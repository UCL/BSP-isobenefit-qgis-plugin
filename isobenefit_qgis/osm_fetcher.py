"""QgsTask that downloads OpenStreetMap data via Overpass and writes a GeoPackage.

The pure Overpass-QL / tag logic lives in :mod:`osm_queries`; this module holds the
QGIS/GDAL-coupled parts: the HTTP POST (via QGIS's own ``QgsBlockingNetworkRequest``
so we add **no** new dependency), parsing the OSM XML with GDAL's built-in ``OSM``
driver, and writing each dataset as a layer in an on-disk GeoPackage that the user
can edit/swap before running the simulation. Downloading and simulating are fully
independent steps — this task only produces layers; it never starts a simulation.
"""

from __future__ import annotations

import time

from osgeo import gdal, ogr, osr
from qgis.core import (
    Qgis,
    QgsBlockingNetworkRequest,
    QgsFeedback,
    QgsMessageLog,
    QgsProject,
    QgsTask,
    QgsVectorLayer,
)
from qgis.PyQt.QtCore import QBuffer, QByteArray, QIODevice, QUrl
from qgis.PyQt.QtNetwork import QNetworkRequest

from . import osm_queries

LOG_TAG = "Isobenefit"
USER_AGENT = b"isobenefit-qgis (QGIS plugin; https://github.com/UCL/BSP-isobenefit-qgis-plugin)"

# Per-endpoint retry backoff (seconds) on rate-limit / gateway errors.
_BACKOFF = (2.0, 5.0)
# HTTP statuses worth retrying (rate-limited / overloaded / gateway timeout).
_RETRY_STATUS = {429, 502, 503, 504}

# Abort a request that has produced no data for this long, so a stuck/overloaded
# mirror fails over instead of the fetch appearing frozen. Set above the server-side
# query timeout so a legitimately slow (but progressing) query is not cut off early.
_TRANSFER_TIMEOUT_MS = 120_000

_OGR_GEOM = {
    "MultiPolygon": ogr.wkbMultiPolygon,
    "MultiLineString": ogr.wkbMultiLineString,
    "Point": ogr.wkbPoint,
}

# Linear barriers (motorways/railways/rivers) are buffered into thin no-build corridor polygons
# before being merged into the unbuildable layer. ~15 m (in degrees, mid-latitude); the
# simulation rasterises unbuildable with ALL_TOUCHED so the corridor reliably carves its cells.
_BARRIER_BUFFER_DEG = 0.00015


class OsmError(Exception):
    """Raised for Overpass / OSM-parsing failures, with a user-facing message."""


def _interruptible_sleep(seconds: float, feedback: QgsFeedback | None) -> None:
    """Sleep in small slices so a cancelled task wakes up promptly."""
    waited = 0.0
    while waited < seconds:
        if feedback is not None and feedback.isCanceled():
            return
        time.sleep(0.1)
        waited += 0.1


def _looks_like_osm(content: bytes) -> bool:
    """Whether a response body is OSM XML (a valid empty result still has an <osm> root).

    Rejects empty bodies and HTML/text error pages that overloaded mirrors return with
    a 200 status — which GDAL's OSM driver would otherwise fail to open downstream.
    """
    return b"<osm" in content[:1024].lstrip().lower()


def _overpass_error(content: bytes) -> str | None:
    """An Overpass error reported as a ``<remark>`` inside otherwise-valid OSM XML.

    A server-side timeout/overload still returns ``[out:xml]`` with an ``<osm>`` root
    plus ``<remark> runtime error: Query timed out … </remark>`` and no data — which
    GDAL would happily parse as zero features. Surface it so we retry another mirror
    instead of silently returning nothing.
    """
    idx = content.find(b"<remark>")
    if idx == -1:
        return None
    snippet = content[idx : idx + 500].lower()
    if b"runtime error" in snippet or b"timed out" in snippet:
        return "Overpass query timed out / overloaded"
    return None


def overpass_post(
    ql: str,
    feedback: QgsFeedback | None = None,
    endpoints: tuple[str, ...] = osm_queries.OVERPASS_ENDPOINTS,
) -> bytes:
    """POST an Overpass-QL query and return the raw OSM-XML bytes.

    Uses ``QgsBlockingNetworkRequest`` (safe on a worker thread). Validates that the
    body is OSM, backs off and retries, and rotates over the mirror list on
    rate-limit/overload. Raises :class:`OsmError` if no mirror yields usable data.
    """
    last_error = "no Overpass endpoint responded"
    for url in endpoints:
        for attempt in range(len(_BACKOFF) + 1):
            if feedback is not None and feedback.isCanceled():
                raise OsmError("cancelled")
            request = QNetworkRequest(QUrl(url))
            request.setHeader(QNetworkRequest.KnownHeaders.ContentTypeHeader, "text/plain; charset=utf-8")
            request.setRawHeader(b"User-Agent", USER_AGENT)
            request.setRawHeader(b"Accept-Encoding", b"identity")  # avoid gzip surprises
            if hasattr(request, "setTransferTimeout"):  # Qt >= 5.15: bound a stuck connection
                request.setTransferTimeout(_TRANSFER_TIMEOUT_MS)
            # Keep both the QByteArray and the QBuffer alive across the blocking call.
            body = QByteArray(ql.encode("utf-8"))
            buffer = QBuffer(body)
            buffer.open(QIODevice.OpenModeFlag.ReadOnly)
            blocking = QgsBlockingNetworkRequest()
            err = blocking.post(request, buffer, True, feedback)
            reply = blocking.reply()
            if err == QgsBlockingNetworkRequest.ErrorCode.NoError:
                content = bytes(reply.content())
                if _looks_like_osm(content):
                    overpass_err = _overpass_error(content)
                    if overpass_err is None:
                        return content
                    last_error = overpass_err  # server-side timeout/overload: retry/rotate
                else:
                    # 200 OK but empty/non-OSM (an overloaded mirror's error page).
                    last_error = "Overpass returned an empty or non-OSM response (mirror may be overloaded)"
            else:
                status = reply.attribute(QNetworkRequest.Attribute.HttpStatusCodeAttribute)
                last_error = blocking.errorMessage() or f"HTTP {status}"
                if status not in _RETRY_STATUS:
                    break  # non-retryable → next endpoint
            if attempt < len(_BACKOFF):
                _interruptible_sleep(_BACKOFF[attempt], feedback)
                continue
            break  # attempts exhausted → next endpoint
    raise OsmError(last_error)


def _feature_tags(feat: "ogr.Feature") -> dict[str, str]:
    """Merge an OGR feature's native tag fields with its ``other_tags`` HSTORE."""
    tags: dict[str, str] = {}
    defn = feat.GetDefnRef()
    for i in range(defn.GetFieldCount()):
        name = defn.GetFieldDefn(i).GetName()
        if not feat.IsFieldSet(i):
            continue
        if name == "other_tags":
            tags.update(osm_queries.parse_hstore(feat.GetFieldAsString(i)))
        elif name in osm_queries.TAG_KEYS:
            value = feat.GetFieldAsString(i)
            if value:
                tags.setdefault(name, value)
    return tags


def _clip(geom, aoi):
    """Trim ``geom`` to the AOI polygon, or None if it falls outside.

    Overpass returns everything in the bounding box; trimming to the actual drawn
    polygon is what makes the download match the area the user asked for. Invalid OSM
    geometries (self-intersections etc.) can make ``Intersection`` raise — in that
    case we keep the bbox-filtered original rather than dropping real data.
    """
    if aoi is None:
        return geom
    try:
        if not geom.Intersects(aoi):
            return None
        clipped = geom.Intersection(aoi)
        if clipped is None or clipped.IsEmpty():
            return None
        return clipped
    except Exception:
        return geom


def read_osm_datasets(
    xml_bytes: bytes, datasets, aoi_wkt: str | None = None
) -> "dict[str, list[tuple[str, dict[str, str]]]]":
    """Parse the combined Overpass OSM-XML ONCE and split it into ``{dataset: [(wkt, attrs), …]}``.

    Datasets are grouped by the GDAL-OSM-driver layer they read (``multipolygons`` / ``lines`` /
    ``points``) so each layer is scanned a single time; a feature is routed to every dataset whose
    tag filter it matches (``attrs`` carries the persisted fields per dataset — e.g. a stop's
    ``kind`` or a street's ``highway``). Features are trimmed to ``aoi_wkt`` (EPSG:4326) when given.

    Reads with interleaved reading OFF (a single-layer scan fully assembles ``multipolygons`` from
    closed ways/relations; interleaved reading yields ``None`` at round boundaries and silently
    drops polygons) and ``OSM_USE_CUSTOM_INDEXING=NO`` (tolerates non-increasing node ids). The
    combined query emits nodes before ways, which GDAL's forward-pass assembly needs.
    """
    aoi = ogr.CreateGeometryFromWkt(aoi_wkt) if aoi_wkt else None
    gdal.SetConfigOption("OSM_USE_CUSTOM_INDEXING", "NO")
    gdal.SetConfigOption("OGR_INTERLEAVED_READING", "NO")
    by_layer: dict[str, list[str]] = {}
    for dataset in datasets:
        by_layer.setdefault(osm_queries.DATASETS[dataset]["osm_layer"], []).append(dataset)
    result: dict[str, list[tuple[str, dict[str, str]]]] = {d: [] for d in datasets}
    # Motorways/railways/rivers are carved into the unbuildable substrate as no-build corridors:
    # read them from the lines layer (even if no line dataset was selected) and buffer to polygons.
    want_barriers = "unbuildable" in result
    if want_barriers and "lines" not in by_layer:
        by_layer["lines"] = []
    vsipath = "/vsimem/overpass_combined.osm"
    gdal.FileFromMemBuffer(vsipath, xml_bytes)
    try:
        ds = gdal.OpenEx(vsipath, gdal.OF_VECTOR)
        if ds is None:
            raise OsmError("GDAL could not parse the Overpass response as OSM data")
        for layer_name, dsets in by_layer.items():
            layer = ds.GetLayerByName(layer_name)
            if layer is None:
                continue
            layer.ResetReading()
            feat = layer.GetNextFeature()
            while feat is not None:
                tags = _feature_tags(feat)
                matched = [d for d in dsets if osm_queries.feature_matches(d, tags)]
                barrier = layer_name == "lines" and want_barriers and osm_queries.is_barrier_line(tags)
                if matched or barrier:
                    geom = feat.GetGeometryRef()
                    if geom is not None and not geom.IsEmpty():
                        kept = _clip(geom, aoi)
                        if kept is not None:
                            if matched:
                                wkt = kept.ExportToWkt()
                                for d in matched:
                                    result[d].append((wkt, osm_queries.feature_attributes(d, tags)))
                            if barrier:  # buffer the line into a thin no-build corridor polygon
                                corridor = kept.Buffer(_BARRIER_BUFFER_DEG)
                                if corridor is not None and not corridor.IsEmpty():
                                    result["unbuildable"].append((corridor.ExportToWkt(), {}))
                feat = layer.GetNextFeature()
        ds = None
    finally:
        gdal.Unlink(vsipath)
    return result


def _wgs84_srs() -> "osr.SpatialReference":
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    return srs


EXTENTS_LAYER = "extents"


def _layer_label(key: str) -> str:
    """Human label for a written GeoPackage layer (datasets + the extents layer)."""
    if key == EXTENTS_LAYER:
        return "Area / extents"
    return osm_queries.DATASETS[key]["label"]


def write_geopackage_layer(
    gpkg_ds: "ogr.DataSource",
    layer_name: str,
    geom_type: str,
    features: list[tuple[str, dict[str, str]]],
    srs,
    fields: "tuple[str, ...] | list[str]" = (),
) -> int:
    """Create ``layer_name`` (of ``geom_type``) in an open GeoPackage and write ``features``.

    Each feature is ``(wkt, attrs)``; ``fields`` names the string attribute columns to
    create (e.g. ``("highway",)`` for the street network), and each feature's ``attrs`` fills them.
    Returns the number of features written. Geometries are forced to the layer's Multi type
    and null/invalid WKT is skipped.
    """
    layer = gpkg_ds.CreateLayer(layer_name, srs, _OGR_GEOM[geom_type], options=["GEOMETRY_NAME=geom"])
    for fname in fields:
        layer.CreateField(ogr.FieldDefn(fname, ogr.OFTString))
    defn = layer.GetLayerDefn()
    count = 0
    for wkt, attrs in features:
        geom = ogr.CreateGeometryFromWkt(wkt)
        if geom is None:
            continue
        if geom_type == "MultiPolygon":
            geom = ogr.ForceToMultiPolygon(geom)
        elif geom_type == "MultiLineString":
            geom = ogr.ForceToMultiLineString(geom)
        feature = ogr.Feature(defn)
        feature.SetGeometry(geom)
        for fname in fields:
            if attrs.get(fname):
                feature.SetField(fname, attrs[fname])
        layer.CreateFeature(feature)
        count += 1
    return count


class OsmFetchTask(QgsTask):
    """Background task: Overpass download → parse → write a GeoPackage → load layers.

    ``bbox`` is ``(xmin, ymin, xmax, ymax)`` in EPSG:4326 and drives the Overpass
    query; ``aoi_wkt`` (an EPSG:4326 polygon) trims the results to the exact area and is
    also saved as an ``extents`` layer (the simulation needs an extents polygon, so the
    retrieval area is retained for the analysis). Output layers stay in 4326 (the
    simulation reprojects on use). Only the datasets in ``datasets`` are fetched.
    """

    def __init__(self, *, iface, bbox, datasets, gpkg_path, group_name, aoi_wkt=None):
        super().__init__("Fetch OpenStreetMap data")
        self.iface = iface
        self.bbox = tuple(float(v) for v in bbox)
        self.aoi_wkt = aoi_wkt
        self.datasets = [d for d in osm_queries.DATASET_ORDER if d in set(datasets)]
        self.gpkg_path = str(gpkg_path)
        self.group_name = group_name
        self._feedback = QgsFeedback()
        # populated during run(): [(dataset_key, feature_count), ...]
        self.results: list[tuple[str, int]] = []
        self.error_message: str | None = None

    def cancel(self) -> None:
        self._feedback.cancel()
        super().cancel()

    @staticmethod
    def _log(message: str, level=Qgis.MessageLevel.Info, notify: bool = False) -> None:
        QgsMessageLog.logMessage(message, LOG_TAG, level=level, notifyUser=notify)

    def run(self) -> bool:
        try:
            if gdal.GetDriverByName("OSM") is None:
                self.error_message = (
                    "This QGIS build has no GDAL 'OSM' driver, so OpenStreetMap data cannot be parsed."
                )
                return False
            if ogr.GetDriverByName("GPKG") is None:
                self.error_message = "This QGIS build has no GDAL 'GPKG' driver, so the output cannot be written."
                return False

            s, w, n, e = osm_queries.bbox_to_overpass(*self.bbox)
            self._log(
                f"Area bbox (EPSG:4326): S {s:.5f}, W {w:.5f}, N {n:.5f}, E {e:.5f}; "
                f"AOI clip = {'on' if self.aoi_wkt else 'off'}."
            )
            srs = _wgs84_srs()
            # ONE combined Overpass request for all datasets. Many small sequential requests get
            # rate-limited / queued by the public mirrors (which turned a small area into minutes —
            # a throttled dataset retrying through repeated server-side timeouts); a single request
            # avoids that. The download happens once here; splitting it per dataset below is fast,
            # local parsing + AOI clipping.
            self._log(f"Downloading {len(self.datasets)} dataset(s) from OpenStreetMap in one request…")
            try:
                ql = osm_queries.build_combined_query(self.datasets, s, w, n, e)
                xml = overpass_post(ql, self._feedback)
                by_dataset = read_osm_datasets(xml, self.datasets, self.aoi_wkt)
            except OsmError as exc:
                if str(exc) == "cancelled":
                    return False
                self.error_message = f"OpenStreetMap download failed: {exc}"
                return False
            self.setProgress(80.0)

            # Only now that the download + parse have succeeded, start a fresh GeoPackage so a
            # re-fetch fully replaces the previous one. (Deleting it before the download meant a
            # failed fetch destroyed the existing data and invalidated any project layers on it.)
            gpkg_drv = ogr.GetDriverByName("GPKG")
            if gdal.VSIStatL(self.gpkg_path) is not None:
                gpkg_drv.DeleteDataSource(self.gpkg_path)
            gpkg_ds = gpkg_drv.CreateDataSource(self.gpkg_path)
            if gpkg_ds is None:
                self.error_message = f"Could not create the output GeoPackage: {self.gpkg_path}"
                return False

            total = len(self.datasets)
            for i, dataset in enumerate(self.datasets):
                if self.isCanceled():
                    self._log("OSM fetch cancelled by user.", Qgis.MessageLevel.Warning)
                    return False
                label = osm_queries.DATASETS[dataset]["label"]
                count = write_geopackage_layer(
                    gpkg_ds,
                    dataset,
                    osm_queries.DATASETS[dataset]["geom_type"],
                    by_dataset.get(dataset, []),
                    srs,
                    osm_queries.DATASET_FIELDS.get(dataset, ()),
                )
                self.results.append((dataset, count))
                self._log(f"{label}: {count} feature(s).")
                self.setProgress(80.0 + (i + 1) / total * 20.0)

            # Retain the retrieval area as an extents layer — the simulation needs an
            # extents polygon, and this keeps the downloaded inputs and the analysis area
            # in lockstep so the user can run straight away.
            if self.aoi_wkt:
                write_geopackage_layer(gpkg_ds, EXTENTS_LAYER, "MultiPolygon", [(self.aoi_wkt, {})], srs)
                self.results.append((EXTENTS_LAYER, 1))
                self._log("Saved the area of interest as an extents layer.")

            gpkg_ds = None  # flush + close
            self._log(f"OSM data written to {self.gpkg_path}")
            return True
        except OsmError as exc:
            if str(exc) == "cancelled":
                return False
            self.error_message = str(exc)
            return False
        except Exception as exc:  # noqa: BLE001 — surface any failure as a friendly message
            self.error_message = str(exc)
            return False

    def finished(self, result: bool) -> None:
        if not result:
            self._log(
                f"OSM fetch did not complete: {self.error_message or 'cancelled'}",
                Qgis.MessageLevel.Warning,
                notify=True,
            )
            return
        # Load each written layer from the GeoPackage (main thread only).
        project = QgsProject.instance()
        root = project.layerTreeRoot()
        group = root.insertGroup(0, self.group_name)
        group.setExpanded(True)
        loaded = 0
        for key, _count in self.results:
            uri = f"{self.gpkg_path}|layername={key}"
            layer = QgsVectorLayer(uri, f"OSM — {_layer_label(key)}", "ogr")
            if not layer.isValid():
                self._log(f"Could not load OSM layer '{key}' from the GeoPackage.", Qgis.MessageLevel.Warning)
                continue
            # tag with the dataset key so the simulation dialog can pre-select it in the right combo
            layer.setCustomProperty("isobenefit/osm_dataset", key)
            project.addMapLayer(layer, addToLegend=False)
            group.addLayer(layer)
            loaded += 1
        summary = ", ".join(f"{_layer_label(k).lower()}: {c}" for k, c in self.results)
        self._log(
            f"Loaded {loaded} OpenStreetMap layer(s) ({summary}). Edit/swap as needed, then run the simulation.",
            notify=True,
        )
