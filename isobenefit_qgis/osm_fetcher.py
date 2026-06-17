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


def read_osm_layer(xml_bytes: bytes, dataset: str, aoi_wkt: str | None = None) -> list[str]:
    """Parse Overpass OSM-XML and return WKT for every feature matching ``dataset``.

    Reads the single target layer of GDAL's ``OSM`` driver directly with interleaved
    reading OFF (the default, as ``ogr2ogr file.osm multipolygons`` does): a single-layer
    scan fully assembles that layer, including ``multipolygons`` built from closed ways
    and relations. (Interleaved reading returns features in rounds and yields ``None`` at
    each round boundary, so a naive loop stops early and silently drops every
    multipolygon — which is why the polygon datasets came back empty.) Features are
    trimmed to ``aoi_wkt`` (an EPSG:4326 polygon) when given.
    """
    want_layer = osm_queries.DATASETS[dataset]["osm_layer"]
    aoi = ogr.CreateGeometryFromWkt(aoi_wkt) if aoi_wkt else None
    # The query emits nodes before ways (see osm_queries.build_query), which is what
    # GDAL's forward-pass assembly needs. This flag is extra insurance: it drops the
    # requirement that node ids be strictly increasing, so any residual ordering quirk
    # still resolves rather than erroring out ("Non increasing node id").
    gdal.SetConfigOption("OSM_USE_CUSTOM_INDEXING", "NO")
    gdal.SetConfigOption("OGR_INTERLEAVED_READING", "NO")
    vsipath = f"/vsimem/overpass_{dataset}.osm"
    gdal.FileFromMemBuffer(vsipath, xml_bytes)
    wkts: list[str] = []
    try:
        ds = gdal.OpenEx(vsipath, gdal.OF_VECTOR)
        if ds is None:
            raise OsmError("GDAL could not parse the Overpass response as OSM data")
        layer = ds.GetLayerByName(want_layer)
        if layer is None:
            return []
        layer.ResetReading()
        feat = layer.GetNextFeature()
        while feat is not None:
            if osm_queries.feature_matches(dataset, _feature_tags(feat)):
                geom = feat.GetGeometryRef()
                if geom is not None and not geom.IsEmpty():
                    kept = _clip(geom, aoi)
                    if kept is not None:
                        wkts.append(kept.ExportToWkt())
            feat = layer.GetNextFeature()
        ds = None
    finally:
        gdal.Unlink(vsipath)
    return wkts


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


def write_geopackage_layer(gpkg_ds: "ogr.DataSource", layer_name: str, geom_type: str, wkts: list[str], srs) -> int:
    """Create ``layer_name`` (of ``geom_type``) in an open GeoPackage and write ``wkts``.

    Returns the number of features written. Geometries are forced to the layer's
    Multi type and null/invalid WKT is skipped.
    """
    layer = gpkg_ds.CreateLayer(layer_name, srs, _OGR_GEOM[geom_type], options=["GEOMETRY_NAME=geom"])
    defn = layer.GetLayerDefn()
    count = 0
    for wkt in wkts:
        geom = ogr.CreateGeometryFromWkt(wkt)
        if geom is None:
            continue
        if geom_type == "MultiPolygon":
            geom = ogr.ForceToMultiPolygon(geom)
        elif geom_type == "MultiLineString":
            geom = ogr.ForceToMultiLineString(geom)
        feature = ogr.Feature(defn)
        feature.SetGeometry(geom)
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
            # Start a fresh GeoPackage so a re-fetch fully replaces the previous one.
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
                self._log(f"Downloading {label}…")
                # Isolate per-dataset failures: a single overloaded query shouldn't
                # discard the datasets that already downloaded — write it empty and warn.
                try:
                    ql = osm_queries.build_query(dataset, s, w, n, e)
                    xml = overpass_post(ql, self._feedback)
                    wkts = read_osm_layer(xml, dataset, self.aoi_wkt)
                except Exception as exc:  # noqa: BLE001
                    if isinstance(exc, OsmError) and str(exc) == "cancelled":
                        return False
                    self._log(f"{label}: skipped ({exc}).", Qgis.MessageLevel.Warning, notify=True)
                    wkts = []
                count = write_geopackage_layer(gpkg_ds, dataset, osm_queries.DATASETS[dataset]["geom_type"], wkts, srs)
                self.results.append((dataset, count))
                self._log(f"{label}: {count} feature(s).")
                self.setProgress((i + 1) / total * 100.0)
                if i + 1 < total:
                    _interruptible_sleep(1.0, self._feedback)  # be polite between Overpass queries

            # Retain the retrieval area as an extents layer — the simulation needs an
            # extents polygon, and this keeps the downloaded inputs and the analysis area
            # in lockstep so the user can run straight away.
            if self.aoi_wkt:
                write_geopackage_layer(gpkg_ds, EXTENTS_LAYER, "MultiPolygon", [self.aoi_wkt], srs)
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
            project.addMapLayer(layer, addToLegend=False)
            group.addLayer(layer)
            loaded += 1
        summary = ", ".join(f"{_layer_label(k).lower()}: {c}" for k, c in self.results)
        self._log(
            f"Loaded {loaded} OpenStreetMap layer(s) ({summary}). Edit/swap as needed, then run the simulation.",
            notify=True,
        )
