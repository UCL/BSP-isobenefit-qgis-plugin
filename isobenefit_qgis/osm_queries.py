"""Pure, QGIS-free helpers for OpenStreetMap extraction.

Overpass-QL query construction, bbox handling, the dataset → OSM-layer mapping and
the tag predicates live here so they can be unit-tested in a plain venv (no
QGIS/GDAL), mirroring the pure/coupled split between ``grid.py`` and ``gis_io.py``.
The QGIS/GDAL-coupled fetch + GeoPackage building lives in ``osm_fetcher.py``.

The model consumes built / green / centre / (water → unbuildable) polygons and, as
groundwork for a later step, the street network and public-transport stops. Centres
are derived from ``landuse=retail|commercial`` and become *true polygon areas* in the
model (every cell they cover is a centre cell).
"""

from __future__ import annotations

import re

# Public Overpass mirrors, tried in order; we rotate on rate-limit / gateway errors.
OVERPASS_ENDPOINTS: tuple[str, ...] = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
)

# Server-side query timeout (seconds), embedded in the Overpass-QL header. Kept modest
# so an overloaded mirror gives up (and we fail over) rather than the client hanging.
OVERPASS_TIMEOUT = 90

# Overpass-QL selectors per dataset: (element, tag-filter) pairs, bbox appended at
# build time. Closed ways tagged landuse/leisure/natural are emitted by GDAL's OSM
# driver as ``multipolygons``; relations carry multipolygons too.
DATASET_SELECTORS: dict[str, list[tuple[str, str]]] = {
    "built": [
        ("way", '["landuse"~"^(residential|commercial|retail|industrial)$"]'),
        ("relation", '["landuse"~"^(residential|commercial|retail|industrial)$"]'),
    ],
    "green": [
        ("way", '["leisure"~"^(park|garden|recreation_ground|nature_reserve)$"]'),
        ("relation", '["leisure"~"^(park|garden|recreation_ground|nature_reserve)$"]'),
        ("way", '["landuse"~"^(grass|meadow|forest|recreation_ground|village_green)$"]'),
        ("relation", '["landuse"~"^(grass|meadow|forest|recreation_ground|village_green)$"]'),
        ("way", '["natural"~"^(wood|scrub|grassland|heath)$"]'),
        ("relation", '["natural"~"^(wood|scrub|grassland|heath)$"]'),
    ],
    "centres": [
        ("way", '["landuse"~"^(retail|commercial)$"]'),
        ("relation", '["landuse"~"^(retail|commercial)$"]'),
    ],
    "streets": [
        ("way", '["highway"]'),
    ],
    "pt": [
        ("node", '["highway"="bus_stop"]'),
        ("node", '["railway"~"^(station|halt|tram_stop)$"]'),
        ("node", '["public_transport"~"^(stop_position|platform|station)$"]'),
    ],
    # Unbuildable land: water, airports/airfields, military, quarries/landfill.
    "unbuildable": [
        ("way", '["natural"="water"]'),
        ("relation", '["natural"="water"]'),
        ("way", '["waterway"~"^(riverbank|dock)$"]'),
        ("relation", '["waterway"~"^(riverbank|dock)$"]'),
        ("way", '["aeroway"~"^(aerodrome|apron|terminal|runway|helipad)$"]'),
        ("relation", '["aeroway"~"^(aerodrome|apron|terminal|runway|helipad)$"]'),
        ("way", '["landuse"~"^(military|quarry|landfill)$"]'),
        ("relation", '["landuse"~"^(military|quarry|landfill)$"]'),
        ("way", '["military"]'),
        ("relation", '["military"]'),
    ],
}

# Per-dataset metadata: human label, which GDAL-OSM-driver layer to read, and the
# output geometry type (used for the GeoPackage layer). The dict key doubles as the
# GeoPackage layer name.
DATASETS: dict[str, dict[str, str]] = {
    "built": {"label": "Built (urban fabric)", "osm_layer": "multipolygons", "geom_type": "MultiPolygon"},
    "green": {"label": "Green space", "osm_layer": "multipolygons", "geom_type": "MultiPolygon"},
    "centres": {"label": "Centres", "osm_layer": "multipolygons", "geom_type": "MultiPolygon"},
    "streets": {"label": "Street network", "osm_layer": "lines", "geom_type": "MultiLineString"},
    "pt": {"label": "Public-transport stops", "osm_layer": "points", "geom_type": "Point"},
    "unbuildable": {
        "label": "Unbuildable (water, airports, military)",
        "osm_layer": "multipolygons",
        "geom_type": "MultiPolygon",
    },
}

# Default display order (also the order datasets are fetched in).
DATASET_ORDER: tuple[str, ...] = ("built", "green", "centres", "streets", "pt", "unbuildable")

# OSM tag keys we read to decide whether a feature matches a dataset. Any not present
# as a native OSM-driver field (e.g. railway / public_transport on some layers) are
# recovered from the ``other_tags`` HSTORE.
TAG_KEYS: frozenset[str] = frozenset(
    {
        "landuse",
        "leisure",
        "natural",
        "highway",
        "railway",
        "public_transport",
        "waterway",
        "aeroway",
        "military",
        "name",
    }
)

# Polygon datasets: a feature matches if ANY field → accepted-values holds. A value of
# ``None`` means "any non-empty value for this key" (e.g. military=*).
_POLYGON_FILTERS: dict[str, dict[str, set[str] | None]] = {
    "built": {"landuse": {"residential", "commercial", "retail", "industrial"}},
    "green": {
        "leisure": {"park", "garden", "recreation_ground", "nature_reserve"},
        "landuse": {"grass", "meadow", "forest", "recreation_ground", "village_green"},
        "natural": {"wood", "scrub", "grassland", "heath"},
    },
    "centres": {"landuse": {"retail", "commercial"}},
    "unbuildable": {
        "natural": {"water"},
        "waterway": {"riverbank", "dock"},
        "aeroway": {"aerodrome", "apron", "terminal", "runway", "helipad"},
        "landuse": {"military", "quarry", "landfill"},
        "military": None,  # any military=* area
    },
}

_PT_RAILWAY = {"station", "halt", "tram_stop"}
_PT_PUBLIC_TRANSPORT = {"stop_position", "platform", "station"}


def bbox_to_overpass(xmin: float, ymin: float, xmax: float, ymax: float) -> tuple[float, float, float, float]:
    """Convert a GIS ``(xmin, ymin, xmax, ymax)`` extent to Overpass ``(s, w, n, e)``.

    Overpass orders its bbox filter ``(south, west, north, east)`` = ``(ymin, xmin,
    ymax, xmax)`` — the transpose of a GIS extent, and a classic source of silently
    wrong queries, so it is isolated here and unit-tested.
    """
    return (ymin, xmin, ymax, xmax)


def build_query(dataset: str, s: float, w: float, n: float, e: float) -> str:
    """Build the Overpass-QL for ``dataset`` over the bbox ``(s, w, n, e)``.

    Unions the matched ways/relations with their recursed members (``(._;>;);``) and
    emits the whole set in a single ``out body;``. This is deliberate: Overpass groups
    a single ``out`` by type (nodes, then ways, then relations) with ascending ids, so
    nodes precede the ways that reference them. GDAL's OSM driver resolves geometry in a
    forward pass and needs that order — the older ``out body; >; out skel qt;`` emits
    ways *before* nodes (and nodes in quadtile order), which GDAL buffers for small
    results but silently drops on large ones, leaving the polygon layers empty.
    """
    if dataset not in DATASET_SELECTORS:
        raise KeyError(f"unknown OSM dataset: {dataset!r}")
    bbox = f"({s},{w},{n},{e})"
    body = "\n".join(f"  {elem}{filt}{bbox};" for elem, filt in DATASET_SELECTORS[dataset])
    return f"[out:xml][timeout:{OVERPASS_TIMEOUT}];\n(\n{body}\n);\n(._;>;);\nout body;\n"


# HSTORE pairs look like ``"key"=>"value"``; values may contain commas, so we match
# quoted pairs rather than splitting on commas.
_HSTORE_RE = re.compile(r'"((?:[^"\\]|\\.)*)"\s*=>\s*"((?:[^"\\]|\\.)*)"')


def _unescape(s: str) -> str:
    return s.replace('\\"', '"').replace("\\\\", "\\")


def parse_hstore(other_tags: str | None) -> dict[str, str]:
    """Parse a GDAL OSM ``other_tags`` HSTORE string into a ``{key: value}`` dict."""
    if not other_tags:
        return {}
    return {_unescape(k): _unescape(v) for k, v in _HSTORE_RE.findall(other_tags)}


def point_is_pt_stop(tags: dict[str, str]) -> bool:
    """Whether an OSM point's tags identify it as a public-transport stop."""
    return (
        tags.get("highway") == "bus_stop"
        or tags.get("railway") in _PT_RAILWAY
        or tags.get("public_transport") in _PT_PUBLIC_TRANSPORT
    )


# Stop "kind", written as a field on the PT layer so the recommended plan can treat
# significant stops (rail/tram stations) differently from ordinary bus stops — only the
# former anchor a centre.
STATION_KINDS: frozenset[str] = frozenset({"rail", "tram"})


def pt_stop_kind(tags: dict[str, str]) -> str:
    """Classify a public-transport stop as ``"rail"``, ``"tram"`` or ``"bus"``.

    Rail/tram stations are the significant interchanges that anchor a centre; everything
    else (bus stops, plain platforms/stop positions) is ``"bus"``.
    """
    if tags.get("railway") == "tram_stop":
        return "tram"
    if tags.get("railway") in {"station", "halt"} or tags.get("public_transport") == "station":
        return "rail"
    return "bus"


# Extra string fields written per dataset (beyond geometry). Default: none.
DATASET_FIELDS: dict[str, list[str]] = {"pt": ["kind"]}


def feature_attributes(dataset: str, tags: dict[str, str]) -> dict[str, str]:
    """Attribute values to persist for a feature, keyed by field name (see DATASET_FIELDS)."""
    if dataset == "pt":
        return {"kind": pt_stop_kind(tags)}
    return {}


def feature_matches(dataset: str, tags: dict[str, str]) -> bool:
    """Whether a feature with merged ``tags`` belongs in ``dataset``.

    ``tags`` is the union of the OSM-driver's native fields and the parsed
    ``other_tags`` HSTORE, so the same predicate works regardless of which tags the
    driver promoted to columns.
    """
    if dataset == "streets":
        return bool(tags.get("highway"))
    if dataset == "pt":
        return point_is_pt_stop(tags)
    for key, values in _POLYGON_FILTERS[dataset].items():
        value = tags.get(key)
        if value is not None and (values is None or value in values):
            return True
    return False
