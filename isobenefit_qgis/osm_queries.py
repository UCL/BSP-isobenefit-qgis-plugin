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

# Server-side query timeout (seconds), embedded in the Overpass-QL header. Kept modest so an
# overloaded mirror gives up (and we fail over) rather than the client hanging. We now send ONE
# combined request per fetch (see build_combined_query), so this cap is paid at most once.
OVERPASS_TIMEOUT = 60

# Overpass-QL selectors per dataset: (element, tag-filter) pairs, bbox appended at
# build time. Closed ways tagged landuse/leisure/natural are emitted by GDAL's OSM
# driver as ``multipolygons``; relations carry multipolygons too.
DATASET_SELECTORS: dict[str, list[tuple[str, str]]] = {
    "built": [
        ("way", '["landuse"~"^(residential|commercial|retail)$"]'),
        ("relation", '["landuse"~"^(residential|commercial|retail)$"]'),
    ],
    # Industrial is its OWN category — not residential fabric (homes) and not a centre. Treated as
    # existing non-residential land: no new housing grows on it (carved via unbuildable below).
    "industrial": [
        ("way", '["landuse"="industrial"]'),
        ("relation", '["landuse"="industrial"]'),
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
    "railways": [
        ("way", '["railway"~"^(rail|light_rail|subway|tram)$"]'),
    ],
    # Public transport split into two layers so each can be edited/swapped on its own:
    # ordinary stops (bus etc.) and significant rail/tram stations (which anchor centres).
    "stops": [
        ("node", '["highway"="bus_stop"]'),
        ("node", '["public_transport"~"^(stop_position|platform)$"]'),
    ],
    "stations": [
        ("node", '["railway"~"^(station|halt|tram_stop)$"]'),
        ("node", '["public_transport"="station"]'),
    ],
    # Unbuildable land: water, airports/airfields, military, quarries/landfill, industrial.
    "unbuildable": [
        ("way", '["natural"="water"]'),
        ("relation", '["natural"="water"]'),
        ("way", '["waterway"~"^(riverbank|dock)$"]'),
        ("relation", '["waterway"~"^(riverbank|dock)$"]'),
        ("way", '["aeroway"~"^(aerodrome|apron|terminal|runway|helipad)$"]'),
        ("relation", '["aeroway"~"^(aerodrome|apron|terminal|runway|helipad)$"]'),
        # industrial is no-build for new housing (also emitted as its own 'industrial' layer)
        ("way", '["landuse"~"^(military|quarry|landfill|industrial)$"]'),
        ("relation", '["landuse"~"^(military|quarry|landfill|industrial)$"]'),
        ("way", '["military"]'),
        ("relation", '["military"]'),
        # Linear barriers carved out of the buildable substrate too (motorways, railways, rivers):
        # fetched as lines and buffered to no-build corridors when read.
        ("way", '["highway"~"^(motorway|motorway_link|trunk|trunk_link|primary|primary_link)$"]'),
        ("way", '["railway"~"^(rail|light_rail|subway|tram)$"]'),
        ("way", '["waterway"~"^(river|canal|stream)$"]'),
    ],
}

# Per-dataset metadata: human label, which GDAL-OSM-driver layer to read, and the
# output geometry type (used for the GeoPackage layer). The dict key doubles as the
# GeoPackage layer name.
DATASETS: dict[str, dict[str, str]] = {
    "built": {"label": "Built (residential fabric)", "osm_layer": "multipolygons", "geom_type": "MultiPolygon"},
    "green": {"label": "Green space", "osm_layer": "multipolygons", "geom_type": "MultiPolygon"},
    "centres": {"label": "Centres", "osm_layer": "multipolygons", "geom_type": "MultiPolygon"},
    "industrial": {"label": "Industrial land", "osm_layer": "multipolygons", "geom_type": "MultiPolygon"},
    "streets": {"label": "Street network", "osm_layer": "lines", "geom_type": "MultiLineString"},
    "railways": {"label": "Railways", "osm_layer": "lines", "geom_type": "MultiLineString"},
    "stops": {"label": "Public-transport stops", "osm_layer": "points", "geom_type": "Point"},
    "stations": {"label": "Rail / tram stations", "osm_layer": "points", "geom_type": "Point"},
    "unbuildable": {
        "label": "Unbuildable (water, airports, military, industrial)",
        "osm_layer": "multipolygons",
        "geom_type": "MultiPolygon",
    },
}

# Default display order (also the order datasets are fetched in).
DATASET_ORDER: tuple[str, ...] = (
    "built",
    "green",
    "centres",
    "industrial",
    "streets",
    "railways",
    "stops",
    "stations",
    "unbuildable",
)

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
    "built": {"landuse": {"residential", "commercial", "retail"}},  # residential fabric (NOT industrial)
    "green": {
        "leisure": {"park", "garden", "recreation_ground", "nature_reserve"},
        "landuse": {"grass", "meadow", "forest", "recreation_ground", "village_green"},
        "natural": {"wood", "scrub", "grassland", "heath"},
    },
    "centres": {"landuse": {"retail", "commercial"}},
    "industrial": {"landuse": {"industrial"}},
    "unbuildable": {
        "natural": {"water"},
        "waterway": {"riverbank", "dock"},
        "aeroway": {"aerodrome", "apron", "terminal", "runway", "helipad"},
        "landuse": {"military", "quarry", "landfill", "industrial"},  # industrial: no new housing
        "military": None,  # any military=* area
    },
}

_STATION_RAILWAY = {"station", "halt", "tram_stop"}  # rail/tram stations — significant, anchor a centre
_STOP_PUBLIC_TRANSPORT = {"stop_position", "platform"}  # ordinary stops (bus etc.)
_RAILWAY_LINES = {"rail", "light_rail", "subway", "tram"}

# Linear barriers carved out of the buildable substrate (never developed). Motorway-class roads,
# railways and rivers/canals are fetched as lines and buffered into the unbuildable layer.
_BARRIER_HIGHWAY = {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link"}
_BARRIER_WATERWAY = {"river", "canal", "stream"}


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


def build_combined_query(datasets, s: float, w: float, n: float, e: float) -> str:
    """One Overpass-QL query covering ALL ``datasets`` over the bbox ``(s, w, n, e)``.

    Public mirrors rate-limit / queue many small sequential requests, which can turn a small area
    into minutes (a throttled dataset retries through repeated server-side timeouts). A single
    request avoids that: every category comes back in one response and the reader splits it per
    dataset by tag. Selectors are de-duplicated (e.g. ``built`` and ``centres`` share landuse
    selectors). Same node-before-way ordering as :func:`build_query`.
    """
    selectors: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for dataset in datasets:
        for sel in DATASET_SELECTORS.get(dataset, []):
            if sel not in seen:
                seen.add(sel)
                selectors.append(sel)
    if not selectors:
        raise ValueError("no known datasets to query")
    bbox = f"({s},{w},{n},{e})"
    body = "\n".join(f"  {elem}{filt}{bbox};" for elem, filt in selectors)
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


def point_is_station(tags: dict[str, str]) -> bool:
    """A significant rail/tram station (anchors a centre in the recommended plan)."""
    return tags.get("railway") in _STATION_RAILWAY or tags.get("public_transport") == "station"


def point_is_stop(tags: dict[str, str]) -> bool:
    """An ordinary public-transport stop (bus stop, platform, stop position)."""
    return tags.get("highway") == "bus_stop" or tags.get("public_transport") in _STOP_PUBLIC_TRANSPORT


def is_barrier_line(tags: dict[str, str]) -> bool:
    """A linear barrier (motorway-class road, railway, or river/canal) — carved into the
    unbuildable substrate as a no-build corridor so it is never developed."""
    return (
        tags.get("highway") in _BARRIER_HIGHWAY
        or tags.get("railway") in _RAILWAY_LINES
        or tags.get("waterway") in _BARRIER_WATERWAY
    )


# Extra string fields written per dataset (beyond geometry). The street network carries its
# highway class so the routing graph can tell walkable streets from motorway-class barriers.
DATASET_FIELDS: dict[str, list[str]] = {"streets": ["highway"]}


def feature_attributes(dataset: str, tags: dict[str, str]) -> dict[str, str]:
    """Attribute values to persist for a feature, keyed by field name (see DATASET_FIELDS)."""
    if dataset == "streets":
        highway = tags.get("highway")
        return {"highway": highway} if highway else {}
    return {}


def feature_matches(dataset: str, tags: dict[str, str]) -> bool:
    """Whether a feature with merged ``tags`` belongs in ``dataset``.

    ``tags`` is the union of the OSM-driver's native fields and the parsed
    ``other_tags`` HSTORE, so the same predicate works regardless of which tags the
    driver promoted to columns.
    """
    if dataset == "streets":
        return bool(tags.get("highway"))
    if dataset == "railways":
        return tags.get("railway") in _RAILWAY_LINES
    if dataset == "stations":
        return point_is_station(tags)
    if dataset == "stops":
        return point_is_stop(tags)
    for key, values in _POLYGON_FILTERS[dataset].items():
        value = tags.get(key)
        if value is not None and (values is None or value in values):
            return True
    return False
