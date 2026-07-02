#!/usr/bin/env python3
"""Snapshot REAL OpenStreetMap data for the website's demonstration town (Cambourne).

The website's input-layer panels and growth demonstrators all run on one shared
geography — actual downloaded data, split into datasets by the plugin's own rules
(``isobenefit_qgis.osm_queries``), exactly as the in-QGIS "Extract from OpenStreetMap"
tool would produce. This script performs that download headlessly and commits the
result as small simplified GeoJSON snapshots, so regenerating the SVGs never needs
the network or QGIS:

    uv run --no-project --with pyproj --with shapely \
        python website/scripts/fetch_data.py            # refresh the snapshots

Window: a 4.2 km square (84 x 84 cells at 50 m -- the demonstrators' native grid)
centred on Cambourne town, chosen from the largest urban polygon in the demo data.
Output: website/scripts/data/<dataset>.geojson in EPSG:27700, clipped + simplified.
"""
from __future__ import annotations

import json
import math
import os
import sys
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from isobenefit_qgis import osm_queries  # noqa: E402  (the plugin's own dataset rules)

# The demonstration window in EPSG:27700 — 84 x 84 cells of 50 m centred on Cambourne.
WINDOW = (530000.0, 257400.0, 534200.0, 261600.0)  # xmin, ymin, xmax, ymax
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SIMPLIFY_M = 6.0  # geometry simplification tolerance
MIN_AREA_M2 = 1500.0  # drop polygon slivers below this (0.15 ha)
MIN_LINE_M = 60.0  # drop line fragments below this


def overpass_xml() -> bytes:
    """One combined Overpass request for every dataset, via the plugin's query builder."""
    import pyproj

    tf = pyproj.Transformer.from_crs(27700, 4326, always_xy=True)
    xmin, ymin, xmax, ymax = WINDOW
    lons, lats = zip(*(tf.transform(x, y) for x, y in ((xmin, ymin), (xmax, ymax), (xmin, ymax), (xmax, ymin))))
    s, w, n, e = osm_queries.bbox_to_overpass(min(lons), min(lats), max(lons), max(lats))
    ql = osm_queries.build_combined_query(list(osm_queries.DATASETS), s, w, n, e)
    req = urllib.request.Request(
        osm_queries.OVERPASS_ENDPOINTS[0],
        data=ql.encode(),
        headers={
            "Content-Type": "text/plain",
            "User-Agent": "isobenefit-qgis-website-diagrams/1.0 (github.com/UCL/BSP-isobenefit-qgis-plugin)",
        },
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return resp.read()


def parse_osm(xml_bytes: bytes):
    """Minimal OSM-XML reader: nodes, ways, and multipolygon relations with merged tags."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml_bytes)
    nodes: dict[str, tuple[float, float]] = {}
    node_tags: dict[str, dict[str, str]] = {}
    ways: dict[str, dict] = {}
    rels: list[dict] = []
    for el in root:
        if el.tag == "node":
            nid = el.attrib["id"]
            nodes[nid] = (float(el.attrib["lon"]), float(el.attrib["lat"]))
            tags = {t.attrib["k"]: t.attrib["v"] for t in el if t.tag == "tag"}
            if tags:
                node_tags[nid] = tags
        elif el.tag == "way":
            ways[el.attrib["id"]] = {
                "refs": [m.attrib["ref"] for m in el if m.tag == "nd"],
                "tags": {t.attrib["k"]: t.attrib["v"] for t in el if t.tag == "tag"},
            }
        elif el.tag == "relation":
            tags = {t.attrib["k"]: t.attrib["v"] for t in el if t.tag == "tag"}
            if tags.get("type") == "multipolygon":
                rels.append(
                    {
                        "outers": [m.attrib["ref"] for m in el if m.tag == "member"
                                   and m.attrib.get("type") == "way" and m.attrib.get("role") in ("outer", "")],
                        "tags": tags,
                    }
                )
    return nodes, node_tags, ways, rels


def stitch_rings(way_coord_lists):
    """Join way fragments end-to-end into closed rings (for multipolygon outers)."""
    frags = [list(c) for c in way_coord_lists if len(c) >= 2]
    rings = []
    while frags:
        ring = frags.pop()
        while ring[0] != ring[-1]:
            for i, f in enumerate(frags):
                if f[0] == ring[-1]:
                    ring += f[1:]
                elif f[-1] == ring[-1]:
                    ring += f[-2::-1]
                elif f[-1] == ring[0]:
                    ring = f[:-1] + ring
                elif f[0] == ring[0]:
                    ring = f[::-1][:-1] + ring
                else:
                    continue
                frags.pop(i)
                break
            else:
                break  # cannot close (clipped at the bbox) — keep as-is, shapely will drop it
        if len(ring) >= 4 and ring[0] == ring[-1]:
            rings.append(ring)
    return rings


def main() -> None:
    from shapely.geometry import LineString, Point, Polygon, box, mapping
    from shapely.ops import unary_union
    import pyproj

    cache = os.environ.get("OSM_XML")  # reuse a downloaded response instead of re-fetching
    xml_bytes = open(cache, "rb").read() if cache else overpass_xml()
    nodes, node_tags, ways, rels = parse_osm(xml_bytes)
    print(f"parsed: {len(nodes)} nodes, {len(ways)} ways, {len(rels)} multipolygon relations")

    tf = pyproj.Transformer.from_crs(4326, 27700, always_xy=True)

    def project(coords):
        return [tf.transform(lon, lat) for lon, lat in coords]

    window = box(*WINDOW)
    polygon_ds = [k for k, v in osm_queries.DATASETS.items() if v["geom_type"] == "MultiPolygon"]
    line_ds = [k for k, v in osm_queries.DATASETS.items() if v["geom_type"] == "MultiLineString"]
    point_ds = [k for k, v in osm_queries.DATASETS.items() if v["geom_type"] == "Point"]
    out: dict[str, list] = {k: [] for k in osm_queries.DATASETS}

    def add(dataset, geom, tags):
        from shapely.validation import make_valid

        if not geom.is_valid:
            geom = make_valid(geom)
        g = geom.intersection(window)
        if g.is_empty:
            return
        g = g.simplify(SIMPLIFY_M)
        if not g.is_valid:  # simplify can introduce self-intersections
            g = make_valid(g)
        parts = getattr(g, "geoms", [g])
        for p in parts:
            if p.geom_type == "Polygon" and p.area >= MIN_AREA_M2:
                out[dataset].append((p, tags))
            elif p.geom_type == "LineString" and p.length >= MIN_LINE_M:
                out[dataset].append((p, tags))
            elif p.geom_type == "Point":
                out[dataset].append((p, tags))

    for way in ways.values():
        coords = [nodes[r] for r in way["refs"] if r in nodes]
        if len(coords) < 2:
            continue
        tags = way["tags"]
        closed = coords[0] == coords[-1] and len(coords) >= 4
        for ds in polygon_ds:
            if closed and osm_queries.feature_matches(ds, tags):
                add(ds, Polygon(project(coords)), tags)
        # linear barriers belong to unbuildable as buffered no-build corridors (the plugin's rule);
        # buffer > half a 50 m cell so the corridor registers under centre-of-cell sampling
        if osm_queries.is_barrier_line(tags):
            add("unbuildable", LineString(project(coords)).buffer(30.0), tags)
        for ds in line_ds:
            if osm_queries.feature_matches(ds, tags):
                add(ds, LineString(project(coords)), tags)

    for rel in rels:
        for ds in polygon_ds:
            if osm_queries.feature_matches(ds, rel["tags"]):
                rings = stitch_rings([[nodes[r] for r in ways[w]["refs"] if r in nodes]
                                      for w in rel["outers"] if w in ways])
                for ring in rings:
                    add(ds, Polygon(project(ring)), rel["tags"])

    for nid, tags in node_tags.items():
        for ds in point_ds:
            if osm_queries.feature_matches(ds, tags):
                add(ds, Point(*tf.transform(*nodes[nid])), tags)

    # the layers must COMPOSE: green never overlaps built/centres (mixed tags happen in OSM),
    # and centres sit within the built fabric by definition (retail/commercial ⊂ built's tags).
    built_u = unary_union([g for g, _ in out["built"]]) if out["built"] else None
    if built_u is not None:
        kept = []
        for g, tags in out["green"]:
            g2 = g.difference(built_u)
            for p in getattr(g2, "geoms", [g2]):
                if p.geom_type == "Polygon" and p.area >= MIN_AREA_M2:
                    kept.append((p, tags))
        out["green"] = kept

    os.makedirs(DATA_DIR, exist_ok=True)
    meta = {"window_27700": WINDOW, "crs": "EPSG:27700", "source": "OpenStreetMap via Overpass (ODbL)",
            "queries": "isobenefit_qgis.osm_queries.build_combined_query"}
    for ds, feats in out.items():
        fc = {
            "type": "FeatureCollection",
            "metadata": meta,
            "features": [
                {"type": "Feature",
                 "properties": osm_queries.feature_attributes(ds, tags) | ({"name": tags["name"]} if "name" in tags else {}),
                 "geometry": json.loads(json.dumps(mapping(g), default=float))}
                for g, tags in feats
            ],
        }
        # round coordinates to 0.1 m to keep the snapshots small
        def rnd(obj):
            if isinstance(obj, list):
                return [rnd(o) for o in obj]
            if isinstance(obj, float):
                return round(obj, 1)
            return obj
        for f in fc["features"]:
            f["geometry"]["coordinates"] = rnd(f["geometry"]["coordinates"])
        path = os.path.join(DATA_DIR, f"{ds}.geojson")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(fc, fh, separators=(",", ":"))
        print(f"{ds}: {len(feats)} features -> {path} ({os.path.getsize(path) // 1024} kB)")

    # the AOI itself, saved like the plugin saves its extents layer
    xmin, ymin, xmax, ymax = WINDOW
    fc = {"type": "FeatureCollection", "metadata": meta, "features": [
        {"type": "Feature", "properties": {},
         "geometry": {"type": "Polygon",
                      "coordinates": [[[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]]]}}]}
    with open(os.path.join(DATA_DIR, "extents.geojson"), "w", encoding="utf-8") as fh:
        json.dump(fc, fh, separators=(",", ":"))
    print("extents: 1 feature (the demonstration window)")

    # sanity: layers must compose
    green_u = unary_union([g for g, _ in out["green"]]) if out["green"] else None
    if built_u is not None and green_u is not None:
        overlap = built_u.intersection(green_u).area
        assert overlap < 1.0, f"green overlaps built by {overlap:.0f} m2"
        print("compose check: green/built overlap = 0 OK")


if __name__ == "__main__":
    main()
