#!/usr/bin/env python3
"""Download the OSM input layers for a scenarios/<scenario>/ folder, headlessly.

Generalises website/scripts/fetch_data.py (the Cambourne snapshot) to any scenario folder:

    .venv/bin/python scripts/fetch_scenario.py scenarios/dnipro [scenarios/…]

Reads ``<folder>/params.json`` for the scenario's metric CRS and every ``extents*.geojson``
(delivered in that CRS). The download window is the CONVEX HULL of all extents features plus
``fetch_buffer_m`` (params.json, default 0): one Overpass fetch covers every pilot area, while the
individual extents files stay the formal simulation boundaries. Datasets are split with the
plugin's own rules (``isobenefit_qgis.osm_queries``), clipped to the hull, simplified, and written
as ``<dataset>.geojson`` in the scenario CRS. The hull itself is written as
``osm_download_extent.geojson`` for the record.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "website", "scripts"))

from fetch_data import parse_osm, stitch_rings  # noqa: E402  (the shared OSM-XML reader)

from isobenefit_qgis import osm_queries  # noqa: E402

SIMPLIFY_M = 6.0
MIN_AREA_M2 = 1500.0
MIN_LINE_M = 60.0
BARRIER_BUFFER_M = 30.0


def fetch(folder: str) -> None:
    import pyproj
    import shapely
    from shapely.geometry import LineString, Point, Polygon, mapping, shape
    from shapely.ops import unary_union
    from shapely.validation import make_valid

    with open(os.path.join(folder, "params.json"), encoding="utf-8") as fh:
        params = json.load(fh)
    crs = params["crs"]
    buffer_m = float(params.get("fetch_buffer_m", 0.0))

    extents = []
    for name in sorted(os.listdir(folder)):
        if name.startswith("extents") and name.endswith(".geojson"):
            with open(os.path.join(folder, name), encoding="utf-8") as fh:
                fc = json.load(fh)
            extents += [shape(f["geometry"]) for f in fc["features"]]
    if not extents:
        raise SystemExit(f"{folder}: no extents*.geojson found")
    hull = unary_union(extents).convex_hull.buffer(buffer_m) if buffer_m else unary_union(extents).convex_hull
    print(f"{folder}: {len(extents)} extents feature(s), hull {hull.area / 1e6:,.0f} km², CRS {crs}")

    to_wgs = pyproj.Transformer.from_crs(crs, 4326, always_xy=True)
    to_loc = pyproj.Transformer.from_crs(4326, crs, always_xy=True)
    lons, lats = zip(*(to_wgs.transform(x, y) for x, y in hull.exterior.coords))
    s, w, n, e = osm_queries.bbox_to_overpass(min(lons), min(lats), max(lons), max(lats))
    ql = osm_queries.build_combined_query(list(osm_queries.DATASETS), s, w, n, e)
    cache = os.path.join(folder, "_overpass_cache.xml")
    if os.path.exists(cache):
        print("  using cached Overpass response")
        xml_bytes = open(cache, "rb").read()
    else:
        xml_bytes = None
        last_err: Exception | None = None
        for attempt, endpoint in enumerate(osm_queries.OVERPASS_ENDPOINTS * 2):
            try:
                req = urllib.request.Request(
                    endpoint,
                    data=ql.encode(),
                    headers={
                        "Content-Type": "text/plain",
                        "User-Agent": "isobenefit-qgis-scenarios/1.0 (github.com/UCL/BSP-isobenefit-qgis-plugin)",
                    },
                )
                with urllib.request.urlopen(req, timeout=600) as resp:
                    xml_bytes = resp.read()
                break
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
                last_err = exc
                wait = 20 * (attempt + 1)
                print(f"  {endpoint} failed ({exc}); retrying in {wait}s")
                time.sleep(wait)
        if xml_bytes is None:
            raise SystemExit(f"{folder}: every Overpass endpoint failed ({last_err})")
        open(cache, "wb").write(xml_bytes)
        print(f"  downloaded {len(xml_bytes) // (1024 * 1024)} MB of OSM XML (cached)")
    nodes, node_tags, ways, rels = parse_osm(xml_bytes)
    print(f"  parsed: {len(nodes)} nodes, {len(ways)} ways, {len(rels)} relations")

    def project(coords):
        return [to_loc.transform(lon, lat) for lon, lat in coords]

    polygon_ds = [k for k, v in osm_queries.DATASETS.items() if v["geom_type"] == "MultiPolygon"]
    line_ds = [k for k, v in osm_queries.DATASETS.items() if v["geom_type"] == "MultiLineString"]
    point_ds = [k for k, v in osm_queries.DATASETS.items() if v["geom_type"] == "Point"]
    out: dict[str, list] = {k: [] for k in osm_queries.DATASETS}

    def add(dataset, geom, tags):
        if not geom.is_valid:
            geom = make_valid(geom)
        g = geom.intersection(hull)
        if g.is_empty:
            return
        g = g.simplify(SIMPLIFY_M)
        if not g.is_valid:
            g = make_valid(g)
        for p in getattr(g, "geoms", [g]):
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
        if osm_queries.is_barrier_line(tags):
            add("unbuildable", LineString(project(coords)).buffer(BARRIER_BUFFER_M), tags)
        for ds in line_ds:
            if osm_queries.feature_matches(ds, tags):
                add(ds, LineString(project(coords)), tags)
    for rel in rels:
        for ds in polygon_ds:
            if osm_queries.feature_matches(ds, rel["tags"]):
                rings = stitch_rings(
                    [[nodes[r] for r in ways[w]["refs"] if r in nodes] for w in rel["outers"] if w in ways]
                )
                for ring in rings:
                    add(ds, Polygon(project(ring)), rel["tags"])
    for nid, tags in node_tags.items():
        for ds in point_ds:
            if osm_queries.feature_matches(ds, tags):
                add(ds, Point(*to_loc.transform(*nodes[nid])), tags)

    # compose: green never overlaps built/centres (mixed OSM tags happen)
    built_u = unary_union([g for g, _ in out["built"]]) if out["built"] else None
    if built_u is not None:
        kept = []
        for g, tags in out["green"]:
            g2 = g.difference(built_u)
            for p in getattr(g2, "geoms", [g2]):
                if p.geom_type == "Polygon" and p.area >= MIN_AREA_M2:
                    kept.append((p, tags))
        out["green"] = kept

    meta = {"crs": crs, "source": "OpenStreetMap via Overpass (ODbL)",
            "queries": "isobenefit_qgis.osm_queries.build_combined_query"}

    for ds, feats in out.items():
        features = []
        for g, tags in feats:
            # snap to a 0.1 m grid to keep files small; set_precision (unlike naive coordinate
            # rounding) guarantees the result stays VALID, so downstream unary_union never fails
            g = shapely.set_precision(g, 0.1)
            if g.is_empty:
                continue
            features.append(
                {
                    "type": "Feature",
                    "properties": osm_queries.feature_attributes(ds, tags)
                    | ({"name": tags["name"]} if "name" in tags else {}),
                    "geometry": json.loads(json.dumps(mapping(g), default=float)),
                }
            )
        fc = {"type": "FeatureCollection", "metadata": meta, "features": features}
        path = os.path.join(folder, f"{ds}.geojson")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(fc, fh, separators=(",", ":"))
        print(f"  {ds}: {len(features)} features -> {path} ({os.path.getsize(path) // 1024} kB)")

    hull_fc = {
        "type": "FeatureCollection",
        "metadata": meta,
        "features": [{"type": "Feature",
                      "properties": {"role": "osm_download", "note": "convex hull of the extents features"},
                      "geometry": json.loads(json.dumps(mapping(shapely.set_precision(hull, 0.1)), default=float))}],
    }
    with open(os.path.join(folder, "osm_download_extent.geojson"), "w", encoding="utf-8") as fh:
        json.dump(hull_fc, fh, separators=(",", ":"))
    print(f"  osm_download_extent.geojson written; delete {os.path.basename(cache)} once happy")


if __name__ == "__main__":
    for folder in sys.argv[1:] or [None]:
        if folder is None:
            raise SystemExit("usage: fetch_scenario.py scenarios/<scenario> [more…]")
        fetch(folder)
