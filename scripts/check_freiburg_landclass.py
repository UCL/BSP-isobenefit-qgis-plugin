#!/usr/bin/env python3
"""Land-classification fairness check for the Freiburg comparison.

Answers one question: is land that is buildable or set aside portrayed the same
way for the regrown plan and for the real districts? Two sources are overlaid on
every cluster of regrown development and on the real district land:

1. The committed scenario inputs (green, unbuildable, industrial).
2. Protection and set-aside designations fetched fresh from Overpass
   (protected_area, nature_reserve, military, cemetery, allotments, and similar),
   cached as scenarios/freiburg_rieselfeld/_protection.geojson.

Run scripts/validate_freiburg.py first; this script reads its saved plan array.

    .venv/bin/python scripts/check_freiburg_landclass.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import urllib.parse
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import shapely  # noqa: E402
import shapely.ops  # noqa: E402
from pyproj import Transformer  # noqa: E402

from isobenefit_qgis import grid as G  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "validate_freiburg", os.path.join(REPO, "scripts", "validate_freiburg.py")
)
_val = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_val)
_gallery = _val._gallery

SCENARIO = _val.SCENARIO
OUT = _val.OUT
PROTECTION_CACHE = os.path.join(SCENARIO, "_protection.geojson")

OVERPASS_QUERY = """
[out:json][timeout:90];
(
  way["boundary"="protected_area"]({bbox});
  relation["boundary"="protected_area"]({bbox});
  way["leisure"="nature_reserve"]({bbox});
  relation["leisure"="nature_reserve"]({bbox});
  way["landuse"~"^(military|cemetery|allotments|landfill|quarry|greenfield)$"]({bbox});
  relation["landuse"~"^(military|cemetery|allotments|landfill|quarry|greenfield)$"]({bbox});
  way["boundary"="water_protection_area"]({bbox});
  relation["boundary"="water_protection_area"]({bbox});
  way["aeroway"~"^(aerodrome|runway)$"]({bbox});
  relation["aeroway"="aerodrome"]({bbox});
);
out geom;
"""


def _way_polygon(coords):
    if len(coords) >= 4 and coords[0] == coords[-1]:
        return shapely.Polygon(coords)
    return None


def fetch_protection(extent) -> list[dict]:
    """Designation polygons in EPSG:25832 with their tags, cached after first fetch."""
    if os.path.exists(PROTECTION_CACHE):
        with open(PROTECTION_CACHE, encoding="utf-8") as fh:
            fc = json.load(fh)
        return [
            {"geom": shapely.make_valid(shapely.geometry.shape(f["geometry"])), "tags": f["properties"]}
            for f in fc["features"]
        ]
    to_wgs = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
    xmin, ymin, xmax, ymax = extent.bounds
    lon0, lat0 = to_wgs.transform(xmin, ymin)
    lon1, lat1 = to_wgs.transform(xmax, ymax)
    bbox = f"{lat0},{lon0},{lat1},{lon1}"
    body = urllib.parse.urlencode({"data": OVERPASS_QUERY.format(bbox=bbox)}).encode()
    req = urllib.request.Request(
        "https://overpass-api.de/api/interpreter",
        data=body,
        headers={"User-Agent": "BSP-isobenefit-qgis-plugin land-classification check"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.load(resp)
    keep_keys = ("boundary", "leisure", "landuse", "aeroway", "protect_class", "protection_title", "name")
    feats = []
    for el in data.get("elements", []):
        tags = {k: v for k, v in el.get("tags", {}).items() if k in keep_keys}
        polys = []
        if el["type"] == "way" and el.get("geometry"):
            p = _way_polygon([(g["lon"], g["lat"]) for g in el["geometry"]])
            if p is not None:
                polys.append(p)
        elif el["type"] == "relation":
            for m in el.get("members", []):
                if m.get("role") == "outer" and m.get("geometry"):
                    p = _way_polygon([(g["lon"], g["lat"]) for g in m["geometry"]])
                    if p is not None:
                        polys.append(p)
        for p in polys:
            p = shapely.make_valid(shapely.ops.transform(to_utm.transform, p))
            if not p.is_empty:
                feats.append({"geom": p, "tags": tags})
    with open(PROTECTION_CACHE, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": f["tags"],
                        "geometry": json.loads(shapely.to_geojson(f["geom"])),
                    }
                    for f in feats
                ],
            },
            fh,
        )
    return feats


def cell_mask(geoms, sub, gran):
    gt, rows, cols = sub["gt"], sub["rows"], sub["cols"]
    xs = gt[0] + (np.arange(cols) + 0.5) * gran
    ys = gt[3] - (np.arange(rows) + 0.5) * gran
    gx, gy = np.meshgrid(xs, ys)
    if not geoms:
        return np.zeros((rows, cols), bool)
    return shapely.contains_xy(shapely.unary_union(geoms), gx, gy) & sub["inside"]


def shares(mask, cover: dict[str, np.ndarray]) -> dict[str, float]:
    n = int(mask.sum())
    if not n:
        return {}
    out = {k: round(float((mask & m).sum()) / n, 3) for k, m in cover.items()}
    covered = np.zeros_like(mask)
    for m in cover.values():
        covered |= m
    out["uncovered"] = round(float((mask & ~covered).sum()) / n, 3)
    out["cells"] = n
    return out


def main():
    params, layers, extents = _gallery.load_scenario(SCENARIO)
    extent = extents["main"]
    gran = float(params["grid_size_m"])
    districts = _val.fetch_districts()
    union = shapely.unary_union(list(districts.values()))

    real_sub = _gallery.substrate(extent, layers, gran)
    val_layers = dict(layers)
    val_layers["built"] = [g.difference(union) for g in layers["built"] if not g.difference(union).is_empty]
    val_layers["green"] = [g.difference(union) for g in layers["green"] if not g.difference(union).is_empty]
    val_layers["centres"] = [
        g for g in layers.get("centres", []) if not union.contains(g.representative_point())
    ]
    val_sub = _gallery.substrate(extent, val_layers, gran)

    plan_path = os.path.join(OUT, "plan.npy")
    if not os.path.exists(plan_path):
        raise SystemExit("run scripts/validate_freiburg.py first (plan.npy missing)")
    plan = np.load(plan_path)

    protection = fetch_protection(extent)
    print(f"protection polygons fetched: {len(protection)}")
    by_tag: dict[str, list] = {}
    for f in protection:
        t = f["tags"]
        key = (
            t.get("boundary")
            or t.get("leisure")
            or (f"landuse={t['landuse']}" if "landuse" in t else None)
            or (f"aeroway={t['aeroway']}" if "aeroway" in t else None)
            or "other"
        )
        by_tag.setdefault(key, []).append(f["geom"])

    cover = {
        "committed_green": cell_mask(layers.get("green", []), real_sub, gran),
        "committed_unbuildable": cell_mask(layers.get("unbuildable", []), real_sub, gran),
        "committed_industrial": cell_mask(
            [shapely.make_valid(shapely.geometry.shape(f["geometry"])) for f in
             json.load(open(os.path.join(SCENARIO, "industrial.geojson"), encoding="utf-8"))["features"]],
            real_sub, gran,
        ),
    }
    for key, geoms in sorted(by_tag.items()):
        cover[f"osm_{key}"] = cell_mask(geoms, real_sub, gran)

    dmask = _val.district_mask(real_sub, districts)
    real_district_built = (real_sub["origin"] == 1) & dmask
    new_built = np.isin(plan, (G.PLAN_BUILT, G.PLAN_CENTRE))

    to_wgs = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)
    gt = real_sub["gt"]
    labels = G._label_components(new_built, queen=True)
    clusters = []
    for i in range(1, int(labels.max()) + 1):
        m = labels == i
        rr, cc = np.nonzero(m)
        x = gt[0] + (cc.mean() + 0.5) * gran
        y = gt[3] - (rr.mean() + 0.5) * gran
        lon, lat = to_wgs.transform(x, y)
        clusters.append(
            {
                "cluster": i,
                "centroid_lonlat": [round(lon, 5), round(lat, 5)],
                "grid_rc": [int(rr.mean()), int(cc.mean())],
                "inside_districts_share": round(float((m & dmask).sum()) / int(m.sum()), 3),
                **shares(m, cover),
            }
        )
    clusters.sort(key=lambda c: -c["cells"])

    # symmetry: could the model rebuild the district land at all, and how is that land classified?
    district_buildable = int(((val_sub["state"] == 0) & (val_sub["origin"] != 0) & real_district_built).sum())
    report = {
        "district_land": {
            "real_built_cells": int(real_district_built.sum()),
            "re_entered_as_buildable": district_buildable,
            **shares(real_district_built, cover),
        },
        "regrown_clusters": clusters,
        "protection_names": sorted(
            {f["tags"].get("name", "?") for f in protection if f["tags"].get("name")}
        ),
    }
    out_path = os.path.join(OUT, "landclass.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))
    print("LANDCLASS-DONE")


if __name__ == "__main__":
    main()
