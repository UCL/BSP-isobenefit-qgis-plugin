#!/usr/bin/env python3
"""Ground-truth check for the Freiburg comparison's northeast growth area.

The built layer derives from OSM landuse polygons (residential, commercial,
retail). Land that carries buildings without such a polygon rasterises as open
and growable. This script fetches, for each regrown cluster outside the
district boundaries and for a control box on Vauban, every landuse polygon and
every building footprint, and reports what actually covers those cells.

Run scripts/validate_freiburg.py first (plan.npy is read from its output).

    .venv/bin/python scripts/check_freiburg_builtgap.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
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

QUERY = """
[out:json][timeout:120];
(
  way["landuse"]({bbox});
  relation["landuse"]({bbox});
  way["building"]({bbox});
);
out geom;
"""


def _polys_from_element(el):
    def ring(coords):
        if len(coords) >= 4 and coords[0] == coords[-1]:
            return shapely.Polygon(coords)
        return None

    polys = []
    if el["type"] == "way" and el.get("geometry"):
        p = ring([(g["lon"], g["lat"]) for g in el["geometry"]])
        if p is not None:
            polys.append(p)
    elif el["type"] == "relation":
        for m in el.get("members", []):
            if m.get("role") == "outer" and m.get("geometry"):
                p = ring([(g["lon"], g["lat"]) for g in m["geometry"]])
                if p is not None:
                    polys.append(p)
    return polys


ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)


def fetch_box(bbox_wgs: str):
    body = urllib.parse.urlencode({"data": QUERY.format(bbox=bbox_wgs)}).encode()
    last = None
    for attempt in range(4):
        url = ENDPOINTS[attempt % len(ENDPOINTS)]
        req = urllib.request.Request(
            url, data=body, headers={"User-Agent": "BSP-isobenefit-qgis-plugin built-gap check"}
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.load(resp)
        except Exception as exc:  # noqa: BLE001 - retry across mirrors on any transport error
            last = exc
            print(f"overpass attempt {attempt + 1} via {url} failed: {exc}", file=sys.stderr)
            time.sleep(10 * (attempt + 1))
    raise last


def analyse_box(name, mask, sub, gran, to_wgs, to_utm):
    rr, cc = np.nonzero(mask)
    gt = sub["gt"]
    pad = 100.0
    x0 = gt[0] + cc.min() * gran - pad
    x1 = gt[0] + (cc.max() + 1) * gran + pad
    y1 = gt[3] - rr.min() * gran + pad
    y0 = gt[3] - (rr.max() + 1) * gran - pad
    lon0, lat0 = to_wgs.transform(x0, y0)
    lon1, lat1 = to_wgs.transform(x1, y1)
    data = fetch_box(f"{lat0},{lon0},{lat1},{lon1}")

    landuse_geoms: dict[str, list] = {}
    building_geoms = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        for p in _polys_from_element(el):
            p = shapely.make_valid(shapely.ops.transform(to_utm.transform, p))
            if p.is_empty:
                continue
            if "building" in tags:
                building_geoms.append(p)
            elif "landuse" in tags:
                landuse_geoms.setdefault(tags["landuse"], []).append(p)

    xs = gt[0] + (np.arange(sub["cols"]) + 0.5) * gran
    ys = gt[3] - (np.arange(sub["rows"]) + 0.5) * gran
    gx, gy = np.meshgrid(xs, ys)
    n = int(mask.sum())
    out = {"box": name, "cells": n, "buildings": len(building_geoms)}
    for value, geoms in sorted(landuse_geoms.items()):
        share = float((shapely.contains_xy(shapely.unary_union(geoms), gx, gy) & mask).sum()) / n
        if share >= 0.01:
            out[f"landuse={value}"] = round(share, 3)
    if building_geoms:
        u = shapely.unary_union(building_geoms)
        out["building_cover_share"] = round(
            float((shapely.contains_xy(u, gx, gy) & mask).sum()) / n, 3
        )
        cluster_cells_area = n * gran * gran
        out["building_footprint_m2_per_cell_area"] = round(float(u.area) / cluster_cells_area, 3)
    return out


def main():
    params, layers, extents = _gallery.load_scenario(_val.SCENARIO)
    extent = extents["main"]
    gran = float(params["grid_size_m"])
    districts = _val.fetch_districts()
    union = shapely.unary_union(list(districts.values()))
    real_sub = _gallery.substrate(extent, layers, gran)
    dmask = _val.district_mask(real_sub, districts)

    plan = np.load(os.path.join(_val.OUT, "plan.npy"))
    new_built = np.isin(plan, (G.PLAN_BUILT, G.PLAN_CENTRE))
    labels = G._label_components(new_built, queen=True)

    to_wgs = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)

    reports = []
    for i in range(1, int(labels.max()) + 1):
        m = labels == i
        if float((m & dmask).sum()) / m.sum() > 0.5:
            continue  # district-side clusters are not in question
        reports.append(analyse_box(f"regrown_cluster_{i}", m, real_sub, gran, to_wgs, to_utm))
        time.sleep(2)

    # control: Vauban's real footprint, which everyone agrees is built-up
    vauban = _val.district_mask(real_sub, {"Vauban": districts["Vauban"]})
    reports.append(analyse_box("control_vauban", vauban & (real_sub["origin"] == 1),
                               real_sub, gran, to_wgs, to_utm))

    with open(os.path.join(_val.OUT, "builtgap.json"), "w", encoding="utf-8") as fh:
        json.dump(reports, fh, indent=2)
    print(json.dumps(reports, indent=2))
    print("BUILTGAP-DONE")


if __name__ == "__main__":
    main()
