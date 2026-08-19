#!/usr/bin/env python3
"""Freiburg comparative case study: remove Rieselfeld and Vauban, regrow, compare.

The protocol from scenarios/freiburg_rieselfeld/params.json notes: delete the two
districts from the built layer (the committed file is never modified; deletion
happens in memory), regrow the same land toward the districts' real combined
population, and compare the result against what was actually built. The comparison
covers the best single run (footprint, population, walks) and the ensemble
build-likelihood surface (does the surface light up where the planners built and
stay dark where they kept parks).

    .venv/bin/python scripts/validate_freiburg.py [--runs 50]

Outputs land in temp/freiburg_validation/ (metrics.json and PNG panels); the two
paper panels are also copied to paper/figures/. District boundaries come from
Nominatim on first run and are cached as scenarios/freiburg_rieselfeld/_districts.geojson
so later runs are offline.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import time
import urllib.parse
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import isobenefit  # noqa: E402
import numpy as np  # noqa: E402
import shapely  # noqa: E402
import shapely.ops  # noqa: E402
from pyproj import Transformer  # noqa: E402

from isobenefit_qgis import grid as G  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "render_scenario_gallery", os.path.join(REPO, "scripts", "render_scenario_gallery.py")
)
_gallery = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gallery)

SCENARIO = os.path.join(REPO, "scenarios", "freiburg_rieselfeld")
OUT = os.path.join(REPO, "temp", "freiburg_validation")
FIGS = os.path.join(REPO, "paper", "figures")
DISTRICT_CACHE = os.path.join(SCENARIO, "_districts.geojson")
REALITY_CACHE = os.path.join(SCENARIO, "_reality.geojson")
DISTRICTS = ("Rieselfeld", "Vauban")

# Developed or withheld land the scenario's landuse-based extraction cannot see. Campus and
# construction land joins the frozen built fabric; cemeteries join unbuildable. Applied
# identically to the real and simulated substrates, so both start from the same availability.
REALITY_QUERY = """
[out:json][timeout:120];
(
  way["landuse"~"^(construction|cemetery|railway)$"]({bbox});
  relation["landuse"~"^(construction|cemetery|railway)$"]({bbox});
  way["amenity"~"^(hospital|university|grave_yard)$"]({bbox});
  relation["amenity"~"^(hospital|university|grave_yard)$"]({bbox});
);
out geom;
"""
REALITY_BUILT = {"landuse=construction", "landuse=railway", "amenity=hospital", "amenity=university"}
REALITY_UNBUILDABLE = {"landuse=cemetery", "amenity=grave_yard"}


def fetch_districts() -> dict[str, shapely.Geometry]:
    """District polygons in EPSG:25832, from the cache or Nominatim."""
    if os.path.exists(DISTRICT_CACHE):
        with open(DISTRICT_CACHE, encoding="utf-8") as fh:
            fc = json.load(fh)
        return {
            f["properties"]["name"]: shapely.make_valid(shapely.geometry.shape(f["geometry"]))
            for f in fc["features"]
        }
    tf = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
    feats = []
    for name in DISTRICTS:
        q = urllib.parse.urlencode(
            {
                "q": f"{name}, Freiburg im Breisgau, Germany",
                "format": "jsonv2",
                "polygon_geojson": 1,
                "limit": 5,
            }
        )
        req = urllib.request.Request(
            f"https://nominatim.openstreetmap.org/search?{q}",
            headers={"User-Agent": "BSP-isobenefit-qgis-plugin validation script"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            results = json.load(resp)
        poly = None
        for res in results:
            geom = res.get("geojson", {})
            if geom.get("type") in ("Polygon", "MultiPolygon"):
                poly = shapely.make_valid(shapely.geometry.shape(geom))
                break
        if poly is None:
            raise SystemExit(f"no polygon result from Nominatim for {name}")
        poly = shapely.ops.transform(tf.transform, poly)
        feats.append(
            {
                "type": "Feature",
                "properties": {"name": name},
                "geometry": json.loads(shapely.to_geojson(poly)),
            }
        )
        time.sleep(1)  # Nominatim asks for at most one request per second
    with open(DISTRICT_CACHE, "w", encoding="utf-8") as fh:
        json.dump({"type": "FeatureCollection", "features": feats}, fh)
    return {f["properties"]["name"]: shapely.geometry.shape(f["geometry"]) for f in feats}


def fetch_reality(extent) -> dict[str, list]:
    """Supplementary developed/withheld polygons in EPSG:25832, cached after first fetch.
    Returns {"built": [...], "unbuildable": [...]}."""
    if not os.path.exists(REALITY_CACHE):
        to_wgs = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)
        xmin, ymin, xmax, ymax = extent.bounds
        lon0, lat0 = to_wgs.transform(xmin, ymin)
        lon1, lat1 = to_wgs.transform(xmax, ymax)
        body = urllib.parse.urlencode(
            {"data": REALITY_QUERY.format(bbox=f"{lat0},{lon0},{lat1},{lon1}")}
        ).encode()
        endpoints = (
            "https://overpass-api.de/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter",
        )
        data = None
        last = None
        for attempt in range(6):
            req = urllib.request.Request(
                endpoints[attempt % len(endpoints)],
                data=body,
                headers={"User-Agent": "BSP-isobenefit-qgis-plugin reality-layer fetch"},
            )
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    data = json.load(resp)
                break
            except Exception as exc:  # noqa: BLE001 - retry across mirrors on any transport error
                last = exc
                print(f"overpass attempt {attempt + 1} failed: {exc}", file=sys.stderr)
                time.sleep(10 * (attempt + 1))
        if data is None:
            raise last
        to_utm = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)

        def rings(el):
            def ring(coords):
                if len(coords) >= 4 and coords[0] == coords[-1]:
                    return shapely.Polygon(coords)
                return None

            out = []
            if el["type"] == "way" and el.get("geometry"):
                p = ring([(g["lon"], g["lat"]) for g in el["geometry"]])
                if p is not None:
                    out.append(p)
            elif el["type"] == "relation":
                for m in el.get("members", []):
                    if m.get("role") == "outer" and m.get("geometry"):
                        p = ring([(g["lon"], g["lat"]) for g in m["geometry"]])
                        if p is not None:
                            out.append(p)
            return out

        feats = []
        for el in data.get("elements", []):
            tags = el.get("tags", {})
            keys = {f"{k}={tags[k]}" for k in ("landuse", "amenity") if k in tags}
            role = "built" if keys & REALITY_BUILT else "unbuildable" if keys & REALITY_UNBUILDABLE else None
            if role is None:
                continue
            for p in rings(el):
                p = shapely.make_valid(shapely.ops.transform(to_utm.transform, p))
                if not p.is_empty:
                    feats.append(
                        {
                            "type": "Feature",
                            "properties": {"role": role, **{k: tags[k] for k in ("landuse", "amenity", "name") if k in tags}},
                            "geometry": json.loads(shapely.to_geojson(p)),
                        }
                    )
        with open(REALITY_CACHE, "w", encoding="utf-8") as fh:
            json.dump({"type": "FeatureCollection", "features": feats}, fh)
    with open(REALITY_CACHE, encoding="utf-8") as fh:
        fc = json.load(fh)
    out: dict[str, list] = {"built": [], "unbuildable": []}
    for f in fc["features"]:
        out[f["properties"]["role"]].append(
            shapely.make_valid(shapely.geometry.shape(f["geometry"]))
        )
    return out


def district_mask(sub, districts) -> np.ndarray:
    gt, rows, cols = sub["gt"], sub["rows"], sub["cols"]
    xs = gt[0] + (np.arange(cols) + 0.5) * gt[1]
    ys = gt[3] - (np.arange(rows) + 0.5) * gt[1]
    gx, gy = np.meshgrid(xs, ys)
    union = shapely.unary_union(list(districts.values()))
    return shapely.contains_xy(union, gx, gy) & sub["inside"]


def real_plan_for(sub, dmask, districts, layers) -> np.ndarray:
    """The place as built, with the two districts tagged as the 'new' development so
    evaluate_plan scores them on the same terms as a regrown plan."""
    plan = np.zeros_like(sub["state"], np.uint8)
    plan[sub["state"] == 0] = G.PLAN_GREEN
    plan[sub["origin"] == 0] = G.PLAN_GREEN
    plan[sub["origin"] == 1] = G.PLAN_EXIST_BUILT
    built = sub["origin"] == 1
    plan[built & dmask] = G.PLAN_BUILT
    union = shapely.unary_union(list(districts.values()))
    gt = sub["gt"]
    for r, c in sub["seeds"]:
        x = gt[0] + (c + 0.5) * gt[1]
        y = gt[3] - (r + 0.5) * gt[1]
        inside_district = union.contains(shapely.Point(x, y))
        plan[r, c] = G.PLAN_CENTRE if inside_district else G.PLAN_EXIST_CENTRE
    return plan


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Probability that a random positive cell outranks a random negative one."""
    if not len(pos) or not len(neg):
        return float("nan")
    neg_sorted = np.sort(neg)
    higher = np.searchsorted(neg_sorted, pos, side="left").astype(np.float64)
    ties = (np.searchsorted(neg_sorted, pos, side="right") - higher).astype(np.float64)
    return float((higher + 0.5 * ties).sum() / (len(pos) * len(neg)))


def likelihood_stats(built_freq, sub, real_district_built, district_parks, dmask):
    """How the build-likelihood surface meshes with what was actually built."""
    growable = (sub["state"] == 0) & (sub["origin"] != 0)
    background = growable & ~dmask
    return {
        "mean_likelihood_real_built": float(built_freq[real_district_built].mean()),
        "mean_likelihood_real_parks": float(built_freq[district_parks].mean()) if district_parks.any() else None,
        "mean_likelihood_other_growable": float(built_freq[background].mean()),
        "auc_real_built_vs_other_growable": _auc(built_freq[real_district_built], built_freq[background]),
        "auc_real_built_vs_real_parks": _auc(built_freq[real_district_built], built_freq[district_parks])
        if district_parks.any()
        else None,
    }


def render_likelihood(built_freq, sub, layers, districts, gran, path):
    """Dot-grid likelihood panel in the gallery's visual language: dot colour ramps with
    the share of runs that built the cell; district boundaries drawn as outlines."""
    from PIL import Image, ImageDraw

    H, W = built_freq.shape
    P, PAD = 8, 10
    cw, ch = W * P + 2 * PAD, H * P + 2 * PAD
    im = Image.new("RGB", (cw, ch), (255, 255, 255))
    draw = ImageDraw.Draw(im)
    gt = sub["gt"]
    for geom in layers.get("streets", []):
        for line in getattr(geom, "geoms", [geom]):
            if line.geom_type != "LineString":
                continue
            pts = [
                (PAD + (x - gt[0]) / gran * P, PAD + (gt[3] - y) / gran * P)
                for x, y in line.simplify(gran / 4).coords
            ]
            if len(pts) >= 2:
                draw.line(pts, fill=_gallery._rgb(_gallery.STREET), width=2)
    inside = sub["inside"]
    exist_built = sub["origin"] == 1
    protected_green = (sub["origin"] == 0) & ~exist_built
    unbuildable = (sub["state"] == -1) & inside
    lo, hi = np.array([236, 236, 232], float), np.array([178, 24, 43], float)
    for r in range(H):
        for c in range(W):
            if not inside[r, c]:
                continue
            if exist_built[r, c]:
                col, radf = _gallery._rgb(_gallery.EXIST_BUILT), 0.42
            elif unbuildable[r, c]:
                col, radf = _gallery._rgb(_gallery.UNBUILDABLE), 0.3
            elif protected_green[r, c]:
                col, radf = _gallery._rgb(_gallery.GREEN), 0.18
            else:
                f = float(built_freq[r, c])
                col, radf = tuple((lo + (hi - lo) * f).astype(int)), 0.18 + 0.28 * f
            cx, cy = PAD + c * P + P / 2, PAD + r * P + P / 2
            rad = P * radf
            draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=tuple(col))
    for geom in districts.values():
        for poly in getattr(geom, "geoms", [geom]):
            pts = [
                (PAD + (x - gt[0]) / gran * P, PAD + (gt[3] - y) / gran * P)
                for x, y in poly.exterior.coords
            ]
            draw.line(pts, fill=(51, 51, 51), width=3)
    im = im.resize((cw // 2, ch // 2), Image.LANCZOS)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    im.save(path, optimize=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument(
        "--keep-green",
        action="store_true",
        help="keep the committed protected-green layer intact inside the district boundaries "
        "(present-day designations, including the Rieselfeld reserve); the default removes it "
        "(the pre-plan reading, where the whole sewage field was available)",
    )
    ap.add_argument(
        "--min-green-span",
        type=float,
        default=None,
        help="override the scenario's minimum green span in metres (sensitivity sweep)",
    )
    args = ap.parse_args()
    out_dir = OUT + ("_keepgreen" if args.keep_green else "")
    if args.min_green_span is not None:
        out_dir += f"_span{args.min_green_span:.0f}"

    params, layers, extents = _gallery.load_scenario(SCENARIO)
    extent = extents["main"]
    gran = float(params["grid_size_m"])
    districts = fetch_districts()
    for name, geom in districts.items():
        print(f"{name}: district area {geom.area / 1e6:.2f} km2")

    # the reality layer closes the availability gap between the real world and the inputs:
    # campus, railway and construction land is developed, cemeteries are withheld
    reality = fetch_reality(extent)
    layers = dict(layers)
    layers["built"] = layers["built"] + reality["built"]
    layers["unbuildable"] = layers.get("unbuildable", []) + reality["unbuildable"]
    print(
        f"reality layer: {len(reality['built'])} developed polygons joined built, "
        f"{len(reality['unbuildable'])} withheld polygons joined unbuildable"
    )

    # the place as built (committed layers), and the validation inputs (districts removed)
    real_sub = _gallery.substrate(extent, layers, gran)
    union = shapely.unary_union(list(districts.values()))
    val_layers = dict(layers)
    val_layers["built"] = [g.difference(union) for g in layers["built"] if not g.difference(union).is_empty]
    if not args.keep_green:
        val_layers["green"] = [
            g.difference(union) for g in layers["green"] if not g.difference(union).is_empty
        ]
    val_layers["centres"] = [
        g for g in layers.get("centres", []) if not union.contains(g.representative_point())
    ]
    val_sub = _gallery.substrate(extent, val_layers, gran)
    print(
        f"grid {real_sub['cols']}x{real_sub['rows']} at {gran:.0f} m; "
        f"centre seeds {len(real_sub['seeds'])} real, {len(val_sub['seeds'])} after removal"
    )

    dmask = district_mask(real_sub, districts)
    real_district_built = (real_sub["origin"] == 1) & dmask
    n_real = int(real_district_built.sum())
    cell_km2 = gran * gran / 1e6
    real_density = params["target_population"] / (n_real * cell_km2)
    print(f"real districts: {n_real} built cells, implied density {real_density:,.0f} people/km2")

    tiers = (params["densities_km2"]["high"], params["densities_km2"]["medium"], params["densities_km2"]["low"])
    shares = (params["shares"]["high"], params["shares"]["medium"], params["shares"]["low"])
    mean_density = sum(s * d for s, d in zip(shares, tiers))
    walk = float(params["centre_walk_m"])
    green_walk = float(params["green_walk_m"])
    max_walk = max(walk, green_walk)
    green_span = (
        float(params["min_green_span_m"]) if args.min_green_span is None else args.min_green_span
    )
    print(f"minimum green span: {green_span:.0f} m")

    # score the real districts on the same yardstick as any candidate plan
    real_plan = real_plan_for(real_sub, dmask, districts, layers)
    real_metrics = G.evaluate_plan(
        real_plan, gran, max_walk, min_green_span_m=green_span,
        centre_distance_m=walk, green_distance_m=green_walk, new_density_km2=real_density,
        existing_green=(real_sub["origin"] == 0),
    )

    # regrow: the full ensemble at full resolution, then the standard selection pipeline
    min_cells = max(1, round(float(params["min_settlement_pop"]) / (mean_density * cell_km2)))
    val_state, val_origin = val_sub["state"].copy(), val_sub["origin"].copy()
    n_pocket = G.green_unviable_pockets(val_state, val_origin, min_cells)
    if n_pocket:
        print(f"unviable pockets marked protected green: {n_pocket} cells")
    val_sub["state"], val_sub["origin"] = val_state, val_origin
    states = []
    t0 = time.time()
    for i in range(args.runs):
        sim = isobenefit.Simulation(
            val_sub["state"].copy(), val_sub["origin"].copy(),
            np.zeros_like(val_sub["state"], np.float32), val_sub["seeds"],
            gran, max_walk, float(params["target_population"]),
            green_span, float(params["build_prob"]), 0.01,
            _gallery.DISPERSAL[str(params["dispersal"])], 0.8, shares, tiers,
            int(params["max_iterations"]), int(params["random_seed"]) + i,
        )
        sim.run()
        states.append(np.asarray(sim.snapshot()["state"]))
    print(f"{args.runs} runs in {time.time() - t0:.1f} s")

    # the ensemble build-likelihood surface: share of runs in which each cell ended built
    built_freq = np.zeros(val_sub["state"].shape, np.float32)
    for st in states:
        built_freq += st >= 1
    built_freq /= len(states)
    built_freq[val_sub["origin"] == 1] = 0.0  # existing fabric is not simulated growth
    district_parks = (real_sub["origin"] == 0) & dmask & (real_sub["origin"] != 1)
    prob_stats = likelihood_stats(built_freq, val_sub, real_district_built, district_parks, dmask)

    plan, metrics, _pre, best_state = G.select_plan(
        states, gran, green_span, max_walk,
        existing_built=(val_sub["origin"] == 1), existing_green=(val_sub["origin"] == 0),
        existing_centres=val_sub["seeds"], centre_mode="placed",
        centre_distance_m=walk, green_distance_m=green_walk, new_density_km2=mean_density,
        centre_min_settlement=min_cells,
    )

    # footprint agreement: does the model choose the land that was actually built?
    new_built = np.isin(plan, (G.PLAN_BUILT, G.PLAN_CENTRE))
    n_new = int(new_built.sum())
    inter = int((new_built & real_district_built).sum())
    union_cells = int((new_built | real_district_built).sum())
    comparison = {
        "real_district_cells": n_real,
        "regrown_new_cells": n_new,
        "overlap_cells": inter,
        "recall_share_of_real_footprint": inter / n_real if n_real else 0.0,
        "precision_share_of_regrown": inter / n_new if n_new else 0.0,
        "iou": inter / union_cells if union_cells else 0.0,
        "share_of_growth_inside_districts": int((new_built & dmask).sum()) / n_new if n_new else 0.0,
        "best_run_index": next((i for i, s in enumerate(states) if s is best_state), -1),
        "runs": args.runs,
        "min_green_span_m": green_span,
    }
    for name, geom in districts.items():
        dm = district_mask(real_sub, {name: geom})
        comparison[f"new_cells_{name.lower()}"] = int((new_built & dm).sum())

    os.makedirs(out_dir, exist_ok=True)
    dens = G.derive_density(plan, gran, walk, tiers, shares)
    disp = G.to_tiered_plan(plan, dens, tiers)
    _gallery.render_png(disp, val_layers, val_sub, gran, os.path.join(out_dir, "regrown.png"))
    real_disp = real_plan.copy()  # real districts shown as mid-tier new fabric, centres mid-tier
    real_disp[real_plan == G.PLAN_BUILT] = G.PLAN_BUILT_MED
    real_disp[real_plan == G.PLAN_CENTRE] = G.PLAN_CENTRE_MED
    _gallery.render_png(real_disp, layers, real_sub, gran, os.path.join(out_dir, "real.png"))
    np.save(os.path.join(out_dir, "plan.npy"), plan)
    np.save(os.path.join(out_dir, "built_freq.npy"), built_freq)
    render_likelihood(built_freq, val_sub, val_layers, districts, gran, os.path.join(out_dir, "likelihood.png"))
    if not args.keep_green and args.min_green_span is None:  # paper figures track the default
        os.makedirs(FIGS, exist_ok=True)
        for name in ("real.png", "regrown.png", "likelihood.png"):
            shutil.copyfile(os.path.join(out_dir, name), os.path.join(FIGS, f"freiburg_validation_{name}"))

    result = {
        "real_districts": {k: real_metrics[k] for k in sorted(real_metrics)},
        "regrown_placed": {k: round(float(v), 4) for k, v in sorted(metrics.items())},
        "footprint": comparison,
        "likelihood": prob_stats,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    print(json.dumps(comparison, indent=2))
    print(json.dumps(prob_stats, indent=2))
    for side in ("real_districts", "regrown_placed"):
        m = result[side]
        print(
            f"{side}: pop {m.get('population', 0):,.0f}, served {m.get('served_coverage', 0):.0%}, "
            f"centre walk {m.get('centre_walk_mean', 0):,.0f} m, green walk {m.get('green_walk_mean', 0):,.0f} m"
        )
    print("VALIDATION-DONE")


if __name__ == "__main__":
    main()
