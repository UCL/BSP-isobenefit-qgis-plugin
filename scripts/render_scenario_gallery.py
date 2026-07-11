#!/usr/bin/env python3
"""Precompute the website's scenario-explorer gallery: for every scenario folder, run a curated
set of parameter presets through the REAL pipeline (rasterise -> CA growth -> post-process) and
render each result as a tiered dot-grid SVG plus coverage metrics.

    .venv/bin/python scripts/render_scenario_gallery.py [scenarios/<name> ...]   # default: all

Outputs: website/public/gallery/<entry>/<preset>.png and website/public/gallery/gallery.json.

Web previews run at a COARSENED grid (max ~150 cells a side) with single deterministic runs, so
the whole gallery stays computable and the images stay light; formal runs happen in QGIS at the
scenario's real resolution. Every preset records the exact dial changes and the seed, so any
panel can be reproduced.
"""

from __future__ import annotations

import json
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import isobenefit  # noqa: E402
import numpy as np  # noqa: E402
import shapely  # noqa: E402

from isobenefit_qgis import grid as G  # noqa: E402

OUT = os.path.join(REPO, "website", "public", "gallery")
MAX_CELLS = 150  # preview grids are capped at ~150 cells a side
DISPERSAL = {"off": 0.0, "moderate": 0.0001, "aggressive": 0.04}

# colours mirror the plugin palette exactly (same convention as website/scripts/demonstrators.py)
def _hex(rgb):
    return "#%02x%02x%02x" % tuple(int(v) for v in rgb)

BUILT_LOW, BUILT_MED, BUILT_HIGH = _hex(G._BUILT_LOW), _hex(G._BUILT_MED), _hex(G._BUILT_HIGH)
CENTRE_LOW, CENTRE_MED, CENTRE_HIGH = _hex(G._CENTRE_LOW), _hex(G._CENTRE_MED), _hex(G._CENTRE_HIGH)
EXIST_BUILT, EXIST_CENTRE = _hex(G._EXIST_BUILT), _hex(G._EXIST_CENTRE)
GREEN, UNBUILDABLE, STREET, INK = _hex((89, 176, 60)), "#6f9fcf", "#a9a9a9", "#333333"
TIER_STYLE = {
    G.PLAN_GREEN: (GREEN, 0.18),
    G.PLAN_EXIST_BUILT: (EXIST_BUILT, 0.42),
    G.PLAN_EXIST_CENTRE: (EXIST_CENTRE, 0.46),
    G.PLAN_BUILT_LOW: (BUILT_LOW, 0.42),
    G.PLAN_BUILT_MED: (BUILT_MED, 0.42),
    G.PLAN_BUILT_HIGH: (BUILT_HIGH, 0.42),
    G.PLAN_CENTRE_LOW: (CENTRE_LOW, 0.46),
    G.PLAN_CENTRE_MED: (CENTRE_MED, 0.46),
    G.PLAN_CENTRE_HIGH: (CENTRE_HIGH, 0.46),
    G.PLAN_BUILT: (BUILT_MED, 0.42),
    G.PLAN_CENTRE: (CENTRE_MED, 0.46),
}

# The curated presets. Each is (id, label, note, overrides); overrides patch the scenario's
# params.json. "clustering" picks the centre-spacing multiple in post-processing.
def presets_for(name: str, params: dict) -> list[dict]:
    base = [
        {"id": "baseline", "label": "Baseline run", "note": "The scenario's own params.json, as shipped."},
        {"id": "walk800", "label": "Longer walk (800 m)",
         "note": "Centres serve an 800 m walk; growth reaches further from each centre.",
         "overrides": {"centre_walk_m": 800.0, "green_walk_m": max(400.0, params.get("green_walk_m", 400.0))}},
        {"id": "compact", "label": "Compact (dispersal off)",
         "note": "No leapfrogging: one contiguous settlement.", "overrides": {"dispersal": "off"}},
        {"id": "dispersed", "label": "Dispersed (aggressive)",
         "note": "Satellites leapfrog readily across the window.", "overrides": {"dispersal": "aggressive"}},
        {"id": "denser", "label": "Denser mix",
         "note": "Shares shifted one step toward the high tier; the same target houses on less land.",
         "overrides": {"shares": {"high": 0.5, "medium": 0.3, "low": 0.2}}},
        {"id": "tight", "label": "Tightly clustered centres",
         "note": "The same growth, post-processed to fewer, larger mixed-use centres.",
         "overrides": {"clustering": 2.5}},
    ]
    if params.get("dispersal") == "off":  # baseline already compact: swap that preset for moderate
        for p in base:
            if p["id"] == "compact":
                p.update(id="moderate", label="Moderate dispersal",
                         note="Some leapfrogging allowed (the baseline is compact).",
                         overrides={"dispersal": "moderate"})
    return base


def load_scenario(folder: str):
    with open(os.path.join(folder, "params.json"), encoding="utf-8") as fh:
        params = json.load(fh)
    layers = {}
    for name in ("built", "green", "unbuildable", "centres", "streets"):
        path = os.path.join(folder, f"{name}.geojson")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                fc = json.load(fh)
            layers[name] = [shapely.make_valid(shapely.geometry.shape(f["geometry"])) for f in fc["features"]]
    # terrain: steep.geojson bands at/above the scenario's slope_max_deg preclude development
    steep_path = os.path.join(folder, "steep.geojson")
    slope_max = params.get("slope_max_deg")
    if slope_max is not None and os.path.exists(steep_path):
        with open(steep_path, encoding="utf-8") as fh:
            fc = json.load(fh)
        layers.setdefault("unbuildable", []).extend(
            shapely.make_valid(shapely.geometry.shape(f["geometry"]))
            for f in fc["features"]
            if float(f["properties"].get("min_slope_deg", 0)) >= float(slope_max)
        )
    extents = {}
    for name in sorted(os.listdir(folder)):
        if name.startswith("extents") and name.endswith(".geojson"):
            with open(os.path.join(folder, name), encoding="utf-8") as fh:
                fc = json.load(fh)
            key = name.replace("extents", "").replace(".geojson", "").strip("_") or "main"
            extents[key] = shapely.unary_union(
                [shapely.make_valid(shapely.geometry.shape(f["geometry"])) for f in fc["features"]]
            )
    return params, layers, extents


def substrate(extent, layers, gran):
    xmin, ymin, xmax, ymax = extent.bounds
    rows, cols = G.align_bounds(xmin, ymin, xmax, ymax, gran)[:2]
    gt = (math.floor(xmin / gran) * gran, gran, 0.0, math.ceil(ymax / gran) * gran, 0.0, -gran)
    xs = gt[0] + (np.arange(cols) + 0.5) * gran
    ys = gt[3] - (np.arange(rows) + 0.5) * gran
    gx, gy = np.meshgrid(xs, ys)

    def mask(geoms):
        if not geoms:
            return np.zeros((rows, cols), bool)
        u = shapely.unary_union(geoms)
        return shapely.contains_xy(u, gx, gy)

    inside = shapely.contains_xy(extent, gx, gy)
    state = np.full((rows, cols), -1, np.int16)
    state[inside] = 0
    origin = np.full((rows, cols), -1, np.int16)
    built = mask(layers.get("built", [])) & inside
    green = mask(layers.get("green", [])) & inside
    unb = mask(layers.get("unbuildable", [])) & inside
    state[built] = 1
    origin[built] = 1
    origin[green & ~built] = 0
    state[unb & ~built] = -1
    seeds = []
    for geom in layers.get("centres", []):
        p = geom if geom.geom_type == "Point" else geom.representative_point()
        c, r = int((p.x - gt[0]) / gran), int((gt[3] - p.y) / gran)
        if 0 <= r < rows and 0 <= c < cols and built[r, c]:
            seeds.append((r, c))
    return {"state": state, "origin": origin, "seeds": sorted(set(seeds)), "gt": gt,
            "rows": rows, "cols": cols, "extent": extent}


def _rgb(hexcol):
    return tuple(int(hexcol[i : i + 2], 16) for i in (1, 3, 5))


def run_preset(sub, params, preset):
    p = dict(params)
    over = preset.get("overrides", {})
    for k, v in over.items():
        if k == "shares":
            p["shares"] = v
        elif k != "clustering":
            p[k] = v
    gran = p["_gran"]
    tiers = (p["densities_km2"]["high"], p["densities_km2"]["medium"], p["densities_km2"]["low"])
    shares = (p["shares"]["high"], p["shares"]["medium"], p["shares"]["low"])
    walk = float(p.get("centre_walk_m", 400.0))
    green_walk = float(p.get("green_walk_m", walk))
    max_walk = max(walk, green_walk)
    sim = isobenefit.Simulation(
        sub["state"].copy(), sub["origin"].copy(), np.zeros_like(sub["state"], np.float32), sub["seeds"],
        gran, max_walk, float(p["target_population"]), float(p.get("min_green_span_m", 400.0)),
        float(p.get("build_prob", 0.25)), 0.01, DISPERSAL.get(str(p.get("dispersal", "moderate")), 0.0001),
        0.8, shares, tiers, int(p.get("max_iterations", 300)), int(p.get("random_seed", 42)),
    )
    sim.run()
    st = np.asarray(sim.snapshot()["state"])
    spacing = float(over.get("clustering", 1.5)) * walk
    plan, metrics, _pre, _best = G.select_plan(
        [st], gran, float(p.get("min_green_span_m", 400.0)), max_walk,
        existing_built=(sub["origin"] == 1), existing_green=(sub["origin"] == 0),
        existing_centres=sub["seeds"], centre_spacing_m=spacing,
        centre_distance_m=walk, green_distance_m=green_walk,
        new_density_km2=sum(s * d for s, d in zip(shares, tiers)),
        centre_min_settlement=max(1, round(2.0 * 1e4 / gran**2)),
    )
    dens = G.derive_density(plan, gran, walk, tiers, shares)
    disp = G.to_tiered_plan(plan, dens, tiers)
    return disp, metrics


def render_png(codes, layers, sub, gran, path):
    """Dot-grid PNG with the street underlay: the same visual language as the site's SVGs at a
    fraction of the size (a 150-cell SVG carries ~20k circle elements; the PNG is tens of kB)."""
    from PIL import Image, ImageDraw

    H, W = codes.shape
    P, PAD = 8, 10  # supersampled 2x then reduced, for smooth dots
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
                draw.line(pts, fill=_rgb(STREET), width=2)
    for r in range(H):
        for c in range(W):
            v = int(codes[r, c])
            if v not in TIER_STYLE:
                continue
            col, radf = TIER_STYLE[v]
            cx, cy = PAD + c * P + P / 2, PAD + r * P + P / 2
            rad = P * radf
            draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=_rgb(col))
    im = im.resize((cw // 2, ch // 2), Image.LANCZOS)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    im.save(path, optimize=True)


_SCHEMA_KEYS = (
    "crs", "grid_size_m", "max_iterations", "target_population", "build_prob", "dispersal",
    "random_seed", "centre_walk_m", "green_walk_m", "optimise_centres", "centre_m2_per_person",
    "min_settlement_ha", "min_green_span_m", "densities_km2", "shares", "ensemble", "ensemble_runs",
)


def merged_formal_params(params: dict, preset: dict, entry_name: str) -> dict:
    """The FULL-resolution parameter set for a preset, in the plugin's params schema, so the file
    downloads straight into the dialog's Load parameters button. Post-processing choices that are
    not dialog parameters (the clustering option) are noted rather than encoded."""
    p = {k: params[k] for k in _SCHEMA_KEYS if k in params}
    for k, v in preset.get("overrides", {}).items():
        if k in ("shares", "densities_km2"):
            p[k] = v
        elif k != "clustering":
            p[k] = v
    p["schema"] = "isobenefit-params/1"
    p["name"] = f"{entry_name}_{preset['id']}"
    note = preset["note"]
    if preset.get("overrides", {}).get("clustering"):
        note += " In QGIS, both clustering options are always written; open the tightly-clustered output layer."
    p["notes"] = note
    return p


def existing_panel(sub):
    """The place as downloaded, before any simulated growth."""
    plan = np.zeros_like(sub["state"], np.uint8)
    plan[sub["state"] == 0] = G.PLAN_GREEN
    plan[sub["origin"] == 0] = G.PLAN_GREEN
    plan[sub["origin"] == 1] = G.PLAN_EXIST_BUILT
    for r, c in sub["seeds"]:
        plan[r, c] = G.PLAN_EXIST_CENTRE
    return plan


def entry_for(folder: str, extent_key: str, extent, params, layers, title, subtitle):
    name = os.path.basename(folder) + ("" if extent_key == "main" else f"_{extent_key}")
    span = max(extent.bounds[2] - extent.bounds[0], extent.bounds[3] - extent.bounds[1])
    gran = max(float(params.get("grid_size_m", 25.0)), math.ceil(span / MAX_CELLS / 5) * 5)
    sub = substrate(extent, layers, gran)
    print(f"{name}: grid {sub['cols']}x{sub['rows']} at {gran:.0f} m, {len(sub['seeds'])} centre seeds")
    p = dict(params, _gran=gran)
    presets_out = []

    # panel 0: the existing situation, so every comparison starts from the before-picture
    rel = f"{name}/existing.png"
    render_png(existing_panel(sub), layers, sub, gran, os.path.join(OUT, rel))
    presets_out.append({
        "id": "existing", "label": "Existing (before growth)",
        "note": "The place as downloaded: existing fabric muted, its mixed-use centres magenta. "
                "No simulation has run.",
        "image": rel, "metrics": None, "settings": None, "params_file": None,
    })

    for preset in presets_for(name, params):
        disp, metrics = run_preset(sub, p, preset)
        rel = f"{name}/{preset['id']}.png"
        render_png(disp, layers, sub, gran, os.path.join(OUT, rel))
        keep = {k: round(float(metrics.get(k, 0)), 3) for k in
                ("served_coverage", "centre_access", "green_access", "population",
                 "centre_m2_per_person", "green_m2_per_person", "built_cells")}
        formal = merged_formal_params(params, preset, name)
        params_rel = f"{name}/{preset['id']}_params.json"
        with open(os.path.join(OUT, params_rel), "w", encoding="utf-8") as fh:
            json.dump(formal, fh, indent=2, ensure_ascii=False)
        presets_out.append({"id": preset["id"], "label": preset["label"], "note": preset["note"],
                            "overrides": preset.get("overrides", {}), "image": rel, "metrics": keep,
                            "settings": {k: v for k, v in formal.items() if k not in ("schema", "notes")},
                            "params_file": params_rel})
        print(f"  {preset['id']}: served {keep['served_coverage']:.0%}, pop {keep['population']:,.0f}")
    return {"id": name, "title": title, "subtitle": subtitle,
            "grid": f"{sub['cols']}x{sub['rows']} cells at {gran:.0f} m (preview resolution)",
            "seed": int(params.get("random_seed", 42)),
            "folder": os.path.basename(folder),
            "github": f"https://github.com/UCL/BSP-isobenefit-qgis-plugin/tree/main/scenarios/{os.path.basename(folder)}",
            "zip": f"scenarios/{os.path.basename(folder)}.zip",
            "presets": presets_out}


TITLES = {
    "cambourne": ("Cambourne, UK", "New-settlement growth: the reference demo"),
    "dnipro_A": ("Dnipro, Ukraine: Area A", "Central right-bank regeneration (DBN norms)"),
    "dnipro_B": ("Dnipro, Ukraine: Area B", "Left-bank edge growth (DBN norms)"),
    "london_crews_hill": ("Crews Hill, London", "Green-belt release at the metropolitan edge"),
    "celina_tx": ("Celina, Texas", "US suburbia at the metropolitan fringe"),
    "kigali_east": ("Kigali, Rwanda", "Plan-guided rapid urbanisation"),
    "medellin_pajarito": ("Medellín, Colombia", "Planned hillside expansion"),
    "freiburg_rieselfeld": ("Freiburg, Germany", "Validation against Rieselfeld and Vauban"),
}


def main():
    folders = sys.argv[1:] or sorted(
        os.path.join("scenarios", d) for d in os.listdir("scenarios")
        if os.path.isdir(os.path.join("scenarios", d))
    )
    entries = []
    for folder in folders:
        params, layers, extents = load_scenario(folder)
        for key, extent in extents.items():
            name = os.path.basename(folder) + ("" if key == "main" else f"_{key}")
            title, subtitle = TITLES.get(name, (name, ""))
            pp = params
            if key == "B":  # dnipro's B preset file
                bpath = os.path.join(folder, "params_B.json")
                if os.path.exists(bpath):
                    with open(bpath, encoding="utf-8") as fh:
                        pp = json.load(fh)
            entries.append(entry_for(folder, key, extent, pp, layers, title, subtitle))
    # one downloadable ZIP per scenario folder (layers + params presets), served by the site
    import zipfile

    zip_dir = os.path.join(REPO, "website", "public", "scenarios")
    os.makedirs(zip_dir, exist_ok=True)
    for folder in folders:
        base = os.path.basename(folder)
        zpath = os.path.join(zip_dir, f"{base}.zip")
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname in sorted(os.listdir(folder)):
                if fname.startswith("_") or fname.endswith(".qgz"):
                    continue  # caches and personal QGIS projects stay out
                zf.write(os.path.join(folder, fname), arcname=f"{base}/{fname}")
        print(f"{zpath}: {os.path.getsize(zpath) // 1024} kB")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "gallery.json"), "w", encoding="utf-8") as fh:
        json.dump({"entries": entries}, fh, indent=1)
    print(f"gallery.json: {len(entries)} entries")


if __name__ == "__main__":
    main()
