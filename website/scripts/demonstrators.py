#!/usr/bin/env python3
"""Recommended-plan DEMONSTRATORS for the website: run the REAL pipeline (CA growth ->
select run -> post-process) on REAL downloaded data — the Cambourne snapshots in
scripts/data (fetched by fetch_data.py with the plugin's own OSM queries) — and render
the results as a fine dot-grid over the town's actual street network.

One geography threads the whole site: the input-layer panels (diagrams.py), these
growth demonstrators, and the hero recommended plan are all the same 4.2 km window.

    uv run --no-project --with core/dist/isobenefit-*.whl --with numpy --with shapely \
        python website/scripts/demonstrators.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

import isobenefit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from isobenefit_qgis import grid as G  # noqa: E402  (repo root added to path above)

GREEN, BUILT, EXIST_BUILT, RED, EXIST_CENTRE = "#2f7d33", "#3a3a3a", "#7d6240", "#D32333", "#962858"
UNBUILDABLE = "#6f9fcf"  # water / industrial / barrier corridors, matching the input-layer panels
STREET = "#a9a9a9"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.environ.get("DEMO_OUT", os.path.join(os.path.dirname(HERE), "public", "images"))
GRAN = 50.0  # m per cell — the demonstration window is 84 x 84 cells (4.2 km)

# the same walk/density dials the dialog defaults to
WALK, GREEN_SPAN, MIN_D, MAX_D, EXIST_D = 400.0, 400.0, 1500.0, 6000.0, 2000.0
DENSITY_TIERS = (MAX_D, 0.5 * (MIN_D + MAX_D), MIN_D)
TIER_PROBS = (1 / 3, 1 / 3, 1 / 3)


def _features(name):
    with open(os.path.join(DATA, f"{name}.geojson"), encoding="utf-8") as fh:
        return json.load(fh)["features"]


def _window():
    ring = _features("extents")[0]["geometry"]["coordinates"][0]
    xs, ys = zip(*ring)
    return min(xs), min(ys), max(xs), max(ys)


def _union(name):
    import shapely

    geoms = [shapely.make_valid(shapely.geometry.shape(f["geometry"])) for f in _features(name)]
    return shapely.unary_union(geoms) if geoms else None


def substrate():
    """Rasterise the downloaded layers to the CA grid, using the plugin's conventions:
    state -1 unbuildable / 0 empty / 1 built; origin -1 free / 0 fixed green / 1 existing
    built; existing centre POLYGONS become true-area seeds (every covered cell)."""
    import shapely

    xmin, ymin, xmax, ymax = _window()
    rows, cols = int(round((ymax - ymin) / GRAN)), int(round((xmax - xmin) / GRAN))
    xs = xmin + (np.arange(cols) + 0.5) * GRAN
    ys = ymax - (np.arange(rows) + 0.5) * GRAN
    gx, gy = np.meshgrid(xs, ys)

    def mask(name):
        u = _union(name)
        return shapely.contains_xy(u, gx, gy) if u is not None else np.zeros((rows, cols), bool)

    state = np.zeros((rows, cols), np.int16)
    origin = np.full((rows, cols), -1, np.int16)
    density = np.zeros((rows, cols), np.float32)
    built, green, unbuild, cent = mask("built"), mask("green"), mask("unbuildable"), mask("centres")
    state[built] = 1
    origin[built] = 1
    density[built] = EXIST_D
    origin[green & ~built] = 0
    state[unbuild & ~built] = -1
    seeds = [(int(r), int(c)) for r, c in np.argwhere(cent & built)]
    return {"state": state, "origin": origin, "density": density, "seeds": seeds,
            "rows": rows, "cols": cols, "window": (xmin, ymin, xmax, ymax)}


def grow(sub, pop=12000.0, isol=0.0001, seed=11, bp=0.3, nb=0.01, iters=400, stages=()):
    """Step one CA run on the substrate; return (final state, {iter: state} at ``stages``)."""
    sim = isobenefit.Simulation(
        sub["state"].copy(), sub["origin"].copy(), sub["density"].copy(), sub["seeds"],
        GRAN, WALK, pop, GREEN_SPAN, bp, nb, isol, 0.8, TIER_PROBS, DENSITY_TIERS, EXIST_D, iters, seed,
    )
    snaps = {}
    while sim.current_iter < iters and sim.pop_target_ratio < 1.0:
        sim.step()
        if sim.current_iter in stages:
            snaps[sim.current_iter] = np.asarray(sim.snapshot()["state"]).copy()
    return np.asarray(sim.snapshot()["state"]), snaps


def to_plan(sub, st, spacing):
    plan, _, _, _ = G.select_plan(
        [st], GRAN, GREEN_SPAN, WALK,
        existing_built=sub["origin"] == 1, existing_centres=sub["seeds"],
        centre_spacing_m=spacing, centre_distance_m=WALK, green_distance_m=WALK,
        centre_min_settlement=8,
    )
    return plan


def state_codes(sub, st):
    """Raw CA state -> plan-style codes for rendering a growth stage (no post-processing)."""
    plan = np.zeros_like(st, np.uint8)
    seeds = set(sub["seeds"])
    plan[st == 1] = G.PLAN_BUILT
    plan[(st == 1) & (sub["origin"] == 1)] = G.PLAN_EXIST_BUILT
    plan[st == 2] = G.PLAN_CENTRE
    for r, c in seeds:
        if st[r, c] == 2:
            plan[r, c] = G.PLAN_EXIST_CENTRE
    plan[(sub["origin"] == 0) & (plan == 0)] = G.PLAN_GREEN
    return plan


def streets_underlay(sub, P, x0, y0):
    """The town's actual street network, in the same transform as the dot grid — vector
    geography under simulation dots, the site's visual contract."""
    xmin, _ymin, _xmax, ymax = sub["window"]
    walkable = {"footway", "path", "cycleway", "steps", "track", "service"}
    out = []
    for f in _features("streets"):
        if f["properties"].get("highway") in walkable:
            continue  # keep the underlay legible: streets, not every footpath
        pts = " ".join(
            f"{x0 + (x - xmin) / GRAN * P:.1f},{y0 + (ymax - y) / GRAN * P:.1f}"
            for x, y in f["geometry"]["coordinates"]
        )
        out.append(f'<polyline points="{pts}" fill="none" stroke="{STREET}" stroke-width="2.2"/>')
    return "".join(out)



def _legend(items, x0, ly, avail_w):
    """Horizontal legend: one column per item, its dot centred ABOVE the label, all
    vertically aligned."""
    col_w = avail_w / len(items)
    out = []
    for i, (label, color, rad, op) in enumerate(items):
        cx = x0 + i * col_w + col_w / 2
        out.append(f'<circle cx="{cx:.1f}" cy="{ly:.1f}" r="{rad}" fill="{color}" opacity="{op}"/>')
        out.append(f'<text x="{cx:.1f}" y="{ly + 26:.1f}" fill="{BUILT}" '
                   f'font-family="Arial" font-size="15" text-anchor="middle">{label}</text>')
    return "".join(out)


def render_likelihood(prob, sub, name, title, underlay=""):
    """The ensemble's development-likelihood layer: dot opacity = share of runs ending built."""
    H, Wc = prob.shape
    P, x0, y0 = 9, 44, 86
    cw, ch = Wc * P + 88, H * P + 230
    out = [f'<rect width="{cw}" height="{ch}" fill="white"/>',
           f'<text x="{x0}" y="52" fill="{RED}" font-weight="800" font-size="26">{title}</text>', underlay]
    exist, unb = sub["origin"] == 1, sub["state"] == -1
    for r in range(H):
        for c in range(Wc):
            x, y = x0 + c * P + P / 2, y0 + r * P + P / 2
            if exist[r, c]:
                out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{P * 0.42:.1f}" fill="{EXIST_BUILT}"/>')
            elif unb[r, c]:
                out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{P * 0.3:.1f}" fill="{UNBUILDABLE}"/>')
            elif prob[r, c] > 0.04:
                op = 0.15 + 0.85 * float(prob[r, c])
                out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{P * 0.42:.1f}" fill="{BUILT}" opacity="{op:.2f}"/>')
            else:
                out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{P * 0.18:.1f}" fill="{GREEN}"/>')
    out.append(_legend([("existing built", EXIST_BUILT, 9, 1.0), ("built in most runs", BUILT, 9, 1.0),
                        ("built in few runs", BUILT, 9, 0.25), ("unbuildable", UNBUILDABLE, 9, 1.0)],
                       x0, y0 + H * P + 40, Wc * P))
    doc = (f'<?xml version="1.0" encoding="UTF-8"?>'
           f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {cw} {ch}" font-family="Arial, sans-serif">'
           f'{"".join(out)}</svg>')
    with open(os.path.join(OUT, f"{name}.svg"), "w", encoding="utf-8") as fh:
        fh.write(doc)
    print(f"wrote {os.path.join(OUT, name)}.svg  ({Wc}x{H} cells)")


def render(plan, name, title, underlay="", unbuildable=None):
    H, Wc = plan.shape
    P, x0, y0 = 9, 44, 86
    cw, ch = Wc * P + 88, H * P + 230
    out = [f'<rect width="{cw}" height="{ch}" fill="white"/>']
    out.append(f'<text x="{x0}" y="52" fill="{RED}" font-weight="800" font-size="26">{title}</text>')
    out.append(underlay)

    def dot(c, r, color, rad):
        return f'<circle cx="{x0 + c * P + P / 2:.1f}" cy="{y0 + r * P + P / 2:.1f}" r="{rad}" fill="{color}"/>'

    for r in range(H):
        for c in range(Wc):
            v = plan[r, c]
            if v == G.PLAN_BUILT:
                out.append(dot(c, r, BUILT, P * 0.42))
            elif v == G.PLAN_EXIST_BUILT:
                out.append(dot(c, r, EXIST_BUILT, P * 0.42))
            elif v == G.PLAN_CENTRE:
                out.append(dot(c, r, RED, P * 0.46))
            elif v == G.PLAN_EXIST_CENTRE:
                out.append(dot(c, r, EXIST_CENTRE, P * 0.46))
            elif unbuildable is not None and unbuildable[r, c]:
                out.append(dot(c, r, UNBUILDABLE, P * 0.3))  # carved out, never developed
            else:  # green + untouched land: ONE consistent small dot, for contrast with built
                out.append(dot(c, r, GREEN, P * 0.18))
    # legend: dots above wrapped labels, one aligned horizontal row
    out.append(_legend([("nature", GREEN, 5, 1.0), ("unbuildable", UNBUILDABLE, 5, 1.0),
                        ("new built", BUILT, 9, 1.0), ("existing built", EXIST_BUILT, 9, 1.0),
                        ("new centre", RED, 9, 1.0), ("existing centre", EXIST_CENTRE, 9, 1.0)],
                       x0, y0 + H * P + 40, Wc * P))
    doc = (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {cw} {ch}" font-family="Arial, sans-serif">'
        f'{"".join(out)}</svg>'
    )
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{name}.svg"), "w", encoding="utf-8") as fh:
        fh.write(doc)
    print(f"wrote {os.path.join(OUT, name)}.svg  ({Wc}x{H} cells)")


def main():
    sub = substrate()
    print(f"substrate: {sub['rows']}x{sub['cols']} cells, {len(sub['seeds'])} existing centre cells, "
          f"{int((sub['origin'] == 1).sum())} existing built cells, "
          f"{int((sub['state'] == -1).sum())} unbuildable cells")
    P, x0, y0 = 9, 44, 86
    streets = streets_underlay(sub, P, x0, y0)
    unb = sub["state"] == -1

    def draw(plan, name, title):
        render(plan, name, title, underlay=streets, unbuildable=unb)

    # The data -> grid bridge: the downloaded layers rasterised to the simulation's cells,
    # including exactly which cells are carved out as unbuildable.
    start = state_codes(sub, sub["state"])
    for r, c in sub["seeds"]:
        start[r, c] = G.PLAN_EXIST_CENTRE
    draw(start, "demo_substrate", "The town as cells — the starting grid")

    # THE worked example: grow the real town, snapshotting the growth on the way
    final, snaps = grow(sub, stages=(6, 30))
    draw(to_plan(sub, final, 1.5 * WALK), "demo_recommended_plan", "Recommended plan — Cambourne, grown")
    for it, st in snaps.items():
        draw(state_codes(sub, st), f"demo_growth_{it:03d}", f"Growth — iteration {it}")
    draw(state_codes(sub, final), "demo_growth_final", "Growth — population target met")

    # The ensemble's uncertainty map: many runs -> share of runs each cell ends built
    sim = isobenefit.Simulation(
        sub["state"].copy(), sub["origin"].copy(), sub["density"].copy(), sub["seeds"],
        GRAN, WALK, 12000.0, GREEN_SPAN, 0.3, 0.01, 0.0001, 0.8, TIER_PROBS, DENSITY_TIERS, EXIST_D, 400, 11,
    )
    prob = np.asarray(isobenefit.ensemble_probability(sim, 11, 24))
    render_likelihood(prob, sub, "demo_likelihood", "Development likelihood — 24 runs blended",
                      underlay=streets)

    # Dispersed development on the same town: Off / Moderate / Aggressive
    for label, isol in (("off", 0.0), ("moderate", 0.0001), ("aggressive", 0.04)):
        st, _ = grow(sub, isol=isol)
        draw(to_plan(sub, st, 1.5 * WALK), f"demo_dispersal_{label}",
             f"Dispersed development — {label.capitalize()}")

    # Centre clustering: the SAME run and buildings, only the centre spacing differs
    st, _ = grow(sub, isol=0.0)
    for label, mult in (("moderate", 1.5), ("tight", 2.5)):
        draw(to_plan(sub, st, mult * WALK), f"demo_clustering_{label}",
             f"Centre clustering — {label.capitalize()}")

    # Build probability: cap iterations (huge pop) so the same elapsed time shows the RATE
    for label, bp in (("slow", 0.08), ("fast", 0.6)):
        st, _ = grow(sub, isol=0.0, bp=bp, pop=1_000_000.0, iters=60)
        draw(to_plan(sub, st, 1.5 * WALK), f"demo_buildprob_{label}",
             f"Build probability — {label.capitalize()} ({bp})")


if __name__ == "__main__":
    main()
