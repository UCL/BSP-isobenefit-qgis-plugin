#!/usr/bin/env python3
"""Scenario DEMONSTRATORS for the website: run the REAL pipeline (CA growth ->
select run -> post-process) on REAL downloaded data — the Cambourne snapshots in
scripts/data (fetched by fetch_data.py with the plugin's own OSM queries) — and render
the results as a fine dot-grid over the town's actual street network.

One geography threads the whole site: the input-layer panels (diagrams.py), these
growth demonstrators, and the hero scenario are all the same 4.2 km window.

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

def _hex(rgb):
    return "#%02x%02x%02x" % tuple(int(v) for v in rgb)


# Colours mirror the plugin's plan palette EXACTLY (isobenefit_qgis/grid.py), so the site reads as the
# plugin's own output: new development is coloured by density tier — built as a yellow->brown ramp,
# mixed-use centres as a reds ramp — and existing fabric is a single muted shade (it carries no density).
BUILT_LOW, BUILT_MED, BUILT_HIGH = _hex(G._BUILT_LOW), _hex(G._BUILT_MED), _hex(G._BUILT_HIGH)
CENTRE_LOW, CENTRE_MED, CENTRE_HIGH = _hex(G._CENTRE_LOW), _hex(G._CENTRE_MED), _hex(G._CENTRE_HIGH)
EXIST_BUILT, EXIST_CENTRE = _hex(G._EXIST_BUILT), _hex(G._EXIST_CENTRE)
GREEN = _hex((89, 176, 60))  # nature
RED = "#D32333"  # site-theme red for headings (centre dots use the CENTRE_* ramp)
UNBUILDABLE = "#6f9fcf"  # water / industrial / barrier corridors, matching the input-layer panels
STREET = "#a9a9a9"
INK = "#333333"  # legend and label text
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.environ.get("DEMO_OUT", os.path.join(os.path.dirname(HERE), "public", "images"))
GRAN = 50.0  # m per cell — the demonstration window is 84 x 84 cells (4.2 km)

# the same walk/density dials the dialog defaults to: three explicit tiers (people/km²) each with a
# probability (summing to 1), high -> low
WALK, GREEN_WALK, GREEN_SPAN = 800.0, 400.0, 400.0  # the plugin's defaults
MIN_SETTLEMENT_POP = 1000.0  # the plugin's default; converted to cells via the mean density
DENSITY_TIERS = (6000.0, 3000.0, 1500.0)  # high, med, low
TIER_PROBS = (0.2, 0.3, 0.5)

# per-cell dot colour + radius factor, keyed by the plan/tier codes to_tiered_plan emits
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
    G.PLAN_BUILT: (BUILT_MED, 0.42),  # fallbacks for an untiered plan
    G.PLAN_CENTRE: (CENTRE_MED, 0.46),
}


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
    # existing fabric carries no density and no population — it is context only
    origin[green & ~built] = 0
    state[unbuild & ~built] = -1
    seeds = [(int(r), int(c)) for r, c in np.argwhere(cent & built)]
    return {"state": state, "origin": origin, "density": density, "seeds": seeds,
            "rows": rows, "cols": cols, "window": (xmin, ymin, xmax, ymax)}


def grow(sub, pop=12000.0, isol=0.0001, seed=11, bp=0.3, nb=0.01, iters=400, stages=()):
    """Step one CA run on the substrate; return (final state, final density, {iter: (state, density)}
    at ``stages``). Density is carried so growth stages can be coloured by their drawn tier."""
    sim = isobenefit.Simulation(
        sub["state"].copy(), sub["origin"].copy(), sub["density"].copy(), sub["seeds"],
        GRAN, WALK, pop, GREEN_SPAN, bp, nb, isol, 0.8, TIER_PROBS, DENSITY_TIERS, iters, seed,
    )
    snaps = {}
    while sim.current_iter < iters and sim.pop_target_ratio < 1.0:
        sim.step()
        if sim.current_iter in stages:
            snap = sim.snapshot()
            snaps[sim.current_iter] = (np.asarray(snap["state"]).copy(), np.asarray(snap["density"]).copy())
    final = sim.snapshot()
    return np.asarray(final["state"]).copy(), np.asarray(final["density"]).copy(), snaps


def to_plan(sub, st, spacing):
    plan, _, _, _ = G.select_plan(
        [st], GRAN, GREEN_SPAN, WALK,
        existing_built=sub["origin"] == 1, existing_centres=sub["seeds"],
        centre_spacing_m=spacing, centre_distance_m=WALK, green_distance_m=GREEN_WALK,
        centre_min_settlement=max(
            1, round(MIN_SETTLEMENT_POP / (sum(p * d for p, d in zip(TIER_PROBS, DENSITY_TIERS)) * GRAN**2 / 1.0e6))
        ),
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


def tiered(plan):
    """Colour a post-processed plan by density tier, exactly as the plugin does: arrange the drawn
    densities by walking distance to the final centres, then map each new cell to its tier code."""
    dens = G.derive_density(plan, GRAN, WALK, DENSITY_TIERS, TIER_PROBS)
    return G.to_tiered_plan(plan, dens, DENSITY_TIERS)


def growth_codes(sub, st, dens):
    """Raw CA stage -> tier display codes, colouring each new cell by its DRAWN density tier (matching
    the plugin's growth animation). ``dens`` is the sim's per-block density, so compare in per-block."""
    plan = state_codes(sub, st)
    block = GRAN**2 / 1.0e6
    tiers_per_block = tuple(d * block for d in DENSITY_TIERS)
    return G.to_tiered_plan(plan, np.asarray(dens), tiers_per_block)


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



def _legend(groups, x0, y_top, avail_w):
    """A tidy grouped legend. Each group is a titled column; within it, rows of (swatch, label) with
    the colour swatch to the LEFT of a left-aligned label. Groups are spread evenly across ``avail_w``,
    so the new-development tiers read as a stacked high/medium/low scale beside the context classes."""
    def esc(s):
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    out = []
    col_w = min(230.0, avail_w / len(groups))  # cap the column width so wide figures don't spread the key out
    total = col_w * len(groups)
    start = x0 + max(0.0, (avail_w - total) / 2.0)  # centre the legend block within the available width
    row_h = 27
    for gi, (title, items) in enumerate(groups):
        cx = start + gi * col_w
        sx = cx + 9  # swatch centre
        tx = sx + 16  # label start (left-aligned)
        out.append(f'<text x="{cx:.1f}" y="{y_top:.1f}" fill="{INK}" font-family="Arial" '
                   f'font-size="15" font-weight="700">{esc(title)}</text>')
        for ri, (label, color, rad, op) in enumerate(items):
            ry = y_top + 24 + ri * row_h
            out.append(f'<circle cx="{sx:.1f}" cy="{ry - 5:.1f}" r="{rad}" fill="{color}" opacity="{op}"/>')
            out.append(f'<text x="{tx:.1f}" y="{ry:.1f}" fill="{INK}" '
                       f'font-family="Arial" font-size="14">{esc(label)}</text>')
    return "".join(out)


def render_likelihood(prob, sub, name, title, underlay=""):
    """The ensemble's development-likelihood layer: dot opacity = share of runs ending built."""
    H, Wc = prob.shape
    P, x0, y0 = 9, 16, 54
    cw, ch = Wc * P + 2 * x0, y0 + H * P + 120
    out = [f'<rect width="{cw}" height="{ch}" fill="white"/>',
           f'<text x="{x0}" y="34" fill="{RED}" font-weight="800" font-size="26">{title}</text>', underlay]
    exist, unb = sub["origin"] == 1, sub["state"] == -1
    for r in range(H):
        for c in range(Wc):
            x, y = x0 + c * P + P / 2, y0 + r * P + P / 2
            if exist[r, c]:
                out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{P * 0.42:.1f}" fill="{EXIST_BUILT}"/>')
            elif unb[r, c]:
                out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{P * 0.3:.1f}" fill="{UNBUILDABLE}"/>')
            elif prob[r, c] > 0.04:
                op = 0.30 + 0.70 * float(prob[r, c])
                out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{P * 0.42:.1f}" '
                           f'fill="{BUILT_MED}" opacity="{op:.2f}"/>')
            else:
                out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{P * 0.18:.1f}" fill="{GREEN}"/>')
    out.append(_legend([
        ("Development likelihood", [("built in most runs", BUILT_MED, 8, 1.0),
                                    ("built in few runs", BUILT_MED, 8, 0.35)]),
        ("Existing", [("existing built", EXIST_BUILT, 8, 1.0)]),
        ("Other", [("nature", GREEN, 6, 1.0), ("unbuildable", UNBUILDABLE, 6, 1.0)]),
    ], x0, y0 + H * P + 24, Wc * P))
    doc = (f'<?xml version="1.0" encoding="UTF-8"?>'
           f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {cw} {ch}" font-family="Arial, sans-serif">'
           f'{"".join(out)}</svg>')
    with open(os.path.join(OUT, f"{name}.svg"), "w", encoding="utf-8") as fh:
        fh.write(doc)
    print(f"wrote {os.path.join(OUT, name)}.svg  ({Wc}x{H} cells)")


def _grid_dots(codes, ox, oy, P, unbuildable) -> str:
    """One dot per cell for a tier-coded grid, drawn at origin (ox, oy) and pitch P."""
    out = []
    H, Wc = codes.shape
    for r in range(H):
        for c in range(Wc):
            cx, cy = ox + c * P + P / 2, oy + r * P + P / 2
            v = int(codes[r, c])
            if v in TIER_STYLE:
                color, radf = TIER_STYLE[v]
                out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{P * radf:.1f}" fill="{color}"/>')
            elif unbuildable is not None and unbuildable[r, c]:
                out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{P * 0.3:.1f}" fill="{UNBUILDABLE}"/>')
            else:
                out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{P * 0.18:.1f}" fill="{GREEN}"/>')
    return "".join(out)


# the shared four-column key: built and centre tiers, then existing fabric and other land
_TIER_LEGEND = [
    ("New built", [("high", BUILT_HIGH, 8, 1.0), ("medium", BUILT_MED, 8, 1.0), ("low", BUILT_LOW, 8, 1.0)]),
    ("Mixed-use centre", [("high", CENTRE_HIGH, 8, 1.0), ("medium", CENTRE_MED, 8, 1.0),
                          ("low", CENTRE_LOW, 8, 1.0)]),
    ("Existing", [("existing built", EXIST_BUILT, 8, 1.0), ("existing centre", EXIST_CENTRE, 8, 1.0)]),
    ("Other", [("nature", GREEN, 6, 1.0), ("unbuildable", UNBUILDABLE, 6, 1.0)]),
]

# the starting grid has no new development, so its key drops the New built / Mixed-use tier columns
_EXISTING_LEGEND = _TIER_LEGEND[2:]


def _write_svg(name, cw, ch, out) -> None:
    doc = (f'<?xml version="1.0" encoding="UTF-8"?>'
           f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {cw} {ch}" font-family="Arial, sans-serif">'
           f'{"".join(out)}</svg>')
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{name}.svg"), "w", encoding="utf-8") as fh:
        fh.write(doc)
    print(f"wrote {os.path.join(OUT, name)}.svg")


def render(plan, name, title, underlay="", unbuildable=None, legend=None):
    """Render a tier-coded plan (from ``tiered`` or ``growth_codes``): each cell takes its tier's
    colour via TIER_STYLE, so new built reads as a yellow->brown ramp and mixed-use centres as reds."""
    H, Wc = plan.shape
    P, x0, y0 = 9, 16, 54  # tight left/right (x0) and title-to-grid (y0) margins
    cw, ch = Wc * P + 2 * x0, y0 + H * P + 120  # just enough room for the grouped legend below
    out = [f'<rect width="{cw}" height="{ch}" fill="white"/>',
           f'<text x="{cw / 2:.1f}" y="34" fill="{RED}" font-weight="800" font-size="26" '
           f'text-anchor="middle">{title}</text>', underlay,
           _grid_dots(plan, x0, y0, P, unbuildable),
           _legend(legend or _TIER_LEGEND, x0, y0 + H * P + 24, Wc * P)]
    _write_svg(name, cw, ch, out)


def render_multi(sub, panels, name, title, unbuildable):
    """Several sub-maps in ONE figure under one title, sharing a single legend (no per-panel legend).
    ``panels`` is a list of ``(tier_codes, subtitle)``. Used for the growth stages and the clustering
    options, so the sub-figures read as one comparison rather than separate images."""
    H, Wc = panels[0][0].shape
    P, x0, y0 = 5, 16, 96  # smaller pitch so the panels sit side by side; y0 leaves a clear gap below the title
    pw, gap = Wc * P, 40
    n = len(panels)
    total_w = n * pw + (n - 1) * gap
    cw, ch = total_w + 2 * x0, y0 + H * P + 134  # extra room for the wider title/legend gaps
    out = [f'<rect width="{cw}" height="{ch}" fill="white"/>',
           f'<text x="{cw / 2:.1f}" y="38" fill="{RED}" font-weight="800" font-size="28" '
           f'text-anchor="middle">{title}</text>']
    for i, (codes, subtitle) in enumerate(panels):
        ox = x0 + i * (pw + gap)
        out.append(f'<text x="{ox + pw / 2:.1f}" y="{y0 - 14:.1f}" fill="{INK}" font-size="16" '
                   f'font-weight="700" text-anchor="middle">{subtitle}</text>')
        out.append(streets_underlay(sub, P, ox, y0))
        out.append(_grid_dots(codes, ox, y0, P, unbuildable))
    out.append(_legend(_TIER_LEGEND, x0, y0 + H * P + 42, total_w))  # clear gap between panels and legend
    _write_svg(name, cw, ch, out)


def main():
    sub = substrate()
    print(f"substrate: {sub['rows']}x{sub['cols']} cells, {len(sub['seeds'])} existing centre cells, "
          f"{int((sub['origin'] == 1).sum())} existing built cells, "
          f"{int((sub['state'] == -1).sum())} unbuildable cells")
    P, x0, y0 = 9, 16, 54  # must match render()'s grid transform so the street underlay aligns
    streets = streets_underlay(sub, P, x0, y0)
    unb = sub["state"] == -1

    def draw(plan, name, title):
        render(plan, name, title, underlay=streets, unbuildable=unb)

    # The data -> grid bridge: the downloaded layers rasterised to the simulation's cells,
    # including exactly which cells are carved out as unbuildable.
    start = state_codes(sub, sub["state"])
    for r, c in sub["seeds"]:
        start[r, c] = G.PLAN_EXIST_CENTRE
    # the starting grid has no new development, so drop the New built / Mixed-use tiers from its key
    render(start, "demo_substrate", "The town as cells: the starting grid",
           underlay=streets, unbuildable=unb, legend=_EXISTING_LEGEND)

    # THE worked example: grow the real town, snapshotting the growth on the way. The three growth
    # stages are ONE figure (three sub-panels, one shared legend), so they read as a single sequence.
    final, final_dens, snaps = grow(sub, stages=(6, 30))
    draw(tiered(to_plan(sub, final, 1.5 * WALK)), "demo_recommended_plan", "An idealised scenario: Cambourne, grown")
    render_multi(sub, [
        (growth_codes(sub, snaps[6][0], snaps[6][1]), "Iteration 6"),
        (growth_codes(sub, snaps[30][0], snaps[30][1]), "Iteration 30"),
        (growth_codes(sub, final, final_dens), "Target met (raw)"),
    ], "demo_growth", "The rules at work on the real town", unb)

    # The ensemble's uncertainty map: many runs -> share of runs each cell ends built
    sim = isobenefit.Simulation(
        sub["state"].copy(), sub["origin"].copy(), sub["density"].copy(), sub["seeds"],
        GRAN, WALK, 12000.0, GREEN_SPAN, 0.3, 0.01, 0.0001, 0.8, TIER_PROBS, DENSITY_TIERS, 400, 11,
    )
    prob = np.asarray(isobenefit.ensemble_probability(sim, 11, 24))
    render_likelihood(prob, sub, "demo_likelihood", "Development likelihood: 24 runs blended",
                      underlay=streets)

    # Dispersed development on the same town: Off / Moderate / Aggressive
    for label, isol in (("off", 0.0), ("moderate", 0.0001), ("aggressive", 0.04)):
        st, _dens, _ = grow(sub, isol=isol)
        draw(tiered(to_plan(sub, st, 1.5 * WALK)), f"demo_dispersal_{label}",
             f"Dispersed development: {label.capitalize()}")

    # Centre clustering: the SAME run and buildings, only the centre spacing differs. The two options
    # are ONE figure (two sub-panels, one shared legend) so they read as a direct comparison.
    st, _dens, _ = grow(sub, isol=0.0)
    render_multi(sub, [
        (tiered(to_plan(sub, st, 1.5 * WALK)), "Moderately clustered"),
        (tiered(to_plan(sub, st, 2.5 * WALK)), "Tightly clustered"),
    ], "demo_clustering", "Centre clustering options", unb)

    # Build probability: cap iterations (huge pop) so the same elapsed time shows the RATE
    for label, bp in (("slow", 0.08), ("fast", 0.6)):
        st, _dens, _ = grow(sub, isol=0.0, bp=bp, pop=1_000_000.0, iters=60)
        draw(tiered(to_plan(sub, st, 1.5 * WALK)), f"demo_buildprob_{label}",
             f"Build probability: {label.capitalize()} ({bp})")


if __name__ == "__main__":
    main()
