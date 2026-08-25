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
BUILD_PROB = 0.25  # paces the edge and varies members; does not shape the plan
# Growth leans toward the transit a scenario supplies, without being dictated by it. A
# scenario with no stops, stations or corridor is unaffected: the field is then absent and
# every draw runs at its ordinary rate.
DEFAULT_TRANSIT_PREFERENCE = 0.35
DISPERSAL = {"off": False, "moderate": True, "aggressive": True}


def centre_quota(p):
    """People a centre must gather before the next is earned: the viability threshold."""
    return float(p.get("min_settlement_pop", 2000.0))


def allow_detached(p):
    """May an earned centre start away from existing fabric?"""
    if "allow_detached" in p:
        return bool(p["allow_detached"])
    return DISPERSAL.get(str(p.get("dispersal", "moderate")), True)

# colours mirror the plugin palette exactly (same convention as website/scripts/demonstrators.py)
def _hex(rgb):
    return "#%02x%02x%02x" % tuple(int(v) for v in rgb)

BUILT_LOW, BUILT_MED, BUILT_HIGH = _hex(G._BUILT_LOW), _hex(G._BUILT_MED), _hex(G._BUILT_HIGH)
CENTRE_LOW, CENTRE_MED, CENTRE_HIGH = _hex(G._CENTRE_LOW), _hex(G._CENTRE_MED), _hex(G._CENTRE_HIGH)
EXIST_BUILT, EXIST_CENTRE = _hex(G._EXIST_BUILT), _hex(G._EXIST_CENTRE)
GREEN, STREET, INK = _hex((89, 176, 60)), "#a9a9a9", "#333333"
# Barriers are drawn solid and dark rather than as pale specks: a carved rail or river
# corridor can sever land that looks open on the map, and a reader who cannot see the
# wall reads the gap as available.
# Water is a saturated azure: intense enough to read as water, and hue-separated from the
# indigo of an existing centre, which reads purple (navy would sit between the two). The
# crossing is magenta, the one hue family the palette leaves free: yellows and oranges are
# the built tiers, reds the new centres, purples the existing fabric, teal the transit, and
# on a dark grey corridor a few magenta dots read as markers rather than as ground.
UNBUILDABLE, STEEP, WATER, CROSSING = "#5f6368", "#43484d", "#2e75b6", "#e0218a"
# One dot size for every class except nature: development, existing fabric, barriers and
# water all draw at the same middle weight, distinguished by colour rather than by bulk,
# and the nature field alone stays finer so the settlements read against it.
DOT_RAD = 0.34
GROUND_RAD = DOT_RAD
NATURE_RAD = 0.24

TRANSIT = "#0b7285"  # transit stop markers, the site's stops colour
TIER_STYLE = {
    G.PLAN_GREEN: (GREEN, NATURE_RAD),
    G.PLAN_EXIST_BUILT: (EXIST_BUILT, DOT_RAD),
    G.PLAN_EXIST_CENTRE: (EXIST_CENTRE, DOT_RAD),
    G.PLAN_BUILT_LOW: (BUILT_LOW, DOT_RAD),
    G.PLAN_BUILT_MED: (BUILT_MED, DOT_RAD),
    G.PLAN_BUILT_HIGH: (BUILT_HIGH, DOT_RAD),
    G.PLAN_CENTRE_LOW: (CENTRE_LOW, DOT_RAD),
    G.PLAN_CENTRE_MED: (CENTRE_MED, DOT_RAD),
    G.PLAN_CENTRE_HIGH: (CENTRE_HIGH, DOT_RAD),
    G.PLAN_BUILT: (BUILT_MED, DOT_RAD),
    G.PLAN_CENTRE: (CENTRE_MED, DOT_RAD),
}

# The curated presets. Each is (id, label, note, overrides); overrides patch the scenario's
# params.json. "centre_mode" picks the post-processing centre option (default "placed").

def transit_attraction(state, gran, stop_cells, hub_cells, stop_reach, hub_reach):
    """How strongly transit favours each cell: 1 at a stop, hub or corridor cell, falling
    linearly to 0 at the edge of that source's catchment, and the strongest pull wins where
    two overlap. A field rather than a mask, so growth is drawn toward transit by degree
    instead of stepping off a cliff at the catchment edge. Returns None when no transit is
    supplied, which leaves growth untouched."""
    import numpy as _np
    field = None
    for cells, reach in ((stop_cells, stop_reach), (hub_cells, hub_reach)):
        if not cells or reach <= 0:
            continue
        m = _np.zeros(state.shape, bool)
        for r, c in cells:
            m[r, c] = True
        d = G._walk_distance(m, gran, reach, blocked=(state == -1))
        pull = _np.where(_np.isfinite(d), 1.0 - _np.minimum(d, reach) / reach, 0.0)
        field = pull if field is None else _np.maximum(field, pull)
    return None if field is None else field.astype(_np.float32)

def presets_for(name: str, params: dict, has_stops: bool = False, has_centres: bool = True) -> list[dict]:
    base = [
        {"id": "baseline", "label": "Baseline run", "note": "The scenario's own params.json, as shipped."},
        {"id": "walk800", "label": "Shorter centre walk (800 m)",
         "note": "Centres serve an 800 m walk; at low densities the pooled demand within so "
                 "small a catchment can fall below viability.",
         "overrides": {"centre_walk_m": 800.0}},
        {"id": "walk1600", "label": "Longer centre walk (1,600 m)",
         "note": "Centres serve a 1,600 m walk; growth reaches much further from each centre.",
         "overrides": {"centre_walk_m": 1600.0}},
        {"id": "compact", "label": "Attached growth only",
         "note": "Every earned centre must join existing development: one contiguous settlement.",
         "overrides": {"allow_detached": False}},
        {"id": "viability4000", "label": "Larger centres (4,000 people)",
         "note": "A centre must gather 4,000 people before the next is earned, so the same "
                 "target arrives as fewer, larger settlements.",
         "overrides": {"min_settlement_pop": 4000.0}},
        {"id": "denser", "label": "Denser mix",
         "note": "Shares shifted one step toward the high tier; the same target houses on less land.",
         "overrides": {"shares": {"high": 0.5, "medium": 0.3, "low": 0.2}}},
        {"id": "fewest", "label": "Fewest centres",
         "note": "The same growth, post-processed to the fewest centres that keep every home "
                 "within the centre walk.",
         "overrides": {"centre_mode": "minimal"}},
    ]
    if params.get("allow_detached") is False:  # baseline already attached: show the converse
        for p in base:
            if p["id"] == "compact":
                p.update(id="detached", label="Detached settlements allowed",
                         note="An earned centre may start away from existing development "
                              "(the baseline keeps growth attached).",
                         overrides={"allow_detached": True})
    if not has_centres:
        # with no centre anywhere, every existing settlement is sterile and attached growth
        # has nothing to nucleate against: the preset would render an empty window
        base = [p for p in base if p["id"] != "compact"]
    if has_stops:
        base.append({"id": "corridor", "label": "Transit corridor preference",
                     "note": "Growth concentrates along transit: development draws outside the "
                             "transit catchments (400 m of a stop or drawn route, 1,200 m of a "
                             "hub) are scaled by one minus the corridor preference (0.95 here). "
                             "The teal line is a proposed bus route and the teal dots the "
                             "existing stops. The large ringed marker is a proposed bus rapid "
                             "transit stop treated as a transit hub: it anchors a pinned "
                             "centre, so a new settlement gathers around it.",
                     "overrides": {"corridor_weight": 0.95, "include_proposed": True}})
    return base


def load_scenario(folder: str):
    with open(os.path.join(folder, "params.json"), encoding="utf-8") as fh:
        params = json.load(fh)
    layers = {}
    for name in ("built", "green", "unbuildable", "centres", "streets", "walkable", "water",
                 "stops", "stations", "proposed_corridor", "proposed_station"):
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
        layers["steep"] = [
            shapely.make_valid(shapely.geometry.shape(f["geometry"]))
            for f in fc["features"]
            if float(f["properties"].get("min_slope_deg", 0)) >= float(slope_max)
        ]
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
    inside_mask = inside.copy()
    origin = np.full((rows, cols), -1, np.int16)
    built = mask(layers.get("built", [])) & inside
    green = mask(layers.get("green", [])) & inside
    unb = mask(layers.get("unbuildable", [])) & inside
    steep = mask(layers.get("steep", [])) & inside
    water = mask(layers.get("water", [])) & inside
    state[built] = 1
    origin[built] = 1
    origin[green & ~built] = 0
    state[unb & ~built] = -1
    state[steep & ~built] = -1
    # Existing centres seed as TRUE AREAS, exactly as the plugin rasterises polygon centre
    # layers: every covered built cell, not one cell at the centroid. A polygon too small to
    # cover any cell centre falls back to its representative point so no centre is lost.
    cent = mask(layers.get("centres", [])) & built
    seeds = [(int(r), int(c)) for r, c in np.argwhere(cent)]
    for geom in layers.get("centres", []):
        p = geom if geom.geom_type == "Point" else geom.representative_point()
        c, r = int((p.x - gt[0]) / gran), int((gt[3] - p.y) / gran)
        if 0 <= r < rows and 0 <= c < cols and built[r, c] and not cent[r, c]:
            seeds.append((r, c))
    # Pedestrian crossings: where a way people can walk crosses a carved barrier, the barrier
    # is passable there. A footbridge over a railway does not make the railway developable, so
    # the cell stays out of the buildable set and only the walk changes. Without this a
    # corridor severs land the planner can plainly reach: at Crews Hill it strands 238 ha of
    # developable ground 75 m from its station.
    crossing_cells = []
    for geom in layers.get("walkable", []):
        for line in getattr(geom, "geoms", [geom]):
            if line.geom_type != "LineString" or line.length <= 0:
                continue
            for f in np.arange(0.0, 1.0 + 1e-9, (gran / 3) / max(line.length, gran)):
                pt = line.interpolate(min(f, 1.0), normalized=True)
                c, r = int((pt.x - gt[0]) / gran), int((gt[3] - pt.y) / gran)
                if 0 <= r < rows and 0 <= c < cols and state[r, c] == -1:
                    crossing_cells.append((r, c))
    for r, c in set(crossing_cells):
        state[r, c] = G.STATE_CROSSING

    # transit stops: point features -> cells, snapped off unbuildable land exactly as the
    # plugin snaps them (a stop often sits on a carved road corridor)
    stops = []
    for geom in layers.get("stops", []):
        p = geom if geom.geom_type == "Point" else geom.representative_point()
        c, r = int((p.x - gt[0]) / gran), int((gt[3] - p.y) / gran)
        if 0 <= r < rows and 0 <= c < cols and inside[r, c]:
            stops.append((r, c))
    stops, _, _ = G.sanitise_seeds(sorted(set(stops)), state, gran, 2 * gran)
    # a proposed transit corridor (a hand-drawn line): every walkable cell the line touches
    # becomes a growth-anchor source, exactly as the plugin rasterises a drawn corridor layer
    corridor = []
    for geom in layers.get("proposed_corridor", []):
        for line in getattr(geom, "geoms", [geom]):
            if line.geom_type != "LineString":
                continue
            for f in np.arange(0.0, 1.0 + 1e-9, (gran / 2) / max(line.length, gran)):
                p = line.interpolate(min(f, 1.0), normalized=True)
                c, r = int((p.x - gt[0]) / gran), int((gt[3] - p.y) / gran)
                if 0 <= r < rows and 0 <= c < cols and inside[r, c] and state[r, c] != -1:
                    corridor.append((r, c))
    # rail and tram stations: pinned centre anchors in every run, exactly as the plugin
    # treats the stations layer (a station seeds a centre and post-processing pins it)
    stations = []
    for geom in layers.get("stations", []):
        p = geom if geom.geom_type == "Point" else geom.representative_point()
        c, r = int((p.x - gt[0]) / gran), int((gt[3] - p.y) / gran)
        if 0 <= r < rows and 0 <= c < cols and inside[r, c]:
            stations.append((r, c))
    stations, _, _ = G.sanitise_seeds(sorted(set(stations)), state, gran, 2 * gran)
    # a proposed transit hub (a hand-drawn point): a pinned centre anchor under the
    # transit presets, exactly as the plugin treats a point in the hubs layer
    hubs = []
    for geom in layers.get("proposed_station", []):
        p = geom if geom.geom_type == "Point" else geom.representative_point()
        c, r = int((p.x - gt[0]) / gran), int((gt[3] - p.y) / gran)
        if 0 <= r < rows and 0 <= c < cols and inside[r, c]:
            hubs.append((r, c))
    hubs, _, _ = G.sanitise_seeds(sorted(set(hubs)), state, gran, 2 * gran)
    return {"state": state, "origin": origin, "seeds": sorted(set(seeds)), "gt": gt,
            "rows": rows, "cols": cols, "extent": extent, "inside": inside_mask,
            "steep": steep & ~built, "water": water & ~built, "stops": stops,
            "corridor": sorted(set(corridor)), "proposed_hubs": hubs,
            "stations": stations}


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
    walk = float(p.get("centre_walk_m", 800.0))
    green_walk = float(p.get("green_walk_m", walk))
    park_m2 = float(p["min_park_area_ha"]) * 1.0e4 if p.get("min_park_area_ha") else None
    max_walk = max(walk, green_walk)
    # min settlement is a population: convert via the mean density (people / (people/km² × km²/cell))
    min_cells = max(
        1,
        round(
            float(p.get("min_settlement_pop", 2000.0))
            / (sum(s * d for s, d in zip(shares, tiers)) * gran**2 / 1.0e6)
        ),
    )
    state, origin = sub["state"].copy(), sub["origin"].copy()
    G.green_unviable_pockets(
        state, origin, min_cells,
        existing_centres=sub["seeds"], granularity_m=gran, centre_distance_m=walk,
    )
    # the corridor preference: cells outside the transit catchments draw at a scaled
    # probability. Corridor cells (existing stops plus any proposed route the scenario
    # ships) project the stop catchment; a proposed hub projects the wider hub catchment,
    # seeds a centre in growth, and is pinned in post-processing (the plugin's behaviour).
    corridor_w = float(p.get("corridor_weight", DEFAULT_TRANSIT_PREFERENCE))
    # rail and tram stations anchor a pinned centre in every run; a hand-drawn proposed
    # hub joins them only when the scenario is asked to develop along transit
    # A drawn corridor and a proposed hub are inputs of the demonstration that asks for
    # them, not of any run whose preference happens to be above zero.
    use_proposed = bool(p.get("include_proposed", False))
    hubs = list(sub.get("stations", []))
    if use_proposed:
        hubs += [h for h in sub.get("proposed_hubs", []) if h not in set(hubs)]
    corridor_cells = list(sub.get("stops", [])) + (
        list(sub.get("corridor", [])) if use_proposed else []
    )
    catchment = transit_attraction(
        state, gran, corridor_cells, hubs,
        float(p.get("stop_catchment_m", 400.0)), float(p.get("hub_catchment_m", 1200.0)),
    )
    sim_seeds = sub["seeds"] + [h for h in hubs if h not in set(sub["seeds"])]
    sim = isobenefit.Simulation(
        state, origin.copy(), np.zeros_like(state, np.float32), sim_seeds,
        gran, walk, green_walk, float(p["target_population"]), float(p.get("min_green_span_m", 400.0)),
        BUILD_PROB, centre_quota(p), allow_detached(p),
        shares, tiers, int(p.get("max_iterations", 300)), int(p.get("random_seed", 42)),
        min_park_area_m2=park_m2,
        sterile=G.sterile_fabric(origin == 1, sub["seeds"]),
        transit_attraction=catchment, corridor_weight=corridor_w,
        provision_seeds=hubs,
    )
    sim.run()
    st = np.asarray(sim.snapshot()["state"])
    plan, metrics, _pre, _best = G.select_plan(
        [st], gran, max_walk,
        existing_built=(origin == 1), existing_green=(origin == 0),
        existing_centres=sub["seeds"], centre_mode=str(over.get("centre_mode", "placed")),
        centre_anchors=hubs or None,
        centre_distance_m=walk, green_distance_m=green_walk,
        new_density_km2=sum(s * d for s, d in zip(shares, tiers)),
        centre_min_settlement=min_cells,
        min_park_area_m2=park_m2,
    )
    dens = G.derive_density(plan, gran, walk, tiers, shares)
    disp = G.to_tiered_plan(plan, dens, tiers)
    return disp, metrics


def render_png(codes, layers, sub, gran, path, stops=None, hubs=None):
    """Dot-grid PNG with the street underlay: the same visual language as the site's SVGs at a
    fraction of the size (a 150-cell SVG carries ~20k circle elements; the PNG is tens of kB)."""
    from PIL import Image, ImageDraw

    H, W = codes.shape
    P, PAD = 8, 10  # supersampled 2x then reduced, for smooth dots
    cw, ch = W * P + 2 * PAD, H * P + 2 * PAD
    im = Image.new("RGB", (cw, ch), (255, 255, 255))
    draw = ImageDraw.Draw(im)
    gt = sub["gt"]
    # Every vector overlay is clipped to the extents polygon before drawing. The extents
    # are usually a rotated rectangle in the scenario CRS (drawn in another projection),
    # and the dot field is clipped to them, so an unclipped underlay spills past the dots
    # and makes the raster read as rotated against the streets. Clipped, the two share one
    # boundary and the panel reads as a single map.
    clip = sub.get("extent")

    def _clipped_lines(geom):
        g2 = geom if clip is None else clip.intersection(geom)
        if g2.is_empty:
            return
        for part in getattr(g2, "geoms", [g2]):
            if part.geom_type == "LineString":
                yield part

    for geom in layers.get("streets", []):
        for line in _clipped_lines(geom):
            pts = [
                (PAD + (x - gt[0]) / gran * P, PAD + (gt[3] - y) / gran * P)
                for x, y in line.simplify(gran / 4).coords
            ]
            if len(pts) >= 2:
                draw.line(pts, fill=_rgb(STREET), width=2)
    inside = sub["inside"]
    unbuildable = (sub["state"] == -1) & inside
    steep = sub.get("steep")
    steep = steep & unbuildable if steep is not None else np.zeros_like(unbuildable)
    water = sub.get("water")
    water = water & unbuildable if water is not None else np.zeros_like(unbuildable)
    # a barrier carrying a pedestrian crossing: the same grey as the wall it breaks, with a
    # light centre, so it reads as a gap in the wall rather than as a fourth kind of ground
    crossing = (sub["state"] == G.STATE_CROSSING) & inside
    for r in range(H):
        for c in range(W):
            v = int(codes[r, c])
            if v in TIER_STYLE:
                col, radf = TIER_STYLE[v]
            elif water[r, c]:
                col, radf = WATER, GROUND_RAD  # water bodies and river corridors
            elif steep[r, c]:
                col, radf = STEEP, GROUND_RAD  # steep terrain: excluded for slope
            elif crossing[r, c]:
                # a plain white dot on the map: the grey ring muddied it back toward the
                # barrier colour, and the wall on either side is context enough. The legend
                # swatch keeps a hairline ring only so white shows against white paper.
                cx, cy = PAD + c * P + P / 2, PAD + r * P + P / 2
                rad = P * GROUND_RAD
                draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=_rgb(CROSSING))
                continue
            elif unbuildable[r, c]:
                col, radf = UNBUILDABLE, GROUND_RAD  # other exclusions: airfields, military, barriers
            elif inside[r, c]:
                col, radf = GREEN, NATURE_RAD  # untouched land inside the extents
            else:
                continue  # outside the study area: blank
            cx, cy = PAD + c * P + P / 2, PAD + r * P + P / 2
            rad = P * radf
            draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=_rgb(col))
    if stops:
        # the proposed corridor route first (a heavier teal line), then the stops on top,
        # white-ringed so they read over any tier colour
        for geom in layers.get("proposed_corridor", []):
            for line in _clipped_lines(geom):
                pts = [
                    (PAD + (x - gt[0]) / gran * P, PAD + (gt[3] - y) / gran * P)
                    for x, y in line.coords
                ]
                if len(pts) >= 2:
                    draw.line(pts, fill=_rgb(TRANSIT), width=5)
        for r, c in stops:
            cx, cy = PAD + c * P + P / 2, PAD + r * P + P / 2
            rad = P * 0.7
            draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad],
                         fill=_rgb(TRANSIT), outline=(255, 255, 255), width=2)
    if hubs:
        # a proposed transit hub: a heavier ringed marker, so the anchor stands apart
        for r, c in hubs:
            cx, cy = PAD + c * P + P / 2, PAD + r * P + P / 2
            rad = P * 1.3
            draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad],
                         fill=_rgb(TRANSIT), outline=(255, 255, 255), width=4)
    im = im.resize((cw // 2, ch // 2), Image.LANCZOS)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    im.save(path, optimize=True)


# the shared map key: the website legend's grouped layout, drawn as a strip the manuscript
# includes under each map figure (the panels themselves stay legend-free for the gallery)
LEGEND_GROUPS = [
    ("New built", [("high", BUILT_HIGH), ("medium", BUILT_MED), ("low", BUILT_LOW)]),
    ("Mixed-use centre", [("high", CENTRE_HIGH), ("medium", CENTRE_MED), ("low", CENTRE_LOW)]),
    ("Existing and nature", [("existing built", EXIST_BUILT), ("existing centre", EXIST_CENTRE),
                             ("nature", GREEN)]),
    # barriers block walking as well as building, which is what makes land beyond them
    # undevelopable however open it looks; a crossing is a gap in one, not a fourth ground
    ("Barriers", [("water", WATER), ("steep terrain", STEEP), ("unbuildable", UNBUILDABLE),
                  ("crossing (walkable)", CROSSING)]),
]


def render_legend(path):
    """One legend strip for all map figures, supersampled 2x like the panels."""
    from PIL import Image, ImageDraw, ImageFont

    col_w, row_h, top, margin = 880, 78, 96, 40
    rows = max(len(items) for _, items in LEGEND_GROUPS)
    cw, ch = margin * 2 + col_w * len(LEGEND_GROUPS), top + rows * row_h + margin
    im = Image.new("RGB", (cw, ch), (255, 255, 255))
    draw = ImageDraw.Draw(im)
    try:
        title_f = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 46)
        label_f = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 42)
    except OSError:
        title_f = label_f = ImageFont.load_default()
    for gi, (title, items) in enumerate(LEGEND_GROUPS):
        x = margin + gi * col_w
        draw.text((x, margin), title, fill=_rgb(INK), font=title_f)
        for ri, (label, colour) in enumerate(items):
            cy = top + ri * row_h + row_h / 2
            draw.ellipse([x + 4, cy - 22, x + 48, cy + 22], fill=_rgb(colour))
            draw.text((x + 68, cy - 24), label, fill=_rgb(INK), font=label_f)
    im = im.resize((cw // 2, ch // 2), Image.LANCZOS)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    im.save(path, optimize=True)


_SCHEMA_KEYS = (
    "crs", "grid_size_m", "max_iterations", "target_population", "allow_detached", "slope_max_deg",
    "random_seed", "centre_walk_m", "green_walk_m", "optimise_centres", "centre_m2_per_person",
    "min_settlement_pop", "min_green_span_m", "densities_km2", "shares", "ensemble", "ensemble_runs",
)


def merged_formal_params(params: dict, preset: dict, entry_name: str) -> dict:
    """The FULL-resolution parameter set for a preset, in the plugin's params schema, so the file
    downloads straight into the dialog's Load parameters button. Post-processing choices that are
    not dialog parameters (the clustering option) are noted rather than encoded."""
    p = {k: params[k] for k in _SCHEMA_KEYS if k in params}
    for k, v in preset.get("overrides", {}).items():
        if k in ("shares", "densities_km2"):
            p[k] = v
        elif k != "centre_mode":
            p[k] = v
    p["schema"] = "isobenefit-params/1"
    p["name"] = f"{entry_name}_{preset['id']}"
    note = preset["note"]
    if preset.get("overrides", {}).get("centre_mode"):
        note += " In QGIS, every centre option is always written; open the fewest-centres output layer."
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

    for preset in presets_for(name, params, has_stops=bool(sub.get("stops")),
                              has_centres=bool(sub.get("seeds"))):
        disp, metrics = run_preset(sub, p, preset)
        rel = f"{name}/{preset['id']}.png"
        render_png(disp, layers, sub, gran, os.path.join(OUT, rel),
                   stops=sub["stops"] if preset["id"] == "corridor" else None,
                   hubs=sub["proposed_hubs"] if preset["id"] == "corridor" else None)
        keep = {k: round(float(metrics.get(k, 0)), 3) for k in
                ("served_coverage", "served_coverage_incl_existing",
                 "centre_access", "green_access", "population",
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
            "grid": f"{sub['cols']}×{sub['rows']} cells at {gran:.0f} m (preview resolution)",
            "seed": int(params.get("random_seed", 42)),
            "folder": os.path.basename(folder),
            "github": f"https://github.com/UCL/BSP-isobenefit-qgis-plugin/tree/main/scenarios/{os.path.basename(folder)}",
            "zip": f"scenarios/{os.path.basename(folder)}.zip",
            "presets": presets_out}


TITLES = {
    "cambourne": ("Cambourne, UK", "New-settlement growth: the reference demo"),
    "dnipro": ("Dnipro, Ukraine", "Regeneration and edge growth on both banks"),
    "london_crews_hill": ("Crews Hill, London", "Green-belt release at the metropolitan edge"),
    "celina_tx": ("Celina, Texas", "US suburbia at the metropolitan fringe"),
    "kigali_east": ("Kigali, Rwanda", "Plan-guided rapid urbanisation"),
    "medellin_pajarito": ("Medellín, Colombia", "Planned hillside expansion"),
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
            if key != "main":  # a per-extent preset file, when a scenario ships one
                ppath = os.path.join(folder, f"params_{key}.json")
                if os.path.exists(ppath):
                    with open(ppath, encoding="utf-8") as fh:
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
    out_path = os.path.join(OUT, "gallery.json")
    # a partial run (explicit folder arguments) updates its entries in place and keeps the rest
    if sys.argv[1:] and os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            old = json.load(fh).get("entries", [])
        by_id = {e["id"]: e for e in entries}
        entries = [by_id.pop(o["id"], o) for o in old] + list(by_id.values())
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"entries": entries}, fh, indent=1)
    print(f"gallery.json: {len(entries)} entries")


if __name__ == "__main__":
    main()
