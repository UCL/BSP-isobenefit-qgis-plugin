#!/usr/bin/env python3
"""Generate the explanatory dot-grid SVG diagrams for the website, in the established style:
warm orange-brown = built, green = green periphery, red ring = centrality, blue reticle = candidate,
with a right-hand explainer panel (context parameters -> step -> the decision -> legend).

These replace the original Illustrator diagrams with versions matching the CURRENT logic (centre-walk
vs green-walk, "Dispersed development") and add the recommended-plan pipeline. Reproducible:

    python website/scripts/diagrams.py        # writes website/public/images/*.svg
"""
from __future__ import annotations

import math
import os

# palette — one shared system, brand-aligned (site theme red #D32333) and worked across diagrams +
# demonstrators: newly developed = warm orange-brown (the mid of the plan's built ramp), nature = forest
# green, centre/accent = brand red, candidate = blue; existing fabric gets its own muted shades so
# "already there" reads apart from "newly recommended".
RED, GREEN, BLUE, BUILT = "#D32333", "#2f7d33", "#1f6fbf", "#cc7a29"
EXIST_BUILT, EXIST_CENTRE = "#96867a", "#962858"  # existing built matches grid.py's cool grey-taupe
W, H = 1200, 680  # cropped tight to the content (grid + panel end ~1110 / ~654) to cut dead margin
X0, Y0, STEP = 92, 46, 50  # grid origin (col 1 / row 1 centre) + cell pitch
COLS, ROWS = 16, 12
PANEL_X = 880  # right of the 16-col grid (which ends at x≈842) so panel text never overlaps the grid
OUT = os.environ.get(
    "DIAGRAMS_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "public", "images")
)


def gx(c: float) -> float:
    return X0 + (c - 1) * STEP


def gy(r: float) -> float:
    return Y0 + (r - 1) * STEP


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# --- base town: an organic blob (union of disks) + scattered green fringe ---------------------
def base_town() -> list[list[str]]:
    disks = [(4.5, 5, 3.0), (8.5, 6, 3.9), (11.5, 7.5, 3.2), (6.5, 9.5, 3.0), (10.5, 10, 2.7)]
    grid = [["." for _ in range(COLS + 2)] for _ in range(ROWS + 2)]  # 1-indexed, padded
    for r in range(1, ROWS + 1):
        for c in range(1, COLS + 1):
            if any((c - dx) ** 2 + (r - dy) ** 2 <= rad * rad for dx, dy, rad in disks):
                grid[r][c] = "#"
    for r in range(1, ROWS + 1):  # green = empty cells touching built, scattered to look like dots
        for c in range(1, COLS + 1):
            if grid[r][c] == ".":
                touches = any(
                    grid[r + dr][c + dc] == "#"
                    for dr in (-1, 0, 1)
                    for dc in (-1, 0, 1)
                    if (dr, dc) != (0, 0)
                )
                if touches:
                    grid[r][c] = "g"
    return grid


# --- primitives ------------------------------------------------------------------------------
def grid_lines() -> str:
    # non-scaling-stroke pins every line to a full device pixel regardless of how the SVG is scaled,
    # so no vertical/horizontal grid line drops out to sub-pixel rounding (e.g. column 11, row 10).
    style = 'stroke="{}" stroke-width="1" opacity="0.35" vector-effect="non-scaling-stroke"'.format(RED)
    out = []
    for c in range(1, COLS + 1):
        out.append(
            f'<line x1="{gx(c):.1f}" y1="{gy(1) - 30:.1f}" x2="{gx(c):.1f}" y2="{gy(ROWS) + 30:.1f}" {style}/>'
        )
    for r in range(1, ROWS + 1):
        out.append(
            f'<line x1="{gx(1) - 30:.1f}" y1="{gy(r):.1f}" x2="{gx(COLS) + 30:.1f}" y2="{gy(r):.1f}" {style}/>'
        )
    return "".join(out)


def axes() -> str:
    out = []
    for c in range(1, COLS + 1):
        out.append(
            f'<text x="{gx(c):.1f}" y="{gy(ROWS) + 58:.1f}" fill="{RED}" font-weight="800" '
            f'font-size="18" text-anchor="middle">{c}</text>'
        )
    for r in range(1, ROWS + 1):
        out.append(
            f'<text x="{gx(1) - 52:.1f}" y="{gy(r) + 6:.1f}" fill="{RED}" font-weight="800" '
            f'font-size="18" text-anchor="middle">{r}</text>'
        )
    return "".join(out)


def cells(grid: list[list[str]], shade: dict | None = None) -> str:
    out = []
    for r in range(1, ROWS + 1):
        for c in range(1, COLS + 1):
            s = grid[r][c]
            if shade and (c, r) in shade:  # probability shading (likelihood diagram)
                op = shade[(c, r)]
                out.append(f'<circle cx="{gx(c):.1f}" cy="{gy(r):.1f}" r="12.8" fill="{BUILT}" opacity="{op:.2f}"/>')
            elif s == "#":
                out.append(f'<circle cx="{gx(c):.1f}" cy="{gy(r):.1f}" r="12.8" fill="{BUILT}"/>')
            elif s == "g":
                out.append(f'<circle cx="{gx(c):.1f}" cy="{gy(r):.1f}" r="7.5" fill="{GREEN}"/>')
    return "".join(out)


def centre(c: float, r: float) -> str:
    x, y = gx(c), gy(r)
    return (
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="white" stroke="{RED}" stroke-width="3.5"/>'
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6" fill="{RED}"/>'
    )


def reticle(c: float, r: float, color: str = BLUE) -> str:
    x, y = gx(c), gy(r)
    out = [f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9.5" fill="none" stroke="{color}" stroke-width="3"/>']
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        out.append(
            f'<line x1="{x + sx * 7:.1f}" y1="{y + sy * 7:.1f}" x2="{x + sx * 16:.1f}" y2="{y + sy * 16:.1f}" '
            f'stroke="{color}" stroke-width="3"/>'
        )
    return "".join(out)


def cand_label(c: float, r: float, label: str) -> str:
    return f'<text x="{gx(c) + 19:.1f}" y="{gy(r) - 13:.1f}" fill="{RED}" font-weight="700" font-size="13">{esc(label)}</text>'


def dist_line(c1, r1, c2, r2, label) -> str:
    x1, y1, x2, y2 = gx(c1), gy(r1), gx(c2), gy(r2)
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
    if ang > 90 or ang < -90:
        ang += 180
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{RED}" '
        f'stroke-width="1.5" stroke-dasharray="4 4"/>'
        f'<text x="{mx:.1f}" y="{my - 7:.1f}" fill="{RED}" font-weight="700" font-size="13" '
        f'text-anchor="middle" transform="rotate({ang:.1f} {mx:.1f} {my:.1f})">{esc(label)}</text>'
    )


def panel(params: list[str], title: str, body: list) -> str:
    x = PANEL_X
    out = [f'<text x="{x}" y="150" fill="{RED}" font-weight="700" font-size="14">context parameters (hypothetical):</text>']
    y = 150
    for p in params:
        y += 21
        out.append(f'<text x="{x}" y="{y}" fill="{RED}" font-weight="700" font-size="14">{esc(p)}</text>')
    out.append(f'<text x="{x}" y="280" fill="{BUILT}" font-weight="700" font-size="22">{esc(title)}</text>')
    y = 296
    for line in body:
        y += 26
        txt, color, weight = line if isinstance(line, tuple) else (line, BUILT, 400)
        if txt:
            out.append(f'<text x="{x}" y="{y}" fill="{color}" font-weight="{weight}" font-size="15">{esc(txt)}</text>')
    return "".join(out), y


def legend(entries: list[str], y0: float) -> str:
    # A quiet key, subordinate to the action comment above it: smaller, grey text and tighter rows.
    # Callers leave a clear gap above y0 so it reads as a separate, secondary thing.
    x, y = PANEL_X, y0
    grey = "#6b6b6b"
    markers = {
        "green": f'<circle cx="{x + 7}" cy="{{y}}" r="6" fill="{GREEN}"/>',
        "built": f'<circle cx="{x + 7}" cy="{{y}}" r="8" fill="{BUILT}"/>',
        "centre": f'<circle cx="{x + 7}" cy="{{y}}" r="6.5" fill="white" stroke="{RED}" stroke-width="2.6"/>'
        f'<circle cx="{x + 7}" cy="{{y}}" r="2" fill="{RED}"/>',
        "candidate": f'<circle cx="{x + 7}" cy="{{y}}" r="6.5" fill="none" stroke="{BLUE}" stroke-width="2.3"/>',
    }
    labels = {"green": "green periphery", "built": "built cell", "centre": "centrality", "candidate": "candidate location"}
    out = []
    for kind in entries:
        out.append(markers[kind].replace("{y}", str(y - 4)))
        out.append(f'<text x="{x + 26}" y="{y}" fill="{grey}" font-size="13">{labels[kind]}</text>')
        y += 30
    return "".join(out)


def write(name: str, body: str) -> None:
    doc = (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'font-family="Arial, sans-serif">{body}</svg>'
    )
    path = os.path.join(OUT, f"{name}.svg")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(doc)
    print(f"wrote {path}")


# --- diagrams --------------------------------------------------------------------------------
def d_step2():  # SAME candidate: its distance to the nearest centre is within the centre walk
    town = worked_town()
    (wc, wr), (ec, er) = WORKED_CENTRES
    cc, cr = WORKED_CANDIDATE
    scene = grid_lines() + axes() + cells(town) + centre(wc, wr) + centre(ec, er)
    scene += dist_line(cc, cr, wc, wr, "424 m") + reticle(cc, cr) + cand_label(cc, cr, f"r{cr} c{cc}")
    p, ey = panel(
        PARAMS, "Step 2: centre walk",
        [("The same candidate.", RED, 700), "", "Distance to the nearest centre?", ("424 m", RED, 700), "",
         "Within the 600 m centre walk?", ("Yes → next check.", BUILT, 700)],
    )
    scene += p + legend(["green", "built", "centre", "candidate"], ey + 56)
    write("step_2", scene)


def d_plan_centring():
    # A grown centre often lands on the EDGE or a corner of its development, where it serves fewer
    # homes. Post-processing moves it to a CENTRAL location in the same built area.
    grid = [["." for _ in range(COLS + 2)] for _ in range(ROWS + 2)]
    for c in range(4, 14):
        for r in range(3, 11):
            grid[r][c] = "#"
    scene = grid_lines() + axes() + cells(grid)
    # the as-grown centre, sitting in a corner (blue, marked as poorly placed)
    ex, ey_ = gx(4), gy(3)
    scene += (
        f'<circle cx="{ex:.1f}" cy="{ey_:.1f}" r="9" fill="white" stroke="{BLUE}" stroke-width="3.5"/>'
        f'<circle cx="{ex:.1f}" cy="{ey_:.1f}" r="2.6" fill="{BLUE}"/>'
        f'<text x="{ex - 6:.1f}" y="{ey_ - 14:.1f}" fill="{BLUE}" font-weight="700" font-size="13" '
        f'text-anchor="middle">as grown: on the edge</text>'
    )
    # the re-placed centre, central to the development (red), with a dashed move arrow
    scene += centre(8, 6)
    scene += (
        f'<line x1="{ex + 13:.1f}" y1="{ey_ + 10:.1f}" x2="{gx(8) - 14:.1f}" y2="{gy(6) - 12:.1f}" '
        f'stroke="{BLUE}" stroke-width="2" stroke-dasharray="5 4"/>'
        f'<text x="{gx(8) + 16:.1f}" y="{gy(6) - 14:.1f}" fill="{RED}" font-weight="700" font-size="13">re-placed: central</text>'
    )
    p, ey = panel(
        ["one contiguous", "built area"],
        "Centre placement",
        [
            "A grown centre often lands on",
            "the edge or in a corner, serving",
            "its population less well than a",
            "central location would.",
            "",
            ("Post-processing moves it to a", BUILT, 700),
            ("central spot in the same", BUILT, 700),
            ("development.", BUILT, 700),
        ],
    )
    scene += p + legend(["built", "centre"], ey + 56)
    write("plan_centring", scene)


PARAMS = ["100 m grid cells", "600 m centre walk", "400 m green span"]
CENTRES = ((5, 6), (12, 8))

# ONE consistent worked example threads Steps 1c -> 2 -> 3 -> 4: the SAME town, the SAME two centres,
# and the SAME candidate cell. Two built lobes are separated by a 500 m green corridor (5 cells); the
# candidate sits on the periphery at the north edge of that corridor. It survives every check and builds.
WORKED_CENTRES = ((4, 6), (14, 6))
WORKED_CANDIDATE = (7, 3)  # west edge of the corridor, touches the west lobe -> on the periphery


def worked_town() -> list[list[str]]:
    g = [["." for _ in range(COLS + 2)] for _ in range(ROWS + 2)]
    for r in range(3, 12):  # two solid lobes, rows 3-11
        for c in range(2, 7):  # west lobe, cols 2-6
            g[r][c] = "#"
        for c in range(12, 17):  # east lobe, cols 12-16
            g[r][c] = "#"
    for r in range(1, ROWS + 1):  # green periphery: empty cells touching built
        for c in range(1, COLS + 1):
            if g[r][c] == "." and any(
                g[r + dr][c + dc] == "#" for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)
            ):
                g[r][c] = "g"
    return g


def build_step(name, title, body, candidate=None, cand_color=BLUE, distline=None, town=None, extra="",
               centres=CENTRES):
    g = town if town is not None else base_town()
    scene = grid_lines() + axes() + cells(g)
    for cc, rr in centres:
        if g[rr][cc] in ("#", "g", "."):
            scene += centre(cc, rr)
    if distline:
        scene += dist_line(*distline)
    if candidate:
        scene += reticle(candidate[0], candidate[1], cand_color)
        scene += cand_label(candidate[0], candidate[1], f"r{candidate[1]} c{candidate[0]}")
    scene += extra
    p, ey = panel(PARAMS, title, body)
    # a clear gap below the action comment sets the (secondary) key apart from the main message
    scene += p + legend(["green", "built", "centre", "candidate"], ey + 56)
    write(name, scene)


def d_step0():
    build_step(
        "step_0", "Step 0: population",
        ["Checked before each", "iteration, not cell by cell.", "",
         ("Below target →", BUILT, 700), ("run the whole iteration.", BUILT, 700), "",
         ("Target met → stop.", BUILT, 700)],
        town=worked_town(), centres=WORKED_CENTRES,  # SAME town as steps 1-4 (was the old base town)
    )


def d_step1a():  # a cell that is already built -> skip (same worked town)
    build_step(
        "step_1a", "Step 1: a candidate?",
        [("Look at row 6, col 5.", RED, 700), "", "It is already a built cell.", ("Skip it.", BUILT, 700)],
        candidate=(5, 6), town=worked_town(), centres=WORKED_CENTRES,
    )


def d_step1b():  # an empty cell in the open green, touching no built land -> not on the periphery
    build_step(
        "step_1b", "Step 1: a candidate?",
        [("Look at row 7, col 9.", RED, 700), "", "Empty, but it touches no built", "land — not on the periphery.",
         ("Skip it.", BUILT, 700)],
        candidate=(9, 7), town=worked_town(), centres=WORKED_CENTRES,
    )


def d_step1c():  # THE worked candidate: empty and on the periphery -> continues into Steps 2, 3, 4
    build_step(
        "step_1c", "Step 1: a candidate?",
        [(f"Look at row {WORKED_CANDIDATE[1]}, col {WORKED_CANDIDATE[0]}.", RED, 700), "",
         "Empty, and it touches built", "land — on the periphery.", ("A candidate → next check.", BUILT, 700)],
        candidate=WORKED_CANDIDATE, town=worked_town(), centres=WORKED_CENTRES,
    )


def d_step2b():  # SAME candidate: the stochastic build draw, between the centre walk and the green checks
    build_step(
        "step_2b", "Step 2b: build chance",
        [("The same candidate.", RED, 700), "",
         "Eligible cells do not all build", "at once: each is drawn against",
         "the build probability first.", "",
         ("Drawn (p = 0.25) → last check.", BUILT, 700),
         "Not drawn → wait for a", "later iteration."],
        candidate=WORKED_CANDIDATE, town=worked_town(), centres=WORKED_CENTRES,
    )


def d_step3():  # SAME candidate: its green corridor stays >= the minimum span, so it passes
    cy = (gy(4) + gy(5)) / 2  # sit the dimension line in the GAP between two dot rows, not over the dots
    x1, x2 = gx(7) - 25, gx(11) + 25  # the green corridor spans cols 7..11 = 5 cells = 500 m
    span = (
        f'<line x1="{x1:.1f}" y1="{cy:.1f}" x2="{x2:.1f}" y2="{cy:.1f}" stroke="{GREEN}" stroke-width="2.5"/>'
        f'<line x1="{x1:.1f}" y1="{cy - 9:.1f}" x2="{x1:.1f}" y2="{cy + 9:.1f}" stroke="{GREEN}" stroke-width="2.5"/>'
        f'<line x1="{x2:.1f}" y1="{cy - 9:.1f}" x2="{x2:.1f}" y2="{cy + 9:.1f}" stroke="{GREEN}" stroke-width="2.5"/>'
        f'<text x="{(x1 + x2) / 2:.1f}" y="{cy - 12:.1f}" fill="{GREEN}" font-weight="700" font-size="13" '
        f'text-anchor="middle">green span 500 m</text>'
    )
    build_step(
        "step_3", "Step 3: green span",
        [("The same candidate, on the", RED, 700), ("green corridor.", RED, 700), "",
         "Building it narrows the corridor", "from 500 m to 400 m.", "",
         ("Still ≥ the 400 m minimum → build.", BUILT, 700)],
        candidate=WORKED_CANDIDATE, town=worked_town(), centres=WORKED_CENTRES, extra=span,
    )


def d_step4():  # SAME candidate builds
    g = worked_town()
    g[WORKED_CANDIDATE[1]][WORKED_CANDIDATE[0]] = "#"  # the candidate cell, now built
    build_step(
        "step_4", "Step 4: build",
        ["Every check passed →", ("the cell becomes built.", BUILT, 700), "",
         "Its density is one of the three", "tiers, arranged later by distance", "to the centres."],
        candidate=WORKED_CANDIDATE, cand_color=BUILT, town=g, centres=WORKED_CENTRES,
    )


def seeding_town():
    # ONE town serves both seeding diagrams: a west blob holding the centre, and an eastern lobe
    # grown to the edge of the centre walk. Its green frontier cell sits at 700 m, just past the
    # 600 m walk — growth can only ever overshoot the walk by a cell, so a frontier far beyond
    # it (say 900 m) could never arise.
    g = [["." for _ in range(COLS + 2)] for _ in range(ROWS + 2)]
    for c in range(2, 8):  # the west blob, holding the centre
        for r in range(4, 10):
            g[r][c] = "#"
    for c in range(8, 12):  # the eastern lobe, grown out to the walk's edge
        for r in range(6, 9):
            g[r][c] = "#"
    for r in range(1, ROWS + 1):
        for c in range(1, COLS + 1):
            if g[r][c] == "." and any(
                g[r + dr][c + dc] == "#" for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)
            ):
                g[r][c] = "g"
    return g


def d_centres_neighbouring():
    build_step(
        "centres_neighbouring", "Seeding: neighbouring",
        [("The eastern lobe has grown to", RED, 700), ("the edge of the centre walk.", RED, 700), "",
         "Its green frontier is 700 m out,", "just past the 600 m walk, so it", "cannot build.", "",
         ("Such a cell may seed a centre:", BUILT, 700), ("a 1% draw each iteration.", BUILT, 700)],
        candidate=(12, 7), distline=(12, 7, 5, 7, "700 m"), town=seeding_town(), centres=((5, 7),),
    )


def d_centres_isolated():
    # the SAME town as the neighbouring diagram; only the seeding location differs
    build_step(
        "centres_isolated", "Seeding: isolated",
        ["The same town.", "", "Dispersed development", "(Moderate / Aggressive) gives", "every far green cell a small",
         "per-iteration chance to seed",
         ("a satellite, with its centre.", BUILT, 700)],
        candidate=(15, 3), town=seeding_town(), centres=((5, 7),),
    )


def d_plan_select():
    g = base_town()
    scene = grid_lines() + axes() + cells(g)
    for cc, rr in CENTRES:
        scene += centre(cc, rr)
    p, ey = panel(
        ["one chosen", "run"],
        "Pick the best run",
        ["From the ensemble, take the", "single run with the shortest", "average walk to amenities.", "",
         ("A coherent, buildable layout —", BUILT, 700), "not a blurry average."],
    )
    scene += p + legend(["green", "built", "centre"], ey + 56)
    write("plan_select", scene)


def d_plan_cleanup():
    g = base_town()
    for r in (2, 3):  # a stranded speck, top-right
        for c in (14, 15):
            g[r][c] = "#"
    g[11][3] = "#"  # a lone speck, bottom-left
    ring = ""
    for (cc, rr, rad) in ((14.5, 2.5, 38), (3, 11, 24)):
        ring += (
            f'<circle cx="{gx(cc):.1f}" cy="{gy(rr):.1f}" r="{rad}" fill="none" stroke="{RED}" '
            f'stroke-width="2" stroke-dasharray="4 4"/>'
        )
    build_step_town = g
    scene = grid_lines() + axes() + cells(build_step_town) + ring
    for cc, rr in CENTRES:
        scene += centre(cc, rr)
    p, ey = panel(
        ["minimum", "settlement", "size"],
        "Cleanup",
        ["Tiny stranded settlements", "(below the minimum size)", ("revert to green.", GREEN, 700), "",
         "The raw plan is kept too, so", "the change stays visible."],
    )
    scene += p + legend(["green", "built", "centre"], ey + 56)
    write("plan_cleanup", scene)


# ---- input-data catalogue: the REAL downloaded layers for the demonstration town -------------
# Rendered from the GeoJSON snapshots in scripts/data (fetched by fetch_data.py with the plugin's
# own OSM queries). Every panel shares one transform over the same 4.2 km window, so the layers
# provably compose — and the SAME window is the growth demonstrators' substrate.
STREET = "#9a9a9a"
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
# one colour language across the whole site: EXISTING fabric/centres take the same brown/magenta as
# the demonstrators' existing dots (the warm built colour + brand red are what the SIMULATION adds)
BUILT_AREA, GREEN_AREA, WATER_AREA, CENTRE_AREA, IND_AREA = EXIST_BUILT, "#3f8f47", "#6f9fcf", EXIST_CENTRE, "#7b6d8f"


def _features(name):
    import json

    with open(os.path.join(DATA, f"{name}.geojson"), encoding="utf-8") as fh:
        return json.load(fh)["features"]


def input_layers():
    ring = _features("extents")[0]["geometry"]["coordinates"][0]
    xs, ys = zip(*ring)
    xmin, ymin, xmax, ymax = min(xs), min(ys), max(xs), max(ys)
    side, ox, oy = 520.0, 20.0, 56.0
    S = side / (xmax - xmin)
    cw, ch = 560, 600

    def X(x):
        return ox + (x - xmin) * S

    def Y(y):
        return oy + (ymax - y) * S

    def poly_path(geom):
        rings = geom["coordinates"] if geom["type"] == "Polygon" else [r for p in geom["coordinates"] for r in p]
        return " ".join(
            "M " + " L ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in ring) + " Z" for ring in rings
        )

    def polys(name, fill, stroke):
        return "".join(
            f'<path d="{poly_path(f["geometry"])}" fill="{fill}" fill-rule="evenodd" '
            f'stroke="{stroke}" stroke-width="1"/>'
            for f in _features(name) if f["geometry"]["type"] in ("Polygon", "MultiPolygon")
        )

    def street_lines(minor=True):
        # stroke by class so arterials read over the residential net
        widths = {"primary": 3.0, "primary_link": 3.0, "secondary": 2.4, "tertiary": 1.9}
        light = {"footway", "path", "cycleway", "steps", "track", "bridleway"}
        out = []
        for f in _features("streets"):
            hw = f["properties"].get("highway", "")
            if hw in light:
                if not minor:
                    continue
                w, op = 0.6, 0.5
            else:
                w, op = widths.get(hw, 1.1), 1.0
            pts = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in f["geometry"]["coordinates"])
            out.append(f'<polyline points="{pts}" fill="none" stroke="{STREET}" stroke-width="{w}" opacity="{op}"/>')
        return "".join(out)

    def stop_markers():
        s = 9.0
        return "".join(
            f'<rect x="{X(f["geometry"]["coordinates"][0]) - s / 2:.1f}" '
            f'y="{Y(f["geometry"]["coordinates"][1]) - s / 2:.1f}" width="{s}" height="{s}" rx="2" fill="{BLUE}"/>'
            for f in _features("stops")
        )

    def extents_rect():
        return (
            f'<rect x="{X(xmin):.1f}" y="{Y(ymax):.1f}" width="{side:.1f}" height="{(ymax - ymin) * S:.1f}" '
            f'rx="10" fill="none" stroke="{RED}" stroke-width="3" stroke-dasharray="9 6"/>'
        )

    tile = (
        f'<rect x="{ox:.1f}" y="{oy:.1f}" width="{side:.1f}" height="{(ymax - ymin) * S:.1f}" '
        f'rx="10" fill="#eef1ee"/>'
    )

    def panel(name, title, body):
        doc = (
            f'<?xml version="1.0" encoding="UTF-8"?>'
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {cw} {ch}" font-family="Arial, sans-serif">'
            f'<rect width="{cw}" height="{ch}" fill="white"/>'
            f'<text x="20" y="38" fill="{RED}" font-weight="800" font-size="22">{esc(title)}</text>'
            f"{tile}{body}</svg>"
        )
        with open(os.path.join(OUT, f"{name}.svg"), "w", encoding="utf-8") as fh:
            fh.write(doc)
        print(f"wrote {os.path.join(OUT, name)}.svg")

    panel("input_extents", "Extents (the area of interest)", extents_rect())
    panel("input_built", "Existing built fabric", polys("built", BUILT_AREA, "#5e4a30"))
    panel("input_green", "Green space", polys("green", GREEN_AREA, "#2f7d33"))
    panel("input_centres", "Urban centres", polys("centres", CENTRE_AREA, "#6e1d40"))
    panel("input_unbuildable", "Unbuildable (water, industrial, barriers)",
          polys("unbuildable", WATER_AREA, "#5a86b5"))
    panel("input_industrial", "Industrial land", polys("industrial", IND_AREA, "#645a78"))
    panel("input_streets", "Street network", street_lines())
    panel("input_pt", "Public-transport stops", stop_markers())
    # everything stacked: the proof the layers are slices of ONE geography
    panel(
        "input_composite", "All layers together",
        polys("green", GREEN_AREA, "#2f7d33")
        + polys("unbuildable", WATER_AREA, "#5a86b5")
        + polys("industrial", IND_AREA, "#645a78")
        + polys("built", BUILT_AREA, "#5e4a30")
        + polys("centres", CENTRE_AREA, "#6e1d40")
        + street_lines(minor=False)
        + stop_markers()
        + extents_rect(),
    )


def main():
    os.makedirs(OUT, exist_ok=True)
    input_layers()
    for fn in (d_step0, d_step1a, d_step1b, d_step1c, d_step2, d_step2b, d_step3, d_step4,
               d_centres_neighbouring, d_centres_isolated,
               d_plan_select, d_plan_cleanup, d_plan_centring):
        fn()
    print("done")


if __name__ == "__main__":
    main()
