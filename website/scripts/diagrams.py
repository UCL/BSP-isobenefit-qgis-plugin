#!/usr/bin/env python3
"""Generate the explanatory dot-grid SVG diagrams for the website, in the established style:
black = built, green = green periphery, red ring = centrality, blue reticle = candidate, with a
right-hand explainer panel (context parameters -> step -> the decision -> legend).

These replace the original Illustrator diagrams with versions matching the CURRENT logic (centre-walk
vs green-walk, "Dispersed development") and add the recommended-plan pipeline. Reproducible:

    python website/scripts/diagrams.py        # writes website/public/images/*.svg
"""
from __future__ import annotations

import math
import os

# palette — one shared system, brand-aligned (site theme red #D32333) and worked across diagrams +
# demonstrators: developed = grey, nature = forest green, centre/accent = brand red, candidate = blue;
# existing fabric gets its own muted shades so "already there" reads apart from "newly recommended".
RED, GREEN, BLUE, BUILT = "#D32333", "#2f7d33", "#1f6fbf", "#3a3a3a"
EXIST_BUILT, EXIST_CENTRE, UNBUILD = "#7d6240", "#962858", "#c4c4c4"
W, H = 1360, 720
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
                if touches and (r + c) % 2 == 0:
                    grid[r][c] = "g"
    return grid


# --- primitives ------------------------------------------------------------------------------
def grid_lines() -> str:
    out = []
    for c in range(1, COLS + 1):
        out.append(
            f'<line x1="{gx(c):.1f}" y1="{gy(1) - 30:.1f}" x2="{gx(c):.1f}" y2="{gy(ROWS) + 30:.1f}" '
            f'stroke="{RED}" stroke-width="0.5" opacity="0.3"/>'
        )
    for r in range(1, ROWS + 1):
        out.append(
            f'<line x1="{gx(1) - 30:.1f}" y1="{gy(r):.1f}" x2="{gx(COLS) + 30:.1f}" y2="{gy(r):.1f}" '
            f'stroke="{RED}" stroke-width="0.5" opacity="0.3"/>'
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
    x, y = PANEL_X, y0
    markers = {
        "green": f'<circle cx="{x + 9}" cy="{{y}}" r="7.5" fill="{GREEN}"/>',
        "built": f'<circle cx="{x + 9}" cy="{{y}}" r="11" fill="{BUILT}"/>',
        "centre": f'<circle cx="{x + 9}" cy="{{y}}" r="8" fill="white" stroke="{RED}" stroke-width="3"/>'
        f'<circle cx="{x + 9}" cy="{{y}}" r="2.3" fill="{RED}"/>',
        "candidate": f'<circle cx="{x + 9}" cy="{{y}}" r="8" fill="none" stroke="{BLUE}" stroke-width="2.6"/>',
    }
    labels = {"green": "green periphery", "built": "built cell", "centre": "centrality", "candidate": "candidate location"}
    out = []
    for kind in entries:
        out.append(markers[kind].replace("{y}", str(y - 5)))
        out.append(f'<text x="{x + 36}" y="{y}" fill="{BUILT}" font-size="16">{labels[kind]}</text>')
        y += 42
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
def d_step2():
    town = base_town()
    scene = grid_lines() + axes() + cells(town) + centre(5, 6) + centre(12, 8)
    scene += dist_line(8, 5, 5, 6, "316 m") + reticle(8, 5) + cand_label(8, 5, "5, 8")
    p, ey = panel(
        ["100 m grid cells", "600 m centre walk", "400 m green walk"],
        "Step 2 — centre access",
        [
            ("Candidate at row 5, col 8.", RED, 700),
            "",
            "Distance to the nearest centre?",
            ("316 m", RED, 700),
            "",
            "Within the 600 m centre walk?",
            ("Yes → continue.", BUILT, 700),
        ],
    )
    scene += p + legend(["green", "built", "centre", "candidate"], ey + 30)
    write("step_2", scene)


def d_plan_centring():
    # one contiguous built area; the centre sits at its INTERIOR, not the centroid (which here is off
    # the built, in the notch) nor an edge.
    grid = [["." for _ in range(COLS + 2)] for _ in range(ROWS + 2)]
    Lc = {(c, r) for c in range(3, 7) for r in range(3, 11)}  # vertical arm
    Lc |= {(c, r) for c in range(3, 13) for r in range(8, 11)}  # horizontal arm -> an L
    for c, r in Lc:
        grid[r][c] = "#"
    scene = grid_lines() + axes() + cells(grid)
    scene += centre(4, 9)  # interior of the (thick) corner of the L
    # mark where the plain centroid would fall (off built, in the notch)
    bx, by = gx(8.5), gy(6.0)
    scene += (
        f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="9" fill="none" stroke="{BLUE}" stroke-width="2.5" '
        f'stroke-dasharray="3 3"/><line x1="{bx - 6:.1f}" y1="{by - 6:.1f}" x2="{bx + 6:.1f}" y2="{by + 6:.1f}" '
        f'stroke="{BLUE}" stroke-width="2.5"/><line x1="{bx + 6:.1f}" y1="{by - 6:.1f}" x2="{bx - 6:.1f}" '
        f'y2="{by + 6:.1f}" stroke="{BLUE}" stroke-width="2.5"/>'
        f'<text x="{bx + 16:.1f}" y="{by - 12:.1f}" fill="{BLUE}" font-weight="700" font-size="13">centroid (off built)</text>'
    )
    p, ey = panel(
        ["one contiguous", "built area", "(an L shape)"],
        "Centre placement",
        [
            "A centre sits at the INTERIOR of",
            "its contiguous built area — the",
            "point furthest from any edge.",
            "",
            ("Not the centroid", BLUE, 700),
            "(which an L / ring / gap puts off",
            "the built, on an edge).",
        ],
    )
    scene += p + legend(["built", "centre"], ey + 30)
    write("plan_centring", scene)


def plan_demonstrator():
    """A worked recommended-plan map in the unified dot-grid language: nature (green) + new built (grey)
    + existing built (brown) + new/existing centres (rings). This is the audit material 'worked into'
    the website style — a clean demonstrator rather than a raw matplotlib panel."""
    Wc, Hc, P, x0, y0 = 30, 20, 30, 50, 78
    cw, ch = 1000, 782
    grid = [["g"] * Wc for _ in range(Hc)]  # default: nature
    new_disks = [(9, 8, 4.2), (15, 6.5, 4.6), (20, 10, 4.2), (13, 13, 4.3), (22, 14.5, 3.6), (7, 13, 3.2), (24, 8.5, 3.0)]
    for r in range(Hc):
        for c in range(Wc):
            if any((c - dx) ** 2 + (r - dy) ** 2 <= rad * rad for dx, dy, rad in new_disks):
                grid[r][c] = "#"  # new built
    for r in range(Hc):
        for c in range(Wc):
            if (c - 14) ** 2 + (r - 9) ** 2 <= 9.0:
                grid[r][c] = "E"  # existing core

    def px(c):
        return x0 + c * P + P / 2

    def py(r):
        return y0 + r * P + P / 2

    def ring(x, y, color, rad=12.5):
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{rad}" fill="white" stroke="{color}" stroke-width="3.5"/><circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}"/>'

    out = [f'<rect width="{cw}" height="{ch}" fill="white"/>']
    out.append(f'<text x="{x0}" y="48" fill="{RED}" font-weight="800" font-size="25">Recommended plan — a worked example</text>')
    for r in range(Hc):
        for c in range(Wc):
            s, x, y = grid[r][c], px(c), py(r)
            if s == "g":
                out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{GREEN}"/>')
            elif s == "#":
                out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="12" fill="{BUILT}"/>')
            elif s == "E":
                out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="12" fill="{EXIST_BUILT}"/>')
    for c, r in [(14, 9)]:  # existing centre
        out.append(ring(px(c), py(r), EXIST_CENTRE))
    for c, r in [(9, 8), (20, 10), (13, 13), (22, 14)]:  # new centres (at built interiors)
        out.append(ring(px(c), py(r), RED))
    # legend strip
    ly = y0 + Hc * P + 40
    lx = x0
    items = [("nature", GREEN, "g"), ("new built", BUILT, "b"), ("existing built", EXIST_BUILT, "b"),
             ("new centre", RED, "r"), ("existing centre", EXIST_CENTRE, "r")]
    for label, color, kind in items:
        if kind == "r":
            out.append(ring(lx + 10, ly - 5, color, 9))
        else:
            out.append(f'<circle cx="{lx + 10}" cy="{ly - 5}" r="{6 if kind == "g" else 11}" fill="{color}"/>')
        out.append(f'<text x="{lx + 28}" y="{ly}" fill="{BUILT}" font-size="16">{label}</text>')
        lx += 50 + len(label) * 9
    doc = (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {cw} {ch}" font-family="Arial, sans-serif">'
        f'{"".join(out)}</svg>'
    )
    with open(os.path.join(OUT, "plan_demonstrator.svg"), "w", encoding="utf-8") as fh:
        fh.write(doc)
    print(f"wrote {os.path.join(OUT, 'plan_demonstrator.svg')}")


PARAMS = ["100 m grid cells", "600 m centre walk", "400 m green walk"]
CENTRES = ((5, 6), (12, 8))


def build_step(name, title, body, candidate=None, cand_color=BLUE, distline=None, town=None, extra=""):
    g = town if town is not None else base_town()
    scene = grid_lines() + axes() + cells(g)
    for cc, rr in CENTRES:
        if g[rr][cc] in ("#", "g", "."):
            scene += centre(cc, rr)
    if distline:
        scene += dist_line(*distline)
    if candidate:
        scene += reticle(candidate[0], candidate[1], cand_color)
        scene += cand_label(candidate[0], candidate[1], f"{candidate[1]}, {candidate[0]}")
    scene += extra
    p, ey = panel(PARAMS, title, body)
    scene += p + legend(["green", "built", "centre", "candidate"], ey + 30)
    write(name, scene)


def d_step0():
    build_step(
        "step_0", "Step 0 — population",
        ["Is the population still below", "the target?", "", ("Yes → visit every cell.", BUILT, 700),
         "If the target is met, stop."],
    )


def d_step1a():
    build_step(
        "step_1a", "Step 1 — periphery",
        [("Candidate at row 7, col 8.", RED, 700), "", "It is already a built cell.", ("Bail.", BUILT, 700)],
        candidate=(8, 7),
    )


def d_step1b():
    build_step(
        "step_1b", "Step 1 — periphery",
        [("Candidate at row 4, col 15.", RED, 700), "", "It is empty but not next to any", "built land — not on the periphery.",
         ("Bail.", BUILT, 700)],
        candidate=(15, 4),
    )


def d_step1c():
    build_step(
        "step_1c", "Step 1 — periphery",
        [("Candidate at row 2, col 8.", RED, 700), "", "It is empty and touches built", "land — on the periphery.",
         ("Continue.", BUILT, 700)],
        candidate=(8, 2),
    )


def d_step3():
    g = [["." for _ in range(COLS + 2)] for _ in range(ROWS + 2)]
    for r in range(3, 11):
        for c in range(2, 7):
            g[r][c] = "#"
        for c in range(11, 16):
            g[r][c] = "#"
    for r in range(3, 11):  # a narrow green corridor between the two blocks
        for c in range(8, 10):
            g[r][c] = "g"
    span = (
        f'<line x1="{gx(8) - 22:.1f}" y1="{gy(3):.1f}" x2="{gx(8) - 22:.1f}" y2="{gy(10):.1f}" '
        f'stroke="{GREEN}" stroke-width="2"/>'
        f'<text x="{gx(8) - 30:.1f}" y="{gy(6.5):.1f}" fill="{GREEN}" font-weight="700" font-size="13" '
        f'text-anchor="end">green span</text>'
    )
    build_step(
        "step_3", "Step 3 — green span",
        ["Would building here shrink a", "green corridor below the", "400 m minimum span?",
         ("Yes → bail (keep the green).", BUILT, 700)],
        candidate=(9, 6), town=g, extra=span,
    )


def d_step4():
    g = base_town()
    g[2][8] = "#"  # the candidate cell, now built
    build_step(
        "step_4", "Step 4 — build",
        ["All checks pass →", ("the cell becomes built.", BUILT, 700), "",
         "Its density is set by distance", "to the nearest centre (denser", "closer in)."],
        candidate=(8, 2), cand_color=BUILT, town=g,
    )


def d_centres_neighbouring():
    build_step(
        "centres_neighbouring", "Seeding — neighbouring",
        [("Candidate at row 11, col 15.", RED, 700), "", "Beyond the 600 m centre walk", "of any centre.",
         "", "A new neighbouring centre may", "seed (per Dispersed development)."],
        candidate=(15, 11), distline=(15, 11, 12, 8, "632 m"),
    )


def d_centres_isolated():
    build_step(
        "centres_isolated", "Seeding — isolated",
        ["Dispersed development", "(Moderate / Aggressive) lets a", "new settlement leapfrog away", "from the built area —",
         ("a satellite, with its centre.", BUILT, 700)],
        candidate=(15, 3),
    )


def d_plan_ensemble():
    g = base_town()
    cy0, cx0 = 6.5, 8.5
    shade = {}
    for r in range(1, ROWS + 1):
        for c in range(1, COLS + 1):
            if g[r][c] == "#":
                d = math.hypot(c - cx0, r - cy0)
                shade[(c, r)] = max(0.25, min(1.0, 1.15 - d / 7.0))
    scene = grid_lines() + axes() + cells(g, shade=shade)
    p, ey = panel(
        ["many runs", "of the same", "simulation"],
        "Ensemble → likelihood",
        ["Run the simulation many times.", "Each cell's shade = the share of", "runs where it ended built.",
         "", ("Dark = robust,", BUILT, 700), "faint = contingent."],
    )
    scene += p + legend(["green", "built"], ey + 30)
    write("plan_ensemble", scene)


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
    scene += p + legend(["green", "built", "centre"], ey + 30)
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
    scene += p + legend(["green", "built", "centre"], ey + 30)
    write("plan_cleanup", scene)


def main():
    os.makedirs(OUT, exist_ok=True)
    for fn in (d_step0, d_step1a, d_step1b, d_step1c, d_step2, d_step3, d_step4,
               d_centres_neighbouring, d_centres_isolated,
               d_plan_ensemble, d_plan_select, d_plan_cleanup, d_plan_centring,
               plan_demonstrator):
        fn()
    print("done")


if __name__ == "__main__":
    main()
