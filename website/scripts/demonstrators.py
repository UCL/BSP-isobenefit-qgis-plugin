#!/usr/bin/env python3
"""High-resolution recommended-plan DEMONSTRATORS for the website: run the real pipeline (CA growth ->
select best run -> post-process) and render the resulting plan as a fine dot-grid in the unified
website palette. Because it's generated, we can use a far higher resolution than the original crude
hand-drawn diagrams. Needs the engine wheel + numpy:

    uv run --no-project --with core/dist/isobenefit-*.whl --with numpy python website/scripts/demonstrators.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

import isobenefit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from isobenefit_qgis import grid as G  # noqa: E402  (repo root added to path above)

GREEN, BUILT, EXIST_BUILT, RED, EXIST_CENTRE = "#2f7d33", "#3a3a3a", "#7d6240", "#D32333", "#962858"
OUT = os.environ.get(
    "DEMO_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "public", "images")
)


def grow(n=84, pop=10000.0, isol=0.0001, seed=3, existing=True, bp=0.3, nb=0.01, iters=400):
    """Grow one CA run. ``nb`` (neighbouring-centre rate) must be > 0 or a single town can't grow past one
    centre's reach. ``isol`` is the Dispersed-development (leapfrog) rate; ``bp`` the per-step build
    probability; ``iters`` the iteration cap (use a low cap + huge pop to show growth RATE)."""
    state = np.zeros((n, n), dtype=np.int16)
    origin = np.full((n, n), -1, dtype=np.int16)
    lo, hi = n // 2 - 5, n // 2 + 5
    state[lo:hi, lo:hi] = 1
    density = np.zeros((n, n), dtype=np.float32)
    if existing:
        origin[lo:hi, lo:hi] = 1
        density[lo:hi, lo:hi] = 2000.0
    seeds = [(n // 2, n // 2)]
    sim = isobenefit.Simulation(
        state, origin, density, seeds, 50.0, 400.0, pop, 400.0, bp, nb, isol, 0.8,
        (0.4, 0.4, 0.2), (6000.0, 3000.0, 1000.0), 2000.0, iters, seed,
    )
    return np.asarray(isobenefit.run_ensemble(sim, seed, 1)[0]), origin, seeds


def to_plan(st, origin, seeds, spacing, existing=True):
    plan, _, _, _ = G.select_plan(
        [st], 50.0, 400.0, 400.0,
        existing_built=(origin == 1) if existing else None,
        existing_centres=seeds if existing else None,
        centre_spacing_m=spacing, centre_distance_m=400.0, green_distance_m=400.0, centre_min_settlement=8,
    )
    return plan


def recommended_plan():
    st, origin, seeds = grow(existing=True)
    return to_plan(st, origin, seeds, 1.5 * 400.0, existing=True)


def render(plan, name, title):
    H, Wc = plan.shape
    P, x0, y0 = 9, 44, 86
    cw, ch = Wc * P + 88, H * P + 230
    out = [f'<rect width="{cw}" height="{ch}" fill="white"/>']
    out.append(f'<text x="{x0}" y="52" fill="{RED}" font-weight="800" font-size="26">{title}</text>')

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
            else:  # green + none -> nature
                out.append(dot(c, r, GREEN, P * 0.24))
    # legend
    ly, lx = y0 + H * P + 44, x0
    for label, color, big in [("nature", GREEN, False), ("new built", BUILT, True),
                              ("existing built", EXIST_BUILT, True), ("new centre", RED, True),
                              ("existing centre", EXIST_CENTRE, True)]:
        out.append(f'<circle cx="{lx + 9}" cy="{ly - 5}" r="{9 if big else 5}" fill="{color}"/>')
        out.append(f'<text x="{lx + 26}" y="{ly}" fill="{BUILT}" font-family="Arial" font-size="16">{label}</text>')
        lx += 46 + len(label) * 9
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
    render(recommended_plan(), "demo_recommended_plan", "Recommended plan — a worked example")

    # Dispersed development: Off / Moderate / Aggressive (pure growth, no existing core)
    for label, isol in (("off", 0.0), ("moderate", 0.0001), ("aggressive", 0.04)):
        st, origin, seeds = grow(isol=isol, existing=False)
        render(to_plan(st, origin, seeds, 1.5 * 400.0, existing=False),
               f"demo_dispersal_{label}", f"Dispersed development — {label.capitalize()}")

    # Centre clustering: a COMPACT town (one big contiguous mass) so clustering visibly thins the
    # centres; same run + buildings, only the spacing differs.
    st, origin, seeds = grow(isol=0.0, existing=False)
    for label, mult in (("moderate", 1.5), ("tight", 2.5)):
        render(to_plan(st, origin, seeds, mult * 400.0, existing=False),
               f"demo_clustering_{label}", f"Centre clustering — {label.capitalize()}")

    # Build probability: the per-step growth RATE. Cap iterations (huge pop) so the SAME elapsed time
    # shows how much further fast growth reaches.
    for label, bp in (("slow", 0.08), ("fast", 0.6)):
        st, origin, seeds = grow(isol=0.0, bp=bp, existing=False, pop=1_000_000.0, iters=80)
        render(to_plan(st, origin, seeds, 1.5 * 400.0, existing=False),
               f"demo_buildprob_{label}", f"Build probability — {label.capitalize()} ({bp})")


if __name__ == "__main__":
    main()
