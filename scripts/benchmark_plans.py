"""Fair, reproducible comparison of recommended-plan methods on the Cambourne demo.

The earlier ad-hoc benchmark compared apples to oranges (a single run kept all its
centres and counted every green speck as a park, while the consensus is capped and
min-span filtered). This routes EVERY candidate plan through the SAME normalisation
before scoring — identical centre budget and identical "qualifying green" rule — so
the only thing that varies is the extraction method.

Run: uv run python scripts/benchmark_plans.py
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import isobenefit
import numpy as np
import shapely

from isobenefit_qgis.grid import (
    PLAN_BUILT,
    PLAN_CENTRE,
    PLAN_GREEN,
    PLAN_NONE,
    _gravity_centres,
    _keep_large_components,
    capacity_summary,
    evaluate_plan,
    optimise_plan,
    recommended_plan,
)

# density tiers (high, med, low) and their probabilities — as in the demo
DENSITY_FACTORS = (6000.0, 3000.0, 1000.0)
PROB_DISTRIBUTION = (0.4, 0.4, 0.2)
MEAN_DENSITY = sum(p * d for p, d in zip(PROB_DISTRIBUTION, DENSITY_FACTORS))
MAX_DENSITY = max(DENSITY_FACTORS)

DEMO = Path(__file__).resolve().parent.parent / "demo_layers"
GRAN = 100.0
MAX_DISTANCE = 800.0  # walk
GREEN_SPAN = 400.0  # a park must be at least this across to "count"
MAX_CENTRES = 50
ENSEMBLE_N = 64
SINGLE_SEEDS = list(range(1, 17))  # 16 independent single runs for the distribution


def _union(name: str):
    data = json.loads((DEMO / f"{name}.geojson").read_text())
    return shapely.unary_union([shapely.geometry.shape(f["geometry"]) for f in data["features"]])


def load_grid():
    extent = _union("extents")
    x0, y0, x1, y1 = extent.bounds
    xmn, ymn = math.floor(x0 / GRAN) * GRAN, math.floor(y0 / GRAN) * GRAN
    xmx, ymx = math.ceil(x1 / GRAN) * GRAN, math.ceil(y1 / GRAN) * GRAN
    cols, rows = int(round((xmx - xmn) / GRAN)), int(round((ymx - ymn) / GRAN))
    gt = (xmn, GRAN, 0.0, ymx, 0.0, -GRAN)
    xs = gt[0] + (np.arange(cols) + 0.5) * GRAN
    ys = gt[3] - (np.arange(rows) + 0.5) * GRAN
    gx, gy = np.meshgrid(xs, ys)

    def burn(arr, name, value):
        arr[shapely.contains_xy(_union(name), gx, gy)] = value
        return arr

    state = burn(np.full((rows, cols), -1, np.int16), "extents", 0)
    origin = np.full((rows, cols), -1, np.int16)
    for name, val in [("urban", 1), ("green", 0)]:
        burn(state, name, val)
        burn(origin, name, val)
    burn(state, "unbuildable", -1)
    seeds = []
    for f in json.loads((DEMO / "centres.geojson").read_text())["features"]:
        pt = shapely.geometry.shape(f["geometry"])
        c, r = int((pt.x - gt[0]) / GRAN), int((gt[3] - pt.y) / GRAN)
        if 0 <= r < rows and 0 <= c < cols:
            seeds.append((r, c))
    return rows, cols, state, origin, seeds


def make(rows, cols, state, origin, seeds, seed):
    return isobenefit.Simulation(
        state, origin, np.zeros((rows, cols), np.float32), seeds,
        GRAN, MAX_DISTANCE, 10_000_000.0, 100.0, 0.25, 0.05, 0.0, 0.8,
        (0.4, 0.4, 0.2), (6000.0, 3000.0, 1000.0), 2000.0, 100, seed,
    )


def normalise(plan: np.ndarray) -> np.ndarray:
    """Put any plan on equal footing: drop sub-span green, then place exactly
    MAX_CENTRES by the same gravity model. This is the fix for the earlier confound."""
    plan = plan.copy()
    green_min = max(1, round((GREEN_SPAN / GRAN) ** 2))
    keep = _keep_large_components(plan == PLAN_GREEN, green_min)
    plan[(plan == PLAN_GREEN) & ~keep] = PLAN_NONE
    plan[plan == PLAN_CENTRE] = PLAN_BUILT
    built = plan == PLAN_BUILT
    walk_r = max(1, round(MAX_DISTANCE / GRAN))
    for y, x in _gravity_centres(built.astype(float), built, walk_r, MAX_CENTRES):
        plan[y, x] = PLAN_CENTRE
    return plan


def score(plan):
    p = normalise(plan)
    m = evaluate_plan(p, GRAN, MAX_DISTANCE)
    m["built"] = int((p == PLAN_BUILT).sum()) + int((p == PLAN_CENTRE).sum())
    m["green"] = int((p == PLAN_GREEN).sum())
    return m


def single_run_plan(rows, cols, state, origin, seeds, seed):
    sim = make(rows, cols, state, origin, seeds, seed)
    sim.run()
    st = sim.snapshot()["state"]
    plan = np.zeros_like(st, np.uint8)
    plan[st == 0], plan[st == 1], plan[st == 2] = PLAN_GREEN, PLAN_BUILT, PLAN_CENTRE
    return plan


def main():
    rows, cols, state, origin, seeds = load_grid()
    print(f"Cambourne {cols}x{rows}; walk {MAX_DISTANCE:.0f} m; park span {GREEN_SPAN:.0f} m; "
          f"centre cap {MAX_CENTRES}; all plans scored identically\n")

    built, green, _c = isobenefit.ensemble_class_counts(make(rows, cols, state, origin, seeds, 1), 2024, ENSEMBLE_N)
    consensus = recommended_plan(built / ENSEMBLE_N, green / ENSEMBLE_N, GRAN, GREEN_SPAN, MAX_DISTANCE)
    optimised = optimise_plan(
        consensus, GRAN, GREEN_SPAN, MAX_DISTANCE, mean_density=MEAN_DENSITY, max_density=MAX_DENSITY
    )
    singles = [score(single_run_plan(rows, cols, state, origin, seeds, s)) for s in SINGLE_SEEDS]

    hdr = f"{'method':28s}{'served':>8s}{'green_cov':>10s}{'centre_cov':>11s}{'worst':>7s}{'built':>7s}{'green':>7s}"
    print(hdr)
    print("-" * len(hdr))

    def row(tag, m):
        print(f"{tag:28s}{m['served_coverage']:>7.1%}{m['green_coverage']:>10.1%}"
              f"{m['centre_coverage']:>11.1%}{m['worst_benefit']:>7.1%}{m['built']:>7d}{m['green']:>7d}")

    row("consensus (ensemble)", score(consensus))
    row("optimised (+greedy green)", score(optimised))

    def med(key):
        return statistics.median(m[key] for m in singles)

    lo, hi = min(m["served_coverage"] for m in singles), max(m["served_coverage"] for m in singles)
    print(f"{'single run (median of ' + str(len(singles)) + ')':28s}"
          f"{med('served_coverage'):>7.1%}{med('green_coverage'):>10.1%}"
          f"{med('centre_coverage'):>11.1%}{med('worst_benefit'):>7.1%}"
          f"{round(med('built')):>7d}{round(med('green')):>7d}")
    print(f"{'  single-run served range':28s}{lo:>7.1%} .. {hi:.1%}")

    # population accounting for the optimised plan (constant-inhabitants check)
    cs = capacity_summary(score(consensus)["built"], score(optimised)["built"], MEAN_DENSITY, MAX_DENSITY)
    print("\nPopulation check (Isobenefit 'constant inhabitants'):")
    print(f"  population held         {cs['population']:,.0f}")
    print(f"  built cells   {cs['built_before']} -> {cs['built_after']}  "
          f"({(cs['built_before'] - cs['built_after']) / cs['built_before']:.0%} freed to green)")
    print(f"  mean density  {cs['density_before']:,.0f} -> {cs['density_after']:,.0f} / cell "
          f"(max {cs['max_density']:,.0f})")
    print(f"  feasible by densifying the rest: {cs['feasible']}")


if __name__ == "__main__":
    main()
