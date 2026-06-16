"""Headless pipeline tests — no QGIS required.

Exercises the plugin's pure logic (``isobenefit_qgis.grid``) and the full
simulation pipeline on the **real Cambourne demo data**, rasterised with shapely
(the geojsons are EPSG:27700, so no reprojection is needed for the test). The
QGIS/GDAL-coupled glue in ``gis_io`` is covered by in-QGIS testing instead.

Run:
    uv run --no-project \
        --with core/dist/isobenefit-*.whl --with numpy --with shapely --with pytest \
        python -m pytest tests -q
"""

from __future__ import annotations

import json
from pathlib import Path

import isobenefit
import numpy as np
import pytest
import shapely

from isobenefit_qgis.grid import (
    EXIST_BUILT,
    NATURE,
    NODATA,
    PLAN_BUILT,
    PLAN_CENTRE,
    PLAN_GREEN,
    PLAN_NONE,
    align_bounds,
    capacity_summary,
    classify,
    evaluate_plan,
    optimise_plan,
    recommended_plan,
)

DEMO = Path(__file__).resolve().parent.parent / "demo_layers"
GRAN = 100.0


def _union(path: Path):
    data = json.loads(path.read_text())
    return shapely.unary_union([shapely.geometry.shape(f["geometry"]) for f in data["features"]])


# --- pure-logic unit tests (numpy only) -------------------------------------


def test_align_bounds_snaps_to_cells():
    rows, cols, gt, bounds = align_bounds(10.0, 20.0, 290.0, 310.0, 100.0)
    assert (rows, cols) == (4, 3)
    assert gt == (0.0, 100.0, 0.0, 400.0, 0.0, -100.0)
    assert bounds == (0.0, 0.0, 300.0, 400.0)


def test_classify_codes_and_precedence():
    per = (60.0, 30.0, 10.0)  # high, med, low
    state = np.array([[0, 1, 2], [1, 1, -1]], dtype=np.int16)
    origin = np.array([[-1, -1, -1], [1, -1, -1]], dtype=np.int16)
    density = np.array([[0, 10, 0], [60, 30, 0]], dtype=np.float32)
    cls = classify(state, origin, density, per)
    assert cls.dtype == np.uint8
    assert cls[0, 0] == NATURE
    assert cls[1, 0] == EXIST_BUILT  # origin==1 overrides the density tier
    assert cls[1, 2] == NODATA  # unbuildable / out of bounds
    assert cls[0, 1] == 1  # NEW_LOW (density 10 == low)


# --- end-to-end on real demo data ------------------------------------------


@pytest.fixture(scope="module")
def grid():
    extent = _union(DEMO / "extents.geojson")
    rows, cols, gt, _ = align_bounds(*extent.bounds, GRAN)
    xs = gt[0] + (np.arange(cols) + 0.5) * GRAN
    ys = gt[3] - (np.arange(rows) + 0.5) * GRAN
    gx, gy = np.meshgrid(xs, ys)

    def burn(arr, name, value):
        arr[shapely.contains_xy(_union(DEMO / f"{name}.geojson"), gx, gy)] = value
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
    return {
        "rows": rows,
        "cols": cols,
        "state": state,
        "origin": origin,
        "density": np.zeros((rows, cols), np.float32),
        "seeds": seeds,
    }


def _make(grid, total_iters=50, seed=42):
    return isobenefit.Simulation(
        grid["state"],
        grid["origin"],
        grid["density"],
        grid["seeds"],
        GRAN,
        800.0,
        10_000_000.0,
        100.0,
        0.25,
        0.05,
        0.0,
        0.8,
        (0.4, 0.4, 0.2),
        (6000.0, 3000.0, 1000.0),
        2000.0,
        total_iters,
        seed,
    )


def test_demo_rasterises_sensibly(grid):
    assert grid["rows"] > 50 and grid["cols"] > 50
    assert len(grid["seeds"]) > 0
    assert int((grid["state"] == 1).sum()) > 0  # some existing urban


def test_single_run_grows_and_is_deterministic(grid):
    a, b = _make(grid), _make(grid)
    before = a.population
    a.run()
    b.run()
    assert a.population > before
    np.testing.assert_array_equal(a.snapshot()["state"], b.snapshot()["state"])


def test_ensemble_probability_on_demo(grid):
    prob = isobenefit.ensemble_probability(_make(grid), 2024, 8)
    assert prob.shape == (grid["rows"], grid["cols"])
    assert prob.dtype == np.float32
    assert float(prob.min()) >= 0.0 and float(prob.max()) <= 1.0
    # cells that start urban are urban in every member -> probability 1.0
    assert float(prob[grid["state"] == 1].min()) == 1.0
    # some speculative (0<p<1) development occurs
    assert bool(((prob > 0.0) & (prob < 1.0)).any())


def test_recommended_plan_synthetic():
    g = 20
    p_green = np.zeros((g, g), np.float32)
    p_built = np.zeros((g, g), np.float32)
    p_green[2:12, 2:12] = 1.0  # large green block (100 cells) -> kept
    p_green[0, 18] = 1.0  # 1-cell sliver -> dropped (min area 9 cells)
    p_built[14:18, 2:8] = 1.0  # built region (24 cells) -> kept
    p_built[19, 19] = 1.0  # 1-cell built drip -> dropped (min 6 cells)
    plan = recommended_plan(p_built, p_green, granularity_m=100.0, min_green_span_m=300.0, max_distance_m=300.0)
    assert set(np.unique(plan)).issubset({PLAN_NONE, PLAN_GREEN, PLAN_BUILT, PLAN_CENTRE})
    assert plan[5, 5] == PLAN_GREEN  # inside the large block
    assert plan[0, 18] == PLAN_NONE  # green sliver dropped by the min-area rule
    assert plan[19, 19] == PLAN_NONE  # built drip dropped by the min-area rule
    assert plan[15, 5] in (PLAN_BUILT, PLAN_CENTRE)  # built region survives
    # a centre is placed by the gravity model, sitting inside the built fabric
    centres = np.argwhere(plan == PLAN_CENTRE)
    assert len(centres) >= 1
    for cy, cx in centres:
        assert 14 <= cy < 18 and 2 <= cx < 8


def test_recommended_plan_on_demo(grid):
    n = 8
    built, green, _centre = isobenefit.ensemble_class_counts(_make(grid), 2024, n)
    plan = recommended_plan(built / n, green / n, GRAN, 100.0, 800.0)
    assert plan.shape == (grid["rows"], grid["cols"])
    assert set(np.unique(plan)).issubset({PLAN_NONE, PLAN_GREEN, PLAN_BUILT, PLAN_CENTRE})
    assert (plan == PLAN_GREEN).any() and (plan == PLAN_BUILT).any()
    assert int((plan == PLAN_CENTRE).sum()) >= 1


def test_evaluate_plan_metrics_in_range():
    g = 24
    plan = np.zeros((g, g), np.uint8)
    plan[4:20, 4:20] = PLAN_BUILT  # a built block
    plan[11, 11] = PLAN_CENTRE  # a centre inside it
    plan[4:20, 20:24] = PLAN_GREEN  # green alongside
    m = evaluate_plan(plan, granularity_m=100.0, max_distance_m=800.0)
    for k in ("centre_coverage", "green_coverage", "served_coverage", "mean_benefit", "worst_benefit", "compactness"):
        assert 0.0 <= m[k] <= 1.0
    assert m["built_cells"] == int((plan == PLAN_BUILT).sum()) + 1  # centre counts as built
    assert m["centre_coverage"] > 0.0  # the centre serves its block
    # a centre embedded in the block beats one shoved into a corner (equity drops)
    far = plan.copy()
    far[11, 11] = PLAN_BUILT
    far[4, 4] = PLAN_CENTRE
    assert evaluate_plan(far, 100.0, 800.0)["worst_benefit"] <= m["worst_benefit"]


def test_evaluate_empty_plan():
    assert evaluate_plan(np.zeros((10, 10), np.uint8), 100.0, 800.0) == {"built_cells": 0}


def test_optimise_plan_improves_green_access():
    g = 40
    plan = np.full((g, g), PLAN_BUILT, np.uint8)  # all built, no green
    plan[20, 20] = PLAN_CENTRE
    before = evaluate_plan(plan, 100.0, 800.0)
    assert before["green_coverage"] == 0.0  # nothing green to reach

    opt = optimise_plan(plan, 100.0, min_green_span_m=400.0, max_distance_m=800.0, max_green_frac=0.3)
    after = evaluate_plan(opt, 100.0, 800.0)
    assert opt.shape == plan.shape
    assert set(np.unique(opt)).issubset({PLAN_NONE, PLAN_GREEN, PLAN_BUILT, PLAN_CENTRE})
    assert after["green_coverage"] > before["green_coverage"]
    assert after["served_coverage"] > before["served_coverage"]
    assert (opt == PLAN_CENTRE).any()  # centres re-placed on the reduced fabric
    # green spent stays within budget (strict, no overshoot)
    n_built0 = int(((plan == PLAN_BUILT) | (plan == PLAN_CENTRE)).sum())
    assert int((opt == PLAN_GREEN).sum()) <= 0.3 * n_built0


def test_optimise_plan_population_aware_never_overhouses():
    # all-built grid; mean density 3800, max 6000 -> may free up to 1-3800/6000 = 36.7%
    g = 40
    plan = np.full((g, g), PLAN_BUILT, np.uint8)
    plan[20, 20] = PLAN_CENTRE
    n0 = int(((plan == PLAN_BUILT) | (plan == PLAN_CENTRE)).sum())
    opt = optimise_plan(plan, 100.0, 400.0, 800.0, mean_density=3800.0, max_density=6000.0)
    n1 = int(((opt == PLAN_BUILT) | (opt == PLAN_CENTRE)).sum())
    freed = n0 - n1
    assert 0 < freed <= round((1.0 - 3800.0 / 6000.0) * n0)  # within the densification headroom
    # the population is genuinely re-housed by densifying the rest, not deleted
    summary = capacity_summary(n0, n1, 3800.0, 6000.0)
    assert summary["feasible"]
    assert summary["density_after"] <= 6000.0
    assert summary["population"] == n0 * 3800.0


def test_capacity_summary_flags_infeasible():
    # freeing too much (no densities given -> flat 20% on a tiny grid) should still re-house
    s = capacity_summary(built_before=1000, built_after=500, mean_density=3800.0, max_density=6000.0)
    assert not s["feasible"]  # would need 7600 > 6000
    assert s["density_after"] == 7600.0
