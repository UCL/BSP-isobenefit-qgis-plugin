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
    PLAN_EXIST_BUILT,
    PLAN_EXIST_CENTRE,
    PLAN_GREEN,
    PLAN_NONE,
    align_bounds,
    capacity_summary,
    class_probabilities,
    classify,
    evaluate_plan,
    optimise_plan,
    recommended_plan,
    select_plan,
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


def test_evaluate_plan_coverage_in_range():
    g = 24
    plan = np.zeros((g, g), np.uint8)
    plan[4:20, 4:20] = PLAN_BUILT  # a built block
    plan[11, 11] = PLAN_CENTRE  # a centre inside it
    plan[4:20, 20:24] = PLAN_GREEN  # green alongside
    m = evaluate_plan(plan, granularity_m=100.0, max_distance_m=800.0)
    for k in ("centre_coverage", "green_coverage", "served_coverage", "unserved_fraction", "compactness"):
        assert 0.0 <= m[k] <= 1.0
    assert abs(m["served_coverage"] + m["unserved_fraction"] - 1.0) < 1e-9  # served + unserved = all
    assert abs(m["access_cost"] - 0.5 * (m["centre_access"] + m["green_access"])) < 1e-6  # combined = mean of halves
    assert m["built_cells"] == int((plan == PLAN_BUILT).sum()) + 1  # centre counts as built
    assert m["centre_coverage"] > 0.0
    # a central centre covers more homes than one shoved into a corner
    far = plan.copy()
    far[11, 11] = PLAN_BUILT
    far[4, 4] = PLAN_CENTRE
    assert evaluate_plan(far, 100.0, 800.0)["centre_coverage"] <= m["centre_coverage"]


def test_evaluate_plan_only_real_parks_count():
    g = 24
    plan = np.full((g, g), PLAN_BUILT, np.uint8)
    plan[0, 0] = PLAN_GREEN  # a single-cell speck, not a park
    assert evaluate_plan(plan, 100.0, 800.0)["green_coverage"] > 0.0  # speck counts when unfiltered
    assert evaluate_plan(plan, 100.0, 800.0, min_green_span_m=400.0)["green_coverage"] == 0.0  # filtered out


def test_evaluate_empty_plan():
    assert evaluate_plan(np.zeros((10, 10), np.uint8), 100.0, 800.0) == {"built_cells": 0}


def test_seed_centres_proximity_covers_built():
    # The original dumb rule: every built cell beyond a walk of a centre becomes one.
    from isobenefit_qgis.grid import _seed_centres_proximity, _walk_distance

    g = 40
    built = np.zeros((g, g), bool)
    built[5:35, 5:35] = True
    seeds = _seed_centres_proximity(built, 100.0, 800.0)
    assert len(seeds) >= 1
    centres = np.zeros((g, g), bool)
    for y, x in seeds:
        centres[y, x] = True
    assert np.isfinite(_walk_distance(centres, 100.0, 800.0)[built]).all()  # every home within a walk


def test_refine_centres_culls_redundant():
    # Two seeds on a small development that one centre already covers -> the redundant one goes.
    from isobenefit_qgis.grid import _refine_centres

    g = 30
    built = np.zeros((g, g), bool)
    built[12:18, 12:18] = True  # small development, well within one 800 m catchment
    out = _refine_centres([(14, 14), (15, 15)], [], built, built, 100.0, 800.0)
    assert len(out) == 1
    assert built[out[0][0], out[0][1]]  # the survivor sits on the development


def test_refine_centres_keeps_separate_developments():
    from isobenefit_qgis.grid import _refine_centres

    g = 70
    built = np.zeros((g, g), bool)
    built[5:25, 5:25] = True  # development A
    built[45:65, 45:65] = True  # development B, beyond a walk away
    out = _refine_centres([(15, 15), (55, 55)], [], built, built, 100.0, 800.0)
    # neither development is dropped — each is served by at least one centre
    assert any(y < 35 and x < 35 for y, x in out)
    assert any(y >= 35 and x >= 35 for y, x in out)
    for y, x in out:
        assert built[y, x]


def test_refine_centres_adds_when_underserved():
    # A development far larger than one catchment gets MORE centres than seeded, so it isn't
    # left underserved.
    from isobenefit_qgis.grid import _refine_centres, _walk_distance

    g = 70
    built = np.zeros((g, g), bool)
    built[5:65, 5:65] = True  # 60x60 dev, far larger than one 800 m catchment
    out = _refine_centres([(34, 34)], [], built, built, 100.0, 800.0)  # under-seeded: one centre
    assert len(out) > 1  # additional centres added for the underserved areas
    cmask = np.zeros((g, g), bool)
    for y, x in out:
        cmask[y, x] = True
    assert np.isfinite(_walk_distance(cmask, 100.0, 800.0)[built]).mean() > 0.8  # most homes served


def test_optimise_plan_centre_optimisation_optional():
    # The centre optimisation is a toggle: off keeps the CA's grown centres exactly where they
    # are; on re-positions them central to their development.
    from isobenefit_qgis.grid import PLAN_BUILT, PLAN_CENTRE, optimise_plan

    g = 40
    plan = np.zeros((g, g), np.uint8)
    plan[10:30, 10:30] = PLAN_BUILT  # a 20x20 development (centroid ~ (19, 19))
    edge = (10, 10)  # the simulation grew a centre in the corner

    off = optimise_plan(plan, 50.0, 400.0, 800.0, max_green_frac=0.0, ca_centres=[edge], optimise_centres=False)
    assert {(int(y), int(x)) for y, x in np.argwhere(off == PLAN_CENTRE)} == {edge}  # kept as-is

    on = optimise_plan(plan, 50.0, 400.0, 800.0, max_green_frac=0.0, ca_centres=[edge], optimise_centres=True)
    centres_on = {(int(y), int(x)) for y, x in np.argwhere(on == PLAN_CENTRE)}
    assert edge not in centres_on  # re-positioned off the corner, toward the centre


def test_evaluate_plan_transit_coverage_reported():
    # Transit access is reported when stops are given, and omitted otherwise; it does not touch
    # the selection metric (access_cost).
    from isobenefit_qgis.grid import PLAN_BUILT, PLAN_CENTRE, evaluate_plan

    g = 50
    plan = np.zeros((g, g), np.uint8)
    plan[10:20, 10:20] = PLAN_BUILT
    plan[14, 14] = PLAN_CENTRE

    stops = np.zeros((g, g), bool)
    stops[14, 14] = True  # one stop, central to the development
    m = evaluate_plan(plan, 50.0, 800.0, transit_stops=stops)
    assert m["transit_coverage"] == 1.0  # whole dev within an 800 m walk of the stop
    assert m["transit_access"] < 800.0

    far = np.zeros((g, g), bool)
    far[48, 48] = True  # stop in the far corner, well beyond a walk of the development
    assert evaluate_plan(plan, 50.0, 800.0, transit_stops=far)["transit_coverage"] == 0.0

    assert "transit_coverage" not in evaluate_plan(plan, 50.0, 800.0)  # omitted with no stops


def test_optimise_plan_anchors_centre_at_station():
    # A significant transit stop on built land anchors a fixed centre, kept where it is; a stop
    # off built land is ignored (a centre needs homes to serve).
    from isobenefit_qgis.grid import PLAN_BUILT, PLAN_CENTRE, optimise_plan

    g = 40
    plan = np.zeros((g, g), np.uint8)
    plan[10:30, 10:30] = PLAN_BUILT

    anchor = (12, 12)  # a station near the development edge
    out = optimise_plan(plan, 50.0, 400.0, 800.0, max_green_frac=0.0, ca_centres=[], centre_anchors=[anchor])
    assert anchor in {(int(y), int(x)) for y, x in np.argwhere(out == PLAN_CENTRE)}  # anchored + kept

    out2 = optimise_plan(plan, 50.0, 400.0, 800.0, max_green_frac=0.0, ca_centres=[], centre_anchors=[(0, 0)])
    assert (0, 0) not in {(int(y), int(x)) for y, x in np.argwhere(out2 == PLAN_CENTRE)}  # off built -> ignored


def test_evaluate_plan_uses_injected_router():
    # evaluate_plan routes via an injected callable (mask -> rows x cols metres) instead of the
    # grid walk, so true network distances can drive the metrics. grid.py stays QGIS-free.
    from isobenefit_qgis.grid import PLAN_BUILT, PLAN_CENTRE, evaluate_plan

    g = 20
    plan = np.zeros((g, g), np.uint8)
    plan[5:15, 5:15] = PLAN_BUILT
    plan[10, 10] = PLAN_CENTRE

    calls = []

    def router(mask):
        calls.append(mask)
        return np.full((g, g), 137.0)  # a distinctive constant the grid walk would never produce

    m = evaluate_plan(plan, 50.0, 800.0, router=router)
    assert calls  # the router was used in place of the grid walk
    assert m["centre_access"] == 137.0 and m["green_access"] == 137.0  # straight from the router
    assert m["centre_coverage"] == 1.0  # 137 m < 800 m walk


def test_optimise_plan_threads_one_distance_model_to_placement():
    # The injected distance model drives centre PLACEMENT too, not only scoring — one metric
    # throughout. With nothing reachable, no centre can justify itself, so none are placed.
    from isobenefit_qgis.grid import PLAN_BUILT, PLAN_CENTRE, optimise_plan

    g = 30
    plan = np.zeros((g, g), np.uint8)
    plan[10:20, 10:20] = PLAN_BUILT
    calls = []

    def router(mask):
        calls.append(True)
        return np.full((g, g), np.inf)  # nothing is within a walk of anything

    out = optimise_plan(plan, 50.0, 400.0, 800.0, max_green_frac=0.0, ca_centres=[(15, 15)], router=router)
    assert calls  # the injected model was used inside the optimiser, not just the scorer
    assert not (out == PLAN_CENTRE).any()  # unreachable -> no centre is warranted


def test_network_router_uses_graph_distance():
    # A U-shaped graph: A-M1-M2-B, each leg 100 m, so A->B ALONG the network is 300 m even though
    # the straight-line A-B is only 100 m. The router must report the network distance.
    from isobenefit_qgis.routing import NetworkRouter

    nodes = np.array([[0, 0], [0, 100], [100, 100], [100, 0]], float)  # A, M1, M2, B
    adj = [[(1, 100.0)], [(0, 100.0), (2, 100.0)], [(1, 100.0), (3, 100.0)], [(2, 100.0)]]
    cell_node = np.array([[0, 3], [-1, -1]])  # cell (0,0)->A, (0,1)->B, bottom row off-network
    cell_access = np.array([[0.0, 0.0], [np.inf, np.inf]])

    router = NetworkRouter(nodes, adj, cell_node, cell_access, 50.0, 1000.0)
    field = router(np.array([[False, True], [False, False]]))  # target = cell (0,1) == node B
    assert field[0, 1] == 0.0  # the target cell itself
    assert field[0, 0] == 300.0  # A -> B is the 300 m network distance, not the 100 m straight line
    assert not np.isfinite(field[1, 0])  # off-network cell is unreachable

    bounded = NetworkRouter(nodes, adj, cell_node, cell_access, 50.0, 200.0)
    field2 = bounded(np.array([[False, True], [False, False]]))
    assert not np.isfinite(field2[0, 0])  # 300 m exceeds the 200 m walk limit -> bounded out


def test_network_router_caches_single_source():
    # single-source queries (the hot path: one centre) are solved once per node and reused
    from isobenefit_qgis.routing import NetworkRouter

    nodes = np.array([[0, 0], [0, 100], [100, 100], [100, 0]], float)
    adj = [[(1, 100.0)], [(0, 100.0), (2, 100.0)], [(1, 100.0), (3, 100.0)], [(2, 100.0)]]
    cell_node = np.array([[0, 3], [-1, -1]])
    cell_access = np.array([[0.0, 0.0], [np.inf, np.inf]])
    router = NetworkRouter(nodes, adj, cell_node, cell_access, 50.0, 1000.0)

    one = np.array([[True, False], [False, False]])  # single source = node 0
    f1 = router(one)
    assert 0 in router._cache  # node 0's distances were solved and cached
    f2 = router(one)
    assert np.array_equal(f1, f2)  # second call reuses the cache, same result


def test_snap_cells_finds_nearest_node():
    from isobenefit_qgis.routing import _snap_cells

    nodes = np.array([[25.0, -25.0]])  # sits exactly at the centre of cell (0, 0)
    gt = (0.0, 50.0, 0.0, 0.0, 0.0, -50.0)  # 50 m cells, north-up
    cell_node, cell_access = _snap_cells(nodes, gt, 1, 1, 800.0)
    assert cell_node[0, 0] == 0 and cell_access[0, 0] < 1.0


def test_audit_centres_reports_served_and_flags_weak():
    from isobenefit_qgis.grid import PLAN_BUILT, PLAN_CENTRE, audit_centres

    g = 40
    plan = np.zeros((g, g), np.uint8)
    plan[20:26, 20:26] = PLAN_BUILT  # a 6x6 = 36-cell development
    plan[22, 22] = PLAN_CENTRE  # one well-placed centre
    a = audit_centres(plan, 50.0, 800.0)
    assert a["summary"]["n_centres"] == 1
    assert a["centres"][0]["served"] == 36  # serves the whole development
    assert 0 < a["centres"][0]["mean_dist_m"] < 800

    plan[0, 0] = PLAN_CENTRE  # a lone speck far from the development — serves only itself
    a2 = audit_centres(plan, 50.0, 800.0)
    assert a2["summary"]["n_centres"] == 2
    assert a2["centres"][0]["served"] == 1  # weakest-first ordering surfaces the dud
    assert a2["summary"]["served_min"] == 1


def test_audit_centres_counts_areas_not_cells():
    # a centre is an AREA — a contiguous block counts as one centre, not one-per-cell
    from isobenefit_qgis.grid import PLAN_BUILT, PLAN_CENTRE, audit_centres

    g = 30
    plan = np.zeros((g, g), np.uint8)
    plan[10:20, 10:20] = PLAN_BUILT
    plan[13:16, 13:16] = PLAN_CENTRE  # a 3x3 = 9-cell centre area
    a = audit_centres(plan, 50.0, 800.0)
    assert a["summary"]["n_centres"] == 1  # one area, not nine cells
    assert a["centres"][0]["cells"] == 9


def test_optimise_plan_grows_centre_into_area():
    # a new centre is grown into an AREA sized by the homes it serves, on the development, not a dot
    from isobenefit_qgis.grid import PLAN_BUILT, PLAN_CENTRE, optimise_plan

    g = 50
    plan = np.zeros((g, g), np.uint8)
    plan[10:40, 10:40] = PLAN_BUILT  # a 30x30 = 900-cell development
    out = optimise_plan(plan, 50.0, 400.0, 800.0, max_green_frac=0.0, ca_centres=[(25, 25)])
    centre_cells = [(int(y), int(x)) for y, x in np.argwhere(out == PLAN_CENTRE)]
    assert len(centre_cells) > 1  # grew into an area, not a single cell
    assert all(10 <= y < 40 and 10 <= x < 40 for y, x in centre_cells)  # stays on the development


def test_optimise_plan_culls_tiny_ca_centre():
    # A CA centre feeding a 2-cell speck is culled; the one for the real development is kept.
    g = 40
    plan = np.zeros((g, g), np.uint8)
    plan[10:30, 10:30] = PLAN_BUILT  # real development
    plan[2, 36:38] = PLAN_BUILT  # an isolated 2-cell speck
    out = optimise_plan(plan, 100.0, 400.0, 800.0, max_green_frac=0.0, ca_centres=[(20, 20), (2, 36)])
    cs = [(int(y), int(x)) for y, x in np.argwhere(out == PLAN_CENTRE)]
    assert any(10 <= y < 30 and 10 <= x < 30 for y, x in cs)  # real development keeps a centre
    assert not any(y < 5 and x >= 35 for y, x in cs)  # the 2-cell speck's centre is culled


def test_walk_distance_routes_around_green_barrier():
    from isobenefit_qgis.grid import _walk_distance

    g = 11
    targets = np.zeros((g, g), bool)
    targets[5, 0] = True
    open_d = _walk_distance(targets, 100.0, 1e6)
    full_wall = np.zeros((g, g), bool)
    full_wall[:, 5] = True  # spans the full height — fully separates the far side
    walled = _walk_distance(targets, 100.0, 1e6, blocked=full_wall)
    assert np.isfinite(open_d[5, 9])  # reachable with no barrier
    assert np.isinf(walled[5, 9])  # a full green wall can't be crossed
    gap_wall = np.zeros((g, g), bool)
    gap_wall[0:9, 5] = True  # leaves a gap at the bottom
    around = _walk_distance(targets, 100.0, 1e6, blocked=gap_wall)
    assert np.isfinite(around[5, 9])  # reachable by routing around the green
    assert around[5, 9] > open_d[5, 9]  # but a longer walk than straight across


def test_centres_do_not_bridge_open_gaps():
    # Two built-up areas separated by OPEN land (not carved green) must not share a centre
    # reaching across the gap — each gets its own, centred in its own contiguous area.
    g = 40
    plan = np.zeros((g, g), np.uint8)
    plan[5:35, 4:16] = PLAN_BUILT  # left built-up area
    plan[5:35, 24:36] = PLAN_BUILT  # right built-up area (open gap cols 16-23)
    out = optimise_plan(plan, 100.0, 400.0, 800.0, max_green_frac=0.0)
    xs = [int(x) for _, x in np.argwhere(out == PLAN_CENTRE)]
    assert any(x < 16 for x in xs)  # left served
    assert any(x >= 24 for x in xs)  # right served
    assert not any(16 <= x < 24 for x in xs)  # nothing stranded in the open gap


def test_optimise_never_prunes_existing_built():
    # The green-carve may free NEW built to parks, but must never touch existing built.
    g = 40
    plan = np.full((g, g), PLAN_BUILT, np.uint8)  # solid built, no green -> carve wants parks
    existing = np.zeros((g, g), bool)
    existing[0:20, :] = True  # the top half is existing built (frozen)
    out = optimise_plan(plan, 100.0, 400.0, 800.0, max_green_frac=0.4, existing_built=existing)
    assert not ((out == PLAN_GREEN) & existing).any()  # not one existing cell pruned to green
    assert (out == PLAN_GREEN).any()  # but new (bottom) land was greened — carve still works


def test_select_plan_freezes_and_tags_existing():
    g = 40
    state = np.ones((g, g), np.int16)  # entirely built
    existing_built = np.zeros((g, g), bool)
    existing_built[0:20, :] = True  # top half already developed
    plan, _m = select_plan(
        [state], 100.0, 400.0, 800.0, existing_centres=[(5, 5)], existing_built=existing_built
    )
    assert plan is not None
    assert not ((plan == PLAN_GREEN) & existing_built).any()  # existing never pruned to green
    assert not ((plan == PLAN_BUILT) & existing_built).any()  # existing built tagged existing, not "new"
    assert (plan == PLAN_EXIST_BUILT).any()
    assert plan[5, 5] == PLAN_EXIST_CENTRE  # existing centre tagged with its own (hue) code


def test_true_area_centres_marked_on_plan():
    # An existing centre AREA (not a point) should come back entirely as PLAN_CENTRE.
    g = 40
    plan = np.zeros((g, g), np.uint8)
    plan[5:35, 5:35] = PLAN_BUILT  # built block
    centre_cells = [(r, c) for r in range(10, 16) for c in range(10, 16)]  # 6x6 centre area
    out = optimise_plan(plan, 100.0, 400.0, 800.0, max_green_frac=0.0, existing_centres=centre_cells)
    assert set(np.unique(out)).issubset({PLAN_NONE, PLAN_GREEN, PLAN_BUILT, PLAN_CENTRE})
    for r, c in centre_cells:
        assert out[r, c] == PLAN_CENTRE  # every covered, built cell is a centre cell


def test_class_probabilities():
    s1 = np.array([[1, 0], [2, 1]], np.int16)
    s2 = np.array([[1, 0], [0, 1]], np.int16)
    pb, pg = class_probabilities([s1, s2])  # centre likelihood intentionally not emitted
    assert pb.dtype == np.float32
    assert pb[0, 0] == 1.0  # built in both runs
    assert pg[0, 1] == 1.0  # green in both


def test_select_plan_on_demo(grid):
    states = isobenefit.run_ensemble(_make(grid), 7, 6)
    md = sum(p * d for p, d in zip((0.4, 0.4, 0.2), (6000.0, 3000.0, 1000.0)))
    plan, m = select_plan(
        states, GRAN, 400.0, 800.0, mean_density=md, max_density=6000.0, existing_centres=grid["seeds"]
    )
    assert plan.shape == (grid["rows"], grid["cols"])
    assert set(np.unique(plan)).issubset(
        {PLAN_NONE, PLAN_GREEN, PLAN_BUILT, PLAN_CENTRE, PLAN_EXIST_BUILT, PLAN_EXIST_CENTRE}
    )
    assert (plan == PLAN_BUILT).any() and ((plan == PLAN_CENTRE) | (plan == PLAN_EXIST_CENTRE)).any()
    assert 0.0 <= m["served_coverage"] <= 1.0 and m["access_cost"] > 0.0


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
