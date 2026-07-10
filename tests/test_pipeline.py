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
    from isobenefit_qgis.grid import CENTRE_HIGH, CENTRE_LOW, CENTRE_MED, EXIST_CENTRE, FIXED_GREEN, NEW_HIGH, NEW_MED

    per = (60.0, 30.0, 10.0)  # high, med, low
    state = np.array([[0, 1, 2], [1, 1, -1], [2, 2, 1], [2, 0, 0]], dtype=np.int16)
    origin = np.array([[-1, -1, -1], [1, -1, -1], [-1, -1, -1], [2, 0, -1]], dtype=np.int16)
    density = np.array([[0, 10, 30], [60, 30, 0], [60, 10, 60], [0, 0, 0]], dtype=np.float32)
    cls = classify(state, origin, density, per)
    assert cls.dtype == np.uint8
    assert cls[0, 0] == NATURE
    assert cls[1, 0] == EXIST_BUILT  # origin==1 overrides the density tier
    assert cls[1, 2] == NODATA  # unbuildable / out of bounds
    assert cls[0, 1] == 1  # NEW_LOW (density 10 == low)
    assert cls[1, 1] == NEW_MED and cls[2, 2] == NEW_HIGH  # the other two built tiers
    # new centres split by their drawn tier, in the centre hue
    assert cls[0, 2] == CENTRE_MED and cls[2, 0] == CENTRE_HIGH and cls[2, 1] == CENTRE_LOW
    # existing centre seeds (origin==2) and fixed green (origin==0) take their own codes
    assert cls[3, 0] == EXIST_CENTRE
    assert cls[3, 1] == FIXED_GREEN


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


def test_interior_point():
    from isobenefit_qgis.grid import _interior_point

    assert _interior_point(np.zeros((5, 5), bool)) is None
    sq = np.zeros((9, 9), bool)
    sq[1:8, 1:8] = True
    assert _interior_point(sq) == (4, 4)  # dead centre of a solid square
    ll = np.zeros((9, 9), bool)  # an L-shape: the interior point must be ON the L, not in the notch
    ll[1:8, 1:3] = True
    ll[5:8, 1:8] = True
    y, x = _interior_point(ll)
    assert ll[y, x]


def test_refine_centres_concave_catchment_centres_on_built_interior():
    # A ring (hollow square) of built: the catchment centroid lands in the HOLE (off built), where the
    # old centroid+nearest-built would snap the centre onto the inner RIM. The fix places it at the
    # catchment's interior instead — on built, deep in the band, not on a rim or in the gap.
    from isobenefit_qgis.grid import _refine_centres

    g = 60
    built = np.zeros((g, g), bool)
    built[10:50, 10:50] = True
    built[24:36, 24:36] = False  # hollow centre -> a ring; its centroid is in the hole
    out = _refine_centres([(30, 30)], [], built, built, 100.0, 4000.0)  # one big catchment = the whole ring
    assert out
    for y, x in out:
        assert built[y, x]  # on built
        assert built[y - 1 : y + 2, x - 1 : x + 2].all()  # interior of the band, not snapped to the rim


def test_optimise_plan_centre_optimisation_optional():
    # The centre optimisation is a toggle: off keeps the CA's grown centres exactly where they
    # are; on re-positions them central to their development.
    from isobenefit_qgis.grid import PLAN_BUILT, PLAN_CENTRE, optimise_plan

    g = 40
    plan = np.zeros((g, g), np.uint8)
    plan[10:30, 10:30] = PLAN_BUILT  # a 20x20 development (centroid ~ (19, 19))
    edge = (10, 10)  # the simulation grew a centre in the corner

    off = optimise_plan(plan, 50.0, 400.0, 800.0, ca_centres=[edge], optimise_centres=False)
    assert {(int(y), int(x)) for y, x in np.argwhere(off == PLAN_CENTRE)} == {edge}  # kept as-is

    on = optimise_plan(plan, 50.0, 400.0, 800.0, ca_centres=[edge], optimise_centres=True)
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
    out = optimise_plan(plan, 50.0, 400.0, 800.0, ca_centres=[], centre_anchors=[anchor])
    assert anchor in {(int(y), int(x)) for y, x in np.argwhere(out == PLAN_CENTRE)}  # anchored + kept

    out2 = optimise_plan(plan, 50.0, 400.0, 800.0, ca_centres=[], centre_anchors=[(0, 0)])
    assert (0, 0) not in {(int(y), int(x)) for y, x in np.argwhere(out2 == PLAN_CENTRE)}  # off built -> ignored


def test_optimise_plan_grows_station_anchor():
    # A station anchor seeds a real centre AREA (grown by its catchment), not a lone cell.
    from isobenefit_qgis.grid import PLAN_BUILT, PLAN_CENTRE, _components, optimise_plan

    g = 50
    plan = np.zeros((g, g), np.uint8)
    plan[10:40, 10:40] = PLAN_BUILT  # 30x30 development around the station
    out = optimise_plan(plan, 50.0, 400.0, 800.0, ca_centres=[], centre_anchors=[(15, 15)])
    centres = {(int(y), int(x)) for y, x in np.argwhere(out == PLAN_CENTRE)}
    assert (15, 15) in centres  # the station is a centre
    comp = next(c for c in _components(out == PLAN_CENTRE) if (15, 15) in c)
    assert len(comp) > 1  # ...and it grew into an area around the station, not a single cell


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

    out = optimise_plan(plan, 50.0, 400.0, 800.0, ca_centres=[(15, 15)], router=router)
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
    out = optimise_plan(plan, 50.0, 400.0, 800.0, ca_centres=[(25, 25)])
    centre_cells = [(int(y), int(x)) for y, x in np.argwhere(out == PLAN_CENTRE)]
    assert len(centre_cells) > 1  # grew into an area, not a single cell
    assert all(10 <= y < 40 and 10 <= x < 40 for y, x in centre_cells)  # stays on the development


def test_optimise_plan_culls_tiny_ca_centre():
    # A CA centre feeding a 2-cell speck is culled; the one for the real development is kept.
    g = 40
    plan = np.zeros((g, g), np.uint8)
    plan[10:30, 10:30] = PLAN_BUILT  # real development
    plan[2, 36:38] = PLAN_BUILT  # an isolated 2-cell speck
    out = optimise_plan(plan, 100.0, 400.0, 800.0, ca_centres=[(20, 20), (2, 36)])
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
    out = optimise_plan(plan, 100.0, 400.0, 800.0)
    xs = [int(x) for _, x in np.argwhere(out == PLAN_CENTRE)]
    assert any(x < 16 for x in xs)  # left served
    assert any(x >= 24 for x in xs)  # right served
    assert not any(16 <= x < 24 for x in xs)  # nothing stranded in the open gap


def test_optimise_never_prunes_existing_built():
    # The failed-satellite prune removes only NEW detached specks — a small EXISTING cluster is frozen.
    g = 40
    plan = np.zeros((g, g), np.uint8)
    plan[10:30, 10:30] = PLAN_BUILT  # main development
    plan[3:5, 3:5] = PLAN_BUILT  # a tiny 2x2 cluster...
    existing = np.zeros((g, g), bool)
    existing[3:5, 3:5] = True  # ...that is EXISTING (frozen), so the prune must leave it
    out = optimise_plan(
        plan, 50.0, 400.0, 800.0, existing_built=existing, ca_centres=[(20, 20)], centre_min_settlement=12
    )
    assert (out[3:5, 3:5] != PLAN_GREEN).all()  # the existing cluster is not pruned to green


def test_select_plan_freezes_and_tags_existing():
    g = 40
    state = np.ones((g, g), np.int16)  # entirely built
    existing_built = np.zeros((g, g), bool)
    existing_built[0:20, :] = True  # top half already developed
    plan, _m, _pre, _st = select_plan(
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
    out = optimise_plan(plan, 100.0, 400.0, 800.0, existing_centres=centre_cells)
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
    plan, m, pre, st = select_plan(states, GRAN, 400.0, 800.0, existing_centres=grid["seeds"])
    assert plan.shape == (grid["rows"], grid["cols"])
    assert pre is not None and pre.shape == plan.shape  # the raw (pre-processing) plan is returned too
    assert st is not None and st.shape == plan.shape  # the chosen run's state, for compactness variants
    assert set(np.unique(plan)).issubset(
        {PLAN_NONE, PLAN_GREEN, PLAN_BUILT, PLAN_CENTRE, PLAN_EXIST_BUILT, PLAN_EXIST_CENTRE}
    )
    assert (plan == PLAN_BUILT).any() and ((plan == PLAN_CENTRE) | (plan == PLAN_EXIST_CENTRE)).any()
    assert 0.0 <= m["served_coverage"] <= 1.0 and m["access_cost"] > 0.0




def test_refine_centres_spacing_consolidates_beyond_walk():
    # A spacing LARGER than the walk genuinely consolidates: fewer, larger centres than the
    # coverage-minimal default. (This is the fix — the spacing used to be capped at the walk.)
    from isobenefit_qgis.grid import _refine_centres

    g = 100
    built = np.zeros((g, g), bool)
    built[15:85, 15:85] = True  # a big block, far larger than one catchment
    coverage_min = _refine_centres([(50, 50)], [], built, built, 50.0, 400.0)  # spacing defaults to the walk
    consolidated = _refine_centres([(50, 50)], [], built, built, 50.0, 400.0, spacing_m=1000.0)  # 2.5x the walk
    assert len(consolidated) < len(coverage_min)  # spacing > walk -> fewer, clumped centres
    assert consolidated  # ...but still at least one


def test_refine_centres_spacing_consolidated_vs_dispersed():
    # The spacing dial sets how far apart centres sit. A big block with a tight spacing keeps many,
    # close centres (dispersed); the default (= the walk) keeps the coverage-minimal few (consolidated).
    from isobenefit_qgis.grid import _refine_centres

    g = 80
    built = np.zeros((g, g), bool)
    built[10:70, 10:70] = True  # 60x60 block, far larger than one catchment
    consolidated = _refine_centres([(40, 40)], [], built, built, 50.0, 800.0)  # spacing defaults to the walk
    dispersed = _refine_centres([(40, 40)], [], built, built, 50.0, 800.0, spacing_m=300.0)
    assert len(dispersed) > len(consolidated)  # tighter spacing -> more centres
    for y, x in dispersed:
        assert built[y, x]


def test_optimise_plan_centre_spacing_disperses():
    from isobenefit_qgis.grid import PLAN_BUILT, PLAN_CENTRE, _components, optimise_plan

    g = 80
    plan = np.zeros((g, g), np.uint8)
    plan[10:70, 10:70] = PLAN_BUILT
    common = dict(ca_centres=[(40, 40)], optimise_centres=True)
    cons = optimise_plan(plan, 50.0, 400.0, 800.0, centre_spacing_m=800.0, **common)
    disp = optimise_plan(plan, 50.0, 400.0, 800.0, centre_spacing_m=300.0, **common)
    # count centre AREAS (connected components), not cells, since centres are grown blobs
    n_cons = len(_components(cons == PLAN_CENTRE))
    n_disp = len(_components(disp == PLAN_CENTRE))
    assert n_disp > n_cons


def test_optimise_plan_centre_area_scales_per_person():
    # Centres are sized by the POPULATION they serve: more m² per person -> bigger centre, and the
    # same dial at a higher density (more people in the catchment) -> bigger centre too.
    from isobenefit_qgis.grid import PLAN_BUILT, PLAN_CENTRE, optimise_plan

    g = 60
    plan = np.zeros((g, g), np.uint8)
    plan[10:50, 10:50] = PLAN_BUILT
    common = dict(ca_centres=[(30, 30)], optimise_centres=True, centre_spacing_m=700.0)
    small = optimise_plan(plan, 50.0, 400.0, 800.0, centre_m2_per_person=5.0, **common)
    large = optimise_plan(plan, 50.0, 400.0, 800.0, centre_m2_per_person=30.0, **common)
    assert int((large == PLAN_CENTRE).sum()) > int((small == PLAN_CENTRE).sum())  # more provision per person
    dense = optimise_plan(plan, 50.0, 400.0, 800.0, centre_m2_per_person=5.0, new_density_km2=9000.0, **common)
    assert int((dense == PLAN_CENTRE).sum()) > int((small == PLAN_CENTRE).sum())  # more people, same dial


def test_evaluate_plan_reports_per_person_provision():
    # The per-person readouts are NEW-ONLY: new population (existing fabric counts zero), new
    # centres, and new green (existing green excluded when a mask is given).
    g = 40
    plan = np.zeros((g, g), np.uint8)
    plan[10:30, 10:30] = PLAN_BUILT
    plan[19:21, 19:21] = PLAN_CENTRE
    plan[10:30, 32:38] = PLAN_GREEN
    m = evaluate_plan(plan, 100.0, 800.0, min_green_span_m=200.0, new_density_km2=4000.0)
    assert m["population"] == pytest.approx(400 * 4000.0 * 0.01)  # 400 cells x 1 ha at 4000/km²
    assert m["centre_m2_per_person"] == pytest.approx(4 * 100.0 * 100.0 / m["population"])
    assert m["green_m2_per_person"] > 0.0
    # existing fabric carries NO population: marking half the fabric existing halves the count
    marked = plan.copy()
    marked[10:30, 10:20] = PLAN_EXIST_BUILT
    m2 = evaluate_plan(marked, 100.0, 800.0, min_green_span_m=200.0, new_density_km2=4000.0)
    assert m2["population"] == pytest.approx(0.5 * m["population"])
    # existing centres do not count as provided amenity (only PLAN_CENTRE does)
    marked2 = plan.copy()
    marked2[19:21, 19:21] = PLAN_EXIST_CENTRE
    m3 = evaluate_plan(marked2, 100.0, 800.0, min_green_span_m=200.0, new_density_km2=4000.0)
    assert m3["centre_m2_per_person"] == 0.0
    # pre-existing green is excluded from the green provision when the mask is supplied
    exist_green = np.zeros((g, g), bool)
    exist_green[10:30, 32:38] = True
    m4 = evaluate_plan(plan, 100.0, 800.0, min_green_span_m=200.0,
                       new_density_km2=4000.0, existing_green=exist_green)
    assert m4["green_m2_per_person"] == 0.0
    # coverage metrics are unaffected by the tagging (basis-independent)
    assert m4["green_coverage"] == m["green_coverage"]


def test_optimise_plan_min_settlement_culls_satellite():
    # A small detached satellite keeps its centre at a low minimum, loses it at a high one.
    from isobenefit_qgis.grid import PLAN_BUILT, PLAN_CENTRE, optimise_plan

    g = 70
    plan = np.zeros((g, g), np.uint8)
    plan[20:60, 20:60] = PLAN_BUILT  # main town
    plan[3:7, 3:7] = PLAN_BUILT  # 4x4 satellite, detached
    common = dict(ca_centres=[(40, 40), (5, 5)], optimise_centres=True)
    kept = optimise_plan(plan, 50.0, 400.0, 800.0, centre_min_settlement=3, **common)
    culled = optimise_plan(plan, 50.0, 400.0, 800.0, centre_min_settlement=40, **common)
    sat = lambda out: any(y < 10 and x < 10 for y, x in np.argwhere(out == PLAN_CENTRE))  # noqa: E731
    assert sat(kept)  # satellite (16 cells) keeps a centre when the minimum is small
    assert not sat(culled)  # ...and loses it when the minimum (40) exceeds its catchment


def test_evaluate_plan_split_centre_green_walks():
    # A home just beyond a short centre walk is "unserved" by centre but served once the walk is long.
    from isobenefit_qgis.grid import PLAN_BUILT, PLAN_CENTRE, evaluate_plan

    g = 40
    plan = np.zeros((g, g), np.uint8)
    plan[20, 5:35] = PLAN_BUILT  # a row of homes
    plan[20, 5] = PLAN_CENTRE  # one centre at the left end; far homes are ~1500 m away
    plan[18, 5:35] = PLAN_GREEN  # green alongside so green coverage isn't the limiter
    short = evaluate_plan(plan, 100.0, 2000.0, centre_distance_m=400.0)
    long = evaluate_plan(plan, 100.0, 2000.0, centre_distance_m=1600.0)
    assert long["centre_coverage"] > short["centre_coverage"]  # a longer centre walk reaches more homes


def test_optimise_plan_prunes_centreless_island():
    # A stranded built speck (no centre, below the minimum settlement) reverts to green (the land
    # returns to nature); the real development is kept. The cleanup is off-switchable via prune_islands.
    from isobenefit_qgis.grid import PLAN_BUILT, PLAN_GREEN, optimise_plan

    g = 60
    plan = np.zeros((g, g), np.uint8)
    plan[20:50, 10:40] = PLAN_BUILT  # main development (kept, gets a centre)
    plan[3:6, 53:56] = PLAN_BUILT  # 3x3 stranded speck (9 cells), far from anything, no centre
    common = dict(ca_centres=[(35, 25)], optimise_centres=True, centre_min_settlement=12)
    pruned = optimise_plan(plan, 50.0, 400.0, 800.0, prune_islands=True, **common)
    kept = optimise_plan(plan, 50.0, 400.0, 800.0, prune_islands=False, **common)
    assert (pruned[3:6, 53:56] == PLAN_GREEN).all()  # stranded speck reverted to green
    assert not (kept[3:6, 53:56] == PLAN_GREEN).any()  # ...only when cleanup is on (off: speck kept/developed)
    assert (pruned[20:50, 10:40] == PLAN_BUILT).any()  # the real development is untouched


def test_plan_variants_compactness_options():
    # plan_variants post-processes ONE run at several centre spacings so the user can compare/pick;
    # a tighter spacing yields more, closer centre areas than the consolidated default.
    from isobenefit_qgis.grid import PLAN_CENTRE, _components, plan_variants

    g = 80
    state = np.zeros((g, g), np.int16)
    state[10:70, 10:70] = 1  # a big built block (larger than one catchment)
    state[40, 40] = 2  # one CA-grown centre
    out = plan_variants(state, 50.0, 400.0, 800.0, {"consolidated": None, "dispersed": 300.0})
    assert set(out) == {"consolidated", "dispersed"}
    cons_plan, cons_m = out["consolidated"]
    disp_plan, _disp_m = out["dispersed"]
    assert "served_coverage" in cons_m  # each option carries its own metrics
    n_cons = len(_components(cons_plan == PLAN_CENTRE))
    n_disp = len(_components(disp_plan == PLAN_CENTRE))
    assert n_disp > n_cons  # dispersed places more, closer centres


def test_evaluate_plan_marked_unmarked_equivalence():
    # evaluate_plan must score a plan identically whether existing development is tagged with the
    # EXIST_* codes or folded into the plain codes — sim_runner compares a marked pre_plan against
    # unmarked variants, so any basis-dependence corrupts the raw-vs-processed accounting.
    rng = np.random.default_rng(42)
    g = 30
    marked = np.zeros((g, g), np.uint8)
    marked[2:12, 2:12] = PLAN_BUILT
    marked[6, 6] = PLAN_CENTRE
    marked[15:28, 4:20] = PLAN_EXIST_BUILT
    marked[20, 10] = PLAN_EXIST_CENTRE
    marked[2:8, 20:28] = PLAN_GREEN
    for y, x in rng.integers(0, g, size=(30, 2)):  # scattered existing specks
        if marked[y, x] == PLAN_NONE:
            marked[y, x] = PLAN_EXIST_BUILT
    unmarked = marked.copy()
    unmarked[unmarked == PLAN_EXIST_BUILT] = PLAN_BUILT
    unmarked[unmarked == PLAN_EXIST_CENTRE] = PLAN_CENTRE
    m1 = evaluate_plan(marked, 100.0, 400.0, min_green_span_m=200.0)
    m2 = evaluate_plan(unmarked, 100.0, 400.0, min_green_span_m=200.0)
    for key in ("built_cells", "served_coverage", "centre_coverage", "green_coverage",
                "unserved_fraction", "access_cost", "compactness",
                "centre_efficiency", "green_efficiency"):
        assert m1[key] == m2[key], f"{key}: marked={m1[key]} unmarked={m2[key]}"


def test_select_plan_cleanup_accounting_non_negative():
    # The "cleaned up N cells" figure is pre_plan built minus optimised built; the marked pre_plan
    # includes existing development, so the two evaluations must share a basis or the count goes
    # negative (the pre-fix symptom).
    g = 30
    state = np.zeros((g, g), np.int16)
    state[2:12, 2:12] = 1
    state[6, 6] = 2
    state[25, 25] = 1  # lone speck -> pruned by the cleanup
    exist = np.zeros((g, g), bool)
    exist[15:20, 4:9] = True
    state[15:20, 4:9] = 1
    _best, best_m, pre_plan, _st = select_plan([state], 100.0, 200.0, 400.0, existing_built=exist)
    pre_m = evaluate_plan(pre_plan, 100.0, 400.0, min_green_span_m=200.0)
    removed = pre_m["built_cells"] - best_m["built_cells"]
    assert removed >= 0
    assert removed >= 1  # the speck really was cleaned up


def test_plan_variants_router_bound_must_cover_spacing():
    # Under a street router, centre clustering only works if the router's bound reaches the centre
    # SPACING (2.5x walk for "tight"), not just the walk — sim_runner passes
    # max(max_distance, max(spacings)). A walk-bounded router clips every spacing decision and the
    # options collapse toward walk-spacing (tight was observed placing MORE centres than moderate).
    from isobenefit_qgis.grid import _components, plan_variants
    from isobenefit_qgis.routing import NetworkRouter

    granularity, walk = 100.0, 400.0
    rows, cols = 3, 80  # an 8 km built strip along a chain-graph street
    nodes = np.array([(c * granularity + granularity / 2, 0.0) for c in range(cols)])
    adj = [[] for _ in range(cols)]
    for c in range(cols - 1):
        adj[c].append((c + 1, granularity))
        adj[c + 1].append((c, granularity))
    cell_node = np.tile(np.arange(cols), (rows, 1))
    cell_access = np.zeros((rows, cols))
    state = np.ones((rows, cols), np.int16)
    state[1, 5] = state[1, 40] = state[1, 75] = 2  # a few CA-grown centres
    spacings = {"moderate": 1.5 * walk, "tight": 2.5 * walk}

    def centre_counts(bound):
        router = NetworkRouter(nodes, adj, cell_node, cell_access, granularity, bound)
        out = plan_variants(state, granularity, 200.0, walk, spacings, router=router)
        return {
            label: len(_components(np.isin(plan, (PLAN_CENTRE, PLAN_EXIST_CENTRE))))
            for label, (plan, _m) in out.items()
        }

    good = centre_counts(max(walk, max(spacings.values())))  # what sim_runner passes
    assert good["tight"] < good["moderate"]  # clustering harder -> strictly fewer centres
    clipped = centre_counts(walk)  # the pre-fix bound: spacing decisions clipped at the walk
    assert not (clipped["tight"] < clipped["moderate"])  # collapse fingerprint the fix removes


def test_derive_density_arranges_tiers_by_distance():
    # Every new cell was built at one of three tiers (the mix set by the probabilities); the density
    # layer ARRANGES those drawn values so the highest sit nearest the final mixed-use centre, then
    # medium, then low. Tier counts follow the probabilities, so the population equals the
    # probability-weighted mean. Existing fabric is not counted (0).
    from isobenefit_qgis.grid import derive_density

    g = 40
    plan = np.zeros((g, g), np.uint8)
    plan[10, 5:35] = PLAN_BUILT  # a 3 km row of homes at 100 m cells
    plan[10, 5] = PLAN_CENTRE  # one centre at the left end
    plan[30, 5:15] = PLAN_EXIST_BUILT
    tiers = (6000.0, 3000.0, 1500.0)  # high, med, low
    probs = (0.2, 0.3, 0.5)
    dens = derive_density(plan, 100.0, 4000.0, tiers, probs)  # walk long enough to cover the whole row
    vals = dens[10, 5:35]
    n = vals.size  # 30 new cells
    assert set(np.unique(vals).tolist()) <= set(tiers)  # only the three tier values appear
    assert (vals == 6000.0).sum() == round(0.2 * n)  # counts follow the probabilities
    assert (vals == 3000.0).sum() == round(0.3 * n)
    assert np.all(np.diff(vals) <= 0)  # nearer the centre (left) = the higher tier: non-increasing
    assert vals.mean() == pytest.approx(sum(p * d for p, d in zip(probs, tiers)), rel=1e-6)  # population held
    assert (dens[30, 5:15] == 0.0).all()  # existing fabric is not counted
    assert dens[0, 0] == 0.0  # non-built land carries no density


def test_to_tiered_plan_maps_new_cells_to_tier_codes():
    # to_tiered_plan recolours new built/centre cells by their arranged density, leaving existing and
    # green untouched, so the categorical raster shows low/medium/high in distinct shades.
    from isobenefit_qgis.grid import (
        PLAN_BUILT_HIGH,
        PLAN_BUILT_LOW,
        PLAN_CENTRE_HIGH,
        PLAN_EXIST_BUILT,
        to_tiered_plan,
    )

    tiers = (6000.0, 3000.0, 1500.0)
    plan = np.array([[PLAN_BUILT, PLAN_BUILT, PLAN_CENTRE, PLAN_EXIST_BUILT]], np.uint8)
    dens = np.array([[1500.0, 6000.0, 6000.0, 0.0]], np.float32)
    out = to_tiered_plan(plan, dens, tiers)
    assert out[0, 0] == PLAN_BUILT_LOW
    assert out[0, 1] == PLAN_BUILT_HIGH
    assert out[0, 2] == PLAN_CENTRE_HIGH
    assert out[0, 3] == PLAN_EXIST_BUILT  # existing untouched


def test_derive_density_without_centres_still_conserves_population():
    # A plan with no centres has no distance basis; every tier still lands somewhere and the
    # population still equals the probability-weighted mean over the new cells.
    from isobenefit_qgis.grid import derive_density

    plan = np.zeros((10, 10), np.uint8)
    plan[2:8, 2:8] = PLAN_BUILT  # 36 new cells, no centre anywhere
    tiers, probs = (6000.0, 3000.0, 1500.0), (0.25, 0.25, 0.5)
    dens = derive_density(plan, 100.0, 400.0, tiers, probs)
    vals = dens[plan == PLAN_BUILT]
    assert set(np.unique(vals).tolist()) <= set(tiers)
    assert vals.mean() == pytest.approx(sum(p * d for p, d in zip(probs, tiers)), rel=1e-6)
    assert (dens[plan == 0] == 0.0).all()


def test_density_share_validator():
    # The dialog's guard logic, extracted pure so it is testable without Qt.
    from isobenefit_qgis.validation import check_density_tiers

    ok = check_density_tiers("6000", "3000", "1500", "0.2", "0.3", "0.5")
    assert ok.ok and ok.total == 1.0 and ok.mean == pytest.approx(2850.0)
    assert not check_density_tiers("6000", "3000", "1500", "0.5", "0.4", "0.2").ok  # sums to 1.1
    assert not check_density_tiers("1500", "3000", "6000", "0.2", "0.3", "0.5").ok  # not descending
    assert not check_density_tiers("6000", "3000", "1500", "1.2", "-0.4", "0.2").ok  # shares outside [0,1]
    assert not check_density_tiers("6000", "x", "1500", "0.2", "0.3", "0.5").ok  # unparseable
    ok2 = check_density_tiers("6000", "3000", "1500", "0.2", "0.3", "0.5005")  # within the 1e-3 tolerance
    assert ok2.ok


def test_run_ensemble_member_offset_reproducible_across_batch_sizes(grid):
    # One logical ensemble split into batches (as the runner does per core count) must equal a
    # single call: the same random_seed then reproduces the same ensemble on any machine.
    whole = isobenefit.run_ensemble(_make(grid, total_iters=8), 7, 5)
    sim2 = _make(grid, total_iters=8)
    batched = list(isobenefit.run_ensemble(sim2, 7, 2, member_offset=0))
    batched += list(isobenefit.run_ensemble(sim2, 7, 3, member_offset=2))
    assert len(whole) == len(batched) == 5
    assert all(np.array_equal(a, b) for a, b in zip(whole, batched))
