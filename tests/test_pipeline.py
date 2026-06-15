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

import numpy as np
import pytest
import shapely

import isobenefit
from isobenefit_qgis.grid import EXIST_BUILT, NATURE, NODATA, align_bounds, classify

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
        grid["state"], grid["origin"], grid["density"], grid["seeds"],
        GRAN, 800.0, 10_000_000.0, 100.0,
        0.25, 0.05, 0.0, 0.8,
        (0.4, 0.4, 0.2), (6000.0, 3000.0, 1000.0), 2000.0,
        total_iters, seed,
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
