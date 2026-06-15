"""Headless end-to-end test + benchmark on the real Cambourne demo data.

Rasterises the demo_layers/*.geojson (all EPSG:27700) onto the simulation grid
without QGIS, runs a single scenario and an ensemble, and prints timings so we can
see the parallel speedup. The plugin's gis_io does the same job via GDAL/QGIS; here
we use shapely so it runs in a plain venv.

Run:
    uv run --no-project \
        --with core/dist/isobenefit-*.whl --with numpy --with shapely \
        python scripts/demo_bench.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import shapely

import isobenefit

DEMO = Path(__file__).resolve().parent.parent / "demo_layers"
GRAN = 100.0
TOTAL_ITERS = 100
MAX_POPULAT = 10_000_000.0  # high so the run does not stop early on existing pop


def _geoms(path: Path):
    data = json.loads(path.read_text())
    return shapely.unary_union([shapely.geometry.shape(f["geometry"]) for f in data["features"]])


def _bounds(geom) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = geom.bounds
    x_min = np.floor(minx / GRAN) * GRAN
    y_min = np.floor(miny / GRAN) * GRAN
    x_max = np.ceil(maxx / GRAN) * GRAN
    y_max = np.ceil(maxy / GRAN) * GRAN
    return x_min, y_min, x_max, y_max


def main() -> None:
    extents = _geoms(DEMO / "extents.geojson")
    x_min, y_min, x_max, y_max = _bounds(extents)
    cols = int(round((x_max - x_min) / GRAN))
    rows = int(round((y_max - y_min) / GRAN))
    # cell-centre coordinates
    xs = x_min + (np.arange(cols) + 0.5) * GRAN
    ys = y_max - (np.arange(rows) + 0.5) * GRAN
    gx, gy = np.meshgrid(xs, ys)

    def burn(arr, path, value):
        mask = shapely.contains_xy(_geoms(path), gx, gy)
        arr[mask] = value
        return arr

    state = np.full((rows, cols), -1, dtype=np.int16)
    burn(state, DEMO / "extents.geojson", 0)
    origin = np.full((rows, cols), -1, dtype=np.int16)
    for layer, val in [("urban", 1), ("green", 0)]:
        burn(state, DEMO / f"{layer}.geojson", val)
        burn(origin, DEMO / f"{layer}.geojson", val)
    burn(state, DEMO / "unbuildable.geojson", -1)
    density = np.zeros((rows, cols), dtype=np.float32)

    seeds = []
    cdata = json.loads((DEMO / "centres.geojson").read_text())
    for f in cdata["features"]:
        pt = shapely.geometry.shape(f["geometry"])
        c = int((pt.x - x_min) / GRAN)
        r = int((y_max - pt.y) / GRAN)
        if 0 <= r < rows and 0 <= c < cols:
            seeds.append((r, c))

    def make():
        return isobenefit.Simulation(
            state, origin, density, seeds,
            GRAN, 800.0, MAX_POPULAT, 100.0,
            0.25, 0.05, 0.0, 0.8,
            (0.4, 0.4, 0.2), (6000.0, 3000.0, 1000.0), 2000.0,
            TOTAL_ITERS, 42,
        )

    cores = os.cpu_count() or 1
    print(f"grid {cols}x{rows} ({cols * rows} cells); {len(seeds)} centre seeds; {cores} cores")
    sim0 = make()
    print(f"starting population {int(sim0.population)} ({sim0.pop_target_ratio:.0%} of target)")

    t = time.time()
    sim0.run()
    single = time.time() - t
    print(f"single run: {single:.2f}s, {sim0.current_iter} iters, final pop {int(sim0.population)}")

    n = cores * 2
    template = make()
    t = time.time()
    prob = isobenefit.ensemble_probability(template, 2024, n)
    ens = time.time() - t
    print(
        f"ensemble of {n}: {ens:.2f}s wall  "
        f"(serial-equivalent ~{single * n:.1f}s -> {single * n / ens:.1f}x speedup)"
    )
    print(f"probability grid: min {prob.min():.2f}, max {prob.max():.2f}, mean {prob.mean():.3f}")


if __name__ == "__main__":
    main()
