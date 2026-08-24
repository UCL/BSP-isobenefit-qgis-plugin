#!/usr/bin/env python3
"""Four-outputs run report for the Cambourne scenario.

Reproduces the plugin's run report headlessly on the committed Cambourne
scenario at full resolution and its baseline settings: a fifty-run
ensemble, the standard selection pipeline, the raw grown state recovered by
re-running the chosen member at its own seed, and the three centre options.
The manuscript's four-outputs table (``tab:fouroutputs``) reproduces this
script's output.

    .venv/bin/python scripts/demo_run_report.py
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import isobenefit  # noqa: E402

from isobenefit_qgis import grid as G  # noqa: E402
from isobenefit_qgis import report as R  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "render_scenario_gallery", os.path.join(REPO, "scripts", "render_scenario_gallery.py")
)
_gallery = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gallery)

RUNS = 50


def _count_centres(plan) -> int:
    return len(G._components(np.isin(plan, (G.PLAN_CENTRE, G.PLAN_EXIST_CENTRE))))


def main():
    folder = os.path.join(REPO, "scenarios", "cambourne")
    params, layers, extents = _gallery.load_scenario(folder)
    gran = float(params["grid_size_m"])
    sub = _gallery.substrate(extents["main"], layers, gran)
    tiers = (
        params["densities_km2"]["high"], params["densities_km2"]["medium"],
        params["densities_km2"]["low"],
    )
    shares = (params["shares"]["high"], params["shares"]["medium"], params["shares"]["low"])
    mean_density = sum(s * d for s, d in zip(shares, tiers))
    walk = float(params.get("centre_walk_m", 800.0))
    green_walk = float(params.get("green_walk_m", 400.0))
    park_m2 = float(params["min_park_area_ha"]) * 1.0e4 if params.get("min_park_area_ha") else None
    target = float(params["target_population"])
    seed = int(params.get("random_seed", 42))
    cell_km2 = gran * gran / 1e6
    min_cells = max(1, round(float(params.get("min_settlement_pop", 2000.0)) / (mean_density * cell_km2)))
    state, origin = sub["state"].copy(), sub["origin"].copy()
    G.green_unviable_pockets(
        state, origin, min_cells,
        existing_centres=sub["seeds"], granularity_m=gran, centre_distance_m=walk,
    )

    template = isobenefit.Simulation(
        state, origin, np.zeros_like(state, np.float32), sub["seeds"],
        gran, walk, green_walk, target,
        float(params.get("min_green_span_m", 400.0)), _gallery.BUILD_PROB,
        _gallery.centre_quota(params), _gallery.allow_detached(params),
        shares, tiers, int(params.get("max_iterations", 300)), seed,
        min_park_area_m2=park_m2,
        sterile=G.sterile_fabric(origin == 1, sub["seeds"]),
    )
    states = [np.asarray(s) for s in isobenefit.run_ensemble(template, seed, RUNS)]

    _plan, _metrics, pre_plan, best_state = G.select_plan(
        states, gran, max(walk, green_walk),
        existing_built=(origin == 1), existing_green=(origin == 0),
        existing_centres=sub["seeds"], centre_mode="placed",
        centre_distance_m=walk, green_distance_m=green_walk,
        new_density_km2=mean_density, centre_min_settlement=min_cells,
        target_population=target, min_park_area_m2=park_m2,
    )

    options = []
    best_idx = next(i for i, s in enumerate(states) if np.array_equal(s, best_state))
    member = isobenefit.run_member(template, seed, best_idx)
    pre_m = G.evaluate_plan(
        pre_plan, gran, max(walk, green_walk),
        centre_distance_m=walk, green_distance_m=green_walk,
        new_density_km2=mean_density, existing_green=(origin == 0),
        min_park_area_m2=park_m2,
    )
    options.append({"short": "raw", "metrics": pre_m, "n_centres": _count_centres(pre_plan)})

    variants = G.plan_variants(
        best_state, gran, max(walk, green_walk),
        {key: key for key in ("grown", "placed", "minimal")},
        existing_centres=sub["seeds"], existing_built=(origin == 1),
        existing_green=(origin == 0),
        centre_distance_m=walk, green_distance_m=green_walk,
        centre_min_settlement=min_cells, new_density_km2=mean_density,
        min_park_area_m2=park_m2,
    )
    for key, short in (("grown", "grown"), ("placed", "placed"), ("minimal", "fewest")):
        vplan, vm = variants[key]
        options.append({"short": short, "metrics": vm, "n_centres": _count_centres(vplan)})

    print(f"member {best_idx} of {RUNS}, seed {seed}, target {target:,.0f}")
    print(f"drawn density recovered: {np.asarray(member['density']).sum():,.0f} people (raw)")
    for line in R.options_table(options, target):
        print(line)
    print("DEMO-RUN-REPORT-DONE")


if __name__ == "__main__":
    main()
