#!/usr/bin/env python3
"""Four-outputs run report for the Cambourne demonstration window.

Reproduces the plugin's run report headlessly on the website's committed
Cambourne window (4.2 km at 50 m, 12,000-person target): a fifty-run
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
    "demonstrators", os.path.join(REPO, "website", "scripts", "demonstrators.py")
)
_demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_demo)

SEED = 42
RUNS = 50
TARGET = 12_000.0
BUILD_PROB = 0.25  # the plugin's default


def _count_centres(plan) -> int:
    return len(G._components(np.isin(plan, (G.PLAN_CENTRE, G.PLAN_EXIST_CENTRE))))


def main():
    sub = _demo.substrate()
    template = isobenefit.Simulation(
        sub["state"].copy(), sub["origin"].copy(), sub["density"].copy(), sub["seeds"],
        _demo.GRAN, max(_demo.WALK, _demo.GREEN_WALK), TARGET, _demo.GREEN_SPAN,
        BUILD_PROB, 0.01, 0.0001, 0.8, _demo.TIER_PROBS, _demo.DENSITY_TIERS, 400, SEED,
    )
    states = [np.asarray(s) for s in isobenefit.run_ensemble(template, SEED, RUNS)]
    mean_density = _demo._mean_density()
    min_cells = _demo._min_settlement()

    _plan, _metrics, pre_plan, best_state = G.select_plan(
        states, _demo.GRAN, _demo.GREEN_SPAN, max(_demo.WALK, _demo.GREEN_WALK),
        existing_built=(sub["origin"] == 1), existing_green=(sub["origin"] == 0),
        existing_centres=sub["seeds"], centre_mode="placed",
        centre_distance_m=_demo.WALK, green_distance_m=_demo.GREEN_WALK,
        new_density_km2=mean_density, centre_min_settlement=min_cells,
    )

    options = []
    best_idx = next(i for i, s in enumerate(states) if np.array_equal(s, best_state))
    member = isobenefit.run_member(template, SEED, best_idx)
    pre_m = G.evaluate_plan(
        pre_plan, _demo.GRAN, max(_demo.WALK, _demo.GREEN_WALK),
        min_green_span_m=_demo.GREEN_SPAN,
        centre_distance_m=_demo.WALK, green_distance_m=_demo.GREEN_WALK,
        new_density_km2=mean_density, existing_green=(sub["origin"] == 0),
    )
    options.append({"short": "raw", "metrics": pre_m, "n_centres": _count_centres(pre_plan)})

    variants = G.plan_variants(
        best_state, _demo.GRAN, _demo.GREEN_SPAN, max(_demo.WALK, _demo.GREEN_WALK),
        {key: key for key in ("grown", "placed", "minimal")},
        existing_centres=sub["seeds"], existing_built=(sub["origin"] == 1),
        existing_green=(sub["origin"] == 0),
        centre_distance_m=_demo.WALK, green_distance_m=_demo.GREEN_WALK,
        centre_min_settlement=min_cells, new_density_km2=mean_density,
    )
    for key, short in (("grown", "grown"), ("placed", "placed"), ("minimal", "fewest")):
        vplan, vm = variants[key]
        options.append({"short": short, "metrics": vm, "n_centres": _count_centres(vplan)})

    print(f"member {best_idx} of {RUNS}, seed {SEED}, target {TARGET:,.0f}")
    print(f"drawn density recovered: {np.asarray(member['density']).sum():,.0f} people (raw)")
    for line in R.options_table(options, TARGET):
        print(line)
    print("DEMO-RUN-REPORT-DONE")


if __name__ == "__main__":
    main()
