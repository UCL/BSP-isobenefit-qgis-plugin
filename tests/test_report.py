"""Unit tests for the pure report composition (``isobenefit_qgis.report``).

No QGIS required: the module formats pre-computed numbers into the fixed-width
tables written to ``<name>_report.txt``.
"""

from __future__ import annotations

import numpy as np

from isobenefit_qgis.grid import (
    PLAN_BUILT_HIGH,
    PLAN_BUILT_LOW,
    PLAN_BUILT_MED,
    PLAN_CENTRE_HIGH,
    PLAN_EXIST_BUILT,
    PLAN_GREEN,
)
from isobenefit_qgis.report import (
    compose_report,
    compose_single_run_report,
    fixed_table,
    options_table,
    tier_breakdown,
    tiers_table,
)

DENSITIES = (6000.0, 3000.0, 1500.0)  # high, medium, low (people/km²)
SHARES = (0.2, 0.3, 0.5)


def _metrics(**overrides) -> dict:
    base = {
        "population": 11_500.0,
        "built_cells": 1_600,
        "served_coverage": 0.96,
        "centre_coverage": 0.97,
        "green_coverage": 0.98,
        "unserved_fraction": 0.04,
        "centre_access": 410.0,
        "green_access": 210.0,
        "centre_walk_mean": 395.0,
        "green_walk_mean": 190.0,
        "centre_m2_per_person": 4.0,
        "green_m2_per_person": 55.0,
        "centre_efficiency": 120.0,
        "compactness": 0.71,
    }
    base.update(overrides)
    return base


def _option(short: str, tiers=None, **metric_overrides) -> dict:
    return {
        "label": short,
        "short": short,
        "metrics": _metrics(**metric_overrides),
        "n_centres": 11,
        "tiers": tiers,
    }


def test_fixed_table_aligns_columns():
    lines = fixed_table(["Metric", "a", "bb"], [["x", "1", "22"], ["longer", "333", "4"]])
    assert len(lines) == 4  # header, rule, two rows
    assert all(len(line) == len(lines[0]) or len(line) <= len(lines[0]) for line in lines)
    # right alignment of numeric columns: the shorter number is padded to the wider one
    assert lines[2].endswith("22")
    assert lines[3].split()[-2] == "333"


def test_tier_breakdown_counts_new_cells_only():
    plan = np.full((4, 4), PLAN_GREEN, dtype=np.uint8)
    plan[0, 0] = PLAN_BUILT_HIGH
    plan[0, 1] = PLAN_BUILT_HIGH
    plan[1, 0] = PLAN_CENTRE_HIGH
    plan[2, 0] = PLAN_BUILT_MED
    plan[3, 0] = PLAN_BUILT_LOW
    plan[3, 3] = PLAN_EXIST_BUILT  # existing fabric: no density, excluded
    tiers = tier_breakdown(plan, granularity_m=100.0, density_factors_km2=DENSITIES)
    cell_km2 = 0.01
    assert tiers["high"]["built_cells"] == 2
    assert tiers["high"]["centre_cells"] == 1
    assert tiers["high"]["population"] == 3 * 6000.0 * cell_km2
    assert tiers["medium"] == {"built_cells": 1, "centre_cells": 0, "population": 3000.0 * cell_km2}
    assert tiers["low"]["population"] == 1500.0 * cell_km2
    # totals cover exactly the new development
    total = sum(t["built_cells"] + t["centre_cells"] for t in tiers.values())
    assert total == 5


def test_options_table_side_by_side_and_missing_metrics():
    sparse = {"label": "raw", "short": "raw", "metrics": {"built_cells": 1700}, "n_centres": 14, "tiers": None}
    lines = options_table([sparse, _option("moderate")], target_population=12_000.0)
    text = "\n".join(lines)
    assert "raw" in lines[0] and "moderate" in lines[0]
    assert "population accommodated" in text
    # the sparse option renders '-' where a metric is absent, the full one renders values
    pop_row = next(line for line in lines if line.lstrip().startswith("population accommodated"))
    assert "-" in pop_row and "11,500" in pop_row
    share_row = next(line for line in lines if line.lstrip().startswith("share of target"))
    assert "96%" in share_row  # 11,500 of 12,000
    # transit rows appear only when some option reports them
    assert "transit" not in text
    with_transit = _option("tight", transit_coverage=0.8, transit_access=300.0)
    assert "transit catchment coverage" in "\n".join(options_table([with_transit], 12_000.0))


def test_tiers_table_totals_and_skip_without_tiers():
    plan = np.full((3, 3), PLAN_GREEN, dtype=np.uint8)
    plan[0, 0] = PLAN_BUILT_HIGH
    plan[1, 1] = PLAN_BUILT_LOW
    tiers = tier_breakdown(plan, 100.0, DENSITIES)
    lines = tiers_table([_option("moderate", tiers=tiers), _option("raw", tiers=None)], DENSITIES, SHARES)
    text = "\n".join(lines)
    assert "moderate: cells" in lines[0] and "raw:" not in lines[0]  # tier-less options are skipped
    assert "high (6,000)" in text and "20%" in text
    total_row = next(line for line in lines if line.lstrip().startswith("total"))
    assert "100%" in total_row and "2" in total_row
    assert tiers_table([_option("raw", tiers=None)], DENSITIES, SHARES) == []


def test_compose_report_sections_in_order():
    audit = {
        "summary": {
            "n_centres": 11, "n_existing": 3, "n_new": 8,
            "served_median": 60, "served_min": 4, "served_max": 200, "mean_dist_median_m": 350.0,
        },
        "centres": [
            {"row": 5, "col": 6, "cells": 2, "served": 4, "mean_dist_m": 700.0, "existing": False},
            {"row": 1, "col": 1, "cells": 9, "served": 180, "mean_dist_m": 200.0, "existing": True},
        ],
    }
    text = compose_report(
        header_lines=["Isobenefit Urbanism — simulation report", "=" * 42],
        param_lines=["  Grid size             : 50 m"],
        run_lines=["  Elapsed: 12 s"],
        options=[_option("moderate")],
        target_population=12_000.0,
        density_factors_km2=DENSITIES,
        shares=SHARES,
        audit=audit,
        file_lines=["  out.tif  development likelihood (built, green)"],
    )
    order = [text.index(s) for s in ("PARAMETERS", "RUN", "PLAN OPTIONS", "CENTRE AUDIT", "FILES")]
    assert order == sorted(order)
    assert "Weakest new centres" in text and "(5, 6)" in text
    assert "(1, 1)" not in text.split("Weakest new centres")[1]  # existing centres are not listed as weak


def test_compose_single_run_report_is_minimal():
    text = compose_single_run_report(
        header_lines=["Isobenefit Urbanism — simulation report"],
        param_lines=["  Grid size             : 50 m"],
        run_lines=["  Population accommodated: 9,000 of 12,000 target (75%)"],
        file_lines=["  out.tif  growth animation (temporal raster)"],
    )
    assert "PLAN OPTIONS" not in text and "CENTRE AUDIT" not in text
    assert "Population accommodated" in text and "FILES" in text
