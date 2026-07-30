"""Run-report composition: fixed-width tables for the ``<name>_report.txt`` sidecar.

Pure logic (no QGIS imports), like ``grid.py``, so the report content is
unit-testable in a plain venv. ``sim_runner`` gathers the numbers and calls
``compose_report`` (ensemble) or ``compose_single_run_report``; both return the
full report text. The report lives in the run's output folder next to the
rasters and is the durable, comprehensive record of what the run produced.
"""

from __future__ import annotations

import numpy as np

from .grid import (
    PLAN_BUILT_HIGH,
    PLAN_BUILT_LOW,
    PLAN_BUILT_MED,
    PLAN_CENTRE_HIGH,
    PLAN_CENTRE_LOW,
    PLAN_CENTRE_MED,
)

TIER_LABELS = ("high", "medium", "low")
_TIER_CODES = {
    "high": (PLAN_BUILT_HIGH, PLAN_CENTRE_HIGH),
    "medium": (PLAN_BUILT_MED, PLAN_CENTRE_MED),
    "low": (PLAN_BUILT_LOW, PLAN_CENTRE_LOW),
}


def fixed_table(headers: list[str], rows: list[list[str]], indent: str = "  ") -> list[str]:
    """Render a fixed-width text table: first column left-aligned, the rest right-aligned,
    with a rule under the header. Every cell is taken as already-formatted text."""
    cells = [headers] + [[str(c) for c in row] for row in rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(headers))]

    def fmt(row: list[str]) -> str:
        first = row[0].ljust(widths[0])
        rest = [c.rjust(w) for c, w in zip(row[1:], widths[1:])]
        return (indent + "  ".join([first] + rest)).rstrip()

    rule = indent + "  ".join("-" * w for w in widths)
    return [fmt(headers), rule] + [fmt(r) for r in rows]


def tier_breakdown(tiered_plan: np.ndarray, granularity_m: float, density_factors_km2) -> dict:
    """Achieved density mix of a tiered plan: per tier (high/medium/low), the NEW built and
    mixed-use-centre cell counts and the population housed at that tier's density. Existing
    fabric carries no density and is excluded by construction (it has its own codes)."""
    cell_km2 = granularity_m * granularity_m / 1.0e6
    out = {}
    for label, density in zip(TIER_LABELS, density_factors_km2):
        built_code, centre_code = _TIER_CODES[label]
        n_built = int((tiered_plan == built_code).sum())
        n_centre = int((tiered_plan == centre_code).sum())
        out[label] = {
            "built_cells": n_built,
            "centre_cells": n_centre,
            "population": (n_built + n_centre) * float(density) * cell_km2,
        }
    return out


def _pct(v) -> str:
    return f"{v:.0%}" if v is not None else "-"


def _num(v, decimals: int = 0) -> str:
    return f"{v:,.{decimals}f}" if v is not None else "-"


def options_table(options: list[dict], target_population: float) -> list[str]:
    """The plan options side by side: one metric per row, one column per option. Each option is
    ``{"short": column header, "metrics": evaluate_plan dict, "n_centres": int}``. Metrics a plan
    lacks render as ``-`` so the rows are stable across configurations."""
    rows_spec = [
        ("population accommodated", lambda m, n: _num(m.get("population"))),
        ("share of target", lambda m, n: _pct(_share(m.get("population"), target_population))),
        ("built cells (incl. existing)", lambda m, n: _num(m.get("built_cells"))),
        ("mixed-use centre areas", lambda m, n: _num(n)),
        ("served coverage (centre AND green)", lambda m, n: _pct(m.get("served_coverage"))),
        ("centre coverage", lambda m, n: _pct(m.get("centre_coverage"))),
        ("green coverage", lambda m, n: _pct(m.get("green_coverage"))),
        ("unserved homes", lambda m, n: _pct(m.get("unserved_fraction"))),
        ("avg walk to a centre (m)", lambda m, n: _num(m.get("centre_access"))),
        ("avg walk to green (m)", lambda m, n: _num(m.get("green_access"))),
        ("mean walk, served homes: centre (m)", lambda m, n: _num(_finite(m.get("centre_walk_mean")))),
        ("mean walk, served homes: green (m)", lambda m, n: _num(_finite(m.get("green_walk_mean")))),
        ("m2 mixed-use centre / person", lambda m, n: _num(m.get("centre_m2_per_person"))),
        ("m2 walkable green / person", lambda m, n: _num(m.get("green_m2_per_person"))),
        ("homes served per centre area", lambda m, n: _num(m.get("centre_efficiency"))),
        ("compactness", lambda m, n: _pct(m.get("compactness"))),
    ]
    if any("transit_coverage" in o["metrics"] for o in options):
        rows_spec += [
            ("transit coverage (reported only)", lambda m, n: _pct(m.get("transit_coverage"))),
            ("avg walk to a stop (m)", lambda m, n: _num(m.get("transit_access"))),
        ]
    headers = ["Metric"] + [o["short"] for o in options]
    rows = [[label] + [cell(o["metrics"], o["n_centres"]) for o in options] for label, cell in rows_spec]
    return fixed_table(headers, rows)


def _finite(v):
    return None if v is None or not np.isfinite(v) else v


def _share(population, target):
    return None if population is None or not target else population / target


def tiers_table(options: list[dict], density_factors_km2, shares) -> list[str]:
    """Achieved density mix per option: for each tier, the drawn share target and, per option,
    the new cells (built + centre) and people housed at that tier. Options without a tiered
    plan (``tiers`` is None) are skipped; returns [] if none carry one."""
    with_tiers = [o for o in options if o.get("tiers")]
    if not with_tiers:
        return []
    headers = ["Tier (people/km2)", "share drawn"]
    for o in with_tiers:
        headers += [f"{o['short']}: cells", f"{o['short']}: people"]
    rows = []
    for label, density, share in zip(TIER_LABELS, density_factors_km2, shares):
        row = [f"{label} ({density:,.0f})", f"{share:.0%}"]
        for o in with_tiers:
            t = o["tiers"][label]
            row += [_num(t["built_cells"] + t["centre_cells"]), _num(t["population"])]
        rows.append(row)
    total = ["total", "100%"]
    for o in with_tiers:
        cells = sum(o["tiers"][t]["built_cells"] + o["tiers"][t]["centre_cells"] for t in TIER_LABELS)
        people = sum(o["tiers"][t]["population"] for t in TIER_LABELS)
        total += [_num(cells), _num(people)]
    rows.append(total)
    return fixed_table(headers, rows)


def audit_tables(audit: dict) -> list[str]:
    """The centre audit as text: the summary line plus a table of the weakest NEW centres
    (thin catchment or off-centre), so the report carries what the log surfaces."""
    s = audit["summary"]
    lines = [
        f"  {s['n_centres']} centres ({s['n_existing']} existing, {s['n_new']} placed by the model); "
        f"each serves a median of {s['served_median']} built cells "
        f"(min {s['served_min']}, max {s['served_max']}); median avg-walk {s['mean_dist_median_m']:.0f} m.",
    ]
    weak_new = [c for c in audit["centres"] if not c["existing"]][:5]
    if weak_new:
        lines.append("")
        lines.append("  Weakest new centres (fewest homes served):")
        rows = [
            [f"({c['row']}, {c['col']})", _num(c["cells"]), _num(c["served"]), _num(c["mean_dist_m"])]
            for c in weak_new
        ]
        lines += fixed_table(["Cell (row, col)", "area cells", "homes served", "avg walk (m)"], rows)
    return lines


def compose_report(
    header_lines: list[str],
    param_lines: list[str],
    run_lines: list[str],
    options: list[dict],
    target_population: float,
    density_factors_km2,
    shares,
    audit: dict | None,
    file_lines: list[str],
) -> str:
    """Assemble the full ensemble report from pre-formatted header/parameter/run/file lines and
    the structured per-option data (see ``options_table`` / ``tiers_table``)."""
    lines = list(header_lines)
    lines += ["", "PARAMETERS", "-" * 10] + param_lines
    lines += ["", "RUN", "-" * 3] + run_lines
    if options:
        lines += ["", "PLAN OPTIONS (side by side; the walkability figures count every home)", "-" * 40]
        lines += options_table(options, target_population)
        tiers = tiers_table(options, density_factors_km2, shares)
        if tiers:
            lines += ["", "ACHIEVED DENSITY MIX (new development only)", "-" * 40]
            lines += tiers
    if audit:
        lines += ["", "CENTRE AUDIT", "-" * 12] + audit_tables(audit)
    lines += ["", "FILES", "-" * 5] + file_lines
    lines.append("")
    return "\n".join(lines)


def compose_single_run_report(
    header_lines: list[str],
    param_lines: list[str],
    run_lines: list[str],
    file_lines: list[str],
) -> str:
    """The single-run (animation) report: parameters and outcome. Single-run mode writes no
    plan options, so there are no per-option tables; the temporal raster is the product."""
    lines = list(header_lines)
    lines += ["", "PARAMETERS", "-" * 10] + param_lines
    lines += ["", "RUN", "-" * 3] + run_lines
    lines += ["", "FILES", "-" * 5] + file_lines
    lines.append("")
    return "\n".join(lines)
