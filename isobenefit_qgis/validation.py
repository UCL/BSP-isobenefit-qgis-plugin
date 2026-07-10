"""Pure input-validation helpers, kept free of any QGIS/Qt import so they are testable headlessly.

The dialog delegates its density-tier checks here; the Rust core re-validates the same invariants,
so these functions define the user-facing messages while the core stays the final guard.
"""

from __future__ import annotations

from typing import NamedTuple


class DensityCheck(NamedTuple):
    """Outcome of validating the three density tiers and their shares.

    ``ok`` gates the Run button; ``total`` is the share sum (None until every field parses);
    ``mean`` is the probability-weighted mean density, for the live feedback line.
    """

    ok: bool
    message: str
    total: float | None
    mean: float | None


def check_density_tiers(
    high_density: str,
    med_density: str,
    low_density: str,
    high_share: str,
    med_share: str,
    low_share: str,
    tolerance: float = 1e-3,
) -> DensityCheck:
    """Validate the dialog's six density fields (raw text, as the user typed them).

    Rules: every field must parse as a number; densities must be positive and strictly
    descending (high > med > low, which the engine requires); each share must lie in [0, 1];
    and the three shares must sum to 1 within ``tolerance``.
    """
    try:
        hd, md, ld = float(high_density), float(med_density), float(low_density)
        hp, mp, lp = float(high_share), float(med_share), float(low_share)
    except ValueError:
        return DensityCheck(False, "Enter valid numbers for every density and share.", None, None)
    if not (ld > 0 and hd > md > ld):
        return DensityCheck(False, "Densities must be positive and High > Medium > Low.", None, None)
    if any(not 0.0 <= p <= 1.0 for p in (hp, mp, lp)):
        return DensityCheck(False, "Each share must be between 0 and 1.", None, None)
    total = round(hp + mp + lp, 3)
    if abs(total - 1.0) > tolerance:
        return DensityCheck(False, f"Shares must sum to 1 (currently {total:.2f}).", total, None)
    mean = hp * hd + mp * md + lp * ld
    return DensityCheck(True, f"Shares sum to 1.00 ✓ · mean ≈ {mean:,.0f} /km²", total, mean)
