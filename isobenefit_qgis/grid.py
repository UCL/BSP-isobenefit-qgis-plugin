"""Pure, QGIS-free grid logic: class taxonomy, classification, grid maths.

This module imports only numpy so it can be unit-tested in a plain virtualenv
(no QGIS, no GDAL). The QGIS/GDAL-coupled IO lives in ``gis_io.py``, which imports
from here.
"""

from __future__ import annotations

import math

import numpy as np

# Categorical class codes for the output raster.
NODATA = 255
NATURE = 0
NEW_LOW = 1
NEW_MED = 2
NEW_HIGH = 3
CENTRE = 4
EXIST_BUILT = 5
FIXED_GREEN = 6

# (class code, (r, g, b), legend label) — echoes the original NetLogo scheme.
PALETTE = [
    (NATURE, (89, 176, 60), "Nature / green"),
    (NEW_LOW, (200, 136, 68), "New built — low density"),
    (NEW_MED, (197, 86, 17), "New built — medium density"),
    (NEW_HIGH, (101, 44, 7), "New built — high density"),
    (CENTRE, (255, 255, 255), "Centrality"),
    (EXIST_BUILT, (114, 114, 114), "Existing built"),
    (FIXED_GREEN, (54, 109, 35), "Existing green / park"),
]


def align_bounds(x_min: float, y_min: float, x_max: float, y_max: float, granularity_m: float):
    """Snap a bounding box out to whole cells and return the grid geometry.

    Returns ``(rows, cols, geotransform, (x_min, y_min, x_max, y_max))`` where the
    geotransform is the GDAL 6-tuple ``(x_min, g, 0, y_max, 0, -g)`` and the bounds
    are the snapped extents in the same CRS units as the inputs.
    """
    g = float(granularity_m)
    xmn = math.floor(x_min / g) * g
    ymn = math.floor(y_min / g) * g
    xmx = math.ceil(x_max / g) * g
    ymx = math.ceil(y_max / g) * g
    cols = int(round((xmx - xmn) / g))
    rows = int(round((ymx - ymn) / g))
    geotransform = (xmn, g, 0.0, ymx, 0.0, -g)
    return rows, cols, geotransform, (xmn, ymn, xmx, ymx)


def classify(state, origin, density, per_block) -> np.ndarray:
    """Map the simulation arrays to a uint8 categorical raster (see class codes).

    ``per_block`` is ``(high, med, low)`` persons-per-block; new-built cells carry
    one of these exact values so density tiers can be matched directly. Existing
    (origin) features take visual precedence.
    """
    high_pb, med_pb, low_pb = per_block
    cls = np.full(state.shape, NODATA, dtype=np.uint8)
    cls[state == 0] = NATURE
    built = state == 1
    cls[built & np.isclose(density, low_pb)] = NEW_LOW
    cls[built & np.isclose(density, med_pb)] = NEW_MED
    cls[built & np.isclose(density, high_pb)] = NEW_HIGH
    cls[state == 2] = CENTRE
    cls[origin == 1] = EXIST_BUILT
    cls[origin == 0] = FIXED_GREEN
    return cls


# --- constraint-aware "recommended plan" derived from the probability surfaces ---

PLAN_NONE = 0
PLAN_GREEN = 1
PLAN_BUILT = 2
PLAN_CENTRE = 3

PLAN_PALETTE = [
    (PLAN_GREEN, (54, 109, 35), "Recommended green network"),
    (PLAN_BUILT, (150, 110, 90), "Recommended development"),
    (PLAN_CENTRE, (200, 30, 30), "Recommended centre"),
]


def _keep_large_components(mask: np.ndarray, min_cells: int) -> np.ndarray:
    """Zero out rook-connected components of ``mask`` smaller than ``min_cells``."""
    rows, cols = mask.shape
    out = np.zeros_like(mask)
    seen = np.zeros_like(mask)
    for sy in range(rows):
        for sx in range(cols):
            if not mask[sy, sx] or seen[sy, sx]:
                continue
            stack = [(sy, sx)]
            seen[sy, sx] = True
            comp = []
            while stack:
                y, x = stack.pop()
                comp.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < rows and 0 <= nx < cols and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            if len(comp) >= min_cells:
                for y, x in comp:
                    out[y, x] = True
    return out


def _box_sum(a: np.ndarray, r: int) -> np.ndarray:
    """Sum of ``a`` over each (2r+1)-square window (edge-clamped) via an integral image."""
    rows, cols = a.shape
    ii = np.zeros((rows + 1, cols + 1), dtype=np.float64)
    ii[1:, 1:] = a.astype(np.float64).cumsum(0).cumsum(1)
    y0 = np.clip(np.arange(rows) - r, 0, rows)
    y1 = np.clip(np.arange(rows) + r + 1, 0, rows)
    x0 = np.clip(np.arange(cols) - r, 0, cols)
    x1 = np.clip(np.arange(cols) + r + 1, 0, cols)
    return ii[np.ix_(y1, x1)] - ii[np.ix_(y0, x1)] - ii[np.ix_(y1, x0)] + ii[np.ix_(y0, x0)]


def _centre_peaks(p_centre, radius, threshold_frac, max_centres):
    """Greedy non-max suppression on the walk-smoothed centre surface.

    Returns up to ``max_centres`` (row, col) peaks, each at least ``threshold_frac``
    of the strongest and spaced at least ``radius`` cells apart.
    """
    suit = _box_sum(p_centre, radius)
    peak_floor = threshold_frac * float(suit.max()) if suit.size and suit.max() > 0 else None
    if peak_floor is None:
        return []
    rows, cols = suit.shape
    peaks = []
    for _ in range(max_centres):
        y, x = divmod(int(np.argmax(suit)), cols)
        if suit[y, x] < peak_floor or suit[y, x] <= 0.0:
            break
        peaks.append((y, x))
        suit[max(0, y - radius) : y + radius + 1, max(0, x - radius) : x + radius + 1] = -1.0
    return peaks


def recommended_plan(
    p_built,
    p_green,
    p_centre,
    granularity_m,
    min_green_span_m,
    max_distance_m,
    green_thresh: float = 0.5,
    built_thresh: float = 0.5,
    max_centres: int = 50,
) -> np.ndarray:
    """Constraint-aware plan from the per-class probability surfaces.

    - green network: ``P(green) >= green_thresh`` kept only as connected components
      at least the min-green-span area (slivers dropped);
    - centres: peaks of the walk-smoothed ``P(centre)``, spaced >= a walk apart;
    - built: ``P(built) >= built_thresh`` elsewhere.

    Returns a uint8 categorical grid using the ``PLAN_*`` codes.
    """
    plan = np.zeros(p_built.shape, dtype=np.uint8)
    plan[p_built >= built_thresh] = PLAN_BUILT
    min_cells = max(1, round((min_green_span_m / granularity_m) ** 2))
    green = _keep_large_components(np.asarray(p_green) >= green_thresh, min_cells)
    plan[green] = PLAN_GREEN
    radius = max(1, round(max_distance_m / granularity_m))
    for y, x in _centre_peaks(np.asarray(p_centre), radius, 0.25, max_centres):
        plan[y, x] = PLAN_CENTRE
    return plan
