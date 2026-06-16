"""Pure, QGIS-free grid logic: class taxonomy, classification, grid maths.

This module imports only numpy so it can be unit-tested in a plain virtualenv
(no QGIS, no GDAL). The QGIS/GDAL-coupled IO lives in ``gis_io.py``, which imports
from here.
"""

from __future__ import annotations

import heapq
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
    (PLAN_BUILT, (170, 120, 60), "Recommended development"),
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


def _gravity_centres(p_built, built_mask, radius, max_centres, threshold_frac=0.25):
    """Place centres by a gravity model.

    Greedily pick the built cell that can reach the most built "population" within a
    walk — a box-sum of ``p_built`` over the ``radius`` catchment — then suppress a
    walk-radius neighbourhood and repeat. Candidates are restricted to ``built_mask``
    so centres sit in the fabric: an isolated spot reaches little built, scores low,
    and is never chosen; the catchment radius bakes in the max-distance constraint.
    Returns up to ``max_centres`` (row, col)s spaced >= ``radius`` apart.
    """
    gravity = _box_sum(np.asarray(p_built), radius)
    gravity = np.where(built_mask, gravity, -1.0)  # a centre must sit in built fabric
    if gravity.max() <= 0.0:
        return []
    peak_floor = threshold_frac * float(gravity.max())
    rows, cols = gravity.shape
    peaks = []
    for _ in range(max_centres):
        y, x = divmod(int(np.argmax(gravity)), cols)
        if gravity[y, x] < peak_floor or gravity[y, x] <= 0.0:
            break
        peaks.append((y, x))
        gravity[max(0, y - radius) : y + radius + 1, max(0, x - radius) : x + radius + 1] = -1.0
    return peaks


def recommended_plan(
    p_built,
    p_green,
    granularity_m,
    min_green_span_m,
    max_distance_m,
    green_thresh: float = 0.5,
    built_thresh: float = 0.5,
    min_built_cells: int = 6,
    max_centres: int = 50,
) -> np.ndarray:
    """Constraint-aware plan from the per-class probability surfaces.

    - built: ``P(built) >= built_thresh``, kept only as connected components of at
      least ``min_built_cells`` cells (slivers / leftover drips dropped);
    - green network: ``P(green) >= green_thresh`` kept as connected components of at
      least the min-green-span area;
    - centres: a gravity model — built cells that maximise reachable built within a
      walk (see ``_gravity_centres``), spaced >= a walk apart.

    Returns a uint8 categorical grid using the ``PLAN_*`` codes. ``P(centre)`` is no
    longer used for placement (centres are derived from access to built fabric) but
    is still emitted as a likelihood layer by the plugin.
    """
    plan = np.zeros(p_built.shape, dtype=np.uint8)
    built = _keep_large_components(np.asarray(p_built) >= built_thresh, min_built_cells)
    plan[built] = PLAN_BUILT
    green_min = max(1, round((min_green_span_m / granularity_m) ** 2))
    green = _keep_large_components(np.asarray(p_green) >= green_thresh, green_min)
    plan[green] = PLAN_GREEN
    radius = max(1, round(max_distance_m / granularity_m))
    for y, x in _gravity_centres(p_built, built, radius, max_centres):
        plan[y, x] = PLAN_CENTRE
    return plan


# --- plan evaluator: the "isobenefit" objective, method-agnostic ----------------
#
# Scores any PLAN_* layout so different extraction methods (consensus, greedy,
# annealing, …) can be compared on the same yardstick, and so an optimiser has an
# objective to maximise. The headline question is equity of *walkable* access:
# is every home within a walk of both a centre and qualifying green, and how badly
# off is the worst-served home? "Isobenefit" = equal benefit, so we report both the
# utilitarian mean and the egalitarian worst-case.


def _walk_distance(targets: np.ndarray, granularity_m: float, max_distance_m: float) -> np.ndarray:
    """Walking distance (metres) from every cell to the nearest target cell.

    A bounded multi-source Dijkstra over an open grid (every cell walkable), queen
    moves with diagonal cost ``sqrt(2) * granularity``. Cells further than
    ``max_distance_m`` from any target stay ``inf``.
    """
    rows, cols = targets.shape
    dist = np.full((rows, cols), math.inf)
    g = float(granularity_m)
    diag = math.sqrt(2.0)
    steps = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
             (1, 1, diag), (1, -1, diag), (-1, 1, diag), (-1, -1, diag))
    heap: list[tuple[float, int, int]] = []
    for y, x in zip(*np.nonzero(targets)):
        y, x = int(y), int(x)
        dist[y, x] = 0.0
        heap.append((0.0, y, x))
    heapq.heapify(heap)
    while heap:
        d, y, x = heapq.heappop(heap)
        if d > dist[y, x]:
            continue
        for dy, dx, w in steps:
            ny, nx = y + dy, x + dx
            if 0 <= ny < rows and 0 <= nx < cols:
                nd = d + w * g
                if nd <= max_distance_m and nd < dist[ny, nx]:
                    dist[ny, nx] = nd
                    heapq.heappush(heap, (nd, ny, nx))
    return dist


def evaluate_plan(plan: np.ndarray, granularity_m: float, max_distance_m: float) -> dict:
    """Score a recommended plan by walkable access to centres and green.

    Per built cell (centres count as built fabric) the "benefit" of each amenity is
    ``1`` at the doorstep, fading linearly to ``0`` at ``max_distance_m`` and ``0``
    beyond. Returns intuitive metrics in ``[0, 1]`` plus walk distances in metres:

    - ``centre_coverage`` / ``green_coverage`` — share of homes within a walk;
    - ``served_coverage`` — share within a walk of *both*;
    - ``mean_benefit`` — utilitarian average benefit;
    - ``worst_benefit`` — 5th-percentile benefit (the egalitarian/isobenefit headline);
    - ``centre_walk_mean`` / ``green_walk_mean`` — mean walk to each (reachable only);
    - ``compactness`` — share of built neighbours that are also built (anti-sprawl).
    """
    built = (plan == PLAN_BUILT) | (plan == PLAN_CENTRE)
    n_built = int(built.sum())
    if n_built == 0:
        return {"built_cells": 0}

    d_cent = _walk_distance(plan == PLAN_CENTRE, granularity_m, max_distance_m)[built]
    d_green = _walk_distance(plan == PLAN_GREEN, granularity_m, max_distance_m)[built]
    a_cent = np.clip(1.0 - d_cent / max_distance_m, 0.0, 1.0)  # inf -> 0 benefit
    a_green = np.clip(1.0 - d_green / max_distance_m, 0.0, 1.0)
    benefit = 0.5 * (a_cent + a_green)

    rows, cols = plan.shape
    adj = 0
    for dy, dx in ((1, 0), (0, 1)):
        a = built[: rows - dy, : cols - dx] & built[dy:, dx:]
        adj += 2 * int(a.sum())  # each shared edge counts for both cells

    return {
        "built_cells": n_built,
        "centre_coverage": float(np.mean(d_cent < math.inf)),
        "green_coverage": float(np.mean(d_green < math.inf)),
        "served_coverage": float(np.mean((d_cent < math.inf) & (d_green < math.inf))),
        "mean_benefit": float(benefit.mean()),
        "worst_benefit": float(np.percentile(benefit, 5)),
        "centre_walk_mean": float(d_cent[d_cent < math.inf].mean()) if (d_cent < math.inf).any() else math.inf,
        "green_walk_mean": float(d_green[d_green < math.inf].mean()) if (d_green < math.inf).any() else math.inf,
        "compactness": adj / (4.0 * n_built),
    }
