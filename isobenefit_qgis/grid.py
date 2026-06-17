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
PLAN_BUILT = 2  # new (speculative) development
PLAN_CENTRE = 3  # new centre
PLAN_EXIST_BUILT = 4  # development that was already there (frozen, shown muted)
PLAN_EXIST_CENTRE = 5  # centre that was already there

PLAN_PALETTE = [
    (PLAN_GREEN, (54, 109, 35), "Recommended green network"),
    (PLAN_EXIST_BUILT, (120, 92, 62), "Existing development"),
    (PLAN_BUILT, (196, 140, 74), "New development"),
    (PLAN_EXIST_CENTRE, (150, 40, 85), "Existing centre"),
    (PLAN_CENTRE, (210, 35, 35), "New centre"),
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


def _nearest_built(built: np.ndarray, y: int, x: int) -> tuple[int, int]:
    """Nearest built cell to (y, x) (itself if already built)."""
    rows, cols = built.shape
    if 0 <= y < rows and 0 <= x < cols and built[y, x]:
        return y, x
    for r in range(1, max(rows, cols)):
        y0, y1 = max(0, y - r), min(rows, y + r + 1)
        x0, x1 = max(0, x - r), min(cols, x + r + 1)
        sub = built[y0:y1, x0:x1]
        if sub.any():
            ys, xs = np.nonzero(sub)
            i = int(np.argmin((y0 + ys - y) ** 2 + (x0 + xs - x) ** 2))
            return int(y0 + ys[i]), int(x0 + xs[i])
    return y, x


def _place_centres(built, granularity_m, max_distance_m, max_new=200, existing=None, centre_cost_frac=0.0):
    """Reverse-engineer good locations for NEW centres (facility location).

    Default (``centre_cost_frac=0``): the **fewest centres that still put every
    reachable home within a walk** — greedy set cover, parameter-free. ``centre_cost_frac``
    (× total homes) is an optional **cost-per-centre dial**: a centre is added only while
    it newly serves at least that many homes, so raising it trims to fewer, busier
    centres (and leaves the thinnest pockets uncovered). Each new centre is snapped to
    the CENTROID of the homes it serves (a Lloyd step) so it sits central to its
    catchment; ``existing`` centres are kept fixed and seed the initial coverage.
    Returns the list of NEW ``(row, col)``.
    """
    built = np.asarray(built, dtype=bool)
    rows, cols = built.shape
    r = max(1, round(max_distance_m / granularity_m))
    yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
    disk = (yy * yy + xx * xx) <= r * r  # circular walk catchment

    def stamp(mask, y, x):  # mark the walk catchment of a centre at (y, x)
        y0, y1 = max(0, y - r), min(rows, y + r + 1)
        x0, x1 = max(0, x - r), min(cols, x + r + 1)
        mask[y0:y1, x0:x1] |= disk[y0 - (y - r) : y1 - (y - r), x0 - (x - r) : x1 - (x - r)]

    covered = np.zeros((rows, cols), dtype=bool)
    for ey, ex in existing or []:
        if 0 <= ey < rows and 0 <= ex < cols:
            stamp(covered, ey, ex)

    min_audience = max(1, int(centre_cost_frac * int(built.sum())))
    new = []
    for _ in range(max_new):
        unserved = built & ~covered
        gain = np.where(built, _box_sum(unserved.astype(np.float64), r), -1.0)
        if gain.max() <= 0.0:
            break
        y, x = divmod(int(np.argmax(gain)), cols)
        audience = 0
        for _ in range(4):  # settle onto the centroid of the homes this centre serves
            y0, y1 = max(0, y - r), min(rows, y + r + 1)
            x0, x1 = max(0, x - r), min(cols, x + r + 1)
            served = unserved[y0:y1, x0:x1] & disk[y0 - (y - r) : y1 - (y - r), x0 - (x - r) : x1 - (x - r)]
            ys, xs = np.nonzero(served)
            if ys.size == 0:
                break
            audience = int(ys.size)
            cy, cx = _nearest_built(built, y0 + int(round(ys.mean())), x0 + int(round(xs.mean())))
            if (cy, cx) == (y, x):
                break
            y, x = cy, cx
        if audience < min_audience:  # cost test — not enough new audience to justify a centre
            break
        new.append((y, x))
        stamp(covered, y, x)

    # Global refinement (Lloyd / k-means): repeatedly assign each home to its nearest
    # NEW centre and move that centre to its members' centroid, settling them jointly
    # into central, balanced positions (≈ the p-median optimum) rather than the greedy
    # placement. Homes already within an existing centre's walk are claimed by those
    # centres and excluded — this keeps new centres out of already-served areas AND
    # avoids an O(homes × centre-cells) matrix when ``existing`` is a whole centre AREA
    # (true-area centres) rather than a few point seeds.
    if new:
        fixed_cover = np.zeros((rows, cols), dtype=bool)
        for ey, ex in existing or []:
            if 0 <= ey < rows and 0 <= ex < cols:
                stamp(fixed_cover, ey, ex)
        hy, hx = np.nonzero(built & ~fixed_cover)
        sy = np.array([s[0] for s in new])
        sx = np.array([s[1] for s in new])
        for _ in range(12):
            if hy.size == 0:
                break
            nearest = np.argmin((hy[:, None] - sy) ** 2 + (hx[:, None] - sx) ** 2, axis=1)
            moved = False
            for j in range(len(new)):
                members = nearest == j
                if not members.any():
                    continue
                ny, nx = _nearest_built(built, round(hy[members].mean()), round(hx[members].mean()))
                if (ny, nx) != new[j]:
                    new[j] = (ny, nx)
                    sy[j], sx[j] = ny, nx
                    moved = True
            if not moved:
                break
    return new


def recommended_plan(
    p_built,
    p_green,
    granularity_m,
    min_green_span_m,
    max_distance_m,
    green_thresh: float = 0.5,
    built_thresh: float = 0.5,
    min_built_cells: int = 6,
    max_centres: int = 200,
    existing_centres=None,
) -> np.ndarray:
    """Constraint-aware plan from the per-class probability surfaces.

    - built: ``P(built) >= built_thresh``, kept only as connected components of at
      least ``min_built_cells`` cells (slivers / leftover drips dropped);
    - green network: ``P(green) >= green_thresh`` kept as connected components of at
      least the min-green-span area;
    - centres: new centres placed centrally to serve homes not already within a walk
      of an ``existing_centres`` location (see ``_place_centres``).

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
    for ey, ex in existing_centres or []:
        if 0 <= ey < plan.shape[0] and 0 <= ex < plan.shape[1] and plan[ey, ex] == PLAN_BUILT:
            plan[ey, ex] = PLAN_CENTRE
    for y, x in _place_centres(built, granularity_m, max_distance_m, max_centres, existing=existing_centres):
        plan[y, x] = PLAN_CENTRE
    return plan


# --- plan evaluator: the "isobenefit" objective, method-agnostic ----------------
#
# Scores any PLAN_* layout on the same yardstick so extraction methods can be
# compared. The standard is a THRESHOLD, not a gradient: being within a walk
# (<= max_distance) of an amenity is "okay", full stop — 800 m is fine, not "almost
# zero". So the score is COVERAGE — is each home within a walk of a centre and of a
# real park? — and the equity headline is simply how many homes are left out.


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


def evaluate_plan(
    plan: np.ndarray, granularity_m: float, max_distance_m: float, min_green_span_m: float | None = None
) -> dict:
    """Score a recommended plan by COVERAGE — who is within a walk of what.

    A home is *served* if it is within ``max_distance_m`` of both a centre and a real
    park (within the walk = okay; not a gradient). Only green patches of at least
    ``min_green_span_m`` across count as parks (specks don't). Returns shares in
    ``[0, 1]``:

    - ``centre_coverage`` / ``green_coverage`` — share of homes within a walk of each;
    - ``served_coverage`` — share within a walk of *both* (the headline);
    - ``unserved_fraction`` — share left out (the equity headline);
    - ``access_cost`` — average walk (m) to amenities over every home (unreachable counted
      at a penalty); the **selection metric** (lower better). ``centre_access`` /
      ``green_access`` are its two halves — avg walk to a centre, and to green, separately;
    - ``centre_walk_mean`` / ``green_walk_mean`` — mean walk to each (reachable only);
    - ``compactness`` — share of built neighbours that are also built (anti-sprawl).
    """
    built = (plan == PLAN_BUILT) | (plan == PLAN_CENTRE)
    n_built = int(built.sum())
    if n_built == 0:
        return {"built_cells": 0}

    green_mask = plan == PLAN_GREEN
    if min_green_span_m:  # only real parks count, matching recommended_plan
        green_min = max(1, round((min_green_span_m / granularity_m) ** 2))
        green_mask = _keep_large_components(green_mask, green_min)

    d_cent = _walk_distance(plan == PLAN_CENTRE, granularity_m, max_distance_m)[built]
    d_green = _walk_distance(green_mask, granularity_m, max_distance_m)[built]
    near_cent = d_cent < math.inf
    near_green = d_green < math.inf
    served = near_cent & near_green

    # selection metric: average walk to amenities over EVERY home, with anyone who
    # can't reach within the limit counted at a penalty distance (so a plan can't
    # score well by abandoning the fringe). Lower is better.
    penalty = 2.0 * max_distance_m
    centre_access = float(np.where(near_cent, d_cent, penalty).mean())
    green_access = float(np.where(near_green, d_green, penalty).mean())
    access_cost = 0.5 * (centre_access + green_access)

    rows, cols = plan.shape
    adj = 0
    for dy, dx in ((1, 0), (0, 1)):
        a = built[: rows - dy, : cols - dx] & built[dy:, dx:]
        adj += 2 * int(a.sum())  # each shared edge counts for both cells

    # supply-side efficiency: how well-used each centre / unit of green is
    n_centres = int((plan == PLAN_CENTRE).sum())
    n_green = int(green_mask.sum())

    return {
        "built_cells": n_built,
        "centre_coverage": float(near_cent.mean()),
        "green_coverage": float(near_green.mean()),
        "served_coverage": float(served.mean()),
        "unserved_fraction": float((~served).mean()),
        "access_cost": access_cost,  # mean of the two below — the selection metric (lower better)
        "centre_access": centre_access,  # avg walk to a centre over all homes (penalised)
        "green_access": green_access,  # avg walk to green over all homes (penalised)
        "centre_walk_mean": float(d_cent[near_cent].mean()) if near_cent.any() else math.inf,
        "green_walk_mean": float(d_green[near_green].mean()) if near_green.any() else math.inf,
        "compactness": adj / (4.0 * n_built),
        "centre_efficiency": float(near_cent.sum()) / n_centres if n_centres else 0.0,  # homes served per centre
        "green_efficiency": float(near_green.sum()) / n_green if n_green else 0.0,  # homes served per green cell
    }


# --- hybrid optimiser: greedy coverage over the consensus prior ------------------
#
# The consensus plan (recommended_plan) is the CA's forecast; the evaluator shows
# its weak spot is usually green access. This polishes that prior toward the
# objective: greedily carve a green network into the built fabric at the spots where
# the worst-served homes are, until coverage stops improving meaningfully or the
# green budget is spent, then re-place centres on the fabric.
#
# Population-aware budget (the "constant inhabitants" principle of Isobenefit
# urbanism): green is paid for by DENSIFYING the remaining built fabric, never by
# deleting homes. If the current fabric houses its population at ``mean_density`` and
# can be densified up to ``max_density``, then a fraction ``1 - mean_density/max_density``
# of built cells can be freed to green while still housing everyone — that is the
# budget. With no densities given it falls back to a flat ``max_green_frac``.
#
# Park placement uses a fast box-sum coverage proxy; the actual coverage that decides
# the loop is recomputed each step by real walk-distance, so this is a deterministic
# greedy heuristic (no global-optimality claim) whose reported coverage is honest.


def optimise_plan(
    plan: np.ndarray,
    granularity_m: float,
    min_green_span_m: float,
    max_distance_m: float,
    mean_density: float | None = None,
    max_density: float | None = None,
    max_green_frac: float = 0.2,
    max_centres: int = 200,
    existing_centres=None,
    centre_cost_frac: float = 0.0,
    existing_built=None,
) -> np.ndarray:
    """Improve a plan's walkable green access by carving parks where access is worst.

    From ``plan`` (the consensus prior), repeatedly place a compact park of side
    ``min_green_span`` at the spot serving the most currently-unserved homes (built ->
    green), stopping when the best park adds too few homes or the green budget is
    spent; then re-place new centres centrally on the reduced fabric (existing centres
    kept). The budget is population-aware when ``mean_density`` and ``max_density`` are
    given (green funded by densification, never by lost housing); otherwise a flat
    ``max_green_frac``. Returns a new plan.

    ``existing_built`` is a bool mask of cells that were already developed before the
    simulation. Those are **frozen**: the plan may green-over only NEW (speculative)
    built land, never prune what is already there.
    """
    plan = plan.copy()
    g = float(granularity_m)
    side = max(2, round(min_green_span_m / g))
    half = side // 2
    walk_r = max(1, round(max_distance_m / g))
    rows, cols = plan.shape

    n_built = int(((plan == PLAN_BUILT) | (plan == PLAN_CENTRE)).sum())
    if n_built == 0:
        return plan
    if mean_density and max_density and max_density > 0:
        budget_frac = max(0.0, 1.0 - mean_density / max_density)
    else:
        budget_frac = max_green_frac
    green_budget = int(budget_frac * n_built)
    min_gain = max(1.0, 0.002 * n_built)  # stop once a park serves only a few stragglers

    # Existing development is frozen: parks may be carved only from NEW (speculative)
    # built land, never from what is already there.
    frozen = np.zeros(plan.shape, dtype=bool)
    if existing_built is not None:
        frozen |= np.asarray(existing_built, dtype=bool)

    spent = 0
    while spent < green_budget:
        built = (plan == PLAN_BUILT) | (plan == PLAN_CENTRE)
        carvable = built & ~frozen  # only new built may be freed to green
        d_green = _walk_distance(plan == PLAN_GREEN, g, max_distance_m)
        unserved = built & ~np.isfinite(d_green)  # every home wants green access...
        if not unserved.any():
            break
        # homes a park here would newly serve: unserved within a walk of the park
        gain = _box_sum(unserved.astype(np.float64), walk_r + half)
        gain = np.where(carvable, gain, -1.0)  # ...but a park may only be carved from new land
        if gain.max() < min_gain:  # diminishing returns — don't over-green for stragglers
            break
        cy, cx = divmod(int(np.argmax(gain)), cols)
        y0, y1 = max(0, cy - half), min(rows, cy + half + 1)
        x0, x1 = max(0, cx - half), min(cols, cx + half + 1)
        block = plan[y0:y1, x0:x1]
        carved = carvable[y0:y1, x0:x1]  # leave any existing built within the footprint intact
        carved_n = int(carved.sum())
        if carved_n == 0 or spent + carved_n > green_budget:  # never overshoot the budget
            break
        block[carved] = PLAN_GREEN
        spent += carved_n

    # re-place centres on the fabric that remains after carving: keep existing
    # centres, add new ones centrally to serve homes they don't already reach
    plan[plan == PLAN_CENTRE] = PLAN_BUILT
    built = plan == PLAN_BUILT
    for ey, ex in existing_centres or []:
        if 0 <= ey < rows and 0 <= ex < cols and plan[ey, ex] == PLAN_BUILT:
            plan[ey, ex] = PLAN_CENTRE
    for y, x in _place_centres(
        built, granularity_m, max_distance_m, max_centres, existing=existing_centres, centre_cost_frac=centre_cost_frac
    ):
        plan[y, x] = PLAN_CENTRE
    return plan


def capacity_summary(built_before: int, built_after: int, mean_density: float, max_density: float) -> dict:
    """Population accounting for an optimised plan (constant-inhabitants check).

    The plan held its population (``built_before * mean_density``) while freeing some
    built cells to green; the remaining cells must absorb everyone by densifying.
    Returns the population held, the density before and the density now required, the
    max permitted, and whether that is feasible (required <= max).
    """
    population = built_before * mean_density
    density_after = population / built_after if built_after else math.inf
    return {
        "population": population,
        "built_before": built_before,
        "built_after": built_after,
        "density_before": mean_density,
        "density_after": density_after,
        "max_density": max_density,
        "feasible": density_after <= max_density + 1e-9,
    }


# --- selecting the recommended plan from an ensemble of single runs --------------
#
# The ensemble gives both the uncertainty (likelihood) surfaces AND a set of coherent
# single-run layouts. The recommended plan is the BEST single run, optimised — not the
# blurred average — because a coherent fabric optimises far better (see the review).


def class_probabilities(states):
    """Per-class likelihood surfaces from a list of final-state grids (0 green / 1
    built / 2 centre). Returns ``(p_built, p_green, p_centre)`` float32 in ``[0, 1]``."""
    arr = np.stack([np.asarray(s) for s in states])
    return (
        (arr == 1).mean(0).astype(np.float32),
        (arr == 0).mean(0).astype(np.float32),
        (arr == 2).mean(0).astype(np.float32),
    )


def _state_to_plan(state, min_green_span_m, granularity_m, existing_green=None) -> np.ndarray:
    """Map a single run's final state to a PLAN_* layout: built/centre -> built (the
    optimiser re-places centres), green kept only as qualifying parks (>= min-span).
    ``existing_green`` cells are always kept as green so existing parks are never lost."""
    state = np.asarray(state)
    plan = np.zeros(state.shape, dtype=np.uint8)
    plan[(state == 1) | (state == 2)] = PLAN_BUILT
    green_min = max(1, round((min_green_span_m / granularity_m) ** 2))
    plan[_keep_large_components(state == 0, green_min)] = PLAN_GREEN
    if existing_green is not None:
        plan[np.asarray(existing_green, dtype=bool)] = PLAN_GREEN  # never drop existing green
    return plan


def _mark_existing(plan: np.ndarray, existing_built=None, existing_centres=None) -> np.ndarray:
    """Relabel existing development with its own PLAN_* codes (a different hue) so the map
    distinguishes what is already there from what is newly recommended.

    Runs once on the chosen plan, AFTER scoring, so the optimiser and evaluator still
    operate on the merged built/centre codes.
    """
    out = plan.copy()
    for ey, ex in existing_centres or []:
        if 0 <= ey < out.shape[0] and 0 <= ex < out.shape[1] and out[ey, ex] == PLAN_CENTRE:
            out[ey, ex] = PLAN_EXIST_CENTRE
    if existing_built is not None:
        out[(out == PLAN_BUILT) & np.asarray(existing_built, dtype=bool)] = PLAN_EXIST_BUILT
    return out


def select_plan(
    states,
    granularity_m,
    min_green_span_m,
    max_distance_m,
    mean_density=None,
    max_density=None,
    existing_centres=None,
    centre_cost_frac=0.0,
    max_eval=None,
    existing_built=None,
    existing_green=None,
):
    """Pick the recommended plan from per-run final states: optimise EVERY run and keep
    the one with the lowest average walk (``access_cost``). Pass ``max_eval`` to optimise
    only that many evenly-sampled runs (faster for very large ensembles; runs are
    similar). ``existing_built``/``existing_green`` (bool masks of already-developed land)
    are frozen — never pruned — and the chosen plan tags them with the existing-* codes.
    Returns ``(best_plan, best_metrics)`` — ``(None, None)`` if ``states`` empty.
    """
    states = list(states)
    if not states:
        return None, None
    if max_eval and len(states) > max_eval:  # optional cap for very large ensembles
        states = states[:: len(states) // max_eval][:max_eval]
    best_plan, best = None, None
    for st in states:
        opt = optimise_plan(
            _state_to_plan(st, min_green_span_m, granularity_m, existing_green=existing_green),
            granularity_m, min_green_span_m, max_distance_m,
            mean_density=mean_density, max_density=max_density,
            existing_centres=existing_centres, centre_cost_frac=centre_cost_frac,
            existing_built=existing_built,
        )
        m = evaluate_plan(opt, granularity_m, max_distance_m, min_green_span_m=min_green_span_m)
        if best is None or m["access_cost"] < best["access_cost"]:
            best_plan, best = opt, m
    if best_plan is not None:
        best_plan = _mark_existing(best_plan, existing_built=existing_built, existing_centres=existing_centres)
    return best_plan, best
