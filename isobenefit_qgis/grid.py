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


def _disk(r: int) -> np.ndarray:
    yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
    return (yy * yy + xx * xx) <= r * r  # circular walk catchment


def _seed_centres_proximity(built, granularity_m, max_distance_m, existing=None):
    """The original isobenefit seeding, by pure proximity: walk the built cells and plant a
    centre at any cell that is beyond a walk of every centre so far. No coverage counting, no
    Lloyd — just "is this cell out of reach of a centre? then it becomes one." ``existing``
    centres seed the initial coverage. Returns NEW ``(row, col)``.
    """
    built = np.asarray(built, dtype=bool)
    rows, cols = built.shape
    r = max(1, round(max_distance_m / granularity_m))
    disk = _disk(r)

    def stamp(mask, y, x):
        y0, y1 = max(0, y - r), min(rows, y + r + 1)
        x0, x1 = max(0, x - r), min(cols, x + r + 1)
        mask[y0:y1, x0:x1] |= disk[y0 - (y - r) : y1 - (y - r), x0 - (x - r) : x1 - (x - r)]

    covered = np.zeros((rows, cols), dtype=bool)
    for ey, ex in existing or []:
        if 0 <= ey < rows and 0 <= ex < cols:
            stamp(covered, int(ey), int(ex))
    new = []
    for y, x in np.argwhere(built):
        if not covered[y, x]:
            new.append((int(y), int(x)))
            stamp(covered, int(y), int(x))
    return new


def _refine_centres(seeds, fixed, built, new_built, granularity_m, max_distance_m, cull_min_unique=3, walk=None):
    """Optimise seeded centres after the fact, measuring catchment by ``walk`` — ONE distance
    model used for every judgment here (the grid walk by default, or true street-network
    distances when a ``walk`` callable ``mask -> rows×cols metres`` is injected). Each new centre
    is re-positioned onto NEW land, central to the NEW homes it serves; ``fixed``/existing centres
    compete in the assignment; centres uniquely serving fewer than ``cull_min_unique`` built cells
    are culled (redundant, or feeding too small a catchment). Returns the optimised new ``(row, col)``.
    """
    built = np.asarray(built, dtype=bool)
    new_built = np.asarray(new_built, dtype=bool) & built
    rows, cols = built.shape
    r = max(1, round(max_distance_m / granularity_m))
    if walk is None:

        def walk(mask):
            return _walk_distance(mask, granularity_m, max_distance_m)

    fixed = [(int(y), int(x)) for y, x in (fixed or [])]
    if not new_built.any():
        return []
    hy, hx = np.nonzero(new_built)

    def onehot(cells):
        m = np.zeros((rows, cols), dtype=bool)
        for y, x in cells:
            m[y, x] = True
        return m

    def reach(cells):  # built cells within a walk of any cell in `cells` (by the one metric)
        if not cells:
            return np.zeros((rows, cols), dtype=bool)
        return np.isfinite(walk(onehot(cells)))

    new = [_nearest_built(new_built, int(y), int(x)) for y, x in seeds]

    def lloyd(centres):  # re-position each new centre central to the NEW homes WITHIN A WALK of it
        if not centres:
            return centres
        for _ in range(8):
            # walk distance from each centre (fixed centres compete) to every NEW home
            stack = np.stack([walk(onehot([c]))[new_built] for c in fixed + centres], axis=1)
            nearest = np.argmin(stack, axis=1)
            within = np.isfinite(stack.min(axis=1))  # homes beyond a walk of every centre pull no one
            moved = False
            for j in range(len(centres)):
                members = (nearest == len(fixed) + j) & within
                if not members.any():
                    continue
                ny, nx = _nearest_built(new_built, round(hy[members].mean()), round(hx[members].mean()))
                if (ny, nx) != centres[j]:
                    centres[j] = (ny, nx)
                    moved = True
            if not moved:
                break
        return centres

    new = lloyd(new)

    # Add centres where NEW development is still beyond a walk of any centre. The densest
    # underserved cluster (a box-sum) only proposes WHERE; whether a centre is warranted there is
    # confirmed by the one metric (how many underserved homes it actually reaches within a walk).
    while True:
        underserved = new_built & ~reach(fixed + new)
        if not underserved.any():
            break
        gain = np.where(new_built, _box_sum(underserved.astype(np.float64), r), -1.0)
        if gain.max() < cull_min_unique:
            break
        y, x = divmod(int(np.argmax(gain)), cols)
        if int((reach([(int(y), int(x))]) & underserved).sum()) < cull_min_unique:
            break  # the largest remaining gap is too small to warrant a centre
        new.append((int(y), int(x)))
    new = lloyd(new)

    # Cull a centre uniquely serving < cull_min_unique built cells (redundant / overly small).
    while new:
        masks = [reach([c]) for c in fixed + new]
        count = np.sum(masks, axis=0)
        unique = [int((built & masks[len(fixed) + j] & (count == 1)).sum()) for j in range(len(new))]
        worst = min(range(len(new)), key=lambda j: unique[j])
        if unique[worst] >= cull_min_unique:
            break
        new.pop(worst)
        new = lloyd(new)
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
    - centres: seeded by proximity then optimised onto new land and culled (see
      ``_seed_centres_proximity`` / ``_refine_centres``).

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
    existing_on_built = [
        (int(ey), int(ex))
        for ey, ex in (existing_centres or [])
        if 0 <= ey < plan.shape[0] and 0 <= ex < plan.shape[1] and plan[ey, ex] == PLAN_BUILT
    ]
    for ey, ex in existing_on_built:
        plan[ey, ex] = PLAN_CENTRE
    # seed centres by proximity (no CA run here), then optimise + cull
    seed_new = _seed_centres_proximity(built, granularity_m, max_distance_m, existing_centres)
    for y, x in _refine_centres(seed_new, existing_on_built, built, built, granularity_m, max_distance_m):
        plan[y, x] = PLAN_CENTRE
    return plan


# --- plan evaluator: the "isobenefit" objective, method-agnostic ----------------
#
# Scores any PLAN_* layout on the same yardstick so extraction methods can be
# compared. The standard is a THRESHOLD, not a gradient: being within a walk
# (<= max_distance) of an amenity is "okay", full stop — 800 m is fine, not "almost
# zero". So the score is COVERAGE — is each home within a walk of a centre and of a
# real park? — and the equity headline is simply how many homes are left out.


def _walk_distance(
    targets: np.ndarray, granularity_m: float, max_distance_m: float, blocked: np.ndarray | None = None
) -> np.ndarray:
    """Walking distance (metres) from every cell to the nearest target cell.

    A bounded multi-source Dijkstra, queen moves with diagonal cost
    ``sqrt(2) * granularity``. Cells further than ``max_distance_m`` from any target stay
    ``inf``. If ``blocked`` is given, the walk cannot enter those cells (it routes around
    them) — used so distances don't cross the green network.
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
            if 0 <= ny < rows and 0 <= nx < cols and (blocked is None or not blocked[ny, nx]):
                nd = d + w * g
                if nd <= max_distance_m and nd < dist[ny, nx]:
                    dist[ny, nx] = nd
                    heapq.heappush(heap, (nd, ny, nx))
    return dist


def evaluate_plan(
    plan: np.ndarray,
    granularity_m: float,
    max_distance_m: float,
    min_green_span_m: float | None = None,
    transit_stops: np.ndarray | None = None,
    router=None,
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

    If ``transit_stops`` (a bool mask of public-transport stop cells) is given, also reports
    ``transit_coverage`` / ``transit_access`` / ``transit_walk_mean`` — walkable access to a
    stop, as a third dimension alongside centre and green. These are *reported only*: transit
    does not yet feed ``access_cost`` (the run-selection metric), so it cannot distort which
    plan is chosen until the dimension is validated.
    """
    built = (plan == PLAN_BUILT) | (plan == PLAN_CENTRE)
    n_built = int(built.sum())
    if n_built == 0:
        return {"built_cells": 0}

    green_mask = plan == PLAN_GREEN
    if min_green_span_m:  # only real parks count, matching recommended_plan
        green_min = max(1, round((min_green_span_m / granularity_m) ** 2))
        green_mask = _keep_large_components(green_mask, green_min)

    # Walking distances: a straight-ish grid walk by default, or true street-network distances
    # when a ``router`` (a callable mask -> rows×cols metres field) is injected. The default keeps
    # this function pure/headless; the network router lives in the QGIS-coupled isobenefit_qgis.routing.
    def _dist(mask):
        return router(mask) if router is not None else _walk_distance(mask, granularity_m, max_distance_m)

    d_cent = _dist(plan == PLAN_CENTRE)[built]
    d_green = _dist(green_mask)[built]
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

    metrics = {
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

    # transit access — a third dimension, REPORTED ONLY for now (not folded into access_cost,
    # so it cannot distort run-selection until validated). See the transit-routing plan.
    if transit_stops is not None:
        stops = np.asarray(transit_stops, dtype=bool)
        if stops.any():
            d_stop = _dist(stops)[built]
            near_stop = d_stop < math.inf
            metrics["transit_coverage"] = float(near_stop.mean())
            metrics["transit_access"] = float(np.where(near_stop, d_stop, penalty).mean())
            metrics["transit_walk_mean"] = float(d_stop[near_stop].mean()) if near_stop.any() else math.inf

    return metrics


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
    ca_centres=None,
    optimise_centres: bool = True,
    centre_anchors=None,
    router=None,
) -> np.ndarray:
    """Improve a plan's walkable green access by carving parks where access is worst.

    From ``plan`` (the consensus prior), repeatedly place a compact park of side
    ``min_green_span`` at the spot serving the most currently-unserved homes (built ->
    green), stopping when the best park adds too few homes or the green budget is
    spent; then, when ``optimise_centres`` (the default), re-place new centres centrally
    on the reduced fabric (existing centres kept) — otherwise the simulation's grown
    centres are kept as-is. The budget is population-aware when ``mean_density`` and ``max_density`` are
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
    # ONE distance model for the whole optimisation (carve + centres): the grid walk by default,
    # or true street-network distances when a ``router`` (mask -> rows×cols metres) is injected.
    if router is None:

        def walk(mask):
            return _walk_distance(mask, g, max_distance_m)
    else:
        walk = router

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
        d_green = walk(plan == PLAN_GREEN)
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

    # Centres: keep the existing ones; take the simulation's grown centres (``ca_centres`` — the
    # original proximity seeding). When ``optimise_centres`` (the default) tidy them — re-position
    # onto new land, central to what they serve, add where development is under-served, and cull
    # redundant ones; otherwise keep them exactly where the simulation grew them. With no
    # ca_centres (direct calls) fall back to the same proximity seeding on the finished fabric.
    plan[plan == PLAN_CENTRE] = PLAN_BUILT
    built = plan == PLAN_BUILT
    new_built = built & ~frozen
    existing_on_built = [
        (int(ey), int(ex))
        for ey, ex in (existing_centres or [])
        if 0 <= ey < rows and 0 <= ex < cols and plan[ey, ex] == PLAN_BUILT
    ]
    # Significant transit stops (rail/tram) on built land anchor a FIXED centre — kept like
    # existing centres (never culled); the other centres optimise around them.
    anchor_on_built = [
        (int(ay), int(ax))
        for ay, ax in (centre_anchors or [])
        if 0 <= ay < rows and 0 <= ax < cols and plan[ay, ax] == PLAN_BUILT
    ]
    fixed_on_built = list(dict.fromkeys(existing_on_built + anchor_on_built))  # dedup, order-stable
    exclude = {(int(ey), int(ex)) for ey, ex in (existing_centres or [])} | set(anchor_on_built)
    if ca_centres is None:
        seed_new = [
            s
            for s in _seed_centres_proximity(built, granularity_m, max_distance_m, existing_centres)
            if s not in exclude
        ]
    else:
        seed_new = [(int(y), int(x)) for y, x in ca_centres if (int(y), int(x)) not in exclude]
    for ey, ex in fixed_on_built:
        plan[ey, ex] = PLAN_CENTRE
    if optimise_centres:
        new_centres = _refine_centres(
            seed_new, fixed_on_built, built, new_built, granularity_m, max_distance_m, walk=walk
        )
    else:  # keep the simulation's grown centres as-is (only those still on built land)
        new_centres = [(y, x) for y, x in seed_new if 0 <= y < rows and 0 <= x < cols and plan[y, x] == PLAN_BUILT]
    for y, x in new_centres:
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
    """Built / green likelihood surfaces from a list of final-state grids (0 green / 1
    built / 2 centre). Returns ``(p_built, p_green)`` float32 in ``[0, 1]``.

    Centre likelihood is intentionally not emitted: the per-run centres are individual
    points that land in different places each run, so averaging them yields a diffuse
    smear rather than a meaningful likelihood. Centres belong to the recommended plan,
    not the uncertainty layers."""
    arr = np.stack([np.asarray(s) for s in states])
    return (
        (arr == 1).mean(0).astype(np.float32),
        (arr == 0).mean(0).astype(np.float32),
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
    optimise_centres=True,
    transit_stops=None,
    centre_anchors=None,
    router=None,
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
        st = np.asarray(st)
        ca_centres = [(int(y), int(x)) for y, x in np.argwhere(st == 2)]  # the CA's grown centres
        opt = optimise_plan(
            _state_to_plan(st, min_green_span_m, granularity_m, existing_green=existing_green),
            granularity_m, min_green_span_m, max_distance_m,
            mean_density=mean_density, max_density=max_density,
            existing_centres=existing_centres, centre_cost_frac=centre_cost_frac,
            existing_built=existing_built, ca_centres=ca_centres,
            optimise_centres=optimise_centres, centre_anchors=centre_anchors,
            router=router,
        )
        m = evaluate_plan(
            opt,
            granularity_m,
            max_distance_m,
            min_green_span_m=min_green_span_m,
            transit_stops=transit_stops,
            router=router,
        )
        if best is None or m["access_cost"] < best["access_cost"]:
            best_plan, best = opt, m
    if best_plan is not None:
        best_plan = _mark_existing(best_plan, existing_built=existing_built, existing_centres=existing_centres)
    return best_plan, best
