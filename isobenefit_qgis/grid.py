"""Pure, QGIS-free grid logic: class taxonomy, classification, grid maths.

This module imports only numpy so it can be unit-tested in a plain virtualenv
(no QGIS, no GDAL). The QGIS/GDAL-coupled IO lives in ``gis_io.py``, which imports
from here.
"""

from __future__ import annotations

import heapq
import math
from collections import deque

import numpy as np

# Categorical class codes for the output raster.
NODATA = 255
NATURE = 0
NEW_LOW = 1
NEW_MED = 2
NEW_HIGH = 3
CENTRE_LOW = 4
CENTRE_MED = 5
CENTRE_HIGH = 6
EXIST_BUILT = 7
FIXED_GREEN = 8
EXIST_CENTRE = 9

# New development reads as two hue families, each in three tiers so high/medium/low is obvious:
# built as a yellow -> amber -> orange-brown ramp, mixed-use centres as a pale -> deep red ramp.
# Existing fabric is a single muted shade (it carries no density and is not counted).
_BUILT_LOW = (255, 237, 160)
_BUILT_MED = (254, 196, 79)
_BUILT_HIGH = (232, 146, 20)  # deep amber, saturated and yellow: brown belongs to the existing fabric
_CENTRE_LOW = (252, 187, 161)
_CENTRE_MED = (239, 101, 72)
_CENTRE_HIGH = (179, 18, 24)
_EXIST_BUILT = (143, 114, 89)  # eased a step: still unmistakably brown and dull beside the amber ramp, less heavy
_EXIST_CENTRE = (48, 42, 38)  # warm near-black: keeps the strongest-mark role without pure black's weight
_GREEN = (54, 109, 35)

# (class code, (r, g, b), legend label) — the single-run animation palette.
PALETTE = [
    (NATURE, (89, 176, 60), "Nature / green"),
    (NEW_LOW, _BUILT_LOW, "New built — low density"),
    (NEW_MED, _BUILT_MED, "New built — medium density"),
    (NEW_HIGH, _BUILT_HIGH, "New built — high density"),
    (CENTRE_LOW, _CENTRE_LOW, "Mixed-use centre — low density"),
    (CENTRE_MED, _CENTRE_MED, "Mixed-use centre — medium density"),
    (CENTRE_HIGH, _CENTRE_HIGH, "Mixed-use centre — high density"),
    (EXIST_BUILT, _EXIST_BUILT, "Existing built"),
    (EXIST_CENTRE, _EXIST_CENTRE, "Existing mixed-use centre"),
    (FIXED_GREEN, _GREEN, "Existing green / park"),
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

    ``per_block`` is ``(high, med, low)`` persons-per-block; new-built and new-centre
    cells each carry one of these exact drawn values, so both are split into density
    tiers (built and centres in distinct hues). Existing (origin) features take visual
    precedence.
    """
    high_pb, med_pb, low_pb = per_block
    cls = np.full(state.shape, NODATA, dtype=np.uint8)
    cls[state == 0] = NATURE
    built = state == 1
    cls[built & np.isclose(density, low_pb)] = NEW_LOW
    cls[built & np.isclose(density, med_pb)] = NEW_MED
    cls[built & np.isclose(density, high_pb)] = NEW_HIGH
    centre = state == 2
    cls[centre & np.isclose(density, low_pb)] = CENTRE_LOW
    cls[centre & np.isclose(density, med_pb)] = CENTRE_MED
    cls[centre & np.isclose(density, high_pb)] = CENTRE_HIGH
    cls[origin == 1] = EXIST_BUILT
    cls[origin == 0] = FIXED_GREEN
    # existing centre seeds carry no density (never counted); tag them so they stay visible
    cls[origin == 2] = EXIST_CENTRE
    return cls


# --- constraint-aware "recommended plan" derived from the probability surfaces ---

PLAN_NONE = 0
PLAN_GREEN = 1
PLAN_BUILT = 2  # new (speculative) development — base code, used by all metric logic
PLAN_CENTRE = 3  # new mixed-use centre — base code
PLAN_EXIST_BUILT = 4  # development that was already there (frozen, shown muted)
PLAN_EXIST_CENTRE = 5  # mixed-use centre that was already there
# Per-tier DISPLAY codes: written to disk by to_tiered_plan so the map shows low/medium/high
# development in distinct shades. The base PLAN_BUILT/PLAN_CENTRE codes stay for all logic.
PLAN_BUILT_LOW = 6
PLAN_BUILT_MED = 7
PLAN_BUILT_HIGH = 8
PLAN_CENTRE_LOW = 9
PLAN_CENTRE_MED = 10
PLAN_CENTRE_HIGH = 11

PLAN_REJECT_UNSERVABLE = 12
# A carved barrier that carries a pedestrian crossing: never developed, like PLAN_NONE, but
# walkable. A footbridge over a railway does not make the railway a building site, and a
# model that refuses the crossing severs land the planner can plainly reach on foot.
PLAN_CROSSING = 13
# state grids use -2 for the same thing, so the value stays out of the way of the built
# (>0) and green (0) tests that run everywhere
STATE_CROSSING = -2

PLAN_PALETTE = [
    (PLAN_GREEN, _GREEN, "Recommended green network"),
    (PLAN_REJECT_UNSERVABLE, (214, 96, 77), "Rejected: no viable centre within the walk"),
    (PLAN_CROSSING, (224, 33, 138), "Walkable route through a barrier"),
    (PLAN_EXIST_BUILT, _EXIST_BUILT, "Existing development"),
    (PLAN_EXIST_CENTRE, _EXIST_CENTRE, "Existing mixed-use centre"),
    (PLAN_BUILT_LOW, _BUILT_LOW, "New development — low density"),
    (PLAN_BUILT_MED, _BUILT_MED, "New development — medium density"),
    (PLAN_BUILT_HIGH, _BUILT_HIGH, "New development — high density"),
    (PLAN_CENTRE_LOW, _CENTRE_LOW, "New mixed-use centre — low density"),
    (PLAN_CENTRE_MED, _CENTRE_MED, "New mixed-use centre — medium density"),
    (PLAN_CENTRE_HIGH, _CENTRE_HIGH, "New mixed-use centre — high density"),
    # base codes kept as a neutral fallback for any raster written without tiering
    (PLAN_BUILT, _BUILT_MED, "New development"),
    (PLAN_CENTRE, _CENTRE_MED, "New mixed-use centre"),
]


def _label_components(mask: np.ndarray, queen: bool):
    """Engine-labelled connected components (0 = background, 1..n), or None when the
    engine predates ``label_components``; callers then run their exact Python fallback."""
    try:
        import isobenefit

        return isobenefit.label_components(np.ascontiguousarray(mask, dtype=bool), queen)
    except (ImportError, AttributeError):
        return None


DEFAULT_MIN_PARK_AREA_M2 = 20_000.0  # 2 ha: Natural England's accessible natural greenspace standard


def walk_ball_cells(granularity_m: float, distance_m: float) -> int:
    """The number of cells within the bounded grid walk of a single open cell: the walk-ball.
    The ball bounds a centre's possible catchment, so ``threshold cells > ball cells`` means
    no centre can ever be viable at that walk, and a threshold near the ball needs the ball
    built nearly solid. Computed exactly on an open grid."""
    r = max(1, int(math.ceil(distance_m / granularity_m)))
    n = 2 * r + 3
    centre = np.zeros((n, n), dtype=bool)
    centre[r + 1, r + 1] = True
    d = _walk_distance(centre, granularity_m, distance_m)
    return int((d <= distance_m).sum())


def park_threshold_cells(granularity_m: float, min_park_area_m2: float | None = None) -> int:
    """Cells a park-qualifying green area must hold: the minimum park area (2 ha default,
    the accessible-greenspace standard) over the cell area, matching the engine's park
    rule. An area of zero yields a one-cell threshold, so any green counts."""
    area = DEFAULT_MIN_PARK_AREA_M2 if min_park_area_m2 is None else float(min_park_area_m2)
    return max(1, round(area / (granularity_m * granularity_m)))


def _keep_large_components(mask: np.ndarray, min_cells: int) -> np.ndarray:
    """Zero out rook-connected components of ``mask`` smaller than ``min_cells``."""
    mask = np.asarray(mask, dtype=bool)
    labels = _label_components(mask, queen=False)
    if labels is not None:
        n = int(labels.max())
        if n == 0:
            return np.zeros_like(mask)
        keep = np.bincount(labels.ravel()) >= min_cells
        keep[0] = False
        return keep[labels]
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


def _erode3(mask):
    """3x3 erosion without wrap-around: a cell survives only if its whole 3x3 block is set."""
    out = mask.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            shifted = np.zeros_like(mask)
            ys = slice(max(dy, 0), mask.shape[0] + min(dy, 0))
            xs = slice(max(dx, 0), mask.shape[1] + min(dx, 0))
            ys_src = slice(max(-dy, 0), mask.shape[0] + min(-dy, 0))
            xs_src = slice(max(-dx, 0), mask.shape[1] + min(-dx, 0))
            shifted[ys, xs] = mask[ys_src, xs_src]
            out &= shifted
    return out


def _dilate3(mask):
    """3x3 dilation without wrap-around: a cell is set if any cell of its 3x3 block is set."""
    out = mask.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            shifted = np.zeros_like(mask)
            ys = slice(max(dy, 0), mask.shape[0] + min(dy, 0))
            xs = slice(max(dx, 0), mask.shape[1] + min(dx, 0))
            ys_src = slice(max(-dy, 0), mask.shape[0] + min(-dy, 0))
            xs_src = slice(max(-dx, 0), mask.shape[1] + min(-dx, 0))
            shifted[ys, xs] = mask[ys_src, xs_src]
            out |= shifted
    return out


def green_unviable_pockets(
    state,
    origin,
    min_settlement_cells: int,
    existing_centres=None,
    granularity_m: float | None = None,
    centre_distance_m: float | None = None,
) -> int:
    """Reclassify land that cannot carry viably served development as protected green, in place.

    Developable land must satisfy a width condition and a service condition. It must be
    locally wide: a cell counts only inside a 3x3 block of open land, or immediately beside
    one (a morphological opening), which drops the one- and two-cell slivers that thread
    between existing buildings and along barriers. And its region must be serviceable: the
    wide land is grouped into rook-connected regions (a diagonal corner touch does not
    connect), and a region qualifies either by holding at least the service viability
    threshold in cells, so it can support a minimal centre of its own, or by lying partly
    within the centre walk of an existing centre, so growth there starts as served infill.
    Viability belongs to the service catchment, not to the land parcel: a small field
    against a centred town is developable, while the same field detached in open country is
    not. Every candidate cell that fails is marked as fixed green (``origin`` 0): it stays
    walkable and qualifies as green space, an enclosed scrap inside a town reads as a pocket
    park, and no growth, seeding, or centre provision is ever spent on it.

    The infill exemption needs ``existing_centres`` (cell coordinates), ``granularity_m``,
    and ``centre_distance_m``; when any is missing, only the capacity test applies.
    ``state``/``origin`` follow the simulation convention (state: -1 unbuildable, 0 nature,
    1 built; origin: 1 existing built, 0 fixed green, -1 free). ``origin`` is modified in
    place; the count of reclassified cells is returned.
    """
    state = np.asarray(state)
    min_cells = max(int(min_settlement_cells), MIN_SETTLEMENT_CELLS)
    buildable = (state == 0) & (np.asarray(origin) != 0)
    if not buildable.any():
        return 0
    core = _erode3(buildable)
    wide = _dilate3(core) & buildable

    served = None
    if existing_centres is not None and granularity_m and centre_distance_m:
        cent = np.zeros_like(buildable)
        for y, x in existing_centres:
            if 0 <= y < cent.shape[0] and 0 <= x < cent.shape[1]:
                cent[y, x] = True
        if cent.any():
            d = _walk_distance(cent, granularity_m, centre_distance_m, blocked=(state == -1))
            served = np.isfinite(d)

    keep = np.zeros_like(buildable)
    for comp in _components(wide, queen=False):
        viable = len(comp) >= min_cells
        if not viable and served is not None:
            viable = any(served[y, x] for y, x in comp)
        if viable:
            for y, x in comp:
                keep[y, x] = True
    unviable = buildable & ~keep
    origin[unviable] = 0
    return int(unviable.sum())


def sterile_fabric(existing_built, existing_centres) -> np.ndarray:
    """Existing built cells whose contiguous settlement holds no centre seed: outlying
    farmsteads and hamlets. Passed to the simulation as its ``sterile`` mask, so new growth
    never nucleates against them; development attaches to centred settlements or arrives as
    a dispersal-seeded centre of its own. Every settlement in a finished run therefore has
    a centre, matching what post-processing enforces."""
    built = np.asarray(existing_built, dtype=bool)
    out = np.zeros_like(built)
    cents = {(int(y), int(x)) for y, x in (existing_centres or [])}
    for comp in _components(built):
        if not any((int(y), int(x)) in cents for y, x in comp):
            for y, x in comp:
                out[y, x] = True
    return out


def sanitise_seeds(seeds, state, granularity_m, max_snap_m):
    """Re-home or drop centre seeds that fall on unbuildable land (state -1).

    Rasterisation can strand a legitimate seed on a carved corridor or water cell (a town-centre
    polygon's representative point can land on a buffered road, say). The core rejects such seeds
    outright, so each is snapped to the nearest buildable cell within ``max_snap_m``; a seed with
    no buildable cell in range is dropped. Returns ``(kept, n_snapped, n_dropped)``; ``kept``
    preserves input order and is deduplicated after snapping.
    """
    rows, cols = state.shape
    buildable = state >= 0
    max_r = max(0, int(max_snap_m / granularity_m))
    kept: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    n_snapped = n_dropped = 0
    for y, x in seeds:
        if buildable[y, x]:
            target = (y, x)
        else:
            target = None
            for r in range(1, max_r + 1):
                y0, y1 = max(0, y - r), min(rows, y + r + 1)
                x0, x1 = max(0, x - r), min(cols, x + r + 1)
                sub = buildable[y0:y1, x0:x1]
                if sub.any():
                    ys, xs = np.nonzero(sub)
                    i = int(np.argmin((y0 + ys - y) ** 2 + (x0 + xs - x) ** 2))
                    target = (int(y0 + ys[i]), int(x0 + xs[i]))
                    n_snapped += 1
                    break
            if target is None:
                n_dropped += 1
                continue
        if target not in seen:
            seen.add(target)
            kept.append(target)
    return kept, n_snapped, n_dropped


def _interior_point(region: np.ndarray):
    """The cell most INTERIOR to a bool region — the one farthest (8-connected) from any non-region
    cell, i.e. its "pole of inaccessibility". It is always inside the region, and naturally lands in
    the THICKEST contiguous part (a thin arm or a detached speck has a small interior distance), so a
    centre placed here sits deep inside its built catchment rather than on an edge or across a gap —
    which the plain centroid does NOT guarantee on an L-shape, a ring, or a multi-blob catchment.
    Cropped to the region's bounding box (+1 margin of "outside") to stay cheap. Returns (row, col) or
    None for an empty region; ties resolve to the first in row-major order (deterministic).
    """
    region = np.asarray(region, dtype=bool)
    ys, xs = np.nonzero(region)
    if len(ys) == 0:
        return None
    y0, x0 = int(ys.min()), int(xs.min())
    sub = region[y0 : int(ys.max()) + 1, x0 : int(xs.max()) + 1]
    cur = np.zeros((sub.shape[0] + 2, sub.shape[1] + 2), dtype=bool)
    cur[1:-1, 1:-1] = sub  # 1-cell False border = "outside", so erosion shrinks inward correctly
    # Chebyshev distance-to-edge by iterative 8-connected erosion: each pass peels one rim, and a cell's
    # depth is how many passes it survives. The deepest surviving cell is the pole of inaccessibility.
    dist = cur.astype(np.int32)
    work = cur
    while True:
        nb = np.ones_like(work)  # nb[i,j] := AND of the 8 neighbours of (i,j) in `work`
        nb[1:, :] &= work[:-1, :]
        nb[:-1, :] &= work[1:, :]
        nb[:, 1:] &= work[:, :-1]
        nb[:, :-1] &= work[:, 1:]
        nb[1:, 1:] &= work[:-1, :-1]
        nb[:-1, :-1] &= work[1:, 1:]
        nb[1:, :-1] &= work[:-1, 1:]
        nb[:-1, 1:] &= work[1:, :-1]
        work = work & nb  # survives erosion iff itself and all 8 neighbours were set
        if not work.any():
            break
        dist += work
    idx = int(np.argmax(dist))
    py, px = divmod(idx, cur.shape[1])
    return (y0 - 1 + py, x0 - 1 + px)


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


def _refine_centres(
    seeds, fixed, built, new_built, granularity_m, max_distance_m, walk=None, anchors=None,
    walk_cache=None, minimise_count=False, infill=None, reposition=True,
):
    """Optimise seeded centres after the fact, measuring catchment by ``walk`` — ONE distance
    model used for every judgment here: the bounded grid walk (callers inject it as the
    ``walk`` callable ``mask -> rows×cols metres``, sharing one barrier mask and cache). Each new
    centre is re-positioned onto NEW land, central to the NEW homes it serves; ``fixed``/existing
    centres compete in the assignment. Returns the optimised new ``(row, col)`` list.

    WALK CONSTRAINT (hard): on return, every NEW built cell is within ``max_distance_m`` (the
    centre walk) of a new centre or an anchor. The growth rules enforce this distance while the
    settlement grows, and nothing here may un-serve a home: a repositioning that would strand one
    is discarded, a centre is added wherever new development would otherwise be beyond the walk,
    and a removal is only allowed when every home stays covered without it.

    ``minimise_count`` picks the placement product. False re-positions the seeded centres (and
    adds where provision is missing) without ever removing one — the run's own centres, walked
    into their best positions. True additionally removes centres one at a time, most redundant
    first, for as long as full coverage and the anchor invariant hold — the fewest centres the
    walk constraint permits.

    Walking distances traverse green, so a cluster may be served by a centre across a green
    gap: nearby settlements pool their demand and their provision as a constellation, and no
    per-settlement anchor is required (the 2026-08-21 centre-first viability doctrine; the
    caller cuts centres below threshold demand and prunes homes no viable centre reaches).

    PROVISION RULE: existing centres serve the existing population and do NOT count as provision
    for new development — new growth hugging an existing town must still earn its own centre, or
    it would sprawl centre-free along existing fabric. ``anchors`` (station-anchored centres,
    a subset of ``fixed``) are new provision and DO count: they are grown and sized by this
    pipeline like any new centre, only their location is pinned. The one exception is ``infill``
    (a bool mask of sub-threshold attached infill, judged by the caller): those few cells are
    served by an existing centre within the walk when one exists, so a scrap of infill does not
    force an orphan centre of its own.
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

    # A single centre's walk field depends only on its coordinate, and the Lloyd/add/cull
    # loops ask for the same coordinates thousands of times (and, via select_plan's shared
    # cache, across every ensemble member). Caching fields per coordinate is what makes a
    # many-centre refinement affordable: measured 42 s -> ~2 s per member at ~80 centres.
    cache = walk_cache if walk_cache is not None else {}

    def centre_field(c) -> np.ndarray:
        field = cache.get(c)
        if field is None:
            field = walk(onehot([c]))
            cache[c] = field
        return field

    col_cache: dict = {}

    def centre_col(c) -> np.ndarray:
        # the placement stack slices each centre's field by new_built every Lloyd iteration;
        # the slice depends only on the coordinate, so cache it per refinement call
        col = col_cache.get(c)
        if col is None:
            col = centre_field(c)[new_built]
            col_cache[c] = col
        return col

    def reach(cells):  # built cells within the centre WALK of any cell in `cells` (by the one metric)
        if not cells:
            return np.zeros((rows, cols), dtype=bool)
        if len(cells) == 1:
            return centre_field(tuple(cells[0])) <= max_distance_m
        return np.minimum.reduce([centre_field(tuple(c)) for c in cells]) <= max_distance_m

    new = [_nearest_built(new_built, int(y), int(x)) for y, x in seeds]

    # settlement components for the anchor invariant: label contiguous built clusters, note which
    # contain new development (those must stay anchored) and which hold a fixed/existing centre
    comps = _components(built)
    comp_label = np.full((rows, cols), -1, dtype=int)
    for i, comp in enumerate(comps):
        for y, x in comp:
            comp_label[y, x] = i

    # Distance/reach to the nearest FIXED (existing) centre, solved ONCE — fixed centres don't move,
    # and true-area centres are many cells, so collapsing them into a single field (rather than one
    # per cell) is what keeps this single-threaded post-processing affordable.
    fixed_field = walk(onehot(fixed)) if fixed else np.full((rows, cols), np.inf)
    fixed_col = fixed_field[new_built]
    comp_col = comp_label[new_built]  # settlement id of every new home (placement is settlement-local)
    # station anchors are the only fixed centres that count as provision for NEW development
    anchors = [(int(y), int(x)) for y, x in (anchors or [])]
    anchor_reach = (
        (walk(onehot(anchors)) <= max_distance_m) if anchors else np.zeros((rows, cols), dtype=bool)
    )
    # sub-threshold attached infill within the walk of an existing centre is served by that
    # centre (the provision-rule exception above); elsewhere infill needs provision like any home
    infill_ok = (
        (np.asarray(infill, dtype=bool) & (fixed_field <= max_distance_m))
        if infill is not None
        else np.zeros((rows, cols), dtype=bool)
    )
    # every contiguous settlement with non-infill new development must host its own centre:
    # demand pools across green for viability, but a blob justifies itself with an attached
    # centre, so placement seeds one wherever a blob lacks one (the caller's viability cut
    # then decides whether it stands)
    needs_anchor = {
        i for i, comp in enumerate(comps)
        if any(new_built[y, x] and not infill_ok[y, x] for y, x in comp)
    }
    fixed_comps = {int(comp_label[y, x]) for y, x in fixed if 0 <= y < rows and 0 <= x < cols} - {-1}

    def covered(centres):  # new homes within the walk of a NEW centre, an anchor, or (infill) an existing centre
        return anchor_reach | infill_ok | reach(centres)

    def lloyd(centres):  # re-position each new centre to the INTERIOR of the NEW homes it serves
        if not centres:
            return centres
        member_mask = np.zeros((rows, cols), dtype=bool)
        for _ in range(8):
            # column 0 = nearest fixed centre; columns 1.. = each new centre (single-source, cached)
            stack = np.column_stack([fixed_col] + [centre_col(tuple(c)) for c in centres])
            nearest = np.argmin(stack, axis=1)
            within = stack.min(axis=1) <= max_distance_m  # homes beyond the walk of every centre pull no one
            moved = False
            for j in range(len(centres)):
                # SETTLEMENT-LOCAL placement: only homes in the centre's own contiguous
                # built component position it. Walks traverse green, so a catchment can
                # span a green gap to a neighbouring cluster; letting those homes pull
                # the centre drags it to a periphery cell facing the green. Cross-gap
                # homes still COUNT as served in scoring — they just don't steer placement.
                members = (nearest == 1 + j) & within
                members &= comp_col == int(comp_label[centres[j][0], centres[j][1]])
                if not members.any():
                    continue
                cy = int(round(hy[members].mean()))
                cx = int(round(hx[members].mean()))
                if 0 <= cy < rows and 0 <= cx < cols and new_built[cy, cx]:
                    pt = (cy, cx)  # centroid is on built: keep it — even spread, so coverage holds
                else:
                    # the centroid fell OFF the development (a concave / ring / multi-blob catchment, where
                    # _nearest_built would snap it onto an edge): place at the catchment's deepest INTERIOR
                    # instead, so the centre lands on built and central to it rather than on a rim or in a gap
                    member_mask.fill(False)
                    member_mask[hy[members], hx[members]] = True
                    pt = _interior_point(member_mask) or _nearest_built(new_built, cy, cx)
                if pt != centres[j]:
                    centres[j] = pt
                    moved = True
            if not moved:
                break
        return centres

    def fill_gaps(centres):
        # Every new home beyond the walk of every new centre and anchor forces a centre in the
        # gap — the walk constraint is hard, however few homes a gap holds. Existing centres
        # suppress an addition only for sub-threshold attached infill (the provision-rule
        # exception above). The centre is proposed from WITHIN
        # the densest underserved cluster, not merely near it: a box-near cell may be unable to
        # actually reach across a barrier. Each addition covers at least its own cell, so this
        # terminates with full coverage.
        while True:
            underserved = new_built & ~covered(centres)
            if not underserved.any():
                return centres
            gain = np.where(underserved, _box_sum(underserved.astype(np.float64), r), -1.0)
            y, x = divmod(int(np.argmax(gain)), cols)
            centres.append((int(y), int(x)))

    def guarded_lloyd(centres):
        # polish placement without breaking the walk constraint: coverage is full before the
        # call, and a repositioning that would strand any home is discarded wholesale
        moved = lloyd(list(centres))
        if not (new_built & ~covered(moved)).any():
            return moved
        return centres

    if reposition:
        new = lloyd(new)  # free first pass: the baseline is the seeded placement, gaps are filled next
    new = fill_gaps(new)
    if reposition:
        new = guarded_lloyd(new)

    def is_last_anchor(j, centres):
        cj = int(comp_label[centres[j][0], centres[j][1]])
        if cj < 0 or cj not in needs_anchor or cj in fixed_comps:
            return False
        return not any(int(comp_label[c[0], c[1]]) == cj for k, c in enumerate(centres) if k != j)

    if minimise_count:
        # Remove centres one at a time for as long as the walk constraint and the anchor
        # invariant hold. A removal keeps every home covered exactly when the centre covers no
        # new cell uniquely (its whole catchment is also someone else's), so zero-unique centres
        # are the removable ones; repositioning after each removal lets the survivors spread and
        # exposes the next redundancy, converging on the fewest centres the constraint permits.
        while new:
            masks = [reach([c]) for c in new]
            counts = np.sum(masks, axis=0)
            unique = [
                int((new_built & masks[j] & (counts == 1) & ~anchor_reach & ~infill_ok).sum())
                for j in range(len(new))
            ]
            cullable = [j for j in range(len(new)) if unique[j] == 0 and not is_last_anchor(j, new)]
            if not cullable:
                break
            new.pop(cullable[0])
            new = guarded_lloyd(new)

    # seed a centre at the interior of any blob still lacking one, so the blob can justify
    # itself; the caller's viability cut removes it (and then the blob) if demand falls short
    anchored = fixed_comps | {int(comp_label[y, x]) for y, x in new}
    for i in sorted(needs_anchor - anchored):
        mask = np.zeros((rows, cols), dtype=bool)
        for y, x in comps[i]:
            if new_built[y, x]:
                mask[y, x] = True
        pt = _interior_point(mask)
        if pt is not None:
            new.append((int(pt[0]), int(pt[1])))
    return new


# New centres are grown into AREAS (not single cells) sized by the POPULATION they serve — a town
# centre spans many cells, a local centre a few. Mixed-use: the cells stay built/homes, just
# designated centre as well. Existing/true-area centres come in pre-sized from the input.
# The provision is a per-person rule of thumb: m² of centre land (retail, services, civic) per
# resident in the catchment. 20 m²/person matches the previous 8%-of-homes sizing at the default
# densities, but adapts when density changes — denser catchments get bigger centres.
CENTRE_M2_PER_PERSON = 20.0
# Fallback per-cell population estimate for new development: the probability-weighted mean of the
# dialog's default tiers (0.2*6000 + 0.3*3000 + 0.5*1500). Existing fabric carries NO population
# anywhere — it is assumed served by its own centres, so only new development is ever counted.
MEAN_NEW_DENSITY_KM2 = 2850.0
CENTRE_AREA_MAX = 100  # cap so a single centre can't sprawl without bound
# Contiguity floor: however coarse the grid, a settlement (and so any mixed-use centre attached to
# it) must span at least this many contiguous cells, or it reverts to green. Keeps the population-based
# minimum-settlement dial resolution-independent.
MIN_SETTLEMENT_CELLS = 4


def _grow_blob(start, target, built, claimed):
    """BFS outward from ``start`` over unclaimed built cells, up to ``target`` cells. Returns a set."""
    rows, cols = built.shape
    sy, sx = start
    if not (0 <= sy < rows and 0 <= sx < cols) or not built[sy, sx] or start in claimed:
        return set()
    out = {start}
    queue = deque([start])
    while queue and len(out) < target:
        y, x = queue.popleft()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                cell = (y + dy, x + dx)
                if (
                    0 <= cell[0] < rows
                    and 0 <= cell[1] < cols
                    and built[cell]
                    and cell not in out
                    and cell not in claimed
                ):
                    out.add(cell)
                    queue.append(cell)
                    if len(out) >= target:
                        return out
    return out


def _grow_centres(points, fixed, built, walk, cell_pop, m2_per_person, cell_area_m2,
                  max_area=CENTRE_AREA_MAX, growable=None):
    """Grow each new centre POINT into a contiguous AREA on built land, sized by the POPULATION it is
    the nearest centre to (its Voronoi catchment within a walk) at ``m2_per_person`` of centre land
    per resident — like a real centre, bigger where it serves more people. Mixed-use: cells stay
    built/homes, just designated centre; existing/fixed centres are left intact and never grown into.
    ``growable`` bounds where a centre may spread, and defaults to all built land. Callers pass the
    NEW development alone: turning existing housing into a mixed-use centre is redevelopment, which
    this model does not do, and it would also count those residents as new.
    ``cell_pop`` is the per-cell population estimate. Returns the set of (row, col) centre cells.
    """
    points = [(int(y), int(x)) for y, x in points]
    if not points:
        return set()
    built = np.asarray(built, dtype=bool)
    rows, cols = built.shape

    def onehot(cells):
        m = np.zeros((rows, cols), dtype=bool)
        for y, x in cells:
            m[y, x] = True
        return m

    # size each centre by the homes it is the NEAREST centre to (no double counting across centres)
    fixed_at_built = walk(onehot(fixed))[built] if fixed else np.full(int(built.sum()), np.inf)
    stack = np.column_stack([fixed_at_built] + [walk(onehot([p]))[built] for p in points])
    nearest = np.argmin(stack, axis=1)  # 0 = nearest fixed centre; 1.. = the new points
    within = np.isfinite(stack.min(axis=1))
    pop_b = np.asarray(cell_pop, dtype=float)[built]
    targets = [
        max(1, min(int(max_area),
                   round(float(pop_b[(nearest == 1 + j) & within].sum()) * m2_per_person / cell_area_m2)))
        for j in range(len(points))
    ]
    claimed = {(int(y), int(x)) for y, x in fixed}  # never grow onto existing/fixed centres
    grown: set = set()
    for j in sorted(range(len(points)), key=lambda k: -targets[k]):  # biggest centres claim first
        blob = _grow_blob(points[j], targets[j],
                          built if growable is None else growable, claimed)
        grown |= blob
        claimed |= blob
    return grown


def _components(mask, queen=True):
    """Connected components of a bool mask, as a list of (row, col) cell lists.

    ``queen`` (the default) counts diagonal neighbours as connected; ``queen=False`` counts
    rook (edge-sharing) neighbours only, so a diagonal corner touch does not connect."""
    mask = np.asarray(mask, dtype=bool)
    labels = _label_components(mask, queen=queen)
    if labels is not None:
        comps: list[list[tuple[int, int]]] = [[] for _ in range(int(labels.max()))]
        for y, x in zip(*np.nonzero(mask)):
            comps[labels[y, x] - 1].append((int(y), int(x)))
        return comps
    rows, cols = mask.shape
    steps = (
        [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        if queen
        else [(-1, 0), (0, -1), (0, 1), (1, 0)]
    )
    seen = np.zeros((rows, cols), dtype=bool)
    comps = []
    for sy in range(rows):
        for sx in range(cols):
            if mask[sy, sx] and not seen[sy, sx]:
                comp = []
                queue = deque([(sy, sx)])
                seen[sy, sx] = True
                while queue:
                    y, x = queue.popleft()
                    comp.append((y, x))
                    for dy, dx in steps:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < rows and 0 <= nx < cols and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            queue.append((ny, nx))
                comps.append(comp)
    return comps


# --- plan evaluator: the "isobenefit" objective, method-agnostic ----------------
#
# Scores any PLAN_* layout on the same yardstick so extraction methods can be
# compared. The standard is a THRESHOLD, not a gradient: being within a walk
# (<= max_distance) of an amenity counts as served, whether 80 m or 800 m. So the
# score is COVERAGE — is each home within a walk of a centre and of a real park? —
# and the equity headline is simply how many homes are left out.


def _walk_distance(
    targets: np.ndarray, granularity_m: float, max_distance_m: float, blocked: np.ndarray | None = None
) -> np.ndarray:
    """Walking distance (metres) from every cell to the nearest target cell.

    A bounded multi-source Dijkstra, queen moves with diagonal cost
    ``sqrt(2) * granularity``. Cells further than ``max_distance_m`` from any target stay
    ``inf``. If ``blocked`` is given, the walk cannot enter those cells (it routes around
    them) — used so distances don't cross the green network.

    The engine computes this field 50-100x faster (and without holding the GIL), so it
    is preferred whenever importable; the Python loop below is the exact-parity fallback
    for engines predating the engine helpers (< 0.12.17).
    """
    try:
        import isobenefit

        return isobenefit.walk_distance(
            np.ascontiguousarray(targets, dtype=bool),
            float(granularity_m),
            float(max_distance_m),
            None if blocked is None else np.ascontiguousarray(blocked, dtype=bool),
        )
    except (ImportError, AttributeError, TypeError):
        pass
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
                # a diagonal step may not squeeze between two blocked cells (mirrors the engine)
                if dy and dx and blocked is not None and blocked[y, nx] and blocked[ny, x]:
                    continue
                nd = d + w * g
                if nd <= max_distance_m and nd < dist[ny, nx]:
                    dist[ny, nx] = nd
                    heapq.heappush(heap, (nd, ny, nx))
    return dist


def evaluate_plan(
    plan: np.ndarray,
    granularity_m: float,
    max_distance_m: float,
    transit_stops: np.ndarray | None = None,
    centre_distance_m: float | None = None,
    green_distance_m: float | None = None,
    new_density_km2: float = MEAN_NEW_DENSITY_KM2,
    existing_green: np.ndarray | None = None,
    target_population: float | None = None,
    min_park_area_m2: float | None = None,
    stop_catchment_m: float | None = None,
    transit_hubs: np.ndarray | None = None,
    hub_catchment_m: float | None = None,
) -> dict:
    """Score a recommended plan by COVERAGE — who is within a walk of what.

    A home is *served* if it is within the walk of both a centre and a real park
    (within the walk = okay; not a gradient). Only green patches of at least the
    minimum park area (``min_park_area_m2``, 2 ha default) qualify as parks,
    matching the growth rules' park definition; an area of zero lets any green
    qualify. The headline metrics cover NEW homes only: the growth
    rules guarantee every new home a centre within the centre walk and a park
    within the green walk, so ``served_coverage`` doubles as the check that
    post-processing preserved the guarantee. Existing fabric carries no guarantee
    (it is assumed served by its own centres, and the growth rules only promise
    never to worsen its park access), so it appears solely in the
    ``incl_existing`` blend and the supply-side ratios. The split follows the
    existing-* plan codes, so pass plans through ``_mark_existing`` first; an
    untagged plan reads as all new. Returns shares in ``[0, 1]``:

    - ``centre_coverage`` / ``green_coverage`` — share of new homes within a walk of each;
    - ``served_coverage`` — share of new homes within a walk of *both* (the headline);
    - ``unserved_fraction`` — share of new homes left out (expected 0 by construction);
    - ``served_coverage_incl_existing`` — the same test over every home in the
      window, existing fabric included (descriptive only: no guarantee attaches);
    - ``existing_served_coverage`` and the existing walk means — the existing
      fabric alone, as context for comparison with the new development (present
      only when the plan carries existing homes);
    - ``access_cost`` — the **selection metric** (lower better): average walk (m) to
      amenities over every new home, with unreachable homes counted at a penalty and,
      when ``target_population`` is given, the unhoused remainder of the target counted
      at the penalty distance too (so housing fewer people never wins selection).
      ``centre_access`` / ``green_access`` are the plain per-amenity averages over new
      homes, without the shortfall term;
    - ``centre_walk_mean`` / ``green_walk_mean`` — mean walk of served new homes to each;
    - ``compactness`` — share of built neighbours that are also built (anti-sprawl).

    If ``transit_stops`` (a bool mask of corridor cells: bus stops and drawn routes) or
    ``transit_hubs`` (a bool mask of hub cells: stations and designated points) is given,
    also reports ``transit_coverage`` / ``transit_access`` / ``transit_walk_mean`` — new
    homes' walkable access to a transit anchor. Each anchor kind has its own catchment
    (``stop_catchment_m`` for corridors, ``hub_catchment_m`` for hubs, each falling back
    to ``max_distance_m``): ``transit_coverage`` counts homes within either, matching the
    catchment the corridor preference acts on during growth, and the walk figures measure
    to the nearest anchor of any kind. These are reported only; transit shapes the runs
    through the growth rules, never the selection.
    """
    new_homes = np.isin(plan, (PLAN_BUILT, PLAN_CENTRE))
    exist_homes = np.isin(plan, (PLAN_EXIST_BUILT, PLAN_EXIST_CENTRE))
    built = new_homes | exist_homes
    n_built = int(built.sum())
    if n_built == 0:
        return {"built_cells": 0}

    green_mask = plan == PLAN_GREEN
    # only real parks count, matching the growth rules
    green_mask = _keep_large_components(green_mask, park_threshold_cells(granularity_m, min_park_area_m2))

    # Walking distances: the bounded grid walk (queen moves, barriers block) — the same
    # metric the growth rules use, so growth and scoring always agree.
    # Split walks: a home is near a centre within ``centre_distance_m`` and near green within
    # ``green_distance_m`` (each defaults to the shared ``max_distance_m``). The distance field is
    # bounded at the larger of the two; coverage compares against each amenity's own threshold.
    centre_distance_m = max_distance_m if centre_distance_m is None else float(centre_distance_m)
    green_distance_m = max_distance_m if green_distance_m is None else float(green_distance_m)
    field_bound = max(centre_distance_m, green_distance_m)

    walk_blocked = plan == PLAN_NONE  # unbuildable land and outside the extents

    def _dist(mask):
        return _walk_distance(mask, granularity_m, field_bound, blocked=walk_blocked)

    d_cent_field = _dist(np.isin(plan, (PLAN_CENTRE, PLAN_EXIST_CENTRE)))
    d_green_field = _dist(green_mask)

    def _near(mask):
        dc, dg = d_cent_field[mask], d_green_field[mask]
        return dc, dg, dc <= centre_distance_m, dg <= green_distance_m

    d_cent, d_green, near_cent, near_green = _near(new_homes)
    served = near_cent & near_green
    _dc_all, _dg_all, near_cent_all, near_green_all = _near(built)
    served_all = near_cent_all & near_green_all

    # selection metric: average walk to amenities over every NEW home, with anyone
    # who can't reach within the limit counted at a penalty distance (so a plan
    # can't score well by abandoning the fringe). When ``target_population`` is
    # given, the unhoused remainder of the target enters the average as homes at
    # the penalty distance, so a run cannot win selection by housing fewer
    # people; with the target met this reduces to the plain mean walk.
    has_new = bool(new_homes.any())
    n_new = int(new_homes.sum())
    cell_km2 = granularity_m * granularity_m / 1e6
    population = n_new * new_density_km2 * cell_km2
    if has_new:
        centre_access = float(np.where(near_cent, d_cent, 2.0 * centre_distance_m).mean())
        green_access = float(np.where(near_green, d_green, 2.0 * green_distance_m).mean())
    else:  # a degenerate plan with no new development: worst-case penalty
        centre_access = 2.0 * centre_distance_m
        green_access = 2.0 * green_distance_m
    shortfall_cells = 0.0
    if target_population and new_density_km2:
        shortfall_cells = max(0.0, float(target_population) - population) / (new_density_km2 * cell_km2)
    denom = n_new + shortfall_cells
    if denom:
        sel_centre = (centre_access * n_new + 2.0 * centre_distance_m * shortfall_cells) / denom
        sel_green = (green_access * n_new + 2.0 * green_distance_m * shortfall_cells) / denom
    else:
        sel_centre, sel_green = 2.0 * centre_distance_m, 2.0 * green_distance_m
    access_cost = 0.5 * (sel_centre + sel_green)

    rows, cols = plan.shape
    adj = 0
    for dy, dx in ((1, 0), (0, 1)):
        a = built[: rows - dy, : cols - dx] & built[dy:, dx:]
        adj += 2 * int(a.sum())  # each shared edge counts for both cells

    # supply-side efficiency: how well-used each centre / unit of green is, over the
    # whole town — existing homes use new centres and parks too.
    n_centres = int(np.isin(plan, (PLAN_CENTRE, PLAN_EXIST_CENTRE)).sum())
    n_green = int(green_mask.sum())

    metrics = {
        "built_cells": n_built,
        "new_cells": int(new_homes.sum()),
        "centre_coverage": float(near_cent.mean()) if has_new else 0.0,
        "green_coverage": float(near_green.mean()) if has_new else 0.0,
        "served_coverage": float(served.mean()) if has_new else 0.0,
        "unserved_fraction": float((~served).mean()) if has_new else 0.0,
        "served_coverage_incl_existing": float(served_all.mean()),
        "access_cost": access_cost,  # the selection metric (lower better; includes the shortfall term)
        "centre_access": centre_access,  # avg walk to a centre over new homes (penalised)
        "green_access": green_access,  # avg walk to green over new homes (penalised)
        "centre_walk_mean": float(d_cent[near_cent].mean()) if near_cent.any() else math.inf,
        "green_walk_mean": float(d_green[near_green].mean()) if near_green.any() else math.inf,
        "compactness": adj / (4.0 * n_built),
        "centre_efficiency": float(near_cent_all.sum()) / n_centres if n_centres else 0.0,  # homes served per centre
        "green_efficiency": float(near_green_all.sum()) / n_green if n_green else 0.0,  # homes served per green cell
    }
    # the existing fabric on its own, as context for comparison with the new
    # development: no guarantee attaches, and nothing selects on these
    if exist_homes.any():
        d_ce, d_ge, near_ce, near_ge = _near(exist_homes)
        served_exist = near_ce & near_ge
        metrics["existing_served_coverage"] = float(served_exist.mean())
        metrics["existing_centre_walk_mean"] = float(d_ce[near_ce].mean()) if near_ce.any() else math.inf
        metrics["existing_green_walk_mean"] = float(d_ge[near_ge].mean()) if near_ge.any() else math.inf
    # per-person provision (rule-of-thumb readouts): NEW amenity over NEW population only. Existing
    # fabric carries no population (it is assumed served by its own centres), so the honest ratio is
    # what the plan ADDS — new mixed-use centre land, and new green — per new resident. Pass
    # ``existing_green`` (a bool mask) to exclude pre-existing green from the provision numerator;
    # without it, all qualifying green counts as new. NB these depend on the existing-* tagging,
    # unlike the coverage metrics above (evaluate marked plans for honest splits).
    n_new_centres = int((plan == PLAN_CENTRE).sum())
    new_green_mask = green_mask if existing_green is None else green_mask & ~np.asarray(existing_green, dtype=bool)
    n_new_green = int(new_green_mask.sum())
    metrics["population"] = population
    metrics["centre_m2_per_person"] = n_new_centres * cell_km2 * 1e6 / population if population else 0.0
    metrics["green_m2_per_person"] = n_new_green * cell_km2 * 1e6 / population if population else 0.0

    # transit access — reported only; transit shapes growth, never selection. Corridors
    # and hubs each carry their own catchment; coverage counts homes within either, and
    # the walk figures measure to the nearest anchor of any kind.
    if has_new:
        d_stop = d_hub = None
        near = None
        if transit_stops is not None and np.asarray(transit_stops, dtype=bool).any():
            d_stop = _dist(np.asarray(transit_stops, dtype=bool))[new_homes]
            near = d_stop <= (stop_catchment_m if stop_catchment_m is not None else max_distance_m)
        if transit_hubs is not None and np.asarray(transit_hubs, dtype=bool).any():
            d_hub = _dist(np.asarray(transit_hubs, dtype=bool))[new_homes]
            near_hub = d_hub <= (hub_catchment_m if hub_catchment_m is not None else max_distance_m)
            near = near_hub if near is None else (near | near_hub)
        if near is not None:
            d_any = d_stop if d_hub is None else (d_hub if d_stop is None else np.minimum(d_stop, d_hub))
            metrics["transit_coverage"] = float(near.mean())
            metrics["transit_access"] = float(np.where(near, d_any, 2.0 * max_distance_m).mean())
            metrics["transit_walk_mean"] = float(d_any[near].mean()) if near.any() else math.inf

    return metrics


def rejected_development(pre_plan, option_plan):
    """Diagnostic layer connecting the raw grown state to the plan options.

    New development present in the raw state (``pre_plan``, tagged with the existing-*
    codes) but absent from an option is the rejected development. Under centre-first
    viability there is one reason: no centre with threshold demand within its walk could
    reach it. Returns ``(raster, counts)``: a uint8 grid of PLAN_REJECT_UNSERVABLE
    (0 elsewhere) and the rejected cell count.
    """
    pre_plan = np.asarray(pre_plan)
    option_plan = np.asarray(option_plan)
    pre_new = np.isin(pre_plan, (PLAN_BUILT, PLAN_CENTRE))
    opt_built = np.isin(
        option_plan, (PLAN_BUILT, PLAN_CENTRE, PLAN_EXIST_BUILT, PLAN_EXIST_CENTRE)
    )
    rejected = pre_new & ~opt_built
    out = np.where(rejected, PLAN_REJECT_UNSERVABLE, 0).astype(np.uint8)
    return out, {"cells": int(rejected.sum())}


def audit_centres(plan, granularity_m, max_distance_m, centre_min_settlement=0):
    """Per-centre-AREA effectiveness audit, by the one distance model (the bounded grid walk).
    Centres are areas (existing true-area + grown new ones), so each
    record is a connected component, with its ``cells`` (area), how many built cells it **serves**
    (within a walk) and the **mean walk** to them (low = well-centred; few served = an ineffective
    centre on a thin/edge catchment).

    ``centre_min_settlement`` is the service viability threshold in cells (the population a
    minimal local centre needs, converted via the mean density). When given, each record carries
    ``new_served`` (NEW built cells within the walk, the demand basis for viability, since
    existing homes are committed to their own centres) and ``viable`` (new catchment at or
    above the threshold), and the summary counts the new centres below it. Post-processing
    enforces viability centre-first (station anchors exempt), so on its outputs the audit
    verifies that the count is zero.

    Run after each plan so weak centres are visible and the cull threshold can be tuned to evidence
    rather than by eye. Returns ``{"centres": [...weakest first...], "summary": {...}}``.
    """
    plan = np.asarray(plan)
    walk_blocked = plan == PLAN_NONE

    def walk(mask):
        return _walk_distance(mask, granularity_m, max_distance_m, blocked=walk_blocked)

    built = np.isin(plan, (PLAN_BUILT, PLAN_CENTRE, PLAN_EXIST_BUILT, PLAN_EXIST_CENTRE))
    new_homes = np.isin(plan, (PLAN_BUILT, PLAN_CENTRE))
    records = []
    # one record per centre AREA (connected component), not per cell — centres are areas now, so
    # per-cell would massively over-count
    for comp in _components(np.isin(plan, (PLAN_CENTRE, PLAN_EXIST_CENTRE))):
        one = np.zeros(plan.shape, dtype=bool)
        for y, x in comp:
            one[y, x] = True
        d = walk(one)  # walk from the whole centre area to all cells
        served_mask = built & np.isfinite(d)
        served = int(served_mask.sum())
        new_served = int((new_homes & np.isfinite(d)).sum())
        records.append(
            {
                "row": int(round(sum(c[0] for c in comp) / len(comp))),
                "col": int(round(sum(c[1] for c in comp) / len(comp))),
                "cells": len(comp),  # the centre's area (a town centre spans many cells)
                "served": served,  # built cells within a walk — the catchment it serves
                "new_served": new_served,  # the demand basis for viability
                "mean_dist_m": float(d[served_mask].mean()) if served else math.inf,  # avg walk to them
                "existing": any(plan[y, x] == PLAN_EXIST_CENTRE for y, x in comp),
                "viable": new_served >= centre_min_settlement,
            }
        )
    records.sort(key=lambda r: r["served"])  # weakest first, so the audit surfaces the dubious ones
    served = np.array([r["served"] for r in records], dtype=float)
    finite_means = [r["mean_dist_m"] for r in records if math.isfinite(r["mean_dist_m"])]
    # split existing (from the input centres layer) vs new (placed by the model) — so a suspicious
    # centre can be traced to the data or to the optimiser
    new_served = np.array([r["served"] for r in records if not r["existing"]], dtype=float)
    summary = {
        "n_centres": len(records),
        "n_new": int((~np.array([r["existing"] for r in records])).sum()) if records else 0,
        "n_existing": int(np.array([r["existing"] for r in records]).sum()) if records else 0,
        "served_min": int(served.min()) if len(served) else 0,
        "served_median": int(np.median(served)) if len(served) else 0,
        "served_max": int(served.max()) if len(served) else 0,
        "new_served_min": int(new_served.min()) if len(new_served) else 0,
        "new_served_median": int(np.median(new_served)) if len(new_served) else 0,
        "mean_dist_median_m": float(np.median(finite_means)) if finite_means else math.inf,
        "viability_threshold_cells": int(centre_min_settlement),
        "n_new_below_viability": sum(1 for r in records if not r["existing"] and not r["viable"]),
    }
    return {"centres": records, "summary": summary}


# --- recommended-plan post-processing -------------------------------------------
#
# Turns a single CA run into a recommended plan: prune failed-satellite specks, then re-place the
# centres (re-centre on their development, add where under-served, cull redundant, grow to area). The
# CA's own green network is kept as-is — the CA already preserves green to the minimum span during
# growth, so the plan does NOT re-carve parks.


def optimise_plan(
    plan: np.ndarray,
    granularity_m: float,
    max_distance_m: float,
    existing_centres=None,
    existing_built=None,
    ca_centres=None,
    centre_mode: str = "placed",
    centre_anchors=None,
    centre_distance_m: float | None = None,
    green_distance_m: float | None = None,
    centre_m2_per_person: float = CENTRE_M2_PER_PERSON,
    new_density_km2: float = MEAN_NEW_DENSITY_KM2,
    centre_min_settlement: int = 3,
    prune_islands: bool = True,
    walk_cache: dict | None = None,
) -> np.ndarray:
    """Post-process a single CA run's plan into a recommended plan: prune failed-satellite specks,
    handle the centres per ``centre_mode``, and grow each centre to
    an area sized by the residents it serves. The CA's green network is kept as-is. Returns a new
    plan. Every mode keeps the walk constraint the growth rules enforced: every new home within
    the centre walk of a centre.

    ``centre_mode`` picks the centre treatment:

    - ``"grown"`` — centres stay exactly where the run grew them (locations and extent); each
      still grows to the area its catchment warrants (never shrinks).
    - ``"placed"`` — the run's centres are re-positioned central to the new homes they serve, and
      centres are added where new development lacks provision within the walk; none are removed.
    - ``"minimal"`` — as ``"placed"``, then centres are removed one at a time for as long as every
      home keeps a centre within the walk: the fewest centres the constraint permits.

    ``existing_built`` is a bool mask of cells already developed before the simulation; those are
    **frozen** (never pruned) and tagged distinctly downstream.

    Centre/green walks are split: ``centre_distance_m`` / ``green_distance_m`` (each defaulting to
    ``max_distance_m``) are the walk thresholds for centre vs green coverage. Each centre grows to
    ``centre_m2_per_person`` of centre land per resident it serves (population estimated per cell
    from ``new_density_km2``; existing fabric counts zero — only new residents size a centre).
    ``centre_min_settlement`` is the minimum settlement size (in cells): a detached new cluster
    below it is pruned, and so is new growth in a sub-threshold settlement of existing fabric
    that has no centre of its own (an outlying hamlet the model should not grow). A new cluster
    grown against a centred town is infill anchored on that town's centres and stays new
    development; sub-threshold infill within the walk of an existing centre is served by it and
    earns no centre of its own.
    """
    if centre_mode not in ("grown", "placed", "minimal"):
        raise ValueError(f"unknown centre_mode: {centre_mode!r}")
    plan = plan.copy()
    g = float(granularity_m)
    # Split walks: centres and green each have their own walk threshold (both default to the
    # shared max_distance). The distance FIELD is bounded at the larger of the two.
    centre_distance_m = max_distance_m if centre_distance_m is None else float(centre_distance_m)
    green_distance_m = max_distance_m if green_distance_m is None else float(green_distance_m)
    field_bound = max(centre_distance_m, green_distance_m)
    rows, cols = plan.shape
    # ONE distance model: the bounded grid walk, the same metric the growth rules use;
    # unbuildable land and cells outside the extents block it, exactly as in growth.
    walk_blocked = plan == PLAN_NONE

    def walk(mask):
        return _walk_distance(mask, g, field_bound, blocked=walk_blocked)

    # Existing development is frozen: never pruned, and the small-settlement cleanup never touches it.
    frozen = np.zeros(plan.shape, dtype=bool)
    if existing_built is not None:
        frozen |= np.asarray(existing_built, dtype=bool)

    # Viability is judged centre-first inside the loop below (the 2026-08-21 doctrine):
    # catchments follow the walk and cross green gaps, so nearby settlements pool demand as
    # a constellation, centres below threshold demand are cut, and homes no viable centre
    # reaches revert to nature. No contiguity prune, hamlet rule, or per-settlement anchor
    # remains; the threshold is floored (MIN_SETTLEMENT_CELLS) so a coarse grid cannot
    # collapse it.
    centre_min_settlement = max(int(centre_min_settlement), MIN_SETTLEMENT_CELLS)
    n_built = int(((plan == PLAN_BUILT) | (plan == PLAN_CENTRE)).sum())
    if n_built == 0:
        return plan

    # Centres: keep the existing ones; take the simulation's grown centres (``ca_centres``), each
    # connected blob of centre cells collapsing to ONE centre (seeded at the blob's interior).
    # With no ca_centres (direct calls) fall back to proximity seeding on the finished fabric.
    #
    # The preparation below runs inside the SERVICE VIABILITY loop. The placed arrangement is
    # checked against the viability threshold; new growth that only an unviable centre could
    # serve is not viable development and reverts to green, and the preparation repeats on the
    # reduced fabric. The loop is fabric-level, so all three centre options share its outcome.
    # Station anchors are pinned and exempt, and infill served by an existing centre never
    # forces a centre of its own.
    exist_cell_set = {(int(ey), int(ex)) for ey, ex in (existing_centres or [])}
    while True:
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
        exclude = exist_cell_set | set(anchor_on_built)
        if ca_centres is None:
            seed_cells = [
                s
                for s in _seed_centres_proximity(built, granularity_m, centre_distance_m, existing_centres)
                if s not in exclude
            ]
        else:
            seed_cells = [(int(y), int(x)) for y, x in ca_centres if (int(y), int(x)) not in exclude]
        seed_cells = [(y, x) for y, x in seed_cells if 0 <= y < rows and 0 <= x < cols and plan[y, x] == PLAN_BUILT]
        seed_mask = np.zeros_like(built)
        for y, x in seed_cells:
            seed_mask[y, x] = True
        seed_areas = _components(seed_mask) if seed_mask.any() else []
        seed_points = []
        for comp in seed_areas:
            m = np.zeros_like(built)
            for y, x in comp:
                m[y, x] = True
            pt = _interior_point(m)
            seed_points.append((int(pt[0]), int(pt[1])) if pt is not None else comp[0])

        # sub-threshold attached infill: a contiguous NEW cluster below the viability threshold
        # inside a settlement that holds an existing centre or station anchor. The provision-rule
        # exception in _refine_centres lets that centre serve these cells. Existing FABRIC alone
        # earns no exemption: a cluster grown against a centre-less farmstead or hamlet must
        # justify a centre like any other settlement, or revert.
        infill = np.zeros(plan.shape, dtype=bool)
        if fixed_on_built and new_built.any():
            comp_lbl = np.full(plan.shape, -1, dtype=int)
            for i, comp in enumerate(_components(built)):
                for y, x in comp:
                    comp_lbl[y, x] = i
            centred_ids = {int(comp_lbl[y, x]) for y, x in fixed_on_built}
            for cluster in _components(new_built):
                y0, x0 = cluster[0]
                if len(cluster) < centre_min_settlement and int(comp_lbl[y0, x0]) in centred_ids:
                    for y, x in cluster:
                        infill[y, x] = True
        # a CA centre planted on an infill scrap is an orphan: the scrap is served by the
        # existing centre nearby, so the seed is dropped rather than repositioned
        refine_seeds = [p for p in seed_points if not infill[p[0], p[1]]]

        if not new_built.any():
            placed_points = []
            break
        placed_points = _refine_centres(
            refine_seeds, fixed_on_built, built, new_built, granularity_m, centre_distance_m,
            walk=walk, anchors=anchor_on_built, walk_cache=walk_cache,
            minimise_count=False, infill=infill,
        )
        # CENTRE-FIRST SERVICE VIABILITY: a centre whose walk catchment holds less than the
        # threshold of demand is cut, and homes beyond the walk of every surviving centre
        # are not viably servable and revert to nature. Catchments cross green gaps, so
        # nearby settlements pool demand and stand or fall together as a constellation. One
        # pass is stable (every home in a surviving centre's catchment is covered by
        # definition, so no cut shrinks another catchment); the loop repeats only because
        # placement reruns on a reduced fabric, and the fabric only shrinks, so it
        # terminates.
        if not prune_islands:
            break
        vcache = walk_cache if walk_cache is not None else {}

        def _cfield(c, _cache=vcache, _shape=plan.shape):
            f = _cache.get(tuple(c))
            if f is None:
                m = np.zeros(_shape, dtype=bool)
                m[c[0], c[1]] = True
                f = walk(m)
                _cache[tuple(c)] = f
            return f

        anchor_cells = {tuple(a) for a in anchor_on_built}
        # demand counts NEW homes only: existing homes are assumed committed to their own
        # centres, so they do not subsidise a new centre's viability
        viable = [
            c for c in placed_points
            if tuple(c) in anchor_cells
            or int((new_built & (_cfield(c) <= centre_distance_m)).sum()) >= centre_min_settlement
        ]
        covered = np.zeros(plan.shape, dtype=bool)
        for c in list(viable) + anchor_on_built:
            covered |= _cfield(c) <= centre_distance_m
        if fixed_on_built:
            fm = np.zeros(plan.shape, dtype=bool)
            for y, x in fixed_on_built:
                fm[y, x] = True
            covered |= infill & (walk(fm) <= centre_distance_m)
        # each contiguous blob justifies itself with an attached surviving centre: demand
        # pools across green for a centre's catchment, but a blob without a viable centre
        # of its own (nor an existing centre or station anchor) is not a settlement, and
        # its new growth reverts; infill served by an existing centre is exempt
        centre_ok = np.zeros(plan.shape, dtype=bool)
        for y, x in list(viable) + anchor_on_built + fixed_on_built:
            centre_ok[y, x] = True
        unanchored = np.zeros(plan.shape, dtype=bool)
        for comp in _components(built):
            if not any(new_built[y, x] and not infill[y, x] for y, x in comp):
                continue
            if any(centre_ok[y, x] for y, x in comp):
                continue
            for y, x in comp:
                if new_built[y, x]:
                    unanchored[y, x] = True
        unservable = (new_built & ~covered) | unanchored
        if not unservable.any():
            placed_points = viable
            break
        plan[unservable] = PLAN_GREEN

    for ey, ex in fixed_on_built:
        plan[ey, ex] = PLAN_CENTRE
    exist_mask = (
        (np.asarray(existing_built, dtype=bool) & built)
        if existing_built is not None
        else np.zeros_like(built)
    )
    # existing fabric contributes NO population: centres are sized by the new residents they serve
    cell_pop = np.where(exist_mask, 0.0, new_density_km2) * (g * g / 1e6)
    if centre_mode == "grown":
        # Locations untouched: every grown centre cell stays a centre, and each area still
        # grows to the size its catchment warrants (mixed-use provision is a ratio, not a
        # location choice); growing only, never shrinking. Pruning is an edit like any other,
        # so where it removed a cluster's serving centre, a gap-filling addition restores the
        # walk constraint; nothing is repositioned or removed. Station anchors grow too.
        kept = [c for comp in seed_areas for c in comp]
        filled = _refine_centres(
            seed_points, fixed_on_built, built, new_built, granularity_m, centre_distance_m,
            walk=walk, anchors=anchor_on_built, walk_cache=walk_cache,
            minimise_count=False, infill=infill, reposition=False,
        )
        grow_points = list(dict.fromkeys(filled + anchor_on_built))
        grown = _grow_centres(grow_points, existing_on_built, built, walk, cell_pop,
                              centre_m2_per_person, g * g, growable=built & ~exist_mask)
        new_centres = list(dict.fromkeys(list(grown) + kept))
    else:
        if centre_mode == "placed":
            new_centres = placed_points  # the viability loop's converged placed arrangement
        else:
            # fewest centres: cull the converged viable set, never the raw seeds, so the
            # option starts from a plan every centre of which reaches threshold demand
            new_centres = _refine_centres(
                placed_points, fixed_on_built, built, new_built, granularity_m, centre_distance_m,
                walk=walk, anchors=anchor_on_built, walk_cache=walk_cache,
                minimise_count=True, infill=infill,
            )
        # Grow each placed centre into an AREA sized by the homes it serves (mixed-use, on built).
        # Station anchors grow too — a station should seed a real centre, not stay a lone cell — while
        # existing/true-area centres come pre-sized from the input and are left intact (claimed, not grown).
        grow_points = list(dict.fromkeys(new_centres + anchor_on_built))
        new_centres = _grow_centres(
            grow_points, existing_on_built, built, walk, cell_pop, centre_m2_per_person, g * g,
            growable=built & ~exist_mask,
        )
    for y, x in new_centres:
        plan[y, x] = PLAN_CENTRE
    return plan


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


def _state_to_plan(state, granularity_m, existing_green=None) -> np.ndarray:
    """Map a single run's final state to a PLAN_* layout: built/centre -> built (the
    optimiser re-places centres) and every nature cell -> green. Sub-park green stays
    green land: it is traversable and rendered, and the scoring's park filter alone
    decides what qualifies for green access (classifying it as PLAN_NONE would block
    walks the growth rules permitted)."""
    state = np.asarray(state)
    plan = np.zeros(state.shape, dtype=np.uint8)
    plan[(state == 1) | (state == 2)] = PLAN_BUILT
    plan[state == 0] = PLAN_GREEN
    if existing_green is not None:
        plan[np.asarray(existing_green, dtype=bool)] = PLAN_GREEN  # never drop existing green
    plan[state == -1] = PLAN_NONE  # unbuildable (rivers / roads / etc.) is never developed OR green
    plan[state == STATE_CROSSING] = PLAN_CROSSING  # a barrier a walk may cross, never built on
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
    max_distance_m,
    existing_centres=None,
    max_eval=None,
    existing_built=None,
    existing_green=None,
    centre_mode="placed",
    transit_stops=None,
    centre_anchors=None,
    centre_distance_m=None,
    green_distance_m=None,
    centre_m2_per_person=CENTRE_M2_PER_PERSON,
    new_density_km2=MEAN_NEW_DENSITY_KM2,
    centre_min_settlement=3,
    prune_islands=True,
    progress=None,
    target_population=None,
    min_park_area_m2=None,
    stop_catchment_m=None,
    transit_hubs=None,
    hub_catchment_m=None,
):
    """Pick the recommended plan from per-run final states: optimise EVERY run (at
    ``centre_mode``; see ``optimise_plan``) and keep the one with the lowest average walk
    (``access_cost``). Pass ``max_eval`` to optimise only that many evenly-sampled runs
    (faster for very large ensembles; runs are similar). ``existing_built``/``existing_green``
    (bool masks of already-developed land)
    are frozen — never pruned — and the chosen plan tags them with the existing-* codes.
    Returns ``(best_plan, best_metrics, pre_plan, best_state)`` — ``(None, None, None, None)`` if empty.
    ``pre_plan`` is the chosen run BEFORE post-processing (its raw CA development, grown centres and
    qualifying green), so the pre/post pair can be compared.

    ``progress`` is an optional callable ``(done, total) -> bool`` invoked after each
    post-processed candidate; return False to abort (the function then
    returns four Nones).
    """
    states = list(states)
    if not states:
        return None, None, None, None
    if max_eval and len(states) > max_eval:  # optional cap for very large ensembles
        states = states[:: len(states) // max_eval][:max_eval]

    # walk fields depend only on the coordinate, so one cache serves every member
    walk_cache: dict = {}

    def optimise_and_score(st):
        st = np.asarray(st)
        ca_centres = [(int(y), int(x)) for y, x in np.argwhere(st == 2)]  # the CA's grown centres
        opt = optimise_plan(
            _state_to_plan(st, granularity_m, existing_green=existing_green),
            granularity_m, max_distance_m,
            existing_centres=existing_centres,
            existing_built=existing_built, ca_centres=ca_centres,
            centre_mode=centre_mode, centre_anchors=centre_anchors,
            centre_distance_m=centre_distance_m, green_distance_m=green_distance_m,
            centre_m2_per_person=centre_m2_per_person,
            new_density_km2=new_density_km2,
            centre_min_settlement=centre_min_settlement, prune_islands=prune_islands,
            walk_cache=walk_cache,
        )
        # tag before scoring: the coverage metrics and the selection metric read
        # the existing-* codes to confine themselves to new homes
        opt = _mark_existing(opt, existing_built=existing_built, existing_centres=existing_centres)
        m = evaluate_plan(
            opt, granularity_m, max_distance_m,
            transit_stops=transit_stops,
            centre_distance_m=centre_distance_m, green_distance_m=green_distance_m,
            new_density_km2=new_density_km2, existing_green=existing_green,
            target_population=target_population, min_park_area_m2=min_park_area_m2,
            stop_catchment_m=stop_catchment_m,
            transit_hubs=transit_hubs, hub_catchment_m=hub_catchment_m,
        )
        return opt, m

    total = len(states)
    done = 0
    best_plan, best, best_state = None, None, None
    for st in states:
        st = np.asarray(st)
        opt, m = optimise_and_score(st)
        done += 1
        if progress is not None and not progress(done, total):
            return None, None, None, None
        # a degenerate run can yield zero built cells (metrics has no access_cost); never select it
        if best is None or m.get("access_cost", math.inf) < best.get("access_cost", math.inf):
            best_plan, best, best_state = opt, m, st
    pre_plan = None
    if best_plan is not None:
        # the chosen run BEFORE post-processing — its raw CA development + grown centres + qualifying
        # green — tagged with existing-* codes so it lines up with the post-processed plan
        pre_plan = _state_to_plan(best_state, granularity_m, existing_green=existing_green)
        pre_plan[np.asarray(best_state) == 2] = PLAN_CENTRE  # show the CA's own grown centres
        pre_plan = _mark_existing(pre_plan, existing_built=existing_built, existing_centres=existing_centres)
    return best_plan, best, pre_plan, best_state


def derive_density(
    plan,
    granularity_m,
    centre_distance_m,
    density_factors_km2,
    prob_distribution,
):
    """Per-cell density (people/km²) for a FINISHED scenario, arranging the three tiers core-out.

    Every new cell was built at one of three densities, drawn at the given probabilities (the mix).
    Here those drawn values are ARRANGED spatially: new cells are ranked by the WALKING DISTANCE
    to the nearest post-processed mixed-use centre, ties broken by interior depth (how far inside
    the new fabric a cell sits), AS A PERCENTILE WITHIN THEIR OWN contiguous settlement. The
    highest densities go to the lowest percentiles, then medium, then low, so density falls away
    from each centre. The gradient runs from the centre outward rather than from the edge inward:
    where a centre stands at a settlement's edge, as a station against a rail line does, density
    grades away from that centre in bands, which is what a centre in that position means. Depth
    breaks ties so that, among cells the same walk from a centre, the one buried in the fabric
    outranks the one on the edge. The within-settlement percentile gives every settlement its own
    gradient; a raw global ranking instead starves whole small settlements into a single low
    tier. The tier counts follow the probabilities (``n_high = round(p_high · N)``, …),
    so the population equals the probability-weighted mean over the cells — the same accounting
    the run's stopping rule uses. Existing fabric is not counted (0); non-built cells are 0
    (nodata).

    Arranged in post-processing, not during growth: mid-run distances measure against whichever
    centres happen to exist at that moment, and post-processing then moves, adds and culls centres.
    """
    plan = np.asarray(plan)
    g = float(granularity_m)
    high, med, low = (float(d) for d in density_factors_km2)
    p_high, p_med, _p_low = (float(p) for p in prob_distribution)
    new_built = np.isin(plan, (PLAN_BUILT, PLAN_CENTRE))
    centres = np.isin(plan, (PLAN_CENTRE, PLAN_EXIST_CENTRE))
    out = np.zeros(plan.shape, dtype=np.float32)
    n = int(new_built.sum())
    if n == 0:
        return out
    # Walking distance to the MIDDLE of the nearest centre, not to whichever of its cells is
    # closest. Measuring to the nearest cell makes every cell inside a centre equidistant at
    # zero and measures the fabric outside it from the centre's rim, which grades density off
    # a belt rather than off the centre. From the middle, density peaks at the heart of the
    # centre, tapers across the centre area, and keeps tapering through the fabric around it.
    dist_field = np.full(plan.shape, np.inf)
    blocked = plan == PLAN_NONE
    for comp in _components(centres):
        m = np.zeros(plan.shape, dtype=bool)
        for y, x in comp:
            m[y, x] = True
        mid = _interior_point(m)
        if mid is None:
            mid = comp[0]
        seed = np.zeros(plan.shape, dtype=bool)
        seed[int(mid[0]), int(mid[1])] = True
        # bounded generously: a centre's own area sits inside this, and the walk beyond it
        # only has to rank cells, not certify access
        d = _walk_distance(seed, g, float(centre_distance_m) * 2.0, blocked=blocked)
        np.minimum(dist_field, d, out=dist_field)
    dists = dist_field[new_built]
    # percentile of distance WITHIN each contiguous settlement, so every settlement grades
    # from dense at its centre to light at its edge, whatever its absolute distances
    built_any = np.isin(plan, (PLAN_BUILT, PLAN_CENTRE, PLAN_EXIST_BUILT, PLAN_EXIST_CENTRE))
    comp_label = np.full(plan.shape, -1, dtype=int)
    for i, comp in enumerate(_components(built_any)):
        for y, x in comp:
            comp_label[y, x] = i
    # interior depth of the new fabric: how many 8-neighbour erosions a cell survives
    # (chebyshev distance to the region's edge), so a settlement's core ranks first
    depth = np.zeros(plan.shape, dtype=np.int32)
    mask = new_built.copy()
    while mask.any():
        depth[mask] += 1
        core = mask[1:-1, 1:-1].copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                core &= mask[1 + dy : mask.shape[0] - 1 + dy, 1 + dx : mask.shape[1] - 1 + dx]
        nxt = np.zeros_like(mask)
        nxt[1:-1, 1:-1] = core
        mask = nxt
    labels = comp_label[new_built]
    depth_b = depth[new_built]
    keys = np.full(n, np.inf)
    for lab in np.unique(labels):
        members = labels == lab
        # nearest the centre first, ties broken by depth so a cell buried inside the
        # fabric outranks one on the edge at the same walk; lexsort ranks by its LAST key
        order_m = np.lexsort((-depth_b[members], dists[members]))
        r = np.empty(order_m.size, dtype=np.int64)
        r[order_m] = np.arange(order_m.size)
        denom = max(int(members.sum()) - 1, 1)
        keys[members] = r / denom
    order = np.argsort(keys, kind="stable")
    n_high = min(int(round(p_high * n)), n)
    n_med = min(int(round(p_med * n)), n - n_high)
    ranked = np.empty(n, dtype=np.float32)
    tiers = np.empty(n, dtype=np.float32)
    tiers[:n_high] = high
    tiers[n_high : n_high + n_med] = med
    tiers[n_high + n_med :] = low
    ranked[order] = tiers  # closest cell (order[0]) gets the first (highest) tier
    out[new_built] = ranked
    return out


def to_tiered_plan(plan, density, density_factors_km2):
    """Remap a plan's new built/centre cells to per-tier DISPLAY codes by their arranged density, so
    the categorical raster shows low/medium/high development in distinct shades (built and centres in
    separate hues). Existing and green cells are unchanged. Returns a uint8 copy for writing/styling;
    the base ``plan`` codes are untouched for all metric logic."""
    plan = np.asarray(plan)
    density = np.asarray(density)
    high, med, low = (float(d) for d in density_factors_km2)
    out = plan.astype(np.uint8).copy()

    def _tier(mask, low_code, med_code, high_code):
        if not mask.any():
            return
        vals = density[mask]
        # nearest tier by value (derive_density assigns the exact tier values, so this is exact)
        dl, dm, dh = np.abs(vals - low), np.abs(vals - med), np.abs(vals - high)
        pick = np.select(
            [(dl <= dm) & (dl <= dh), (dm <= dl) & (dm <= dh)],
            [low_code, med_code],
            default=high_code,
        )
        out[mask] = pick.astype(np.uint8)

    _tier(plan == PLAN_BUILT, PLAN_BUILT_LOW, PLAN_BUILT_MED, PLAN_BUILT_HIGH)
    _tier(plan == PLAN_CENTRE, PLAN_CENTRE_LOW, PLAN_CENTRE_MED, PLAN_CENTRE_HIGH)
    return out


def plan_variants(
    state,
    granularity_m,
    max_distance_m,
    modes,
    *,
    existing_centres=None,
    existing_built=None,
    existing_green=None,
    centre_anchors=None,
    centre_distance_m=None,
    green_distance_m=None,
    centre_m2_per_person=CENTRE_M2_PER_PERSON,
    new_density_km2=MEAN_NEW_DENSITY_KM2,
    centre_min_settlement=3,
    prune_islands=True,
    min_park_area_m2=None,
):
    """Post-process one chosen CA run ``state`` at several centre modes, so the user can compare
    the options and pick rather than choosing up front. ``modes`` maps a label to a
    ``centre_mode`` (``"grown"`` / ``"placed"`` / ``"minimal"``; see ``optimise_plan``). Returns
    ``{label: (plan, metrics)}``; each plan is tagged with the existing-* codes."""
    state = np.asarray(state)
    ca_centres = [(int(y), int(x)) for y, x in np.argwhere(state == 2)]
    base = _state_to_plan(state, granularity_m, existing_green=existing_green)
    out: dict = {}
    for label, mode in modes.items():
        plan = optimise_plan(
            base, granularity_m, max_distance_m,
            existing_centres=existing_centres, existing_built=existing_built, ca_centres=ca_centres,
            centre_mode=mode, centre_anchors=centre_anchors,
            centre_distance_m=centre_distance_m, green_distance_m=green_distance_m,
            centre_m2_per_person=centre_m2_per_person,
            new_density_km2=new_density_km2,
            centre_min_settlement=centre_min_settlement, prune_islands=prune_islands,
        )
        marked = _mark_existing(plan, existing_built=existing_built, existing_centres=existing_centres)
        # scored on the MARKED plan: the coverage metrics, the selection metric, and the
        # per-person readouts all need the existing/new split
        metrics = evaluate_plan(
            marked, granularity_m, max_distance_m,
            centre_distance_m=centre_distance_m, green_distance_m=green_distance_m,
            new_density_km2=new_density_km2, existing_green=existing_green,
            min_park_area_m2=min_park_area_m2,
        )
        out[label] = (marked, metrics)
    return out
