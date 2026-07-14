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
_BUILT_HIGH = (204, 122, 41)
_CENTRE_LOW = (252, 187, 161)
_CENTRE_MED = (239, 101, 72)
_CENTRE_HIGH = (179, 18, 24)
_EXIST_BUILT = (150, 134, 122)  # a cool grey-taupe so existing fabric recedes and reads apart from the warm new ramp
_EXIST_CENTRE = (150, 40, 85)
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

PLAN_PALETTE = [
    (PLAN_GREEN, _GREEN, "Recommended green network"),
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
    seeds, fixed, built, new_built, granularity_m, max_distance_m, cull_min_unique=3, walk=None,
    spacing_m=None, anchors=None, walk_cache=None,
):
    """Optimise seeded centres after the fact, measuring catchment by ``walk`` — ONE distance
    model used for every judgment here: the bounded grid walk (callers inject it as the
    ``walk`` callable ``mask -> rows×cols metres``, sharing one barrier mask and cache). Each new centre
    is re-positioned onto NEW land, central to the NEW homes it serves; ``fixed``/existing centres
    compete in the assignment; centres uniquely serving fewer than ``cull_min_unique`` built cells
    are culled (redundant, or feeding too small a catchment). Returns the optimised new ``(row, col)``.

    ``spacing_m`` sets how far apart centres sit — their catchment scale (defaults to the walk,
    ``max_distance_m`` = the coverage-minimal arrangement). A LARGER spacing CONSOLIDATES — fewer,
    larger, more central centres, accepting that some homes end up beyond a single walk of one; a
    smaller spacing disperses (more, closer centres). It may exceed the walk (the ``walk`` field is
    bounded to reach it), which is how aggressive consolidation actually clumps centres together.

    ANCHOR INVARIANT: every contiguous built settlement containing new development keeps at least
    one directly attached centre (its own, or a fixed one within it). Walking distances traverse
    green, so without this a cluster can look "served" by a neighbouring cluster's centre across a
    green gap and be stripped bare by consolidation; a settlement without its own centre is not a
    settlement in this model's terms.

    PROVISION RULE: existing centres serve the existing population and do NOT count as provision
    for new development — new growth hugging an existing town must still earn its own centre, or
    it would sprawl centre-free along existing fabric. ``anchors`` (station-anchored centres,
    a subset of ``fixed``) are new provision and DO count: they are grown and sized by this
    pipeline like any new centre, only their location is pinned.
    """
    built = np.asarray(built, dtype=bool)
    new_built = np.asarray(new_built, dtype=bool) & built
    rows, cols = built.shape
    spacing = max_distance_m if spacing_m is None else float(spacing_m)
    r = max(1, round(spacing / granularity_m))
    if walk is None:
        bound = max(max_distance_m, spacing)  # the field must reach the spacing (which may exceed the walk)

        def walk(mask):
            return _walk_distance(mask, granularity_m, bound)

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

    def reach(cells):  # built cells within the centre SPACING of any cell in `cells` (by the one metric)
        if not cells:
            return np.zeros((rows, cols), dtype=bool)
        if len(cells) == 1:
            return centre_field(tuple(cells[0])) <= spacing
        return np.minimum.reduce([centre_field(tuple(c)) for c in cells]) <= spacing

    new = [_nearest_built(new_built, int(y), int(x)) for y, x in seeds]

    # settlement components for the anchor invariant: label contiguous built clusters, note which
    # contain new development (those must stay anchored) and which hold a fixed/existing centre
    comps = _components(built)
    comp_label = np.full((rows, cols), -1, dtype=int)
    for i, comp in enumerate(comps):
        for y, x in comp:
            comp_label[y, x] = i
    needs_anchor = {i for i, comp in enumerate(comps) if any(new_built[y, x] for y, x in comp)}
    fixed_comps = {int(comp_label[y, x]) for y, x in fixed if 0 <= y < rows and 0 <= x < cols} - {-1}

    # Distance/reach to the nearest FIXED (existing) centre, solved ONCE — fixed centres don't move,
    # and true-area centres are many cells, so collapsing them into a single field (rather than one
    # per cell) is what keeps this single-threaded post-processing affordable.
    fixed_field = walk(onehot(fixed)) if fixed else np.full((rows, cols), np.inf)
    fixed_col = fixed_field[new_built]
    comp_col = comp_label[new_built]  # settlement id of every new home (placement is settlement-local)
    # station anchors are the only fixed centres that count as provision for NEW development
    anchors = [(int(y), int(x)) for y, x in (anchors or [])]
    anchor_reach = (
        (walk(onehot(anchors)) <= spacing) if anchors else np.zeros((rows, cols), dtype=bool)
    )

    def lloyd(centres):  # re-position each new centre to the INTERIOR of the NEW homes it serves
        if not centres:
            return centres
        member_mask = np.zeros((rows, cols), dtype=bool)
        for _ in range(8):
            # column 0 = nearest fixed centre; columns 1.. = each new centre (single-source, cached)
            stack = np.column_stack([fixed_col] + [centre_col(tuple(c)) for c in centres])
            nearest = np.argmin(stack, axis=1)
            within = stack.min(axis=1) <= spacing  # homes beyond the spacing of every centre pull no one
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

    new = lloyd(new)

    # Add centres where NEW development is still beyond a walk of any NEW centre or anchor.
    # Existing centres do not suppress an addition (the provision rule above). The densest
    # underserved cluster (a box-sum) only proposes WHERE; whether a centre is warranted there is
    # confirmed by the one metric (how many underserved homes it actually reaches within a walk).
    while True:
        underserved = new_built & ~(anchor_reach | reach(new))
        if not underserved.any():
            break
        # propose the new centre from WITHIN the underserved area (densest spot), not merely
        # near it: a box-near cell may be unable to actually reach across a barrier
        gain = np.where(underserved, _box_sum(underserved.astype(np.float64), r), -1.0)
        if gain.max() < cull_min_unique:
            break
        y, x = divmod(int(np.argmax(gain)), cols)
        if int((reach([(int(y), int(x))]) & underserved).sum()) < cull_min_unique:
            break  # the largest remaining gap is too small to warrant a centre
        new.append((int(y), int(x)))
    new = lloyd(new)

    # Cull a centre uniquely serving < cull_min_unique NEW built cells (redundant / overly small).
    # A new centre's unique coverage = new cells it reaches that no OTHER new centre and no anchor
    # does; coverage by an existing centre does not discount it (the provision rule above).
    # The anchor invariant caps the cull: a centre that is its settlement's LAST anchor is never
    # removed, however redundant its coverage looks through the green to a neighbouring cluster.
    def is_last_anchor(j, centres):
        cj = int(comp_label[centres[j][0], centres[j][1]])
        if cj < 0 or cj not in needs_anchor or cj in fixed_comps:
            return False
        return not any(int(comp_label[c[0], c[1]]) == cj for k, c in enumerate(centres) if k != j)

    while new:
        new_masks = [reach([c]) for c in new]
        new_count = np.sum(new_masks, axis=0)
        unique = [int((new_built & new_masks[j] & (new_count == 1) & ~anchor_reach).sum()) for j in range(len(new))]
        cullable = [j for j in range(len(new)) if unique[j] < cull_min_unique and not is_last_anchor(j, new)]
        if not cullable:
            break
        new.pop(min(cullable, key=lambda j: unique[j]))
        new = lloyd(new)

    # Backstop for the same invariant: if a settlement with new development still has no attached
    # centre (the CA never seeded one there, or Lloyd drifted its centre into another cluster),
    # anchor it at the interior of its new development.
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
# Redundancy floor for the centre cull/add — a centre that uniquely serves fewer than this many built
# cells is dropped. This is the CENTRE catchment minimum, kept small and SEPARATE from the
# minimum-SETTLEMENT size (which prunes failed-satellite clusters); conflating them made large
# min-settlement values stop centres forming at all.
CENTRE_CULL_MIN = 3


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
                  max_area=CENTRE_AREA_MAX):
    """Grow each new centre POINT into a contiguous AREA on built land, sized by the POPULATION it is
    the nearest centre to (its Voronoi catchment within a walk) at ``m2_per_person`` of centre land
    per resident — like a real centre, bigger where it serves more people. Mixed-use: cells stay
    built/homes, just designated centre; existing/fixed centres are left intact and never grown into.
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
        blob = _grow_blob(points[j], targets[j], built, claimed)
        grown |= blob
        claimed |= blob
    return grown


def _components(mask):
    """8-connected components of a bool mask, as a list of (row, col) cell lists."""
    mask = np.asarray(mask, dtype=bool)
    labels = _label_components(mask, queen=True)
    if labels is not None:
        comps: list[list[tuple[int, int]]] = [[] for _ in range(int(labels.max()))]
        for y, x in zip(*np.nonzero(mask)):
            comps[labels[y, x] - 1].append((int(y), int(x)))
        return comps
    rows, cols = mask.shape
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
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
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
    centre_distance_m: float | None = None,
    green_distance_m: float | None = None,
    new_density_km2: float = MEAN_NEW_DENSITY_KM2,
    existing_green: np.ndarray | None = None,
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
    # Count existing fabric too, so scoring is the same whether the plan is still raw or has been
    # tagged with the existing-* codes (otherwise a marked pre_plan silently drops existing built/centres
    # and its metrics — and the raw-vs-option built-cell delta — come out inconsistent).
    built = np.isin(plan, (PLAN_BUILT, PLAN_CENTRE, PLAN_EXIST_BUILT, PLAN_EXIST_CENTRE))
    n_built = int(built.sum())
    if n_built == 0:
        return {"built_cells": 0}

    green_mask = plan == PLAN_GREEN
    if min_green_span_m:  # only real parks count, matching recommended_plan
        green_min = max(1, round((min_green_span_m / granularity_m) ** 2))
        green_mask = _keep_large_components(green_mask, green_min)

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

    d_cent = _dist(np.isin(plan, (PLAN_CENTRE, PLAN_EXIST_CENTRE)))[built]
    d_green = _dist(green_mask)[built]
    near_cent = d_cent <= centre_distance_m
    near_green = d_green <= green_distance_m
    served = near_cent & near_green

    # selection metric: average walk to amenities over EVERY home, with anyone who
    # can't reach within the limit counted at a penalty distance (so a plan can't
    # score well by abandoning the fringe). Lower is better.
    centre_access = float(np.where(near_cent, d_cent, 2.0 * centre_distance_m).mean())
    green_access = float(np.where(near_green, d_green, 2.0 * green_distance_m).mean())
    access_cost = 0.5 * (centre_access + green_access)

    rows, cols = plan.shape
    adj = 0
    for dy, dx in ((1, 0), (0, 1)):
        a = built[: rows - dy, : cols - dx] & built[dy:, dx:]
        adj += 2 * int(a.sum())  # each shared edge counts for both cells

    # supply-side efficiency: how well-used each centre / unit of green is. Existing centres
    # count too — the numerator (homes near ANY centre) does, and no metric may depend on
    # whether the plan is tagged with the EXIST_* codes.
    n_centres = int(np.isin(plan, (PLAN_CENTRE, PLAN_EXIST_CENTRE)).sum())
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
    # per-person provision (rule-of-thumb readouts): NEW amenity over NEW population only. Existing
    # fabric carries no population (it is assumed served by its own centres), so the honest ratio is
    # what the plan ADDS — new mixed-use centre land, and new green — per new resident. Pass
    # ``existing_green`` (a bool mask) to exclude pre-existing green from the provision numerator;
    # without it, all qualifying green counts as new. NB these depend on the existing-* tagging,
    # unlike the coverage metrics above (evaluate marked plans for honest splits).
    cell_km2 = granularity_m * granularity_m / 1e6
    n_exist = int(np.isin(plan, (PLAN_EXIST_BUILT, PLAN_EXIST_CENTRE)).sum())
    population = (n_built - n_exist) * new_density_km2 * cell_km2
    n_new_centres = int((plan == PLAN_CENTRE).sum())
    new_green_mask = green_mask if existing_green is None else green_mask & ~np.asarray(existing_green, dtype=bool)
    n_new_green = int(new_green_mask.sum())
    metrics["population"] = population
    metrics["centre_m2_per_person"] = n_new_centres * cell_km2 * 1e6 / population if population else 0.0
    metrics["green_m2_per_person"] = n_new_green * cell_km2 * 1e6 / population if population else 0.0

    # transit access — a third dimension, REPORTED ONLY for now (not folded into access_cost,
    # so it cannot distort run-selection until validated). See the transit-routing plan.
    if transit_stops is not None:
        stops = np.asarray(transit_stops, dtype=bool)
        if stops.any():
            d_stop = _dist(stops)[built]
            near_stop = d_stop <= max_distance_m
            metrics["transit_coverage"] = float(near_stop.mean())
            metrics["transit_access"] = float(np.where(near_stop, d_stop, 2.0 * max_distance_m).mean())
            metrics["transit_walk_mean"] = float(d_stop[near_stop].mean()) if near_stop.any() else math.inf

    return metrics


def audit_centres(plan, granularity_m, max_distance_m):
    """Per-centre-AREA effectiveness audit, by the one distance model (the bounded grid walk).
    Centres are areas (existing true-area + grown new ones), so each
    record is a connected component, with its ``cells`` (area), how many built cells it **serves**
    (within a walk) and the **mean walk** to them (low = well-centred; few served = an ineffective
    centre on a thin/edge catchment).

    Run after each plan so weak centres are visible and the cull threshold can be tuned to evidence
    rather than by eye. Returns ``{"centres": [...weakest first...], "summary": {...}}``.
    """
    plan = np.asarray(plan)
    walk_blocked = plan == PLAN_NONE

    def walk(mask):
        return _walk_distance(mask, granularity_m, max_distance_m, blocked=walk_blocked)

    built = np.isin(plan, (PLAN_BUILT, PLAN_CENTRE, PLAN_EXIST_BUILT, PLAN_EXIST_CENTRE))
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
        records.append(
            {
                "row": int(round(sum(c[0] for c in comp) / len(comp))),
                "col": int(round(sum(c[1] for c in comp) / len(comp))),
                "cells": len(comp),  # the centre's area (a town centre spans many cells)
                "served": served,  # built cells within a walk — the catchment it serves
                "mean_dist_m": float(d[served_mask].mean()) if served else math.inf,  # avg walk to them
                "existing": any(plan[y, x] == PLAN_EXIST_CENTRE for y, x in comp),
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
    min_green_span_m: float,  # accepted for signature stability; green qualification happens in _state_to_plan
    max_distance_m: float,
    existing_centres=None,
    existing_built=None,
    ca_centres=None,
    optimise_centres: bool = True,
    centre_anchors=None,
    centre_distance_m: float | None = None,
    green_distance_m: float | None = None,
    centre_spacing_m: float | None = None,
    centre_m2_per_person: float = CENTRE_M2_PER_PERSON,
    new_density_km2: float = MEAN_NEW_DENSITY_KM2,
    centre_min_settlement: int = 3,
    prune_islands: bool = True,
    walk_cache: dict | None = None,
) -> np.ndarray:
    """Post-process a single CA run's plan into the recommended plan: prune failed-satellite specks,
    then (when ``optimise_centres``, the default) re-place the centres central to their development,
    add centres where new development is under-served, cull redundant ones and grow each to an area —
    otherwise keep the simulation's grown centres as-is. The CA's green network is kept as-is. Returns
    a new plan.

    ``existing_built`` is a bool mask of cells already developed before the simulation; those are
    **frozen** (never pruned) and tagged distinctly downstream.

    Centre/green walks are split: ``centre_distance_m`` / ``green_distance_m`` (each defaulting to
    ``max_distance_m``) are the walk thresholds for centre vs green coverage. ``centre_spacing_m``
    sets centre consolidation (consolidated↔dispersed; see ``_refine_centres``); each centre grows to
    ``centre_m2_per_person`` of centre land per resident it serves (population estimated per cell from
    ``new_density_km2``; existing fabric counts zero — only new residents size a centre);
    ``centre_min_settlement`` is the minimum settlement size below which a detached new cluster is pruned.
    """
    plan = plan.copy()
    g = float(granularity_m)
    # Split walks: centres and green each have their own walk threshold (both default to the shared
    # max_distance). The distance FIELD is bounded at the larger of the walks AND the centre spacing
    # (consolidation may place centres more than a walk apart, so the field must reach that far).
    centre_distance_m = max_distance_m if centre_distance_m is None else float(centre_distance_m)
    green_distance_m = max_distance_m if green_distance_m is None else float(green_distance_m)
    field_bound = max(centre_distance_m, green_distance_m, float(centre_spacing_m or 0.0))
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

    # Cleanup FIRST — prune "failed satellites": an entirely-NEW built settlement (its own connected
    # component) smaller than the minimum settlement size is a stranded speck, not viable development.
    # Remove it up front so nothing is greened or centred on it, and so a dispersed CA run can't leave
    # a lone centre on a 3-cell orphan (the catchment cull keeps such a centre because it can SEE many
    # scattered homes within a walk; judging by SETTLEMENT SIZE is what actually removes the orphan).
    # A small cluster contiguous with existing/frozen fabric is kept (it extends a real settlement).
    # This is the CONTIGUITY rule: a mixed-use centre only exists attached to a contiguous built
    # settlement of at least the minimum size, so it applies whether or not centres are optimised,
    # and the threshold is floored (MIN_SETTLEMENT_CELLS) so a coarse grid cannot collapse it.
    centre_min_settlement = max(int(centre_min_settlement), MIN_SETTLEMENT_CELLS)
    if prune_islands:
        for comp in _components((plan == PLAN_BUILT) | (plan == PLAN_CENTRE)):
            if len(comp) < centre_min_settlement and not any(frozen[y, x] for y, x in comp):
                for y, x in comp:
                    plan[y, x] = PLAN_GREEN  # the land reverts to nature, blending with its surroundings

    n_built = int(((plan == PLAN_BUILT) | (plan == PLAN_CENTRE)).sum())
    if n_built == 0:
        return plan

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
            for s in _seed_centres_proximity(built, granularity_m, centre_distance_m, existing_centres)
            if s not in exclude
        ]
    else:
        seed_new = [(int(y), int(x)) for y, x in ca_centres if (int(y), int(x)) not in exclude]
    for ey, ex in fixed_on_built:
        plan[ey, ex] = PLAN_CENTRE
    if optimise_centres:
        new_centres = _refine_centres(
            seed_new, fixed_on_built, built, new_built, granularity_m, centre_distance_m,
            cull_min_unique=CENTRE_CULL_MIN, walk=walk, spacing_m=centre_spacing_m,
            anchors=anchor_on_built, walk_cache=walk_cache,
        )
        # Grow each placed centre into an AREA sized by the homes it serves (mixed-use, on built).
        # Station anchors grow too — a station should seed a real centre, not stay a lone cell — while
        # existing/true-area centres come pre-sized from the input and are left intact (claimed, not grown).
        grow_points = list(dict.fromkeys(new_centres + anchor_on_built))
        exist_mask = (
            (np.asarray(existing_built, dtype=bool) & built)
            if existing_built is not None
            else np.zeros_like(built)
        )
        # existing fabric contributes NO population: centres are sized by the new residents they serve
        cell_pop = np.where(exist_mask, 0.0, new_density_km2) * (g * g / 1e6)
        new_centres = _grow_centres(
            grow_points, existing_on_built, built, walk, cell_pop, centre_m2_per_person, g * g
        )
    else:  # keep the simulation's grown centres as-is (only those still on built land)
        new_centres = [(y, x) for y, x in seed_new if 0 <= y < rows and 0 <= x < cols and plan[y, x] == PLAN_BUILT]
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
    plan[state == -1] = PLAN_NONE  # unbuildable (rivers / roads / etc.) is never developed OR green
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
    existing_centres=None,
    max_eval=None,
    existing_built=None,
    existing_green=None,
    optimise_centres=True,
    transit_stops=None,
    centre_anchors=None,
    centre_distance_m=None,
    green_distance_m=None,
    centre_spacing_m=None,
    centre_m2_per_person=CENTRE_M2_PER_PERSON,
    new_density_km2=MEAN_NEW_DENSITY_KM2,
    centre_min_settlement=3,
    prune_islands=True,
    progress=None,
):
    """Pick the recommended plan from per-run final states: optimise EVERY run and keep
    the one with the lowest average walk (``access_cost``). Pass ``max_eval`` to optimise
    only that many evenly-sampled runs (faster for very large ensembles; runs are
    similar). ``existing_built``/``existing_green`` (bool masks of already-developed land)
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
            _state_to_plan(st, min_green_span_m, granularity_m, existing_green=existing_green),
            granularity_m, min_green_span_m, max_distance_m,
            existing_centres=existing_centres,
            existing_built=existing_built, ca_centres=ca_centres,
            optimise_centres=optimise_centres, centre_anchors=centre_anchors,
            centre_distance_m=centre_distance_m, green_distance_m=green_distance_m,
            centre_spacing_m=centre_spacing_m, centre_m2_per_person=centre_m2_per_person,
            new_density_km2=new_density_km2,
            centre_min_settlement=centre_min_settlement, prune_islands=prune_islands,
            walk_cache=walk_cache,
        )
        m = evaluate_plan(
            opt, granularity_m, max_distance_m, min_green_span_m=min_green_span_m,
            transit_stops=transit_stops,
            centre_distance_m=centre_distance_m, green_distance_m=green_distance_m,
            new_density_km2=new_density_km2,
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
        best_plan = _mark_existing(best_plan, existing_built=existing_built, existing_centres=existing_centres)
        # re-score the marked plan: coverage metrics are basis-independent, but the per-person
        # readouts need the existing/new split (new amenity over new population only)
        best = evaluate_plan(
            best_plan, granularity_m, max_distance_m, min_green_span_m=min_green_span_m,
            transit_stops=transit_stops,
            centre_distance_m=centre_distance_m, green_distance_m=green_distance_m,
            new_density_km2=new_density_km2, existing_green=existing_green,
        )
        # the chosen run BEFORE post-processing — its raw CA development + grown centres + qualifying
        # green — tagged with existing-* codes so it lines up with the post-processed plan
        pre_plan = _state_to_plan(best_state, min_green_span_m, granularity_m, existing_green=existing_green)
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
    """Per-cell density (people/km²) for a FINISHED scenario, arranging the three tiers by distance.

    Every new cell was built at one of three densities, drawn at the given probabilities (the mix).
    Here those drawn values are ARRANGED spatially: new cells are ranked by walking distance to the
    nearest (post-processed) mixed-use centre, and the highest densities go to the closest cells,
    then medium, then low. The tier counts follow the probabilities (``n_high = round(p_high · N)``,
    …), so the population equals the probability-weighted mean over the cells — the same accounting
    the run's stopping rule uses — while the layout is coherent (densest by the centres). Existing
    fabric is not counted (0); non-built cells are 0 (nodata).

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
    # walking distance to the nearest centre for every new cell (inf where none is within a walk)
    if centres.any():
        dist_field = _walk_distance(centres, g, float(centre_distance_m), blocked=plan == PLAN_NONE)
    else:
        dist_field = np.full(plan.shape, np.inf)
    dists = dist_field[new_built]
    # rank ascending; ties and unreachable (inf) sort last, so they take the lowest tier
    order = np.argsort(dists, kind="stable")
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
    min_green_span_m,
    max_distance_m,
    spacings,
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
):
    """Post-process one chosen CA run ``state`` at several centre-SPACING settings, so the user can
    compare compactness options and pick rather than choosing up front. ``spacings`` maps a label to a
    centre spacing in metres (``None`` = consolidated / coverage-minimal). Returns ``{label: (plan,
    metrics)}``; each plan is fully optimised (centre placement at that spacing) and tagged with the
    existing-* codes."""
    state = np.asarray(state)
    ca_centres = [(int(y), int(x)) for y, x in np.argwhere(state == 2)]
    base = _state_to_plan(state, min_green_span_m, granularity_m, existing_green=existing_green)
    out: dict = {}
    for label, spacing_m in spacings.items():
        plan = optimise_plan(
            base, granularity_m, min_green_span_m, max_distance_m,
            existing_centres=existing_centres, existing_built=existing_built, ca_centres=ca_centres,
            optimise_centres=True, centre_anchors=centre_anchors,
            centre_distance_m=centre_distance_m, green_distance_m=green_distance_m,
            centre_spacing_m=spacing_m, centre_m2_per_person=centre_m2_per_person,
            new_density_km2=new_density_km2,
            centre_min_settlement=centre_min_settlement, prune_islands=prune_islands,
        )
        marked = _mark_existing(plan, existing_built=existing_built, existing_centres=existing_centres)
        # scored on the MARKED plan: coverage is basis-independent, and the per-person readouts
        # need the existing/new split (new amenity over new population only)
        metrics = evaluate_plan(
            marked, granularity_m, max_distance_m, min_green_span_m=min_green_span_m,
            centre_distance_m=centre_distance_m, green_distance_m=green_distance_m,
            new_density_km2=new_density_km2, existing_green=existing_green,
        )
        out[label] = (marked, metrics)
    return out
