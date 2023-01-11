"""
Can't handle border regions in numba stencil: only 'constant' mode is currently supported, i.e. border buffer = 0

Don't use anything from QGIS so that it is easier to test this module.
"""
from __future__ import annotations

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from futurb.isobenefit.logger import get_logger
from numba import njit
from rasterio import features, transform
from shapely import geometry

LOGGER: logging.Logger = get_logger()


@njit
def _random_density(
    prob_distribution: tuple[float, float, float], density_factors: tuple[float, float, float]
) -> float:
    """Numba compatible method for determining a land use density"""
    p = np.random.rand()
    if p >= 0 and p < prob_distribution[0]:
        return density_factors[0]
    elif p >= prob_distribution[1] and p < prob_distribution[2]:
        return density_factors[1]
    else:
        return density_factors[2]


@njit
def _inc_access(y_idx: int, x_idx: int, arr: Any, granularity_m: int, max_distance_m: int) -> Any:
    """increment access"""
    return _agg_access(y_idx, x_idx, arr, granularity_m, max_distance_m, positive=True)


@njit
def _decr_access(y_idx: int, x_idx: int, arr: Any, granularity_m: int, max_distance_m: int) -> Any:
    """increment access"""
    return _agg_access(y_idx, x_idx, arr, granularity_m, max_distance_m, positive=False)


@njit
def _agg_access(y_idx: int, x_idx: int, arr: Any, granularity_m: int, max_distance_m: int, positive: bool) -> Any:
    """Aggregates access - from an x, y to provided array, either positive or negative"""
    for cy_idx, cx_idx in np.ndindex(arr.shape):
        y_dist = int(abs(y_idx - cy_idx) * granularity_m)
        x_dist = int(abs(x_idx - cx_idx) * granularity_m)
        dist = np.hypot(x_dist, y_dist)
        if dist > max_distance_m:
            continue
        val = 1  #  - dist / max_distance_m
        if positive:
            arr[cy_idx, cx_idx] += val
        else:
            arr[cy_idx, cx_idx] -= val
    return arr


@njit
def _iter_nbs(arr: Any, y_idx: int, x_idx: int, rook: bool) -> Any:
    """Returns rook or queen neighbours - return in order - no shuffling?"""
    idxs: list[list[int]] = []
    if rook:
        y_offsets = [1, 0, -1, 0]
        x_offsets = [0, 1, 0, -1]
    else:
        y_offsets = [1, 1, 1, 0, -1, -1, -1, 0]
        x_offsets = [-1, 0, 1, 1, 1, 0, -1, -1]
    # add offset and check extents
    for y_offset, x_offset in zip(y_offsets, x_offsets):
        y_nb_idx = y_idx + y_offset
        if y_nb_idx < 0 or y_nb_idx >= arr.shape[0]:
            continue
        x_nb_idx = x_idx + x_offset
        if x_nb_idx < 0 or x_nb_idx >= arr.shape[1]:
            continue
        idxs.append([y_nb_idx, x_nb_idx])

    return idxs


@njit
def _agg_dijkstra_cont(
    state_arr: Any,
    y_idx: int,
    x_idx: int,
    path_state: list[int],
    target_state: list[int],
    max_distance_m: int,
    granularity_m: int,
    break_first: bool = False,
    break_count: int | None = None,
    rook: bool = False,
) -> Any:
    """ """
    if break_first is True and break_count is not None:
        raise ValueError("Only one of break_first and break_count can be specified at once.")
    # targets
    targets_arr = np.full(state_arr.shape, 0, np.int_)
    if state_arr[y_idx, x_idx] in target_state:
        targets_arr[y_idx, x_idx] = 1
    # distances
    distances_arr = np.full(state_arr.shape, np.inf)
    distances_arr[y_idx, x_idx] = 0
    # claimed
    claimed_arr = np.full(state_arr.shape, False)
    claimed_arr[y_idx, x_idx] = True
    # pending - prime with current cell
    pending_arr = np.full(state_arr.shape, False)
    pending_arr[y_idx, x_idx] = True
    # breadth-first search
    while np.any(pending_arr):
        # find the item in the queue which is currently nearest
        min_dist = np.inf
        next_y = -1
        next_x = -1
        pending_ys, pending_xs = np.nonzero(pending_arr)
        for pending_y, pending_x in zip(pending_ys, pending_xs):
            if distances_arr[pending_y, pending_x] < min_dist:
                next_y = pending_y
                next_x = pending_x
                min_dist = distances_arr[pending_y, pending_x]
        # reset pending
        pending_arr[next_y, next_x] = False
        # retrieve the current distance
        next_dist = distances_arr[next_y, next_x]
        # explore neighbours
        for nb_y, nb_x in _iter_nbs(distances_arr, next_y, next_x, rook=rook):
            y_step = abs(nb_y - next_y)
            x_step = abs(nb_x - next_x)
            d_step = np.hypot(y_step, x_step) * granularity_m
            # compound distance to neighbour
            nb_dist = next_dist + d_step
            # check for max distance
            if nb_dist > max_distance_m:
                continue
            # aggregate targets
            if state_arr[nb_y, nb_x] in target_state:
                targets_arr[nb_y, nb_x] = 1
            # don't follow this neighbour if the wrong path state
            if not state_arr[nb_y, nb_x] in path_state:
                continue
            # if the nb distance is less than already set (or inf) then claim
            if nb_dist < distances_arr[nb_y, nb_x]:
                distances_arr[nb_y, nb_x] = nb_dist
                if not claimed_arr[nb_y, nb_x]:
                    claimed_arr[nb_y, nb_x] = True
                    pending_arr[nb_y, nb_x] = True
        # break if search for first instance
        if break_first is True and np.any(targets_arr):
            break
        # break if target cell count has been met
        if break_count is not None and np.sum(targets_arr) >= break_count:
            break
        # otherwise finds all in distance
    return targets_arr


@njit
def _count_cont_nbs(state_arr: Any, y_idx: int, x_idx: int, target_vals: list[int]) -> tuple[int, int, int]:
    """Counts continuous green space neighbours"""
    circle: list[int] = []
    for y_nb_idx, x_nb_idx in _iter_nbs(state_arr, y_idx, x_idx, rook=False):
        if state_arr[y_nb_idx, x_nb_idx] in target_vals:
            circle.append(1)
        else:
            circle.append(0)
    # additions
    adds: list[int] = []
    run = 0
    for c in circle:
        if c == 1:
            run += 1
        elif run > 0:
            adds.append(run)
            run = 0
    if run > 0:
        adds.append(run)
    # check for wraparound if fully circled
    if len(adds) > 1 and len(circle) == 8 and circle[0] == 1 and circle[-1] == 1:
        # add last to first
        adds[0] += adds[-1]
        # remove last
        adds = adds[:-1]
    if not adds:
        return 0, 0, 0
    return sum(adds), max(adds), len(adds)


@njit
def green_span(arr_1d: Any, start_idx: int, positive: bool) -> int:
    """ """
    if positive:
        idxs = list(range(start_idx + 1, len(arr_1d)))
    else:
        idxs = list(range(start_idx - 1, -1, -1))
    span = 0
    for span_idx in idxs:
        if span_idx == len(arr_1d):
            break
        if arr_1d[span_idx] != 0:
            break
        span += 1
    return span


@njit
def green_spans(
    arr: Any,
    y_idx: int,
    x_idx: int,
    granularity_m: int,
    min_green_span_m: int,
) -> bool:
    """ """
    # x spans
    x_span_l = green_span(arr[y_idx, :], x_idx, positive=False)
    x_span_r = green_span(arr[y_idx, :], x_idx, positive=True)
    x_spans = sorted([x_span_l, x_span_r])
    # y spans
    y_span_l = green_span(arr[:, x_idx], y_idx, positive=False)
    y_span_r = green_span(arr[:, x_idx], y_idx, positive=True)
    y_spans = sorted([y_span_l, y_span_r])
    # mins and maxes
    xy_mins = sorted([x_spans[0], y_spans[0]])
    xy_maxs = sorted([x_spans[-1], y_spans[-1]])
    # gaps greater than zero must meet the span
    span_blocks = min_green_span_m / granularity_m
    if xy_mins[0] == 0 and xy_maxs[1] > span_blocks:
        return True
    return False


# @njit
def _green_to_built(
    y_idx: int,
    x_idx: int,
    state_arr: Any,
    old_green_itx_arr: Any,
    old_green_acc_arr: Any,
    buildable_arr: Any,
    granularity_m: int,
    max_distance_m: int,
    min_green_cont_km2: int = 1,
) -> tuple[bool, Any, Any]:
    """
    can't track state directly... because local actions have non-local impact
    avoids checking each reachable cell's green access causes exponential complexity
    """
    new_green_itx_arr = np.copy(old_green_itx_arr)
    new_green_acc_arr = np.copy(old_green_acc_arr)
    # check neighbours situation
    _tot_urban_nbs, cont_urban_nbs, urban_regions = _count_cont_nbs(state_arr, y_idx, x_idx, [1, 2])
    # if buildable_arr indicates that a green area should not be developed, then special conditions apply
    if buildable_arr[y_idx, x_idx] < 1:
        # allow filling in crimped areas
        if cont_urban_nbs < 5:
            return False, old_green_itx_arr, old_green_acc_arr
    # enforce green spans
    ## if not green_spans(state_arr, y_idx, x_idx, granularity_m, min_green_span_m=200):
    ##     return False, old_green_itx_arr, old_green_acc_arr
    # if splitting green into two regions
    ## if urban_regions > 1:
    ##     # required number of contiguous green cells
    ##     target_count = int((min_green_cont_km2 * 1000**2) / granularity_m**2)
    ##     # use a mock state - otherwise dijkstra doesn't know that current y, x is tentatively built
    ##     mock_state_arr = np.copy(state_arr)
    ##     # mock built state
    ##     mock_state_arr[y_idx, x_idx] = 1
    ##     # review neighbours in turn to check that each has access to continuous green space
    ##     for y_nb_idx, x_nb_idx in _iter_nbs(new_green_itx_arr, y_idx, x_idx, rook=False):
    ##         if not state_arr[y_nb_idx, x_nb_idx] == 0:
    ##             continue
    ##         nb_green_acc_arr = _agg_dijkstra_cont(
    ##             mock_state_arr,
    ##             y_nb_idx,
    ##             x_nb_idx,
    ##             [0],
    ##             [0],
    ##             max_distance_m=max_distance_m * 2,
    ##             granularity_m=granularity_m,
    ##             break_count=target_count,
    ##             rook=True,  # rook has to be true otherwise diagonal steps are allowed
    ##         )
    ##         if nb_green_acc_arr.sum() < target_count:
    ##             return False, old_green_itx_arr, old_green_acc_arr
    # check if cell is currently green_itx
    # if so, set itx to off and decrement green access accordingly
    if new_green_itx_arr[y_idx, x_idx] == 2:
        new_green_itx_arr[y_idx, x_idx] = 1
        # decrement green access as consequence of converting cell from itx to built
        new_green_acc_arr -= _agg_dijkstra_cont(
            new_green_itx_arr, y_idx, x_idx, [0, 1, 2], [0, 1, 2], max_distance_m, granularity_m
        )
    # scan through neighbours - set new green itx - use rook for checking contiguity
    for y_nb_idx, x_nb_idx in _iter_nbs(new_green_itx_arr, y_idx, x_idx, rook=True):
        # convert green space to itx
        if new_green_itx_arr[y_nb_idx, x_nb_idx] == 0:
            new_green_itx_arr[y_nb_idx, x_nb_idx] = 2
            # increment green access to existing built cells
            new_green_acc_arr += _agg_dijkstra_cont(
                new_green_itx_arr, y_nb_idx, x_nb_idx, [0, 1, 2], [0, 1, 2], max_distance_m, granularity_m
            )
    # check that green access has not been cut off for built areas
    # green_acc_diff = new_green_acc_arr - old_green_acc_arr
    ny_idxs, nx_idxs = np.nonzero(np.logical_and(state_arr > 0, new_green_acc_arr <= 0))
    for ny_idx, nx_idx in zip(ny_idxs, nx_idxs):
        # bail if built and below zero
        if old_green_acc_arr[ny_idx, nx_idx] > new_green_acc_arr[ny_idx, nx_idx]:
            return False, old_green_itx_arr, old_green_acc_arr
    return True, new_green_itx_arr, new_green_acc_arr


@njit
def _prepare_green_arrs(state_arr: Any, max_distance_m: int, granularity_m: int) -> Any:
    """
    Initialises green itx and green acc arrays.
    Finds cells on the periphery of green areas and adjacent to built areas.
    """
    # prepare green itx array
    green_itx_arr = np.full(state_arr.shape, 0)
    for y_idx, x_idx in np.ndindex(state_arr.shape):
        # if built space
        if state_arr[y_idx, x_idx] > 0:
            # set to 1
            green_itx_arr[y_idx, x_idx] = 1
            # find green neighbours
            for y_nb_idx, x_nb_idx in _iter_nbs(state_arr, y_idx, x_idx, rook=True):
                # set to 2
                if state_arr[y_nb_idx, x_nb_idx] == 0:
                    green_itx_arr[y_nb_idx, x_nb_idx] = 2
    # prepare green access arr
    green_acc_arr = np.full(state_arr.shape, 0, dtype=np.int_)
    for y_idx, x_idx in np.ndindex(state_arr.shape):
        # agg green itx cells to surrounding space
        if green_itx_arr[y_idx, x_idx] == 2:
            green_acc_arr += _agg_dijkstra_cont(
                green_itx_arr,
                y_idx,
                x_idx,
                path_state=[0, 1, 2],
                target_state=[0, 1, 2],
                max_distance_m=max_distance_m,
                granularity_m=granularity_m,
            )
    return green_itx_arr, green_acc_arr


class Land:
    """
    state_arr - is this necessary?
    -1 - out of bounds
    0 - nature
    1 - built
    2 - centre
    """

    # substrate
    iters: int
    granularity_m: int
    max_distance_m: int
    trf: transform.Affine
    # parameters
    build_prob: float
    cent_prob_nb: float  # TODO: pending
    cent_prob_isol: float  # TODO: pending
    max_local_pop: int
    prob_distribution: tuple[float, float, float]
    density_factors: tuple[float, float, float]
    # state - QGIS / gdal numpy veresion doesn't yet support numpy typing for NDArray
    state_arr: Any
    green_itx_arr: Any
    green_acc_arr: Any
    cent_acc_arr: Any
    density_arr: Any
    min_green_km2: int | float

    def __init__(
        self,
        granularity_m: int,
        max_distance_m: int,
        extents_transform: transform.Affine,
        extents_arr: Any,
        centre_seeds: list[tuple[int, int]] = [],
        build_prob: float = 0.1,
        cent_prob_nb: float = 0.05,
        cent_prob_isol: float = 0,
        max_local_pop: int = 10000,
        prob_distribution: tuple[float, float, float] = (0.7, 0.3, 0),
        density_factors: tuple[float, float, float] = (1, 0.1, 0.01),
        min_green_km2: int | float = 1,
        random_seed: int = 0,
    ):
        """ """
        np.random.seed(random_seed)
        self.iters = 0
        # prepare extents
        self.granularity_m = granularity_m
        self.max_distance_m = max_distance_m
        self.trf = extents_transform
        # params
        self.build_prob = build_prob
        self.cent_prob_nb = cent_prob_nb
        self.cent_prob_isol = cent_prob_isol
        # the assumption is that T_star is the number of blocks
        # that equals to a 15 minutes walk, i.e. roughly 1 km. 1 block has size 1000/T_star metres
        if not np.sum(prob_distribution) == 1:
            raise ValueError("The prob_distribution parameter must sum to 1.")
        self.max_local_pop = max_local_pop
        self.prob_distribution = prob_distribution
        self.density_factors = density_factors
        # 0 = green, 1 = built, 2 = centrality
        self.state_arr = np.full(extents_arr.shape, 0, dtype=np.int16)
        self.state_arr[:, :] = extents_arr
        # seed centres
        self.cent_acc_arr = np.full(extents_arr.shape, 0, dtype=np.int16)
        y_trf: int
        x_trf: int
        for east, north in centre_seeds:
            x_trf, y_trf = transform.rowcol(self.trf, east, north)  # type: ignore
            self.state_arr[y_trf, x_trf] = 2
            # agg centrality to surroundings
            self.cent_acc_arr += _agg_dijkstra_cont(
                self.state_arr,
                y_trf,
                x_trf,
                path_state=[0, 1, 2],
                target_state=[0, 1, 2],
                max_distance_m=max_distance_m,
                granularity_m=granularity_m,
            )
        # check size vs. min green
        area = self.state_arr.shape[0] * granularity_m * self.state_arr.shape[1] * granularity_m
        if area / 1000**2 < 2 * min_green_km2:
            raise ValueError("Please decrease min_green_km2 in relation to provided extents.")
        # find boundary of built land
        # 0 = green, 1 = built, 2 = itx bounds
        self.green_itx_arr, self.green_acc_arr = _prepare_green_arrs(self.state_arr, max_distance_m, granularity_m)
        # density
        self.density_arr = np.full(extents_arr.shape, 0, dtype=np.float32)
        self.min_green_km2 = min_green_km2
        # buildable_arr is set by iter
        self.buildable_arr = np.full(extents_arr.shape, 0, dtype=np.int16)

    def iterate(self):
        """ """
        self.iters += 1
        # extract green space features
        feats: list[tuple[dict, float]] = features.shapes(  # type: ignore
            self.state_arr, mask=self.state_arr == 0, connectivity=4, transform=self.trf
        )
        # prime buildable_arr
        self.buildable_arr.fill(0)
        for feat, _val in feats:  # type: ignore
            poly = geometry.shape(feat)  # convert from geo interface to shapely geom
            # convert to square km - continue if below min threshold
            if poly.area < self.min_green_km2 * 1000**2:
                continue
            # reverse buffer step 1
            buffer_dist = 100
            rev_buf: geometry.Polygon | geometry.MultiPolygon = poly.buffer(
                -buffer_dist, cap_style="square", join_style="mitre"
            )
            geoms: list[geometry.Polygon] = []
            # if an area is split, a MultiPolygon is returned
            if isinstance(rev_buf, geometry.Polygon):
                geoms.append(rev_buf)
            elif isinstance(rev_buf, geometry.MultiPolygon):
                geoms += rev_buf.geoms
            else:
                raise ValueError("Unexpected geometry")
            buildable_geom: geometry.MultiPolygon = geometry.MultiPolygon(polygons=None)
            for geom in geoms:
                back_buf = geom.buffer(buffer_dist, cap_style="square", join_style="mitre")
                # clip for situations where approaching borders
                back_buf = back_buf.intersection(poly)
                # add to buildable if larger than min threshold
                if back_buf.area >= self.min_green_km2 * 1000**2:
                    buildable_geom = buildable_geom.union(back_buf)
            # generate a negative of green extents vs. deemed buildable extents
            unbuildable_geom: geometry.MultiPolygon = geometry.MultiPolygon(polygons=None)
            neg_extents: geometry.Polygon | geometry.MultiPolygon = poly.difference(buildable_geom)
            if not neg_extents.is_empty:
                # cycle buffer to cleanly separate
                neg_geoms: list[geometry.Polygon] = []
                # if an area is split, a MultiPolygon is returned
                if isinstance(neg_extents, geometry.Polygon) and not neg_extents.is_empty:
                    neg_geoms.append(neg_extents)
                elif isinstance(neg_extents, geometry.MultiPolygon) and not neg_extents.is_empty:
                    neg_geoms += neg_extents.geoms
                # sort through neg geoms and assign to buildable or non buildable based on sizes
                for neg_geom in neg_geoms:
                    # look for smaller chunks
                    neg_buf: geometry.Polygon | geometry.MultiPolygon = neg_geom.buffer(
                        self.granularity_m, cap_style="square", join_style="mitre"
                    )
                    # if smaller than min - then discard
                    if neg_buf.area < self.min_green_km2 * 1000**2:
                        unbuildable_geom = unbuildable_geom.union(neg_buf)  # type: ignore
                        # difference padded from buildable to shelter new parks from rapid infill
                        buildable_geom = buildable_geom.difference(neg_buf)
                    # otherwise salvage as buildable
                    else:
                        buildable_geom = buildable_geom.union(neg_buf)
            # burn raster
            if not buildable_geom.is_empty:
                features.rasterize(  # type: ignore
                    shapes=[(geometry.mapping(buildable_geom), 1)],  # convert back to geo interface
                    out=self.buildable_arr,
                    transform=self.trf,
                    all_touched=False,
                )
            if not unbuildable_geom.is_empty:
                features.rasterize(  # type: ignore
                    shapes=[(geometry.mapping(unbuildable_geom), -1)],  # convert back to geo interface
                    out=self.buildable_arr,
                    transform=self.trf,
                    all_touched=False,
                )
            # plt.plot(poly.exterior.xy[0], poly.exterior.xy[1])
            # plt.plot(pos_poly.exterior.xy[0], pos_poly.exterior.xy[1])
            # plt.show()
            if False:
                scale_factor = 1 / self.granularity_m
                plt.plot(poly.exterior.xy[0], poly.exterior.xy[1])
                plt.plot(pos_poly.exterior.xy[0], pos_poly.exterior.xy[1])
                plt.show()
        # shuffle indices
        arr_idxs = list(np.ndindex(self.state_arr.shape))
        np.random.shuffle(arr_idxs)
        old_state_arr = np.copy(self.state_arr)
        for y_idx, x_idx in arr_idxs:
            # bail if already built
            if self.state_arr[y_idx, x_idx] > 0:
                continue
            # a cell can be developed if it is on the green periphery adjacent to built areas
            if self.green_itx_arr[y_idx, x_idx] == 2:
                # don't allow double steps, i.e. a new built cell has to have at least one built neighbour in
                tot_urban_nbs, _cont_urban_nbs, _urban_regions = _count_cont_nbs(old_state_arr, y_idx, x_idx, [1, 2])
                if tot_urban_nbs == 0:
                    continue
                # if centrality is accessible
                if self.cent_acc_arr[y_idx, x_idx] > 0:
                    if np.random.rand() < self.build_prob:
                        # update green state
                        success, self.green_itx_arr, self.green_acc_arr = _green_to_built(
                            y_idx,
                            x_idx,
                            self.state_arr,
                            self.green_itx_arr,
                            self.green_acc_arr,
                            self.buildable_arr,
                            self.granularity_m,
                            self.max_distance_m,
                        )
                        # claim as built
                        if success is True:
                            # state
                            self.state_arr[y_idx, x_idx] = 1
                            # set random density
                            self.density_arr[y_idx, x_idx] = _random_density(
                                self.prob_distribution, self.density_factors
                            )
                # otherwise, consider adding a new centrality
                elif np.random.rand() < self.cent_prob_nb:
                    # update green state
                    success, self.green_itx_arr, self.green_acc_arr = _green_to_built(
                        y_idx,
                        x_idx,
                        self.state_arr,
                        self.green_itx_arr,
                        self.green_acc_arr,
                        self.buildable_arr,
                        self.granularity_m,
                        self.max_distance_m,
                    )
                    if success is True:
                        # state
                        self.state_arr[y_idx, x_idx] = 2
                        # increment centrality access
                        self.cent_acc_arr += _agg_dijkstra_cont(
                            self.state_arr,
                            y_idx,
                            x_idx,
                            path_state=[0, 1, 2],
                            target_state=[0, 1, 2],
                            max_distance_m=self.max_distance_m,
                            granularity_m=self.granularity_m,
                        )
                        # set random density
                        self.density_arr[y_idx, x_idx] = _random_density(self.prob_distribution, self.density_factors)
            # handle random conversion of green space to centralities
            elif self.state_arr[y_idx, x_idx] == 0:
                if np.random.rand() < 0:  # self.cent_prob_isol:
                    # if self.nature_stays_extended(x, y):
                    # if self.nature_stays_reachable(x, y):
                    # update green state
                    success, self.green_itx_arr, self.green_acc_arr = _green_to_built(
                        y_idx,
                        x_idx,
                        self.state_arr,
                        self.green_itx_arr,
                        self.green_acc_arr,
                        self.buildable_arr,
                        self.granularity_m,
                        self.max_distance_m,
                    )
                    if success is True:
                        # claim as built
                        self.state_arr[y_idx, x_idx] = 2
                        self.cent_acc_arr = _inc_access(
                            y_idx, x_idx, self.cent_acc_arr, self.granularity_m, self.max_distance_m
                        )
                        # set random density
                        self.density_arr[y_idx, x_idx] = _random_density(self.prob_distribution, self.density_factors)

    @property
    def population_density(self) -> dict[str, float]:
        """ """
        return {
            "high": self.density_factors[0],
            "medium": self.density_factors[1],
            "low": self.density_factors[2],
            "empty": 0,
        }


'''
@njit
def _compute_arr_cont_access(
    state_arr: Any, seed_state: int, path_state: list[int], max_distance_m: int, granularity_m: int
) -> Any:
    """Computes accessibility surface to centres"""
    agg_arr = np.full(state_arr.shape, 0, dtype=np.float32)
    for y_idx, x_idx in np.ndindex(state_arr.shape):
        if state_arr[y_idx, x_idx] != seed_state:
            continue
        claimed_arr = _agg_dijkstra_cont(state_arr, y_idx, x_idx, path_state, max_distance_m, granularity_m)
        agg_arr += claimed_arr
    return agg_arr

@njit
def _compute_arr_access(state_arr: Any, target_state: int, granularity_m: int, max_distance_m: int) -> Any:
    """Computes accessibility surface to centres"""
    arr = np.full(state_arr.shape, 0, dtype=np.float32)
    for y_idx, x_idx in np.ndindex(state_arr.shape):
        # bail if not a centre
        if state_arr[y_idx, x_idx] != target_state:
            continue
        # otherwise, agg access to surrounding extents
        arr = _inc_access(y_idx, x_idx, arr, granularity_m, max_distance_m)
    return arr

@njit
def recurse_gobble(
    arr: Any,
    y_idx: int,
    x_idx: int,
    target_state: int,
    cell_counter: int,
    target_cell_count: int,
    max_recurse_depth: int,
    last_recurse_depth: int,
    visited_arr: Any,
) -> tuple[int, Any]:
    """
    0 - not visited
    1 - visited and claimed
    """
    # explore neighbours
    for nb_y_idx, nb_x_idx in _iter_nbs(arr, y_idx, x_idx, rook=False):
        # ignore if already visited
        if visited_arr[nb_y_idx, nb_x_idx] != 1:
            continue
        # ignore if not target value
        if arr[nb_y_idx, nb_x_idx] != target_state:
            continue
        # otherwise claim
        visited_arr[nb_y_idx, nb_x_idx] = 1
        cell_counter += 1
        # break if target cells reached
        if cell_counter >= target_cell_count:
            break
        # halt recursion if max recursion depth reached
        this_recurse_depth = last_recurse_depth + 1
        if this_recurse_depth == max_recurse_depth:
            continue
        # otherwise recurse
        cell_counter, visited_arr = recurse_gobble(
            arr,
            nb_y_idx,
            nb_x_idx,
            target_state,
            cell_counter,
            target_cell_count,
            max_recurse_depth,
            this_recurse_depth,
            visited_arr,
        )
    return cell_counter, visited_arr

@njit
def continuous_state_extents(
    arr: Any,
    y_idx: int,
    x_idx: int,
    target_state: int,
    target_area_m: int,
    max_dist_m: int,
    granularity_m: int,
) -> bool:
    """ """
    cell_counter = 0
    target_cell_count = int(np.ceil(target_area_m / granularity_m**2))
    max_recurse_depth = int(np.floor(max_dist_m / granularity_m))
    visited_arr = np.full(arr.shape, 0)
    visited_arr[y_idx, x_idx] = 1
    cell_counter, visited_arr = recurse_gobble(
        arr,
        y_idx,
        x_idx,
        target_state,
        cell_counter,
        target_cell_count,
        max_recurse_depth,
        last_recurse_depth=0,
        visited_arr=visited_arr,
    )
    return cell_counter >= target_cell_count
'''
