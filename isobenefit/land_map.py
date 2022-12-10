"""
Can't handle border regions in numba stencil: only 'constant' mode is currently supported, i.e. border buffer = 0

Don't use anything from QGIS so that it is easier to test this module.
"""
from __future__ import annotations

import copy
import logging
from functools import partial
from typing import Any

import numpy as np
from numba import njit
from rasterio import features, transform
from scipy.ndimage import measurements as measure
from shapely.geometry import shape

from .logger import get_logger

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
    """Returns rook or queen neighbours"""
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
    # change to numpy array for shuffling
    rand_idxs = np.array(idxs)
    np.random.shuffle(rand_idxs)
    return rand_idxs


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
) -> Any:
    """ """
    if break_first is True and break_count is not None:
        raise ValueError("Only one of break_first and break_count can be specified at once.")
    # targets
    targets_arr = np.full(state_arr.shape, False)
    if state_arr[y_idx, x_idx] in target_state:
        targets_arr[y_idx, x_idx] = True
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
        # reset pending
        pending_arr[next_y, next_x] = False
        # retrieve the current distance
        next_dist = distances_arr[next_y, next_x]
        # compound distance to neighbour
        nb_dist = next_dist + granularity_m
        # check for max distance
        if nb_dist > max_distance_m:
            continue
        # explore neighbours
        for nb_y, nb_x in _iter_nbs(distances_arr, next_y, next_x, rook=True):
            # aggregate targets
            if state_arr[nb_y, nb_x] in target_state:
                targets_arr[nb_y, nb_x] = True
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
def _count_cont_nbs(state_arr: Any, y_idx: int, x_idx: int, target_vals: list[int]) -> tuple[int, int]:
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
        return 0, 0
    return max(adds), len(adds)


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
    # encourage filling in from the side, i.e.
    # allow slotting into cells with four, three, or two built neighbours
    # this helps keep growth compact rather than stringy around narrow park corridors
    if sum(xy_mins) == 0 and sum(xy_maxs) == 0:
        return True
    if sum(xy_mins) == 0 and xy_maxs[0] == 0:
        return True
    if sum(xy_mins) == 0:
        return True
    # otherwise, gaps greater than zero must meet the span
    for span in [*xy_mins, *xy_maxs]:
        if span > 0 and span < min_green_span_m / granularity_m:
            return False
    return True


@njit
def _green_to_built(
    y_idx: int,
    x_idx: int,
    state_arr: Any,
    old_green_itx_arr: Any,
    old_green_acc_arr: Any,
    granularity_m: int,
    max_distance_m: int,
    min_green_cont_km2: int = 1,
    min_green_span: int = 100,
) -> tuple[bool, Any, Any]:
    """Check a built cell's neighbours - set to itx if green space - update green access"""
    new_green_itx_arr = np.copy(old_green_itx_arr)
    new_green_acc_arr = np.copy(old_green_acc_arr)
    # check that cell is spatially sensible
    if not green_spans(state_arr, y_idx, x_idx, granularity_m, min_green_span):
        return False, old_green_itx_arr, old_green_acc_arr
    # check if cell is currently green_itx
    # if so, set itx to off and decrement green access accordingly
    if new_green_itx_arr[y_idx, x_idx] == 2:
        new_green_itx_arr[y_idx, x_idx] = 1
        # decrement green access to existing built cells
        new_green_acc_arr -= _agg_dijkstra_cont(state_arr, y_idx, x_idx, [1, 2], [1, 2], max_distance_m, granularity_m)
        # calculate green access for new (current) cell
        new_cell_green_acc = _agg_dijkstra_cont(
            new_green_itx_arr, y_idx, x_idx, [1], [2], max_distance_m, granularity_m
        )
        new_green_acc_arr[y_idx, x_idx] = np.sum(new_cell_green_acc)
    # scan through neighbours - check green contiguity and set new green itx
    target_cell_count = int(np.ceil((min_green_cont_km2 * 1000**2) / granularity_m**2))
    max_search_dist = np.sqrt(min_green_cont_km2) * 1.25 * 1000
    # use rook for checking contiguity
    for y_nb_idx, x_nb_idx in _iter_nbs(new_green_itx_arr, y_idx, x_idx, rook=False):
        # check that if cell is built, the neighbours have sufficient contiguous access to green space
        if new_green_itx_arr[y_nb_idx, x_nb_idx] in [0, 2]:
            claimed_arr = _agg_dijkstra_cont(
                new_green_itx_arr,
                y_nb_idx,
                x_nb_idx,
                [0, 2],
                [0, 2],
                max_search_dist,
                granularity_m,
                break_count=target_cell_count,
            )
            # bail if not enough
            if np.sum(claimed_arr) < target_cell_count:
                return False, old_green_itx_arr, old_green_acc_arr
            # if neighbour is currently green space, on the same axis, and not already itx, then set as new itx
            if y_nb_idx == y_idx or x_nb_idx == x_idx:
                new_green_itx_arr[y_nb_idx, x_nb_idx] = 2
                # increment green access to existing built cells
                new_green_acc_arr += _agg_dijkstra_cont(
                    state_arr, y_nb_idx, x_nb_idx, [1, 2], [1, 2], max_distance_m, granularity_m
                )
    # check that green access has not been cut off
    # mg_y_idxs, mg_x_idxs = np.nonzero(new_green_acc_arr <= 0)
    # for mg_y_idx, mg_x_idx in zip(mg_y_idxs, mg_x_idxs):
    #     if old_green_acc_arr[mg_y_idx, mg_x_idx] > new_green_acc_arr[mg_y_idx, mg_x_idx]:
    #         return False, old_green_itx_arr, old_green_acc_arr
    # return accordingly
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
    green_acc_arr = np.full(state_arr.shape, 0)
    for y_idx, x_idx in np.ndindex(state_arr.shape):
        # agg green itx cells to built space
        if green_itx_arr[y_idx, x_idx] == 2:
            green_acc_arr += _agg_dijkstra_cont(
                green_itx_arr,
                y_idx,
                x_idx,
                path_state=[1],
                target_state=[1],
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
    density_arr: Any
    min_green_km2: int

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
        min_green_km2: int = 5,
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
        # state
        self.state_arr = np.full(extents_arr.shape, 0)
        self.state_arr[:, :] = extents_arr
        # seed centres
        y_trf: int
        x_trf: int
        for east, north in centre_seeds:
            x_trf, y_trf = transform.rowcol(self.trf, east, north)  # type: ignore
            self.state_arr[y_trf, x_trf] = 2
        # check size vs. min green
        area = self.state_arr.shape[0] * granularity_m * self.state_arr.shape[1] * granularity_m
        if area / 1000**2 < 2 * min_green_km2:
            raise ValueError("Please decrease min_green_km2 in relation to extents.")
        # find boundary of built land
        self.green_itx_arr, self.green_acc_arr = _prepare_green_arrs(self.state_arr, max_distance_m, granularity_m)
        # density
        self.density_arr = np.full(extents_arr.shape, 0, dtype=np.float32)
        self.min_green_km2 = min_green_km2

    def iter_land_isobenefit(self):
        """ """
        self.iters += 1
        # shuffle indices
        arr_idxs = list(np.ndindex(self.state_arr.shape))
        np.random.shuffle(arr_idxs)
        for y_idx, x_idx in arr_idxs:
            # if a cell is on the green periphery adjacent to built areas
            if self.green_itx_arr[y_idx, x_idx] == 2:
                # green_nbs, green_regions = _count_cont_nbs(self.state_arr, y_idx, x_idx, [0])
                # urban_nbs, urban_regions = _count_cont_nbs(self.state_arr, y_idx, x_idx, [1, 2])
                # if urban_regions > 1:
                #     continue
                # if centrality is accessible
                if True:  # TODO: self.cent_acc_arr[y_idx, x_idx] > 0:
                    if np.random.rand() < self.build_prob:
                        # update green state
                        success, self.green_itx_arr, self.green_acc_arr = _green_to_built(
                            y_idx,
                            x_idx,
                            self.state_arr,
                            self.green_itx_arr,
                            self.green_acc_arr,
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
                else:
                    if np.random.rand() < self.cent_prob_nb:
                        # if self.nature_stays_extended(x, y):
                        # if self.nature_stays_reachable(x, y):
                        # update green state
                        success, self.green_itx_arr, self.green_acc_arr = _green_to_built(
                            y_idx,
                            x_idx,
                            self.state_arr,
                            self.green_itx_arr,
                            self.green_acc_arr,
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
                            self.density_arr[y_idx, x_idx] = _random_density(
                                self.prob_distribution, self.density_factors
                            )
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
