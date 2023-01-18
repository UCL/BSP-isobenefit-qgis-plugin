"""
Algos for Numba
"""
from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit


@njit
def random_density(prob_distribution: tuple[float, float, float], density_factors: tuple[float, float, float]) -> float:
    """Numba compatible method for determining a land use density"""
    p = np.random.rand()
    if p >= 0 and p < prob_distribution[0]:
        return density_factors[0]
    elif p >= prob_distribution[1] and p < prob_distribution[2]:
        return density_factors[1]
    else:
        return density_factors[2]


# @njit
def _inc_access(y_idx: int, x_idx: int, arr: Any, granularity_m: int, max_distance_m: int) -> Any:
    """increment access"""
    return _agg_access(y_idx, x_idx, arr, granularity_m, max_distance_m, positive=True)


# @njit
def _decr_access(y_idx: int, x_idx: int, arr: Any, granularity_m: int, max_distance_m: int) -> Any:
    """increment access"""
    return _agg_access(y_idx, x_idx, arr, granularity_m, max_distance_m, positive=False)


# @njit
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
def iter_nbs(arr: Any, y_idx: int, x_idx: int, rook: bool) -> Any:
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
def agg_dijkstra_cont(
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
        for nb_y, nb_x in iter_nbs(distances_arr, next_y, next_x, rook=rook):
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
def count_cont_nbs(state_arr: Any, y_idx: int, x_idx: int, target_vals: list[int]) -> tuple[int, int, int]:
    """Counts continuous green space neighbours"""
    circle: list[int] = []
    for y_nb_idx, x_nb_idx in iter_nbs(state_arr, y_idx, x_idx, rook=False):
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


# @njit
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


# @njit
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


@njit
def prepare_green_arrs(state_arr: Any, max_distance_m: int, granularity_m: int) -> Any:
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
            for y_nb_idx, x_nb_idx in iter_nbs(state_arr, y_idx, x_idx, rook=True):
                # set to 2
                if state_arr[y_nb_idx, x_nb_idx] == 0:
                    green_itx_arr[y_nb_idx, x_nb_idx] = 2
    # prepare green access arr
    green_acc_arr = np.full(state_arr.shape, 0, dtype=np.int_)
    for y_idx, x_idx in np.ndindex(state_arr.shape):
        # agg green itx cells to surrounding space
        if green_itx_arr[y_idx, x_idx] == 2:
            green_acc_arr += agg_dijkstra_cont(
                green_itx_arr,
                y_idx,
                x_idx,
                path_state=[0, 1, 2],
                target_state=[0, 1, 2],
                max_distance_m=max_distance_m,
                granularity_m=granularity_m,
            )
    return green_itx_arr, green_acc_arr
