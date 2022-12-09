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
def _inc_access(y_idx: int, x_idx: int, arr: Any, granularity_m: int, walk_dist_m: int) -> Any:
    """increment access"""
    return _agg_access(y_idx, x_idx, arr, granularity_m, walk_dist_m, positive=True)


@njit
def _decr_access(y_idx: int, x_idx: int, arr: Any, granularity_m: int, walk_dist_m: int) -> Any:
    """increment access"""
    return _agg_access(y_idx, x_idx, arr, granularity_m, walk_dist_m, positive=False)


@njit
def _agg_access(y_idx: int, x_idx: int, arr: Any, granularity_m: int, walk_dist_m: int, positive: bool) -> Any:
    """Aggregates access - from an x, y to provided array, either positive or negative"""
    for cy_idx, cx_idx in np.ndindex(arr.shape):
        y_dist = int(abs(y_idx - cy_idx) * granularity_m)
        x_dist = int(abs(x_idx - cx_idx) * granularity_m)
        dist = np.hypot(x_dist, y_dist)
        if dist > walk_dist_m:
            continue
        val = 1  #  - dist / walk_dist_m
        if positive:
            arr[cy_idx, cx_idx] += val
        else:
            arr[cy_idx, cx_idx] -= val
    return arr


@njit
def _compute_arr_access(state_arr: Any, target_state: int, granularity_m: int, walk_dist_m: int) -> Any:
    """Computes accessibility surface to centres"""
    arr = np.full(state_arr.shape, 0, dtype=np.float32)
    for y_idx, x_idx in np.ndindex(state_arr.shape):
        # bail if not a centre
        if state_arr[y_idx, x_idx] != target_state:
            continue
        # otherwise, agg access to surrounding extents
        arr = _inc_access(y_idx, x_idx, arr, granularity_m, walk_dist_m)
    return arr


@njit
def _iter_nbs(arr: Any, y_idx: int, x_idx: int, rook: bool) -> list[tuple[int, int]]:
    """Returns rook or queen neighbours"""
    idxs: list[tuple[int, int]] = []
    if rook:
        y_offsets = [1, 0, -1, 0]
        x_offsets = [0, 1, 0, -1]
    else:
        y_offsets = [1, 1, 1, 0, -1, -1, -1, 0]
        x_offsets = [-1, 0, 1, 1, 1, 0, -1, -1]
    for y_offset, x_offset in zip(y_offsets, x_offsets):
        y_nb_idx = y_idx + y_offset
        if y_nb_idx < 0 or y_nb_idx >= arr.shape[0]:
            continue
        x_nb_idx = x_idx + x_offset
        if x_nb_idx < 0 or x_nb_idx >= arr.shape[1]:
            continue
        idxs.append((y_nb_idx, x_nb_idx))
    return idxs


@njit
def _count_cont_nbs(state_arr: Any, y_idx: int, x_idx: int, target_vals: list[int]) -> tuple[int, int]:
    """Counts continuous green space neighbours"""
    circle: list[int] = []
    for x_nb_idx, y_nb_idx in _iter_nbs(state_arr, y_idx, x_idx, rook=False):
        if state_arr[x_nb_idx, y_nb_idx] in target_vals:
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


# @njit
def _green_to_built(
    y_idx: int,
    x_idx: int,
    state_arr: Any,
    old_green_itx_arr: Any,
    old_green_acc_arr: Any,
    granularity_m: int,
    walk_dist_m: int,
    min_green_access: int = 50,
    min_green_contig: int = 100,
) -> tuple[bool, Any, Any]:
    """Check a built cell's neighbours - set to itx if green space - update green access"""
    # check if cell is currently green_itx
    # if so, set to itx to off and decrement green access accordingly
    new_green_itx_arr = np.copy(old_green_itx_arr)
    new_green_acc_arr = np.copy(old_green_acc_arr)
    if new_green_itx_arr[y_idx, x_idx] == 1:
        new_green_itx_arr[y_idx, x_idx] = 0
        new_green_acc_arr = _decr_access(y_idx, x_idx, new_green_acc_arr, granularity_m, walk_dist_m)
    # bail if the new built cell breaks green access for built areas
    min_green_idxs = np.where(new_green_acc_arr < min_green_access)
    if np.any(old_green_acc_arr[min_green_idxs] > new_green_acc_arr[min_green_idxs]):
        return False, old_green_itx_arr, old_green_acc_arr
    # check green space contiguity
    if min_green_contig is not None:
        new_clip_idxs = np.where(new_green_acc_arr < min_green_contig)
        if np.any(
            np.logical_and(
                # vs > min_green_contig[new_clip_idxs]
                old_green_acc_arr[new_clip_idxs] > new_green_acc_arr[new_clip_idxs],
                state_arr[new_clip_idxs] == 0,
            )
        ):
            return False, old_green_itx_arr, old_green_acc_arr
    # scan through neighbours
    for x_nb_idx, y_nb_idx in _iter_nbs(new_green_itx_arr, y_idx, x_idx, rook=True):
        # if neighbour is currently green space and not already itx, set as new itx
        if state_arr[x_nb_idx, y_nb_idx] == 0 and new_green_itx_arr[x_nb_idx, y_nb_idx] == 0:
            new_green_itx_arr[x_nb_idx, y_nb_idx] = 1
    # return accordingly
    return True, new_green_itx_arr, new_green_acc_arr


# @njit
def _prepare_green_arrs(state_arr: Any, granularity_m: int, walk_dist_m: int) -> tuple[Any, Any]:
    """
    Initialises green itx and green acc arrays.
    Finds cells on the periphery of green areas and adjacent to built areas.
    """
    green_itx_arr = np.full(state_arr.shape, 0, np.int16)
    green_acc_arr = _compute_arr_access(state_arr, target_state=0, granularity_m=granularity_m, walk_dist_m=walk_dist_m)
    for y_idx, x_idx in np.ndindex(state_arr.shape):
        # bail if not built space
        if state_arr[y_idx, x_idx] < 1:
            continue
        # otherwise, look to see if the cell borders green space
        _success, green_itx_arr, green_acc_arr = _green_to_built(
            y_idx, x_idx, state_arr, green_itx_arr, green_acc_arr, granularity_m, walk_dist_m
        )
    return green_itx_arr, green_acc_arr


@njit
def green_ray(arr_1d: Any, start_idx: int, positive: bool) -> int:
    """ """
    if positive:
        idxs = list(range(start_idx + 1, len(arr_1d)))
    else:
        idxs = list(range(start_idx - 1, -1, -1))
    ray = 0
    for ray_idx in idxs:
        if ray_idx == len(arr_1d):
            break
        if arr_1d[ray_idx] != 0:
            break
        ray += 1
    return ray


@njit
def green_rays(
    arr: Any,
    y_idx: int,
    x_idx: int,
    granularity_m: int,
    min_long_green_span_m: int,
    min_short_green_span_m: int,
) -> bool:
    """ """
    x_ray_l = green_ray(arr[y_idx, :], x_idx, positive=False)
    x_ray_r = green_ray(arr[y_idx, :], x_idx, positive=True)
    #
    x_rays = sorted([x_ray_l, x_ray_r])
    y_ray_l = green_ray(arr[:, x_idx], y_idx, positive=False)
    y_ray_r = green_ray(arr[:, x_idx], y_idx, positive=True)
    y_rays = sorted([y_ray_l, y_ray_r])
    xy_max = sorted([x_rays[-1], y_rays[-1]])
    # allow filling in single cell spaces - i.e. directly neighboured on each side by built area
    if sum(x_rays) != 0:
        # allow filling in one-sided spaces - i.e. directly neighboured on one side
        if x_rays[0] != 0:
            # otherwise, minimum side must be at least short green span
            if x_rays[0] < min_short_green_span_m / granularity_m:
                return False
    # per above on y axis
    if sum(y_rays) != 0:
        if y_rays[0] != 0:
            if y_rays[0] < min_short_green_span_m / granularity_m:
                return False
    # the overall max must be larger than min long
    if xy_max[-1] < min_long_green_span_m / granularity_m:
        return False
    # the secondary max must be larger than min short
    # if xy_max[0] < min_short_green_span_m / granularity_m:
    #     return False
    return True


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
        if visited_arr[nb_y_idx, nb_x_idx] != target_state:
            continue
        # ignore if not target value
        if arr[nb_y_idx, nb_x_idx] != target_state:
            break
        # otherwise claim
        visited_arr[nb_y_idx, nb_x_idx] = 1
        cell_counter += 1
        # break if target cells reached
        if cell_counter >= target_cell_count:
            break
        # halt recursion if max recursion depth reached
        this_recurse_depth = last_recurse_depth + 1
        if this_recurse_depth == max_recurse_depth:
            break
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
    visited_arr = np.full(arr.shape, 0, dtype=np.uint)
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
    walk_dist_m: int
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
    cent_acc_arr: Any
    areas_arr: Any | None
    min_green_km2: int
    min_long_green_span_m: int
    min_short_green_span_m: int

    def __init__(
        self,
        granularity_m: int,
        walk_dist_m: int,
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
        min_long_green_span_m: int = 500,
        min_short_green_span_m: int = 100,
        random_seed: int = 0,
    ):
        """ """
        np.random.seed(random_seed)
        self.iters = 0
        # prepare extents
        self.granularity_m = granularity_m
        self.walk_dist_m = walk_dist_m
        self.trf = extents_transform
        # params
        self.build_prob = build_prob
        self.cent_prob_nb = cent_prob_nb
        self.cent_prob_isol = cent_prob_isol
        # the assumption is that T_star is the number of blocks
        # that equals to a 15 minutes walk, i.e. roughly 1 km. 1 block has size 1000/T_star metres
        self.max_local_pop = max_local_pop
        self.prob_distribution = prob_distribution
        self.density_factors = density_factors
        # state - cast type to int16
        self.state_arr = np.full(extents_arr.shape, 0, dtype=np.int16)
        self.state_arr[:, :] = extents_arr
        # check size vs. min green
        area = self.state_arr.shape[0] * granularity_m * self.state_arr.shape[1] * granularity_m
        if area / (1000 * 1000) < 2 * min_green_km2:
            raise ValueError("Please decrease min_green_km2 in relation to extents.")
        # seed centres
        for east, north in centre_seeds:
            x, y = transform.rowcol(self.trf, east, north)  # type: ignore
            self.state_arr[x, y] = 2
        # find boundary of built land
        self.green_itx_arr, self.green_acc_arr = _prepare_green_arrs(
            state_arr=self.state_arr, granularity_m=granularity_m, walk_dist_m=walk_dist_m
        )
        # density
        self.density_arr = np.full(extents_arr.shape, 0, dtype=np.float32)
        # areas array is set by iter
        self.areas_arr = None
        # access to centres
        self.cent_acc_arr = _compute_arr_access(
            state_arr=self.state_arr,
            target_state=2,
            granularity_m=granularity_m,
            walk_dist_m=walk_dist_m,
        )
        self.min_green_km2 = min_green_km2
        self.min_long_green_span_m = min_long_green_span_m
        self.min_short_green_span_m = min_short_green_span_m

    def iter_land_isobenefit(self):
        """ """
        self.iters += 1
        # extract green space features
        feats: list[tuple[dict, float]] = features.shapes(  # type: ignore
            self.state_arr, mask=self.state_arr == 0, connectivity=4, transform=self.trf
        )
        # unpack each feature's area
        out_feats: list[tuple[dict, int]] = []  # type: ignore
        for feat, _val in feats:  # type: ignore
            # convert to square km
            area = int(shape(feat).area / (1000 * 1000))  # type: ignore
            out_feats.append((feat, area))  # type: ignore
        # create an array showing available
        # use int32 for large enough area integers
        self.areas_arr = features.rasterize(  # type: ignore
            shapes=out_feats,
            out_shape=self.state_arr.shape,
            fill=-1,
            transform=self.trf,
            all_touched=True,
            dtype=np.int32,
        )
        added_blocks = 0
        added_centrality = 0
        for y_idx, x_idx in np.ndindex(self.state_arr.shape):
            # if a cell is on the green periphery adjacent to built areas
            if self.green_itx_arr[y_idx, x_idx] == 1:
                # TODO: set minimum contiguous green access, e.g. 16?
                # only develop a cell if it has at least two urban neighbours - i.e. not diagonally
                urban_nbs, urban_regions = _count_cont_nbs(self.state_arr, y_idx, x_idx, [1, 2])
                green_nbs, green_regions = _count_cont_nbs(self.state_arr, y_idx, x_idx, [0])
                # if urban_regions > 1:
                #     continue
                if not continuous_state_extents(self.state_arr, y_idx, x_idx, 0, 4000, 100, 50):
                    continue
                if not green_rays(self.state_arr, y_idx, x_idx, self.granularity_m, 200, 200):
                    continue
                # if centrality is accessible
                if self.cent_acc_arr[y_idx, x_idx] > 0:
                    # if self.nature_stays_extended(y_idx, x_idx):
                    # if self.nature_stays_reachable(x, y):
                    if np.random.rand() < self.build_prob:
                        # update green state
                        success, self.green_itx_arr, self.green_acc_arr = _green_to_built(
                            y_idx,
                            x_idx,
                            self.state_arr,
                            self.green_itx_arr,
                            self.green_acc_arr,
                            self.granularity_m,
                            self.walk_dist_m,
                        )
                        # claim as built
                        if success is True:
                            # state
                            self.state_arr[y_idx, x_idx] = 1
                            added_blocks += 1
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
                            self.walk_dist_m,
                        )
                        if success is True:
                            # claim as built
                            self.state_arr[y_idx, x_idx] = 2
                            self.cent_acc_arr = _inc_access(
                                y_idx, x_idx, self.cent_acc_arr, self.granularity_m, self.walk_dist_m
                            )
                            added_centrality += 1
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
                        self.walk_dist_m,
                    )
                    if success is True:
                        # claim as built
                        self.state_arr[y_idx, x_idx] = 2
                        self.cent_acc_arr = _inc_access(
                            y_idx, x_idx, self.cent_acc_arr, self.granularity_m, self.walk_dist_m
                        )
                        added_centrality += 1
                        # set random density
                        self.density_arr[y_idx, x_idx] = _random_density(self.prob_distribution, self.density_factors)
        LOGGER.info(f"added blocks: {added_blocks}")
        LOGGER.info(f"added centralities: {added_centrality}")

    @property
    def population_density(self) -> dict[str, float]:
        """ """
        return {
            "high": self.density_factors[0],
            "medium": self.density_factors[1],
            "low": self.density_factors[2],
            "empty": 0,
        }
