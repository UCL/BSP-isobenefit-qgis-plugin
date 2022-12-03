"""
Can't handle border regions in numba stencil: only 'constant' mode is currently supported, i.e. border buffer = 0

Don't use anything from QGIS so that it is easier to test this module.

xs, ys = np.where(state_arr == 2)
for x, y in zip(xs, ys):
    for x_idx in range(arr.shape[0]):
        x_dist = abs(x - x_idx) * granularity_m
        if x_dist > walk_dist_m:
            continue
        for y_idx in range(arr.shape[1]):
            y_dist = abs(y - y_idx) * granularity_m
            if y_dist > walk_dist_m:
                continue
            dist = np.hypot(x_dist, y_dist)
            if dist > walk_dist_m:
                continue
            arr[x_idx, y_idx] += dist / walk_dist_m
return arr
"""
from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np
from numba import njit
from rasterio import features

from .logger import get_logger

LOGGER: logging.Logger = get_logger()


@njit
def _inc_access(x_idx: int, y_idx: int, arr: Any, granularity_m: int, walk_dist_m: int) -> Any:
    """increment access"""
    return _agg_access(x_idx, y_idx, arr, granularity_m, walk_dist_m, positive=True)


@njit
def _decr_access(x_idx: int, y_idx: int, arr: Any, granularity_m: int, walk_dist_m: int) -> Any:
    """increment access"""
    return _agg_access(x_idx, y_idx, arr, granularity_m, walk_dist_m, positive=False)


@njit
def _agg_access(x_idx: int, y_idx: int, arr: Any, granularity_m: int, walk_dist_m: int, positive: bool) -> Any:
    """Aggregates access - from an x, y to provided array, either positive or negative"""
    for cx_idx, cy_idx in np.ndindex(arr.shape):
        x_dist = int(abs(x_idx - cx_idx) * granularity_m)
        y_dist = int(abs(y_idx - cy_idx) * granularity_m)
        dist = np.hypot(x_dist, y_dist)
        if dist > walk_dist_m:
            continue
        val = 1 - dist / walk_dist_m
        if positive:
            arr[cx_idx, cy_idx] += val
        else:
            arr[cx_idx, cy_idx] -= val
    return arr


@njit
def _compute_centres_access(state_arr: Any, granularity_m: int, walk_dist_m: int) -> Any:
    """Computes accessibility surface to centres"""
    cent_acc_arr = np.full(state_arr.shape, 0, dtype=np.float_)
    for sx_idx, sy_idx in np.ndindex(state_arr.shape):
        # bail if not a centre
        if state_arr[sx_idx, sy_idx] != 2:
            continue
        # otherwise, agg access to surrounding extents
        cent_acc_arr = _inc_access(sx_idx, sy_idx, cent_acc_arr, granularity_m, walk_dist_m)
    return cent_acc_arr


@njit
def _green_state(
    x_idx: int, y_idx: int, state_arr: Any, green_itx_arr: Any, green_acc_arr: Any, granularity_m: int, walk_dist_m: int
) -> tuple[Any, Any]:
    """Check a built cell's neighbours - set to itx if green space - update green access"""
    # check if cell is currently green_itx
    # if so, set to False and decrement green access accordingly
    if green_itx_arr[x_idx, y_idx] == 1:
        green_itx_arr[x_idx, y_idx] = 0
        green_acc_arr = _decr_access(x_idx, y_idx, green_acc_arr, granularity_m, walk_dist_m)
    # scan through neighbours
    for x_nb_idx in range(x_idx - 1, x_idx + 2):
        if x_nb_idx < 0 or x_nb_idx >= green_itx_arr.shape[0]:
            continue
        for y_nb_idx in range(y_idx - 1, y_idx + 2):
            if y_nb_idx < 0 or y_nb_idx >= green_itx_arr.shape[1]:
                continue
            # if neighbour is currently green space and not already itx, set as new itx
            if state_arr[x_nb_idx, y_nb_idx] == 0 and green_itx_arr[x_nb_idx, y_nb_idx] == 0:
                green_itx_arr[x_nb_idx, y_nb_idx] = 1
                green_acc_arr = _inc_access(x_nb_idx, y_nb_idx, green_acc_arr, granularity_m, walk_dist_m)
    return green_itx_arr, green_acc_arr


@njit
def _compute_green_itx(state_arr: Any, granularity_m: int, walk_dist_m: int) -> Any:
    """Finds cells on the periphery of green areas and adjacent to built areas"""
    green_itx_arr = np.full(state_arr.shape, 0, np.int_)
    green_acc_arr = np.full(state_arr.shape, 0, np.float_)
    for x_idx, y_idx in np.ndindex(state_arr.shape):
        # bail if not built space
        if state_arr[x_idx, y_idx] < 1:
            continue
        # otherwise, look to see if the cell borders green space
        green_itx_arr, green_acc_arr = _green_state(
            x_idx, y_idx, state_arr, green_itx_arr, green_acc_arr, granularity_m, walk_dist_m
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
    granularity_m: int
    walk_dist_m: int
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
    density_arr: Any
    cent_acc_arr: Any
    green_acc_arr: Any

    def __init__(
        self,
        granularity_m: int,
        walk_dist_m: int,
        bounds: tuple[int, int, int, int],
        extents_arr: Any,
        centre_seeds: list[tuple[int, int]] = [],
        build_prob: float = 0.1,
        cent_prob_nb: float = 0.05,
        cent_prob_isol: float = 0.001,
        max_local_pop: int = 10000,
        prob_distribution: tuple[float, float, float] = (0.7, 0.3, 0),
        density_factors: tuple[float, float, float] = (1, 0.1, 0.01),
    ):
        """ """
        # prepare extents
        self.granularity_m = granularity_m
        self.walk_dist_m = walk_dist_m
        # params
        self.build_prob = build_prob
        self.cent_prob_nb = cent_prob_nb
        self.cent_prob_isol = cent_prob_isol
        # the assumption is that T_star is the number of blocks
        # that equals to a 15 minutes walk, i.e. roughly 1 km. 1 block has size 1000/T_star metres
        self.max_local_pop = max_local_pop
        self.prob_distribution = prob_distribution
        self.density_factors = density_factors
        # state
        self.state_arr = np.copy(extents_arr)
        # seed centres
        s, w, _n, _e = bounds
        for east, north in centre_seeds:
            x = int((east - w) / granularity_m)
            y = int((north - s) / granularity_m)
            self.state_arr[x, y] = 2
        # find boundary of built land
        self.green_itx_arr, self.green_acc_arr = _compute_green_itx(
            state_arr=self.state_arr, granularity_m=granularity_m, walk_dist_m=walk_dist_m
        )
        # density
        self.density_arr = np.full(extents_arr.shape, 0, dtype=np.float_)
        # access to centres
        self.cent_acc_arr = _compute_centres_access(
            state_arr=self.state_arr,
            granularity_m=granularity_m,
            walk_dist_m=walk_dist_m,
        )

    def iter_land_isobenefit(self):
        """ """
        (
            self.state_arr,
            self.green_itx_arr,
            self.cent_acc_arr,
            self.green_acc_arr,
            added_blocks,
            added_centrality,
        ) = _iter_land_isobenefit(
            self.state_arr,
            self.green_itx_arr,
            self.cent_acc_arr,
            self.green_acc_arr,
            self.density_arr,
            self.walk_dist_m,
            self.granularity_m,
            self.build_prob,
            self.cent_prob_isol,
            self.cent_prob_nb,
            self.prob_distribution,
            self.density_factors,
        )
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


# @njit
def _iter_land_isobenefit(
    state_arr: Any,
    green_itx_arr: Any,
    cent_acc_arr: Any,
    green_acc_arr: Any,
    density_arr: Any,
    walk_dist_m: int,
    granularity_m: int,
    build_prob: float,
    cent_prob_isol: float,
    cent_prob_nb: float,
    prob_distribution: tuple[float, float, float],
    density_factors: tuple[float, float, float],
) -> tuple[Any, Any, Any, Any, int, int]:
    """ """
    added_blocks = 0
    added_centrality = 0
    for x_idx, y_idx in np.ndindex(state_arr.shape):
        # if a cell is on the green periphery adjacent to built areas
        if green_itx_arr[x_idx, y_idx] == 1:
            # if centrality is accessible
            if cent_acc_arr[x_idx, y_idx] > 0:
                # if self.nature_stays_extended(x, y):
                # if self.nature_stays_reachable(x, y):
                if np.random.rand() < build_prob:
                    # claim as built
                    state_arr[x_idx, y_idx] = 1
                    added_blocks += 1
                    # set random density
                    density_arr[x_idx, y_idx] = _random_density(prob_distribution, density_factors)
                    # update green state
                    green_itx_arr, green_acc_arr = _green_state(
                        x_idx, y_idx, state_arr, green_itx_arr, green_acc_arr, granularity_m, walk_dist_m
                    )
            # otherwise, consider adding a new centrality
            else:
                if np.random.rand() < cent_prob_nb:
                    # if self.nature_stays_extended(x, y):
                    # if self.nature_stays_reachable(x, y):
                    # claim as built
                    state_arr[x_idx, y_idx] = 2
                    cent_acc_arr = _inc_access(x_idx, y_idx, cent_acc_arr, granularity_m, walk_dist_m)
                    added_centrality += 1
                    # set random density
                    density_arr[x_idx, y_idx] = _random_density(prob_distribution, density_factors)
                    # update green state
                    green_itx_arr, green_acc_arr = _green_state(
                        x_idx, y_idx, state_arr, green_itx_arr, green_acc_arr, granularity_m, walk_dist_m
                    )
        # handle random conversion of green space to centralities
        elif state_arr[x_idx, y_idx] == 0:
            if np.random.rand() < cent_prob_isol:
                # if self.nature_stays_extended(x, y):
                # if self.nature_stays_reachable(x, y):
                state_arr[x_idx, y_idx] = 2
                # set random density
                density_arr[x_idx, y_idx] = _random_density(prob_distribution, density_factors)
                cent_acc_arr = _inc_access(x_idx, y_idx, cent_acc_arr, granularity_m, walk_dist_m)
                added_centrality += 1
                # update green state
                green_itx_arr, green_acc_arr = _green_state(
                    x_idx, y_idx, state_arr, green_itx_arr, green_acc_arr, granularity_m, walk_dist_m
                )
    return state_arr, green_itx_arr, cent_acc_arr, green_acc_arr, added_blocks, added_centrality


def update_map_classical() -> tuple[int, int]:
    """ """
    added_blocks = 0
    added_centrality = 0
    copy_land = copy.deepcopy(self)
    for x in range(self.T_star, self.cells_x - self.T_star):
        for y in range(self.T_star, self.cells_y - self.T_star):
            block = self.map[x][y]
            assert (block.is_nature and not block.is_built) or (
                block.is_built and not block.is_nature
            ), f"({x},{y}) block has ambiguous coordinates"
            if block.is_nature:
                if copy_land.is_any_neighbor_built(x, y):
                    if np.random.rand() < self.build_prob:
                        density_level = np.random.choice(DENSITY_LEVELS, p=self.prob_distribution)
                        block.is_nature = False
                        block.is_built = True
                        block.set_block_population(self.block_pop, density_level, self.population_density)
                        added_blocks += 1

                else:
                    if (
                        np.random.rand() < self.cent_prob_isol / np.sqrt(self.cells_x * self.cells_y)
                        and (self.current_built_blocks / self.current_centralities) > 100
                    ):
                        block.is_centrality = True
                        block.set_block_population(self.block_pop, "empty", self.population_density)
                        added_centrality += 1
            else:
                if not block.is_centrality:
                    if block.density_level == "low":
                        if np.random.rand() < 0.1:
                            block.set_block_population(self.block_pop, "medium", self.population_density)
                    elif block.density_level == "medium":
                        if np.random.rand() < 0.01:
                            block.set_block_population(self.block_pop, "high", self.population_density)
                    elif (
                        block.density_level == "high" and (self.current_built_blocks / self.current_centralities) > 100
                    ):
                        if self.is_any_neighbor_centrality(x, y):
                            if np.random.rand() < self.cent_prob_nb:
                                block.is_centrality = True
                                block.set_block_population(self.block_pop, "empty", self.population_density)
                                added_centrality += 1
                        else:
                            if np.random.rand() < self.cent_prob_isol:  # /np.sqrt(self.current_built_blocks):
                                block.is_centrality = True
                                block.set_block_population(self.block_pop, "empty", self.population_density)
                                added_centrality += 1

    LOGGER.info(f"added blocks: {added_blocks}")
    LOGGER.info(f"added centralities: {added_centrality}")
    return added_blocks, added_centrality
