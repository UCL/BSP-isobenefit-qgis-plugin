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

import matplotlib.pyplot as plt
import numpy as np
from numba import njit

from .logger import get_logger

LOGGER: logging.Logger = get_logger()


@njit
def compute_centres_access(state_arr: Any, granularity_m: int, walk_dist_m: int):
    """ """
    cent_acc_arr = np.full(state_arr.shape, 0, dtype=np.float_)
    # iterate through state array to find centres
    for sx_idx in range(state_arr.shape[0]):
        for sy_idx in range(state_arr.shape[1]):
            # ignore if not a centre
            if state_arr[sx_idx, sy_idx] != 2:
                continue
            # if centre found: agg access to surrounding extents
            for cx_idx in range(cent_acc_arr.shape[0]):
                x_dist = int(abs(sx_idx - cx_idx) * granularity_m)
                if x_dist > walk_dist_m:
                    continue
                for cy_idx in range(cent_acc_arr.shape[1]):
                    y_dist = int(abs(sy_idx - cy_idx) * granularity_m)
                    if y_dist > walk_dist_m:
                        continue
                    dist = np.hypot(x_dist, y_dist)
                    if dist > walk_dist_m:
                        continue
                    cent_acc_arr[cx_idx, cy_idx] += 1 - dist / walk_dist_m
    return cent_acc_arr


@njit
def compute_itx(state_arr: Any):
    """Returns an array indicating whether a cell has a built neighbour."""
    # reset
    itx_arr = np.full(state_arr.shape, 0, np.int_)
    for x_idx in range(state_arr.shape[0]):
        for y_idx in range(state_arr.shape[1]):
            # bail if land can't be developed
            if state_arr[x_idx, y_idx] != 0:
                continue
            # otherwise, look to see if the cell borders existing built land
            for x_nb_idx in range(x_idx - 1, x_idx + 2):
                if x_nb_idx < 0 or x_nb_idx >= state_arr.shape[0]:
                    continue
                for y_nb_idx in range(y_idx - 1, y_idx + 2):
                    if y_nb_idx < 0 or y_nb_idx >= state_arr.shape[1]:
                        continue
                    # itx if a neighbour has built land
                    if state_arr[x_nb_idx, y_nb_idx] > 0:
                        itx_arr[x_idx, y_idx] = 1
    return itx_arr


@njit
def compute_green_access(state_arr: Any, itx_arr: Any, granularity_m: int, walk_dist_m: int):
    """ """
    green_acc_arr = np.full(state_arr.shape, 0, np.int_)
    # find all border regions
    for sx_idx in range(state_arr.shape[0]):
        for sy_idx in range(state_arr.shape[1]):
            # if a cell is already greenspace
            if state_arr[sx_idx, sy_idx] == 0:
                green_acc_arr[sx_idx, sy_idx] = 1
            # cells available for development are access to green space
            if not itx_arr[sx_idx, sy_idx] == 1:
                continue
            # aggregate green access to reachable built areas
            for gx_idx in range(green_acc_arr.shape[0]):
                x_dist = int(abs(sx_idx - gx_idx) * granularity_m)
                if x_dist > walk_dist_m:
                    continue
                for gy_idx in range(green_acc_arr.shape[1]):
                    y_dist = int(abs(sy_idx - gy_idx) * granularity_m)
                    if y_dist > walk_dist_m:
                        continue
                    dist = np.hypot(x_dist, y_dist)
                    if dist > walk_dist_m:
                        continue
                    # skip cells that are already green space
                    if green_acc_arr[gx_idx, gy_idx] == 1:
                        continue
                    # agg
                    green_acc_arr[gx_idx, gy_idx] += dist / walk_dist_m
    return green_acc_arr


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
    itx_arr: Any
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
        build_prob: float = 0.5,
        cent_prob_nb: float = 0.005,
        cent_prob_isol: float = 0.1,
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
        self.itx_arr = compute_itx(state_arr=self.state_arr)
        # density
        self.density_arr = np.full(extents_arr.shape, 0, dtype=np.float_)
        # access to centres
        self.cent_acc_arr = compute_centres_access(
            state_arr=self.state_arr,
            granularity_m=granularity_m,
            walk_dist_m=walk_dist_m,
        )
        # access to greenspace
        self.green_acc_arr = compute_green_access(
            state_arr=self.state_arr, itx_arr=self.itx_arr, granularity_m=granularity_m, walk_dist_m=walk_dist_m
        )
        fig, axes = plt.subplots(2, 2, squeeze=True, sharex=True, sharey=True)
        axes[0][0].imshow(self.state_arr)
        axes[0][1].imshow(self.itx_arr)
        axes[1][0].imshow(self.cent_acc_arr)
        axes[1][1].imshow(self.green_acc_arr)
        plt.show()
        print("here")

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
def reachables_arr(substrate: Any, granularity_m: int, walk_dist_m: int):
    """Returns two arrays indicating whether centrality is reachable and whether nature is reachable."""
    cell_span = np.ceil(walk_dist_m / granularity_m)
    centrality_arr = np.full(substrate.shape, fill_value=False)
    nature_arr = np.full(substrate.shape, fill_value=False)
    for x_idx in range(substrate.shape[0]):
        for y_idx in range(substrate.shape[1]):
            found_centrality = False
            found_nature = False
            for x_nb_idx in range(x_idx - cell_span, x_idx + cell_span):
                if x_nb_idx < 0 or x_nb_idx >= substrate.shape[0]:
                    continue
                for y_nb_idx in range(y_idx - cell_span, y_idx + cell_span):
                    if y_nb_idx < 0 or y_nb_idx >= substrate.shape[1]:
                        continue
                    if substrate[x_nb_idx][y_nb_idx] == 0:
                        found_nature = True
                    elif substrate[x_nb_idx][y_nb_idx] == 2:
                        found_centrality = True
                    if found_nature and found_centrality:
                        break
                if found_nature and found_centrality:
                    break
            if found_centrality:
                centrality_arr[x_idx][y_idx] = True
            if found_nature:
                nature_arr[x_idx][y_idx] = True
    return centrality_arr, nature_arr


def update_map_isobenefit(substrate: Any, walk_dist_m: int, granularity_m: int) -> tuple[int, int]:
    """ """
    added_blocks = 0
    added_centrality = 0
    for x in range(self.T_star, self.cells_x - self.T_star):
        for y in range(self.T_star, self.cells_y - self.T_star):
            block = self.map[x][y]
            assert (block.is_nature and not block.is_built) or (
                block.is_built and not block.is_nature
            ), f"({x},{y}) block has ambiguous coordinates"
            if block.is_nature:
                if copy_land.is_any_neighbor_built(x, y):
                    if copy_land.is_centrality_near(x, y):
                        if self.nature_stays_extended(x, y):
                            if np.random.rand() < self.build_prob:
                                if self.nature_stays_reachable(x, y):
                                    density_level = np.random.choice(
                                        list(range(4)),
                                        p=self.prob_distribution,
                                    )
                                    block.is_built = True
                                    block.set_block_population(
                                        self.block_pop,
                                        density_level,
                                        self.population_density,
                                    )
                                    added_blocks += 1
                    else:
                        if np.random.rand() < self.cent_prob_nb:
                            if self.nature_stays_extended(x, y):
                                if self.nature_stays_reachable(x, y):
                                    block.is_centrality = True
                                    block.set_block_population(
                                        self.block_pop,
                                        "empty",
                                        self.population_density,
                                    )
                                    added_centrality += 1
                else:
                    if np.random.rand() < self.cent_prob_isol / (self.cells_x * self.cells_y):
                        if self.nature_stays_extended(x, y):
                            if self.nature_stays_reachable(x, y):
                                block.is_centrality = True
                                block.set_block_population(self.block_pop, "empty", self.population_density)
                                added_centrality += 1
    LOGGER.info(f"added blocks: {added_blocks}")
    LOGGER.info(f"added centralities: {added_centrality}")
    return added_blocks, added_centrality


def is_any_neighbor_centrality(self, x: int, y: int) -> bool:
    return (
        self.map[x - 1][y].is_centrality
        or self.map[x + 1][y].is_centrality
        or self.map[x][y - 1].is_centrality
        or self.map[x][y + 1].is_centrality
    )


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
