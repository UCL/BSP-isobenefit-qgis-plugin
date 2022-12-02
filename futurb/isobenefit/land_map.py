from __future__ import annotations

import copy
import logging
import os
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from numba import njit, stencil
from scipy.ndimage import measurements as measure

from .logger import get_logger

LOGGER: logging.Logger = get_logger()


class Land:
    """Represents land."""

    # substrate
    granularity_m: int
    walk_dist_m: int
    # parameters
    build_prob: float
    nb_cent_prob: float
    isol_cent_prob: float
    max_population: int
    max_pop_walk_dist: int
    prob_distribution: tuple[float, float, float]
    density_factors: tuple[float, float, float]
    # state
    substrate: Any  # QGIS / gdal doesn't yet support numpy typing for NDArray
    avg_dist_from_nature: float
    avg_dist_from_centr: float
    max_dist_from_nature: float
    max_dist_from_centr: float
    current_population: float
    current_centralities: float
    current_built_blocks: float
    current_free_nature: float
    avg_dist_from_nature_wide: float
    max_dist_from_nature_wide: float

    def __init__(
        self,
        cells_x: int,
        cells_y: int,
        granularity_m: int,
        walk_dist_m: int,
        build_prob: float = 0.5,
        nb_cent_prob: float = 0.005,
        isol_cent_prob: float = 0.1,
        max_population: int = 500000,
        max_pop_walk_dist: int = 10000,
        prob_distribution: tuple[float, float, float] = (0.7, 0.3, 0),
        density_factors: tuple[float, float, float] = (1, 0.1, 0.01),
    ):
        """
        block state
        -1 - out of bounds
        0 - nature
        1 - built
        2 - centrality
        inhabitants
        density_level
        0 - empty
        1 - low
        2 - medium
        3 - high
        """
        self.granularity_m = granularity_m
        self.walk_dist_m = walk_dist_m
        self.substrate = np.full((cells_x, cells_y, 3), 0, dtype=np.int_)
        # params
        self.build_prob = build_prob
        self.nb_cent_prob = nb_cent_prob
        self.isol_cent_prob = isol_cent_prob
        # the assumption is that T_star is the number of blocks
        # that equals to a 15 minutes walk, i.e. roughly 1 km. 1 block has size 1000/T_star metres
        self.max_population = max_population
        self.max_pop_walk_dist = max_pop_walk_dist
        self.prob_distribution = prob_distribution
        self.density_factors = density_factors
        # state
        self.avg_dist_from_nature = 0
        self.avg_dist_from_centr = 0
        self.max_dist_from_nature = 0
        self.max_dist_from_centr = 0
        self.current_population = 0
        self.current_centralities = 0
        self.current_built_blocks = 0
        self.current_free_nature = np.inf
        self.avg_dist_from_nature_wide = 0
        self.max_dist_from_nature_wide = 0

    @property
    def block_pop(self) -> float:
        """ """
        return self.max_pop_walk_dist / (np.pi * self.walk_dist_m**2) * (self.granularity_m**2)

    @property
    def population_density(self) -> dict[str, float]:
        """ """
        return {
            "high": self.density_factors[0],
            "medium": self.density_factors[1],
            "low": self.density_factors[2],
            "empty": 0,
        }

    def nature_stays_extended(self, x: int, y: int) -> bool:
        """ """
        # this method assumes that x,y belongs to a natural region
        land_array, _ = self.get_map_as_array()
        land_array[x, y] = 1
        nature_array = np.where(land_array == 0, 1, 0)
        _labels, num_features = measure.label(nature_array)
        is_nature_extended = False
        if num_features == 1:
            is_nature_extended = True

        is_wide_enough_height = np.apply_along_axis(
            partial(is_nature_wide_along_axis, T_star=self.T_star),
            axis=1,
            arr=nature_array,
        )
        is_wide_enough_width = np.apply_along_axis(
            partial(is_nature_wide_along_axis, T_star=self.T_star),
            axis=0,
            arr=nature_array,
        )
        narrow_places_h = len(is_wide_enough_height) - is_wide_enough_height.sum()
        narrow_places_w = len(is_wide_enough_width) - is_wide_enough_width.sum()

        return narrow_places_h == 0 and narrow_places_w == 0 and is_nature_extended

    def set_current_counts(self, urbanism_model: str):
        """ """
        land_array, population_array = self.get_map_as_array()
        self.current_population = population_array.sum()
        self.current_centralities = np.where(land_array == 2, 1, 0).sum()
        self.current_built_blocks = np.where(land_array > 0, 1, 0).sum()
        self.current_free_nature = np.where(land_array == 0, 1, 0).sum()
        tot_inhabited_blocks = np.where(land_array == 1, 1, 0).sum()

        if tot_inhabited_blocks == 0:
            self.avg_dist_from_nature = 0
            self.avg_dist_from_centr = 0
            self.max_dist_from_nature = 0
            self.max_dist_from_centr = 0
        else:
            x_centr, y_centr = np.where(land_array == 2)
            x_built, y_built = np.where(land_array == 1)
            distances_from_centr = np.sqrt((x_built[:, None] - x_centr) ** 2 + (y_built[:, None] - y_centr) ** 2).min(
                axis=1
            )
            self.avg_dist_from_centr = distances_from_centr.sum() / tot_inhabited_blocks
            self.max_dist_from_centr = distances_from_centr.max()

            x_nature, y_nature = np.where(land_array == 0)

            if urbanism_model == "classical":
                nature_array = np.where(land_array == 0, 1, 0)
                features, _labels = measure.label(nature_array)
                unique, counts = np.unique(features, return_counts=True)
                large_natural_regions = counts[1:] >= self.T_star**2
                large_natural_regions_labels = unique[1:][large_natural_regions]
                x_nature_wide, y_nature_wide = np.where(np.isin(features, large_natural_regions_labels))
                distances_from_nature_wide = np.sqrt(
                    (x_built[:, None] - x_nature_wide) ** 2 + (y_built[:, None] - y_nature_wide) ** 2
                ).min(axis=1)
                self.avg_dist_from_nature_wide = distances_from_nature_wide.sum() / tot_inhabited_blocks
                self.max_dist_from_nature_wide = distances_from_nature_wide.max()

            distances_from_nature = np.sqrt(
                (x_built[:, None] - x_nature) ** 2 + (y_built[:, None] - y_nature) ** 2
            ).min(axis=1)
            self.avg_dist_from_nature = distances_from_nature.sum() / tot_inhabited_blocks
            self.max_dist_from_nature = distances_from_nature.max()

    def set_record_counts_header(self, output_path: Path, urbanism_model: str):
        """ """
        filename = os.path.join(output_path, "current_counts.csv")
        with open(filename, "a") as f:
            if urbanism_model == "isobenefit":
                f.write(
                    "iteration,added_blocks,added_centralities,current_built_blocks,current_centralities,"
                    "current_free_nature,current_population,avg_dist_from_nature,avg_dist_from_centr,max_dist_from_nature,max_dist_from_centr\n"
                )
            elif urbanism_model == "classical":
                f.write(
                    "iteration,added_blocks,added_centralities,current_built_blocks,current_centralities,"
                    "current_free_nature,current_population,avg_dist_from_nature,avg_dist_from_wide_nature,"
                    "avg_dist_from_centr,max_dist_from_nature,max_dist_from_wide_nature,max_dist_from_centr\n"
                )
            else:
                raise ValueError(
                    f"Invalid urbanism_model value: {urbanism_model}. Must be 'classical' or 'isobenefit'."
                )

    def record_current_counts(
        self, output_path: Path, iteration: int, added_blocks: int, added_centralities: int, urbanism_model: str
    ) -> None:
        """ """
        filename = os.path.join(output_path, "current_counts.csv")
        with open(filename, "a") as f:
            if urbanism_model == "isobenefit":
                f.write(
                    f"{iteration},{added_blocks},{added_centralities},"
                    f"{self.current_built_blocks},{self.current_centralities},"
                    f"{self.current_free_nature},{self.current_population},"
                    f"{self.avg_dist_from_nature},{self.avg_dist_from_centr},"
                    f"{self.max_dist_from_nature},{self.max_dist_from_centr}\n"
                )
            elif urbanism_model == "classical":
                f.write(
                    f"{iteration},{added_blocks},{added_centralities},"
                    f"{self.current_built_blocks},{self.current_centralities},"
                    f"{self.current_free_nature},{self.current_population},"
                    f"{self.avg_dist_from_nature},{self.avg_dist_from_nature_wide},{self.avg_dist_from_centr},"
                    f"{self.max_dist_from_nature},{self.max_dist_from_nature_wide},{self.max_dist_from_centr}\n"
                )
            else:
                raise ValueError(
                    f"Invalid urbanism_model value: {urbanism_model}. Must be 'classical' or 'isobenefit'."
                )


# TODO: check array_1d type
def is_nature_wide_along_axis(array_1d, T_star: int) -> bool:
    """ """
    features, _labels = measure.label(array_1d)
    _unique, counts = np.unique(features, return_counts=True)
    if len(counts) > 1:
        return counts[1:].min() >= T_star
    else:
        return True


# @stencil
# def nbs_built_stencil(a):
#     """Can't handle border regions in stencil - only 'constant' mode is currently supported, i.e. border buffer."""
#     for x_idx in range(-1, 2):
#         for y_idx in range(-1, 2):
#             if a[x_idx, y_idx] > 0:
#                 return True
#     return False


@njit
def nbs_built_arr(substrate: Any):
    """Returns an array indicating whether a cell has a built neighbour."""
    out_arr = np.full(substrate.shape, fill_value=False)
    for x_idx in range(substrate.shape[0]):
        for y_idx in range(substrate.shape[1]):
            found_built = False
            for x_nb_idx in range(x_idx - 1, x_idx + 1):
                if x_nb_idx < 0 or x_nb_idx >= substrate.shape[0]:
                    continue
                for y_nb_idx in range(y_idx - 1, y_idx + 1):
                    if y_nb_idx < 0 or y_nb_idx >= substrate.shape[1]:
                        continue
                    if substrate[x_nb_idx][y_nb_idx] > 0:
                        found_built = True
                        break
                if found_built:
                    break
            if found_built:
                out_arr[x_idx][y_idx] = True
    return out_arr


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
                        if np.random.rand() < self.nb_cent_prob:
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
                    if np.random.rand() < self.isol_cent_prob / (self.cells_x * self.cells_y):
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
                        np.random.rand() < self.isol_cent_prob / np.sqrt(self.cells_x * self.cells_y)
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
                            if np.random.rand() < self.nb_cent_prob:
                                block.is_centrality = True
                                block.set_block_population(self.block_pop, "empty", self.population_density)
                                added_centrality += 1
                        else:
                            if np.random.rand() < self.isol_cent_prob:  # /np.sqrt(self.current_built_blocks):
                                block.is_centrality = True
                                block.set_block_population(self.block_pop, "empty", self.population_density)
                                added_centrality += 1

    LOGGER.info(f"added blocks: {added_blocks}")
    LOGGER.info(f"added centralities: {added_centrality}")
    return added_blocks, added_centrality
