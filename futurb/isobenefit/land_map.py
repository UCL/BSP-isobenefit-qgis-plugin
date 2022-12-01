from __future__ import annotations

import copy
import logging
import os
from functools import partial
from pathlib import Path

import numpy as np
from scipy.ndimage import measurements as measure

from .logger import get_logger

LOGGER: logging.Logger = get_logger()

DENSITY_LEVELS = ["high", "medium", "low"]


class MapBlock:
    """Represents a cell."""

    x: int
    y: int
    _is_nature: bool
    _is_built: bool
    _is_centrality: bool
    _inhabitants: float
    _density_level: str | None

    def __init__(self, x: int, y: int, inhabitants: float = 0.0):
        """ """
        self.x = x
        self.y = y
        self._is_nature = True
        self._is_built = False
        self._is_centrality = False
        self._inhabitants = inhabitants
        self._density_level = None

    @property
    def is_nature(self) -> bool:
        """ """
        return self._is_nature

    @is_nature.setter
    def is_nature(self, nature_state: bool) -> None:
        """ """
        self._is_nature = nature_state
        if nature_state is True:
            self._is_built = False
            self._is_centrality = False

    @property
    def is_built(self) -> bool:
        """ """
        return self._is_built

    @is_built.setter
    def is_built(self, built_state: bool) -> None:
        """ """
        self._is_built = built_state
        if built_state is True:
            self._is_nature = False

    @property
    def is_centrality(self) -> bool:
        """ """
        return self._is_centrality

    @is_centrality.setter
    def is_centrality(self, centrality_state: bool) -> None:
        """ """
        self._is_centrality = centrality_state
        if centrality_state is True:
            self._is_built = True
            self._is_nature = False

    @property
    def inhabitants(self) -> float:
        """ """
        return self._inhabitants

    @property
    def density_level(self) -> str | None:
        """ """
        return self._density_level

    # TODO: confirm population_density type
    def set_block_population(
        self, block_population: float, density_level: str, population_density: dict[str, float]
    ) -> None:
        """ """
        self._inhabitants = block_population * population_density[density_level]
        self._density_level = density_level


class Land:
    """Represents land."""

    # input
    size_x: int
    size_y: int
    build_probability: float
    neighboring_centrality_probability: float
    isolated_centrality_probability: float
    T_star: int
    max_population: int
    max_ab_km2: int
    prob_distribution: tuple[float, float, float]
    density_factors: tuple[float, float, float]
    # state
    map: list[list[MapBlock]]
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
        size_x: int,
        size_y: int,
        build_probability: float = 0.5,
        neighboring_centrality_probability: float = 0.005,
        isolated_centrality_probability: float = 0.1,
        T_star: int = 10,
        max_population: int = 500000,
        max_ab_km2: int = 10000,
        prob_distribution: tuple[float, float, float] = (0.7, 0.3, 0),
        density_factors: tuple[float, float, float] = (1, 0.1, 0.01),
    ):
        """ """
        self.size_x = size_x
        self.size_y = size_y
        self.build_probability = build_probability
        self.neighboring_centrality_probability = neighboring_centrality_probability
        self.isolated_centrality_probability = isolated_centrality_probability
        # the assumption is that T_star is the number of blocks
        # that equals to a 15 minutes walk, i.e. roughly 1 km. 1 block has size 1000/T_star metres
        self.T_star = T_star
        self.max_population = max_population
        self.max_ab_km2 = max_ab_km2
        self.prob_distribution = prob_distribution
        self.density_factors = density_factors
        # state
        self.map = [[MapBlock(x, y, inhabitants=0) for x in range(size_y)] for y in range(size_x)]
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
        return self.max_ab_km2 / (self.T_star**2)

    @property
    def population_density(self) -> dict[str, float]:
        """ """
        return {
            "high": self.density_factors[0],
            "medium": self.density_factors[1],
            "low": self.density_factors[2],
            "empty": 0,
        }

    def get_map_as_array(self):
        """ """
        map_array = np.full((self.size_x, self.size_y), 0)
        population_array = np.full((self.size_x, self.size_y), 0.0, dtype=np.float_)
        for x in range(self.size_x):
            for y in range(self.size_y):
                if self.map[x][y].is_built:
                    map_array[x, y] = 1
                if self.map[x][y].is_centrality:
                    map_array[x, y] = 2
                population_array[x, y] = self.map[x][y].inhabitants

        return map_array, population_array

    def is_any_neighbor_built(self, x: int, y: int) -> bool:
        """ """
        assert self.T_star <= x <= self.size_x - self.T_star, f"point ({x},{y}) is not in the 'interior' of the land"
        assert self.T_star <= y <= self.size_y - self.T_star, f"point ({x},{y}) is not in the 'interior' of the land"
        return (
            self.map[x - 1][y].is_built
            or self.map[x + 1][y].is_built
            or self.map[x][y - 1].is_built
            or self.map[x][y + 1].is_built
        )

    def is_centrality_near(self, x: int, y: int) -> bool:
        """ """
        assert self.T_star <= x <= self.size_x - self.T_star, f"point ({x},{y}) is not in the 'interior' of the land"
        assert self.T_star <= y <= self.size_y - self.T_star, f"point ({x},{y}) is not in the 'interior' of the land"
        for i in range(x - self.T_star, x + self.T_star + 1):
            for j in range(y - self.T_star, y + self.T_star + 1):
                if self.map[i][j].is_centrality:
                    if d(x, y, i, j) <= self.T_star:
                        return True
        return False

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

    def nature_stays_reachable(self, x: int, y: int) -> bool:
        """ """
        land_array, _ = self.get_map_as_array()
        land_array[x, y] = 1
        x_built, y_built = np.where(land_array > 0)
        x_nature, y_nature = np.where(land_array == 0)
        return (
            np.sqrt((x_built[:, None] - x_nature) ** 2 + (y_built[:, None] - y_nature) ** 2).min(axis=1).max()
            <= self.T_star
        )

    def set_configuration_from_image(self, filepath: Path):
        """ """
        array_map = import_2Darray_from_image(filepath)
        for x in range(self.size_x):
            for y in range(self.size_y):
                if array_map[x, y] == 1:
                    self.map[x][y].is_centrality = True

                if array_map[x, y] == 0:
                    self.map[x][y].is_built = True
                    self.map[x][y].is_centrality = False

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


def d(x1: int, y1: int, x2: int, y2: int) -> float:
    """ """
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# TODO: check array_1d type
def is_nature_wide_along_axis(array_1d, T_star: int) -> bool:
    """ """
    features, _labels = measure.label(array_1d)
    _unique, counts = np.unique(features, return_counts=True)
    if len(counts) > 1:
        return counts[1:].min() >= T_star
    else:
        return True


class IsobenefitScenario(Land):
    """ """

    def update_map(self) -> tuple[int, int]:
        """ """
        added_blocks = 0
        added_centrality = 0
        copy_land = copy.deepcopy(self)
        for x in range(self.T_star, self.size_x - self.T_star):
            for y in range(self.T_star, self.size_y - self.T_star):
                block = self.map[x][y]
                assert (block.is_nature and not block.is_built) or (
                    block.is_built and not block.is_nature
                ), f"({x},{y}) block has ambiguous coordinates"
                if block.is_nature:
                    if copy_land.is_any_neighbor_built(x, y):
                        if copy_land.is_centrality_near(x, y):
                            if self.nature_stays_extended(x, y):
                                if np.random.rand() < self.build_probability:
                                    if self.nature_stays_reachable(x, y):
                                        density_level = np.random.choice(
                                            DENSITY_LEVELS,
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
                            if np.random.rand() < self.neighboring_centrality_probability:
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
                        if np.random.rand() < self.isolated_centrality_probability / (self.size_x * self.size_y):
                            if self.nature_stays_extended(x, y):
                                if self.nature_stays_reachable(x, y):
                                    block.is_centrality = True
                                    block.set_block_population(self.block_pop, "empty", self.population_density)
                                    added_centrality += 1
        LOGGER.info(f"added blocks: {added_blocks}")
        LOGGER.info(f"added centralities: {added_centrality}")
        return added_blocks, added_centrality


class ClassicalScenario(Land):
    """ """

    def is_any_neighbor_centrality(self, x: int, y: int) -> bool:
        return (
            self.map[x - 1][y].is_centrality
            or self.map[x + 1][y].is_centrality
            or self.map[x][y - 1].is_centrality
            or self.map[x][y + 1].is_centrality
        )

    def update_map(self) -> tuple[int, int]:
        """ """
        added_blocks = 0
        added_centrality = 0
        copy_land = copy.deepcopy(self)
        for x in range(self.T_star, self.size_x - self.T_star):
            for y in range(self.T_star, self.size_y - self.T_star):
                block = self.map[x][y]
                assert (block.is_nature and not block.is_built) or (
                    block.is_built and not block.is_nature
                ), f"({x},{y}) block has ambiguous coordinates"
                if block.is_nature:
                    if copy_land.is_any_neighbor_built(x, y):
                        if np.random.rand() < self.build_probability:
                            density_level = np.random.choice(DENSITY_LEVELS, p=self.prob_distribution)
                            block.is_nature = False
                            block.is_built = True
                            block.set_block_population(self.block_pop, density_level, self.population_density)
                            added_blocks += 1

                    else:
                        if (
                            np.random.rand() < self.isolated_centrality_probability / np.sqrt(self.size_x * self.size_y)
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
                            block.density_level == "high"
                            and (self.current_built_blocks / self.current_centralities) > 100
                        ):
                            if self.is_any_neighbor_centrality(x, y):
                                if np.random.rand() < self.neighboring_centrality_probability:
                                    block.is_centrality = True
                                    block.set_block_population(self.block_pop, "empty", self.population_density)
                                    added_centrality += 1
                            else:
                                if (
                                    np.random.rand() < self.isolated_centrality_probability
                                ):  # /np.sqrt(self.current_built_blocks):
                                    block.is_centrality = True
                                    block.set_block_population(self.block_pop, "empty", self.population_density)
                                    added_centrality += 1

        LOGGER.info(f"added blocks: {added_blocks}")
        LOGGER.info(f"added centralities: {added_centrality}")
        return added_blocks, added_centrality
