import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .image_io import save_image_from_2Darray
from .initialization_utils import get_central_coord
from .land_map import ClassicalScenario, IsobenefitScenario, Land, MapBlock
from .logger import configure_logging, get_logger

N_AMENITIES = 1


def run_isobenefit_simulation(
    size_x: int,
    size_y: int,
    n_steps: int,
    output_path_prefix: str,
    build_probability: float,
    neighboring_centrality_probability: float,
    isolated_centrality_probability: float,
    T_star: int,
    random_seed: int,
    input_filepath: Path,
    initialization_mode: str,
    max_population: int,
    max_ab_km2: int,
    urbanism_model: str,
    prob_distribution: tuple[float, float, float],
    density_factors: tuple[float, float, float],
) -> None:
    """ """
    configure_logging()
    LOGGER = get_logger()
    np.random.seed(random_seed)

    output_path = make_output_path(output_path_prefix)
    metadata = {
        "size_x": size_x,
        "size_y": size_y,
        "n_steps": n_steps,
        "output_path": str(output_path),
        "build_probability": build_probability,
        "neighboring_centrality_probability": neighboring_centrality_probability,
        "isolated_centrality_probability": isolated_centrality_probability,
        "T_star": T_star,
        "random_seed": random_seed,
        "input_filepath": str(input_filepath),
        "initialization_mode": initialization_mode,
        "max_population": max_population,
        "max_ab_km2": max_ab_km2,
        "urbanism_model": urbanism_model,
        "prob_distribution": prob_distribution,
        "density_factors": density_factors,
    }

    output_path.mkdir(parents=True, exist_ok=True)
    save_metadata(metadata, output_path)

    t_zero = time.time()
    land: IsobenefitScenario | ClassicalScenario = initialize_land(
        size_x,
        size_y,
        amenities_list=get_central_coord(size_x=size_x, size_y=size_y),
        neighboring_centrality_probability=neighboring_centrality_probability,
        isolated_centrality_probability=isolated_centrality_probability,
        build_probability=build_probability,
        T=T_star,
        mode=initialization_mode,
        filepath=input_filepath,
        max_population=max_population,
        max_ab_km2=max_ab_km2,
        urbanism_model=urbanism_model,
        prob_distribution=prob_distribution,
        density_factors=density_factors,
    )

    canvas: npt.NDArray[np.float_] = np.full((size_x, size_y, 4), 1.0, dtype=np.float_)
    update_map_snapshot(land, canvas)
    save_snapshot(canvas, output_path=output_path, step=0)
    land.set_record_counts_header(output_path=output_path, urbanism_model=urbanism_model)
    land.set_current_counts(urbanism_model)
    i = 0
    added_blocks, added_centralities = (0, 0)
    land.record_current_counts(
        output_path=output_path,
        iteration=i,
        added_blocks=added_blocks,
        added_centralities=added_centralities,
        urbanism_model=urbanism_model,
    )

    while i <= n_steps and land.current_population <= land.max_population:
        start = time.time()
        added_blocks, added_centralities = land.update_map()
        land.set_current_counts(urbanism_model)
        i += 1
        land.record_current_counts(
            output_path=output_path,
            iteration=i,
            added_blocks=added_blocks,
            added_centralities=added_centralities,
            urbanism_model=urbanism_model,
        )
        LOGGER.info(f"step: {i}, duration: {time.time() - start} seconds")
        LOGGER.info(f"step: {i}, current population: {land.current_population} inhabitants")
        update_map_snapshot(land, canvas)
        save_snapshot(canvas, output_path=output_path, step=i)

    save_min_distances(land, output_path)

    LOGGER.info(f"Simulation ended. Total duration: {time.time() - t_zero} seconds")


def make_output_path(output_path_prefix: str) -> Path:
    """ """
    if output_path_prefix is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        output_path = Path(f"simulations/{timestamp}")
    else:
        output_path = Path(f"simulations/{output_path_prefix}")

    return output_path


def save_metadata(metadata: dict[str, Any], output_path: Path) -> None:
    """ """
    metadata_filepath: Path = output_path / "metadata.json"
    with open(metadata_filepath, "w") as f:
        f.write(json.dumps(metadata))


def initialize_land(
    size_x: int,
    size_y: int,
    build_probability: float,
    neighboring_centrality_probability: float,
    isolated_centrality_probability: float,
    T: int,
    max_population: int,
    max_ab_km2: int,
    mode: str,
    filepath: Path,
    amenities_list: list[tuple[int, int]],
    urbanism_model: str,
    prob_distribution: tuple[float, float, float],
    density_factors: tuple[float, float, float],
) -> IsobenefitScenario | ClassicalScenario:
    """ """
    assert (
        size_x > 2 * T and size_y > 2 * T
    ), f"size of the map is too small: {size_x}x{size_y}. Dimensions should be larger than {2 * T}"
    assert (
        sum(prob_distribution) == 1
    ), f"pobability distribution does not sum-up to 1: sum{prob_distribution} = {sum(prob_distribution)}."
    assert (
        density_factors[0] >= density_factors[1] >= density_factors[2]
    ), f"density factors are not decreasing in value: {density_factors}."

    if urbanism_model == "isobenefit":
        land = IsobenefitScenario(
            size_x=size_x,
            size_y=size_y,
            neighboring_centrality_probability=neighboring_centrality_probability,
            isolated_centrality_probability=isolated_centrality_probability,
            build_probability=build_probability,
            T_star=T,
            max_population=max_population,
            max_ab_km2=max_ab_km2,
            prob_distribution=prob_distribution,
            density_factors=density_factors,
        )
    elif urbanism_model == "classical":
        land = ClassicalScenario(
            size_x=size_x,
            size_y=size_y,
            neighboring_centrality_probability=neighboring_centrality_probability,
            isolated_centrality_probability=isolated_centrality_probability,
            build_probability=build_probability,
            T_star=T,
            max_population=max_population,
            max_ab_km2=max_ab_km2,
            prob_distribution=prob_distribution,
            density_factors=density_factors,
        )
    else:
        raise ValueError("Invalid urbanism model. Choose one of 'isobenefit' and 'classical'")

    if mode == "image" and filepath is not None:
        land.set_configuration_from_image(filepath)
    elif mode == "list":
        amenities: list[MapBlock] = [MapBlock(x, y, inhabitants=0) for (x, y) in amenities_list]
        for amenity in amenities:
            amenity.is_centrality = True
    else:
        raise Exception('Invalid initialization mode. Valid modes are "image" and "list".')

    return land


def update_map_snapshot(land: Land, canvas: npt.NDArray[np.float_]) -> None:
    """ """
    for row in land.map:
        for block in row:
            if block.is_nature:
                color = (0 / 255, 158 / 255, 96 / 255)  # green
            elif block.is_centrality:
                color = np.ones(3)
            else:
                if block.is_built and block.density_level == "high":
                    color = np.zeros(3)
                if block.is_built and block.density_level == "medium":
                    color = np.ones(3) / 3
                if block.is_built and block.density_level == "low":
                    color = np.ones(3) * 2 / 3

            canvas[block.y, block.x] = np.array([color[0], color[1], color[2], 1])


def save_snapshot(canvas: npt.NDArray[np.float_], output_path: Path, step: int, format: str = "png") -> Path:
    """ """
    final_path: Path = output_path / f"{step:05d}.png"
    save_image_from_2Darray(canvas, filepath=final_path, format=format)
    return final_path


def save_min_distances(land: Land, output_path: Path) -> None:
    """ """
    land_array, _population_array = land.get_map_as_array()
    x_centr, y_centr = np.where(land_array == 2)
    x_built, y_built = np.where(land_array == 1)
    x_nature, y_nature = np.where(land_array == 0)
    distances_from_nature = np.sqrt((x_built[:, None] - x_nature) ** 2 + (y_built[:, None] - y_nature) ** 2).min(axis=1)
    distances_from_centr = np.sqrt((x_built[:, None] - x_centr) ** 2 + (y_built[:, None] - y_centr) ** 2).min(axis=1)
    distances_mapping_filepath = os.path.join(output_path, "minimal_distances_map.csv")
    array_of_data = np.concatenate(
        [
            x_built.reshape(-1, 1),
            y_built.reshape(-1, 1),
            distances_from_nature.reshape(-1, 1),
            distances_from_centr.reshape(-1, 1),
        ],
        axis=1,
    )
    header = "X,Y,min_nature_dist, min_centr_dist"
    np.savetxt(
        fname=distances_mapping_filepath,
        X=array_of_data,
        delimiter=",",
        newline="\n",
        header=header,
    )
