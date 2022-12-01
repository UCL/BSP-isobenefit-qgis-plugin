from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import rasterio as rio
from qgis.core import (
    QgsContrastEnhancement,
    QgsCoordinateReferenceSystem,
    QgsLayerTreeGroup,
    QgsMultiBandColorRenderer,
    QgsProject,
    QgsRasterLayer,
    QgsVectorLayer,
)
from rasterio import transform

from .initialization_utils import get_central_coord
from .land_map import ClassicalScenario, IsobenefitScenario, Land, MapBlock
from .logger import configure_logging, get_logger

N_AMENITIES = 1


def run_isobenefit_simulation(
    extents_layer: QgsVectorLayer,
    target_crs: QgsCoordinateReferenceSystem,
    granularity_m: int,
    n_steps: int,
    out_dir_path: Path,
    out_file_name: str,
    build_probability: float,
    neighboring_centrality_probability: float,
    isolated_centrality_probability: float,
    T_star: int,
    random_seed: int,
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
    # prepare extents
    x_min = extents_layer.extent().xMinimum()
    x_max = extents_layer.extent().xMaximum()
    y_min = extents_layer.extent().yMinimum()
    y_max = extents_layer.extent().yMaximum()
    x_min = int(x_min - x_min % granularity_m)
    x_max = int(x_max - x_max % granularity_m) + granularity_m
    y_min = int(y_min - y_min % granularity_m)
    y_max = int(y_max - y_max % granularity_m) + granularity_m
    size_x = int((x_max - x_min) / granularity_m)
    size_y = int((y_max - y_min) / granularity_m)
    # write metadata
    metadata = {
        "size_x": size_x,
        "size_y": size_y,
        "n_steps": n_steps,
        "output_path": str(out_dir_path / out_file_name),
        "build_probability": build_probability,
        "neighboring_centrality_probability": neighboring_centrality_probability,
        "isolated_centrality_probability": isolated_centrality_probability,
        "T_star": T_star,
        "random_seed": random_seed,
        "initialization_mode": initialization_mode,
        "max_population": max_population,
        "max_ab_km2": max_ab_km2,
        "urbanism_model": urbanism_model,
        "prob_distribution": prob_distribution,
        "density_factors": density_factors,
    }
    save_metadata(metadata, out_dir_path)
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
        max_population=max_population,
        max_ab_km2=max_ab_km2,
        urbanism_model=urbanism_model,
        prob_distribution=prob_distribution,
        density_factors=density_factors,
    )
    canvas = np.full((size_x, size_y, 3), 1.0, dtype=np.float_)
    # prepare QGIS menu
    layer_root = QgsProject.instance().layerTreeRoot()
    layer_group = layer_root.insertGroup(0, f"{out_file_name} outputs")
    update_map_snapshot(land, canvas)
    save_snapshot(
        canvas,
        0,
        out_dir_path,
        out_file_name,
        x_min,
        x_max,
        y_min,
        y_max,
        size_x,
        size_y,
        target_crs,
        layer_group,
    )
    land.set_record_counts_header(output_path=out_dir_path, urbanism_model=urbanism_model)
    land.set_current_counts(urbanism_model)
    i = 0
    added_blocks, added_centralities = (0, 0)
    land.record_current_counts(
        output_path=out_dir_path,
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
            output_path=out_dir_path,
            iteration=i,
            added_blocks=added_blocks,
            added_centralities=added_centralities,
            urbanism_model=urbanism_model,
        )
        LOGGER.info(f"step: {i}, duration: {time.time() - start} seconds")
        LOGGER.info(f"step: {i}, current population: {land.current_population} inhabitants")
        update_map_snapshot(land, canvas)
        save_snapshot(
            canvas,
            i,
            out_dir_path,
            out_file_name,
            x_min,
            x_max,
            y_min,
            y_max,
            size_x,
            size_y,
            target_crs,
            layer_group,
        )
    # save_min_distances(land, out_dir_path)
    LOGGER.info(f"Simulation ended. Total duration: {time.time() - t_zero} seconds")


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
    if mode == "list":
        amenities: list[MapBlock] = [MapBlock(x, y, inhabitants=0) for (x, y) in amenities_list]
        for amenity in amenities:
            amenity.is_centrality = True
    else:
        raise Exception('Invalid initialization mode. Valid modes are "image" and "list".')

    return land


def update_map_snapshot(land: Land, canvas) -> None:
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
            canvas[block.y, block.x] = color


def save_snapshot(
    rast_arr,
    step: int,
    out_dir_path: Path,
    out_file_name: str,
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
    size_x: int,
    size_y: int,
    target_crs: QgsCoordinateReferenceSystem,
    layer_group: QgsLayerTreeGroup,
) -> None:
    """ """
    out_name = f"{out_file_name}_{step:05d}"
    out_path: str = str(out_dir_path / f"{out_name}.tif")
    crs_wkt: str = target_crs.toWkt()
    trf = transform.from_bounds(x_min, y_min, x_max, y_max, size_x, size_y)
    with rio.open(
        out_path,
        "w",
        driver="GTiff",
        height=size_y,
        width=size_x,
        count=3,
        dtype=rast_arr.dtype,
        crs=crs_wkt,
        transform=trf,
        nodata=np.nan,
    ) as out_rast:
        # expects bands, rows, columns order
        out_rast.write(rast_arr.transpose(2, 0, 1))
        # add to QGIS
        rast_layer = QgsRasterLayer(out_path, out_name, providerType="gdal")
        QgsProject.instance().addMapLayer(rast_layer, addToLegend=False)
        layer_group.addLayer(rast_layer)
        rast_layer.setCrs(target_crs)


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
