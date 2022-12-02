from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import rasterio as rio
from qgis.core import (
    QgsContrastEnhancement,
    QgsCoordinateReferenceSystem,
    QgsDateTimeRange,
    QgsLayerTreeGroup,
    QgsMultiBandColorRenderer,
    QgsProject,
    QgsRasterLayer,
    QgsRasterLayerTemporalProperties,
    QgsVectorLayer,
)
from rasterio import transform

from .land_map import ClassicalScenario, IsobenefitScenario, Land, MapBlock
from .logger import configure_logging, get_logger

N_AMENITIES = 1


def get_central_coord(size_x_m: int, size_y_m: int) -> list[tuple[int, int]]:
    """Returns centre coordinates given the x, y size."""
    return [(int(size_x_m / 2), int(size_y_m / 2))]


def run_isobenefit_simulation(
    extents_layer: QgsVectorLayer,
    target_crs: QgsCoordinateReferenceSystem,
    granularity_m: int,
    walk_dist_m: int,
    n_steps: int,
    out_dir_path: Path,
    out_file_name: str,
    build_prob: float,
    nb_cent_prob: float,
    isol_cent_prob: float,
    random_seed: int,
    initialization_mode: str,
    max_population: int,
    max_pop_walk_dist: int,
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
    size_x_m = int(x_max - x_min)
    size_y_m = int(y_max - y_min)
    cells_x = int(size_x_m / granularity_m)
    cells_y = int(size_y_m / granularity_m)
    # checks
    if not cells_x * granularity_m > 2 * walk_dist_m or not cells_y * granularity_m > 2 * walk_dist_m:
        raise ValueError(f"The provided extents is too small. It should be larger than 2x walking distance.")
    prob_sum = sum(prob_distribution)
    if not prob_sum == 1:
        raise ValueError(f"The probability distribution doesn't sum to 1 ({prob_sum})")
    if not density_factors[0] >= density_factors[1] >= density_factors[2]:
        raise ValueError(f"The density factors are not decreasing in value: {density_factors}.")
    # write metadata
    metadata_filepath: Path = out_dir_path / "metadata.json"
    with open(metadata_filepath, "w") as f:
        f.write(
            json.dumps(
                {
                    "size_x_m": size_x_m,
                    "size_y_m": size_y_m,
                    "cells_x_m": cells_x,
                    "cells_y_m": cells_y,
                    "granularity_m": granularity_m,
                    "walk_dist_m": walk_dist_m,
                    "n_steps": n_steps,
                    "output_path": str(out_dir_path / out_file_name),
                    "build_prob": build_prob,
                    "nb_cent_prob": nb_cent_prob,
                    "isol_cent_prob": isol_cent_prob,
                    "walk_dist_m": walk_dist_m,
                    "random_seed": random_seed,
                    "initialization_mode": initialization_mode,
                    "max_population": max_population,
                    "max_pop_walk_dist": max_pop_walk_dist,
                    "urbanism_model": urbanism_model,
                    "prob_distribution": prob_distribution,
                    "density_factors": density_factors,
                }
            )
        )
    # start simulation
    t_zero = time.time()
    land = Land(
        cells_x,
        cells_y,
        granularity_m,
        walk_dist_m,
        build_prob,
        nb_cent_prob,
        isol_cent_prob,
        max_population,
        max_pop_walk_dist,
        prob_distribution,
        density_factors,
    )
    # prepare QGIS menu
    layer_root = QgsProject.instance().layerTreeRoot()
    layer_group = layer_root.insertGroup(0, f"{out_file_name} outputs")
    # initialise simulation
    update_map_snapshot(land)
    save_snapshot(
        land,
        0,
        out_dir_path,
        out_file_name,
        x_min,
        x_max,
        y_min,
        y_max,
        size_x_m,
        size_y_m,
        target_crs,
        layer_group,
    )
    land.set_record_counts_header(output_path=out_dir_path, urbanism_model=urbanism_model)
    land.set_current_counts(urbanism_model)
    # first step is already written, so use 1
    added_blocks, added_centralities = (0, 0)
    land.record_current_counts(
        output_path=out_dir_path,
        iteration=0,
        added_blocks=added_blocks,
        added_centralities=added_centralities,
        urbanism_model=urbanism_model,
    )
    for idx in range(1, n_steps + 1):
        if land.current_population <= land.max_population:
            break
        print(idx, n_steps)
        start = time.time()
        added_blocks, added_centralities = land.update_map()
        if urbanism_model == "isobenefit":
            
        elif urbanism_model == "classical":

        else:
            raise ValueError("Invalid urbanism model. Choose one of 'isobenefit' or 'classical'")
        land.set_current_counts(urbanism_model)
        land.record_current_counts(
            output_path=out_dir_path,
            iteration=idx,
            added_blocks=added_blocks,
            added_centralities=added_centralities,
            urbanism_model=urbanism_model,
        )
        LOGGER.info(f"step: {idx}, duration: {time.time() - start} seconds")
        LOGGER.info(f"step: {idx}, current population: {land.current_population} inhabitants")
        update_map_snapshot(land)
        save_snapshot(
            land,
            idx,
            out_dir_path,
            out_file_name,
            x_min,
            x_max,
            y_min,
            y_max,
            size_x_m,
            size_y_m,
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


def update_map_snapshot(land: Land) -> None:
    """ """
    for row in land.substrate:
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
            land.substrate[block.y, block.x] = color


def save_snapshot(
    land: Land,  # type: ignore
    step: int,
    out_dir_path: Path,
    out_file_name: str,
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
    size_x_m: int,
    size_y_m: int,
    target_crs: QgsCoordinateReferenceSystem,
    layer_group: QgsLayerTreeGroup,
) -> None:
    """ """
    out_name = f"{out_file_name}_{step:05d}"
    out_path: str = str(out_dir_path / f"{out_name}.tif")
    crs_wkt: str = target_crs.toWkt()
    trf = transform.from_bounds(x_min, y_min, x_max, y_max, size_x_m, size_y_m)  # type: ignore
    with rio.open(  # type: ignore
        out_path,
        "w",
        driver="GTiff",
        height=size_y_m,
        width=size_x_m,
        count=3,
        dtype=land.substrate.dtype,  # type: ignore
        crs=crs_wkt,
        transform=trf,
        nodata=np.nan,
    ) as out_rast:  # type: ignore
        # expects bands, rows, columns order
        out_rast.write(land.substrate.transpose(2, 0, 1))  # type: ignore
        # create QGIS layer and renderer
        rast_layer = QgsRasterLayer(out_path, f"step {step}", providerType="gdal")
        rast_layer.setCrs(target_crs)
        # help out type hinting via cast as IDE doesn't know ahead of time re: multiband renderer
        rast_renderer: QgsMultiBandColorRenderer = cast(QgsMultiBandColorRenderer, rast_layer.renderer())
        # setup renderer
        rast_renderer.setRedBand(1)
        rast_renderer.setGreenBand(2)
        rast_renderer.setBlueBand(3)
        red_ce = rast_renderer.redContrastEnhancement()
        red_ce.setContrastEnhancementAlgorithm(QgsContrastEnhancement.StretchAndClipToMinimumMaximum)
        red_ce.setMinimumValue(0)
        red_ce.setMaximumValue(1)
        green_ce = rast_renderer.greenContrastEnhancement()
        green_ce.setContrastEnhancementAlgorithm(QgsContrastEnhancement.StretchAndClipToMinimumMaximum)
        green_ce.setMinimumValue(0)
        green_ce.setMaximumValue(1)
        blue_ce = rast_renderer.blueContrastEnhancement()
        blue_ce.setContrastEnhancementAlgorithm(QgsContrastEnhancement.StretchAndClipToMinimumMaximum)
        blue_ce.setMinimumValue(0)
        blue_ce.setMaximumValue(1)
        # setup temporal
        temp_props: QgsRasterLayerTemporalProperties = cast(
            QgsRasterLayerTemporalProperties, rast_layer.temporalProperties()
        )
        temp_props.setMode(QgsRasterLayerTemporalProperties.ModeFixedTemporalRange)  # type: ignore
        start_date = datetime.now()
        this_date = start_date.replace(year=start_date.year + step)
        next_date = start_date.replace(year=start_date.year + step + 1)
        time_range = QgsDateTimeRange(begin=this_date, end=next_date)
        temp_props.setFixedTemporalRange(time_range)
        temp_props.isVisibleInTemporalRange(time_range)
        temp_props.setIsActive(True)
        # add to QGIS
        QgsProject.instance().addMapLayer(rast_layer, addToLegend=False)
        lt_layer = layer_group.addLayer(rast_layer)
        lt_layer.setExpanded(False)
