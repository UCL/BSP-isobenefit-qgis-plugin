from __future__ import annotations

import json
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

from .land_map import Land
from .logger import configure_logging, get_logger

N_AMENITIES = 1


def get_central_coord(size_x_m: int, size_y_m: int) -> list[tuple[int, int]]:
    """Returns centre coordinates given the x, y size."""
    return [(int(size_x_m / 2), int(size_y_m / 2))]


def simulate(
    extents_layer: QgsVectorLayer,
    target_crs: QgsCoordinateReferenceSystem,
    granularity_m: int,
    walk_dist_m: int,
    n_steps: int,
    out_dir_path: Path,
    out_file_name: str,
    build_prob: float,
    cent_prob_nb: float,
    cent_prob_isol: float,
    max_local_pop: int,
    prob_distribution: tuple[float, float, float],
    density_factors: tuple[float, float, float],
    random_seed: int,
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
    # extents
    extents_arr = np.full((cells_y, cells_x), 0, dtype=np.int_)
    # transform
    extents_trf: transform.Affine = transform.from_bounds(x_min, y_min, x_max, y_max, cells_x, cells_y)
    # start simulation
    t_zero = time.time()
    land = Land(
        granularity_m,
        walk_dist_m,
        extents_trf,
        extents_arr,
        centre_seeds=[],
        build_prob=build_prob,
        cent_prob_nb=cent_prob_nb,
        cent_prob_isol=cent_prob_isol,
        max_local_pop=max_local_pop,
        prob_distribution=prob_distribution,
        density_factors=density_factors,
        random_seed=random_seed,
    )
    # prepare QGIS menu
    layer_root = QgsProject.instance().layerTreeRoot()
    layer_group = layer_root.insertGroup(0, f"{out_file_name} outputs")
    # initialise simulation
    save_snapshot(
        land,
        0,
        out_dir_path,
        out_file_name,
        extents_trf,
        target_crs,
        layer_group,
    )
    for idx in range(1, n_steps + 1):
        print(idx, n_steps)
        land.iterate()
        start = time.time()
        LOGGER.info(f"step: {idx}, duration: {time.time() - start} seconds")
        save_snapshot(
            land,
            idx,
            out_dir_path,
            out_file_name,
            extents_trf,
            target_crs,
            layer_group,
        )
    LOGGER.info(f"Simulation ended. Total duration: {time.time() - t_zero} seconds")


def save_snapshot(
    land: Land,  # type: ignore
    step: int,
    out_dir_path: Path,
    out_file_name: str,
    trf: transform.Affine,
    target_crs: QgsCoordinateReferenceSystem,
    layer_group: QgsLayerTreeGroup,
) -> None:
    """ """
    out_name = f"{out_file_name}_{step:05d}"
    out_path: str = str(out_dir_path / f"{out_name}.tif")
    crs_wkt: str = target_crs.toWkt()
    with rio.open(  # type: ignore
        out_path,
        mode="w",
        driver="GTiff",
        count=3,
        width=land.state_arr.shape[1],
        height=land.state_arr.shape[0],
        crs=crs_wkt,
        transform=trf,
        dtype=np.int16,  # type: ignore
        nodata=-1,
    ) as out_rast:  # type: ignore
        rgb = np.full((land.state_arr.shape[0], land.state_arr.shape[1], 3), 0, dtype=np.int16)
        green_idx = np.nonzero(land.state_arr == 0)
        print(green_idx)
        rgb[green_idx] = [70, 183, 42]
        # expects bands, rows, columns order
        print(rgb.transpose(2, 0, 1))
        out_rast.write(rgb.transpose(2, 0, 1))
        # create QGIS layer and renderer
        rast_layer = QgsRasterLayer(out_path, f"step {step}", providerType="gdal")
        rast_layer.setCrs(target_crs)
        # # help out type hinting via cast as IDE doesn't know ahead of time re: multiband renderer
        print(rast_layer)
        print(rast_layer.renderer())
        # rast_renderer: QgsMultiBandColorRenderer = cast(QgsMultiBandColorRenderer, rast_layer.renderer())
        # print(rast_renderer)
        # # setup renderer
        # rast_renderer.setRedBand(1)
        # rast_renderer.setGreenBand(2)
        # rast_renderer.setBlueBand(3)
        # red_ce = rast_renderer.redContrastEnhancement()
        # red_ce.setContrastEnhancementAlgorithm(QgsContrastEnhancement.StretchAndClipToMinimumMaximum)
        # red_ce.setMinimumValue(0)
        # red_ce.setMaximumValue(1)
        # green_ce = rast_renderer.greenContrastEnhancement()
        # green_ce.setContrastEnhancementAlgorithm(QgsContrastEnhancement.StretchAndClipToMinimumMaximum)
        # green_ce.setMinimumValue(0)
        # green_ce.setMaximumValue(1)
        # blue_ce = rast_renderer.blueContrastEnhancement()
        # blue_ce.setContrastEnhancementAlgorithm(QgsContrastEnhancement.StretchAndClipToMinimumMaximum)
        # blue_ce.setMinimumValue(0)
        # blue_ce.setMaximumValue(1)
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
