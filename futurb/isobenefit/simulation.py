from __future__ import annotations

import logging
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
from rasterio import features, transform
from shapely import geometry, wkt

from .land_map import Land
from .logger import configure_logging, get_logger

N_AMENITIES = 1
LOGGER: logging.Logger = get_logger()


def get_central_coord(size_x_m: int, size_y_m: int) -> list[tuple[int, int]]:
    """Returns centre coordinates given the x, y size."""
    return [(int(size_x_m / 2), int(size_y_m / 2))]


def prepare_filepath(step: int, out_dir_path: Path, out_file_name: str) -> str:
    """ """
    return str(out_dir_path / f"{out_file_name}_{step:05d}.tif")


def simulate(
    bounds_layer: QgsVectorLayer,
    extents_layer: QgsVectorLayer,
    built_areas_layer: QgsVectorLayer | None,
    green_areas_layer: QgsVectorLayer | None,
    unbuildable_areas_layer: QgsVectorLayer | None,
    centre_seeds_layer: QgsVectorLayer | None,
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
    # prepare bounds
    x_min = bounds_layer.extent().xMinimum()
    x_max = bounds_layer.extent().xMaximum()
    y_min = bounds_layer.extent().yMinimum()
    y_max = bounds_layer.extent().yMaximum()
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
    extents_arr = np.full((cells_y, cells_x), -1, dtype=np.int_)
    # transform
    extents_trf: transform.Affine = transform.from_bounds(x_min, y_min, x_max, y_max, cells_x, cells_y)
    # review input layers and configure accordingly
    for feature in extents_layer.getFeatures():
        geom_wkt = feature.geometry().asWkt()
        shapely_geom: geometry.Polygon = wkt.loads(geom_wkt)
        features.rasterize(  # type: ignore
            shapes=[(geometry.mapping(shapely_geom), 0)],  # convert back to geo interface
            out=extents_arr,
            transform=extents_trf,
            all_touched=False,
        )
    if built_areas_layer is not None:
        for feature in built_areas_layer.getFeatures():
            geom_wkt = feature.geometry().asWkt()
            shapely_geom: geometry.Polygon = wkt.loads(geom_wkt)
            features.rasterize(  # type: ignore
                shapes=[(geometry.mapping(shapely_geom), 1)],  # convert back to geo interface
                out=extents_arr,
                transform=extents_trf,
                all_touched=False,
            )
    if green_areas_layer is not None:
        for feature in green_areas_layer.getFeatures():
            geom_wkt = feature.geometry().asWkt()
            shapely_geom: geometry.Polygon = wkt.loads(geom_wkt)
            features.rasterize(  # type: ignore
                shapes=[(geometry.mapping(shapely_geom), 0)],  # convert back to geo interface
                out=extents_arr,
                transform=extents_trf,
                all_touched=False,
            )
    if unbuildable_areas_layer is not None:
        for feature in unbuildable_areas_layer.getFeatures():
            geom_wkt = feature.geometry().asWkt()
            shapely_geom: geometry.Polygon = wkt.loads(geom_wkt)
            features.rasterize(  # type: ignore
                shapes=[(geometry.mapping(shapely_geom), -1)],  # convert back to geo interface
                out=extents_arr,
                transform=extents_trf,
                all_touched=False,
            )
    centre_seeds: list[tuple[int, int]] = []
    if centre_seeds_layer is not None:
        for feature in centre_seeds_layer.getFeatures():
            point = feature.geometry().asPoint()
            centre_seeds.append((int(point.x()), int(point.y())))
    t_zero = time.time()
    print("instancing land")
    # start simulation
    land = Land(
        granularity_m,
        walk_dist_m,
        extents_trf,
        extents_arr,
        centre_seeds=centre_seeds,
        build_prob=build_prob,
        cent_prob_nb=cent_prob_nb,
        cent_prob_isol=cent_prob_isol,
        max_local_pop=max_local_pop,
        prob_distribution=prob_distribution,
        density_factors=density_factors,
        random_seed=random_seed,
    )
    print("done instancing land", time.time() - t_zero)
    # prepare QGIS menu
    layer_root = QgsProject.instance().layerTreeRoot()
    layer_group = layer_root.insertGroup(0, f"{out_file_name} outputs")
    # initialise simulation
    out_path = prepare_filepath(0, out_dir_path, out_file_name)
    save_snapshot(
        land,
        out_path,
        extents_trf,
        target_crs,
    )
    for idx in range(1, n_steps + 1):
        start = time.time()
        land.iterate()
        out_path = prepare_filepath(idx, out_dir_path, out_file_name)
        save_snapshot(
            land,
            out_path,
            extents_trf,
            target_crs,
        )
        LOGGER.info(f"step: {idx}, duration: {time.time() - start} seconds")
    # load snapshots to menu
    for idx in range(1, n_steps + 1):
        in_path = prepare_filepath(idx, out_dir_path, out_file_name)
        load_snapshot(in_path, idx, target_crs, layer_group)
    LOGGER.info(f"Simulation ended. Total duration: {time.time() - t_zero} seconds")


def save_snapshot(
    land: Land,  # type: ignore
    out_path: str,
    trf: transform.Affine,
    target_crs: QgsCoordinateReferenceSystem,
) -> None:
    """ """
    with rio.open(  # type: ignore
        out_path,
        mode="w",
        driver="GTiff",
        count=4,
        width=land.state_arr.shape[1],
        height=land.state_arr.shape[0],
        crs=target_crs.authid(),
        transform=trf,
        dtype=np.byte,  # type: ignore
        nodata=-1,
    ) as out_rast:  # type: ignore
        rgb = np.full((land.state_arr.shape[0], land.state_arr.shape[1], 4), 0, dtype=np.byte)
        # green areas
        green_idx = np.nonzero(land.state_arr == 0)
        rgb[green_idx] = [84, 171, 67, 255]
        # built areas
        built_idx = np.nonzero(land.state_arr == 1)
        rgb[built_idx] = [179, 124, 105, 255]
        # centres
        centre_idx = np.nonzero(land.state_arr == 2)
        rgb[centre_idx] = [194, 50, 50, 255]
        # expects bands, rows, columns order
        out_rast.write(rgb.transpose(2, 0, 1))


def load_snapshot(in_path: str, step: int, target_crs: QgsCoordinateReferenceSystem, layer_group: QgsLayerTreeGroup):
    """ """
    # create QGIS layer and renderer
    rast_layer = QgsRasterLayer(
        in_path,
        f"step {step}",
        providerType="gdal",
    )
    if not rast_layer.isValid():
        LOGGER.error(f"Invalid layer: {in_path}")
    rast_layer.setCrs(target_crs)
    # # help out type hinting via cast as IDE doesn't know ahead of time re: multiband renderer
    rast_renderer: QgsMultiBandColorRenderer = cast(QgsMultiBandColorRenderer, rast_layer.renderer())
    # # setup renderer
    rast_renderer.setRedBand(1)
    rast_renderer.setGreenBand(2)
    rast_renderer.setBlueBand(3)
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
    rast_layer = layer_group.addLayer(rast_layer)
    rast_layer.setExpanded(False)
