"""
Can't handle border regions in numba stencil: only 'constant' mode is currently supported, i.e. border buffer = 0

Don't use anything from QGIS so that it is easier to test this module.
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
from numba import njit
from qgis.core import (
    Qgis,
    QgsCoordinateReferenceSystem,
    QgsDateTimeRange,
    QgsInterval,
    QgsLayerTreeGroup,
    QgsMessageLog,
    QgsMultiBandColorRenderer,
    QgsProject,
    QgsRasterLayer,
    QgsRasterLayerTemporalProperties,
    QgsTask,
    QgsTemporalNavigationObject,
    QgsVectorLayer,
)
from qgis.gui import QgisInterface
from rasterio import features, transform
from shapely import BufferCapStyle, BufferJoinStyle, geometry, wkt

from . import algos


@njit
def green_to_built(
    y_idx: int,
    x_idx: int,
    state_arr: Any,
    old_green_itx_arr: Any,
    old_green_acc_arr: Any,
    buildable_arr: Any,
    granularity_m: int,
    max_distance_m: int,
) -> tuple[bool, Any, Any]:
    """
    can't track state directly... because local actions have non-local impact
    avoids checking each reachable cell's green access causes exponential complexity
    """
    new_green_itx_arr = np.copy(old_green_itx_arr)
    new_green_acc_arr = np.copy(old_green_acc_arr)
    # check neighbours situation
    _tot_urban_nbs, cont_urban_nbs, _urban_regions = algos.count_cont_nbs(state_arr, y_idx, x_idx, [1, 2])
    # if buildable_arr indicates that a green area should not be developed, then special conditions apply
    if buildable_arr[y_idx, x_idx] < 1:
        # allow filling in crimped areas
        if cont_urban_nbs < 5:
            return False, old_green_itx_arr, old_green_acc_arr
    # check if cell is currently green_itx
    # if so, set itx to off and decrement green access accordingly
    if new_green_itx_arr[y_idx, x_idx] == 2:
        new_green_itx_arr[y_idx, x_idx] = 1
        # decrement green access as consequence of converting cell from itx to built
        new_green_acc_arr -= algos.agg_dijkstra_cont(
            new_green_itx_arr, y_idx, x_idx, [0, 1, 2], [0, 1, 2], max_distance_m, granularity_m
        )
    # scan through neighbours - set new green itx - use rook for checking contiguity
    for y_nb_idx, x_nb_idx in algos.iter_nbs(new_green_itx_arr, y_idx, x_idx, rook=True):
        # convert green space to itx
        if new_green_itx_arr[y_nb_idx, x_nb_idx] == 0:
            new_green_itx_arr[y_nb_idx, x_nb_idx] = 2
            # increment green access to existing built cells
            new_green_acc_arr += algos.agg_dijkstra_cont(
                new_green_itx_arr, y_nb_idx, x_nb_idx, [0, 1, 2], [0, 1, 2], max_distance_m, granularity_m
            )
    # check that green access has not been cut off for built areas
    # green_acc_diff = new_green_acc_arr - old_green_acc_arr
    ny_idxs, nx_idxs = np.nonzero(np.logical_and(state_arr > 0, new_green_acc_arr <= 0))
    for ny_idx, nx_idx in zip(ny_idxs, nx_idxs):
        # bail if built and below zero
        if old_green_acc_arr[ny_idx, nx_idx] > new_green_acc_arr[ny_idx, nx_idx]:
            return False, old_green_itx_arr, old_green_acc_arr
    return True, new_green_itx_arr, new_green_acc_arr


class Land(QgsTask):
    """
    state_arr - is this necessary?
    -1 - out of bounds
    0 - nature
    1 - built
    2 - centre
    """

    iface_ref: QgisInterface
    # file naming
    out_file_name: str
    path_template: str
    plot_themes: list[str]
    # crs
    target_crs: QgsCoordinateReferenceSystem
    extents_transform: transform.Affine
    # substrate
    total_iters: int
    current_iter: int
    granularity_m: int
    max_distance_m: int
    max_populat: int
    min_green_km2: int | float
    trf: transform.Affine
    # parameters
    build_prob: float
    cent_prob_nb: float  # TODO: pending
    cent_prob_isol: float  # TODO: pending
    prob_distribution: tuple[float, float, float]
    density_factors: tuple[float, float, float]
    # state - QGIS / gdal numpy veresion doesn't yet support numpy typing for NDArray
    state_arr: Any
    green_itx_arr: Any
    green_acc_arr: Any
    cent_acc_arr: Any
    density_arr: Any

    def __init__(
        self,
        iface_ref: QgisInterface,
        out_dir_path: Path,
        out_file_name: str,
        target_crs: QgsCoordinateReferenceSystem,
        bounds_layer: QgsVectorLayer,
        extents_layer: QgsVectorLayer,
        built_areas_layer: QgsVectorLayer | None,
        green_areas_layer: QgsVectorLayer | None,
        unbuildable_areas_layer: QgsVectorLayer | None,
        centre_seeds_layer: QgsVectorLayer | None,
        total_iters: int,
        granularity_m: int,
        max_distance_m: int,
        max_populat: int = 10000,
        min_green_km2: int | float = 1,
        build_prob: float = 0.1,
        cent_prob_nb: float = 0.05,
        cent_prob_isol: float = 0,
        prob_distribution: tuple[float, float, float] = (0.7, 0.3, 0),
        density_factors: tuple[float, float, float] = (1, 0.1, 0.01),
        random_seed: int = 0,
    ):
        """ """
        QgsMessageLog.logMessage("Instancing Isobenefit simulation.", level=Qgis.Info)
        QgsTask.__init__(self, "Isobenefit")
        # set reference to interface
        self.iface_ref = iface_ref
        # set random state
        np.random.seed(random_seed)
        # file naming template
        self.out_file_name = out_file_name
        self.path_template = str(out_dir_path / out_file_name) + "_{theme}_{iter}.tif"
        self.plot_themes = ["state"]  # 'green', 'centre', 'density'
        # crs
        self.target_crs = target_crs
        # initialise iters state
        self.total_iters = total_iters
        self.current_iter = 0
        # prepare extents
        self.granularity_m = granularity_m
        self.max_distance_m = max_distance_m
        # params
        self.build_prob = build_prob
        self.cent_prob_nb = cent_prob_nb
        self.cent_prob_isol = cent_prob_isol
        # the assumption is that T_star is the number of blocks
        # that equals to a 15 minutes walk, i.e. roughly 1 km. 1 block has size 1000/T_star metres
        if not np.sum(prob_distribution) == 1:
            raise ValueError("The prob_distribution parameter must sum to 1.")
        self.max_populat = max_populat
        self.prob_distribution = prob_distribution
        self.density_factors = density_factors
        self.min_green_km2 = min_green_km2
        # checks
        prob_sum = sum(prob_distribution)
        if not prob_sum == 1:
            raise ValueError(f"The probability distribution doesn't sum to 1 ({prob_sum})")
        if not density_factors[0] >= density_factors[1] >= density_factors[2]:
            raise ValueError(f"The density factors are not decreasing in value: {density_factors}.")
        # prepare state
        self.prepare_state(
            bounds_layer,
            extents_layer,
            built_areas_layer,
            green_areas_layer,
            unbuildable_areas_layer,
            centre_seeds_layer,
        )
        self.save_snapshot()
        QgsMessageLog.logMessage("Isobenefit instance ready.", level=Qgis.Info)

    def prepare_state(
        self,
        bounds_layer: QgsVectorLayer,
        extents_layer: QgsVectorLayer,
        built_areas_layer: QgsVectorLayer | None,
        green_areas_layer: QgsVectorLayer | None,
        unbuildable_areas_layer: QgsVectorLayer | None,
        centre_seeds_layer: QgsVectorLayer | None,
    ) -> None:
        """ """
        # prepare bounds
        x_min = bounds_layer.extent().xMinimum()
        x_max = bounds_layer.extent().xMaximum()
        y_min = bounds_layer.extent().yMinimum()
        y_max = bounds_layer.extent().yMaximum()
        size_x_m = int(x_max - x_min)
        size_y_m = int(y_max - y_min)
        cells_x = int(np.ceil(size_x_m / self.granularity_m))
        cells_y = int(np.ceil(size_y_m / self.granularity_m))
        # checks
        if (
            not cells_x * self.granularity_m > 2 * self.max_distance_m
            or not cells_y * self.granularity_m > 2 * self.max_distance_m
        ):
            raise ValueError(f"The provided extents is too small. It should be larger than 2x walking distance.")
        # -1 unbuildable, 0 = green, 1 = built, 2 = centrality
        self.state_arr = np.full((cells_y, cells_x), -1, dtype=np.int16)
        # transform
        self.extents_transform: transform.Affine = transform.from_bounds(x_min, y_min, x_max, y_max, cells_x, cells_y)
        # review input layers and configure accordingly
        for feature in extents_layer.getFeatures():
            geom_wkt = feature.geometry().asWkt()
            shapely_geom: geometry.Polygon = wkt.loads(geom_wkt)
            features.rasterize(  # type: ignore
                shapes=[(geometry.mapping(shapely_geom), 0)],  # convert back to geo interface
                out=self.state_arr,
                transform=self.extents_transform,
                all_touched=False,
            )
        if built_areas_layer is not None:
            for feature in built_areas_layer.getFeatures():
                geom_wkt = feature.geometry().asWkt()
                shapely_geom: geometry.Polygon = wkt.loads(geom_wkt)
                features.rasterize(  # type: ignore
                    shapes=[(geometry.mapping(shapely_geom), 1)],  # convert back to geo interface
                    out=self.state_arr,
                    transform=self.extents_transform,
                    all_touched=False,
                )
        if green_areas_layer is not None:
            for feature in green_areas_layer.getFeatures():
                geom_wkt = feature.geometry().asWkt()
                shapely_geom: geometry.Polygon = wkt.loads(geom_wkt)
                features.rasterize(  # type: ignore
                    shapes=[(geometry.mapping(shapely_geom), 0)],  # convert back to geo interface
                    out=self.state_arr,
                    transform=self.extents_transform,
                    all_touched=False,
                )
        if unbuildable_areas_layer is not None:
            for feature in unbuildable_areas_layer.getFeatures():
                geom_wkt = feature.geometry().asWkt()
                shapely_geom: geometry.Polygon = wkt.loads(geom_wkt)
                features.rasterize(  # type: ignore
                    shapes=[(geometry.mapping(shapely_geom), -1)],  # convert back to geo interface
                    out=self.state_arr,
                    transform=self.extents_transform,
                    all_touched=False,
                )
        centre_seeds: list[tuple[int, int]] = []
        if centre_seeds_layer is not None:
            for feature in centre_seeds_layer.getFeatures():
                point = feature.geometry().asPoint()
                centre_seeds.append((int(point.x()), int(point.y())))
        # seed centres
        self.cent_acc_arr = np.full(self.state_arr.shape, 0, dtype=np.int16)
        y_trf: int
        x_trf: int
        for east, north in centre_seeds:
            y_trf, x_trf = transform.rowcol(self.extents_transform, east, north)  # type: ignore
            self.state_arr[y_trf, x_trf] = 2
            # agg centrality to surroundings
            self.cent_acc_arr += algos.agg_dijkstra_cont(
                self.state_arr,
                y_trf,
                x_trf,
                path_state=[0, 1, 2],
                target_state=[0, 1, 2],
                max_distance_m=self.max_distance_m,
                granularity_m=self.granularity_m,
            )
        # check size vs. min green
        area = self.state_arr.shape[0] * self.granularity_m * self.state_arr.shape[1] * self.granularity_m
        if area / 1000**2 < 2 * self.min_green_km2:
            raise ValueError("Please decrease min_green_km2 in relation to provided extents.")
        # find boundary of built land
        # 0 = green, 1 = built, 2 = itx bounds
        self.green_itx_arr, self.green_acc_arr = algos.prepare_green_arrs(
            self.state_arr, self.max_distance_m, self.granularity_m
        )
        # density
        self.density_arr = np.full(self.state_arr.shape, 0, dtype=np.float32)
        # buildable_arr is set by iter
        self.buildable_arr = np.full(self.state_arr.shape, 0, dtype=np.int16)

    def save_snapshot(self) -> None:
        """ """
        # plot state arr
        state_path = self.path_template.format(theme="state", iter=self.current_iter)
        with rio.open(  # type: ignore
            state_path,
            mode="w",
            driver="GTiff",
            count=4,
            width=self.state_arr.shape[1],
            height=self.state_arr.shape[0],
            crs=self.target_crs.authid(),
            transform=self.extents_transform,
            dtype=np.byte,  # type: ignore
            nodata=-1,
        ) as out_rast:  # type: ignore
            rgb = np.full((self.state_arr.shape[0], self.state_arr.shape[1], 4), 0, dtype=np.byte)
            # green areas
            green_idx = np.nonzero(self.state_arr == 0)
            rgb[green_idx] = [84, 171, 67, 255]
            # built areas
            built_idx = np.nonzero(self.state_arr == 1)
            rgb[built_idx] = [179, 124, 105, 255]
            # centres
            centre_idx = np.nonzero(self.state_arr == 2)
            rgb[centre_idx] = [194, 50, 50, 255]
            # expects bands, rows, columns order
            out_rast.write(rgb.transpose(2, 0, 1))

    def load_snapshots(self):
        """ """
        # prepare QGIS menu
        layer_root = QgsProject.instance().layerTreeRoot()
        for plot_theme in self.plot_themes:
            layer_group: QgsLayerTreeGroup = layer_root.insertGroup(0, f"{self.out_file_name} {plot_theme} outputs")
            for iter in range(1, self.total_iters + 1):
                load_path: str = self.path_template.format(theme=plot_theme, iter=iter)
                # create QGIS layer and renderer
                rast_layer = QgsRasterLayer(
                    load_path,
                    f"step {iter}",
                    providerType="gdal",
                )
                if not rast_layer.isValid():
                    QgsMessageLog.logMessage("Raster layer is  not valid.", level=Qgis.Critical, notifyUser=True)
                rast_layer.setCrs(self.target_crs)
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
                this_date = start_date.replace(year=start_date.year + iter)
                next_date = start_date.replace(year=start_date.year + iter + 1)
                time_range = QgsDateTimeRange(begin=this_date, end=next_date)
                temp_props.setFixedTemporalRange(time_range)
                temp_props.isVisibleInTemporalRange(time_range)
                temp_props.setIsActive(True)
                # add to QGIS
                QgsProject.instance().addMapLayer(rast_layer, addToLegend=False)
                rast_layer = layer_group.addLayer(rast_layer)
                rast_layer.setExpanded(False)

    def run(self) -> bool:
        """
        QgsTask uses 'run' as entry point for managing task.
        https://qgis.org/pyqgis/master/core/QgsTask.html
        """
        QgsMessageLog.logMessage("Starting Isobenefit simulation.", level=Qgis.Info)
        t_zero = time.time()
        for this_iter in range(self.total_iters + 1):
            if self.isCanceled():
                return False
            self.iterate()
            self.setProgress(self.current_iter / self.total_iters * 100)
            QgsMessageLog.logMessage(f"iter: {this_iter}", level=Qgis.Info)
        QgsMessageLog.logMessage(f"Simulation ended. Duration: {round(time.time() - t_zero)}s", level=Qgis.Info)
        self.load_snapshots()
        # setup temporal controller
        start_date = datetime.now()
        end_date = start_date.replace(year=start_date.year + self.total_iters)
        temporal: QgsTemporalNavigationObject = cast(
            QgsTemporalNavigationObject, self.iface_ref.mapCanvas().temporalController()
        )
        temporal.setTemporalExtents(QgsDateTimeRange(begin=start_date, end=end_date))
        temporal.rewindToStart()
        temporal.setLooping(False)
        temporal.setFrameDuration(QgsInterval(1, 0, 0, 0, 0, 0, 0))  # one year
        temporal.setFramesPerSecond(5)
        temporal.setAnimationState(QgsTemporalNavigationObject.Forward)
        return True

    def iterate(self):
        """ """
        self.current_iter += 1
        # extract green space features
        feats: list[tuple[dict, float]] = features.shapes(  # type: ignore
            self.state_arr, mask=self.state_arr == 0, connectivity=4, transform=self.extents_transform
        )
        # prime buildable_arr
        self.buildable_arr.fill(0)
        for feat, _val in feats:  # type: ignore
            poly = geometry.shape(feat)  # convert from geo interface to shapely geom
            # convert to square km - continue if below min threshold
            if poly.area < self.min_green_km2 * 1000**2:
                continue
            # reverse buffer step 1
            buffer_dist = 100
            rev_buf: geometry.Polygon | geometry.MultiPolygon = poly.buffer(
                -buffer_dist, cap_style=BufferCapStyle.square, join_style=BufferJoinStyle.mitre
            )
            geoms: list[geometry.Polygon] = []
            # if an area is split, a MultiPolygon is returned
            if isinstance(rev_buf, geometry.Polygon):
                geoms.append(rev_buf)
            elif isinstance(rev_buf, geometry.MultiPolygon):
                geoms += rev_buf.geoms
            else:
                raise ValueError("Unexpected geometry")
            buildable_geom: geometry.MultiPolygon = geometry.MultiPolygon(polygons=None)
            for geom in geoms:
                back_buf = geom.buffer(buffer_dist, cap_style=BufferCapStyle.square, join_style=BufferJoinStyle.mitre)
                # clip for situations where approaching borders
                back_buf = back_buf.intersection(poly)
                # add to buildable if larger than min threshold
                if back_buf.area >= self.min_green_km2 * 1000**2:
                    buildable_geom = buildable_geom.union(back_buf)
            # generate a negative of green extents vs. deemed buildable extents
            unbuildable_geom: geometry.MultiPolygon = geometry.MultiPolygon(polygons=None)
            neg_extents: geometry.Polygon | geometry.MultiPolygon = poly.difference(buildable_geom)
            if not neg_extents.is_empty:
                # cycle buffer to cleanly separate
                neg_geoms: list[geometry.Polygon] = []
                # if an area is split, a MultiPolygon is returned
                if isinstance(neg_extents, geometry.Polygon) and not neg_extents.is_empty:
                    neg_geoms.append(neg_extents)
                elif isinstance(neg_extents, geometry.MultiPolygon) and not neg_extents.is_empty:
                    neg_geoms += neg_extents.geoms
                # sort through neg geoms and assign to buildable or non buildable based on sizes
                for neg_geom in neg_geoms:
                    # look for smaller chunks
                    neg_buf: geometry.Polygon | geometry.MultiPolygon = neg_geom.buffer(
                        self.granularity_m, cap_style=BufferCapStyle.square, join_style=BufferJoinStyle.mitre
                    )
                    # if smaller than min - then discard
                    if neg_buf.area < self.min_green_km2 * 1000**2:
                        unbuildable_geom = unbuildable_geom.union(neg_buf)  # type: ignore
                        # difference padded from buildable to shelter new parks from rapid infill
                        buildable_geom = buildable_geom.difference(neg_buf)
                    # otherwise salvage as buildable
                    else:
                        buildable_geom = buildable_geom.union(neg_buf)
            # burn raster
            if not buildable_geom.is_empty:
                features.rasterize(  # type: ignore
                    shapes=[(geometry.mapping(buildable_geom), 1)],  # convert back to geo interface
                    out=self.buildable_arr,
                    transform=self.extents_transform,
                    all_touched=False,
                )
            if not unbuildable_geom.is_empty:
                features.rasterize(  # type: ignore
                    shapes=[(geometry.mapping(unbuildable_geom), -1)],  # convert back to geo interface
                    out=self.buildable_arr,
                    transform=self.extents_transform,
                    all_touched=False,
                )
            # plt.plot(poly.exterior.xy[0], poly.exterior.xy[1])
            # plt.plot(pos_poly.exterior.xy[0], pos_poly.exterior.xy[1])
            # plt.show()
            if False:
                scale_factor = 1 / self.granularity_m
                plt.plot(poly.exterior.xy[0], poly.exterior.xy[1])
                plt.plot(pos_poly.exterior.xy[0], pos_poly.exterior.xy[1])
                plt.show()
        # shuffle indices
        arr_idxs = list(np.ndindex(self.state_arr.shape))
        np.random.shuffle(arr_idxs)
        old_state_arr = np.copy(self.state_arr)
        for y_idx, x_idx in arr_idxs:
            # bail if already built
            if self.state_arr[y_idx, x_idx] > 0:
                continue
            # a cell can be developed if it is on the green periphery adjacent to built areas
            if self.green_itx_arr[y_idx, x_idx] == 2:
                # don't allow double steps, i.e. a new built cell has to have at least one built neighbour in
                tot_urban_nbs, _cont_urban_nbs, _urban_regions = algos.count_cont_nbs(
                    old_state_arr, y_idx, x_idx, [1, 2]
                )
                if tot_urban_nbs == 0:
                    continue
                # if centrality is accessible
                if self.cent_acc_arr[y_idx, x_idx] > 0:
                    if np.random.rand() < self.build_prob:
                        # update green state
                        success, self.green_itx_arr, self.green_acc_arr = green_to_built(
                            y_idx,
                            x_idx,
                            self.state_arr,
                            self.green_itx_arr,
                            self.green_acc_arr,
                            self.buildable_arr,
                            self.granularity_m,
                            self.max_distance_m,
                        )
                        # claim as built
                        if success is True:
                            # state
                            self.state_arr[y_idx, x_idx] = 1
                            # set random density
                            self.density_arr[y_idx, x_idx] = algos.random_density(
                                self.prob_distribution, self.density_factors
                            )
                # otherwise, consider adding a new centrality
                elif np.random.rand() < self.cent_prob_nb:
                    # update green state
                    success, self.green_itx_arr, self.green_acc_arr = green_to_built(
                        y_idx,
                        x_idx,
                        self.state_arr,
                        self.green_itx_arr,
                        self.green_acc_arr,
                        self.buildable_arr,
                        self.granularity_m,
                        self.max_distance_m,
                    )
                    if success is True:
                        # state
                        self.state_arr[y_idx, x_idx] = 2
                        # increment centrality access
                        self.cent_acc_arr += algos.agg_dijkstra_cont(
                            self.state_arr,
                            y_idx,
                            x_idx,
                            path_state=[0, 1, 2],
                            target_state=[0, 1, 2],
                            max_distance_m=self.max_distance_m,
                            granularity_m=self.granularity_m,
                        )
                        # set random density
                        self.density_arr[y_idx, x_idx] = algos.random_density(
                            self.prob_distribution, self.density_factors
                        )
            # handle random conversion of green space to centralities
            elif self.state_arr[y_idx, x_idx] == 0:
                if np.random.rand() < 0:  # self.cent_prob_isol:
                    # if self.nature_stays_extended(x, y):
                    # if self.nature_stays_reachable(x, y):
                    # update green state
                    success, self.green_itx_arr, self.green_acc_arr = green_to_built(
                        y_idx,
                        x_idx,
                        self.state_arr,
                        self.green_itx_arr,
                        self.green_acc_arr,
                        self.buildable_arr,
                        self.granularity_m,
                        self.max_distance_m,
                    )
                    if success is True:
                        # claim as built
                        self.state_arr[y_idx, x_idx] = 2
                        self.cent_acc_arr = algos._inc_access(
                            y_idx, x_idx, self.cent_acc_arr, self.granularity_m, self.max_distance_m
                        )
                        # set random density
                        self.density_arr[y_idx, x_idx] = algos.random_density(
                            self.prob_distribution, self.density_factors
                        )
        # write iter snapshot
        self.save_snapshot()

    @property
    def population_density(self) -> dict[str, float]:
        """ """
        return {
            "high": self.density_factors[0],
            "medium": self.density_factors[1],
            "low": self.density_factors[2],
            "empty": 0,
        }
