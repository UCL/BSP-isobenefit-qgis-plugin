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
from shapely import geometry, wkt

from . import algos


@njit
def green_to_built(
    y_idx: int,
    x_idx: int,
    state_arr: Any,
    old_green_itx_arr: Any,
    old_green_acc_arr: Any,
    granularity_m: int,
    max_distance_m: int,
    min_green_span: int,
) -> tuple[bool, Any, Any]:
    """
    can't track state directly... because local actions have non-local impact
    avoids checking each reachable cell's green access causes exponential complexity
    """
    new_green_itx_arr = np.copy(old_green_itx_arr)
    new_green_acc_arr = np.copy(old_green_acc_arr)
    # check neighbours situation
    _tot_urban_nbs, cont_urban_nbs, urban_regions = algos.count_cont_nbs(state_arr, y_idx, x_idx, [1, 2])
    # if a single neighbour, don't proceed unless that neighbour is a centrality
    # this prevents runaway single streaks of built areas
    if cont_urban_nbs == 1:
        _cent_tot_urban_nbs, cent_cont_urban_nbs, _cent_urban_regions = algos.count_cont_nbs(
            state_arr, y_idx, x_idx, [2]
        )
        if not cent_cont_urban_nbs == 1:
            return False, old_green_itx_arr, old_green_acc_arr
    # bail if a green span would be crimped below min
    if not algos.green_spans(state_arr, y_idx, x_idx, granularity_m, min_green_span):
        return False, old_green_itx_arr, old_green_acc_arr
    # if splitting green into two regions
    if urban_regions > 1:
        # required number of contiguous green cells for min green area
        target_count = int((min_green_span**2 * 1000**2) / granularity_m**2)
        # use a mock state - otherwise dijkstra doesn't know that current y, x is tentatively built
        mock_state_arr = np.copy(state_arr)
        # mock built state
        mock_state_arr[y_idx, x_idx] = 1
        # review neighbours in turn to check that each has access to continuous green space
        for y_nb_idx, x_nb_idx in algos.iter_nbs(new_green_itx_arr, y_idx, x_idx, rook=False):
            if not state_arr[y_nb_idx, x_nb_idx] == 0:
                continue
            nb_green_acc_arr = algos.agg_dijkstra_cont(
                mock_state_arr,
                y_nb_idx,
                x_nb_idx,
                [0],
                [0],
                max_distance_m=max_distance_m * 2,
                granularity_m=granularity_m,
                break_count=target_count,
                rook=True,  # rook has to be true otherwise diagonal steps are allowed
            )
            if nb_green_acc_arr.sum() < target_count:
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
    ny_idxs, nx_idxs = np.nonzero(np.logical_and(state_arr > 0, new_green_acc_arr <= 0))
    for ny_idx, nx_idx in zip(ny_idxs, nx_idxs):
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
    # save input layers
    bounds_layer: QgsVectorLayer
    extents_layer: QgsVectorLayer
    built_areas_layer: QgsVectorLayer | None
    green_areas_layer: QgsVectorLayer | None
    unbuildable_areas_layer: QgsVectorLayer | None
    centre_seeds_layer: QgsVectorLayer | None
    # substrate
    total_iters: int
    current_iter: int
    granularity_m: int
    max_distance_m: int
    max_populat: int
    min_green_span: int | float
    trf: transform.Affine
    # parameters
    build_prob: float
    cent_prob_nb: float
    cent_prob_isol: float
    pop_target_ratio: float  # for tracking current ratio of target population
    pop_target_cent_threshold: float  # parameter above which new centralities are not created
    prob_distribution: tuple[float, float, float]
    exist_built_density_per_block: float
    high_density_per_block: float
    med_density_per_block: float
    low_density_per_block: float
    pop_target_ratio: float
    # state - QGIS / gdal numpy veresion doesn't yet support numpy typing for NDArray
    state_arr: Any  # for tracking simulation state
    origin_arr: Any  # copy of origin state for plots
    green_itx_arr: Any  # green periphery - i.e. candidate for buildable
    green_acc_arr: Any  # access to green space
    cent_acc_arr: Any  # access to centres
    density_arr: Any  # density - low, mid, high

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
        max_populat: int,
        exist_built_density: int,
        min_green_span: int,
        build_prob: float,
        cent_prob_nb: float,
        cent_prob_isol: float,
        pop_target_cent_threshold: float,
        prob_distribution: tuple[float, float, float] = (0.6, 0.3, 0.1),
        density_factors: tuple[float, float, float] = (8000, 4000, 2000),
        random_seed: int = 0,
    ):
        """ """
        QgsMessageLog.logMessage("Instancing Isobenefit simulation.", level=Qgis.Info)
        QgsTask.__init__(self, "Future Urban Growth")
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
        # save input layers
        self.bounds_layer = bounds_layer
        self.extents_layer = extents_layer
        self.built_areas_layer = built_areas_layer
        self.green_areas_layer = green_areas_layer
        self.unbuildable_areas_layer = unbuildable_areas_layer
        self.centre_seeds_layer = centre_seeds_layer
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
        self.pop_target_cent_threshold = pop_target_cent_threshold
        self.max_populat = max_populat
        self.prob_distribution = prob_distribution
        # convert density factors to per block
        self.exist_built_density_per_block = exist_built_density / 1000**2 * granularity_m**2
        self.high_density_per_block = density_factors[0] / 1000**2 * granularity_m**2
        self.med_density_per_block = density_factors[1] / 1000**2 * granularity_m**2
        self.low_density_per_block = density_factors[2] / 1000**2 * granularity_m**2
        self.pop_target_ratio = 0
        self.min_green_span = min_green_span
        # checks
        prob_sum = round(sum(prob_distribution), 2)
        if not prob_sum == 1:
            raise ValueError("The prob_distribution parameter must sum to 1.")
        if not density_factors[0] > density_factors[1] or not density_factors[1] > density_factors[2]:
            raise ValueError("Density factors should be in descending order")
        QgsMessageLog.logMessage("Isobenefit instance ready.", level=Qgis.Info)

    def prepare_state(self) -> None:
        """ """
        # prepare bounds
        x_min = self.bounds_layer.extent().xMinimum()
        x_max = self.bounds_layer.extent().xMaximum()
        y_min = self.bounds_layer.extent().yMinimum()
        y_max = self.bounds_layer.extent().yMaximum()
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
        # -1 unbuildable, 0 = fixed green, 1 = exist built, 2 = exist centrality
        self.origin_arr = np.full((cells_y, cells_x), -1, dtype=np.int16)
        # transform
        self.extents_transform: transform.Affine = transform.from_bounds(x_min, y_min, x_max, y_max, cells_x, cells_y)
        # review input layers and configure accordingly
        for feature in self.extents_layer.getFeatures():
            geom_wkt = feature.geometry().asWkt()
            shapely_geom: geometry.Polygon = wkt.loads(geom_wkt)
            features.rasterize(  # type: ignore
                shapes=[(geometry.mapping(shapely_geom), 0)],  # convert back to geo interface
                out=self.state_arr,
                transform=self.extents_transform,
                all_touched=False,
            )
        # density
        self.density_arr = np.full(self.state_arr.shape, 0, dtype=np.float32)
        if self.built_areas_layer is not None:
            for feature in self.built_areas_layer.getFeatures():
                geom_wkt = feature.geometry().asWkt()
                shapely_geom: geometry.Polygon = wkt.loads(geom_wkt)
                features.rasterize(  # type: ignore
                    shapes=[(geometry.mapping(shapely_geom), 1)],  # convert back to geo interface
                    out=self.state_arr,
                    transform=self.extents_transform,
                    all_touched=False,
                )
                features.rasterize(  # type: ignore
                    shapes=[(geometry.mapping(shapely_geom), 1)],  # convert back to geo interface
                    out=self.origin_arr,
                    transform=self.extents_transform,
                    all_touched=False,
                )
            # initialise density based on existing urban areas
            self.density_arr[self.state_arr > 0] = self.exist_built_density_per_block
        if self.green_areas_layer is not None:
            for feature in self.green_areas_layer.getFeatures():
                geom_wkt = feature.geometry().asWkt()
                shapely_geom: geometry.Polygon = wkt.loads(geom_wkt)
                features.rasterize(  # type: ignore
                    shapes=[(geometry.mapping(shapely_geom), 0)],  # convert back to geo interface
                    out=self.state_arr,
                    transform=self.extents_transform,
                    all_touched=False,
                )
                # treated as intentional (preserved) park space
                features.rasterize(  # type: ignore
                    shapes=[(geometry.mapping(shapely_geom), 0)],  # convert back to geo interface
                    out=self.origin_arr,
                    transform=self.extents_transform,
                    all_touched=False,
                )
        if self.unbuildable_areas_layer is not None:
            for feature in self.unbuildable_areas_layer.getFeatures():
                geom_wkt = feature.geometry().asWkt()
                shapely_geom: geometry.Polygon = wkt.loads(geom_wkt)
                features.rasterize(  # type: ignore
                    shapes=[(geometry.mapping(shapely_geom), -1)],  # convert back to geo interface
                    out=self.state_arr,
                    transform=self.extents_transform,
                    all_touched=False,
                )
        centre_seeds: list[tuple[int, int]] = []
        if self.centre_seeds_layer is not None:
            for feature in self.centre_seeds_layer.getFeatures():
                point = feature.geometry().asPoint()
                centre_seeds.append((int(point.x()), int(point.y())))
        # seed centres
        self.cent_acc_arr = np.full(self.state_arr.shape, 0, dtype=np.int16)
        y_trf: int
        x_trf: int
        for east, north in centre_seeds:
            y_trf, x_trf = transform.rowcol(self.extents_transform, east, north)  # type: ignore
            self.state_arr[y_trf, x_trf] = 2
            self.origin_arr[y_trf, x_trf] = 2
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
        # find boundary of built land
        # 0 = green, 1 = built, 2 = itx bounds
        self.green_itx_arr, self.green_acc_arr = algos.prepare_green_arrs(
            self.state_arr, self.max_distance_m, self.granularity_m
        )

    def save_snapshot(self) -> None:
        """
        Colours to match netlogo scenarios, using netlogo scheme:
        https://ccl.northwestern.edu/netlogo/docs/programming.html#colors
        RGB colours taken with color picker and included below:
        water_bodies: blue (105) - 52, 93, 169
        prin_transport: black (0) - 0, 0, 0
        park: dark green (53) - 54, 109, 35
        green_area: green (55) - 89, 176, 60
        low_den_built: dark grey (2) - 59, 59, 59
        med_den_built: med-grey (4) - 114, 114, 114
        high_den_built: light grey (6) - 164, 164, 164
        centrality: white (9.9) - 255, 255, 255
        new_high_den_built: dark orange (22) - 101, 44, 7
        new_med_den_built: med-orange (24) - 197, 86, 17
        new_low_den_built: light-orange (26) - 242, 136, 68
        """
        # colours
        col_water_bodies = [52, 93, 169, 255]
        col_transport = [0, 0, 0, 255]
        col_park = [54, 109, 35, 255]
        col_green_area = [89, 176, 60, 255]
        col_low_den_built = [59, 59, 59, 255]
        col_med_den_built = [114, 114, 114, 255]
        col_high_den_built = [164, 164, 164, 255]
        col_centrality = [255, 255, 255, 255]
        col_new_high_den_built = [101, 44, 7, 255]
        col_new_med_den_built = [197, 86, 17, 255]
        col_new_low_den_built = [242, 136, 68, 255]
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
            # initialise with fully transparent (fallback to unbuildable)
            rgb = np.full((self.state_arr.shape[0], self.state_arr.shape[1], 4), 0, dtype=np.byte)
            # start with state array (do origin array later)
            # green areas
            green_idx = np.nonzero(self.state_arr == 0)
            rgb[green_idx] = col_green_area
            # built areas
            low_dens_idx = np.nonzero(self.density_arr == self.low_density_per_block)
            rgb[low_dens_idx] = col_new_low_den_built
            med_dens_idx = np.nonzero(self.density_arr == self.med_density_per_block)
            rgb[med_dens_idx] = col_new_med_den_built
            high_dens_idx = np.nonzero(self.density_arr == self.high_density_per_block)
            rgb[high_dens_idx] = col_new_high_den_built
            # centres
            centre_idx = np.nonzero(self.state_arr == 2)
            rgb[centre_idx] = col_centrality
            # origin built
            built_idx = np.nonzero(self.origin_arr == 1)
            rgb[built_idx] = col_med_den_built
            # origin green
            built_idx = np.nonzero(self.origin_arr == 0)
            rgb[built_idx] = col_park
            # expects bands, rows, columns order
            out_rast.write(rgb.transpose(2, 0, 1))

    def load_snapshots(self):
        """ """
        # prepare QGIS menu
        layer_root = QgsProject.instance().layerTreeRoot()
        for plot_theme in self.plot_themes:
            layer_group: QgsLayerTreeGroup = layer_root.insertGroup(0, f"{self.out_file_name} {plot_theme} outputs")
            layer_group.setExpanded(False)
            # use current iter because not all iters will have runned if population target has been reached
            for iter in range(1, self.current_iter + 1):
                load_path: str = self.path_template.format(theme=plot_theme, iter=iter)
                # small wait to reduce crashes on file loading
                time.sleep(0.05)
                # first test for existence in case previously written files are still settling
                # prompted by QGIS crashes when loading files to GUI
                wait_idx = 10
                while wait_idx > 0:
                    wait_idx -= 1
                    if not Path(load_path).exists():
                        QgsMessageLog.logMessage(f"{load_path} is not yet available, waiting 1s.", level=Qgis.Critical)
                        time.sleep(1)
                    else:
                        break
                    if wait_idx == 0 and not Path(load_path).exists():
                        QgsMessageLog.logMessage(
                            f"{load_path} does not appear to be available.", level=Qgis.Critical, notifyUser=True
                        )
                        break
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
        t_zero = time.time()
        # prepare state
        QgsMessageLog.logMessage("Preparing starting state.", level=Qgis.Info)
        self.prepare_state()
        self.save_snapshot()
        self.pop_target_ratio = self.density_arr.sum() / self.max_populat
        QgsMessageLog.logMessage(
            f"Starting population count {int(self.density_arr.sum())}; "
            f"which is {self.pop_target_ratio:.0%} of the {self.max_populat} persons target.",
            level=Qgis.Info,
        )
        if self.pop_target_ratio >= 1:
            QgsMessageLog.logMessage(
                f"Randomly assigned population for existing urban areas exceeds the target population; aborting.",
                level=Qgis.Info,
                notifyUser=True,
            )
        else:
            QgsMessageLog.logMessage("Starting iterations.", level=Qgis.Info)
            for _ in range(self.total_iters):
                if self.isCanceled():
                    return False
                self.iterate()
                self.setProgress(self.current_iter / self.total_iters * 100)
                self.pop_target_ratio = self.density_arr.sum() / self.max_populat
                QgsMessageLog.logMessage(
                    f"iter: {self.current_iter}; {self.pop_target_ratio:.0%} of population target", level=Qgis.Info
                )
                if self.pop_target_ratio >= 1:
                    QgsMessageLog.logMessage(f"Population target reached", level=Qgis.Info, notifyUser=True)
                    break
            QgsMessageLog.logMessage(
                f"Simulation ended; total duration: {round(time.time() - t_zero)}s", level=Qgis.Info
            )
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
            QgsMessageLog.logMessage(
                f"Ending population count {int(self.density_arr.sum())}; "
                f"which is {self.pop_target_ratio:.0%} of the {self.max_populat} persons target.",
                level=Qgis.Info,
            )
        return True

    def iterate(self):
        """ """
        self.current_iter += 1
        # track new centralities (whether neighbouring or isolated) - to enforce max of 1 per iter
        centrality_this_iter = False
        # shuffle indices
        arr_idxs = list(np.ndindex(self.state_arr.shape))
        np.random.shuffle(arr_idxs)
        old_state_arr = np.copy(self.state_arr)
        for y_idx, x_idx in arr_idxs:
            # bail if already built or if unbuildable
            if self.state_arr[y_idx, x_idx] != 0:
                continue
            # bail if existing green space
            if self.origin_arr[y_idx, x_idx] == 0:
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
                            self.granularity_m,
                            self.max_distance_m,
                            self.min_green_span,
                        )
                        # claim as built
                        if success is True:
                            # state
                            self.state_arr[y_idx, x_idx] = 1
                            # set random density
                            self.density_arr[y_idx, x_idx] = algos.random_density(
                                self.prob_distribution,
                                self.high_density_per_block,
                                self.med_density_per_block,
                                self.low_density_per_block,
                            )
                # otherwise, consider adding a new centrality
                elif (
                    centrality_this_iter is False
                    and self.pop_target_ratio <= self.pop_target_cent_threshold
                    and np.random.rand() < self.cent_prob_nb
                ):
                    # update green state
                    success, self.green_itx_arr, self.green_acc_arr = green_to_built(
                        y_idx,
                        x_idx,
                        self.state_arr,
                        self.green_itx_arr,
                        self.green_acc_arr,
                        self.granularity_m,
                        self.max_distance_m,
                        self.min_green_span,
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
                            self.prob_distribution,
                            self.high_density_per_block,
                            self.med_density_per_block,
                            self.low_density_per_block,
                        )
                        centrality_this_iter = True
            # handle random conversion of green space to centralities
            elif centrality_this_iter is False and self.state_arr[y_idx, x_idx] == 0:
                if self.pop_target_ratio <= self.pop_target_cent_threshold and np.random.rand() < self.cent_prob_isol:
                    # update green state
                    success, self.green_itx_arr, self.green_acc_arr = green_to_built(
                        y_idx,
                        x_idx,
                        self.state_arr,
                        self.green_itx_arr,
                        self.green_acc_arr,
                        self.granularity_m,
                        self.max_distance_m,
                        self.min_green_span,
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
                            self.prob_distribution,
                            self.high_density_per_block,
                            self.med_density_per_block,
                            self.low_density_per_block,
                        )
                        centrality_this_iter = True
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
