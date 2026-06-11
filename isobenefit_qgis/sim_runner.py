"""QgsTask wrapper that drives the Rust simulation core.

Reads the input layers via :mod:`gis_io` (reprojecting to the target CRS),
constructs an ``isobenefit.Simulation``, runs it iteration-by-iteration with
QGIS progress/cancellation, writes a categorical GeoTIFF per step, and — on the
main thread in ``finished()`` — loads them as a temporal animation.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from qgis.core import (
    Qgis,
    QgsDateTimeRange,
    QgsInterval,
    QgsMessageLog,
    QgsProject,
    QgsRasterLayer,
    QgsRasterLayerTemporalProperties,
    QgsTask,
    QgsTemporalNavigationObject,
)

from . import gis_io


class IsobenefitTask(QgsTask):
    """Background task: build inputs -> run core -> write rasters -> load them."""

    def __init__(
        self,
        *,
        iface,
        out_dir_path,
        out_file_name,
        target_crs,
        extents_layer,
        built_layer,
        green_layer,
        unbuildable_layer,
        centre_seeds_layer,
        total_iters,
        granularity_m,
        max_distance_m,
        max_populat,
        exist_built_density,
        min_green_span,
        build_prob,
        cent_prob_nb,
        cent_prob_isol,
        pop_target_cent_threshold,
        prob_distribution,
        density_factors,
        random_seed,
    ):
        super().__init__("Isobenefit simulation")
        self.iface = iface
        self.out_file_name = out_file_name
        self.path_template = str(Path(out_dir_path) / out_file_name) + "_{iter}.tif"
        self.target_crs = target_crs
        self.extents_layer = extents_layer
        self.built_layer = built_layer
        self.green_layer = green_layer
        self.unbuildable_layer = unbuildable_layer
        self.centre_seeds_layer = centre_seeds_layer
        self.total_iters = int(total_iters)
        self.granularity_m = float(granularity_m)
        self.max_distance_m = float(max_distance_m)
        self.max_populat = float(max_populat)
        self.exist_built_density = float(exist_built_density)
        self.min_green_span = float(min_green_span)
        self.build_prob = float(build_prob)
        self.cent_prob_nb = float(cent_prob_nb)
        self.cent_prob_isol = float(cent_prob_isol)
        self.pop_target_cent_threshold = float(pop_target_cent_threshold)
        self.prob_distribution = tuple(float(p) for p in prob_distribution)
        self.density_factors = tuple(float(d) for d in density_factors)
        self.random_seed = int(random_seed)
        # populated during run()
        self.geotransform = None
        self.per_block = None
        self.written_paths: list[tuple[int, str]] = []
        self.error_message: str | None = None

    def _per_block(self) -> tuple[float, float, float]:
        block = self.granularity_m**2 / 1.0e6
        return tuple(d * block for d in self.density_factors)

    def run(self) -> bool:
        try:
            import isobenefit
        except Exception as exc:  # core not importable for some reason
            self.error_message = f"Could not import the simulation engine: {exc}"
            return False
        try:
            rows, cols, geotransform, _bounds = gis_io.prepare_grid(
                self.extents_layer, self.target_crs, self.granularity_m
            )
            self.geotransform = geotransform
            if (
                cols * self.granularity_m <= 2 * self.max_distance_m
                or rows * self.granularity_m <= 2 * self.max_distance_m
            ):
                self.error_message = "The extents are too small — they must exceed 2x the walking distance."
                return False

            state = np.full((rows, cols), -1, dtype=np.int16)
            state = gis_io.burn_layer(state, self.extents_layer, self.target_crs, geotransform, 0)
            origin = np.full((rows, cols), -1, dtype=np.int16)
            if self.built_layer is not None:
                state = gis_io.burn_layer(state, self.built_layer, self.target_crs, geotransform, 1)
                origin = gis_io.burn_layer(origin, self.built_layer, self.target_crs, geotransform, 1)
            if self.green_layer is not None:
                state = gis_io.burn_layer(state, self.green_layer, self.target_crs, geotransform, 0)
                origin = gis_io.burn_layer(origin, self.green_layer, self.target_crs, geotransform, 0)
            if self.unbuildable_layer is not None:
                state = gis_io.burn_layer(state, self.unbuildable_layer, self.target_crs, geotransform, -1)
            density = np.zeros((rows, cols), dtype=np.float32)
            seeds = []
            if self.centre_seeds_layer is not None:
                seeds = gis_io.point_cells(self.centre_seeds_layer, self.target_crs, geotransform, rows, cols)

            self.per_block = self._per_block()
            sim = isobenefit.Simulation(
                state,
                origin,
                density,
                seeds,
                self.granularity_m,
                self.max_distance_m,
                self.max_populat,
                self.min_green_span,
                self.build_prob,
                self.cent_prob_nb,
                self.cent_prob_isol,
                self.pop_target_cent_threshold,
                self.prob_distribution,
                self.density_factors,
                self.exist_built_density,
                self.total_iters,
                self.random_seed,
            )
            self._write_snapshot(sim, 0)
            for i in range(self.total_iters):
                if self.isCanceled():
                    return False
                sim.step()
                self.setProgress((i + 1) / self.total_iters * 100.0)
                self._write_snapshot(sim, i + 1)
                if sim.pop_target_ratio >= 1.0:
                    break
            return True
        except Exception as exc:
            self.error_message = str(exc)
            return False

    def _write_snapshot(self, sim, iteration: int) -> None:
        snap = sim.snapshot()
        cls = gis_io.classify(snap["state"], snap["origin"], snap["density"], self.per_block)
        path = self.path_template.format(iter=iteration)
        gis_io.write_class_raster(path, cls, self.geotransform, self.target_crs)
        self.written_paths.append((iteration, path))

    def finished(self, result: bool) -> None:
        if not result:
            QgsMessageLog.logMessage(
                f"Isobenefit simulation did not complete: {self.error_message or 'cancelled'}",
                level=Qgis.MessageLevel.Warning,
                notifyUser=True,
            )
            return
        root = QgsProject.instance().layerTreeRoot()
        group = root.insertGroup(0, f"{self.out_file_name} outputs")
        group.setExpanded(False)
        start = datetime.now()
        for iteration, path in self.written_paths:
            layer = QgsRasterLayer(path, f"step {iteration}", "gdal")
            if not layer.isValid():
                QgsMessageLog.logMessage(f"Invalid output raster: {path}", level=Qgis.MessageLevel.Warning)
                continue
            layer.setCrs(self.target_crs)
            gis_io.apply_palette(layer)
            tprops = layer.temporalProperties()
            tprops.setMode(QgsRasterLayerTemporalProperties.TemporalMode.ModeFixedTemporalRange)
            begin = start.replace(year=start.year + iteration)
            end = start.replace(year=start.year + iteration + 1)
            tprops.setFixedTemporalRange(QgsDateTimeRange(begin, end))
            tprops.setIsActive(True)
            QgsProject.instance().addMapLayer(layer, addToLegend=False)
            node = group.addLayer(layer)
            node.setExpanded(False)
        self._setup_temporal_controller(start)
        QgsMessageLog.logMessage("Isobenefit simulation complete.", level=Qgis.MessageLevel.Info, notifyUser=True)

    def _setup_temporal_controller(self, start: datetime) -> None:
        temporal = self.iface.mapCanvas().temporalController()
        if temporal is None:
            return
        end = start.replace(year=start.year + max(1, len(self.written_paths)))
        temporal.setTemporalExtents(QgsDateTimeRange(start, end))
        temporal.rewindToStart()
        temporal.setLooping(False)
        temporal.setFrameDuration(QgsInterval(1, 0, 0, 0, 0, 0, 0))
        temporal.setFramesPerSecond(5)
        temporal.setAnimationState(QgsTemporalNavigationObject.AnimationState.Forward)
