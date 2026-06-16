"""QgsTask wrapper that drives the Rust simulation core.

Reads the input layers via :mod:`gis_io` (reprojecting to the target CRS),
constructs an ``isobenefit.Simulation``, runs it iteration-by-iteration with QGIS
progress/cancellation and verbose logging, accumulates one categorical frame per
step, and — on the main thread in ``finished()`` — writes a **single multi-band
GeoTIFF** (one band per step) loaded as a temporal animation (``FixedRangePerBand``).
"""

from __future__ import annotations

import os
import time
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
    QgsTask,
    QgsTemporalNavigationObject,
)

from . import gis_io, grid

LOG_TAG = "Isobenefit"


class IsobenefitTask(QgsTask):
    """Background task: build inputs -> run core -> write one temporal raster -> load it."""

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
        n_ensemble=1,
    ):
        super().__init__("Isobenefit simulation")
        self.iface = iface
        self.out_file_name = out_file_name
        self.out_path = str(Path(out_dir_path) / f"{out_file_name}.tif")
        self.plan_path = str(Path(out_dir_path) / f"{out_file_name}_plan.tif")
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
        self.n_ensemble = int(n_ensemble)
        self.is_ensemble = self.n_ensemble > 1
        # populated during run()
        self.geotransform = None
        self.per_block = None
        self.frames: list[np.ndarray] = []  # one categorical (uint8) frame per step
        self.error_message: str | None = None

    @staticmethod
    def _log(message: str, level=Qgis.MessageLevel.Info, notify: bool = False) -> None:
        QgsMessageLog.logMessage(message, LOG_TAG, level=level, notifyUser=notify)

    def _per_block(self) -> tuple[float, float, float]:
        block = self.granularity_m**2 / 1.0e6
        return tuple(d * block for d in self.density_factors)

    def run(self) -> bool:
        t_zero = time.time()
        try:
            import isobenefit
        except Exception as exc:  # core not importable for some reason
            self.error_message = f"Could not import the simulation engine: {exc}"
            return False
        try:
            self._log("Preparing simulation grid from the extents layer…")
            rows, cols, geotransform, _bounds = gis_io.prepare_grid(
                self.extents_layer, self.target_crs, self.granularity_m
            )
            self.geotransform = geotransform
            self._log(
                f"Grid: {cols}×{rows} cells at {self.granularity_m:.0f} m "
                f"({cols * self.granularity_m / 1000:.1f}×{rows * self.granularity_m / 1000:.1f} km); "
                f"up to {self.total_iters} iterations; CRS {self.target_crs.authid()}."
            )
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
                self._log("Burned existing built areas.")
            if self.green_layer is not None:
                state = gis_io.burn_layer(state, self.green_layer, self.target_crs, geotransform, 0)
                origin = gis_io.burn_layer(origin, self.green_layer, self.target_crs, geotransform, 0)
                self._log("Burned existing green areas.")
            if self.unbuildable_layer is not None:
                state = gis_io.burn_layer(state, self.unbuildable_layer, self.target_crs, geotransform, -1)
                self._log("Burned unbuildable areas.")
            density = np.zeros((rows, cols), dtype=np.float32)
            seeds = []
            if self.centre_seeds_layer is not None:
                seeds = gis_io.point_cells(self.centre_seeds_layer, self.target_crs, geotransform, rows, cols)
                self._log(f"Placed {len(seeds)} centre seed(s).")

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
            self._log(
                f"Starting population {int(sim.population)} "
                f"({sim.pop_target_ratio:.0%} of the {self.max_populat:.0f} target)."
            )
            if sim.pop_target_ratio >= 1.0:
                self.error_message = (
                    f"The existing built area already holds {int(sim.population)} people — "
                    f"{sim.pop_target_ratio:.0%} of the {self.max_populat:.0f} target, so there is nothing "
                    "to simulate. Raise the target population (or lower the built density) and rerun."
                )
                return False

            if self.is_ensemble:
                cores = os.cpu_count() or 4
                n = self.n_ensemble
                batch = max(1, cores)  # ~one run per core keeps all cores busy
                self._log(f"Running an ensemble of {n} simulations across {cores} cores…")
                # Blend in batches so we can drive the progress bar and allow
                # cancellation. Each batch returns per-class counts; we sum them and
                # divide by N to get P(built), P(green), P(centre).
                built = green = centre = None
                done = 0
                batch_idx = 0
                while done < n:
                    if self.isCanceled():
                        self._log("Simulation cancelled by user.", Qgis.MessageLevel.Warning)
                        return False
                    members = min(batch, n - done)
                    # distinct, reproducible seed per batch so members never repeat
                    b, g, c = isobenefit.ensemble_class_counts(sim, self.random_seed + batch_idx, members)
                    built = b if built is None else built + b
                    green = g if green is None else green + g
                    centre = c if centre is None else centre + c
                    done += members
                    batch_idx += 1
                    self.setProgress(done / n * 100.0)
                    self._log(f"ensemble: {done}/{n} runs blended")
                gis_io.write_probability_bands(
                    self.out_path,
                    [built / n, green / n, centre / n],
                    ["built likelihood", "green likelihood", "centre likelihood"],
                    geotransform,
                    self.target_crs,
                )
                plan = grid.recommended_plan(
                    built / n,
                    green / n,
                    self.granularity_m,
                    self.min_green_span,
                    self.max_distance_m,
                )
                # hybrid polish: carve an equitable green network into the consensus.
                # Population-aware — green is funded by densifying the rest up to the
                # max tier, never by deleting homes (Isobenefit "constant inhabitants").
                mean_density = sum(p * d for p, d in zip(self.prob_distribution, self.density_factors))
                plan = grid.optimise_plan(
                    plan,
                    self.granularity_m,
                    self.min_green_span,
                    self.max_distance_m,
                    mean_density=mean_density,
                    max_density=max(self.density_factors),
                )
                gis_io.write_plan_raster(self.plan_path, plan, geotransform, self.target_crs)
                self._log(
                    f"Ensemble finished in {time.time() - t_zero:.0f}s; "
                    f"wrote built/green/centre likelihood: {self.out_path}"
                )
                return True

            self._log("Running…")
            self.frames.append(self._frame(sim))  # step 0 (initial state)
            for i in range(self.total_iters):
                if self.isCanceled():
                    self._log("Simulation cancelled by user.", Qgis.MessageLevel.Warning)
                    return False
                sim.step()
                self.setProgress((i + 1) / self.total_iters * 100.0)
                self.frames.append(self._frame(sim))
                self._log(
                    f"iter {i + 1}/{self.total_iters}: "
                    f"{sim.pop_target_ratio:.0%} of population target "
                    f"(population {int(sim.population)})"
                )
                if sim.pop_target_ratio >= 1.0:
                    self._log("Population target reached — stopping early.", Qgis.MessageLevel.Success)
                    break

            self._log(f"Writing {len(self.frames)} steps to a single temporal raster: {self.out_path}")
            gis_io.write_temporal_class_raster(self.out_path, self.frames, geotransform, self.target_crs)
            self._log(f"Simulation finished in {time.time() - t_zero:.0f}s ({len(self.frames)} steps).")
            return True
        except Exception as exc:
            self.error_message = str(exc)
            return False

    def _frame(self, sim) -> np.ndarray:
        snap = sim.snapshot()
        return gis_io.classify(snap["state"], snap["origin"], snap["density"], self.per_block)

    def finished(self, result: bool) -> None:
        if not result:
            self._log(
                f"Isobenefit simulation did not complete: {self.error_message or 'cancelled'}",
                Qgis.MessageLevel.Warning,
                notify=True,
            )
            return
        if self.is_ensemble:
            root = QgsProject.instance().layerTreeRoot()
            group = root.insertGroup(0, f"{self.out_file_name} likelihood")
            group.setExpanded(True)
            for band, label in [(1, "built"), (2, "green"), (3, "centre")]:
                lyr = QgsRasterLayer(self.out_path, f"{self.out_file_name} — {label} likelihood", "gdal")
                if not lyr.isValid():
                    self._log(f"Output raster is not valid: {self.out_path}", Qgis.MessageLevel.Critical, notify=True)
                    return
                lyr.setCrs(self.target_crs)
                gis_io.apply_probability_style(lyr, band, gis_io.PROB_RAMPS[label])
                QgsProject.instance().addMapLayer(lyr, addToLegend=False)
                group.addLayer(lyr)
            plan_layer = QgsRasterLayer(self.plan_path, f"{self.out_file_name} — recommended plan", "gdal")
            if plan_layer.isValid():
                plan_layer.setCrs(self.target_crs)
                gis_io.apply_plan_style(plan_layer)
                QgsProject.instance().addMapLayer(plan_layer, addToLegend=False)
                group.insertLayer(0, plan_layer)
            self._log(f"Loaded likelihood + recommended plan for '{self.out_file_name}'.", notify=True)
            return

        layer = QgsRasterLayer(self.out_path, self.out_file_name, "gdal")
        if not layer.isValid():
            self._log(f"Output raster is not valid: {self.out_path}", Qgis.MessageLevel.Critical, notify=True)
            return
        layer.setCrs(self.target_crs)
        gis_io.apply_palette(layer)

        # Each band is one yearly step; FixedRangePerBand animates through the bands.
        n = len(self.frames)
        start = datetime.now()
        ranges = {
            band: QgsDateTimeRange(
                start.replace(year=start.year + band - 1),
                start.replace(year=start.year + band),
            )
            for band in range(1, n + 1)
        }
        tprops = layer.temporalProperties()
        tprops.setMode(Qgis.RasterTemporalMode.FixedRangePerBand)
        tprops.setFixedRangePerBand(ranges)
        tprops.setIsActive(True)

        QgsProject.instance().addMapLayer(layer)
        self._setup_temporal_controller(start, n)
        self._log(
            f"Loaded '{self.out_file_name}' with {n} temporal steps — press play in the Temporal Controller.",
            notify=True,
        )

    def _setup_temporal_controller(self, start: datetime, n_steps: int) -> None:
        temporal = self.iface.mapCanvas().temporalController()
        if temporal is None:
            return
        end = start.replace(year=start.year + max(1, n_steps))
        temporal.setTemporalExtents(QgsDateTimeRange(start, end))
        temporal.rewindToStart()
        temporal.setLooping(False)
        temporal.setFrameDuration(QgsInterval(1, 0, 0, 0, 0, 0, 0))
        temporal.setFramesPerSecond(5)
        temporal.setAnimationState(QgsTemporalNavigationObject.AnimationState.Forward)
