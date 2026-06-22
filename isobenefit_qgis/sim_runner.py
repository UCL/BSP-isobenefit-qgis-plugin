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
        transit_stops_layer=None,
        stations_layer=None,
        streets_layer=None,
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
        optimise_centres=True,
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
        self.transit_stops_layer = transit_stops_layer
        self.stations_layer = stations_layer
        self.streets_layer = streets_layer
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
        self.optimise_centres = bool(optimise_centres)
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
                # Centres may be supplied as polygon AREAS (every covered cell becomes a
                # true centre cell) or as point seeds (one cell each).
                if self.centre_seeds_layer.geometryType() == Qgis.GeometryType.Polygon:
                    seeds = gis_io.polygon_cells(self.centre_seeds_layer, self.target_crs, geotransform, rows, cols)
                    self._log(f"Placed {len(seeds)} centre cell(s) from polygon areas.")
                else:
                    seeds = gis_io.point_cells(self.centre_seeds_layer, self.target_crs, geotransform, rows, cols)
                    self._log(f"Placed {len(seeds)} centre seed(s).")

            # Public-transport access: ordinary stops and rail/tram stations are two layers
            # (each edited/swapped on its own). The scored transit dimension uses BOTH — every
            # stop is transit access — while only stations anchor a centre in the plan.
            stop_cells = []
            if self.transit_stops_layer is not None:
                stop_cells = gis_io.point_cells(self.transit_stops_layer, self.target_crs, geotransform, rows, cols)
            station_anchors = []
            if self.stations_layer is not None:
                station_anchors = gis_io.point_cells(self.stations_layer, self.target_crs, geotransform, rows, cols)
            transit_stops = None
            all_stop_cells = stop_cells + station_anchors
            if all_stop_cells:
                transit_stops = np.zeros((rows, cols), dtype=bool)
                for sr, sc in all_stop_cells:
                    transit_stops[sr, sc] = True
                self._log(
                    f"Placed {len(stop_cells)} stop(s) + {len(station_anchors)} station(s)"
                    + ("; stations anchor centres." if station_anchors else ".")
                )

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
                # Collect each run's final layout (not just the blended average): the
                # likelihood layers come from all runs, and the recommended plan is the
                # best single run, optimised. Batched for progress + cancellation.
                states = []
                while len(states) < n:
                    if self.isCanceled():
                        self._log("Simulation cancelled by user.", Qgis.MessageLevel.Warning)
                        return False
                    members = min(batch, n - len(states))
                    states.extend(isobenefit.run_ensemble(sim, self.random_seed + len(states), members))
                    self.setProgress(len(states) / n * 80.0)
                    self._log(f"ensemble: {len(states)}/{n} runs")

                # likelihood (uncertainty) layers from all runs (centres belong to the plan, not here)
                p_built, p_green = grid.class_probabilities(states)
                gis_io.write_probability_bands(
                    self.out_path,
                    [p_built, p_green],
                    ["built likelihood", "green likelihood"],
                    geotransform,
                    self.target_crs,
                )

                # recommended plan = the best single run, optimised. Population-aware
                # green (funded by densification, not lost homes) + facility-location
                # centres; existing centre seeds kept. Picked by shortest average walk.
                self._log("Selecting and optimising the recommended plan…")
                self.setProgress(90.0)
                mean_density = sum(p * d for p, d in zip(self.prob_distribution, self.density_factors))
                # ONE distance model: with a streets layer, walking is measured along the network
                # (built once here, reused for every run); without one, the open-grid walk. No silent
                # fallback — if the graph can't be built, make_router raises and the run fails clearly.
                router = None
                if self.streets_layer is not None:
                    from . import routing

                    router = routing.make_router(
                        self.streets_layer,
                        self.target_crs,
                        geotransform,
                        rows,
                        cols,
                        self.granularity_m,
                        self.max_distance_m,
                    )
                    self._log("Walking distances measured along the street network.")
                plan, metrics = grid.select_plan(
                    states,
                    self.granularity_m,
                    self.min_green_span,
                    self.max_distance_m,
                    mean_density=mean_density,
                    max_density=max(self.density_factors),
                    existing_centres=seeds,
                    # existing development is frozen (never pruned) and tagged distinctly
                    existing_built=(origin == 1),
                    existing_green=(origin == 0),
                    optimise_centres=self.optimise_centres,
                    transit_stops=transit_stops,
                    centre_anchors=station_anchors,
                    router=router,
                )
                if plan is not None:
                    gis_io.write_plan_raster(self.plan_path, plan, geotransform, self.target_crs)
                if metrics:
                    self._log(
                        f"Recommended plan: {metrics['served_coverage']:.0%} of homes within a walk of both "
                        f"green and a centre (avg walk to a centre {metrics['centre_access']:.0f} m, "
                        f"to green {metrics['green_access']:.0f} m)."
                    )
                    if "transit_coverage" in metrics:
                        self._log(
                            f"Transit: {metrics['transit_coverage']:.0%} of homes within a walk of a "
                            f"public-transport stop (avg walk {metrics['transit_access']:.0f} m)."
                        )
                self._log(
                    f"Ensemble finished in {time.time() - t_zero:.0f}s; "
                    f"wrote likelihood + recommended plan: {self.out_path}"
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
            for band, label in [(1, "built"), (2, "green")]:
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
