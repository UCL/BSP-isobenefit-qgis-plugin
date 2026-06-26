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
        centre_min_settlement=3,
        centre_distance_m=None,
        green_distance_m=None,
    ):
        super().__init__("Isobenefit simulation")
        self.iface = iface
        self.out_file_name = out_file_name
        self.out_path = str(Path(out_dir_path) / f"{out_file_name}.tif")
        self.plan_path = str(Path(out_dir_path) / f"{out_file_name}_plan.tif")  # post-processed
        self.pre_path = str(Path(out_dir_path) / f"{out_file_name}_pre.tif")  # raw CA, pre-processing
        self.existing_path = str(Path(out_dir_path) / f"{out_file_name}_existing.tif")  # pre-simulation fabric
        self.report_path = str(Path(out_dir_path) / f"{out_file_name}_report.txt")  # human-readable run record
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
        self.centre_min_settlement = int(centre_min_settlement)
        self.centre_distance_m = None if centre_distance_m is None else float(centre_distance_m)
        self.green_distance_m = None if green_distance_m is None else float(green_distance_m)
        self.is_ensemble = self.n_ensemble > 1
        # populated during run()
        self.geotransform = None
        self.per_block = None
        self.frames: list[np.ndarray] = []  # one categorical (uint8) frame per step
        self._plan_outputs: list[tuple[str, str]] = []  # (raster path, layer label) for finished()
        self.error_message: str | None = None

    @staticmethod
    def _log(message: str, level=Qgis.MessageLevel.Info, notify: bool = False) -> None:
        QgsMessageLog.logMessage(message, LOG_TAG, level=level, notifyUser=notify)

    def _per_block(self) -> tuple[float, float, float]:
        block = self.granularity_m**2 / 1.0e6
        return tuple(d * block for d in self.density_factors)

    @staticmethod
    def _count_centres(plan) -> int:
        """Number of centre AREAS (connected components) in a plan — new and existing."""
        return len(grid._components((plan == grid.PLAN_CENTRE) | (plan == grid.PLAN_EXIST_CENTRE)))

    def _compose_report(self, report_stats, audit, rows, cols, start_pop, iter_summary, elapsed) -> str:
        """A plain-text record of the run — parameters, run summary, per-plan statistics and the centre
        audit — so there is a durable account of exactly what was done and how each option scored."""
        from datetime import datetime

        dispersal = {0.0: "Off", 0.005: "Low", 0.02: "Medium", 0.05: "High"}.get(
            round(self.cent_prob_isol, 4), f"{self.cent_prob_isol:g}"
        )
        cwalk = self.centre_distance_m or self.max_distance_m
        gwalk = self.green_distance_m or self.max_distance_m
        min_ha = self.centre_min_settlement * self.granularity_m**2 / 1.0e4
        km = self.granularity_m / 1000.0
        dens = f"{min(self.density_factors):,.0f}-{max(self.density_factors):,.0f} /km²"
        lines = [
            "Isobenefit Urbanism — simulation report",
            "=" * 42,
            f"Output:    {self.out_file_name}",
            f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
            f"CRS:       {self.target_crs.authid()}",
            "",
            "PARAMETERS",
            "-" * 10,
            f"  Grid size             : {self.granularity_m:.0f} m",
            f"  Max iterations        : {self.total_iters}",
            f"  Target population     : {self.max_populat:,.0f}",
            f"  Build probability     : {self.build_prob:g}",
            f"  Dispersed development : {dispersal}",
            f"  Centre walk           : {cwalk:.0f} m",
            f"  Green walk            : {gwalk:.0f} m",
            f"  Min green span        : {self.min_green_span:.0f} m",
            f"  Density               : {dens} (existing built {self.exist_built_density:,.0f})",
            f"  Min settlement        : {min_ha:.0f} ha ({self.centre_min_settlement} cells)",
            f"  Optimise centres      : {'on' if self.optimise_centres else 'off'}",
            f"  Ensemble              : {self.n_ensemble} run(s)",
            "",
            "RUN",
            "-" * 3,
            f"  Grid: {cols} × {rows} cells ({cols * km:.1f} × {rows * km:.1f} km)",
            f"  Starting population: {start_pop:,.0f} ({start_pop / self.max_populat:.0%} of target)",
        ]
        if iter_summary:
            lines.append(f"  {iter_summary}")
        lines.append(f"  Elapsed: {elapsed:.0f} s")
        lines += ["", "STATISTICS (per plan — homes within a walk of an amenity)", "-" * 40]
        for label, m, ncent in report_stats:
            lines.append(
                f"  {label}: served {m.get('served_coverage', 0):.0%}, "
                f"centre walk {m.get('centre_access', 0):.0f} m, green walk {m.get('green_access', 0):.0f} m, "
                f"{ncent} centres, {m.get('built_cells', 0):,} built cells"
            )
        if report_stats and "transit_coverage" in report_stats[-1][1]:
            tm = report_stats[-1][1]
            lines.append(
                f"  transit: {tm['transit_coverage']:.0%} within a walk of a stop "
                f"(avg {tm['transit_access']:.0f} m) [reported only]"
            )
        if audit:
            s = audit["summary"]
            lines += [
                "",
                "CENTRE AUDIT (balanced option)",
                "-" * 30,
                f"  {s['n_centres']} centres ({s['n_existing']} existing, {s['n_new']} new); each serves a "
                f"median of {s['served_median']} built cells (min {s['served_min']}, max {s['served_max']}).",
            ]
        lines += ["", "FILES", "-" * 5, f"  {Path(self.out_path).name}  — development likelihood (built, green)"]
        for path, label in self._plan_outputs:
            lines.append(f"  {Path(path).name}  — {label}")
        lines.append(f"  {Path(self.report_path).name}  — this report")
        lines.append("")
        return "\n".join(lines)

    def _log_iterations_to_target(self, isobenefit, state, origin, density, seeds) -> str:
        """Step ONE representative run to the population target and log how many iterations it took,
        so the user sees that typically only ~N steps run before the target of M is met (well under
        the max) — or a clear warning if the cap is hit first. Returns the summary line (for the
        report). The engine COPIES its inputs (the binding takes read-only arrays), so this throwaway
        run does not disturb the ensemble; it is a good proxy because runs with the same parameters
        reach the target at a similar point."""
        sample = isobenefit.Simulation(
            state, origin, density, seeds,
            self.granularity_m, self.max_distance_m, self.max_populat, self.min_green_span,
            self.build_prob, self.cent_prob_nb, self.cent_prob_isol, self.pop_target_cent_threshold,
            self.prob_distribution, self.density_factors, self.exist_built_density,
            self.total_iters, self.random_seed,
        )
        iters = 0
        while sample.current_iter < self.total_iters and sample.pop_target_ratio < 1.0:
            if self.isCanceled():
                return ""
            sample.step()
            iters += 1
        if sample.pop_target_ratio >= 1.0:
            summary = (
                f"A representative run reached the target population of {int(self.max_populat):,} after "
                f"{iters} iterations (of the {self.total_iters} max)."
            )
            self._log(summary + " The other runs stop similarly.", Qgis.MessageLevel.Info)
        else:
            summary = (
                f"A representative run hit the {self.total_iters}-iteration cap at only "
                f"{sample.pop_target_ratio:.0%} of the target population "
                f"({int(sample.population):,} of {int(self.max_populat):,})."
            )
            self._log(summary + " Raise max iterations or build probability.", Qgis.MessageLevel.Warning)
        return summary

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
                # Carve unbuildable land (water, airports, military, quarries) AND the buffered
                # motorway/railway/river barrier corridors from the OSM tool — these cells must
                # never develop. all_touched so thin corridors leave no gaps. Done before the CA.
                state = gis_io.burn_layer(
                    state, self.unbuildable_layer, self.target_crs, geotransform, -1, all_touched=True
                )
                self._log("Carved unbuildable land + barrier corridors (motorways/railways/rivers).")
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
                iter_summary = self._log_iterations_to_target(isobenefit, state, origin, density, seeds)
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
                centre_walk = self.centre_distance_m or self.max_distance_m
                # compactness options the user compares + picks visually (centre spacing in metres;
                # None = consolidated / coverage-minimal). Generated for the ONE chosen run.
                spacings = {"consolidated": None, "balanced": 0.7 * centre_walk, "dispersed": 0.45 * centre_walk}
                plan, metrics, pre_plan, best_state = grid.select_plan(
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
                    centre_spacing_m=(spacings["balanced"] if self.optimise_centres else None),
                    centre_min_settlement=self.centre_min_settlement,
                    centre_distance_m=self.centre_distance_m,
                    green_distance_m=self.green_distance_m,
                )
                self._plan_outputs = []  # (path, label) for finished() to load, in display order
                report_stats = []  # (label, metrics, n_centres) for the run report
                # existing fabric (before any simulation) so the existing -> raw -> options chain is visible
                if (origin == 0).any() or (origin == 1).any():
                    existing_plan = np.full((rows, cols), grid.PLAN_NONE, dtype=np.uint8)
                    existing_plan[origin == 0] = grid.PLAN_GREEN
                    existing_plan[origin == 1] = grid.PLAN_EXIST_BUILT
                    for sy, sx in seeds:
                        if 0 <= sy < rows and 0 <= sx < cols:
                            existing_plan[sy, sx] = grid.PLAN_EXIST_CENTRE
                    gis_io.write_plan_raster(self.existing_path, existing_plan, geotransform, self.target_crs)
                    self._plan_outputs.append((self.existing_path, "existing development"))
                if pre_plan is not None:  # the chosen run BEFORE post-processing (raw CA), for comparison
                    gis_io.write_plan_raster(self.pre_path, pre_plan, geotransform, self.target_crs)
                    self._plan_outputs.append((self.pre_path, "raw plan (pre-processing)"))
                    pre_m = grid.evaluate_plan(
                        pre_plan, self.granularity_m, self.max_distance_m, min_green_span_m=self.min_green_span,
                        router=router, centre_distance_m=self.centre_distance_m, green_distance_m=self.green_distance_m,
                    )
                    report_stats.append(("raw (pre-processing)", pre_m, self._count_centres(pre_plan)))
                if self.optimise_centres and best_state is not None:
                    self._log("Post-processing the chosen run at each compactness option…")
                    variants = grid.plan_variants(
                        best_state, self.granularity_m, self.min_green_span, self.max_distance_m, spacings,
                        mean_density=mean_density, max_density=max(self.density_factors),
                        existing_centres=seeds, existing_built=(origin == 1), existing_green=(origin == 0),
                        centre_anchors=station_anchors, router=router,
                        centre_distance_m=self.centre_distance_m, green_distance_m=self.green_distance_m,
                        centre_min_settlement=self.centre_min_settlement,
                    )
                    for label in ("consolidated", "balanced", "dispersed"):
                        vplan, vm = variants[label]
                        vpath = str(Path(self.out_path).with_name(f"{self.out_file_name}_{label}.tif"))
                        gis_io.write_plan_raster(vpath, vplan, geotransform, self.target_crs)
                        self._plan_outputs.append((vpath, f"{label} centres"))
                        report_stats.append((label, vm, self._count_centres(vplan)))
                        self._log(  # per-option metrics so the choice is informed, not just visual
                            f"  {label}: {vm['served_coverage']:.0%} served, centre walk "
                            f"{vm['centre_access']:.0f} m, green {vm['green_access']:.0f} m"
                        )
                    plan, metrics = variants["balanced"]  # headline metrics + audit use the middle option
                elif plan is not None:  # centre optimisation off -> a single plan (CA centres kept)
                    gis_io.write_plan_raster(self.plan_path, plan, geotransform, self.target_crs)
                    self._plan_outputs.append((self.plan_path, "recommended plan"))
                    if metrics:
                        report_stats.append(("recommended", metrics, self._count_centres(plan)))
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
                audit = None
                if plan is not None:
                    # Per-centre effectiveness audit, by the same distance model — surfaces weak
                    # centres (thin catchment / off-centre) every run, so they're not just eyeballed.
                    audit = grid.audit_centres(plan, self.granularity_m, self.max_distance_m, router=router)
                    s = audit["summary"]
                    self._log(
                        f"Centre audit: {s['n_centres']} centres ({s['n_existing']} existing from input, "
                        f"{s['n_new']} placed by the model) serve a median of {s['served_median']} built cells "
                        f"each (min {s['served_min']}, max {s['served_max']}); "
                        f"median avg-walk {s['mean_dist_median_m']:.0f} m."
                    )
                    weak_new = [c for c in audit["centres"] if not c["existing"]][:5]  # the model's own worst
                    if weak_new:
                        self._log(
                            f"Weakest NEW centres of {s['n_new']} (row, col, served, avg-walk m): "
                            + "; ".join(
                                f"({c['row']},{c['col']},{c['served']},{c['mean_dist_m']:.0f})" for c in weak_new
                            )
                        )
                # durable run record (best-effort — never fail the run over the report)
                try:
                    report = self._compose_report(
                        report_stats, audit, rows, cols, int(sim.population), iter_summary, time.time() - t_zero
                    )
                    with open(self.report_path, "w", encoding="utf-8") as fh:
                        fh.write(report)
                    self._log(f"Wrote run report: {Path(self.report_path).name}")
                except Exception as exc:  # noqa: BLE001 — the report is a nicety, not worth failing for
                    self._log(f"Could not write the run report: {exc}", Qgis.MessageLevel.Warning)
                self._log(
                    f"Ensemble finished in {time.time() - t_zero:.0f}s; wrote likelihood, "
                    f"{len(self._plan_outputs)} plan(s) and a report: {self.out_path}"
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
                    self._log(
                        f"Reached the target population of {int(self.max_populat)} after {i + 1} "
                        f"iterations (of the {self.total_iters} max) — stopping early.",
                        Qgis.MessageLevel.Success,
                    )
                    break
            else:  # loop ran to the cap without reaching the target
                self._log(
                    f"Ran all {self.total_iters} iterations and reached {sim.pop_target_ratio:.0%} of the "
                    f"target population ({int(sim.population)} of {int(self.max_populat)}).",
                    Qgis.MessageLevel.Warning,
                )

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
            # The raw CA plan (pre-processing) and each post-processed compactness option, inserted
            # above the likelihood bands so the difference the post-processing makes is plain to see.
            for path, label in [(self.pre_path, "raw plan (pre-processing)"), *self._plan_outputs]:
                lyr = QgsRasterLayer(path, f"{self.out_file_name} — {label}", "gdal")
                if lyr.isValid():
                    lyr.setCrs(self.target_crs)
                    gis_io.apply_plan_style(lyr)
                    QgsProject.instance().addMapLayer(lyr, addToLegend=False)
                    group.insertLayer(0, lyr)
            self._log(
                f"Loaded likelihood, the raw (pre-processing) plan and {len(self._plan_outputs)} "
                f"post-processed option(s) for '{self.out_file_name}' — compare and pick.",
                notify=True,
            )
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
