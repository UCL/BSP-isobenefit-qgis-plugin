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
    QgsFeatureRequest,
    QgsInterval,
    QgsMessageLog,
    QgsProject,
    QgsRasterLayer,
    QgsTask,
    QgsTemporalNavigationObject,
)

from . import gis_io, grid, report

LOG_TAG = "Isobenefit"


def _snapshot(layer):
    """Detached in-memory copy of a vector layer, taken on the MAIN thread.

    The task's ``run()`` executes on a worker thread, and reading a live project
    layer there is unsafe: the user can edit, reproject or remove it while the
    run is in flight, and QgsVectorLayer is not thread-safe. Each input layer is
    copied up front in the constructor; the copies belong to the task alone.
    """
    if layer is None:
        return None
    return layer.materialize(QgsFeatureRequest())


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
        steep_layer=None,
        slope_max_deg=None,
        walkable_layer=None,
        centre_seeds_layer,
        transit_stops_layer=None,
        stations_layer=None,
        total_iters,
        granularity_m,
        max_distance_m,
        max_populat,
        min_green_span,
        build_prob,
        allow_detached,
        prob_distribution,
        density_factors,
        random_seed,
        n_ensemble=1,
        optimise_centres=True,
        centre_min_settlement=3,
        centre_m2_per_person=grid.CENTRE_M2_PER_PERSON,
        centre_distance_m=None,
        green_distance_m=None,
        min_park_area_m2=None,
        corridor_weight=0.0,
        stop_catchment_m=400.0,
        hub_catchment_m=1200.0,
    ):
        super().__init__("Isobenefit simulation")
        self.iface = iface
        self.out_file_name = out_file_name
        self.out_path = str(Path(out_dir_path) / f"{out_file_name}.tif")
        self.pre_path = str(Path(out_dir_path) / f"{out_file_name}_pre.tif")  # raw CA, pre-processing
        self.existing_path = str(Path(out_dir_path) / f"{out_file_name}_existing.tif")  # pre-simulation fabric
        self.report_path = str(Path(out_dir_path) / f"{out_file_name}_report.txt")  # human-readable run record
        # the output layers group under "<folder> - <run>", so successive runs (scenarios,
        # scenarios_2, ...) stay organised by their folder in the layers pane
        self.group_name = f"{Path(out_dir_path).name} \u2014 {out_file_name}"
        self.target_crs = target_crs
        # snapshot every input layer while still on the main thread (see _snapshot)
        self.extents_layer = _snapshot(extents_layer)
        self.built_layer = _snapshot(built_layer)
        self.green_layer = _snapshot(green_layer)
        self.unbuildable_layer = _snapshot(unbuildable_layer)
        self.steep_layer = _snapshot(steep_layer)
        self.walkable_layer = _snapshot(walkable_layer)
        self.slope_max_deg = None if slope_max_deg is None else float(slope_max_deg)
        self.centre_seeds_layer = _snapshot(centre_seeds_layer)
        self.transit_stops_layer = _snapshot(transit_stops_layer)
        self.stations_layer = _snapshot(stations_layer)
        self.total_iters = int(total_iters)
        self.granularity_m = float(granularity_m)
        self.max_distance_m = float(max_distance_m)
        self.max_populat = float(max_populat)
        self.min_green_span = float(min_green_span)
        self.build_prob = float(build_prob)
        self.allow_detached = bool(allow_detached)
        self.prob_distribution = tuple(float(p) for p in prob_distribution)
        self.density_factors = tuple(float(d) for d in density_factors)
        self.random_seed = int(random_seed)
        self.n_ensemble = int(n_ensemble)
        self.optimise_centres = bool(optimise_centres)
        self.centre_min_settlement = int(centre_min_settlement)
        self.centre_m2_per_person = float(centre_m2_per_person)
        self.centre_distance_m = None if centre_distance_m is None else float(centre_distance_m)
        self.green_distance_m = None if green_distance_m is None else float(green_distance_m)
        self.min_park_area_m2 = None if min_park_area_m2 is None else float(min_park_area_m2)
        self.corridor_weight = float(corridor_weight)
        self.stop_catchment_m = float(stop_catchment_m)
        self.hub_catchment_m = float(hub_catchment_m)
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

    def _mean_new_density_km2(self) -> float:
        """Probability-weighted mean density of new development (people/km²). This is the expected
        density of a new block over the three tiers, so it is what the population accounting, the
        per-person metrics and the centre provision size against."""
        return float(sum(p * d for p, d in zip(self.prob_distribution, self.density_factors)))

    def _centre_quota(self) -> float:
        """The population a centre must gather before the next one is earned: the service
        viability threshold itself, so growth creates the centres post-processing keeps."""
        return float(
            self.centre_min_settlement * self._mean_new_density_km2() * self.granularity_m**2 / 1.0e6
        )

    def _write_tiered_plan(self, path: str, plan, label: str) -> np.ndarray:
        """Arrange the drawn density tiers by walking distance to the final mixed-use centres, then
        write the plan as one categorical raster in which each new cell takes its tier's colour
        (built and centres in distinct hues). Registers the raster for finished() to load and
        returns the tiered plan, so the report can break the achieved densities down per tier."""
        dens = grid.derive_density(
            plan, self.granularity_m, self.centre_distance_m or self.max_distance_m,
            self.density_factors, self.prob_distribution,
        )
        disp = grid.to_tiered_plan(plan, dens, self.density_factors)
        gis_io.write_plan_raster(path, disp, self.geotransform, self.target_crs)
        self._plan_outputs.append((path, label))
        return disp

    def _report_option(self, label: str, short: str, metrics: dict, n_centres: int, tiered) -> dict:
        """One plan option's report entry: label, short column header, metrics, centre count and
        (when a tiered raster was written) the per-tier density breakdown."""
        tiers = None
        if tiered is not None:
            tiers = report.tier_breakdown(tiered, self.granularity_m, self.density_factors)
        return {"label": label, "short": short, "metrics": metrics, "n_centres": n_centres, "tiers": tiers}

    @staticmethod
    def _count_centres(plan) -> int:
        """Number of centre AREAS (connected components) in a plan — new and existing."""
        return len(grid._components((plan == grid.PLAN_CENTRE) | (plan == grid.PLAN_EXIST_CENTRE)))

    def _report_header_lines(self) -> list[str]:
        return [
            "Isobenefit Urbanism — simulation report",
            "=" * 42,
            f"Output:    {self.out_file_name}",
            f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
            f"CRS:       {self.target_crs.authid()}",
        ]

    def _centre_walk(self) -> float:
        """The centre walk in metres, falling back to the single max distance."""
        return self.centre_distance_m or self.max_distance_m

    def _green_walk(self) -> float:
        """The green walk in metres, falling back to the single max distance."""
        return self.green_distance_m or self.max_distance_m

    def _report_param_lines(self) -> list[str]:
        dispersal = "Detached settlements allowed" if self.allow_detached else "Attached growth only"
        cwalk = self._centre_walk()
        gwalk = self._green_walk()
        min_pop = self.centre_min_settlement * self._mean_new_density_km2() * self.granularity_m**2 / 1.0e6
        hi, md, lo = self.density_factors
        ph, pm, pl = self.prob_distribution
        dens = (
            f"high {hi:,.0f} ({ph:.0%}), med {md:,.0f} ({pm:.0%}), low {lo:,.0f} ({pl:.0%}) /km²; "
            f"mean {self._mean_new_density_km2():,.0f}"
        )
        return [
            f"  Grid size             : {self.granularity_m:.0f} m",
            f"  Max iterations        : {self.total_iters}",
            f"  Target population     : {self.max_populat:,.0f}",
            f"  Dispersed development : {dispersal}",
            f"  Centre walk           : {cwalk:.0f} m",
            f"  Green walk            : {gwalk:.0f} m",
            f"  Stop catchment        : {self.stop_catchment_m:.0f} m",
            f"  Hub catchment         : {self.hub_catchment_m:.0f} m",
            f"  Corridor preference   : {self.corridor_weight:g}",
            f"  Min green span        : {self.min_green_span:.0f} m",
            f"  Min park area         : {(self.min_park_area_m2 or grid.DEFAULT_MIN_PARK_AREA_M2) / 1.0e4:.0f} ha",
            f"  Density               : {dens}",
            f"  Service viability     : ~{min_pop:,.0f} people ({self.centre_min_settlement} cells)",
            f"  Optimise centres      : {'on' if self.optimise_centres else 'off'}",
            f"  Ensemble              : {self.n_ensemble} run(s)",
        ]

    def _report_file_lines(self, first: tuple[str, str] | None = None) -> list[str]:
        entries = ([first] if first else []) + [(Path(p).name, label) for p, label in self._plan_outputs]
        entries.append((Path(self.report_path).name, "this report"))
        width = max(len(name) for name, _ in entries)
        return [f"  {name.ljust(width)}  {label}" for name, label in entries]

    def _compose_report(self, options, audit, rows, cols, start_pop, iter_summary, elapsed) -> str:
        """A plain-text record of the run — parameters, run summary, the plan options side by side,
        the achieved density mix and the centre audit — so there is a durable, comprehensive account
        of exactly what was done and how each option scored."""
        km = self.granularity_m / 1000.0
        run_lines = [
            f"  Grid: {cols} × {rows} cells ({cols * km:.1f} × {rows * km:.1f} km)",
            f"  Starting population: {start_pop:,.0f} ({start_pop / self.max_populat:.0%} of target)",
        ]
        if iter_summary:
            run_lines.append(f"  {iter_summary}")
        run_lines.append(f"  Elapsed: {elapsed:.0f} s")
        return report.compose_report(
            self._report_header_lines(),
            self._report_param_lines(),
            run_lines,
            options,
            self.max_populat,
            self.density_factors,
            self.prob_distribution,
            audit,
            self._report_file_lines(first=(Path(self.out_path).name, "development likelihood (built, green)")),
        )

    def _selection_progress(self, done: int, total: int) -> bool:
        """Progress + cancellation for the post-processing selection: the stage occupies the
        40–99% band of the task bar (it is where the time goes), logs every candidate, and returning False aborts
        ``select_plan`` mid-stage."""
        self.setProgress(40.0 + 59.0 * done / max(1, total))
        self._log(f"post-processing candidates: {done}/{total}")
        return not self.isCanceled()

    def _log_iterations_to_target(
        self, isobenefit, state, origin, density, seeds, sterile=None, transit_catchment=None,
        provision_seeds=None,
    ) -> str:
        """Step ONE representative run to the population target and log how many iterations it took,
        so the user sees that typically only ~N steps run before the target of M is met (well under
        the max) — or a clear warning if the cap is hit first. Returns the summary line (for the
        report). The engine COPIES its inputs (the binding takes read-only arrays), so this throwaway
        run does not disturb the ensemble; it is a good proxy because runs with the same parameters
        reach the target at a similar point."""
        sample = isobenefit.Simulation(
            state, origin, density, seeds,
            self.granularity_m, self._centre_walk(), self._green_walk(),
            self.max_populat, self.min_green_span,
            self.build_prob, self._centre_quota(), self.allow_detached,
            self.prob_distribution, self.density_factors,
            self.total_iters, self.random_seed,
            min_park_area_m2=self.min_park_area_m2, sterile=sterile,
            transit_catchment=transit_catchment, corridor_weight=self.corridor_weight,
            provision_seeds=provision_seeds,
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
            self._log(
                summary + " Raise max iterations or the build probability. Note the target counts"
                " new residents only, and gaps narrower than the min green span never fill;"
                " the plugin guide's troubleshooting section lists the common causes.",
                Qgis.MessageLevel.Warning,
            )
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
            # the extents polygon is the study area, and the layers that follow are not clipped
            # to it: a built or green feature reaching past the boundary would otherwise put
            # fabric, and developable land, outside the area the planner drew
            inside = state == 0
            origin = np.full((rows, cols), -1, dtype=np.int16)
            if self.built_layer is not None:
                state = gis_io.burn_layer(state, self.built_layer, self.target_crs, geotransform, 1)
                origin = gis_io.burn_layer(origin, self.built_layer, self.target_crs, geotransform, 1)
                self._log("Burned existing built areas.")
            if self.green_layer is not None:
                state = gis_io.burn_layer(state, self.green_layer, self.target_crs, geotransform, 0)
                origin = gis_io.burn_layer(origin, self.green_layer, self.target_crs, geotransform, 0)
                self._log("Burned existing green areas.")
            # everything the two burns placed beyond the boundary goes back to nothing
            state = np.where(inside, state, -1)
            origin = np.where(inside, origin, -1)
            if self.unbuildable_layer is not None:
                # Carve unbuildable land (water, airports, military, quarries) AND the buffered
                # motorway/railway/river barrier corridors from the OSM tool — these cells must
                # never develop. Burned by cell centre, like every other layer: burning every
                # touched cell carved up to a cell beyond each edge, which cost real developable
                # land (at Cambourne it doubled the carved area). Walks cannot cut a diagonal
                # corner between two carved cells, so a corridor still blocks them.
                carved = gis_io.burn_layer(
                    np.full_like(state, 0), self.unbuildable_layer, self.target_crs, geotransform, -1
                )
                # existing fabric is fixed: a corridor clipping a mapped building does not
                # turn that building into unbuildable land
                state = np.where((carved == -1) & (origin != 1), -1, state)
                self._log("Carved unbuildable land + barrier corridors (motorways/railways/rivers).")
            if self.slope_max_deg is not None and self.steep_layer is None:
                # the limit alone does nothing: it selects bands from a layer that was not given
                self._log(
                    f"A slope limit of {self.slope_max_deg:g} degrees was set but no steep-slopes "
                    "layer was selected, so no ground is excluded for slope. Choose the scenario's "
                    "steep.geojson in the Steep slopes box.",
                    level=Qgis.MessageLevel.Warning,
                    notify=True,
                )
            if self.steep_layer is not None and self.slope_max_deg is not None:
                # Slope bands are supplied as a separate layer so they can be edited from local
                # knowledge; the bands at or above the limit preclude development. Existing
                # fabric stands whatever the ground does under it.
                limit = self.slope_max_deg

                def _at_or_above(feat, limit=limit):
                    idx = feat.fields().indexOf("min_slope_deg")
                    band = feat.attribute(idx) if idx >= 0 else None
                    try:
                        return band is not None and float(band) >= limit
                    except (TypeError, ValueError):
                        return False

                steep = gis_io.burn_layer(
                    np.full_like(state, 0), self.steep_layer, self.target_crs, geotransform, -1,
                    feature_filter=_at_or_above,
                )
                n_steep = int(((steep == -1) & (origin != 1) & (state != -1)).sum())
                state = np.where((steep == -1) & (origin != 1), -1, state)
                self._log(
                    f"Carved ground at or above {self.slope_max_deg:g} degrees: {n_steep:,} cell(s)."
                )
            if self.walkable_layer is not None:
                # Applied last, over the carves: where a way people can walk crosses a barrier,
                # the barrier is walkable at that cell. It stays unbuildable — a footbridge does
                # not make a motorway a building site — so only the walk changes.
                crossed = gis_io.burn_layer(
                    np.full_like(state, 0), self.walkable_layer, self.target_crs, geotransform, 1,
                    all_touched=True,
                )
                mask = (crossed == 1) & (state == -1)
                state = np.where(mask, grid.STATE_CROSSING, state)
                self._log(
                    f"Marked {int(mask.sum()):,} barrier cell(s) as walkable crossings from the "
                    "walkable-ways layer."
                )
            density = np.zeros((rows, cols), dtype=np.float32)
            seeds = []
            if self.centre_seeds_layer is not None:
                # Centres may be supplied as polygon AREAS (every covered cell becomes a
                # true centre cell) or as point seeds (one cell each).
                if self.centre_seeds_layer.geometryType() == Qgis.GeometryType.Polygon:
                    seeds = gis_io.polygon_cells(self.centre_seeds_layer, self.target_crs, geotransform, rows, cols)
                    # a centre stands in the fabric it serves: a polygon reaching over open
                    # land or water would otherwise plant centres on ground nobody lives on
                    seeds = [(r, c) for r, c in seeds if origin[r, c] == 1]
                    self._log(f"Placed {len(seeds)} centre cell(s) from polygon areas.")
                else:
                    seeds = gis_io.point_cells(self.centre_seeds_layer, self.target_crs, geotransform, rows, cols)
                    self._log(f"Placed {len(seeds)} centre seed(s).")
                # rasterisation can strand a seed on a carved corridor or water cell, which the
                # core rejects; snap those to the nearest buildable cell within two blocks
                seeds, n_snapped, n_dropped = grid.sanitise_seeds(
                    seeds, state, self.granularity_m, 2 * self.granularity_m
                )
                if n_snapped or n_dropped:
                    self._log(
                        f"Seeds on unbuildable land: {n_snapped} snapped to the nearest buildable "
                        f"cell, {n_dropped} dropped (no buildable cell within two blocks)."
                    )

            # Transit inputs: two layers with two roles. Corridors (bus stops or drawn
            # corridor lines) are growth anchors — their walkable catchment attracts
            # development through the corridor preference — while hubs (stations) anchor a
            # pinned centre. Point features are snapped off carved cells: a station sits ON
            # its railway and a stop on its road, and those corridors are carved as
            # unbuildable, so the raw cell is often one a walk can never reach (which would
            # silently void the anchor and the reported access).
            stop_cells = []
            if self.transit_stops_layer is not None:
                if self.transit_stops_layer.geometryType() == Qgis.GeometryType.Point:
                    stop_cells = gis_io.point_cells(
                        self.transit_stops_layer, self.target_crs, geotransform, rows, cols
                    )
                    stop_cells, _, n_lost = grid.sanitise_seeds(stop_cells, state, self.granularity_m,
                                                                2 * self.granularity_m)
                    if n_lost:
                        self._log(f"{n_lost} stop(s) dropped: no walkable cell within two blocks.")
                else:
                    # a drawn corridor (line or polygon): burn every touched cell, keep the
                    # walkable ones — a dense trace needs no per-cell snapping
                    corridor = np.zeros((rows, cols), dtype=np.int16)
                    corridor = gis_io.burn_layer(
                        corridor, self.transit_stops_layer, self.target_crs, geotransform, 1, all_touched=True
                    )
                    stop_cells = [(int(y), int(x)) for y, x in np.argwhere((corridor == 1) & (state != -1))]
                    self._log(f"Rasterised the corridor layer to {len(stop_cells)} walkable cell(s).")
            station_anchors = []
            if self.stations_layer is not None:
                station_anchors = gis_io.point_cells(self.stations_layer, self.target_crs, geotransform, rows, cols)
                station_anchors, _, n_lost = grid.sanitise_seeds(station_anchors, state, self.granularity_m,
                                                                 2 * self.granularity_m)
                if n_lost:
                    self._log(
                        f"{n_lost} station(s) dropped: no walkable cell within two blocks.",
                        Qgis.MessageLevel.Warning,
                    )
            # Land that can host no viable settlement is set aside before the run. This has to
            # follow the transit block: a hub anchors a centre, so a pocket within its walk is
            # servable and must not be ruled out.
            n_pocket = grid.green_unviable_pockets(
                state, origin, self.centre_min_settlement,
                existing_centres=list(seeds) + [a for a in station_anchors if a not in set(seeds)],
                granularity_m=self.granularity_m,
                centre_distance_m=self._centre_walk(),
            )
            if n_pocket:
                self._log(
                    f"Marked {n_pocket} cell(s) of unserviceable pockets as protected green: "
                    "land unable to hold a viable settlement and beyond the centre walk of "
                    "any existing centre or transit hub."
                )

            def _mask(cells):
                if not cells:
                    return None
                m = np.zeros((rows, cols), dtype=bool)
                for sr, sc in cells:
                    m[sr, sc] = True
                return m

            transit_stops = _mask(stop_cells)
            transit_hubs = _mask(station_anchors)
            if transit_stops is not None or transit_hubs is not None:
                self._log(
                    f"Placed {len(stop_cells)} corridor cell(s) + {len(station_anchors)} hub(s)"
                    + ("; hubs anchor centres." if station_anchors else ".")
                )
            # The corridor preference field: cells within a transit catchment draw at full
            # probability during growth; the rest are scaled by (1 - corridor_weight).
            # Corridors and hubs each project their own catchment (the stop catchment and
            # the wider hub catchment), and the field is their union. With the weight at 0
            # (the default) nothing changes and the masks are skipped entirely.
            transit_catchment = None
            if (transit_stops is not None or transit_hubs is not None) and self.corridor_weight > 0.0:
                transit_catchment = np.zeros((rows, cols), dtype=bool)
                for mask, reach in ((transit_stops, self.stop_catchment_m),
                                    (transit_hubs, self.hub_catchment_m)):
                    if mask is not None:
                        d = grid._walk_distance(mask, self.granularity_m, reach, blocked=(state == -1))
                        transit_catchment |= np.isfinite(d)
                cell_pop = self._mean_new_density_km2() * self.granularity_m**2 / 1.0e6
                capacity = float(((state == 0) & transit_catchment).sum()) * cell_pop
                self._log(
                    f"Corridor preference {self.corridor_weight:.2f}: growth favours the "
                    f"{int(transit_catchment.sum()):,}-cell catchment within "
                    f"{self.stop_catchment_m:.0f} m of a corridor cell or "
                    f"{self.hub_catchment_m:.0f} m of a hub."
                )
                if capacity < self.max_populat:
                    self._log(
                        f"The catchment's developable land holds about {capacity:,.0f} people at the mean "
                        f"density, under the target of {self.max_populat:,.0f}. Growth beyond the corridor "
                        f"is throttled by the preference"
                        + (
                            " and fully blocked at weight 1.0, so the run cannot reach its target."
                            if self.corridor_weight >= 1.0
                            else ", so the run will take more iterations to reach the target."
                        ),
                        Qgis.MessageLevel.Warning,
                    )
            # Stations join the CA centre seeds: a station triggers development around itself by
            # default, exactly like an existing centre, and the post-processing anchor then pins,
            # grows and protects the centre it earns. The stations stay OUT of ``seeds`` proper so
            # the plan does not mislabel their new centres as existing fabric.
            sim_seeds = seeds + [s for s in station_anchors if s not in set(seeds)]
            if len(sim_seeds) > len(seeds):
                self._log(f"{len(sim_seeds) - len(seeds)} station(s) added as centre seeds for growth.")

            self.per_block = self._per_block()
            # feasibility of the viability threshold at this walk and density: a centre's
            # catchment cannot exceed the walk-ball, so state the arithmetic up front
            ball = grid.walk_ball_cells(self.granularity_m, self._centre_walk())
            if self.centre_min_settlement > ball:
                self._log(
                    f"Service viability is infeasible at these settings: the threshold needs "
                    f"{self.centre_min_settlement} cells of new homes within the centre walk, but the "
                    f"walk-ball holds only {ball} cells. Every new centre will fail viability; "
                    "raise the centre walk, the densities, or lower the threshold.",
                    Qgis.MessageLevel.Warning,
                )
            elif self.centre_min_settlement > ball // 2:
                self._log(
                    f"Service viability is tight at these settings: the threshold needs "
                    f"{self.centre_min_settlement} of the {ball} cells in the centre walk-ball to be "
                    "new homes, more than half the ball built solid.",
                    Qgis.MessageLevel.Warning,
                )
            # existing settlements without a centre seed (outlying farmsteads and hamlets)
            # are sterile: new growth never nucleates against them, so every settlement a
            # run grows carries a centre
            sterile = grid.sterile_fabric(origin == 1, sim_seeds)
            if not sim_seeds and not self.allow_detached:
                # nothing can nucleate: growth attaches only to fabric that carries a centre,
                # and without a centre layer every existing settlement is sterile
                self._log(
                    "No centres were supplied and detached settlements are switched off, so "
                    "growth has nothing to start from. Supply an urban-centres layer, or allow "
                    "detached settlements.",
                    level=Qgis.MessageLevel.Warning,
                    notify=True,
                )
            if sterile.any():
                self._log(
                    f"{int(sterile.sum())} existing cells sit in settlements without a centre; "
                    "they are kept as context and never anchor new growth."
                )
            sim = isobenefit.Simulation(
                state,
                origin,
                density,
                sim_seeds,
                self.granularity_m,
                self._centre_walk(),
                self._green_walk(),
                self.max_populat,
                self.min_green_span,
                self.build_prob,
                self._centre_quota(),
                self.allow_detached,
                self.prob_distribution,
                self.density_factors,
                self.total_iters,
                self.random_seed,
                min_park_area_m2=self.min_park_area_m2,
                sterile=sterile,
                transit_catchment=transit_catchment,
                corridor_weight=self.corridor_weight,
                provision_seeds=station_anchors,
            )
            # Only NEW development is counted, so a run always starts from zero population and grows
            # toward the new-only target; existing fabric is context and never contributes here.
            self._log(f"Growing toward a target of {self.max_populat:.0f} new residents.")

            if self.is_ensemble:
                cores = os.cpu_count() or 4
                n = self.n_ensemble
                batch = max(1, cores)  # ~one run per core keeps all cores busy
                self._log(f"Running an ensemble of {n} simulations across {cores} cores…")
                iter_summary = self._log_iterations_to_target(
                    isobenefit, state, origin, density, sim_seeds, sterile=sterile,
                    provision_seeds=station_anchors,
                    transit_catchment=transit_catchment,
                )
                # Collect each run's final layout (not just the blended average): the
                # likelihood layers come from all runs, and the idealised scenario is the
                # best single run, optimised. Batched for progress + cancellation; one fixed
                # base seed with member_offset continuing the global member index, so the
                # seed sequence is identical whatever the batch size (i.e. the core count) —
                # the same random_seed reproduces the same ensemble on any machine.
                states = []
                while len(states) < n:
                    if self.isCanceled():
                        self._log("Simulation cancelled by user.", Qgis.MessageLevel.Warning)
                        return False
                    members = min(batch, n - len(states))
                    states.extend(
                        isobenefit.run_ensemble(sim, self.random_seed, members, member_offset=len(states))
                    )
                    self.setProgress(len(states) / n * 40.0)
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

                # idealised scenario = the best single run, optimised. Population-aware
                # green (funded by densification, not lost homes) + facility-location
                # centres; existing centre seeds kept. Picked by shortest average walk.
                self._log("Selecting and refining the idealised scenario…")
                self.setProgress(40.0)
                # ONE distance model: the bounded grid walk, the same metric the growth rules
                # use. Street-network routing was removed: new development's streets do not
                # exist yet, so a network metric compares new and existing fabric on
                # different terms and punishes exactly the thing being designed.
                # THREE centre options that share the SAME built fabric, arranged density and hard
                # walk constraint, differing only in the centres: kept as grown, optimally placed
                # (same centres, walked into position, plus any the provision rule requires), and
                # the fewest centres full coverage permits. The user compares and picks, against
                # the raw plan saved separately. The headline/selection mode is "placed".
                headline_mode = "placed" if self.optimise_centres else "grown"
                plan, metrics, pre_plan, best_state = grid.select_plan(
                    states,
                    self.granularity_m,
                    self.max_distance_m,
                    existing_centres=seeds,
                    # existing development is frozen (never pruned) and tagged distinctly
                    existing_built=(origin == 1),
                    existing_green=(origin == 0),
                    centre_mode=headline_mode,
                    transit_stops=transit_stops,
                    transit_hubs=transit_hubs,
                    centre_anchors=station_anchors,
                    target_population=self.max_populat,
                    min_park_area_m2=self.min_park_area_m2,
                    stop_catchment_m=self.stop_catchment_m,
                    hub_catchment_m=self.hub_catchment_m,
                    centre_min_settlement=self.centre_min_settlement,
                    centre_m2_per_person=self.centre_m2_per_person,
                    new_density_km2=self._mean_new_density_km2(),
                    centre_distance_m=self.centre_distance_m,
                    green_distance_m=self.green_distance_m,
                    progress=self._selection_progress,
                )
                if self.isCanceled():
                    return False
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
                    existing_plan[state == -1] = grid.PLAN_NONE  # unbuildable stays empty, never green
                    gis_io.write_plan_raster(self.existing_path, existing_plan, geotransform, self.target_crs)
                    self._plan_outputs.append((self.existing_path, "existing development"))
                if pre_plan is not None:  # the chosen run BEFORE post-processing — saved for comparison
                    # TRULY raw: coloured by the densities the run actually DREW, exactly where it
                    # drew them — no post-processing arrangement. The ensemble keeps only each
                    # member's state, so the winning member is re-run at its own seed
                    # (deterministic) to recover its drawn per-block density grid.
                    best_idx = next((i for i, s in enumerate(states) if s is best_state), None)
                    if best_idx is None:
                        best_idx = next(i for i, s in enumerate(states) if np.array_equal(s, best_state))
                    member = isobenefit.run_member(sim, self.random_seed, best_idx)
                    drawn_km2 = np.asarray(member["density"], dtype=np.float32) / (
                        self.granularity_m**2 / 1.0e6
                    )
                    pre_tiered = grid.to_tiered_plan(pre_plan, drawn_km2, self.density_factors)
                    gis_io.write_plan_raster(self.pre_path, pre_tiered, geotransform, self.target_crs)
                    self._plan_outputs.append((self.pre_path, "raw (before post-processing)"))
                    pre_m = grid.evaluate_plan(
                        pre_plan, self.granularity_m, self.max_distance_m,
                        centre_distance_m=self.centre_distance_m, green_distance_m=self.green_distance_m,
                        new_density_km2=self._mean_new_density_km2(), existing_green=(origin == 0),
                        min_park_area_m2=self.min_park_area_m2,
                    )
                    report_stats.append(self._report_option(
                        "raw (before post-processing)", "raw", pre_m, self._count_centres(pre_plan), pre_tiered
                    ))
                if best_state is not None:
                    self._log("Post-processing the chosen run at each centre option…")
                    # FOUR outputs: the raw above (untouched, drawn densities in place), then the
                    # processed options — same fabric, same arranged density, same hard walk
                    # constraint, different centres. With centre optimisation off only the
                    # as-grown option is produced.
                    mode_keys = ("grown", "placed", "minimal") if self.optimise_centres else ("grown",)
                    variants = grid.plan_variants(
                        best_state, self.granularity_m, self.max_distance_m,
                        {key: key for key in mode_keys},
                        existing_centres=seeds, existing_built=(origin == 1), existing_green=(origin == 0),
                        centre_anchors=station_anchors,
                        centre_distance_m=self.centre_distance_m, green_distance_m=self.green_distance_m,
                        centre_min_settlement=self.centre_min_settlement,
                        centre_m2_per_person=self.centre_m2_per_person,
                        new_density_km2=self._mean_new_density_km2(),
                        min_park_area_m2=self.min_park_area_m2,
                    )
                    labels = {
                        "grown": "centres as grown",
                        "placed": "optimised placement",
                        "minimal": "fewest centres",
                    }
                    shorts = {"grown": "grown", "placed": "placed", "minimal": "fewest"}
                    files = {"grown": "grown", "placed": "placed", "minimal": "fewest"}
                    for key in mode_keys:
                        vplan, vm = variants[key]
                        ncent = self._count_centres(vplan)
                        vpath = str(Path(self.out_path).with_name(f"{self.out_file_name}_{files[key]}.tif"))
                        # put the centre COUNT in the layer name so the difference between the options is
                        # obvious in the QGIS layer panel itself, not only by eyeballing the map. The
                        # density tiers are arranged onto this plan and coloured per tier (built vs centre).
                        vtiered = self._write_tiered_plan(vpath, vplan, f"{labels[key]} ({ncent} centres)")
                        report_stats.append(self._report_option(labels[key], shorts[key], vm, ncent, vtiered))
                        self._log(  # per-option metrics so the choice is informed, not just visual
                            f"  {labels[key]}: {ncent} centres, {vm.get('served_coverage', 0):.0%} served, "
                            f"centre walk {vm.get('centre_access', 0):.0f} m, green {vm.get('green_access', 0):.0f} m, "
                            f"{vm.get('centre_m2_per_person', 0):.0f} m² centre / person"
                        )
                    # headline metrics + audit use the same mode that drove run selection
                    plan, metrics = variants[headline_mode]
                    if pre_plan is not None:  # surface the gentle cleanup (raw is kept un-cleaned to compare)
                        removed = pre_m["built_cells"] - metrics["built_cells"]
                        min_pop = (
                            self.centre_min_settlement * self._mean_new_density_km2() * self.granularity_m**2 / 1.0e6
                        )
                        self._log(
                            f"Building cleanup reverted {removed:,} built cell(s) to green (growth "
                            f"beyond the walk of every centre able to reach ~{min_pop:,.0f} people of "
                            f"demand); the raw plan is kept un-cleaned so you can see exactly what "
                            f"the cleanup changed."
                        )
                        # the rejected-development diagnostic: the raw-vs-plan difference as its
                        # own layer, coded by reason, so a user can see WHY the plans differ from
                        # the run rather than inferring it from two rasters
                        rejected, rcounts = grid.rejected_development(pre_plan, plan)
                        if rejected.any():
                            cell_pop = self._mean_new_density_km2() * self.granularity_m**2 / 1.0e6
                            rpath = str(Path(self.out_path).with_name(f"{self.out_file_name}_rejected.tif"))
                            gis_io.write_plan_raster(rpath, rejected, geotransform, self.target_crs)
                            self._plan_outputs.append((rpath, "rejected development (diagnostic)"))
                            self._log(
                                f"Rejected development: {rcounts['cells']:,} cell(s) "
                                f"(~{rcounts['cells'] * cell_pop:,.0f} people) had no viable centre "
                                "within the walk; the diagnostic layer maps them."
                            )
                if metrics:
                    self._log(
                        f"Idealised scenario: {metrics.get('served_coverage', 0):.0%} of homes within a walk of "
                        f"both green and a centre (avg walk to a centre {metrics.get('centre_access', 0):.0f} m, "
                        f"to green {metrics.get('green_access', 0):.0f} m)."
                    )
                    if "transit_coverage" in metrics:
                        self._log(
                            f"Transit: {metrics['transit_coverage']:.0%} of new homes within "
                            f"{self.stop_catchment_m:.0f} m of a corridor cell or "
                            f"{self.hub_catchment_m:.0f} m of a hub "
                            f"(avg walk to a transit anchor {metrics['transit_access']:.0f} m)."
                        )
                audit = None
                if plan is not None:
                    # Per-centre effectiveness audit, by the same distance model — surfaces weak
                    # centres (thin catchment / off-centre) every run, so they're not just eyeballed.
                    audit = grid.audit_centres(
                        plan, self.granularity_m, self.max_distance_m,
                        centre_min_settlement=self.centre_min_settlement,
                    )
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
                    report_text = self._compose_report(
                        report_stats, audit, rows, cols, int(sim.population), iter_summary, time.time() - t_zero
                    )
                    with open(self.report_path, "w", encoding="utf-8") as fh:
                        fh.write(report_text)
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
            # durable run record for single-run mode too (best-effort, never fail the run over it)
            try:
                km = self.granularity_m / 1000.0
                run_lines = [
                    f"  Grid: {cols} × {rows} cells ({cols * km:.1f} × {rows * km:.1f} km)",
                    f"  Mode: single run, one categorical band per growth step ({len(self.frames)} steps)",
                    f"  Population accommodated: {int(sim.population):,} of {self.max_populat:,.0f} target "
                    f"({sim.pop_target_ratio:.0%})",
                    f"  Iterations: {sim.current_iter} of {self.total_iters} max",
                    f"  Elapsed: {time.time() - t_zero:.0f} s",
                ]
                text = report.compose_single_run_report(
                    self._report_header_lines(),
                    self._report_param_lines(),
                    run_lines,
                    self._report_file_lines(first=(Path(self.out_path).name, "growth animation (temporal raster)")),
                )
                with open(self.report_path, "w", encoding="utf-8") as fh:
                    fh.write(text)
                self._log(f"Wrote run report: {Path(self.report_path).name}")
            except Exception as exc:  # noqa: BLE001 — the report is a nicety, not worth failing for
                self._log(f"Could not write the run report: {exc}", Qgis.MessageLevel.Warning)
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
            group = root.insertGroup(0, self.group_name)
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
            # existing -> raw (pre) -> post-processed options, inserted above the likelihood bands so the
            # difference post-processing makes is plain. Reversed so the list ends existing-on-top.
            for path, label in reversed(self._plan_outputs):
                lyr = QgsRasterLayer(path, f"{self.out_file_name} — {label}", "gdal")
                if lyr.isValid():
                    lyr.setCrs(self.target_crs)
                    # every plan raster is categorical now — new development is coloured per density
                    # tier (built vs mixed-use centre in distinct hues) directly in the palette
                    gis_io.apply_plan_style(lyr)
                    QgsProject.instance().addMapLayer(lyr, addToLegend=False)
                    group.insertLayer(0, lyr)
            self._log(
                f"Loaded likelihood + {len(self._plan_outputs)} plan layer(s) "
                f"(existing, raw pre-processing, and the post-processing options) for "
                f"'{self.out_file_name}' — compare and pick.",
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
        # Anchor at Jan 1 so year arithmetic below can never hit Feb 29.
        n = len(self.frames)
        start = datetime(datetime.now().year, 1, 1)
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

        root = QgsProject.instance().layerTreeRoot()
        group = root.insertGroup(0, self.group_name)
        group.setExpanded(True)
        QgsProject.instance().addMapLayer(layer, addToLegend=False)
        group.addLayer(layer)
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
