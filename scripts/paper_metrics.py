#!/usr/bin/env python3
"""Full-resolution case-study metrics for the paper.

Runs the three paper scenarios at their committed grid resolution with the
full fifty-run ensemble and the standard selection pipeline, one preset at a
time, and writes metrics plus full-resolution figure panels. This replaces
the preview-resolution gallery numbers cited by the manuscript.

    .venv/bin/python scripts/paper_metrics.py

Outputs: temp/paper_metrics/metrics.json and PNG panels; the panels the
manuscript includes are also written to paper/figures/ under the same names
the tex already references.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import isobenefit  # noqa: E402

from isobenefit_qgis import grid as G  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "render_scenario_gallery", os.path.join(REPO, "scripts", "render_scenario_gallery.py")
)
_gallery = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gallery)

OUT = os.path.join(REPO, "temp", "paper_metrics")
FIGS = os.path.join(REPO, "paper", "figures")

# scenario -> presets the manuscript cites (id, params overrides). Each case-study FIGURE
# demonstrates one dial where its scenario binds hardest (Cambourne: centre walk; Crews Hill:
# density mix; Pajarito: attachment and the viability threshold); the walk presets are
# computed for every case because the cross-scenario sweep table cites them.
PLAN = {
    "cambourne": [
        ("baseline", {}),
        ("walk800", {"centre_walk_m": 800.0}),
        ("walk1600", {"centre_walk_m": 1600.0}),
        ("corridor", {"corridor_weight": 0.95}),
    ],
    "london_crews_hill": [
        ("baseline", {}),
        ("walk800", {"centre_walk_m": 800.0}),
        ("walk1600", {"centre_walk_m": 1600.0}),
        ("denser", {"shares": {"high": 0.5, "medium": 0.3, "low": 0.2}}),
        ("lower", {"shares": {"high": 0.1, "medium": 0.3, "low": 0.6}}),
    ],
    "medellin_pajarito": [
        ("baseline", {}),
        ("walk800", {"centre_walk_m": 800.0}),
        ("walk1600", {"centre_walk_m": 1600.0}),
        ("compact", {"allow_detached": False}),
        ("quota4000", {"min_settlement_pop": 4000}),
    ],
}

# panels the manuscript includes, copied into paper/figures under the tex names
PAPER_PANELS = {
    ("cambourne", "baseline"): "cambourne_baseline.png",
    ("cambourne", "walk1600"): "cambourne_walk1600.png",
    ("london_crews_hill", "lower"): "london_crews_hill_lower.png",
    ("london_crews_hill", "denser"): "london_crews_hill_denser.png",
    ("medellin_pajarito", "baseline"): "medellin_pajarito_baseline.png",
    ("medellin_pajarito", "compact"): "medellin_pajarito_compact.png",
    ("medellin_pajarito", "quota4000"): "medellin_pajarito_quota4000.png",
}

# the transit demonstration pair: the baseline plan and the corridor-preference plan,
# rendered again with the stops overlaid, straight into the paper's figure names
TRANSIT_PANELS = {
    ("cambourne", "baseline"): "cambourne_transit_off.png",
    ("cambourne", "corridor"): "cambourne_transit_corridor.png",
}


def run_preset(sub, params, overrides, runs=50):
    p = dict(params)
    for k, v in overrides.items():
        p[k] = v
    gran = float(p["grid_size_m"])
    tiers = (p["densities_km2"]["high"], p["densities_km2"]["medium"], p["densities_km2"]["low"])
    shares = (p["shares"]["high"], p["shares"]["medium"], p["shares"]["low"])
    mean_density = sum(s * d for s, d in zip(shares, tiers))
    walk = float(p.get("centre_walk_m", 800.0))
    green_walk = float(p.get("green_walk_m", 400.0))
    park_m2 = float(p["min_park_area_ha"]) * 1.0e4 if p.get("min_park_area_ha") else None
    max_walk = max(walk, green_walk)
    cell_km2 = gran * gran / 1e6
    min_cells = max(1, round(float(p.get("min_settlement_pop", 2000.0)) / (mean_density * cell_km2)))
    base_state, base_origin = sub["state"].copy(), sub["origin"].copy()
    G.green_unviable_pockets(
        base_state, base_origin, min_cells,
        existing_centres=sub["seeds"], granularity_m=gran, centre_distance_m=walk,
    )

    # transit: the anchor masks report catchment coverage for every preset; the corridor
    # preference additionally weights the growth draws when an override raises it above 0.
    # At weight 0 the reference is the existing stops; a weighted run adds the proposed
    # corridor and hub (the inputs it is asked to develop along) to the sources. Corridor
    # cells project the stop catchment, hubs the wider hub catchment.
    catchment_m = float(p.get("stop_catchment_m", 400.0))
    hub_catchment_m = float(p.get("hub_catchment_m", 1200.0))
    corridor_w = float(p.get("corridor_weight", 0.0) or 0.0)
    # rail and tram stations anchor a pinned centre in every run; a hand-drawn proposed
    # hub joins them only when the scenario is asked to develop along transit
    hubs = list(sub.get("stations", []))
    if corridor_w > 0.0:
        hubs += [h for h in sub.get("proposed_hubs", []) if h not in set(hubs)]
    corridor_cells = list(sub.get("stops", []))
    if corridor_w > 0.0:
        corridor_cells += list(sub.get("corridor", []))

    def _mask(cells):
        if not cells:
            return None
        m = np.zeros_like(base_state, bool)
        for r, c in cells:
            m[r, c] = True
        return m

    stop_mask, hub_mask = _mask(corridor_cells), _mask(hubs)
    catchment = None
    if corridor_w > 0.0 and (stop_mask is not None or hub_mask is not None):
        catchment = np.zeros_like(base_state, bool)
        for mask, reach in ((stop_mask, catchment_m), (hub_mask, hub_catchment_m)):
            if mask is not None:
                d = G._walk_distance(mask, gran, reach, blocked=(base_state == -1))
                catchment |= np.isfinite(d)

    # the plugin's own ensemble runner, so the paper's numbers come from the same
    # member seeding the plugin (and demo_run_report.py) would produce
    seed = int(p.get("random_seed", 42))
    t0 = time.time()
    sim_seeds = sub["seeds"] + [h for h in hubs if h not in set(sub["seeds"])]
    template = isobenefit.Simulation(
        base_state.copy(), base_origin.copy(),
        np.zeros_like(base_state, np.float32), sim_seeds,
        gran, walk, green_walk, float(p["target_population"]),
        float(p.get("min_green_span_m", 400.0)), _gallery.BUILD_PROB,
        _gallery.centre_quota(p), _gallery.allow_detached(p),
        shares, tiers, int(p.get("max_iterations", 300)), seed,
        min_park_area_m2=park_m2,
        sterile=G.sterile_fabric(base_origin == 1, sub["seeds"]),
        transit_catchment=catchment, corridor_weight=corridor_w,
        provision_seeds=hubs,
    )
    states = [np.asarray(s) for s in isobenefit.run_ensemble(template, seed, runs)]
    sim_s = time.time() - t0

    plan, metrics, _pre, _best = G.select_plan(
        states, gran, max_walk,
        existing_built=(base_origin == 1), existing_green=(base_origin == 0),
        existing_centres=sub["seeds"], centre_mode="placed",
        centre_anchors=hubs or None,
        centre_distance_m=walk, green_distance_m=green_walk, new_density_km2=mean_density,
        centre_min_settlement=min_cells,
        target_population=float(p["target_population"]),
        min_park_area_m2=park_m2,
        transit_stops=stop_mask, stop_catchment_m=catchment_m,
        transit_hubs=hub_mask, hub_catchment_m=hub_catchment_m,
    )
    keep = {
        k: round(float(metrics.get(k, 0)), 3)
        for k in (
            "served_coverage", "served_coverage_incl_existing",
            "centre_coverage", "green_coverage", "centre_access",
            "green_access", "centre_walk_mean", "green_walk_mean", "population",
            "centre_m2_per_person", "green_m2_per_person", "built_cells",
            "transit_coverage", "transit_access",
        )
    }
    keep["sim_seconds"] = round(sim_s, 1)
    disp = G.to_tiered_plan(plan, G.derive_density(plan, gran, walk, tiers, shares), tiers)
    return disp, keep


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIGS, exist_ok=True)
    _gallery.render_legend(os.path.join(FIGS, "legend.png"))
    # optional scenario filter (e.g. `paper_metrics.py cambourne`): recompute only the named
    # scenarios and merge into the existing metrics.json rather than starting it afresh
    wanted = sys.argv[1:]
    plan_items = {k: v for k, v in PLAN.items() if not wanted or k in wanted}
    results = {}
    metrics_path = os.path.join(OUT, "metrics.json")
    if wanted and os.path.exists(metrics_path):
        with open(metrics_path, encoding="utf-8") as fh:
            results = json.load(fh)
    for name, presets in plan_items.items():
        folder = os.path.join(REPO, "scenarios", name)
        params, layers, extents = _gallery.load_scenario(folder)
        extent = extents["main"]
        gran = float(params["grid_size_m"])
        sub = _gallery.substrate(extent, layers, gran)
        print(f"{name}: full grid {sub['cols']}x{sub['rows']} at {gran:.0f} m, "
              f"{len(sub['seeds'])} centre seeds", flush=True)

        existing = _gallery.existing_panel(sub)
        path = os.path.join(OUT, f"{name}_existing.png")
        _gallery.render_png(existing, layers, sub, gran, path)
        if (name, "existing") in PAPER_PANELS:
            import shutil

            shutil.copyfile(path, os.path.join(FIGS, PAPER_PANELS[(name, "existing")]))

        results[name] = {"grid": f"{sub['cols']}x{sub['rows']} at {gran:.0f} m"}
        for preset_id, overrides in presets:
            disp, keep = run_preset(sub, dict(params), overrides)
            results[name][preset_id] = keep
            path = os.path.join(OUT, f"{name}_{preset_id}.png")
            # a scenario with stations marks them, so a reader can see where the pinned
            # centres come from (Crews Hill's release is organised around its station)
            _gallery.render_png(disp, layers, sub, gran, path, hubs=sub.get("stations") or None)
            if (name, preset_id) in PAPER_PANELS:
                import shutil

                shutil.copyfile(path, os.path.join(FIGS, PAPER_PANELS[(name, preset_id)]))
            if (name, preset_id) in TRANSIT_PANELS:
                _gallery.render_png(
                    disp, layers, sub, gran,
                    os.path.join(FIGS, TRANSIT_PANELS[(name, preset_id)]), stops=sub["stops"],
                    hubs=(sub.get("stations", []) + sub.get("proposed_hubs", [])) or None,
                )
            print(f"  {preset_id}: served {keep['served_coverage']:.0%}, "
                  f"pop {keep['population']:,.0f}, centre walk {keep['centre_walk_mean']:,.0f} m "
                  f"({keep['sim_seconds']} s sims)", flush=True)
            with open(os.path.join(OUT, "metrics.json"), "w", encoding="utf-8") as fh:
                json.dump(results, fh, indent=2)

    print("PAPER-METRICS-DONE")


if __name__ == "__main__":
    main()
