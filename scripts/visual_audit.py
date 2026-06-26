"""Headless visual audit of the recommended-plan dynamics (no QGIS).

Runs the PURE pipeline (``isobenefit_qgis.grid`` + the pure parts of ``isobenefit_qgis.routing``)
on a LARGE synthetic substrate and renders side-by-side panels, so every planning lever can be
eyeballed and kept as an audit trail. The substrate is deliberately big (a 6 km square at 50 m
cells) so the effect of fiddling a parameter — consolidated vs dispersed centres, short vs long
walks, more vs less green — is visible at a glance.

Two tracks:

  POST-PROCESS (pure grid.py — what optimise_plan/evaluate_plan enforce on a given development):
    01  centre optimisation        OFF (CA seeds) vs ON (re-centred + grown to areas)
    02  centre spacing             consolidated -> balanced -> dispersed   (the disperse/compact dial)
    03  centre area                small vs large (area grows with the catchment)
    04  minimum settlement size    low vs high (a failed satellite's centre is culled)
    05  centre walk distance       short vs long
    09  station anchoring          off vs on (a rail/tram stop pins a centre)
    10  network routing            open-grid vs street-network (a barrier to detour)
    11  frozen existing fabric     existing built/centres kept; new development added around them
    15  failed-satellite cleanup   stranded specks pruned to green

  (Green is the CA's own preserved network now — there is no post-process green carve — so the plan's
  green = the simulation's green; there are no green-carve scenarios.)

  CA GROWTH (runs the isobenefit engine — what the cellular automaton itself does):
    13  dispersed development      isolated-seeding Off / Low / Med / High (satellite formation)
    14  build probability          slow vs fast growth

Run (post-process track only — pure numpy, no engine/QGIS needed):

    uv run --no-project --with numpy --with matplotlib python scripts/visual_audit.py

Run (both tracks — adds the engine wheel so the CA track can run):

    uv run --no-project --with core/dist/isobenefit-*.whl --with numpy --with matplotlib \
        python scripts/visual_audit.py

Images are written to ``visual_audit/``. The script is deterministic, so re-running reproduces the
same plans — it is the audit trail. The PNGs are gitignored (regenerate on demand).
"""

from __future__ import annotations

import math
import os
import sys
from collections import namedtuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isobenefit_qgis import grid  # noqa: E402
from isobenefit_qgis.grid import (  # noqa: E402
    PLAN_BUILT,
    PLAN_CENTRE,
    PLAN_EXIST_BUILT,
    PLAN_EXIST_CENTRE,
    PLAN_GREEN,
    PLAN_NONE,
    evaluate_plan,
    optimise_plan,
)
from isobenefit_qgis.routing import NetworkRouter  # noqa: E402

GRAN, MIN_GREEN_SPAN, MAX_DIST = 50.0, 400.0, 800.0
G = 120  # substrate side in cells -> a 6 km x 6 km square at 50 m, big enough to see the differences
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "visual_audit")

# render colours: the real PLAN_PALETTE plus empty (none) and unbuildable
COLORS = {
    -1: (90, 90, 90),  # unbuildable / no-build
    PLAN_NONE: (245, 245, 245),
    PLAN_GREEN: (54, 109, 35),
    PLAN_BUILT: (196, 140, 74),
    PLAN_CENTRE: (210, 35, 35),
    PLAN_EXIST_BUILT: (120, 92, 62),
    PLAN_EXIST_CENTRE: (150, 40, 85),
}
LEGEND = [
    (PLAN_NONE, "empty"),
    (-1, "unbuildable"),
    (PLAN_GREEN, "green"),
    (PLAN_BUILT, "new built"),
    (PLAN_CENTRE, "new centre"),
    (PLAN_EXIST_BUILT, "existing built"),
    (PLAN_EXIST_CENTRE, "existing centre"),
]

# One panel of a figure: a plan plus the distance settings used to build/score it (so the caption's
# metrics match the plan). router/cdist/gdist default to the shared walk. ``stations`` are overlaid as
# a distinct marker; when a ``router`` is present its street segments are drawn over the plan.
STATION_RGB = "#1f9bff"  # bright blue, distinct from the centre red
NETWORK_RGB = "#1456b0"
Panel = namedtuple("Panel", "plan label router cdist gdist stations")


def P(plan, label, router=None, cdist=None, gdist=None, stations=()):
    return Panel(plan, label, router, cdist, gdist, tuple(stations))


def _rgb(plan):
    img = np.zeros((*plan.shape, 3), np.uint8)
    for code, color in COLORS.items():
        img[plan == code] = color
    return img


def _caption(panel):
    m = evaluate_plan(
        panel.plan, GRAN, MAX_DIST, min_green_span_m=MIN_GREEN_SPAN, router=panel.router,
        centre_distance_m=panel.cdist, green_distance_m=panel.gdist,
    )
    s = grid.audit_centres(panel.plan, GRAN, panel.cdist or MAX_DIST, router=panel.router)["summary"]
    # report centre and green coverage SEPARATELY (a "served" headline needs both, so it reads as 0%
    # on the centre-only scenarios that carry no green — these per-amenity figures are honest there).
    return (
        f"centre {m.get('centre_coverage', 0):.0%} ({m.get('centre_access', 0):.0f} m)  |  "
        f"green {m.get('green_coverage', 0):.0%} ({m.get('green_access', 0):.0f} m)\n"
        f"{s['n_centres']} centres ({s['n_new']} new)  |  {int((panel.plan == PLAN_GREEN).sum())} green cells"
    )


def _draw_network(ax, router):
    """Overlay the street graph used for routing — node-to-node segments in cell coordinates — so the
    network (and where a barrier breaks it) is visible. Nodes are stored in metres; /GRAN -> cells."""
    nodes = getattr(router, "_nodes", None)
    adj = getattr(router, "_adj", None)
    if nodes is None or adj is None or len(nodes) == 0:
        return
    for a, nbrs in enumerate(adj):
        ax_col, ax_row = nodes[a][0] / GRAN, nodes[a][1] / GRAN
        for b, _ in nbrs:
            if b > a:  # draw each undirected edge once
                ax.plot([ax_col, nodes[b][0] / GRAN], [ax_row, nodes[b][1] / GRAN],
                        color=NETWORK_RGB, lw=0.6, alpha=0.5, zorder=2)


def figure(name, suptitle, panels):
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(max(9.0, 5.4 * n), 6.7))
    if n == 1:
        axes = [axes]
    any_station = any(p.stations for p in panels)
    any_network = any(p.router is not None for p in panels)
    for ax, panel in zip(axes, panels):
        ax.imshow(_rgb(panel.plan), interpolation="nearest")
        if panel.router is not None:
            _draw_network(ax, panel.router)
        if panel.stations:  # hollow star so the centre that forms underneath stays visible
            ax.scatter([s[1] for s in panel.stations], [s[0] for s in panel.stations],
                       marker="*", s=320, facecolors="none", edgecolors=STATION_RGB, linewidths=2.2, zorder=6)
        ax.set_title(f"{panel.label}\n{_caption(panel)}", fontsize=8.5)
        ax.set_xticks([])
        ax.set_yticks([])
    handles = [mpatches.Patch(color=np.array(COLORS[c]) / 255.0, label=lbl) for c, lbl in LEGEND]
    if any_station:
        handles.append(Line2D([], [], marker="*", color="none", markerfacecolor="none",
                              markeredgecolor=STATION_RGB, markeredgewidth=1.6, markersize=15, label="station"))
    if any_network:
        handles.append(Line2D([], [], color=NETWORK_RGB, lw=1.4, alpha=0.7, label="street network"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8, frameon=False)
    fig.suptitle(suptitle, fontsize=12.5, y=0.985)
    fig.tight_layout(rect=(0, 0.05, 1, 0.90))
    path = os.path.join(OUT_DIR, f"{name}.png")
    fig.savefig(path, dpi=104)
    plt.close(fig)
    print(f"wrote {path}")


def block(plan, r0, r1, c0, c1, code=PLAN_BUILT):
    plan[r0:r1, c0:c1] = code
    return plan


def empty():
    return np.full((G, G), PLAN_NONE, np.uint8)


def _lattice_router(rows, cols, step=3, barrier_col=None, bridge_rows=()):
    """A synthetic street-graph on a lattice (nodes every ``step`` cells). Horizontal edges that
    cross ``barrier_col`` are omitted unless the row is in ``bridge_rows`` — so the two sides connect
    only via the bridge, forcing the network to detour where open-grid would cut straight across."""
    lrs, lcs = list(range(0, rows, step)), list(range(0, cols, step))
    node_id = {(r, c): i for i, (r, c) in enumerate([(r, c) for r in lrs for c in lcs])}
    nodes = np.array([(c * GRAN, r * GRAN) for r in lrs for c in lcs], float)
    adj: list = [[] for _ in nodes]

    def link(a, b):
        d = math.hypot(nodes[a][0] - nodes[b][0], nodes[a][1] - nodes[b][1])
        adj[a].append((b, d))
        adj[b].append((a, d))

    for i, r in enumerate(lrs):
        for j, c in enumerate(lcs):
            a = node_id[(r, c)]
            if j + 1 < len(lcs):
                c2 = lcs[j + 1]
                crosses = barrier_col is not None and c < barrier_col <= c2
                if not crosses or r in bridge_rows:
                    link(a, node_id[(r, c2)])
            if i + 1 < len(lrs):
                link(a, node_id[(lrs[i + 1], c)])

    cell_node = np.full((rows, cols), -1, np.int64)
    cell_access = np.full((rows, cols), np.inf)
    for rr in range(rows):
        lr = min(lrs, key=lambda v: abs(v - rr))
        for cc in range(cols):
            lc = min(lcs, key=lambda v: abs(v - cc))
            cell_node[rr, cc] = node_id[(lr, lc)]
            cell_access[rr, cc] = math.hypot(cc - lc, rr - lr) * GRAN
    cell_node[cell_access > MAX_DIST] = -1
    return NetworkRouter(nodes, adj, cell_node, cell_access, GRAN, MAX_DIST)


# A big square town used by most scenarios (rows/cols 24..96 -> a ~3.6 km block of new built fabric).
TOWN = (24, 96, 24, 96)
TOWN_CENTRE = (60, 60)


def _opt(plan, **kw):
    return optimise_plan(plan.copy(), GRAN, MIN_GREEN_SPAN, MAX_DIST, **kw)


# ---------------------------------------------------------------- post-process track


def s01_centre_optimisation():
    plan = block(empty(), *TOWN)
    seeds = [(28, 28), (28, 92), (92, 28), (92, 92)]  # CA centres planted at the corners (off-centre)
    common = dict(ca_centres=seeds)
    figure(
        "01_centre_optimisation",
        "Centre optimisation: CA corner seeds (A) vs re-centred + grown to areas (B)",
        [
            P(_opt(plan, optimise_centres=False, **common), "A — optimise OFF (raw CA seeds)"),
            P(_opt(plan, optimise_centres=True, **common), "B — optimise ON (re-centred, grown)"),
        ],
    )


def s02_centre_spacing():
    plan = block(empty(), *TOWN)
    common = dict(ca_centres=[TOWN_CENTRE], optimise_centres=True)
    figure(
        "02_centre_spacing",
        "Centre spacing — the consolidated <-> dispersed dial (smaller spacing = more, closer centres)",
        [
            P(_opt(plan, centre_spacing_m=800, **common), "consolidated (spacing 800 m)"),
            P(_opt(plan, centre_spacing_m=500, **common), "balanced (spacing 500 m)"),
            P(_opt(plan, centre_spacing_m=300, **common), "dispersed (spacing 300 m)"),
        ],
    )


def s03_centre_area():
    plan = block(empty(), *TOWN)
    common = dict(ca_centres=[TOWN_CENTRE], optimise_centres=True, centre_spacing_m=600)
    figure(
        "03_centre_area",
        "Centre area: each centre grows with the population it serves (area per home)",
        [
            P(_opt(plan, centre_area_frac=0.02, **common), "small centres (area 0.02 / home)"),
            P(_opt(plan, centre_area_frac=0.12, **common), "large centres (area 0.12 / home)"),
        ],
    )


def s04_min_settlement():
    # a big town plus a small detached satellite; a CA centre is seeded on each
    plan = block(empty(), *TOWN)
    block(plan, 8, 14, 8, 14)  # 6x6 satellite, far from the town
    common = dict(ca_centres=[TOWN_CENTRE, (11, 11)], optimise_centres=True)
    figure(
        "04_min_settlement",
        "Minimum settlement size: a small satellite keeps its centre (A) or is too small and loses it (B)",
        [
            P(_opt(plan, centre_min_settlement=3, **common), "A — min size 3 (satellite kept)"),
            P(_opt(plan, centre_min_settlement=60, **common), "B — min size 60 (satellite culled)"),
        ],
    )


def s05_centre_walk():
    plan = block(empty(), *TOWN)
    common = dict(ca_centres=[TOWN_CENTRE], optimise_centres=True)
    figure(
        "05_centre_walk",
        "Centre walk distance: a shorter walk needs more centres to keep everyone covered",
        [
            P(_opt(plan, centre_distance_m=400, **common), "short walk (400 m)", cdist=400),
            P(_opt(plan, centre_distance_m=1000, **common), "long walk (1000 m)", cdist=1000),
        ],
    )


def s09_station_anchor():
    # consolidated spacing + a larger area fraction (few, sizable centres) so the station's grown
    # centre is clearly visible rather than lost among many small dispersed ones
    plan = block(empty(), *TOWN)
    common = dict(ca_centres=[TOWN_CENTRE], optimise_centres=True, centre_area_frac=0.16)
    figure(
        "09_station_anchor",
        "Station anchoring: no station (A) vs a rail/tram station seeding a (grown) centre at (40, 84) (B)",
        [
            P(_opt(plan, **common), "A — no station"),
            P(_opt(plan, centre_anchors=[(40, 84)], **common), "B — station seeds a sizable centre",
              stations=[(40, 84)]),
        ],
    )


def s10_network_routing():
    # two small blocks separated by a narrow gap; ONE centre can cover both only if the network links them
    plan = empty()
    block(plan, 55, 65, 54, 60)  # left block
    block(plan, 55, 65, 64, 70)  # right block (gap at cols 60-64)
    linked = _lattice_router(G, G, step=3, barrier_col=62, bridge_rows=(54, 57, 60, 63, 66))  # roads bridge the gap
    split = _lattice_router(G, G, step=3, barrier_col=62, bridge_rows=())  # no road across the gap
    common = dict(ca_centres=[(60, 57)], optimise_centres=True)
    figure(
        "10_network_routing",
        "Network linking vs separating the same clusters: a bridge serves both from one centre; severing needs two",
        [
            P(_opt(plan, router=None, **common), "open-grid (no streets) — one centre"),
            P(_opt(plan, router=linked, **common), "network LINKS them (road bridges the gap)", router=linked),
            P(_opt(plan, router=split, **common), "network SEPARATES them (no crossing)", router=split),
        ],
    )


def s11_frozen_existing():
    # an existing town (frozen) on the left, new growth on the right
    plan = empty()
    block(plan, 30, 90, 20, 56, PLAN_BUILT)  # existing
    block(plan, 30, 90, 58, 96, PLAN_BUILT)  # new
    existing_built = np.zeros((G, G), bool)
    existing_built[30:90, 20:56] = True
    existing_centres = [(60, 38)]
    plan[60, 38] = PLAN_CENTRE
    opt = _opt(
        plan, existing_built=existing_built, existing_centres=existing_centres,
        ca_centres=[(60, 77)], optimise_centres=True, centre_spacing_m=700,
    )
    opt = grid._mark_existing(opt, existing_built=existing_built, existing_centres=existing_centres)
    figure(
        "11_frozen_existing",
        "Frozen existing fabric: existing kept untouched, new green + centres added only on new land",
        [P(opt, "existing frozen (dark) + new development (light)")],
    )


def s15_island_cleanup():
    # a main town + a viable satellite (each with a CA centre) + two stranded specks with no centre
    plan = empty()
    block(plan, 30, 82, 20, 64)  # main town
    block(plan, 18, 30, 82, 96)  # viable satellite (12x14 = 168 cells)
    block(plan, 8, 11, 8, 11)  # stranded speck A (3x3 = 9 cells, no centre)
    block(plan, 96, 98, 30, 33)  # stranded speck B (2x3 = 6 cells, no centre)
    common = dict(ca_centres=[(56, 42), (24, 89)], optimise_centres=True,
                  centre_min_settlement=12)
    figure(
        "15_island_cleanup",
        "Failed-satellite cleanup: stranded residential specks (no centre, below min settlement) pruned (B)",
        [
            P(_opt(plan, prune_islands=False, **common), "A — no cleanup (specks stranded, no centre)"),
            P(_opt(plan, prune_islands=True, **common), "B — cleanup on (stranded specks pruned)"),
        ],
    )


POST_PROCESS = [
    s01_centre_optimisation, s02_centre_spacing, s03_centre_area, s04_min_settlement,
    s05_centre_walk,
    s09_station_anchor, s10_network_routing, s11_frozen_existing,
    s15_island_cleanup,
]


# ---------------------------------------------------------------- CA-growth track (needs the engine)


def _ca_track():
    """Run the isobenefit engine to show what the CELLULAR AUTOMATON does (vs the post-process above).
    Skipped with a clear message if the engine wheel isn't installed in this environment."""
    try:
        import isobenefit
    except ImportError:
        print("\n[CA track skipped] engine not installed — re-run with `--with core/dist/isobenefit-*.whl` "
              "to render the dispersal / build-probability plots.")
        return
    try:
        _run_ca_track(isobenefit)
    except Exception as exc:  # noqa: BLE001 — the audit must never hard-fail on an engine API mismatch
        print(f"\n[CA track skipped] engine ran but the audit could not drive it: {exc}")


def _ca_plan(isobenefit, *, cent_prob_isol, build_prob, seed):
    """Grow one CA run to its population target and return the recommended plan (so the CA's
    development pattern is shown through the same plan codes as the post-process track). Array
    dtypes mirror sim_runner: state/origin int16, density float32, seeds a list of (row, col)."""
    n = 80
    state = np.zeros((n, n), dtype=np.int16)  # 0 = green/buildable everywhere (extents = whole grid)
    state[37:44, 37:44] = 1  # a small built core in the middle to grow from
    origin = np.full((n, n), -1, dtype=np.int16)
    origin[37:44, 37:44] = 1
    density = np.zeros((n, n), dtype=np.float32)
    seeds = [(40, 40)]  # one centre seed so the core has centre access and growth can start
    sim = isobenefit.Simulation(
        state, origin, density, seeds, GRAN, MAX_DIST, 20000.0, MIN_GREEN_SPAN,
        build_prob, 0.0, cent_prob_isol, 0.8,
        (0.4, 0.4, 0.2), (6000.0, 3000.0, 1000.0), 2000.0, 200, seed,  # density factors descending
    )
    st = np.asarray(isobenefit.run_ensemble(sim, seed, 1)[0])
    ca = [(int(y), int(x)) for y, x in np.argwhere(st == 2)]
    base = grid._state_to_plan(st, MIN_GREEN_SPAN, GRAN)
    # min settlement size 8: stranded specks are pruned and tiny hamlets don't keep their own centre,
    # so a dispersed run yields only viable settlements (each with a centre), not residential islands.
    return optimise_plan(base, GRAN, MIN_GREEN_SPAN, MAX_DIST,
                         ca_centres=ca, optimise_centres=True, centre_min_settlement=8)


def _run_ca_track(isobenefit):
    figure(
        "13_dispersed_development",
        "Dispersed development (CA isolated seeding): Off -> High lets new settlements form away from the core",
        [
            P(_ca_plan(isobenefit, cent_prob_isol=0.0, build_prob=0.3, seed=1), "Off (compact, contiguous)"),
            P(_ca_plan(isobenefit, cent_prob_isol=0.01, build_prob=0.3, seed=1), "Low"),
            P(_ca_plan(isobenefit, cent_prob_isol=0.05, build_prob=0.3, seed=1), "Medium / High (satellites)"),
        ],
    )
    figure(
        "14_build_probability",
        "Build probability: the per-step growth rate (lower = tighter, more compact development)",
        [
            P(_ca_plan(isobenefit, cent_prob_isol=0.0, build_prob=0.1, seed=2), "slow growth (0.1)"),
            P(_ca_plan(isobenefit, cent_prob_isol=0.0, build_prob=0.5, seed=2), "fast growth (0.5)"),
        ],
    )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for scenario in POST_PROCESS:
        scenario()
    _ca_track()
    print(f"\nDone — {OUT_DIR}")


if __name__ == "__main__":
    main()
