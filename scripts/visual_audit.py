"""Headless visual A/B audit of the core recommended-plan logic (no QGIS, no engine).

Runs the PURE pipeline (``isobenefit_qgis.grid`` + the pure parts of ``isobenefit_qgis.routing``)
on small synthetic scenarios and renders side-by-side A/B PNGs, so the core behaviours can be
eyeballed and kept as an audit trail:

  01  centre optimisation OFF vs ON   (CA point seeds  ->  re-centred + grown to areas)
  02  station anchoring OFF vs ON      (a rail/tram stop pins a centre)
  03  network routing  open-grid vs network  (a barrier the network must detour around)
  04  green network    before vs after (parks carved where green access is worst)

Run (no QGIS needed — grid.py is pure numpy; the NetworkRouter is built from synthetic data):

    uv run --no-project --with numpy --with matplotlib python scripts/visual_audit.py

Images are written to ``visual_audit/``. The script is deterministic, so re-running reproduces the
same plans — it is the audit trail; commit the PNGs too if you want a frozen visual record.
"""

from __future__ import annotations

import math
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

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


def _rgb(plan):
    img = np.zeros((*plan.shape, 3), np.uint8)
    for code, color in COLORS.items():
        img[plan == code] = color
    return img


def _caption(plan, router=None):
    m = evaluate_plan(plan, GRAN, MAX_DIST, min_green_span_m=MIN_GREEN_SPAN, router=router)
    n_centres = grid.audit_centres(plan, GRAN, MAX_DIST, router=router)["summary"]["n_centres"]
    return (
        f"served {m['served_coverage']:.0%}  |  centre {m['centre_access']:.0f} m  "
        f"green {m['green_access']:.0f} m  |  {n_centres} centre(s)"
    )


def ab(name, suptitle, a_plan, a_label, b_plan, b_label, a_router=None, b_router=None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    for ax, plan, label, router in ((axes[0], a_plan, a_label, a_router), (axes[1], b_plan, b_label, b_router)):
        ax.imshow(_rgb(plan), interpolation="nearest")
        ax.set_title(f"{label}\n{_caption(plan, router)}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    handles = [mpatches.Patch(color=np.array(COLORS[c]) / 255.0, label=lbl) for c, lbl in LEGEND]
    fig.legend(handles=handles, loc="lower center", ncol=len(LEGEND), fontsize=7, frameon=False)
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    path = os.path.join(OUT_DIR, f"{name}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"wrote {path}")


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


def scenario_centre_optimisation():
    g = 60
    plan = np.full((g, g), PLAN_NONE, np.uint8)
    plan[12:40, 10:50] = PLAN_BUILT
    ca = [(14, 12), (14, 47)]  # CA centres planted at corners (off-centre)
    common = dict(max_green_frac=0.0, ca_centres=ca)
    a = optimise_plan(plan.copy(), GRAN, MIN_GREEN_SPAN, MAX_DIST, optimise_centres=False, **common)
    b = optimise_plan(plan.copy(), GRAN, MIN_GREEN_SPAN, MAX_DIST, optimise_centres=True, **common)
    ab(
        "01_centre_optimisation",
        "Centre optimisation: CA point seeds (A) vs re-centred + grown to areas (B)",
        a, "A — optimise OFF (CA seeds, single cells)",
        b, "B — optimise ON (re-centred, grown to areas)",
    )


def scenario_station_anchor():
    g = 60
    plan = np.full((g, g), PLAN_NONE, np.uint8)
    plan[12:40, 10:50] = PLAN_BUILT
    common = dict(max_green_frac=0.0, ca_centres=[(25, 30)], optimise_centres=True)
    a = optimise_plan(plan.copy(), GRAN, MIN_GREEN_SPAN, MAX_DIST, **common)
    b = optimise_plan(plan.copy(), GRAN, MIN_GREEN_SPAN, MAX_DIST, centre_anchors=[(16, 44)], **common)
    ab(
        "02_station_anchor",
        "Station anchoring: no station (A) vs a station at (16,44) pinning a centre (B)",
        a, "A — no station",
        b, "B — station anchors a centre there",
    )


def scenario_network_routing():
    g = 60
    plan = np.full((g, g), PLAN_NONE, np.uint8)
    plan[26:33, 22:29] = PLAN_BUILT  # left block
    plan[26:33, 31:38] = PLAN_BUILT  # right block (a narrow gap at cols 29-30)
    # the network connects the two sides only via a far bridge, so it must detour; a barrier sits at col 30
    router = _lattice_router(g, g, step=3, barrier_col=30, bridge_rows=(0, 3))
    common = dict(max_green_frac=0.0, ca_centres=[(29, 25)], optimise_centres=True)
    a = optimise_plan(plan.copy(), GRAN, MIN_GREEN_SPAN, MAX_DIST, router=None, **common)
    b = optimise_plan(plan.copy(), GRAN, MIN_GREEN_SPAN, MAX_DIST, router=router, **common)
    ab(
        "03_network_routing",
        "Routing: open-grid serves both blocks from one centre (A) vs network needs one per block (B)",
        a, "A — open-grid walk (one centre reaches both)",
        b, "B — street-network walk (barrier -> one centre each)",
        b_router=router,
    )


def scenario_green_carve():
    g = 60
    plan = np.full((g, g), PLAN_NONE, np.uint8)
    plan[10:50, 10:50] = PLAN_BUILT  # solid built, no green access
    a = plan.copy()
    b = optimise_plan(
        plan.copy(), GRAN, MIN_GREEN_SPAN, MAX_DIST, max_green_frac=0.3, ca_centres=[(30, 30)], optimise_centres=True
    )
    ab(
        "04_green_carve",
        "Green network: consensus with no parks (A) vs parks carved where access is worst (B)",
        a, "A — before (no green network)",
        b, "B — after (green carved + centres)",
    )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    scenario_centre_optimisation()
    scenario_station_anchor()
    scenario_network_routing()
    scenario_green_carve()
    print(f"\nDone — {OUT_DIR}")


if __name__ == "__main__":
    main()
