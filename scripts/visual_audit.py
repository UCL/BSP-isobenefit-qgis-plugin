"""Headless visual audit of the recommended-plan dynamics (no QGIS).

Runs the PURE pipeline (``isobenefit_qgis.grid``)
on a LARGE synthetic substrate and renders side-by-side panels, so every planning lever can be
eyeballed and kept as an audit trail. The substrate is deliberately big (a 6 km square at 50 m
cells) so the effect of fiddling a parameter — spread vs clustered centres, short vs long
walks, more vs less green — is visible at a glance.

Two tracks:

  POST-PROCESS (pure grid.py — what optimise_plan/evaluate_plan enforce on a given development):
    01  centre optimisation        OFF (CA seeds) vs ON (re-centred + grown to areas)
    02  centre clustering          moderate (1.5x) vs tight (2.5x) — the two options the run saves
    03  centre area                small vs large (area grows with the catchment)
    04  minimum settlement size    low vs high (a failed satellite's centre is culled)
    05  centre walk distance       short vs long
    06  centre centering           concave/irregular built: centre sits at the interior, not a rim/gap
    07  clustering on dispersed    moderate vs tight on scattered blobs (small blobs can't cluster)
    09  station anchoring          off vs on (a rail/tram stop pins a centre)
    11  frozen existing fabric     existing built/centres kept; new development added around them
    15  failed-satellite cleanup   stranded specks pruned to green

  (Green is the CA's own preserved network now — there is no post-process green carve — so the plan's
  green = the simulation's green; there are no green-carve scenarios.)

  CA GROWTH (runs the isobenefit engine — what the cellular automaton itself does):
    13  dispersed development      isolated-seeding Off / Moderate / Aggressive (satellite formation)
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
# metrics match the plan). cdist/gdist default to the shared walk. ``stations`` are overlaid as
# a distinct marker.
STATION_RGB = "#1f9bff"  # bright blue, distinct from the centre red
Panel = namedtuple("Panel", "plan label cdist gdist stations")


def P(plan, label, cdist=None, gdist=None, stations=()):
    return Panel(plan, label, cdist, gdist, tuple(stations))


def _rgb(plan):
    img = np.zeros((*plan.shape, 3), np.uint8)
    for code, color in COLORS.items():
        img[plan == code] = color
    return img


def _caption(panel):
    m = evaluate_plan(
        panel.plan, GRAN, MAX_DIST, min_green_span_m=MIN_GREEN_SPAN, 
        centre_distance_m=panel.cdist, green_distance_m=panel.gdist,
    )
    s = grid.audit_centres(panel.plan, GRAN, panel.cdist or MAX_DIST, )["summary"]
    # report centre and green coverage SEPARATELY (a "served" headline needs both, so it reads as 0%
    # on the centre-only scenarios that carry no green — these per-amenity figures are honest there).
    return (
        f"centre {m.get('centre_coverage', 0):.0%} ({m.get('centre_access', 0):.0f} m)  |  "
        f"green {m.get('green_coverage', 0):.0%} ({m.get('green_access', 0):.0f} m)\n"
        f"{s['n_centres']} centres ({s['n_new']} new)  |  {int((panel.plan == PLAN_GREEN).sum())} green cells"
    )



def figure(name, suptitle, panels):
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(max(9.0, 5.4 * n), 6.7))
    if n == 1:
        axes = [axes]
    any_station = any(p.stations for p in panels)
    for ax, panel in zip(axes, panels):
        ax.imshow(_rgb(panel.plan), interpolation="nearest")
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


def s02_centre_clustering():
    # the two options the run actually saves (alongside the raw): moderate (1.5x walk) vs tight (2.5x walk),
    # at the real default 400 m centre walk. Same buildings; only the centres differ. A larger spacing
    # clusters harder (fewer, larger, more central). NB much larger multiples would saturate — once the
    # town only needs N centres to be covered within the spacing, bigger spacings all give the same N.
    cw = 400.0
    plan = block(empty(), *TOWN)
    common = dict(ca_centres=[TOWN_CENTRE], optimise_centres=True, centre_distance_m=cw)
    figure(
        "02_centre_clustering",
        "Centre clustering options — same buildings, fewer & larger centres as spacing grows (coverage trades off)",
        [
            P(_opt(plan, centre_spacing_m=1.5 * cw, **common), "moderately clustered (1.5x walk)", cdist=cw),
            P(_opt(plan, centre_spacing_m=2.5 * cw, **common), "tightly clustered (2.5x walk)", cdist=cw),
        ],
    )


def s03_centre_area():
    plan = block(empty(), *TOWN)
    common = dict(ca_centres=[TOWN_CENTRE], optimise_centres=True, centre_spacing_m=600)
    figure(
        "03_centre_area",
        "Centre area: each centre grows with the population it serves (area per home)",
        [
            P(_opt(plan, centre_m2_per_person=5.0, **common), "small centres (5 m2 / person)"),
            P(_opt(plan, centre_m2_per_person=30.0, **common), "large centres (30 m2 / person)"),
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


def s06_centre_centering():
    # Irregular / concave developments, where a plain catchment CENTROID lands in a gap or on a rim:
    # the centre is instead placed at the catchment's deepest INTERIOR, so it sits ON built and central
    # to the development it anchors. (On a solid convex block the centroid is already central, so this
    # only changes the awkward shapes — exactly the ones the CA's dispersed growth produces.)
    ell = empty()
    block(ell, 30, 92, 30, 50)  # vertical arm
    block(ell, 72, 92, 30, 92)  # horizontal arm -> an L; the union's centroid sits off built, in the notch
    ring = empty()
    block(ring, 24, 96, 24, 96)  # a solid block...
    block(ring, 46, 74, 46, 74, code=PLAN_NONE)  # ...hollowed out -> a ring; its centroid is in the hole
    figure(
        "06_centre_centering",
        "Centre centering on concave built: placed at the catchment interior (on built), not in a gap or on a rim",
        [
            P(_opt(ell, ca_centres=[(82, 40)], optimise_centres=True, centre_distance_m=1200),
              "L-shaped development (1200 m walk)", cdist=1200),
            P(_opt(ring, ca_centres=[(60, 60)], optimise_centres=True, centre_distance_m=1200),
              "ring (hollow centre, 1200 m walk)", cdist=1200),
        ],
    )


def s07_clustering_on_dispersed():
    # The clustering options on a FRAGMENTED development (scattered blobs of different sizes — what
    # dispersed CA growth actually produces), at the real 400 m walk. Clustering only thins centres
    # WITHIN a blob big enough to hold several; a small blob keeps its one centre at either setting
    # (nothing to cluster there). This is why moderate vs tight differ a lot on a big contiguous town
    # but barely on a finely-dispersed one. Centres still sit at each blob's interior, on built.
    plan = empty()
    seeds = []
    for r, c, h, w in [
        (16, 16, 40, 40),  # large blob (holds several centres)
        (20, 78, 30, 34),  # large blob
        (74, 24, 28, 46),  # large blob
        (86, 86, 18, 20),  # medium
        (58, 60, 12, 12),  # small (one centre regardless)
        (40, 52, 10, 10),  # small
        (102, 64, 12, 24),  # medium strip
    ]:
        block(plan, r, r + h, c, c + w)
        seeds.append((r + h // 2, c + w // 2))
    common = dict(ca_centres=seeds, optimise_centres=True, centre_distance_m=400)
    figure(
        "07_clustering_on_dispersed",
        "Clustering on a fragmented development: thins centres only within blobs big enough to hold several",
        [
            P(_opt(plan, centre_spacing_m=1.5 * 400, **common), "moderately clustered (1.5x walk)", cdist=400),
            P(_opt(plan, centre_spacing_m=2.5 * 400, **common), "tightly clustered (2.5x walk)", cdist=400),
        ],
    )


def s09_station_anchor():
    # a larger area fraction (few, sizable centres) so the station's grown
    # centre is clearly visible rather than lost among many small spread-out ones
    plan = block(empty(), *TOWN)
    common = dict(ca_centres=[TOWN_CENTRE], optimise_centres=True, centre_m2_per_person=40.0)
    figure(
        "09_station_anchor",
        "Station anchoring: no station (A) vs a rail/tram station seeding a (grown) centre at (40, 84) (B)",
        [
            P(_opt(plan, **common), "A — no station"),
            P(_opt(plan, centre_anchors=[(40, 84)], **common), "B — station seeds a sizable centre",
              stations=[(40, 84)]),
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
    s01_centre_optimisation, s02_centre_clustering, s03_centre_area, s04_min_settlement,
    s05_centre_walk, s06_centre_centering, s07_clustering_on_dispersed,
    s09_station_anchor, s11_frozen_existing,
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
        "Dispersed development (CA isolated seeding): Off -> Aggressive lets new settlements form away from the core",
        [
            P(_ca_plan(isobenefit, cent_prob_isol=0.0, build_prob=0.3, seed=1), "Off (compact, contiguous)"),
            P(_ca_plan(isobenefit, cent_prob_isol=0.0001, build_prob=0.3, seed=1), "Moderate"),
            P(_ca_plan(isobenefit, cent_prob_isol=0.04, build_prob=0.3, seed=1), "Aggressive (satellites)"),
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
