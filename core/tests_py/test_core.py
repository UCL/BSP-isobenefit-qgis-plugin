"""Python-level tests for the isobenefit extension.

These require the compiled extension (``maturin develop`` or an installed wheel).
They cover the public API surface the QGIS plugin relies on: construction,
stepping/running, snapshots, determinism, ensemble runs, and input validation.
"""

from __future__ import annotations

import numpy as np
import pytest

import isobenefit
from isobenefit import Simulation, ensemble_class_counts, ensemble_probability, run_ensemble


def make_sim(grid: int = 30, seed: int = 0, total_iters: int = 25) -> Simulation:
    """An all-green grid with one centre seed in the middle and permissive params."""
    state = np.zeros((grid, grid), dtype=np.int16)
    origin = np.full((grid, grid), -1, dtype=np.int16)  # -1 => not fixed green
    density = np.zeros((grid, grid), dtype=np.float32)
    return Simulation(
        state,
        origin,
        density,
        [(grid // 2, grid // 2)],  # centre_seeds (row, col)
        100.0,  # granularity_m
        600.0,  # max_distance_m
        1_000_000.0,  # max_populat
        100.0,  # min_green_span_m
        0.6,  # build_prob
        0.1,  # cent_prob_nb
        0.0,  # cent_prob_isol
        0.8,  # pop_target_cent_threshold
        (0.4, 0.4, 0.2),  # prob_distribution
        (6000.0, 3000.0, 1000.0),  # density_factors_km2
        2000.0,  # exist_built_km2
        total_iters,
        seed,
    )


def test_version_exposed() -> None:
    assert isinstance(isobenefit.__version__, str)
    assert isobenefit.__version__


def test_construction_seeds_centre() -> None:
    sim = make_sim(grid=12, seed=1)
    snap = sim.snapshot()
    assert snap["state"][6, 6] == 2
    assert snap["cent_acc"].sum() > 0
    assert snap["state"].dtype == np.int16
    assert snap["density"].dtype == np.float32


def test_growth_occurs_and_population_increases() -> None:
    sim = make_sim(grid=30, seed=5)
    before = sim.population
    sim.run()
    assert sim.current_iter > 0
    assert sim.population > before


def test_same_seed_is_reproducible() -> None:
    a = make_sim(grid=30, seed=99)
    b = make_sim(grid=30, seed=99)
    a.run()
    b.run()
    sa, sb = a.snapshot(), b.snapshot()
    np.testing.assert_array_equal(sa["state"], sb["state"])
    np.testing.assert_array_equal(sa["density"], sb["density"])


def test_step_matches_run() -> None:
    a = make_sim(grid=24, seed=7, total_iters=10)
    b = make_sim(grid=24, seed=7, total_iters=10)
    a.run()
    for _ in range(10):
        b.step()
    np.testing.assert_array_equal(a.snapshot()["state"], b.snapshot()["state"])


def test_ensemble_returns_members_and_diverges() -> None:
    template = make_sim(grid=40, seed=11)
    results = run_ensemble(template, 123, 6)
    assert len(results) == 6
    assert all(r.shape == (40, 40) for r in results)
    # independent seeds should not all coincide
    assert not all(np.array_equal(r, results[0]) for r in results)


def test_ensemble_is_deterministic_across_calls() -> None:
    template = make_sim(grid=30, seed=3)
    first = run_ensemble(template, 2024, 4)
    second = run_ensemble(template, 2024, 4)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


def test_ensemble_probability() -> None:
    template = make_sim(grid=30, seed=7)
    prob = ensemble_probability(template, 2024, 6)
    assert prob.shape == (30, 30)
    assert prob.dtype == np.float32
    assert (prob >= 0.0).all() and (prob <= 1.0).all()
    # the seeded centre is urban in every member
    assert prob[15, 15] == 1.0


def test_ensemble_class_counts() -> None:
    built, green, centre = ensemble_class_counts(make_sim(grid=30, seed=7), 2024, 8)
    for a in (built, green, centre):
        assert a.shape == (30, 30)
        assert a.dtype == np.uint32
    # cells partition into exactly one class -> per-cell counts sum to n (8)
    total = built + green + centre
    assert set(np.unique(total)).issubset({0, 8})
    # the seeded centre is a centre in every member
    assert centre[15, 15] == 8


def test_bad_prob_distribution_raises() -> None:
    grid = 10
    state = np.zeros((grid, grid), dtype=np.int16)
    origin = np.full((grid, grid), -1, dtype=np.int16)
    density = np.zeros((grid, grid), dtype=np.float32)
    with pytest.raises(ValueError):
        Simulation(
            state, origin, density, [(5, 5)],
            100.0, 600.0, 1_000_000.0, 100.0,
            0.6, 0.1, 0.0, 0.8,
            (0.5, 0.4, 0.2),  # sums to 1.1 -> invalid
            (6000.0, 3000.0, 1000.0), 2000.0, 5, 0,
        )
