//! The Isobenefit growth simulation: state machine, the `green_to_built`
//! decision, and the parallel `run_ensemble`.
//!
//! State grid values: -1 unbuildable, 0 nature/green, 1 built, 2 centre.
//! All GIS concerns (rasterization, CRS, IO) live in the QGIS plugin; this
//! module only sees numpy-shaped integer/float grids.

use crate::access::{agg_dijkstra_cont, prepare_green_arrs, DijkstraOpts};
use crate::density::{random_density, rng_for, splitmix64};
use crate::neighbours::{count_cont_nbs, green_spans, iter_nbs};
use ndarray::Array2;
use rayon::prelude::*;

/// Scalar simulation parameters (densities already converted to per-block).
#[derive(Clone, Copy)]
pub struct Params {
    pub granularity_m: f64,
    pub max_distance_m: f64,
    pub max_populat: f64,
    /// Minimum green span / min-park side length, in **metres** (consistent
    /// throughout — this is the fix for the original metres-vs-km units bug).
    pub min_green_span_m: f64,
    pub build_prob: f64,
    pub cent_prob_nb: f64,
    pub cent_prob_isol: f64,
    pub pop_target_cent_threshold: f64,
    pub prob_distribution: (f64, f64, f64),
    pub high_per_block: f64,
    pub med_per_block: f64,
    pub low_per_block: f64,
}

impl Params {
    /// Builds parameters from raw UI values, converting densities from
    /// persons/km^2 to persons/block and validating invariants.
    #[allow(clippy::too_many_arguments)]
    pub fn from_raw(
        granularity_m: f64,
        max_distance_m: f64,
        max_populat: f64,
        min_green_span_m: f64,
        build_prob: f64,
        cent_prob_nb: f64,
        cent_prob_isol: f64,
        pop_target_cent_threshold: f64,
        prob_distribution: (f64, f64, f64),
        density_factors_km2: (f64, f64, f64),
    ) -> Result<Params, String> {
        let prob_sum = ((prob_distribution.0 + prob_distribution.1 + prob_distribution.2) * 100.0)
            .round()
            / 100.0;
        if (prob_sum - 1.0).abs() > f64::EPSILON {
            return Err("The prob_distribution parameter must sum to 1.".to_string());
        }
        if density_factors_km2.0 <= density_factors_km2.1
            || density_factors_km2.1 <= density_factors_km2.2
        {
            return Err("Density factors should be in descending order".to_string());
        }
        let block = granularity_m * granularity_m / 1.0e6;
        Ok(Params {
            granularity_m,
            max_distance_m,
            max_populat,
            min_green_span_m,
            build_prob,
            cent_prob_nb,
            cent_prob_isol,
            pop_target_cent_threshold,
            prob_distribution,
            high_per_block: density_factors_km2.0 * block,
            med_per_block: density_factors_km2.1 * block,
            low_per_block: density_factors_km2.2 * block,
        })
    }
}

/// Attempt to convert green cell `(y, x)` to built land, returning the updated
/// `(green_itx, green_acc)` arrays on success or `None` if the move is rejected
/// (would create a runaway streak, crimp a green corridor, split green below the
/// minimum area, or cut off a built cell's green access).
#[allow(clippy::too_many_arguments)]
pub fn green_to_built(
    y: usize,
    x: usize,
    state: &Array2<i16>,
    old_itx: &Array2<i16>,
    old_acc: &Array2<i32>,
    granularity_m: f64,
    max_distance_m: f64,
    min_green_span_m: f64,
) -> Option<(Array2<i16>, Array2<i32>)> {
    let (rows, cols) = state.dim();
    let mut new_itx = old_itx.clone();
    let mut new_acc = old_acc.clone();

    let (_tot, cont_urban, urban_regions) = count_cont_nbs(state, y, x, &[1, 2]);
    // a single urban neighbour is only allowed if that neighbour is a centrality
    if cont_urban == 1 {
        let (_t, cent_cont, _r) = count_cont_nbs(state, y, x, &[2]);
        if cent_cont != 1 {
            return None;
        }
    }
    // don't crimp a green corridor below the minimum span
    if !green_spans(state, y, x, granularity_m, min_green_span_m) {
        return None;
    }
    // if this build would split green into multiple regions, ensure each green
    // neighbour still reaches at least a minimum contiguous green area
    if urban_regions > 1 {
        // number of cells for a min_green_span x min_green_span area (metres)
        let target_count =
            (min_green_span_m * min_green_span_m / (granularity_m * granularity_m)).round() as i64;
        let mut mock = state.clone();
        mock[[y, x]] = 1; // tentatively built
        let mut opts = DijkstraOpts::new(max_distance_m * 2.0, granularity_m);
        opts.break_count = Some(target_count);
        opts.rook = true; // rook only, else diagonal hops cheat the contiguity
        for (ny, nx) in iter_nbs(rows, cols, y, x, false) {
            if state[[ny, nx]] != 0 {
                continue;
            }
            let nb_acc = agg_dijkstra_cont(&mock, ny, nx, &[0], &[0], &opts);
            if (nb_acc.sum() as i64) < target_count {
                return None;
            }
        }
    }

    let acc_opts = DijkstraOpts::new(max_distance_m, granularity_m);
    // if the cell is currently periphery, demote it and decrement its access footprint
    if new_itx[[y, x]] == 2 {
        new_itx[[y, x]] = 1;
        let dec = agg_dijkstra_cont(&new_itx, y, x, &[0, 1, 2], &[0, 1, 2], &acc_opts);
        new_acc = new_acc - dec;
    }
    // newly exposed green neighbours become periphery; add their access footprint
    for (ny, nx) in iter_nbs(rows, cols, y, x, true) {
        if new_itx[[ny, nx]] == 0 {
            new_itx[[ny, nx]] = 2;
            let inc = agg_dijkstra_cont(&new_itx, ny, nx, &[0, 1, 2], &[0, 1, 2], &acc_opts);
            new_acc = new_acc + inc;
        }
    }

    // reject if any existing built cell would lose all green access
    for y2 in 0..rows {
        for x2 in 0..cols {
            if state[[y2, x2]] > 0
                && new_acc[[y2, x2]] <= 0
                && old_acc[[y2, x2]] > new_acc[[y2, x2]]
            {
                return None;
            }
        }
    }
    Some((new_itx, new_acc))
}

/// The full simulation state. Construct via [`Simulation::new`], then drive with
/// [`Simulation::step`] / [`Simulation::run`].
#[derive(Clone)]
pub struct Simulation {
    pub state: Array2<i16>,
    pub origin: Array2<i16>,
    pub density: Array2<f32>,
    pub green_itx: Array2<i16>,
    pub green_acc: Array2<i32>,
    pub cent_acc: Array2<i32>,
    pub params: Params,
    pub total_iters: usize,
    pub current_iter: usize,
    pub master_seed: u64,
    pub pop_target_ratio: f64,
}

impl Simulation {
    /// Builds the initial state: seeds existing-built density, plants centre
    /// seeds and their accessibility, and computes the green periphery/access.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mut state: Array2<i16>,
        mut origin: Array2<i16>,
        mut density: Array2<f32>,
        centre_seeds: &[(usize, usize)],
        params: Params,
        exist_built_per_block: f32,
        total_iters: usize,
        master_seed: u64,
    ) -> Result<Self, String> {
        let dim = state.dim();
        if origin.dim() != dim || density.dim() != dim {
            return Err("state, origin and density must share the same shape".to_string());
        }
        // existing built cells get the baseline density (before centres are seeded)
        for ((y, x), &s) in state.indexed_iter() {
            if s > 0 {
                density[[y, x]] = exist_built_per_block;
            }
        }
        // seed centres and aggregate their accessibility
        let mut cent_acc = Array2::<i32>::zeros(dim);
        let cent_opts = DijkstraOpts::new(params.max_distance_m, params.granularity_m);
        for &(r, c) in centre_seeds {
            if r >= dim.0 || c >= dim.1 {
                return Err("centre seed falls outside the grid".to_string());
            }
            state[[r, c]] = 2;
            origin[[r, c]] = 2;
            cent_acc =
                cent_acc + agg_dijkstra_cont(&state, r, c, &[0, 1, 2], &[0, 1, 2], &cent_opts);
        }
        let (green_itx, green_acc) =
            prepare_green_arrs(&state, params.max_distance_m, params.granularity_m);
        let pop_target_ratio = density.sum() as f64 / params.max_populat;
        Ok(Simulation {
            state,
            origin,
            density,
            green_itx,
            green_acc,
            cent_acc,
            params,
            total_iters,
            current_iter: 0,
            master_seed,
            pop_target_ratio,
        })
    }

    fn assign_density(&mut self, rng: &mut rand_chacha::ChaCha8Rng, y: usize, x: usize) {
        self.density[[y, x]] = random_density(
            rng,
            self.params.prob_distribution,
            self.params.high_per_block,
            self.params.med_per_block,
            self.params.low_per_block,
        ) as f32;
    }

    fn plant_centre(&mut self, y: usize, x: usize) {
        self.state[[y, x]] = 2;
        let opts = DijkstraOpts::new(self.params.max_distance_m, self.params.granularity_m);
        let inc = agg_dijkstra_cont(&self.state, y, x, &[0, 1, 2], &[0, 1, 2], &opts);
        self.cent_acc = &self.cent_acc + &inc;
    }

    /// Runs a single iteration. RNG is seeded from `(master_seed, current_iter)`
    /// and consumed in a fixed sequence (shuffle, then per-cell), so the result is
    /// deterministic and independent of any outer parallelism.
    pub fn step(&mut self) {
        use rand::Rng;
        self.current_iter += 1;
        let (rows, cols) = self.state.dim();
        let mut rng = rng_for(self.master_seed, self.current_iter as u64);
        let mut centrality_this_iter = false;

        // shuffle the visiting order (Fisher-Yates with the iteration RNG)
        let mut idxs: Vec<(usize, usize)> = (0..rows)
            .flat_map(|y| (0..cols).map(move |x| (y, x)))
            .collect();
        for i in (1..idxs.len()).rev() {
            let j = (rng.gen::<f64>() * ((i + 1) as f64)) as usize;
            idxs.swap(i, j.min(i));
        }

        let old_state = self.state.clone();
        let p = self.params;
        for (y, x) in idxs {
            if self.state[[y, x]] != 0 {
                continue;
            }
            // preserve intentionally-fixed (origin) green space
            if self.origin[[y, x]] == 0 {
                continue;
            }
            if self.green_itx[[y, x]] == 2 {
                // require at least one urban neighbour (no double steps)
                let (tot_urban, _, _) = count_cont_nbs(&old_state, y, x, &[1, 2]);
                if tot_urban == 0 {
                    continue;
                }
                if self.cent_acc[[y, x]] > 0 {
                    if rng.gen::<f64>() < p.build_prob {
                        if let Some((ni, na)) = green_to_built(
                            y,
                            x,
                            &self.state,
                            &self.green_itx,
                            &self.green_acc,
                            p.granularity_m,
                            p.max_distance_m,
                            p.min_green_span_m,
                        ) {
                            self.state[[y, x]] = 1;
                            self.green_itx = ni;
                            self.green_acc = na;
                            self.assign_density(&mut rng, y, x);
                        }
                    }
                } else if !centrality_this_iter
                    && self.pop_target_ratio <= p.pop_target_cent_threshold
                    && rng.gen::<f64>() < p.cent_prob_nb
                {
                    if let Some((ni, na)) = green_to_built(
                        y,
                        x,
                        &self.state,
                        &self.green_itx,
                        &self.green_acc,
                        p.granularity_m,
                        p.max_distance_m,
                        p.min_green_span_m,
                    ) {
                        self.green_itx = ni;
                        self.green_acc = na;
                        self.plant_centre(y, x);
                        self.assign_density(&mut rng, y, x);
                        centrality_this_iter = true;
                    }
                }
            } else if !centrality_this_iter
                && self.state[[y, x]] == 0
                && self.pop_target_ratio <= p.pop_target_cent_threshold
                && rng.gen::<f64>() < p.cent_prob_isol
            {
                if let Some((ni, na)) = green_to_built(
                    y,
                    x,
                    &self.state,
                    &self.green_itx,
                    &self.green_acc,
                    p.granularity_m,
                    p.max_distance_m,
                    p.min_green_span_m,
                ) {
                    self.green_itx = ni;
                    self.green_acc = na;
                    self.plant_centre(y, x);
                    self.assign_density(&mut rng, y, x);
                    centrality_this_iter = true;
                }
            }
        }
        self.pop_target_ratio = self.density.sum() as f64 / p.max_populat;
    }

    /// Runs up to `total_iters` iterations, stopping early once the population
    /// target is reached. No-op if the starting population already meets it.
    pub fn run(&mut self) {
        if self.pop_target_ratio >= 1.0 {
            return;
        }
        for _ in 0..self.total_iters {
            self.step();
            if self.pop_target_ratio >= 1.0 {
                break;
            }
        }
    }

    pub fn population(&self) -> f64 {
        self.density.sum() as f64
    }
}

/// Runs `n_members` independent simulations from the same initial `template`,
/// each with its own deterministic seed, across all available cores. Returns the
/// final `state` grid of each member (the basis for probability-of-development
/// maps). Output is independent of thread count.
pub fn run_ensemble(template: &Simulation, base_seed: u64, n_members: usize) -> Vec<Array2<i16>> {
    (0..n_members)
        .into_par_iter()
        .map(|member| {
            let mut sim = template.clone();
            sim.master_seed = splitmix64(base_seed ^ splitmix64(member as u64));
            sim.current_iter = 0;
            sim.run();
            sim.state
        })
        .collect()
}

/// Runs `n_members` independent simulations from `template` in parallel and
/// returns, per cell, the fraction of members in which it ended urban
/// (`state > 0`) — a probability-of-development surface in `[0, 1]`. The result
/// is independent of thread count (integer-count reduction, per-member seeding).
pub fn ensemble_probability(
    template: &Simulation,
    base_seed: u64,
    n_members: usize,
) -> Array2<f32> {
    let dim = template.state.dim();
    if n_members == 0 {
        return Array2::<f32>::zeros(dim);
    }
    let counts = (0..n_members)
        .into_par_iter()
        .map(|member| {
            let mut sim = template.clone();
            sim.master_seed = splitmix64(base_seed ^ splitmix64(member as u64));
            sim.current_iter = 0;
            sim.run();
            sim.state.mapv(|v| u32::from(v > 0))
        })
        .reduce(|| Array2::<u32>::zeros(dim), |a, b| a + b);
    counts.mapv(|c| c as f32 / n_members as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn growth_params() -> Params {
        Params::from_raw(
            100.0,           // granularity_m
            600.0,           // max_distance_m
            1_000_000.0,     // max_populat (high, so tests never short-circuit on target)
            100.0,           // min_green_span_m (1 block -> the span check never blocks)
            0.6,             // build_prob
            0.1,             // cent_prob_nb
            0.0,             // cent_prob_isol
            0.8,             // pop_target_cent_threshold
            (0.4, 0.4, 0.2), // prob distribution
            (6000.0, 3000.0, 1000.0),
        )
        .unwrap()
    }

    fn seeded_sim(grid: usize, seed: u64) -> Simulation {
        // an all-green grid with a single existing centre seed near the middle
        let state = Array2::<i16>::zeros((grid, grid));
        let origin = Array2::<i16>::from_elem((grid, grid), -1); // -1 origin => not fixed green
        let density = Array2::<f32>::zeros((grid, grid));
        Simulation::new(
            state,
            origin,
            density,
            &[(grid / 2, grid / 2)],
            growth_params(),
            0.0,
            25,
            seed,
        )
        .unwrap()
    }

    #[test]
    fn rejects_bad_prob_distribution() {
        let err = Params::from_raw(
            100.0,
            800.0,
            1.0,
            800.0,
            0.1,
            0.0,
            0.0,
            0.8,
            (0.5, 0.4, 0.2),
            (3.0, 2.0, 1.0),
        );
        assert!(err.is_err());
    }

    #[test]
    fn rejects_non_descending_densities() {
        let err = Params::from_raw(
            100.0,
            800.0,
            1.0,
            800.0,
            0.1,
            0.0,
            0.0,
            0.8,
            (0.4, 0.4, 0.2),
            (1.0, 2.0, 3.0),
        );
        assert!(err.is_err());
    }

    #[test]
    fn construction_seeds_centre() {
        let sim = seeded_sim(12, 1);
        assert_eq!(sim.state[[6, 6]], 2);
        assert!(sim.cent_acc.sum() > 0);
    }

    #[test]
    fn same_seed_is_reproducible() {
        let mut a = seeded_sim(30, 99);
        let mut b = seeded_sim(30, 99);
        a.run();
        b.run();
        assert_eq!(a.state, b.state);
        assert_eq!(a.density, b.density);
    }

    #[test]
    fn ensemble_is_thread_count_independent() {
        let template = seeded_sim(30, 7);
        let run_with = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| run_ensemble(&template, 2024, 8))
        };
        let one = run_with(1);
        let many = run_with(4);
        assert_eq!(one, many);
    }

    #[test]
    fn growth_actually_occurs() {
        let mut sim = seeded_sim(30, 5);
        let before = sim.population();
        sim.run();
        assert!(
            sim.population() > before,
            "expected growth: before={before}, after={}",
            sim.population()
        );
    }

    #[test]
    fn different_seeds_diverge() {
        let template = seeded_sim(40, 11);
        let results = run_ensemble(&template, 123, 6);
        // at least two members should differ given independent seeds
        let all_same = results.iter().all(|r| *r == results[0]);
        assert!(!all_same);
    }

    #[test]
    fn ensemble_probability_is_unit_range_and_thread_independent() {
        let template = seeded_sim(30, 7);
        let prob = ensemble_probability(&template, 2024, 6);
        assert_eq!(prob.dim(), template.state.dim());
        assert!(prob.iter().all(|&p| (0.0..=1.0).contains(&p)));
        // the seeded centre is urban in every member -> probability 1.0
        assert_eq!(prob[[15, 15]], 1.0);
        // identical regardless of how many threads run the members
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let single = pool.install(|| ensemble_probability(&template, 2024, 6));
        assert_eq!(prob, single);
    }
}
