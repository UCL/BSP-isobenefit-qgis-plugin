//! The Isobenefit growth simulation: state machine, the `try_build` decision,
//! and the parallel `run_ensemble`.
//!
//! State grid values: -1 unbuildable, 0 nature/green, 1 built, 2 centre.
//! All GIS concerns (rasterization, CRS, IO) live in the QGIS plugin; this
//! module only sees numpy-shaped integer/float grids.

use crate::access::{
    agg_dijkstra_cont, agg_dijkstra_dist, park_threshold_cells, prepare_park_arrs, DijkstraOpts,
};
use crate::density::{random_density, rng_for, splitmix64};
use crate::neighbours::{count_cont_nbs, green_spans, iter_nbs};
use ndarray::Array2;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

/// The default minimum park area: 2 hectares, Natural England's accessible
/// natural greenspace standard.
pub const DEFAULT_MIN_PARK_AREA_M2: f64 = 20_000.0;

/// Scalar simulation parameters (densities already converted to per-block).
#[derive(Clone, Copy)]
pub struct Params {
    pub granularity_m: f64,
    /// The centre walk: centre accessibility (building and seeding) is bounded here.
    pub centre_distance_m: f64,
    /// The green walk: the green-access guard in `try_build` is bounded here.
    pub green_distance_m: f64,
    pub max_populat: f64,
    /// Minimum green span, in **metres**, consistent
    /// throughout. (An earlier numba port of this project mixed metres and km
    /// here; the published simulator works in cells and never had the bug.)
    pub min_green_span_m: f64,
    /// Minimum park area in square metres: the size a green component must hold
    /// to qualify as a park. Defaults to [`DEFAULT_MIN_PARK_AREA_M2`]
    /// (2 ha) when the caller passes none.
    pub min_park_area_m2: f64,
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
        centre_distance_m: f64,
        green_distance_m: f64,
        max_populat: f64,
        min_green_span_m: f64,
        build_prob: f64,
        cent_prob_nb: f64,
        cent_prob_isol: f64,
        pop_target_cent_threshold: f64,
        prob_distribution: (f64, f64, f64),
        density_factors_km2: (f64, f64, f64),
        min_park_area_m2: Option<f64>,
    ) -> Result<Params, String> {
        let min_park_area_m2 = min_park_area_m2.unwrap_or(DEFAULT_MIN_PARK_AREA_M2);
        let all_inputs = [
            granularity_m,
            centre_distance_m,
            green_distance_m,
            max_populat,
            min_green_span_m,
            min_park_area_m2,
            build_prob,
            cent_prob_nb,
            cent_prob_isol,
            pop_target_cent_threshold,
            prob_distribution.0,
            prob_distribution.1,
            prob_distribution.2,
            density_factors_km2.0,
            density_factors_km2.1,
            density_factors_km2.2,
        ];
        if all_inputs.iter().any(|v| !v.is_finite()) {
            return Err("All parameters must be finite numbers".to_string());
        }
        if granularity_m <= 0.0 {
            return Err("granularity_m must be positive".to_string());
        }
        if centre_distance_m < granularity_m {
            return Err("centre_distance_m must be at least one block (granularity_m)".to_string());
        }
        if green_distance_m < granularity_m {
            return Err("green_distance_m must be at least one block (granularity_m)".to_string());
        }
        if min_green_span_m < 0.0 {
            return Err("min_green_span_m must not be negative".to_string());
        }
        if min_park_area_m2 < 0.0 {
            return Err("min_park_area_m2 must not be negative".to_string());
        }
        for (name, p) in [
            ("build_prob", build_prob),
            ("cent_prob_nb", cent_prob_nb),
            ("cent_prob_isol", cent_prob_isol),
            ("pop_target_cent_threshold", pop_target_cent_threshold),
        ] {
            if !(0.0..=1.0).contains(&p) {
                return Err(format!("{name} must lie in [0, 1]"));
            }
        }
        if prob_distribution.0 < 0.0 || prob_distribution.1 < 0.0 || prob_distribution.2 < 0.0 {
            return Err("prob_distribution components must not be negative".to_string());
        }
        let prob_sum = ((prob_distribution.0 + prob_distribution.1 + prob_distribution.2) * 100.0)
            .round()
            / 100.0;
        if (prob_sum - 1.0).abs() > f64::EPSILON {
            return Err("The prob_distribution parameter must sum to 1.".to_string());
        }
        if !(density_factors_km2.0 > density_factors_km2.1
            && density_factors_km2.1 > density_factors_km2.2)
        {
            return Err("Density factors should be in descending order".to_string());
        }
        if density_factors_km2.2 <= 0.0 {
            return Err("Density factors must be positive".to_string());
        }
        if max_populat <= 0.0 {
            return Err("The population target must be positive".to_string());
        }
        let block = granularity_m * granularity_m / 1.0e6;
        Ok(Params {
            granularity_m,
            centre_distance_m,
            green_distance_m,
            max_populat,
            min_green_span_m,
            min_park_area_m2,
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

/// Attempt to convert green cell `(y, x)` to built land. Every check runs before
/// any mutation; on success `park` and `green_acc` are updated in place and the
/// caller sets the cell's state. The candidate is rejected when building it
/// would create a runaway streak, crimp a green corridor below the minimum
/// span, leave an adjacent park fragment below the park area (whether by
/// splitting the park or by building along its edge), or leave the new home,
/// or any built cell that currently has one (the existing fabric included),
/// without a park within the green walk.
#[allow(clippy::too_many_arguments)]
pub fn try_build(
    y: usize,
    x: usize,
    state: &Array2<i16>,
    park: &mut Array2<bool>,
    green_acc: &mut Array2<i32>,
    granularity_m: f64,
    green_distance_m: f64,
    min_green_span_m: f64,
    min_park_area_m2: f64,
) -> bool {
    let (rows, cols) = state.dim();

    let (_tot, longest_urban_run, _urban_regions) = count_cont_nbs(state, y, x, &[1, 2]);
    // a single urban neighbour is only allowed if that neighbour is a centrality
    if longest_urban_run == 1 {
        let (_t, longest_cent_run, _r) = count_cont_nbs(state, y, x, &[2]);
        if longest_cent_run != 1 {
            return false;
        }
    }
    // don't crimp a green corridor below the minimum span
    if !green_spans(state, y, x, granularity_m, min_green_span_m) {
        return false;
    }

    // no adjacent park fragment may fall below the park area. A connected green
    // region of >= N cells always shows N cells within N-1 rook steps of any
    // member, so bounding the search at N steps is exact and terminates fast.
    // Sub-park green (never promised to anyone) is exempt and stays consumable.
    let threshold = park_threshold_cells(min_park_area_m2, granularity_m);
    let mut frag_opts = DijkstraOpts::new(threshold as f64 * granularity_m, granularity_m);
    frag_opts.break_count = Some(threshold);
    frag_opts.rook = true; // rook only, else diagonal hops cheat the contiguity
    frag_opts.exclude = Some((y, x)); // probe with the candidate treated as built
    for (ny, nx) in iter_nbs(rows, cols, y, x, true) {
        if state[[ny, nx]] != 0 || !park[[ny, nx]] {
            continue;
        }
        let frag = agg_dijkstra_cont(state, ny, nx, &[0], &[0], &frag_opts);
        if (frag.sum() as i64) < threshold {
            return false;
        }
    }

    if !park[[y, x]] {
        // no park cell is removed, so no one's access changes; the new home
        // itself must already reach a park
        return green_acc[[y, x]] > 0;
    }

    // the cell leaves the park: subtract its footprint — every cell it served.
    // Footprints are static (built land stays traversable, unbuildable land
    // never changes), so this decrement is exact.
    let acc_opts = DijkstraOpts::new(green_distance_m, granularity_m);
    let dec = agg_dijkstra_cont(state, y, x, &[0, 1, 2], &[0, 1, 2], &acc_opts);
    let reach = (green_distance_m / granularity_m).ceil() as usize + 1;
    let (y0, y1) = (y.saturating_sub(reach), (y + reach + 1).min(rows));
    let (x0, x1) = (x.saturating_sub(reach), (x + reach + 1).min(cols));
    for y2 in y0..y1 {
        for x2 in x0..x1 {
            if dec[[y2, x2]] == 0 {
                continue;
            }
            let left = green_acc[[y2, x2]] - 1;
            if (y2, x2) == (y, x) {
                // the new home loses its own park cell; another must remain
                if left <= 0 {
                    return false;
                }
            } else if state[[y2, x2]] > 0 && left <= 0 {
                // a served built cell would be stranded (cells that never had
                // access have acc 0 and no footprint covers them)
                return false;
            }
        }
    }
    for y2 in y0..y1 {
        for x2 in x0..x1 {
            green_acc[[y2, x2]] -= dec[[y2, x2]];
        }
    }
    park[[y, x]] = false;
    true
}

/// The full simulation state. Construct via [`Simulation::new`], then drive with
/// [`Simulation::step`] / [`Simulation::run`].
#[derive(Clone)]
pub struct Simulation {
    pub state: Array2<i16>,
    pub origin: Array2<i16>,
    pub density: Array2<f32>,
    /// Green cells belonging to a park-qualifying component (the guard's
    /// destination set). Only ever loses cells, one per accepted build.
    pub park: Array2<bool>,
    /// Per cell: how many park cells lie within the green walk.
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
        density: Array2<f32>,
        centre_seeds: &[(usize, usize)],
        params: Params,
        total_iters: usize,
        master_seed: u64,
    ) -> Result<Self, String> {
        let dim = state.dim();
        if origin.dim() != dim || density.dim() != dim {
            return Err("state, origin and density must share the same shape".to_string());
        }
        // Existing built fabric carries no density and no population: it is assumed to be served by
        // its own centres, so it is spatial context only. Only new development is counted, so the
        // population target is a new-only target. Its cells stay at density 0 here (they are never
        // re-visited: `step` skips any cell whose state is already built).
        // seed centres and aggregate their accessibility
        let mut cent_acc = Array2::<i32>::zeros(dim);
        let cent_opts = DijkstraOpts::new(params.centre_distance_m, params.granularity_m);
        for &(r, c) in centre_seeds {
            if r >= dim.0 || c >= dim.1 {
                return Err("centre seed falls outside the grid".to_string());
            }
            if state[[r, c]] < 0 {
                return Err("centre seed falls on unbuildable land".to_string());
            }
            state[[r, c]] = 2;
            origin[[r, c]] = 2;
            cent_acc =
                cent_acc + agg_dijkstra_cont(&state, r, c, &[0, 1, 2], &[0, 1, 2], &cent_opts);
        }
        let (park, green_acc) = prepare_park_arrs(
            &state,
            params.green_distance_m,
            params.granularity_m,
            params.min_park_area_m2,
        );
        let pop_target_ratio = density.sum() as f64 / params.max_populat;
        Ok(Simulation {
            state,
            origin,
            density,
            park,
            green_acc,
            cent_acc,
            params,
            total_iters,
            current_iter: 0,
            master_seed,
            pop_target_ratio,
        })
    }

    fn assign_density(&mut self, y: usize, x: usize, rng: &mut ChaCha8Rng) {
        // Every new block is built at one of three density tiers, drawn at the configured
        // probabilities. This is real population accounting: the run stops once the drawn densities
        // reach the (new-only) target. Post-processing later re-arranges these values spatially so
        // the highest sit nearest the FINAL (post-processed) mixed-use centres — assigning the
        // arrangement here would measure distances against centres that later steps then move, add
        // or cull, so the placement is a post-processing product; only the mix is fixed now.
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
        let opts = DijkstraOpts::new(self.params.centre_distance_m, self.params.granularity_m);
        // the distance field gives the access footprint (finite == reachable within a walk,
        // matching the old path==target==[0,1,2] agg)
        let d = agg_dijkstra_dist(&self.state, y, x, &[0, 1, 2], &opts);
        let inc = d.mapv(|v| if v.is_finite() { 1 } else { 0 });
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
            // candidacy against the pre-iteration state, so a cell built this
            // iteration cannot seed its neighbours in the same pass
            let attached = iter_nbs(rows, cols, y, x, true)
                .into_iter()
                .any(|(ny, nx)| old_state[[ny, nx]] > 0);
            if attached {
                if self.cent_acc[[y, x]] > 0 {
                    if rng.gen::<f64>() < p.build_prob
                        && try_build(
                            y,
                            x,
                            &self.state,
                            &mut self.park,
                            &mut self.green_acc,
                            p.granularity_m,
                            p.green_distance_m,
                            p.min_green_span_m,
                            p.min_park_area_m2,
                        )
                    {
                        self.state[[y, x]] = 1;
                        self.assign_density(y, x, &mut rng);
                    }
                } else if !centrality_this_iter
                    && self.pop_target_ratio <= p.pop_target_cent_threshold
                    && rng.gen::<f64>() < p.cent_prob_nb
                    && try_build(
                        y,
                        x,
                        &self.state,
                        &mut self.park,
                        &mut self.green_acc,
                        p.granularity_m,
                        p.green_distance_m,
                        p.min_green_span_m,
                        p.min_park_area_m2,
                    )
                {
                    self.plant_centre(y, x);
                    self.assign_density(y, x, &mut rng);
                    centrality_this_iter = true;
                }
            } else if !centrality_this_iter
                && self.pop_target_ratio <= p.pop_target_cent_threshold
                && rng.gen::<f64>() < p.cent_prob_isol
                && try_build(
                    y,
                    x,
                    &self.state,
                    &mut self.park,
                    &mut self.green_acc,
                    p.granularity_m,
                    p.green_distance_m,
                    p.min_green_span_m,
                    p.min_park_area_m2,
                )
            {
                self.plant_centre(y, x);
                self.assign_density(y, x, &mut rng);
                centrality_this_iter = true;
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
/// `member_offset` is the global index of the first member: member `i` of this
/// call is seeded as global member `member_offset + i`, so callers may split one
/// logical ensemble into batches (for progress/cancellation) and still draw the
/// exact seed sequence of a single call, independent of batch size.
pub fn run_ensemble(
    template: &Simulation,
    base_seed: u64,
    n_members: usize,
    member_offset: usize,
) -> Vec<Array2<i16>> {
    (member_offset..member_offset + n_members)
        .into_par_iter()
        .map(|member| {
            let mut sim = member_simulation(template, base_seed, member);
            sim.run();
            sim.state
        })
        .collect()
}

/// The simulation for global ensemble member `member`: a clone of `template`
/// re-seeded exactly as `run_ensemble` seeds that member, ready to run. Lets a
/// caller re-run one member deterministically (e.g. to recover the drawn density
/// grid of the selected run, which the ensemble does not retain).
pub fn member_simulation(template: &Simulation, base_seed: u64, member: usize) -> Simulation {
    let mut sim = template.clone();
    sim.master_seed = splitmix64(base_seed ^ splitmix64(member as u64));
    sim.current_iter = 0;
    sim
}

/// Runs `n_members` independent simulations from `template` in parallel and
/// returns, per cell, **counts** of how many members ended in each class:
/// `(built, green, centre)` for `state == 1 / 0 / 2`. The reduction is integer
/// sums, so the result is independent of thread count; callers divide by
/// `n_members` for per-class probabilities. P(green) reveals the robust green
/// network; smoothed P(centre) reveals natural centre locations.
pub fn ensemble_class_counts(
    template: &Simulation,
    base_seed: u64,
    n_members: usize,
) -> (Array2<u32>, Array2<u32>, Array2<u32>) {
    let dim = template.state.dim();
    if n_members == 0 {
        return (Array2::zeros(dim), Array2::zeros(dim), Array2::zeros(dim));
    }
    (0..n_members)
        .into_par_iter()
        .map(|member| {
            let mut sim = template.clone();
            sim.master_seed = splitmix64(base_seed ^ splitmix64(member as u64));
            sim.current_iter = 0;
            sim.run();
            (
                sim.state.mapv(|v| u32::from(v == 1)),
                sim.state.mapv(|v| u32::from(v == 0)),
                sim.state.mapv(|v| u32::from(v == 2)),
            )
        })
        .reduce(
            || {
                (
                    Array2::<u32>::zeros(dim),
                    Array2::<u32>::zeros(dim),
                    Array2::<u32>::zeros(dim),
                )
            },
            |(b1, g1, c1), (b2, g2, c2)| (b1 + b2, g1 + g2, c1 + c2),
        )
}

/// Probability that each cell ends urban (`state > 0`) across `n_members` runs —
/// a convenience over [`ensemble_class_counts`] (built + centre).
pub fn ensemble_probability(
    template: &Simulation,
    base_seed: u64,
    n_members: usize,
) -> Array2<f32> {
    if n_members == 0 {
        return Array2::<f32>::zeros(template.state.dim());
    }
    let (built, _green, centre) = ensemble_class_counts(template, base_seed, n_members);
    (built + centre).mapv(|c| c as f32 / n_members as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn growth_params() -> Params {
        Params::from_raw(
            100.0,           // granularity_m
            600.0,           // centre_distance_m
            600.0,           // green_distance_m
            1_000_000.0,     // max_populat (high, so tests never short-circuit on target)
            100.0,           // min_green_span_m (1 block -> the span check never blocks)
            0.6,             // build_prob
            0.1,             // cent_prob_nb
            0.0,             // cent_prob_isol
            0.8,             // pop_target_cent_threshold
            (0.4, 0.4, 0.2), // prob distribution
            (6000.0, 3000.0, 1000.0),
            None, // min_park_area_m2 -> the 2 ha default
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
            400.0,
            1.0,
            800.0,
            0.1,
            0.0,
            0.0,
            0.8,
            (0.5, 0.4, 0.2),
            (3.0, 2.0, 1.0),
            None,
        );
        assert!(err.is_err());
    }

    #[test]
    fn rejects_non_descending_densities() {
        let err = Params::from_raw(
            100.0,
            800.0,
            400.0,
            1.0,
            800.0,
            0.1,
            0.0,
            0.0,
            0.8,
            (0.4, 0.4, 0.2),
            (1.0, 2.0, 3.0),
            None,
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
            pool.install(|| run_ensemble(&template, 2024, 8, 0))
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
        let results = run_ensemble(&template, 123, 6, 0);
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

    #[test]
    fn density_is_drawn_from_the_three_tiers() {
        // every new-built cell is built at one of the three density tiers (the spatial
        // arrangement is a post-processing product; the mix is fixed here)
        let mut sim = seeded_sim(40, 5);
        sim.run();
        let tiers = [
            sim.params.high_per_block as f32,
            sim.params.med_per_block as f32,
            sim.params.low_per_block as f32,
        ];
        let mut checked = 0;
        for (&s, &d) in sim.state.iter().zip(sim.density.iter()) {
            if s == 1 {
                assert!(
                    tiers.contains(&d),
                    "density {d} is not one of the tiers {tiers:?}"
                );
                checked += 1;
            }
        }
        assert!(checked > 0, "expected some built cells");
    }

    #[test]
    fn existing_built_is_not_counted_in_population() {
        // seed a grid with existing built cells (state 1) already present; they must carry no
        // density, so the population target is a new-only target.
        let grid = 20;
        let mut state = Array2::<i16>::zeros((grid, grid));
        state[[0, 0]] = 1; // an existing built cell
        state[[0, 1]] = 1;
        let origin = Array2::<i16>::from_elem((grid, grid), -1);
        let density = Array2::<f32>::zeros((grid, grid));
        let sim = Simulation::new(
            state,
            origin,
            density,
            &[(grid / 2, grid / 2)],
            growth_params(),
            25,
            3,
        )
        .unwrap();
        assert_eq!(sim.density[[0, 0]], 0.0);
        assert_eq!(sim.density[[0, 1]], 0.0);
        assert_eq!(sim.population(), 0.0);
    }

    #[test]
    fn batched_ensemble_equals_single_call() {
        // splitting one ensemble into batches via member_offset draws the exact
        // same seed sequence as a single call, whatever the batch size
        let template = seeded_sim(30, 7);
        let whole = run_ensemble(&template, 99, 6, 0);
        let mut batched = run_ensemble(&template, 99, 2, 0);
        batched.extend(run_ensemble(&template, 99, 3, 2));
        batched.extend(run_ensemble(&template, 99, 1, 5));
        assert_eq!(whole, batched);
    }

    #[test]
    fn tier_draw_proportions_follow_probabilities() {
        // over many draws, a mixed (0.5, 0.3, 0.2) distribution should yield roughly
        // those tier shares (loose tolerance; the draw is plain inverse-CDF sampling)
        use crate::density::{random_density, rng_for};
        let mut rng = rng_for(7, 1);
        let (mut hi, mut med, mut lo) = (0u32, 0u32, 0u32);
        let n = 20_000;
        for _ in 0..n {
            let d = random_density(&mut rng, (0.5, 0.3, 0.2), 9.0, 4.0, 1.0);
            if d > 8.0 {
                hi += 1;
            } else if d > 3.0 {
                med += 1;
            } else {
                lo += 1;
            }
        }
        let f = |c: u32| c as f64 / n as f64;
        assert!((f(hi) - 0.5).abs() < 0.02, "high share {}", f(hi));
        assert!((f(med) - 0.3).abs() < 0.02, "med share {}", f(med));
        assert!((f(lo) - 0.2).abs() < 0.02, "low share {}", f(lo));
    }

    #[test]
    fn rejects_non_positive_population_target() {
        let err = Params::from_raw(
            100.0,
            800.0,
            400.0,
            0.0,
            800.0,
            0.1,
            0.0,
            0.0,
            0.8,
            (0.4, 0.4, 0.2),
            (3.0, 2.0, 1.0),
            None,
        );
        assert!(err.is_err());
    }

    #[test]
    fn incremental_green_arrays_match_fresh_rebuild() {
        // golden consistency: after growth over a grid with an unbuildable strip
        // and isolated-centre seeding, the incrementally maintained green arrays
        // must equal a fresh rebuild on the final state, cell for cell
        let grid = 24;
        let mut state = Array2::<i16>::zeros((grid, grid));
        for y in 0..grid {
            state[[y, 10]] = -1; // a river/motorway strip
        }
        let origin = Array2::<i16>::from_elem((grid, grid), -1);
        let density = Array2::<f32>::zeros((grid, grid));
        let params = Params::from_raw(
            100.0,
            600.0,
            600.0,
            1_000_000.0,
            100.0,
            0.6,
            0.1,
            0.01, // isolated centres ON: exercises the itx-0 planting branch
            0.8,
            (0.4, 0.4, 0.2),
            (6000.0, 3000.0, 1000.0),
            None,
        )
        .unwrap();
        let mut sim = Simulation::new(state, origin, density, &[(12, 4)], params, 40, 42).unwrap();
        sim.run();
        assert!(sim.population() > 0.0, "expected growth");
        let (fresh_park, fresh_acc) = prepare_park_arrs(
            &sim.state,
            params.green_distance_m,
            params.granularity_m,
            params.min_park_area_m2,
        );
        assert_eq!(sim.park, fresh_park);
        assert_eq!(sim.green_acc, fresh_acc);
        // the strip itself must never be built on or counted as park
        for y in 0..grid {
            assert_eq!(sim.state[[y, 10]], -1);
            assert!(!sim.park[[y, 10]]);
        }
    }

    #[test]
    fn rejects_build_that_leaves_itself_without_park_access() {
        // 1x5 row: green | built | built | centre | candidate green. Threshold 1
        // (span == granularity), so both single green cells are parks. The only
        // other park (index 0) is 400m from the candidate, beyond the 300m green
        // walk, so building index 4 leaves the new home without a park.
        let state = Array2::from_shape_vec((1, 5), vec![0i16, 1, 1, 2, 0]).unwrap();
        let (mut park, mut acc) = prepare_park_arrs(&state, 300.0, 100.0, 10_000.0);
        assert!(!try_build(
            0, 4, &state, &mut park, &mut acc, 100.0, 300.0, 100.0, 10_000.0
        ));
        // rejection must leave the arrays untouched
        let (park2, acc2) = prepare_park_arrs(&state, 300.0, 100.0, 10_000.0);
        assert_eq!(park, park2);
        assert_eq!(acc, acc2);
        // the built cells themselves keep access via index 0
        for x in 1..4 {
            assert!(acc[[0, x]] > 0);
        }
    }

    #[test]
    fn green_guard_uses_green_walk_not_the_centre_walk() {
        // as above at a 300m green walk the build is rejected; rebuilt at a 600m
        // green walk, the far park is within reach and the same move is accepted
        let state = Array2::from_shape_vec((1, 5), vec![0i16, 1, 1, 2, 0]).unwrap();
        let (mut park, mut acc) = prepare_park_arrs(&state, 300.0, 100.0, 10_000.0);
        assert!(!try_build(
            0, 4, &state, &mut park, &mut acc, 100.0, 300.0, 100.0, 10_000.0
        ));
        let (mut park6, mut acc6) = prepare_park_arrs(&state, 600.0, 100.0, 10_000.0);
        assert!(try_build(
            0, 4, &state, &mut park6, &mut acc6, 100.0, 600.0, 100.0, 10_000.0
        ));
        // the accepted build removed the candidate from the park set and its
        // footprint from the counts
        assert!(!park6[[0, 4]]);
        assert_eq!(acc6[[0, 4]], 1); // index 0 still covers it at 400m
    }

    #[test]
    fn sub_park_green_does_not_satisfy_the_guard() {
        // [-1, green, green, centre]: the 2-cell green region is below the park
        // threshold of 4 (span 200 at 100m cells), so although green sits right
        // next door, the candidate has no park within the walk and is rejected
        let state = Array2::from_shape_vec((1, 4), vec![-1i16, 0, 0, 2]).unwrap();
        let (mut park, mut acc) = prepare_park_arrs(&state, 300.0, 100.0, 40_000.0);
        assert!(!park[[0, 1]] && !park[[0, 2]]);
        assert!(!try_build(
            0, 2, &state, &mut park, &mut acc, 100.0, 300.0, 200.0, 40_000.0
        ));

        // extend the row with a 4-cell park within the walk: the same build now
        // passes, and consuming the sub-park green never touches the counts
        let state = Array2::from_shape_vec((1, 8), vec![-1i16, 0, 0, 2, 0, 0, 0, 0]).unwrap();
        let (mut park, mut acc) = prepare_park_arrs(&state, 300.0, 100.0, 40_000.0);
        assert!(!park[[0, 1]] && !park[[0, 2]] && park[[0, 4]]);
        let acc_before = acc.clone();
        assert!(try_build(
            0, 2, &state, &mut park, &mut acc, 100.0, 300.0, 200.0, 40_000.0
        ));
        assert_eq!(acc, acc_before);
    }

    #[test]
    fn parks_cannot_shrink_below_the_threshold() {
        // [-1, g, g, g, g, centre]: one park of exactly the threshold (4 cells at
        // span 200, 100m cells). Building any of its cells would leave a 3-cell
        // fragment, so the nibble is rejected even though every home keeps access.
        let state = Array2::from_shape_vec((1, 6), vec![-1i16, 0, 0, 0, 0, 2]).unwrap();
        let (mut park, mut acc) = prepare_park_arrs(&state, 300.0, 100.0, 40_000.0);
        assert!(park[[0, 4]]);
        assert!(!try_build(
            0, 4, &state, &mut park, &mut acc, 100.0, 300.0, 200.0, 40_000.0
        ));

        // with one spare cell (a 5-cell park) the same build passes and the
        // remaining fragment stays a valid park
        let state = Array2::from_shape_vec((1, 7), vec![-1i16, 0, 0, 0, 0, 0, 2]).unwrap();
        let (mut park, mut acc) = prepare_park_arrs(&state, 300.0, 100.0, 40_000.0);
        assert!(try_build(
            0, 5, &state, &mut park, &mut acc, 100.0, 300.0, 200.0, 40_000.0
        ));
        assert!(!park[[0, 5]]);
        let mut after = state.clone();
        after[[0, 5]] = 1;
        let (fresh_park, fresh_acc) = prepare_park_arrs(&after, 300.0, 100.0, 40_000.0);
        assert_eq!(park, fresh_park);
        assert_eq!(acc, fresh_acc);
    }

    #[test]
    fn park_area_parameter_decouples_from_the_span() {
        // the park test follows min_park_area_m2, not the corridor span: with a
        // 4-cell park area, a 2-cell green region is not a park regardless of span
        let state = Array2::from_shape_vec((1, 4), vec![-1i16, 0, 0, 2]).unwrap();
        let (park, acc) = prepare_park_arrs(&state, 300.0, 100.0, 40_000.0);
        assert!(!park[[0, 1]] && !park[[0, 2]]);
        assert_eq!(acc.sum(), 0);
        // and Params defaults the area to the 2 ha standard when none is given
        let p = growth_params();
        assert_eq!(p.min_park_area_m2, DEFAULT_MIN_PARK_AREA_M2);
    }

    #[test]
    fn rejects_seed_on_unbuildable() {
        let grid = 8;
        let mut state = Array2::<i16>::zeros((grid, grid));
        state[[4, 4]] = -1;
        let origin = Array2::<i16>::from_elem((grid, grid), -1);
        let density = Array2::<f32>::zeros((grid, grid));
        let err = Simulation::new(state, origin, density, &[(4, 4)], growth_params(), 5, 1);
        assert!(err.is_err());
    }

    #[test]
    fn rejects_non_finite_and_out_of_range_params() {
        let base = |build_prob: f64, granularity_m: f64, high: f64| {
            Params::from_raw(
                granularity_m,
                600.0,
                400.0,
                1000.0,
                100.0,
                build_prob,
                0.1,
                0.0,
                0.8,
                (0.4, 0.4, 0.2),
                (high, 3000.0, 1000.0),
                None,
            )
        };
        assert!(base(f64::NAN, 100.0, 6000.0).is_err()); // NaN probability
        assert!(base(1.5, 100.0, 6000.0).is_err()); // probability out of range
        assert!(base(0.5, 0.0, 6000.0).is_err()); // zero granularity
        assert!(base(0.5, 100.0, f64::NAN).is_err()); // NaN density passes no check silently
        assert!(base(0.5, 100.0, 6000.0).is_ok());
    }

    #[test]
    fn ensemble_class_counts_partition() {
        let template = seeded_sim(30, 7);
        let n = 8u32;
        let (built, green, centre) = ensemble_class_counts(&template, 2024, n as usize);
        // every cell ends in exactly one class, so the counts partition: each cell
        // sums to n (or 0 where permanently unbuildable — none here).
        let total = &built + &green + &centre;
        assert!(total.iter().all(|&t| t == 0 || t == n));
        // the seeded centre is a centre in every member
        assert_eq!(centre[[15, 15]], n);
    }
}
