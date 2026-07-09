//! Quick timing harness for a representative single-scenario run.
//!
//! Run with: `cargo run --release --manifest-path core/Cargo.toml --example bench`
//! It is deliberately simple (median of a few runs) so we can measure before and
//! after optimisation rather than guess.

use isobenefit::sim::{Params, Simulation};
use ndarray::Array2;
use std::time::Instant;

fn make_sim(rows: usize, cols: usize, seed: u64) -> Simulation {
    let state = Array2::<i16>::zeros((rows, cols));
    let origin = Array2::<i16>::from_elem((rows, cols), -1);
    let density = Array2::<f32>::zeros((rows, cols));
    let params = Params::from_raw(
        100.0,       // granularity_m
        800.0,       // max_distance_m
        5_000_000.0, // max_populat (high so it never stops early)
        100.0,       // min_green_span_m
        0.25,        // build_prob
        0.05,        // cent_prob_nb
        0.0,         // cent_prob_isol
        0.8,         // pop_target_cent_threshold
        (0.4, 0.4, 0.2),
        (6000.0, 3000.0, 1000.0),
    )
    .unwrap();
    Simulation::new(
        state,
        origin,
        density,
        &[(rows / 2, cols / 2)],
        params,
        100,
        seed,
    )
    .unwrap()
}

fn main() {
    let (rows, cols) = (130, 105); // Cambourne-sized
                                   // construction (includes prepare_green_arrs)
    let t = Instant::now();
    let _warm = make_sim(rows, cols, 1);
    println!("construct {rows}x{cols}: {:?}", t.elapsed());

    let runs = 3;
    let mut times = Vec::new();
    let mut last_pop = 0.0;
    let mut last_iters = 0;
    for r in 0..runs {
        let mut sim = make_sim(rows, cols, 1000 + r as u64);
        let t = Instant::now();
        sim.run();
        times.push(t.elapsed());
        last_pop = sim.population();
        last_iters = sim.current_iter;
    }
    times.sort();
    println!(
        "run {rows}x{cols}, {last_iters} iters: median {:?} (min {:?}, max {:?}); final pop {last_pop:.0}",
        times[times.len() / 2],
        times[0],
        times[times.len() - 1],
    );
}
