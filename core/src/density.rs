//! Deterministic, parallel-safe RNG helpers and density draws.
//!
//! Reproducibility is by construction: each unit of work derives its own
//! `ChaCha8Rng` from `(master_seed, work_id)` via a SplitMix64 mix, so results
//! never depend on thread count or scheduling order.

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// SplitMix64 finalizer — cheap, well-distributed mixing of a 64-bit value.
#[inline]
pub fn splitmix64(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// An independent RNG stream for a given `(master_seed, stream)` pair.
#[inline]
pub fn rng_for(master_seed: u64, stream: u64) -> ChaCha8Rng {
    let mixed = splitmix64(master_seed ^ splitmix64(stream));
    ChaCha8Rng::seed_from_u64(mixed)
}

/// Picks a per-block density tier from `prob_distribution` (high, med, low).
///
/// `prob_distribution.0` is the high-density probability, `.1` medium; the
/// remainder is low. Mirrors the original `random_density`.
#[inline]
pub fn random_density(
    rng: &mut ChaCha8Rng,
    prob_distribution: (f64, f64, f64),
    high_per_block: f64,
    med_per_block: f64,
    low_per_block: f64,
) -> f64 {
    let p: f64 = rng.gen(); // [0, 1)
    if p < prob_distribution.0 {
        high_per_block
    } else if p < prob_distribution.0 + prob_distribution.1 {
        med_per_block
    } else {
        low_per_block
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_stream_is_reproducible() {
        let mut a = rng_for(42, 7);
        let mut b = rng_for(42, 7);
        let xs: Vec<f64> = (0..10).map(|_| a.gen::<f64>()).collect();
        let ys: Vec<f64> = (0..10).map(|_| b.gen::<f64>()).collect();
        assert_eq!(xs, ys);
    }

    #[test]
    fn different_streams_differ() {
        let mut a = rng_for(42, 1);
        let mut b = rng_for(42, 2);
        let xs: Vec<f64> = (0..10).map(|_| a.gen::<f64>()).collect();
        let ys: Vec<f64> = (0..10).map(|_| b.gen::<f64>()).collect();
        assert_ne!(xs, ys);
    }

    #[test]
    fn density_tiers_partition_unit_interval() {
        // with prob (1,0,0) every draw is high; (0,0,1) every draw is low.
        let mut rng = rng_for(1, 1);
        for _ in 0..100 {
            assert_eq!(
                random_density(&mut rng, (1.0, 0.0, 0.0), 9.0, 4.0, 1.0),
                9.0
            );
        }
        let mut rng = rng_for(1, 1);
        for _ in 0..100 {
            assert_eq!(
                random_density(&mut rng, (0.0, 0.0, 1.0), 9.0, 4.0, 1.0),
                1.0
            );
        }
    }
}
