//! Bounded accessibility search.
//!
//! `agg_dijkstra_cont` replaces the original O(N^2) "scan the whole pending grid
//! each step" search with a proper binary-heap Dijkstra bounded by `max_distance`.
//! `prepare_green_arrs` builds the green-periphery / green-access surfaces as a
//! parallel (rayon) map-reduce; the reduction is an integer sum, so the result is
//! identical regardless of how the work is split across threads.

use crate::neighbours::iter_nbs;
use ndarray::Array2;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Options for [`agg_dijkstra_cont`].
#[derive(Clone, Copy)]
pub struct DijkstraOpts {
    pub max_distance_m: f64,
    pub granularity_m: f64,
    /// Stop as soon as any target cell is found.
    pub break_first: bool,
    /// Stop once at least this many target cells are found.
    pub break_count: Option<i64>,
    /// Rook (orthogonal-only) traversal; otherwise queen (diagonals allowed).
    pub rook: bool,
}

impl DijkstraOpts {
    pub fn new(max_distance_m: f64, granularity_m: f64) -> Self {
        Self {
            max_distance_m,
            granularity_m,
            break_first: false,
            break_count: None,
            rook: false,
        }
    }
}

#[derive(Copy, Clone)]
struct HeapItem {
    dist: f64,
    y: usize,
    x: usize,
}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}
impl Eq for HeapItem {}
impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap; reverse on distance to pop the nearest first.
        other
            .dist
            .partial_cmp(&self.dist)
            .unwrap_or(Ordering::Equal)
    }
}

/// From `(y0, x0)`, returns a 0/1 grid marking every cell whose state is in
/// `target_state` that is reachable within `max_distance_m`, traversing only cells
/// whose state is in `path_state`. Distances use Euclidean steps scaled by
/// `granularity_m` (diagonal = sqrt(2) * granularity).
pub fn agg_dijkstra_cont(
    state: &Array2<i16>,
    y0: usize,
    x0: usize,
    path_state: &[i16],
    target_state: &[i16],
    opts: &DijkstraOpts,
) -> Array2<i32> {
    let (rows, cols) = state.dim();
    let mut targets = Array2::<i32>::zeros((rows, cols));
    let mut target_count: i64 = 0;
    if target_state.contains(&state[[y0, x0]]) {
        targets[[y0, x0]] = 1;
        target_count = 1;
    }

    let mut dist = Array2::<f64>::from_elem((rows, cols), f64::INFINITY);
    dist[[y0, x0]] = 0.0;
    let mut heap = BinaryHeap::new();
    heap.push(HeapItem {
        dist: 0.0,
        y: y0,
        x: x0,
    });

    while let Some(HeapItem { dist: d, y, x }) = heap.pop() {
        // skip stale heap entries
        if d > dist[[y, x]] {
            continue;
        }
        for (ny, nx) in iter_nbs(rows, cols, y, x, opts.rook) {
            let ystep = (ny as f64 - y as f64).abs();
            let xstep = (nx as f64 - x as f64).abs();
            let nd = d + ystep.hypot(xstep) * opts.granularity_m;
            if nd > opts.max_distance_m {
                continue;
            }
            // a target is aggregated even if it is not itself traversable
            if targets[[ny, nx]] == 0 && target_state.contains(&state[[ny, nx]]) {
                targets[[ny, nx]] = 1;
                target_count += 1;
            }
            if !path_state.contains(&state[[ny, nx]]) {
                continue;
            }
            if nd < dist[[ny, nx]] {
                dist[[ny, nx]] = nd;
                heap.push(HeapItem {
                    dist: nd,
                    y: ny,
                    x: nx,
                });
            }
        }
        if opts.break_first && target_count > 0 {
            break;
        }
        if let Some(bc) = opts.break_count {
            if target_count >= bc {
                break;
            }
        }
    }
    targets
}

/// Like [`agg_dijkstra_cont`] but returns the distance (metres) from `(y0, x0)` to
/// every cell reachable within `max_distance_m` traversing `path_state` cells;
/// `f64::INFINITY` elsewhere. Uses the same bounded binary-heap Dijkstra; only the
/// target aggregation is dropped (the raw `dist` array is returned instead).
pub fn agg_dijkstra_dist(
    state: &Array2<i16>,
    y0: usize,
    x0: usize,
    path_state: &[i16],
    opts: &DijkstraOpts,
) -> Array2<f64> {
    let (rows, cols) = state.dim();
    let mut dist = Array2::<f64>::from_elem((rows, cols), f64::INFINITY);
    dist[[y0, x0]] = 0.0;
    let mut heap = BinaryHeap::new();
    heap.push(HeapItem {
        dist: 0.0,
        y: y0,
        x: x0,
    });

    while let Some(HeapItem { dist: d, y, x }) = heap.pop() {
        // skip stale heap entries
        if d > dist[[y, x]] {
            continue;
        }
        for (ny, nx) in iter_nbs(rows, cols, y, x, opts.rook) {
            let ystep = (ny as f64 - y as f64).abs();
            let xstep = (nx as f64 - x as f64).abs();
            let nd = d + ystep.hypot(xstep) * opts.granularity_m;
            if nd > opts.max_distance_m {
                continue;
            }
            if !path_state.contains(&state[[ny, nx]]) {
                continue;
            }
            if nd < dist[[ny, nx]] {
                dist[[ny, nx]] = nd;
                heap.push(HeapItem {
                    dist: nd,
                    y: ny,
                    x: nx,
                });
            }
        }
    }
    dist
}

/// Builds the green-periphery (`green_itx`) and green-access (`green_acc`) arrays.
///
/// `green_itx`: unbuildable cells -> -1; built/centre cells -> 1; green cells
/// rook-adjacent to built -> 2 (the developable periphery). `green_acc`: for every
/// periphery (==2) cell, the accessibility footprint summed together — computed in
/// parallel. Footprints traverse `[0, 1, 2]` only, so unbuildable cells (water,
/// carved corridors) block green access exactly as they block centre access.
pub fn prepare_green_arrs(
    state: &Array2<i16>,
    max_distance_m: f64,
    granularity_m: f64,
) -> (Array2<i16>, Array2<i32>) {
    let (rows, cols) = state.dim();
    let mut green_itx = Array2::<i16>::zeros((rows, cols));
    for y in 0..rows {
        for x in 0..cols {
            if state[[y, x]] < 0 {
                green_itx[[y, x]] = -1;
            } else if state[[y, x]] > 0 {
                green_itx[[y, x]] = 1;
                for (ny, nx) in iter_nbs(rows, cols, y, x, true) {
                    if state[[ny, nx]] == 0 {
                        green_itx[[ny, nx]] = 2;
                    }
                }
            }
        }
    }

    let sources: Vec<(usize, usize)> = (0..rows)
        .flat_map(|y| (0..cols).map(move |x| (y, x)))
        .filter(|&(y, x)| green_itx[[y, x]] == 2)
        .collect();

    let opts = DijkstraOpts::new(max_distance_m, granularity_m);
    let green_acc = sources
        .par_iter()
        .map(|&(y, x)| agg_dijkstra_cont(&green_itx, y, x, &[0, 1, 2], &[0, 1, 2], &opts))
        .reduce(|| Array2::<i32>::zeros((rows, cols)), |a, b| a + b);

    (green_itx, green_acc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn reaches_all_cells_within_distance_on_open_grid() {
        // 1x5 all green; from index 0, granularity 1, max distance 4 -> all 5 reachable
        let state = Array2::<i16>::zeros((1, 5));
        let opts = DijkstraOpts::new(4.0, 1.0);
        let targets = agg_dijkstra_cont(&state, 0, 0, &[0], &[0], &opts);
        assert_eq!(targets.sum(), 5);
    }

    #[test]
    fn respects_max_distance() {
        let state = Array2::<i16>::zeros((1, 5));
        let opts = DijkstraOpts::new(2.0, 1.0); // reach indices 0,1,2 only
        let targets = agg_dijkstra_cont(&state, 0, 0, &[0], &[0], &opts);
        assert_eq!(targets.sum(), 3);
    }

    #[test]
    fn walls_block_traversal_but_targets_still_marked_if_adjacent() {
        // a built wall (1) is not a path cell for path=[0], but is a target for target=[1]
        let mut state = Array2::<i16>::zeros((1, 3));
        state[[0, 1]] = 1; // wall between 0 and 2
        let opts = DijkstraOpts::new(10.0, 1.0);
        // path only through green(0); target the wall(1). Start at 0.
        let targets = agg_dijkstra_cont(&state, 0, 0, &[0], &[1], &opts);
        // cell 1 is adjacent to start within distance -> marked; cell 2 unreachable (wall)
        assert_eq!(targets[[0, 1]], 1);
        assert_eq!(targets[[0, 2]], 0);
    }

    #[test]
    fn break_first_stops_early() {
        let mut state = Array2::<i16>::zeros((1, 5));
        state[[0, 4]] = 2;
        let mut opts = DijkstraOpts::new(100.0, 1.0);
        opts.break_first = true;
        let targets = agg_dijkstra_cont(&state, 0, 0, &[0, 2], &[2], &opts);
        assert!(targets.sum() >= 1);
    }

    #[test]
    fn dist_increases_with_steps_and_is_inf_beyond_max() {
        // 1x5 open green row; from index 0, granularity 1, max distance 3.
        let state = Array2::<i16>::zeros((1, 5));
        let opts = DijkstraOpts::new(3.0, 1.0);
        let dist = agg_dijkstra_dist(&state, 0, 0, &[0], &opts);
        assert_eq!(dist[[0, 0]], 0.0);
        assert_eq!(dist[[0, 1]], 1.0);
        assert_eq!(dist[[0, 2]], 2.0);
        assert_eq!(dist[[0, 3]], 3.0);
        // distance strictly increases with steps along the open grid
        assert!(dist[[0, 1]] < dist[[0, 2]]);
        assert!(dist[[0, 2]] < dist[[0, 3]]);
        // index 4 is 4m away -> beyond max_distance -> infinite
        assert!(dist[[0, 4]].is_infinite());
    }

    #[test]
    fn unbuildable_blocks_green_access() {
        // built column at x=0, unbuildable column at x=2: periphery footprints
        // from x=1 must not cross the barrier to reach x>=3
        let mut state = Array2::<i16>::zeros((5, 5));
        for y in 0..5 {
            state[[y, 0]] = 1;
            state[[y, 2]] = -1;
        }
        let (itx, acc) = prepare_green_arrs(&state, 300.0, 100.0);
        for y in 0..5 {
            assert_eq!(itx[[y, 2]], -1);
            assert_eq!(itx[[y, 1]], 2);
            assert_eq!(acc[[y, 3]], 0);
            assert_eq!(acc[[y, 4]], 0);
            assert!(acc[[y, 1]] > 0);
        }
    }

    #[test]
    fn prepare_green_arrs_marks_periphery() {
        // a single built cell in a green field -> its rook neighbours become itx==2
        let mut state = Array2::<i16>::zeros((3, 3));
        state[[1, 1]] = 1;
        let (itx, acc) = prepare_green_arrs(&state, 100.0, 100.0);
        assert_eq!(itx[[1, 1]], 1);
        assert_eq!(itx[[0, 1]], 2);
        assert_eq!(itx[[1, 0]], 2);
        assert_eq!(itx[[2, 1]], 2);
        assert_eq!(itx[[1, 2]], 2);
        // corners are not rook-adjacent to the built cell
        assert_eq!(itx[[0, 0]], 0);
        assert!(acc.sum() > 0);
    }
}
