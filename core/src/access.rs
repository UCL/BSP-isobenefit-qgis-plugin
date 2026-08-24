//! Bounded accessibility search.
//!
//! `agg_dijkstra_cont` replaces the original O(N^2) "scan the whole pending grid
//! each step" search with a proper binary-heap Dijkstra bounded by `max_distance`.
//! `prepare_park_arrs` builds the park mask and park-access surfaces as a
//! parallel (rayon) map-reduce; the reduction is an integer sum, so the result is
//! identical regardless of how the work is split across threads.

use crate::neighbours::{iter_nbs, label_components};
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
    /// Treat this cell as absent: never traversed and never counted as a
    /// target. Lets a caller probe a tentative build without cloning the grid.
    pub exclude: Option<(usize, usize)>,
}

impl DijkstraOpts {
    pub fn new(max_distance_m: f64, granularity_m: f64) -> Self {
        Self {
            max_distance_m,
            granularity_m,
            break_first: false,
            break_count: None,
            rook: false,
            exclude: None,
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
            if opts.exclude == Some((ny, nx)) {
                continue;
            }
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
            // a diagonal step may not squeeze between two cells the path cannot use,
            // so growth measures the walk exactly as the scoring does
            if ny != y
                && nx != x
                && !path_state.contains(&state[[y, nx]])
                && !path_state.contains(&state[[ny, x]])
            {
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
            // as above: no diagonal squeeze between two cells the path cannot use
            if ny != y
                && nx != x
                && !path_state.contains(&state[[y, nx]])
                && !path_state.contains(&state[[ny, x]])
            {
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

/// Multi-source bounded walk: the distance (metres) from every cell to the nearest
/// `true` cell in `targets`, traversing all cells (queen moves, diagonal cost
/// sqrt(2) x granularity), `inf` beyond `max_distance_m`. This is the plugin's
/// post-processing walk field; it lives in the engine because the Python loop was
/// the whole cost of post-processing on large windows.
pub fn walk_distance(
    targets: &Array2<bool>,
    granularity_m: f64,
    max_distance_m: f64,
    blocked: Option<&Array2<bool>>,
) -> Array2<f64> {
    let (rows, cols) = targets.dim();
    let mut dist = Array2::<f64>::from_elem((rows, cols), f64::INFINITY);
    let mut heap = BinaryHeap::new();
    for ((y, x), &is_target) in targets.indexed_iter() {
        if is_target {
            dist[[y, x]] = 0.0;
            heap.push(HeapItem { dist: 0.0, y, x });
        }
    }
    while let Some(HeapItem { dist: d, y, x }) = heap.pop() {
        if d > dist[[y, x]] {
            continue;
        }
        for (ny, nx) in iter_nbs(rows, cols, y, x, false) {
            if let Some(b) = blocked {
                if b[[ny, nx]] {
                    continue;
                }
                // a diagonal step may not squeeze between two blocked cells: a walk
                // cannot pass through the corner where a carved corridor, a river or a
                // steep band meets itself
                if ny != y && nx != x && b[[y, nx]] && b[[ny, x]] {
                    continue;
                }
            }
            let ystep = (ny as f64 - y as f64).abs();
            let xstep = (nx as f64 - x as f64).abs();
            let nd = d + ystep.hypot(xstep) * granularity_m;
            if nd <= max_distance_m && nd < dist[[ny, nx]] {
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

/// The number of cells a park-qualifying green area must hold, from the minimum
/// park area in square metres. This is the scoring rule's park test, shared here
/// so the growth guard and the served-coverage metric use one park definition.
pub fn park_threshold_cells(min_park_area_m2: f64, granularity_m: f64) -> i64 {
    ((min_park_area_m2 / (granularity_m * granularity_m)).round() as i64).max(1)
}

/// Builds the park mask and park-access count for the green-access guard.
///
/// `park`: green (state 0) cells whose rook-connected green component holds at
/// least [`park_threshold_cells`] cells. `green_acc[c]`: how many park cells lie
/// within the green walk of `c`, traversing everything except unbuildable land —
/// each park cell's bounded footprint, summed in parallel. Unbuildable land never
/// changes and built land stays traversable, so every footprint is constant for
/// the whole run and the count stays exact under the single-cell removals
/// `try_build` applies.
pub fn prepare_park_arrs(
    state: &Array2<i16>,
    green_distance_m: f64,
    granularity_m: f64,
    min_park_area_m2: f64,
) -> (Array2<bool>, Array2<i32>) {
    let (rows, cols) = state.dim();
    let threshold = park_threshold_cells(min_park_area_m2, granularity_m);
    let mask = state.mapv(|v| v == 0);
    let labels = label_components(&mask, false);
    let n_labels = labels.iter().copied().max().unwrap_or(0) as usize;
    let mut areas = vec![0i64; n_labels + 1];
    for &l in labels.iter() {
        areas[l as usize] += 1;
    }
    let park = Array2::from_shape_fn((rows, cols), |(y, x)| {
        let l = labels[[y, x]];
        l > 0 && areas[l as usize] >= threshold
    });

    let sources: Vec<(usize, usize)> = park
        .indexed_iter()
        .filter(|&(_, &is_park)| is_park)
        .map(|(idx, _)| idx)
        .collect();

    let opts = DijkstraOpts::new(green_distance_m, granularity_m);
    let green_acc = sources
        .par_iter()
        .map(|&(y, x)| agg_dijkstra_cont(state, y, x, &[0, 1, 2], &[0, 1, 2], &opts))
        .reduce(|| Array2::<i32>::zeros((rows, cols)), |a, b| a + b);

    (park, green_acc)
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
    fn unbuildable_blocks_park_access() {
        // built column at x=0, unbuildable column at x=2: the park east of the
        // barrier is unreachable from the west side within the walk
        let mut state = Array2::<i16>::zeros((5, 5));
        for y in 0..5 {
            state[[y, 0]] = 1;
            state[[y, 2]] = -1;
        }
        // 9 ha at 100 m cells -> threshold 9: west green (5 cells) is not a
        // park; east green (10 cells) is
        let (park, acc) = prepare_park_arrs(&state, 300.0, 100.0, 90_000.0);
        for y in 0..5 {
            assert!(!park[[y, 1]]);
            assert!(park[[y, 3]] && park[[y, 4]]);
            // west of the barrier no park is reachable; east cells reach their own
            assert_eq!(acc[[y, 0]], 0);
            assert_eq!(acc[[y, 1]], 0);
            assert!(acc[[y, 3]] > 0);
        }
    }

    #[test]
    fn park_threshold_matches_scoring_rule() {
        // max(1, round(area / cell area)), the scoring park test; 16 ha default
        assert_eq!(park_threshold_cells(160_000.0, 50.0), 64);
        assert_eq!(park_threshold_cells(160_000.0, 100.0), 16);
        assert_eq!(park_threshold_cells(10_000.0, 100.0), 1);
        assert_eq!(park_threshold_cells(0.0, 100.0), 1);
    }

    #[test]
    fn small_components_are_not_parks() {
        // a lone green cell walled by unbuildable land, and a big open field:
        // only the field qualifies at threshold 4, and access counts field cells
        let mut state = Array2::<i16>::zeros((5, 5));
        for y in 0..5 {
            state[[y, 1]] = -1;
        }
        // column 0 is a 5-cell strip (>= 4 -> park); make it 3 cells instead
        state[[0, 0]] = -1;
        state[[1, 0]] = -1;
        let (park, acc) = prepare_park_arrs(&state, 200.0, 100.0, 40_000.0);
        for y in 2..5 {
            assert!(
                !park[[y, 0]],
                "3-cell strip must not qualify at threshold 4"
            );
            assert_eq!(acc[[y, 0]], 0);
        }
        assert!(park[[0, 2]]);
        assert!(acc[[0, 2]] > 0);
    }

    #[test]
    fn exclude_cell_blocks_traversal_and_counting() {
        // 1x5 all green; excluding the middle cell cuts the row in two
        let state = Array2::<i16>::zeros((1, 5));
        let mut opts = DijkstraOpts::new(10.0, 1.0);
        opts.exclude = Some((0, 2));
        let targets = agg_dijkstra_cont(&state, 0, 0, &[0], &[0], &opts);
        assert_eq!(targets.sum(), 2); // cells 0 and 1 only
    }
}
