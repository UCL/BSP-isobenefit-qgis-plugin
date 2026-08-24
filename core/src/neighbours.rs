//! Grid neighbour iteration and contiguity helpers.
//!
//! Ports `iter_nbs`, `count_cont_nbs`, `green_span` and `green_spans` from the
//! original numba `algos.py`, preserving the neighbour *ordering* (the queen ring
//! order matters for contiguity / wraparound detection).

use ndarray::{Array2, ArrayView1};

/// In-bounds neighbours of `(y, x)` in a fixed order.
///
/// `rook` = the 4 orthogonal neighbours; otherwise the 8 queen neighbours walked
/// as a ring (so consecutive entries are spatially adjacent — required by
/// [`count_cont_nbs`]). Out-of-bounds neighbours are simply omitted.
#[inline]
pub fn iter_nbs(rows: usize, cols: usize, y: usize, x: usize, rook: bool) -> Vec<(usize, usize)> {
    const ROOK_Y: [i64; 4] = [1, 0, -1, 0];
    const ROOK_X: [i64; 4] = [0, 1, 0, -1];
    const QUEEN_Y: [i64; 8] = [1, 1, 1, 0, -1, -1, -1, 0];
    const QUEEN_X: [i64; 8] = [-1, 0, 1, 1, 1, 0, -1, -1];

    let (ys, xs): (&[i64], &[i64]) = if rook {
        (&ROOK_Y, &ROOK_X)
    } else {
        (&QUEEN_Y, &QUEEN_X)
    };
    let mut out = Vec::with_capacity(ys.len());
    for k in 0..ys.len() {
        let ny = y as i64 + ys[k];
        if ny < 0 || ny >= rows as i64 {
            continue;
        }
        let nx = x as i64 + xs[k];
        if nx < 0 || nx >= cols as i64 {
            continue;
        }
        out.push((ny as usize, nx as usize));
    }
    out
}

/// Counts contiguous neighbour groups whose state is in `targets`.
///
/// Returns `(total, longest_run, num_runs)`:
/// - `total` — number of neighbours in `targets`
/// - `longest_run` — length of the longest contiguous run around the ring
/// - `num_runs` — number of separate runs (i.e. distinct adjacent regions)
///
/// Wraparound (first and last ring entries both set) is only merged for interior
/// cells that have the full 8 neighbours, matching the original.
pub fn count_cont_nbs(state: &Array2<i16>, y: usize, x: usize, targets: &[i16]) -> (i64, i64, i64) {
    let (rows, cols) = state.dim();
    let nbs = iter_nbs(rows, cols, y, x, false);
    let circle: Vec<u8> = nbs
        .iter()
        .map(|&(ny, nx)| u8::from(targets.contains(&state[[ny, nx]])))
        .collect();

    let mut adds: Vec<i64> = Vec::new();
    let mut run: i64 = 0;
    for &c in &circle {
        if c == 1 {
            run += 1;
        } else if run > 0 {
            adds.push(run);
            run = 0;
        }
    }
    if run > 0 {
        adds.push(run);
    }
    // wraparound only for full 8-neighbour rings
    if adds.len() > 1 && circle.len() == 8 && circle[0] == 1 && *circle.last().unwrap() == 1 {
        let last = adds.pop().unwrap();
        adds[0] += last;
    }
    if adds.is_empty() {
        return (0, 0, 0);
    }
    let total: i64 = adds.iter().sum();
    let longest: i64 = *adds.iter().max().unwrap();
    (total, longest, adds.len() as i64)
}

/// Connected-component labels of a bool mask: 0 = background, 1..=n per component,
/// rook or queen connectivity. Serves the plugin's pruning/anchoring logic; the
/// Python loop it replaces was a visible share of post-processing on large windows.
pub fn label_components(mask: &Array2<bool>, queen: bool) -> Array2<i32> {
    let (rows, cols) = mask.dim();
    let mut labels = Array2::<i32>::zeros((rows, cols));
    let mut next = 0i32;
    let mut stack: Vec<(usize, usize)> = Vec::new();
    for sy in 0..rows {
        for sx in 0..cols {
            if !mask[[sy, sx]] || labels[[sy, sx]] != 0 {
                continue;
            }
            next += 1;
            labels[[sy, sx]] = next;
            stack.push((sy, sx));
            while let Some((y, x)) = stack.pop() {
                for (ny, nx) in iter_nbs(rows, cols, y, x, !queen) {
                    if mask[[ny, nx]] && labels[[ny, nx]] == 0 {
                        labels[[ny, nx]] = next;
                        stack.push((ny, nx));
                    }
                }
            }
        }
    }
    labels
}

/// Length of the run of green (`== 0`) cells from `start` along a 1-D line, in
/// the positive or negative direction, stopping at the first non-green cell
/// (built `> 0` or unbuildable `< 0`) or the array edge. Returns the run length
/// and the terminating cell value (`None` at the array edge). Unbuildable land
/// is not usable green, so water or a carved corridor terminates a span rather
/// than extending it.
pub fn green_span(line: ArrayView1<i16>, start: usize, positive: bool) -> (i64, Option<i16>) {
    let n = line.len();
    let mut span: i64 = 0;
    if positive {
        let mut i = start + 1;
        while i < n {
            if line[i] != 0 {
                return (span, Some(line[i]));
            }
            span += 1;
            i += 1;
        }
    } else {
        if start == 0 {
            return (0, None);
        }
        let mut i = start as i64 - 1;
        while i >= 0 {
            if line[i as usize] != 0 {
                return (span, Some(line[i as usize]));
            }
            span += 1;
            i -= 1;
        }
    }
    (span, None)
}

/// True if filling `(y, x)` would not crimp a green corridor **between distinct
/// developments** below the minimum span.
///
/// The span is measured along each axis from the candidate to the first
/// non-green cell. A run is only protected when it separates the candidate from
/// built land belonging to a *different* settlement than the one the candidate
/// is joining: that is a corridor between developments, and pinching it would
/// leave a strip too narrow to serve as green space between them.
///
/// A settlement's own arms do not constrain it. Where growth forms a C, U or N,
/// the green enclosed by those arms is a bay of one settlement rather than a
/// corridor between two, so the settlement may thicken into it; the park-area
/// rule and the green-walk guard still apply to what is built. Without this the
/// bay could never be filled, and runs left narrow fingers with a permanent
/// green slot on either side.
///
/// A span of 0 (the candidate already touches built land) is always allowed, and
/// unbuildable land (water, a carved corridor) terminates a run without
/// protecting it, since it is not usable green.
///
/// `built_labels` carries connected-component labels of built land; with `None`
/// every built terminator is treated as foreign, which is the conservative
/// behaviour.
pub fn green_spans(
    state: &Array2<i16>,
    y: usize,
    x: usize,
    granularity_m: f64,
    min_green_span_m: f64,
    built_labels: Option<&Array2<i32>>,
) -> bool {
    let row = state.row(y);
    let col = state.column(x);
    // the settlements this candidate is joining: their own arms may not block it
    let mut own: Vec<i32> = Vec::new();
    if let Some(lbl) = built_labels {
        for (ny, nx) in iter_nbs(state.dim().0, state.dim().1, y, x, false) {
            let l = lbl[[ny, nx]];
            if l != 0 && !own.contains(&l) {
                own.push(l);
            }
        }
    }
    let (rows, cols) = state.dim();
    let spans = [
        (
            green_span(row, x, false),
            (y as i64, x as i64 - 1),
            (0i64, -1i64),
        ),
        (green_span(row, x, true), (y as i64, x as i64 + 1), (0, 1)),
        (green_span(col, y, false), (y as i64 - 1, x as i64), (-1, 0)),
        (green_span(col, y, true), (y as i64 + 1, x as i64), (1, 0)),
    ];
    let span_blocks = min_green_span_m / granularity_m;
    for ((s, terminator), _first, (dy, dx)) in spans {
        match terminator {
            // the array edge, or unbuildable land: nothing to keep a corridor from
            None => continue,
            Some(v) if v < 0 => continue,
            _ => {}
        }
        if s == 0 || (s as f64) >= span_blocks {
            continue;
        }
        // the cell that terminates this run
        let ty = y as i64 + dy * (s + 1);
        let tx = x as i64 + dx * (s + 1);
        if ty < 0 || tx < 0 || ty >= rows as i64 || tx >= cols as i64 {
            continue;
        }
        if let Some(lbl) = built_labels {
            let l = lbl[[ty as usize, tx as usize]];
            // the candidate's own settlement across a bay does not constrain it
            if l != 0 && own.contains(&l) {
                continue;
            }
        }
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn a_settlements_own_arms_do_not_block_thickening() {
        // a U of one settlement: arms in columns 0 and 4, base along the bottom row.
        // The bay between the arms is 3 cells wide, under a 300 m (3-block) span.
        let state = array![[1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1],];
        let labels = label_components(&state.mapv(|v| v > 0), true);
        // without labels the far arm reads as the other side of a corridor
        assert!(!green_spans(&state, 1, 1, 100.0, 300.0, None));
        // with them it is the same settlement, so the arm may thicken into its own bay
        assert!(green_spans(&state, 1, 1, 100.0, 300.0, Some(&labels)));
    }

    #[test]
    fn a_corridor_between_two_settlements_is_still_protected() {
        // two separate settlements with a 3-cell green corridor between them
        let state = array![[1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1],];
        let labels = label_components(&state.mapv(|v| v > 0), true);
        assert!(!green_spans(&state, 1, 1, 100.0, 300.0, Some(&labels)));
    }

    #[test]
    fn rook_corner_has_two_neighbours() {
        let nbs = iter_nbs(3, 3, 0, 0, true);
        assert_eq!(nbs.len(), 2);
        assert!(nbs.contains(&(1, 0)));
        assert!(nbs.contains(&(0, 1)));
    }

    #[test]
    fn queen_interior_has_eight_in_ring_order() {
        let nbs = iter_nbs(3, 3, 1, 1, false);
        assert_eq!(
            nbs,
            vec![
                (2, 0),
                (2, 1),
                (2, 2),
                (1, 2),
                (0, 2),
                (0, 1),
                (0, 0),
                (1, 0)
            ]
        );
    }

    #[test]
    fn full_ring_of_targets_is_one_region() {
        // centre surrounded entirely by built (1)
        let state = array![[1, 1, 1], [1, 0, 1], [1, 1, 1]];
        let (total, longest, regions) = count_cont_nbs(&state, 1, 1, &[1, 2]);
        assert_eq!((total, longest, regions), (8, 8, 1));
    }

    #[test]
    fn wraparound_merges_split_runs() {
        // built on the whole top row and the two side cells -> wraps around
        // ring order: (2,0)=0 (2,1)=0 (2,2)=0 (1,2)=1 (0,2)=1 (0,1)=1 (0,0)=1 (1,0)=1
        // -> single contiguous run of 5 after wraparound (last+first)
        let state = array![[1, 1, 1], [1, 0, 1], [0, 0, 0]];
        let (total, longest, regions) = count_cont_nbs(&state, 1, 1, &[1, 2]);
        assert_eq!(total, 5);
        assert_eq!(regions, 1);
        assert_eq!(longest, 5);
    }

    #[test]
    fn two_separate_regions() {
        // opposite corners/edges set but not adjacent in the ring
        let state = array![[0, 1, 0], [0, 0, 0], [0, 1, 0]];
        let (total, _longest, regions) = count_cont_nbs(&state, 1, 1, &[1, 2]);
        assert_eq!(total, 2);
        assert_eq!(regions, 2);
    }

    #[test]
    fn green_span_counts_until_built() {
        let line = array![0i16, 0, 0, 1, 0];
        // from index 0 going positive: indices 1,2 are green, index 3 is built -> 2
        assert_eq!(green_span(line.view(), 0, true), (2, Some(1)));
        // from index 4 going negative: index 3 is built immediately -> 0
        assert_eq!(green_span(line.view(), 4, false), (0, Some(1)));
    }

    #[test]
    fn green_span_stops_at_unbuildable() {
        // water/carved corridors terminate a green span; they do not extend it
        let line = array![0i16, 0, -1, 0, 0];
        assert_eq!(green_span(line.view(), 0, true), (1, Some(-1)));
        assert_eq!(green_span(line.view(), 4, false), (1, Some(-1)));
    }

    #[test]
    fn unbuildable_bounded_spans_do_not_block() {
        // a short green strip against a river/carved corridor may be built on;
        // the same strip against built land is a protected corridor and blocks
        let riverside = array![[1i16, 0, 0, 0, -1]];
        assert!(green_spans(&riverside, 0, 1, 100.0, 300.0, None));
        let between_built = array![[1i16, 0, 0, 0, 1]];
        assert!(!green_spans(&between_built, 0, 1, 100.0, 300.0, None));
    }

    #[test]
    fn a_run_reaching_the_grid_edge_is_not_a_corridor() {
        // a single green cell flanked by built on the row, green to the edge above and
        // below. min span 3 cells (300 m / 100 m). The short runs up and down end at the
        // array edge, not at another development, so they constrain nothing.
        let state = array![[0, 0, 0], [1, 0, 1], [0, 0, 0]];
        let labels = label_components(&state.mapv(|v| v > 0), true);
        assert!(green_spans(&state, 1, 1, 100.0, 300.0, Some(&labels)));
    }

    #[test]
    fn a_short_run_to_a_foreign_settlement_still_blocks() {
        // the same candidate, but with a second settlement two cells below: that run is
        // a corridor between developments and is protected.
        let state = array![[0, 0, 0], [1, 0, 1], [0, 0, 0], [0, 3, 0]];
        let labels = label_components(&state.mapv(|v| v > 0), true);
        assert!(!green_spans(&state, 1, 1, 100.0, 300.0, Some(&labels)));
    }
}
