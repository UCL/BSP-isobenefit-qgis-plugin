//! PyO3 bindings: the `isobenefit._core` extension module.
//!
//! Only compiled with `--features python` (maturin). Exposes the [`Simulation`]
//! state machine and the parallel `run_ensemble`, marshalling numpy arrays in and
//! out. All heavy compute releases the GIL where it runs across threads.

use crate::sim::{
    ensemble_class_counts as core_ensemble_class_counts,
    ensemble_probability as core_ensemble_probability, run_ensemble as core_run_ensemble, Params,
    Simulation,
};
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyclass(name = "Simulation")]
pub struct PySimulation {
    inner: Simulation,
}

#[pymethods]
impl PySimulation {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        state, origin, density, centre_seeds,
        granularity_m, max_distance_m, max_populat, min_green_span_m,
        build_prob, cent_prob_nb, cent_prob_isol, pop_target_cent_threshold,
        prob_distribution, density_factors_km2,
        total_iters, random_seed,
    ))]
    fn new(
        state: PyReadonlyArray2<i16>,
        origin: PyReadonlyArray2<i16>,
        density: PyReadonlyArray2<f32>,
        centre_seeds: Vec<(usize, usize)>,
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
        total_iters: usize,
        random_seed: u64,
    ) -> PyResult<Self> {
        let params = Params::from_raw(
            granularity_m,
            max_distance_m,
            max_populat,
            min_green_span_m,
            build_prob,
            cent_prob_nb,
            cent_prob_isol,
            pop_target_cent_threshold,
            prob_distribution,
            density_factors_km2,
        )
        .map_err(PyValueError::new_err)?;
        let inner = Simulation::new(
            state.as_array().to_owned(),
            origin.as_array().to_owned(),
            density.as_array().to_owned(),
            &centre_seeds,
            params,
            total_iters,
            random_seed,
        )
        .map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }

    /// Run one iteration.
    fn step(&mut self) {
        self.inner.step();
    }

    /// Run to completion (or until the population target is reached).
    fn run(&mut self, py: Python<'_>) {
        py.allow_threads(|| self.inner.run());
    }

    #[getter]
    fn current_iter(&self) -> usize {
        self.inner.current_iter
    }

    #[getter]
    fn total_iters(&self) -> usize {
        self.inner.total_iters
    }

    #[getter]
    fn pop_target_ratio(&self) -> f64 {
        self.inner.pop_target_ratio
    }

    #[getter]
    fn population(&self) -> f64 {
        self.inner.population()
    }

    /// Current state as a dict of numpy arrays (copies).
    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("state", self.inner.state.to_pyarray_bound(py))?;
        d.set_item("density", self.inner.density.to_pyarray_bound(py))?;
        d.set_item("origin", self.inner.origin.to_pyarray_bound(py))?;
        d.set_item("green_acc", self.inner.green_acc.to_pyarray_bound(py))?;
        d.set_item("cent_acc", self.inner.cent_acc.to_pyarray_bound(py))?;
        Ok(d)
    }
}

/// Run `n_members` independent simulations from `template`, in parallel, returning
/// each member's final `state` grid. Releases the GIL during compute.
#[pyfunction]
fn run_ensemble(
    py: Python<'_>,
    template: &PySimulation,
    base_seed: u64,
    n_members: usize,
) -> Vec<Py<PyArray2<i16>>> {
    let results: Vec<Array2<i16>> =
        py.allow_threads(|| core_run_ensemble(&template.inner, base_seed, n_members));
    results
        .into_iter()
        .map(|arr| arr.into_pyarray_bound(py).unbind())
        .collect()
}

/// Run `n_members` simulations from `template` in parallel and return a
/// probability-of-development grid (fraction of members urban per cell) as a
/// float32 numpy array. Releases the GIL during compute.
#[pyfunction]
fn ensemble_probability(
    py: Python<'_>,
    template: &PySimulation,
    base_seed: u64,
    n_members: usize,
) -> Py<PyArray2<f32>> {
    let prob =
        py.allow_threads(|| core_ensemble_probability(&template.inner, base_seed, n_members));
    prob.into_pyarray_bound(py).unbind()
}

/// Run `n_members` simulations from `template` in parallel and return per-class
/// development counts as three uint32 numpy arrays: (built, green, centre).
#[pyfunction]
fn ensemble_class_counts(
    py: Python<'_>,
    template: &PySimulation,
    base_seed: u64,
    n_members: usize,
) -> (Py<PyArray2<u32>>, Py<PyArray2<u32>>, Py<PyArray2<u32>>) {
    let (b, g, c) =
        py.allow_threads(|| core_ensemble_class_counts(&template.inner, base_seed, n_members));
    (
        b.into_pyarray_bound(py).unbind(),
        g.into_pyarray_bound(py).unbind(),
        c.into_pyarray_bound(py).unbind(),
    )
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<PySimulation>()?;
    m.add_function(wrap_pyfunction!(run_ensemble, m)?)?;
    m.add_function(wrap_pyfunction!(ensemble_probability, m)?)?;
    m.add_function(wrap_pyfunction!(ensemble_class_counts, m)?)?;
    Ok(())
}
