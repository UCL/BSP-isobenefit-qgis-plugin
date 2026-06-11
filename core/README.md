# isobenefit

The Rust simulation core for the [Isobenefit Urbanism QGIS plugin](https://github.com/UCL/BSP-isobenefit-qgis-plugin).

It is a pure compute engine: **arrays in, arrays out**. It never imports QGIS or
performs any GIS IO — the plugin handles reading layers, reprojection,
rasterization and writing rasters, then hands plain numpy arrays to this core.

## Layout

- `src/neighbours.rs` — neighbour iteration, contiguity counting, green-span checks
- `src/access.rs` — heap-based bounded Dijkstra accessibility (parallel map-reduce)
- `src/density.rs` — deterministic per-work-item RNG + density draws
- `src/sim.rs` — the `Simulation` state machine and `run_ensemble`
- `src/lib.rs` / `src/py_bindings.rs` — PyO3 bindings (built only with `--features python`)

## Development

```bash
cargo test                 # pure-Rust algorithm tests (no Python needed)
maturin develop --release  # build + install the extension into the active venv
pytest tests_py            # Python-level golden + property tests
```

Parallelism (rayon) and determinism are first-class: results are independent of
thread count. Reproducibility comes from per-work-item seeding
(`ChaCha8Rng` derived from `(seed, work_id)`) and order-independent integer-sum
reductions in the accessibility kernels.

Licensed under AGPL-3.0-or-later.
