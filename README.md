# BSP Isobenefit Urbanism QGIS Plugin

QGIS plugin for Isobenefit Urbanism — a tool for brainstorming walkable urban
development against real-world datasets, based on walkable access to centralities
(shops, services) and green spaces.

## Repository layout

This repository contains two deliverables:

- [`isobenefit/`](isobenefit/) — the **QGIS plugin** (thin). It handles the UI and
  all GIS IO (reading layers, reprojection, rasterization, writing rasters,
  temporal animation), and depends only on libraries QGIS already bundles (numpy +
  GDAL) plus the simulation engine below.
- [`core/`](core/) — the **simulation engine** (`isobenefit-core`), a Rust
  extension built with PyO3/maturin and published to PyPI as abi3 wheels. It is
  pure compute (arrays in, arrays out) and never imports QGIS.

The split exists because the QGIS plugin repository does not permit shipping
binaries: the compiled engine is installed from PyPI rather than bundled.

## Installation

1. In QGIS, enable experimental plugins (Plugins → Manage and Install Plugins →
   Settings → "Show also experimental plugins"), then search for **isobenefit**
   and install it.
2. The first time you run the plugin it will check for the `isobenefit-core`
   engine and, if it is missing, offer to install it into the QGIS Python
   environment for you. This needs an internet connection; **restart QGIS** once
   it finishes.

That's it — there is no longer any manual `pip install numba/rasterio` step.

If the automatic install is blocked (e.g. a locked-down environment), the dialog
shows the exact command to run yourself, which is simply:

```bash
<qgis-python> -m pip install "isobenefit-core>=0.1,<0.2"
```

## Usage

Open the plugin, choose an output `.tif` path, a polygon **extents** layer, a
projected **CRS**, and optionally layers for existing built areas, green areas,
unbuildable areas and centre seeds. Set the parameters and run. Each iteration is
written as a categorical GeoTIFF and loaded as a temporal animation; press play in
the Temporal Controller. A demo project is provided in
[`demo_layers/`](demo_layers/) (`cambourne.qgz`).

## Development

### Simulation core (Rust)

```bash
cargo test --manifest-path core/Cargo.toml      # pure-Rust unit tests (no Python)
uvx maturin develop --manifest-path core/Cargo.toml   # build + install into the active venv
uvx maturin build   --manifest-path core/Cargo.toml   # produce an abi3 wheel in core/dist
```

Then exercise the Python API:

```bash
python -m pytest core/tests_py
```

The core is parallel by design (rayon) and deterministic regardless of thread
count — see [`core/README.md`](core/README.md).

### QGIS plugin

Lint with `ruff check isobenefit`.

For live development, link the plugin folder into your QGIS profile, e.g. on macOS:

```bash
ln -s "$(pwd)/isobenefit" "$HOME/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/"
```

On Windows, copy/paste the `isobenefit` folder into
`%APPDATA%/QGIS/QGIS3/profiles/default/python/plugins`. Restart QGIS after linking.

### CI

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs the Rust tests,
builds and smoke-tests abi3 wheels on Linux/macOS/Windows, builds the plugin zip,
and (on a `core-v*` tag) publishes the engine to PyPI via trusted publishing.

## Packaging the plugin

The plugin zip is the `isobenefit/` folder only (never `core/`):

```bash
zip -r isobenefit.zip isobenefit -x '*/__pycache__/*' '*.pyc'
```

## Licensing

This project is licensed under the **GNU AGPL-3.0-or-later** (see [LICENSE](LICENSE)).

## Original version

This work is an outgrowth of the [original work](https://github.com/mitochevole/isobenefit-cities)
(forked to [BSP-isobenefit-original](https://github.com/UCL/BSP-isobenefit-original))
developed by Michele Voto and Luca D'Acci, and has been developed as part of the
Future Urban Growth project at the Bartlett School of Planning.

## Website

An overview of this plugin is at
[BSP-isobenefit-urbanism](https://github-pages.ucl.ac.uk/BSP-isobenefit-urbanism)
([repo](https://github.com/UCL/BSP-isobenefit-urbanism)).

## References

- [PyQGIS](https://qgis.org/pyqgis/3.28/)
- [PyQGIS Developer Cookbook](https://docs.qgis.org/testing/en/docs/pyqgis_developer_cookbook/index.html)
- [maturin](https://www.maturin.rs/) · [PyO3](https://pyo3.rs/)
