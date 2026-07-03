# BSP Isobenefit Urbanism QGIS Plugin

QGIS plugin for Isobenefit Urbanism — a tool for brainstorming walkable urban
development against real-world datasets, based on walkable access to centralities
(shops, services) and green spaces.

📖 **Documentation & overview website:** <https://github-pages.ucl.ac.uk/BSP-isobenefit-qgis-plugin/>

## Repository layout

This repository contains two deliverables:

- [`isobenefit_qgis/`](isobenefit_qgis/) — the **QGIS plugin** (thin). It handles the UI and
  all GIS IO (reading layers, reprojection, rasterization, writing rasters,
  temporal animation), and depends only on libraries QGIS already bundles (numpy +
  GDAL) plus the simulation engine below.
- [`core/`](core/) — the **simulation engine** (`isobenefit`), a Rust
  extension built with PyO3/maturin and published to PyPI as abi3 wheels. It is
  pure compute (arrays in, arrays out) and never imports QGIS.

The split exists because the QGIS plugin repository does not permit shipping
binaries: the compiled engine is installed from PyPI rather than bundled.

## Installation

1. In QGIS (4.x; the 3.40 LTR should also work, but is untested), enable experimental plugins (Plugins → Manage and Install Plugins →
   Settings → "Show also experimental plugins"), then search for **isobenefit**
   and install it.
2. The first time you run the plugin it will check for the `isobenefit`
   engine and, if it is missing, offer to install it into the QGIS Python
   environment for you. This needs an internet connection; **restart QGIS** once
   it finishes.

That's it — there is no longer any manual `pip install numba/rasterio` step.

If the automatic install is blocked (e.g. a locked-down environment), the dialog
shows the exact command to run yourself, which is simply:

```bash
<qgis-python> -m pip install "isobenefit>=0.10,<0.11"
```

## Usage

Open the plugin and choose an output `.tif` path, a polygon **extents** layer, and a
projected **CRS**. Optionally add layers for existing built areas, green space,
unbuildable land, urban centres, public-transport stops, rail/tram stations, and a
street network. No layers prepared? The companion **Extract from OpenStreetMap** tool
downloads them for an area of interest.

Set the parameters and run. A single run is written iteration-by-iteration as a
categorical GeoTIFF and loaded as a temporal animation (press play in the Temporal
Controller). An **ensemble** of runs instead produces development-likelihood layers
plus **idealised planning scenarios** to compare — the existing fabric, the raw
(as-grown) state, and moderately- vs tightly-clustered centre arrangements —
alongside a `_report.txt` summarising the run. A demo project is in
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

Lint with `ruff check isobenefit_qgis`.

The plugin's pure pipeline (grid/routing/OSM helpers) has a headless test suite that
needs no QGIS — run it against a locally built engine wheel:

```bash
uv run --no-project \
  --with core/dist/isobenefit-*.whl --with numpy --with shapely --with pytest \
  python -m pytest tests -q
```

CI runs the same suite against a freshly built wheel on every push, so
plugin-vs-engine drift is caught before it reaches QGIS.

For live development, link the plugin folder into your QGIS profile. The profile
directory is named for your QGIS major version (`QGIS4` for QGIS 4.x, `QGIS3` for
QGIS 3.x). On macOS:

```bash
ln -s "$(pwd)/isobenefit_qgis" "$HOME/Library/Application Support/QGIS/QGIS4/profiles/default/python/plugins/"
```

On Windows, copy/paste the `isobenefit_qgis` folder into
`%APPDATA%/QGIS/QGIS4/profiles/default/python/plugins`. Restart QGIS after linking.

### CI

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs the Rust tests,
builds and smoke-tests abi3 wheels on Linux/macOS/Windows, runs the plugin's
headless test suite against a freshly built wheel, builds the plugin zip, and
(on a `v*` version tag, e.g. `v0.1.0`) publishes the engine to PyPI via trusted publishing.

## Packaging the plugin

The plugin zip is the `isobenefit_qgis/` folder only (never `core/`):

```bash
zip -r isobenefit_qgis.zip isobenefit_qgis -x '*/__pycache__/*' '*.pyc'
```

## Licensing

This project is licensed under the **GNU AGPL-3.0-or-later** (see [LICENSE](LICENSE)).

## Original version

This work is an outgrowth of the [original work](https://github.com/mitochevole/isobenefit-cities)
(forked to [BSP-isobenefit-original](https://github.com/UCL/BSP-isobenefit-original))
developed by Michele Voto and Luca D'Acci, and has been developed as part of the
Future Urban Growth project at the Bartlett School of Planning.

## Website

A full overview — the concept, the workflow and its outputs, installation, and the
parameters — is published from [`website/`](website/) in this repo:

**<https://github-pages.ucl.ac.uk/BSP-isobenefit-qgis-plugin/>**

(The previous standalone `BSP-isobenefit-urbanism` repo is deprecated; the site now
lives here and deploys via [`.github/workflows/website.yml`](.github/workflows/website.yml).)

## References

- [PyQGIS](https://qgis.org/pyqgis/3.28/)
- [PyQGIS Developer Cookbook](https://docs.qgis.org/testing/en/docs/pyqgis_developer_cookbook/index.html)
- [maturin](https://www.maturin.rs/) · [PyO3](https://pyo3.rs/)
