# BSP Isobenefit Urbanism QGIS Plugin

A QGIS plugin for exploring walkable urban development on real-world datasets,
after D'Acci's Isobenefit Urbanism: every new home within a walk of a mixed-use
centre (shops, services) and of green space.

This is research software for discussion and debate. Its scenarios are speculative
sketches of walkable growth in a place, made to be discussed with domain experts and
weighed by them when developing actual planning strategies and developments. The
scenarios are not plans to build from.

📖 **Documentation & overview website:** <https://github-pages.ucl.ac.uk/BSP-isobenefit-qgis-plugin/>

⬇ **Get the plugin:** the ready-to-install `isobenefit_qgis.zip` is attached to every
[GitHub release](https://github.com/UCL/BSP-isobenefit-qgis-plugin/releases/latest)
(QGIS: *Plugins → Manage and Install Plugins → Install from ZIP*). A listing in the
official QGIS plugin repository as an experimental plugin is in progress.

## Repository layout

This repository contains two deliverables:

- [`isobenefit_qgis/`](isobenefit_qgis/): the **QGIS plugin** (thin). It handles the UI and
  all GIS IO (reading layers, reprojection, rasterization, writing rasters,
  temporal animation), and depends only on libraries QGIS already bundles (numpy +
  GDAL) plus the simulation engine below.
- [`core/`](core/): the **simulation engine** (`isobenefit`), a Rust
  extension built with PyO3/maturin and published to PyPI as abi3 wheels. It is
  pure compute (arrays in, arrays out) and never imports QGIS.

The split exists because the QGIS plugin repository does not permit shipping
binaries: the compiled engine is installed from PyPI rather than bundled.

## Installation

1. Either install from the plugin manager, once the repository listing is live
   (QGIS 4: *Plugins → Manage and Install Plugins → Settings →
   "Show also experimental plugins"*, then search for **isobenefit**), or download
   `isobenefit_qgis.zip` from the
   [latest release](https://github.com/UCL/BSP-isobenefit-qgis-plugin/releases/latest)
   and use *Plugins → Manage and Install Plugins → Install from ZIP*.
2. The first time you run the plugin it will check for the `isobenefit`
   engine and, if it is missing, offer to install it into the QGIS Python
   environment for you. This needs an internet connection; **restart QGIS** once
   it finishes.

If the automatic install is not available or not working on your system, the dialog
shows the exact command to run yourself, which is simply:

```bash
<qgis-python> -m pip install "isobenefit>=0.12.11,<0.13"
```

## Usage

Open the plugin and choose an output folder and run name, a polygon **extents** layer, and a
projected **CRS**. Optionally add layers for existing built areas, green space,
unbuildable land, urban centres, public-transport stops, rail/tram stations, and a
street network. No layers prepared? The companion **Extract from OpenStreetMap** tool
downloads them for an area of interest.

Set the parameters and run. A single run is written iteration-by-iteration as a
categorical GeoTIFF and loaded as a temporal animation (press play in the Temporal
Controller). An **ensemble** of runs instead produces development-likelihood layers
plus **idealised planning scenarios** to compare (the existing fabric, the raw
as-grown state, and moderately and tightly clustered centre arrangements),
alongside a `_report.txt` summarising the run. A demo project is in
[`scenarios/cambourne/`](scenarios/cambourne/) (`cambourne.qgz`), and further worked scenarios sit alongside it in [`scenarios/`](scenarios/).

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
count; see [`core/README.md`](core/README.md).

### QGIS plugin

Lint with `ruff check isobenefit_qgis`.

The plugin's pure pipeline (grid/routing/OSM helpers) has a headless test suite that
needs no QGIS; run it against a locally built engine wheel:

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

### Regenerating everything

Every artefact rebuilds from committed sources. Setup once with `uv sync` (dev environment +
engine) and `npm install` in `website/`. Then there are four commands:

```bash
.venv/bin/python scripts/verify.py                            # full local verification (mirrors CI)
.venv/bin/python scripts/fetch_scenario.py scenarios/<name>   # rebuild a scenario's layers (OSM + terrain)
.venv/bin/python scripts/render_scenario_gallery.py           # rebuild the website gallery, ZIPs and presets
cd website && npm run dev                                     # preview the site (npm run build for production)
```

The website's schematic figures regenerate with `website/scripts/diagrams.py` and
`demonstrators.py`. Scenario data is committed, so the network is only needed for a
deliberate re-fetch; gallery runs use fixed seeds, so regeneration is deterministic.
A release bumps the version in `core/Cargo.toml`, `pyproject.toml` and
`isobenefit_qgis/metadata.txt` in lockstep, then a `v*` tag publishes.

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

A full overview (the concept, the workflow and its outputs, installation, and the
parameters) is published from [`website/`](website/) in this repo:

**<https://github-pages.ucl.ac.uk/BSP-isobenefit-qgis-plugin/>**

(The previous standalone `BSP-isobenefit-urbanism` repo is deprecated; the site now
lives here and deploys via [`.github/workflows/website.yml`](.github/workflows/website.yml).)

## References

- [PyQGIS](https://qgis.org/pyqgis/3.28/)
- [PyQGIS Developer Cookbook](https://docs.qgis.org/testing/en/docs/pyqgis_developer_cookbook/index.html)
- [maturin](https://www.maturin.rs/) · [PyO3](https://pyo3.rs/)
