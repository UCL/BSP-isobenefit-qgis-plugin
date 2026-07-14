# Scenario library

Each subfolder is one worked scenario: a real place with prepared input layers and a full
parameter set. The website's scenario page describes each one:
<https://github-pages.ucl.ac.uk/BSP-isobenefit-qgis-plugin/scenarios/>

Folder anatomy:

- `extents.geojson`: the formal simulation boundary, in the scenario's metric CRS. A
  scenario may hold several boundary features (e.g. Dnipro's two growth areas); one run
  grows them all together.
- `params.json`: the plugin's parameters format. Load it via the run dialog's *Load parameters*
  button; every plugin run also writes this format back as a `*_params.json` sidecar next to its
  output.
- `built / green / centres / unbuildable / streets / stops / stations / railways / industrial
  .geojson`: OSM input layers, pre-fetched with the plugin's own extraction rules.
- `osm_download_extent.geojson`: the convex hull of the extents features, the window the OSM
  data was fetched for.
- `steep.geojson`: terrain slope bands from Copernicus GLO-30, where terrain matters; the
  scenario's `slope_max_deg` says which bands preclude development.

Boundaries and parameters are indicative, to be confirmed against local plans and norms.

## Adding a scenario (contributors)

Everything runs from a repo checkout with the dev environment set up once:

```bash
git clone https://github.com/UCL/BSP-isobenefit-qgis-plugin
cd BSP-isobenefit-qgis-plugin
uv sync                      # dev environment + simulation engine
```

Then:

1. Create `scenarios/<name>/` with an `extents.geojson` in a local metric CRS (one or more
   boundary features) and a `params.json` the run dialog's *Load parameters* button accepts
   (copy a neighbour's and adjust). Translate the local density norms into the three tiers and
   shares, and set the target population with its basis.
2. Fetch the OSM layers for the extents:

   ```bash
   .venv/bin/python scripts/fetch_scenario.py scenarios/<name>
   ```

   The fetch caches the raw Overpass response as `_overpass_cache.xml` (delete it once the
   layers look right; it is not committed).
3. Review the layers in QGIS and refine from local sources where OSM falls short.
4. Rebuild the website gallery, ZIPs and presets, and add a short context note to
   `website/src/pages/_scenario_notes.md`:

   ```bash
   .venv/bin/python scripts/render_scenario_gallery.py
   ```
