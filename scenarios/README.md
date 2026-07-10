# Scenario library

Each subfolder is one worked scenario: a real place with prepared input layers and a full
parameter set. The website's scenario page describes each one:
<https://github-pages.ucl.ac.uk/BSP-isobenefit-qgis-plugin/scenarios/>

Folder anatomy:

- `extents*.geojson` — the formal simulation boundary (or boundaries, e.g. Dnipro's pilot areas
  A and B), in the scenario's metric CRS.
- `params.json` — the plugin's parameters format. Load it via the run dialog's *Load parameters*
  button; every plugin run also writes this format back as a `*_params.json` sidecar next to its
  output. Additional presets (e.g. `params_B.json`) cover further pilot areas.
- `built / green / centres / unbuildable / streets / stops / stations / railways / industrial
  .geojson` — OSM input layers, pre-fetched with the plugin's own extraction rules.
- `osm_download_extent.geojson` — the convex hull of the extents features, the window the OSM
  data was fetched for.
- Scenario-specific extras, e.g. Dnipro's curated `centralities.csv` / `centralities.geojson`.

Refresh or add a scenario's OSM layers with:

```bash
.venv/bin/python scripts/fetch_scenario.py scenarios/<name>
```

The fetch caches the raw Overpass response as `_overpass_cache.xml` (delete it once the layers
look right; it is not committed). Scenarios marked draft on the website have indicative
parameters and draft boundaries, to be confirmed against local plans and norms.
