## The scenarios in detail

Each scenario is a real place with a worked parameter set: local density norms translated into
the plugin's controls, prepared input layers, and a population target to grow toward. Anyone can
rerun a scenario with the same data and settings.

Every scenario lives in a folder under
[`scenarios/`](https://github.com/UCL/BSP-isobenefit-qgis-plugin/tree/main/scenarios) in the
repository, containing:

- `extents*.geojson`, the formal simulation boundary (or boundaries), in the scenario's metric CRS;
- `params.json`, the full parameter set in the plugin's format. The run dialog's
  *Load parameters* button reads this file directly, and every plugin run writes the same format
  back as a `*_params.json` sidecar next to its output, so any past run can be reloaded too;
- the OSM input layers (`built`, `green`, `centres`, `unbuildable`, `streets`, `stops`,
  `stations`, `railways`, `industrial`), pre-fetched with the plugin's own extraction rules by
  `scripts/fetch_scenario.py`. The download window is the convex hull of the extents features, so
  one fetch covers every pilot area; the hull is kept as `osm_download_extent.geojson`;
- `steep.geojson`, terrain slope bands (15° / 20° / 25° / 30°) from the Copernicus GLO-30
  elevation model. This ships as a separate, editable layer so local knowledge can trim or extend
  it; the scenario's `slope_max_deg` parameter specifies which bands preclude development. Where it
  applies, review the layer, then merge the selected bands into the unbuildable layer for a run.

Every scenario downloads as a single ZIP (extents, all input layers including the editable
`steep.geojson`, and the parameter presets), or browse the folders on
[GitHub](https://github.com/UCL/BSP-isobenefit-qgis-plugin/tree/main/scenarios):

| # | Scenario | Theme | Status | Download |
|---|---|---|---|---|
| 1 | [Cambourne, UK](#cambourne) | New-settlement growth (the reference demo) | Worked | [ZIP](cambourne.zip) |
| 2 | [Dnipro, Ukraine](#dnipro) | Regeneration and edge growth | Worked | [ZIP](dnipro.zip) |
| 3 | [Crews Hill, London](#crews-hill) | Green-belt release at the metropolitan edge | Draft | [ZIP](london_crews_hill.zip) |
| 4 | [Celina, Texas](#celina) | US suburbia at the metropolitan fringe | Draft | [ZIP](celina_tx.zip) |
| 5 | [Kigali, Rwanda](#kigali) | Plan-guided rapid urbanisation | Draft | [ZIP](kigali_east.zip) |
| 6 | [Medellín, Colombia](#medellin) | Planned hillside expansion on steep terrain | Draft | [ZIP](medellin_pajarito.zip) |
| 7 | [Freiburg, Germany](#freiburg) | Validation against built walkable districts | Draft | [ZIP](freiburg_rieselfeld.zip) |

<h2 id="cambourne">1. Cambourne, UK: the reference demo</h2>

Cambourne is a fast-growing Cambridgeshire new settlement, and the worked example used
throughout the [overview page](../). The scenario folder is the demo project
([`scenarios/cambourne/`](https://github.com/UCL/BSP-isobenefit-qgis-plugin/tree/main/scenarios/cambourne),
with `cambourne.qgz`): a 30,000-person target across the demo extents, 50 m cells, EPSG:27700,
400 m walks, and tiers of 6,000 / 3,000 / 1,500 people/km² at shares 0.2 / 0.3 / 0.5. The
overview page's own demonstrators use a smaller 4.2 km window with a 12,000-person target.

As the reference demo, Cambourne also illustrates the input layers every scenario shares:

| Layer | Plugin role | Geometry |
|---|---|---|
| `extents` | The simulation boundary; one or more polygons | Polygons |
| `built` | Existing built fabric, frozen as context | Polygons |
| `green` | Green space to preserve | Polygons |
| `unbuildable` | Water, floodplain, slopes and other exclusions | Polygons |
| `centres` | Existing or planned urban centres | Points or polygons |
| `streets` | Walking network for routed distances | Lines |
| `stops`, `stations` | Public transport; stations anchor centres | Points |
| `railways`, `industrial` | Carved as barriers / unbuildable | Lines / polygons |

<h2 id="dnipro">2. Dnipro, Ukraine: regeneration and edge growth</h2>

One window covers two growth areas on either side of the river
([`scenarios/dnipro/`](https://github.com/UCL/BSP-isobenefit-qgis-plugin/tree/main/scenarios/dnipro)):
the central right bank (regeneration and infill) and the left-bank edge, where the Samara
floodplain is a hard unbuildable limit. The extents layer holds both boundaries as drafts to
adjust in QGIS, and a single run grows both areas together. Density tiers and the population
target follow the national residential norms and load from `params.json`; urban centres are
supplied as a plain point layer. 25 m grid, EPSG:32636.

<h2 id="crews-hill">3. Crews Hill, London: a green-belt release (draft)</h2>

The Crews Hill area of Enfield, at London's northern edge inside the M25, is one of the largest
green-belt releases proposed in an emerging London local plan, at about 5,500 homes around an
existing rail station. The scenario examines whether a release can deliver a walkable
settlement rather than car-led sprawl, and which green network survives.
Folder: [`scenarios/london_crews_hill/`](https://github.com/UCL/BSP-isobenefit-qgis-plugin/tree/main/scenarios/london_crews_hill).

- **Tiers (indicative UK design-code values, household 2.36):** 9,500 / 15,500 / 26,000
  people/km² (40 / 65 / 110 dwellings/ha), shares 0.3 / 0.5 / 0.2 (low / medium / high).
- **Target:** about 13,000 new residents (5,500 homes). **Walks:** 800 m to a centre, 400 m to
  green. **Grid:** 25 m, EPSG:27700. **Dispersal:** off (a contiguous urban extension).
- **Status:** draft boundary and indicative numbers, to be confirmed against the local plan.

<h2 id="celina">4. Celina, Texas: US suburbia at the fringe (draft)</h2>

Celina, on the Dallas–Fort Worth northern fringe, has repeatedly been the fastest-growing city in
the United States, converting ranchland into master-planned subdivisions at speed. The scenario
examines what the walkable-access rules change where the low density tier dominates and the
street grid is coarse.
Folder: [`scenarios/celina_tx/`](https://github.com/UCL/BSP-isobenefit-qgis-plugin/tree/main/scenarios/celina_tx).

- **Tiers (indicative suburban values, household 2.5):** 1,500 / 7,500 / 18,500 people/km²
  (2.5 / 12 / 30 dwellings per acre), shares 0.6 / 0.3 / 0.1 (low / medium / high).
- **Target:** about 50,000 new residents, of the order of the city's own growth projections.
  **Walks:** 800 m. **Grid:** 30 m, EPSG:32614. **Dispersal:** moderate (leapfrog growth is
  characteristic).
- **Status:** draft boundary and indicative numbers, to be confirmed against city projections.

<h2 id="kigali">5. Kigali, Rwanda: plan-guided rapid urbanisation (draft)</h2>

Kigali manages rapid urbanisation through a city-wide master plan that steers growth into
designated expansion zones while protecting a network of green corridors and wetlands. The
model works on the same principle, a small set of rules guiding new growth under green
protection, so the scenario applies it to one designated expansion direction on the eastern
fringe, toward Ndera and Masaka.
Folder: [`scenarios/kigali_east/`](https://github.com/UCL/BSP-isobenefit-qgis-plugin/tree/main/scenarios/kigali_east).

- **Tiers (indicative values, household 4.3):** 10,000 / 20,000 / 35,000 people/km²
  (roughly 25 / 45 / 80 dwellings/ha), shares 0.4 / 0.4 / 0.2 (low / medium / high).
- **Target:** about 60,000 new residents for one expansion area. **Walks:** 400 m.
  **Grid:** 25 m, EPSG:32736. **Dispersal:** moderate.
- **Terrain:** slopes over 15° preclude development (`slope_max_deg: 15`, matching Rwanda's
  percent-slope planning limits; a few percent of the window). Bands in the editable
  `steep.geojson`, from Copernicus GLO-30.
- **Status:** draft boundary and indicative numbers, to be confirmed against the Kigali master
  plan zoning.

<h2 id="medellin">6. Medellín, Colombia: planned hillside expansion (draft)</h2>

Pajarito and Ciudadela Nuevo Occidente on Medellín's northwestern slopes are a planned expansion
of high-rise social housing served by the Metrocable. The scenario tests the growth rules where
topography is the binding constraint: steep terrain sits in the unbuildable layer, and the green
network and short walking distances have to work around it.
Folder: [`scenarios/medellin_pajarito/`](https://github.com/UCL/BSP-isobenefit-qgis-plugin/tree/main/scenarios/medellin_pajarito).

- **Tiers (indicative values, household 3.1):** 8,000 / 18,000 / 35,000 people/km², shares
  0.3 / 0.4 / 0.3 (low / medium / high); the high tier reflects the area's housing towers.
- **Target:** about 40,000 new residents. **Walks:** 400 m. **Grid:** 25 m, EPSG:32618.
  **Dispersal:** moderate.
- **Terrain:** slopes over 20° preclude development (`slope_max_deg: 20`; about 30% of the study
  window). The bands are in the editable `steep.geojson`, from Copernicus GLO-30.
- **Status:** draft boundary and indicative numbers, to be confirmed against the POT zoning.

<h2 id="freiburg">7. Freiburg, Germany: a validation scenario (draft)</h2>

Rieselfeld and Vauban in western Freiburg are two widely studied walkable districts, planned in
the 1990s and often cited as models of the form this plugin aims for. The scenario runs the
model where a good answer already exists: delete the two districts from the `built` layer
(keeping a reference copy), let the model regrow the same land toward the districts' real
population, and compare the result against what was built. The comparison shows which behaviours
the growth rules capture and which they miss.
Folder: [`scenarios/freiburg_rieselfeld/`](https://github.com/UCL/BSP-isobenefit-qgis-plugin/tree/main/scenarios/freiburg_rieselfeld).

- **Tiers (indicative values, household 2.0, centred on the districts' real densities):**
  8,000 / 14,000 / 22,000 people/km², shares 0.2 / 0.6 / 0.2 (low / medium / high).
- **Target:** about 16,000 residents (the two districts' combined population). **Walks:** 400 m.
  **Grid:** 25 m, EPSG:25832. **Dispersal:** off (both districts are contiguous extensions).
- **Status:** draft boundary; the validation protocol (remove, regrow, compare) is described in
  the folder's `params.json` notes.

## Adding a scenario

Scenario contributions happen in the repository; the steps are described in
[`scenarios/README.md`](https://github.com/UCL/BSP-isobenefit-qgis-plugin/tree/main/scenarios).
