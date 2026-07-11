## The scenarios in detail

Each scenario is a real place with a worked parameter set: local density norms translated into the
plugin's dials, prepared input layers, and a population target to grow toward. The aim is that a
scenario can be rerun by anyone with the same data and the same settings.

Every scenario lives in a folder under
[`scenarios/`](https://github.com/UCL/BSP-isobenefit-qgis-plugin/tree/main/scenarios) in the
repository, containing:

- `extents*.geojson`, the formal simulation boundary (or boundaries), in the scenario's metric CRS;
- `params.json`, the full dial set in the plugin's parameters format. The run dialog's
  *Load parameters* button reads this file directly, and every plugin run writes the same format
  back as a `*_params.json` sidecar next to its output, so any past run can be reloaded too;
- the OSM input layers (`built`, `green`, `centres`, `unbuildable`, `streets`, `stops`,
  `stations`, `railways`, `industrial`), pre-fetched with the plugin's own extraction rules by
  `scripts/fetch_scenario.py`. The download window is the convex hull of the extents features, so
  one fetch covers every pilot area; the hull is kept as `osm_download_extent.geojson`;
- `steep.geojson`, terrain slope bands (15° / 20° / 25° / 30°) from the Copernicus GLO-30
  elevation model. This ships as a separate, editable layer so local knowledge can trim or extend
  it; the scenario's `slope_max_deg` parameter says which bands preclude development. Where it
  applies, review the layer, then merge the selected bands into the unbuildable layer for a run.

Every scenario downloads as a single ZIP (extents, all input layers including the editable
`steep.geojson`, and the parameter presets), or browse the folders on
[GitHub](https://github.com/UCL/BSP-isobenefit-qgis-plugin/tree/main/scenarios):

| # | Scenario | Theme | Status | Download |
|---|---|---|---|---|
| 1 | [Cambourne, UK](#cambourne) | New-settlement growth (the reference demo) | Worked | [ZIP](cambourne.zip) |
| 2 | [Dnipro, Ukraine](#dnipro) | Post-war regeneration and edge growth, DBN norms | Worked | [ZIP](dnipro.zip) (areas A + B) |
| 3 | [Crews Hill, London](#crews-hill) | Green-belt release at the metropolitan edge | Draft | [ZIP](london_crews_hill.zip) |
| 4 | [Celina, Texas](#celina) | US suburbia at the metropolitan fringe | Draft | [ZIP](celina_tx.zip) |
| 5 | [Kigali, Rwanda](#kigali) | Plan-guided rapid urbanisation | Draft | [ZIP](kigali_east.zip) |
| 6 | [Medellín, Colombia](#medellin) | Planned hillside expansion on steep terrain | Draft | [ZIP](medellin_pajarito.zip) |
| 7 | [Freiburg, Germany](#freiburg) | Validation against celebrated walkable districts | Draft | [ZIP](freiburg_rieselfeld.zip) |

<h2 id="cambourne">1. Cambourne, UK: the reference demo</h2>

A fast-growing Cambridgeshire new settlement and the worked example that threads the whole
[overview page](./). The scenario folder is the demo project
([`scenarios/cambourne/`](https://github.com/UCL/BSP-isobenefit-qgis-plugin/tree/main/scenarios/cambourne),
with `cambourne.qgz`): a 30,000-person target across the demo extents, 50 m cells, EPSG:27700,
400 m walks, and tiers of 6,000 / 3,000 / 1,500 people/km² at shares 0.2 / 0.3 / 0.5. The
overview page's own demonstrators use a smaller 4.2 km window with a 12,000-person target.

<h2 id="dnipro">2. Dnipro, Ukraine: regeneration and edge growth under DBN norms</h2>

Two pilot districts run as separate simulations
([`scenarios/dnipro/`](https://github.com/UCL/BSP-isobenefit-qgis-plugin/tree/main/scenarios/dnipro)):
**Area A (regeneration)**, the central right bank with riverside brownfield and infill, and
**Area B (edge growth)** on the left bank, where the Samara floodplain is a hard unbuildable
limit. Both use UTM zone 36N (EPSG:32636) and a 25 m grid (Area B may coarsen to 30 m). The
boundaries are drafts to be adjusted in QGIS; `params.json` holds Area A and `params_B.json`
Area B.

### Density tiers from the Ukrainian norms (DBN Б.2.2-12)

Residential densities follow the national planning norms, assuming an average household of
**2.55 persons per dwelling**. The DBN net residential density band is 150–450 persons/ha, up to
+20% in large cities under conditions; Dnipro qualifies. The three built forms map directly onto
the plugin's three density tiers (1 person/ha = 100 people/km²):

| Tier | Built form (Ukraine) | DBN storeys | Net density (persons/ha) | Avg dwelling (m²) | Plugin density (people/km²) | Persons per 25 m cell |
|---|---|---|---|---|---|---|
| Low | 1–4 storey, садибна / townhouse | 1–4 | 120 | 70 | **12,000** | 7.5 |
| Medium | 5–9 storey mid-rise | 5–9 | 250 | 58 | **25,000** | 15.6 |
| High | 10–25 storey high-rise | 10–25 | 400 | 52 | **40,000** | 25.0 |

These are *net* residential densities (people per hectare of residential land), which suits the
plugin's per-cell accounting at a fine grid. The shares are a scenario dial: Area A starts at
0.2 / 0.6 / 0.2 (high / medium / low) for a medium-led compact reconstruction, Area B at
0.1 / 0.4 / 0.5 for a low-to-medium edge mix.

### Target populations

Buildable area is the part of each large district realistically available for new housing (about
7% of Area A's 8,756 ha and 9% of Area B's 5,515 ha). The target population is buildable area
times net density, and is itself a scenario dial:

| Pilot area | Buildable area (ha) | Net density (persons/ha) | Target population |
|---|---|---|---|
| Area A, regeneration | 613 | 250 | **153,250** |
| Area B, edge growth | 496 | 200 | **99,200** |

### Model parameters

| Parameter | Meaning in the model | Area A | Area B | Plugin control | Notes |
|---|---|---|---|---|---|
| Walkable distance (m) | Max distance from a new home to the nearest centre | 400 | 400 | Centre walk / Green walk | Also run 800 m for comparison |
| Minimum green span | Smallest protected green corridor kept | tune | tune | Min green span (m) | So existing parks and the river edge survive |
| Density level | People per built cell (see the tiers above) | Medium | Low–Med | Density tiers + shares | High for compact reconstruction |
| Centrality seeding | Chance the model adds a new local centre | 0.8 | 0.8 | Dispersed development | 0 disables new centres; the sheet's 0.8 maps to the Moderate/Aggressive end of the dial |
| Cell resolution (m) | Grid cell size (spatial precision) | 25 | 25–30 | Grid size (m) | Coarser for large districts |
| CRS | Coordinate system (metres) | EPSG:32636 | EPSG:32636 | CRS picker | UTM 36N, Dnipro |
| Time steps | Iterations until the target population is reached | model | model | Max iterations | The run stops itself at the target |

### Curated centralities

Twenty-two real attraction points (city centre, stations, markets, hospitals, universities,
malls, parks) are curated in
[`centralities.csv`](https://github.com/UCL/BSP-isobenefit-qgis-plugin/blob/main/scenarios/dnipro/centralities.csv)
and `centralities.geojson`, each with a type and a relative pull weight (3 city-wide, 2 district,
1.5 local) and tagged with the pilot area it falls in. Coordinates are approximate; snap to OSM in
QGIS. The plugin's Urban centres input currently uses the point locations (weights are recorded
for future use); the OSM-derived `centres.geojson` is the uncurated alternative.

### Input layers

All layers share one extent, one CRS (EPSG:32636) and one cell size, with consistent nodata.
`<area>` is `A` or `B`.

| # | Layer / file | Plugin role | Geometry | Key attribute | Deliver as | Dnipro source |
|---|---|---|---|---|---|---|
| 1 | `extent_<area>` | Pilot study-area boundary | Polygon (1 feature) | — | GeoJSON / GPKG | Drawn in QGIS / Google Maps |
| 2 | `built_<area>` | Existing built-up areas | Polygons | `built=1` | GeoJSON → GeoTIFF | OSM buildings; city GIS; satellite |
| 3 | `green_<area>` | Green areas to preserve | Polygons | `preserve=1` | GeoJSON → GeoTIFF | OSM landuse/leisure; ESA WorldCover |
| 4 | `centralities_<area>` | Shops / services / centres / transport | Points | `type`, `weight` | GeoJSON / CSV(xy) | Centralities sheet + OSM |
| 5 | `unbuildable_<area>` | Water / floodplain / slopes / hazards | Polygons | `reason` | GeoJSON → GeoTIFF | OSM water; DEM slope; flood maps |

The scenario folder already holds OSM first drafts of layers 2–5 for the whole hull; local
sources then refine them.

<h2 id="crews-hill">3. Crews Hill, London: a green-belt release (draft)</h2>

The Crews Hill area of Enfield, at London's northern edge inside the M25, is one of the largest
green-belt releases proposed in an emerging London local plan, at about 5,500 homes around an
existing rail station. It tests the model on the green-belt question directly: can a release
deliver a walkable settlement rather than car-led sprawl, and what green network survives?
Folder: [`scenarios/london_crews_hill/`](https://github.com/UCL/BSP-isobenefit-qgis-plugin/tree/main/scenarios/london_crews_hill).

- **Tiers (indicative UK design-code values, household 2.36):** 9,500 / 15,500 / 26,000
  people/km² (40 / 65 / 110 dwellings/ha), shares 0.3 / 0.5 / 0.2 (low / medium / high).
- **Target:** about 13,000 new residents (5,500 homes). **Walks:** 800 m to a centre, 400 m to
  green. **Grid:** 25 m, EPSG:27700. **Dispersal:** off (a contiguous urban extension).
- **Status:** draft boundary and indicative numbers, to be confirmed against the local plan.

<h2 id="celina">4. Celina, Texas: US suburbia at the fringe (draft)</h2>

Celina, on the Dallas–Fort Worth northern fringe, has repeatedly been the fastest-growing city in
the United States, converting ranchland into master-planned subdivisions at speed. It tests the
model on low-density, leapfrog suburbia: what do walkable-access rules change when the low tier
dominates and the street grid is coarse?
Folder: [`scenarios/celina_tx/`](https://github.com/UCL/BSP-isobenefit-qgis-plugin/tree/main/scenarios/celina_tx).

- **Tiers (indicative suburban values, household 2.5):** 1,500 / 7,500 / 18,500 people/km²
  (2.5 / 12 / 30 dwellings per acre), shares 0.6 / 0.3 / 0.1 (low / medium / high).
- **Target:** about 50,000 new residents, of the order of the city's own growth projections.
  **Walks:** 800 m. **Grid:** 30 m, EPSG:32614. **Dispersal:** moderate (leapfrog growth is
  characteristic).
- **Status:** draft boundary and indicative numbers, to be confirmed against city projections.

<h2 id="kigali">5. Kigali, Rwanda: plan-guided rapid urbanisation (draft)</h2>

Kigali is Africa's clearest case of plan-guided growth: a city-wide master plan steers rapid
urbanisation into designated expansion zones with managed green networks, which is exactly the
regime this model speaks to, simple rules guiding new growth while protecting green. The draft
boundary covers the eastern expansion direction (toward Ndera and Masaka).
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
of metrocable-served social housing at height. Steep terrain makes the model's green network and
short walks decisive, and the slopes themselves belong in the unbuildable layer, so the scenario
tests the rules where topography, not policy, is the binding constraint.
Folder: [`scenarios/medellin_pajarito/`](https://github.com/UCL/BSP-isobenefit-qgis-plugin/tree/main/scenarios/medellin_pajarito).

- **Tiers (indicative values, household 3.1):** 8,000 / 18,000 / 35,000 people/km², shares
  0.3 / 0.4 / 0.3 (low / medium / high); the high tier reflects the area's housing towers.
- **Target:** about 40,000 new residents. **Walks:** 400 m. **Grid:** 25 m, EPSG:32618.
  **Dispersal:** moderate.
- **Terrain:** slopes over 20° preclude development (`slope_max_deg: 20`; about 30% of the study
  window). The bands are in the editable `steep.geojson`, from Copernicus GLO-30.
- **Status:** draft boundary and indicative numbers, to be confirmed against the POT zoning.

<h2 id="freiburg">7. Freiburg, Germany: a validation scenario (draft)</h2>

Rieselfeld and Vauban in western Freiburg are among Europe's most celebrated walkable districts.
This scenario runs the model where a known-good answer already exists: delete the two districts
from the `built` layer (keeping a reference copy), let the model regrow the same land toward the
districts' real population, and compare the result against what was actually built. Agreement
strengthens confidence in the model; disagreement locates what the rules miss.
Folder: [`scenarios/freiburg_rieselfeld/`](https://github.com/UCL/BSP-isobenefit-qgis-plugin/tree/main/scenarios/freiburg_rieselfeld).

- **Tiers (indicative values, household 2.0, centred on the districts' real densities):**
  8,000 / 14,000 / 22,000 people/km², shares 0.2 / 0.6 / 0.2 (low / medium / high).
- **Target:** about 16,000 residents (the two districts' combined population). **Walks:** 400 m.
  **Grid:** 25 m, EPSG:25832. **Dispersal:** off (both districts are contiguous extensions).
- **Status:** draft boundary; the validation protocol (remove, regrow, compare) is described in
  the folder's `params.json` notes.

## Adding a scenario

Copy an existing folder's structure: `extents.geojson` in a local metric CRS, a `params.json`
(the dialog's *Load parameters* button must accept it), and a short context note here covering the
local density norms translated to three tiers and shares, the target population with its basis,
and any curated centralities. Then fetch the OSM layers:

```bash
.venv/bin/python scripts/fetch_scenario.py scenarios/<name>
```
