---
layout: ../layouts/BaseLayout.astro
title: Plugin guide
---

# Plugin guide

This page covers installation, a first run, the data downloader, the run dialog, the outputs,
and troubleshooting. The [introduction](../) explains the model itself, the
[theory page](../theory/) relates it to the published model, and the
[scenario library](../scenarios/) provides prepared data and parameters to start from.

## Install

1. In QGIS (4.x; the 3.40 LTR should also work, but is untested): *Plugins → Manage and Install
   Plugins → Settings*, tick *"Show also experimental plugins"*, then search for
   **isobenefit** and install it.
2. Two toolbar buttons appear: **Isobenefit Urbanism** (the simulation) and **Extract from
   OpenStreetMap** (the data downloader).
3. The first time you run the simulation, the plugin checks for its `isobenefit` engine and, if
   missing, offers to install it into the QGIS Python environment (this requires an internet connection).
   **Restart QGIS** once it finishes.

If the automatic install is blocked (a locked-down environment), run the shown command yourself
with the QGIS Python:

```
<qgis-python> -m pip install "isobenefit>=0.12.6,<0.13"
```

## Quick start: your first run

The fastest route uses the OSM downloader for the data and accepts most defaults.

1. Zoom the map to a place you want to test (a town and its surroundings; the area must be more
   than twice the walking distance across, and a window of a few kilometres works well).
2. Open **Extract from OpenStreetMap**. Click *Draw area on map…*: the dialog hides,
   left-clicks add corners, a right-click finishes the polygon (Esc cancels).
3. Leave all datasets ticked, choose an output GeoPackage path, and press **Fetch**. The layers
   are saved to the GeoPackage and added to the project as an "OSM" group.
4. Open **Isobenefit Urbanism**. The dialog pre-fills its layer pickers from the OSM download,
   suggests a local projected CRS, and validates as you type; the status line under the form
   lists what is still missing.
5. Choose an **output folder** and give the run a **name**. All of the run's files take that
   name.
6. Set the **target population**: the number of new residents to house. Existing buildings are
   context only and are never counted.
7. Check the **Development density** group: three densities (people per km²) and the share of
   new blocks built at each. The shares must sum to 1; the feedback line shows the running
   total and the resulting mean density.
8. Press **Run**. The simulation runs as a background task; the progress bar tracks it, and a
   run can be cancelled safely. With the default *Development likelihood* mode, several layers
   load on completion; start with the *moderately clustered centres* plan.
9. The run's full settings are saved next to the output as `<name>_params.json`. To repeat or
   adjust the run later, use *Load parameters* at the top of the dialog.

To start from a prepared case instead, use a scenario download, described in the next section.

<h2 id="use-a-scenario">Using a downloaded scenario</h2>

Each entry in the [scenario library](../scenarios/) downloads as one ZIP with the data and
parameters for a run.

- **Contents**: `extents*.geojson` (the study boundary), the input layers (`built`, `green`,
  `centres`, `unbuildable`, `industrial`, `streets`, `railways`, `stops`, `stations`), the
  terrain bands (`steep.geojson`), and one or more `params*.json` presets.
- **Layers**: drag the GeoJSON files onto the QGIS map.
- **Local adjustment**: the layers are ordinary editable data, and the scenarios are prepared
  on the assumption that they will be revised. The built fabric, the extents, the green areas
  and the unbuildable land can each be corrected wherever local knowledge or stakeholder
  feedback improves on the downloaded state.
- **Centre seeding**: the urban-centres layer steers the simulation and is the most direct
  expression of local intent. A point or area added where a mixed-use centre is planned draws
  growth around it; removing one examines a future without it.
- **Terrain**: `steep.geojson` holds slope bands (15° / 20° / 25° / 30°) from the Copernicus
  GLO-30 elevation model. The bands at or above the scenario's maximum slope belong in the
  unbuildable layer (*Vector → Data Management Tools → Merge Vector Layers*).
- **Parameters**: in **Isobenefit Urbanism**, *Load parameters* with the scenario's
  `params.json` fills in the dialog. Dnipro provides one preset per pilot area.
- **Running**: select the layers in the *Input layers* group, confirm the suggested CRS,
  choose an output folder and run name, and press **Run**.
- **Reproducing a published panel**: the explorer's per-run parameter files carry the exact
  seed and settings at the scenario's full resolution. The gallery previews are computed on a
  coarser grid, so a rerun gives a finer, slower result of the same kind rather than an
  identical image.

## The Extract from OpenStreetMap tool

- **Area of interest**: a polygon, either drawn on the map (left-click corners, right-click to
  finish, Esc to cancel) or the union of an existing polygon layer. The polygon's bounding box
  drives the download; results are trimmed to the polygon.
- **Datasets**: built-up areas, green space, mixed-use centres, industrial land, streets,
  railways, public-transport stops, rail/tram stations, and unbuildable land (water, airfields,
  military, quarries, plus buffered motorway/railway/river corridors). Untick anything not
  needed.
- **Output**: one GeoPackage; every dataset becomes a layer in it, added to the project under an
  "OSM" group.

Downloading is deliberately separate from simulating: the layers are on disk, so they can be
edited, corrected, or swapped before any run. The simulation dialog recognises the downloaded
layers and pre-selects them.

## The run dialog, group by group

**Parameters.** *Load parameters* repopulates the dialog from a previous run's
`*_params.json` sidecar or from a scenario preset. Every run writes such a sidecar next to its
output.

**Simulation.**

| Field | Default | What it does |
| --- | --- | --- |
| Max iterations | 100 | Cap on growth steps; a run stops early at the target population |
| Grid size (m) | 50 | Cell size of the simulation grid |
| Target population | 100,000 | New residents to house; growth stops once reached (checked between iterations, so the final count can slightly overshoot) |
| Build probability | 0.25 | Per-step chance an eligible cell develops (the growth rate) |
| Dispersed development | Moderate | Leapfrog rate: Off / Moderate / Aggressive |
| Random seed | 42 | The same seed reproduces the same run and the same ensemble, independent of core count |

**Walkable access.** Centre walk (400 m) and Green walk (400 m): how far people walk to a
mixed-use centre and to a park. The simulation grows by the larger of the two; the finished plan
is scored against each separately.

**Post-processing.**

| Field | Default | What it does |
| --- | --- | --- |
| Optimise centre placement | on | Re-position centres central to their development, add where under-served, cull redundant ones; saves moderately and tightly clustered options. Off keeps the grown centres (one plan) |
| Centre area (m² per person) | 20 | Mixed-use centre land provided per new resident served |
| Min settlement area (ha) | 2 | Detached new clusters smaller than this revert to green |
| Min green span (m) | 400 | A green patch must span this to count as a park; also a build rule protecting corridors |

**Development density.** Three densities (people per km²) for the high, medium and low tiers,
each with a share. The dialog requires positive, strictly descending densities and shares
between 0 and 1 that sum to 1; the feedback line shows the running total and the mean. Every new block is built at one of
the three densities; post-processing arranges the highest nearest the mixed-use centres.

**Output.** *Development likelihood* (the default) blends many runs; the *Detail* picker sets how
many (Quick 10 / Standard 50 / Thorough 100). Untick it for a single run written as a growth
animation. The output folder and run name determine where the run's files land (see Outputs
below); the CRS must be a local projected CRS (a suggestion
is made from the extents layer; geographic lat/lon CRSs are rejected so the model always works in
metres).

**Input layers.** Extents (required, polygon) plus optional existing urban, existing green,
unbuildable, urban centres (points or polygon areas), PT stops, rail/tram stations, and a street
network (line layer; when given, walking distances are measured along it). All layers may be in
any CRS; they are reprojected to the chosen run CRS.

The **Run button stays disabled** until four things are set: an extents layer, an output folder
and run name, a projected CRS, and valid densities and shares. The red status line names whichever are
missing.

## Outputs and how to read them

**Ensemble mode** writes a family of files into the output folder, sharing the run name:
`<name>.tif` (the
built and green likelihood bands), `<name>_existing.tif` (the starting fabric),
`<name>_pre.tif` (the chosen run before post-processing), `<name>_moderate.tif` and
`<name>_tight.tif` (the two clustering options, each coloured by density tier: built as a
yellow-to-brown ramp, mixed-use centres as a reds ramp, existing fabric muted),
`<name>_report.txt` (parameters and per-option statistics) and `<name>_params.json` (the
reloadable settings). QGIS loads them as a layer group in that order.

Every population figure counts **new residents only**; existing fabric is assumed served by its
own centres. The per-person readouts follow: m² of mixed-use centre per person is new centre
land over new residents, and m² of green per person is new green over new residents. Coverage
percentages include every home, existing and new.

**Single-run mode** writes one band per growth step. QGIS loads it as a temporal animation: open
*View → Panels → Temporal Controller*, press the play button, and the town grows step by step.

## Troubleshooting

- **The engine installed but the tools stay disabled**: restart QGIS; the check runs again on
  the next launch.
- **Run is greyed out**: read the red status line; it names the missing pieces (extents layer,
  output folder and run name, projected CRS, densities/shares).
- **"Select a local projected CRS"**: geographic (degrees) CRSs are rejected. Accept the
  suggested UTM zone, or pick the national grid for the area.
- **"The extents are too small"**: the area must exceed twice the walking distance in both
  directions; enlarge the extents polygon or shorten the walks.
- **OSM fetch fails**: the Overpass servers are shared and sometimes busy; retry after a minute,
  or draw a smaller area.
- **Finding the logs**: *View → Panels → Log Messages*, under the **Isobenefit** tab, records
  grid size, per-stage progress, per-option metrics and any warnings.
- **Recovering a run's settings**: they are saved next to the output raster as
  `<name>_params.json`, and load back with the dialog's *Load parameters* button.
