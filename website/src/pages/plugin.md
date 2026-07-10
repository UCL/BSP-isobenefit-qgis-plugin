---
layout: ../layouts/BaseLayout.astro
title: Plugin guide
---

# Plugin guide

Everything needed to drive the plugin, from installation to reading the outputs. The
[introduction](./) explains how the model thinks; this page explains how to use it. The
[scenario library](./scenarios/) provides ready-made data and parameters to start from.

## Install

1. In QGIS (4.x; the 3.40 LTR should also work, but is untested): *Plugins → Manage and Install
   Plugins → Settings*, tick *"Show also experimental plugins"*, then search for
   **isobenefit** and install it.
2. Two toolbar buttons appear: **Isobenefit Urbanism** (the simulation) and **Extract from
   OpenStreetMap** (the data downloader).
3. The first time you run the simulation, the plugin checks for its `isobenefit` engine and, if
   missing, offers to install it into the QGIS Python environment (needs internet).
   **Restart QGIS** once it finishes.

If the automatic install is blocked (a locked-down environment), run the shown command yourself
with the QGIS Python:

```
<qgis-python> -m pip install "isobenefit>=0.12,<0.13"
```

## Quick start: your first run

The fastest route uses the OSM downloader for the data and accepts most defaults.

1. Zoom the map to a place you want to test (a town and its surroundings; the area must be at
   least twice the walking distance across, so 2 km or more).
2. Open **Extract from OpenStreetMap**. Click *Draw area on map…*: the dialog hides,
   left-clicks add corners, a right-click finishes the polygon (Esc cancels).
3. Leave all datasets ticked, choose an output GeoPackage path, and press **Fetch**. The layers
   download, land in the GeoPackage, and are added to the project as an "OSM" group. They are
   ordinary editable layers; adjust them if you want.
4. Open **Isobenefit Urbanism**. The dialog pre-fills its layer pickers from the OSM download,
   suggests a local projected CRS, and validates as you type; the status line under the form
   says exactly what is still missing.
5. Set an **output file** (a `.tif` path). This is usually the only required field left.
6. Set the **target population**: how many NEW residents to house. Existing buildings are
   context only and are never counted.
7. Check the **Development density** group: three densities (people per km²) and the share of
   new blocks built at each. The shares must sum to 1; the feedback line shows the running
   total and the resulting mean density.
8. Press **Run**. The simulation runs as a background task (watch the progress bar; cancelling
   is safe). With the default *Development likelihood* mode, several layers load when it
   finishes; start with the *moderately clustered centres* plan.
9. The run's full settings are saved next to the output as `<name>_params.json`. To repeat or
   tweak the run later, use *Load parameters* at the top of the dialog.

To start from a ready-made case instead, download a folder from the
[scenario library](https://github.com/UCL/BSP-isobenefit-qgis-plugin/tree/main/scenarios), load
its layers into QGIS, and load its `params.json` via the dialog's *Load parameters* button.

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

**Parameters.** *Load parameters* repopulates every dial from a previous run's
`*_params.json` sidecar or from a scenario preset. Every run writes such a sidecar next to its
output.

**Simulation.**

| Field | Default | What it does |
| --- | --- | --- |
| Max iterations | 100 | Cap on growth steps; a run stops early at the target population |
| Grid size (m) | 50 | Cell size of the simulation grid |
| Target population | 100,000 | NEW residents to house; growth stops once reached (checked between iterations, so the final count can slightly overshoot) |
| Build probability | 0.25 | Per-step chance an eligible cell develops (the growth rate) |
| Dispersed development | Moderate | Leapfrog rate: Off / Moderate / Aggressive |
| Random seed | 42 | The same seed reproduces the same run and the same ensemble, on any machine |

**Walkable access.** Centre walk (400 m) and Green walk (400 m): how far people walk to a
mixed-use centre and to a park. The simulation grows by the larger of the two; the finished plan
is scored against each separately.

**Post-processing.**

| Field | Default | What it does |
| --- | --- | --- |
| Optimise centre placement | on | Re-position centres central to their development, add where under-served, cull redundant ones; saves moderately- and tightly-clustered options. Off keeps the grown centres (one plan) |
| Centre area (m² per person) | 20 | Mixed-use centre land provided per NEW resident served |
| Min settlement area (ha) | 2 | Detached new clusters smaller than this revert to green |
| Min green span (m) | 400 | A green patch must span this to count as a park; also a build rule protecting corridors |

**Development density.** Three densities (people per km²) for the high, medium and low tiers,
each with a share. Guards: densities positive and strictly descending; each share in 0–1; shares
summing to 1 (the feedback line shows the total and the mean). Every new block is built at one of
the three densities; post-processing arranges the highest nearest the mixed-use centres.

**Output.** *Development likelihood* (the default) blends many runs; the *Detail* picker sets how
many (Quick 10 / Standard 50 / Thorough 100). Untick it for a single run written as a growth
animation. The output file must be a `.tif`; the CRS must be a local projected CRS (a suggestion
is made from the extents layer; geographic lat/lon CRSs are rejected so the model always works in
metres).

**Input layers.** Extents (required, polygon) plus optional existing urban, existing green,
unbuildable, urban centres (points or polygon areas), PT stops, rail/tram stations, and a street
network (line layer; when given, walking distances are measured along it). All layers may be in
any CRS; they are reprojected to the chosen run CRS.

The **Run button stays disabled** until four things are set: an extents layer, an output `.tif`,
a projected CRS, and valid densities and shares. The red status line lists exactly which are
missing.

## Outputs and how to read them

**Ensemble mode** loads, top to bottom: existing development; the raw chosen run (before
post-processing); the moderately and tightly clustered scenario options (each coloured by
density tier: built as a yellow-to-brown ramp, mixed-use centres as a reds ramp, existing fabric
muted); and the built/green likelihood bands. A `_report.txt` records the parameters and
per-option statistics.

Every population figure counts **new residents only**; existing fabric is assumed served by its
own centres. The per-person readouts follow: m² of mixed-use centre per person is new centre
land over new residents, and m² of green per person is new green over new residents. Coverage
percentages include every home, existing and new.

**Single-run mode** writes one band per growth step. QGIS loads it as a temporal animation: open
*View → Panels → Temporal Controller*, press the play button, and the town grows step by step.

## Troubleshooting

- **"Install the engine?" then nothing works**: restart QGIS after the engine installs; the
  check runs again on the next launch.
- **Run is greyed out**: read the red status line; it names the missing pieces (extents layer,
  output `.tif`, projected CRS, densities/shares).
- **"Select a local projected CRS"**: geographic (degrees) CRSs are rejected. Accept the
  suggested UTM zone, or pick the national grid for the area.
- **"The extents are too small"**: the area must exceed twice the walking distance in both
  directions; enlarge the extents polygon or shorten the walks.
- **OSM fetch fails**: the Overpass servers are shared and sometimes busy; retry after a minute,
  or draw a smaller area.
- **Where are the logs?** *View → Panels → Log Messages*, under the **Isobenefit** tab: grid
  size, per-stage progress, per-option metrics and any warnings land there.
- **Where did my settings go?** Next to the output raster, as `<name>_params.json`; load it back
  with the dialog's *Load parameters* button.
