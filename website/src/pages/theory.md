---
layout: ../layouts/BaseLayout.astro
title: Theory
description: How the plugin's growth rules relate to D'Acci's published Isobenefit Urbanism model
---

# Theory: the published model and this implementation

The plugin's growth rules descend from D'Acci's published Isobenefit Urbanism model.
This page records where they are unchanged, where they differ, and why. The
[introduction](../) covers the pipeline around the rules.

## The published model

Isobenefit Urbanism (D'Acci 2019) proposes settlements whose benefits are distributed
evenly: wherever one lives, a local centre with shops and services, and open green
land, are both within a walking distance. The *iso* means equal. D'Acci formalised the
idea as a cellular-automaton morphogenesis, published as the Isobenefit-cities
simulator (D'Acci and Voto 2023) with Python code by Michele Voto. The automaton grows
a grid of land cells from seeded centralities under a small set of rules, so that the
walkability guarantee holds at every step no matter how large the settlement becomes.

The published model works on an abstract grid: every cell starts as nature, distances
are measured in cells, and the walkable radius T* is about one kilometre, described as
a 15-minute walk (later summaries give one to two kilometres). Each simulation step
scans the grid; a nature cell adjacent to built land may become built if a centrality
is within T* and the move would not break the nature rules; new centralities seed
stochastically; each built block draws a population density; the run stops at a
population cap.

## The rules

Each row gives a rule of the published simulator, what this implementation does, and a
status: **unchanged** (same rule), **reparameterised** (same rule, different units or
defaults), **modified** (behaviour differs), or **extension** (no counterpart in the
published model). Parameter names in brackets are the plugin's settings.

| Published rule | This implementation | Status |
|---|---|---|
| Build only on nature cells adjacent to built land (periphery growth) | Same, restricted to centred fabric: centre-less farmsteads and hamlets never anchor growth | Modified |
| A centrality must be within T* of the new cell, measured as straight-line distance | A centre must be within the centre walk, measured as a bounded walk over the grid that cannot cross unbuildable land [Centre walk] | Modified: walkable distance |
| Build with probability 0.5 once the checks pass | Same mechanism at a fixed rate: the draw paces growth and varies the ensemble, and no longer decides how many centres appear | Unchanged |
| Nature remains one connected region, and every nature strip stays at least T* wide | Local rules instead: no green corridor between developments may be pinched below the minimum span (a strip bounded by water or a carved road corridor is exempt, so land beside a barrier can be built right up to it), and new development may not shrink a park (a green area of at least the minimum park area, 2 ha default) below that area, whether by splitting it or by building along its edge [Min green span, Min park area] | Modified: local rules |
| Every built cell keeps nature within T* | Same, as a bounded walk whose destination must be a park; the check also applies to the newly built cell itself. The Green walk bounds the check during growth and at scoring, and both use the same park definition [Green walk] | Unchanged in substance |
| New centralities seed near built land that has outgrown its centres, with probability 0.005 | Released rather than drawn: each new home is charged to the centre it is nearest, and a centre earns its successor once that population reaches the service viability threshold, or once it can no longer grow [Service viability] | Modified: earned, not drawn |
| Isolated centralities seed on open land with probability 0.1 scaled by grid area | No probability: a released centre may start away from existing development when the scenario allows detached settlements [Dispersed development] | Modified: earned, not drawn |
| One walkable radius T* = 5 cells, cell side 1000/T* m, so T* ≈ 1 km | Explicit metres throughout: cell size [Grid size] and two walking distances, defaulting to 1,200 m for centres and 400 m for green | Reparameterised |
| Each built block draws a density from three tiers at fixed probabilities (0.7, 0.3, 0) | Same draw; tiers and shares are settings, and post-processing then arranges the drawn values so the highest occupy each settlement's core, graded to a light edge [Development density] | Unchanged draw, extended placement |
| Run stops at a population cap (500,000 default) | Same stop; the target counts new residents only, since existing fabric is treated as served by its own centres [Target population] | Modified: new-only accounting |
| Cells are scanned in a fixed raster order against an unchanged copy of the grid | Cells are visited in a shuffled order each iteration, which removes the scan-direction bias | Modified |
| One run, one output | Ensembles: many runs blended into likelihood layers, with the best single run selected as the scenario | Extension |
| The grid starts as uniform nature | Real inputs from OpenStreetMap: existing built fabric (fixed), protected green, unbuildable land including corridors carved from major roads, railways and rivers, centre areas, stops and stations | Extension |
| All nature is available to build | Developable land must be locally wide, and its region must either hold a viable settlement (the service viability threshold) or lie within the centre walk of an existing centre, where growth is served infill; all other open land is set aside as protected green before the run, so slivers and enclaves read as pocket parks | Extension |
| Growth is spatially uniform in probability | Transit shapes growth in two ways. Hubs (rail/tram stations, or points a planner designates) anchor a pinned mixed-use centre that seeds growth and survives post-processing. Corridors (bus stops, or a proposed route drawn as a line) attract growth: outside the stop catchment of a corridor and the wider hub catchment of a hub, the build and seeding draws are scaled by one minus the corridor preference, so growth concentrates along existing or proposed transit; at the default preference of 0 growth is unchanged [Corridor preference, Stop catchment, Hub catchment] | Extension |
| Centralities stay where they seeded | Post-processing offers centre options that all keep every home within the centre walk: as grown (locations untouched; a centre is added only where pruning removed a cluster's serving centre), optimised placement (re-positioned central to the development served, added wherever new development lacks provision; existing centres serve the existing town and small infill within their walk, while a new district earns its own centre), and fewest centres (redundant ones removed while full coverage holds). Each settlement hosts an attached centre of its own; catchments cross green gaps, so nearby settlements pool their demand, centres below threshold demand are cut, growth without a viable centre reverts to nature, the centre audit verifies the result, and each centre is sized by the population in its catchment | Extension |
| Distances are straight lines over an abstract plain | One metric for growth and scoring: bounded walks over the grid, blocked by unbuildable land. Street-network distances were tried and removed, since a new settlement's streets do not exist yet and a network metric measures new and existing fabric on different terms | Modified: walkable distance |

Most of the modifications have one motivation: the published model assumes a uniform
abstract plain, while in real-world use the model must deal with fragmented green,
rivers, motorways and existing towns.

## Where the automaton ends and post-processing begins

The growth rules above produce the raw grown state, which the plugin always saves.
Everything else is post-processing on that state: pruning stranded settlements below
the minimum size, re-positioning and sizing centres, arranging the drawn densities,
scoring runs, and selecting the best. The walkability guarantees are enforced by the
growth rules during the run. Pruning is the only post-processing step that removes
population; the rest re-arranges what the run grew, and never adds anything. The
[recommended-plan notes](https://github.com/UCL/BSP-isobenefit-qgis-plugin/blob/main/docs/recommended-plan.md)
document that pipeline.

## Terminology

*Centrality* is D'Acci's term for a local centre of shops and services among homes;
the plugin's interface says *centre* and treats the words as synonyms. *Morphogenesis*
refers to D'Acci's growth model specifically. *Isobenefit* describes the goal: equal
benefit wherever one lives. The run report states the achieved coverage.

## Sources

- D'Acci, L. (2019). [A new type of cities for liveable futures. Isobenefit Urbanism
  morphogenesis](https://doi.org/10.1016/j.jenvman.2019.05.129). *Journal of
  Environmental Management*, 246, 128–140.
- D'Acci, L., and Voto, M. (2023). [Morphogenesis of Isobenefit urbanism:
  Isobenefit-cities simulator](https://doi.org/10.1016/j.softx.2023.101408).
  *SoftwareX*, 22, 101408.
- Voto, M. [isobenefit-cities](https://github.com/mitochevole/isobenefit-cities), the
  original Python simulator. The published parameter values cited above (build
  probability 0.5, neighbouring-centrality probability 0.005, isolated-centrality
  probability 0.1 over the grid area, T* of 5 cells, population cap 500,000, density
  tiers drawn at 0.7 / 0.3 / 0) are the defaults of this code.
- D'Acci, L. [Isobenefit Urbanism
  overview](https://lucadacci.wixsite.com/dacci/isobenefit-urbanism-morphogenesis).
