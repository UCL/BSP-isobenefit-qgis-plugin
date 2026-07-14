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
| Build only on nature cells adjacent to built land (periphery growth) | Same: a candidate must be empty and touch built land | Unchanged |
| A centrality must be within T* of the new cell, measured as straight-line distance | A centre must be within the centre walk, measured as a bounded walk over the grid that cannot cross unbuildable land [Centre walk] | Modified: walkable distance |
| Build with probability 0.5 once the checks pass | Same mechanism; the default is 0.25 [Build probability] | Reparameterised |
| Nature remains one connected region, and every nature strip stays at least T* wide | Local rules instead: no green corridor between developments may be pinched below the minimum span (a strip bounded by water or a carved road corridor is exempt, so land beside a barrier can be built right up to it), and a build that splits green must leave each fragment a minimum contiguous area [Min green span] | Modified: local rules |
| Every built cell keeps nature within T* | Same, as a bounded walk; the check also applies to the newly built cell itself [Green walk] | Unchanged in substance |
| New centralities seed near built land that has outgrown its centres, with probability 0.005 | Same trigger; the probability is fixed at 0.01, at most one new centre seeds per iteration, and seeding stops at 80% of the population target | Modified: added guards |
| Isolated centralities seed on open land with probability 0.1 scaled by grid area | A per-cell probability without area scaling, exposed as Off, Moderate or Aggressive [Dispersed development] | Reparameterised |
| One walkable radius T* = 5 cells, cell side 1000/T* m, so T* ≈ 1 km | Explicit metres throughout: cell size [Grid size] and two walking distances, defaulting to 800 m for centres and 400 m for green | Reparameterised |
| Each built block draws a density from three tiers at fixed probabilities (0.7, 0.3, 0) | Same draw; tiers and shares are settings, and post-processing then arranges the drawn values so the highest sit nearest the final centres [Development density] | Unchanged draw, extended placement |
| Run stops at a population cap (500,000 default) | Same stop; the target counts new residents only, since existing fabric is treated as served by its own centres [Target population] | Modified: new-only accounting |
| Cells are scanned in a fixed raster order against a frozen copy of the grid | Cells are visited in a shuffled order each iteration, which removes the scan-direction bias | Modified |
| One run, one output | Ensembles: many runs blended into likelihood layers, with the best single run selected as the scenario | Extension |
| The grid starts as uniform nature | Real inputs: existing built fabric (frozen), protected green, unbuildable land, centre areas, streets, stops and stations from OpenStreetMap | Extension |
| Centralities stay where they seeded | Post-processing re-positions centres central to the development they serve, adds one wherever new development lacks a centre of its own (existing centres serve the existing town, not new growth), culls redundant ones, and sizes each by the population in its catchment | Extension |
| Distances are grid geometry | When a street layer is supplied, post-processing measures walking distances along the network; the growth rules themselves do not use streets yet | Extension |

Most of the modifications have one motivation: the published model assumes a uniform
abstract plain, while real places arrive with fragmented green, rivers, motorways and
existing towns.

## Where the automaton ends and post-processing begins

The growth rules above produce the raw grown state, which the plugin always saves.
Everything else is post-processing on that state: pruning stranded settlements below
the minimum size, re-positioning and sizing centres, arranging the drawn densities,
scoring runs, and selecting the best. The walkability guarantees are enforced by the
growth rules during the run; post-processing improves the presentation and the centre
arrangement without adding or removing population. The
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
