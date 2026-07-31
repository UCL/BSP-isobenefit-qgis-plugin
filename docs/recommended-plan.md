# The recommended plan: how it is produced

How the plugin turns a stochastic growth simulation into the single prescriptive
layout it labels the recommended plan. For the relationship between the growth
rules and D'Acci's published model, see the theory page on the project website.

## Two products, two questions

The tool produces two different outputs.

| Output | What it is | What it answers |
|---|---|---|
| Likelihood layers | The fraction of N simulated runs in which each cell ended built or green | Where development is robust across futures, and where it is contingent |
| Recommended plan | One categorical layout: green network, built fabric, mixed-use centres | One coherent, walkable arrangement, presented as one scenario among many valid ones |

A probability surface is not a plan. Averaging many runs blurs each run's
coherent green network and centre spacing into a pattern that no single future
would produce, so the plan is never read off the averaged surfaces.

## The pipeline

1. **Ensemble.** The cellular automaton runs N times from the same inputs,
   each with its own seed. Each run is a complete, valid settlement pattern:
   the growth rules keep every new home within a walk of a centre and of
   green, and preserve green corridors, as it grows.
2. **Post-process every run.** Each run's final state is tidied into a
   candidate plan:
   - Failed satellites are pruned. An entirely new settlement smaller than
     the minimum settlement size is a stranded speck, not viable development,
     and reverts to nature. Existing fabric is frozen and never pruned.
   - Sub-threshold infill is absorbed. A new cluster grown against existing
     fabric that is smaller than the same minimum is not called new
     development: the odd free cell inside a town is usually an unmapped
     road, park or awkward lot rather than developable land. Those cells
     join the existing fabric on the map and drop out of the population,
     density and centre accounting. The raw plan keeps them for comparison.
   - The green network is kept exactly as the run grew it. The growth rules
     already enforce the minimum green span, so the plan does not re-carve
     parks.
   - Centres start from the run's own grown centres and are handled per
     option (below). Every option keeps the walk constraint the growth rules
     enforced: no edit may leave a new home beyond the centre walk of a
     centre. Each centre grows to an area sized by the residents it serves.
     Existing centres and rail or tram station anchors are fixed and never
     removed, and every settlement with new development keeps at least one
     attached centre.
   - The centre options: **as grown** keeps every centre exactly where the
     run grew it; **optimised placement** re-positions the run's centres
     central to the homes they serve and adds one wherever new development
     lacks provision within the walk (existing centres serve the existing
     town, not new growth); **fewest centres** additionally removes centres
     one at a time for as long as every home keeps one within the walk, the
     smallest number the constraint permits. A repositioning that would
     strand a home is discarded.
3. **Select the best run.** Every candidate is scored on the same yardstick
   and the plan with the lowest mean walk to amenities wins. The score is
   threshold coverage: a home within the chosen walking distance of a centre,
   and of a qualifying park, counts as served. Centre and green walks are
   scored separately against their own distances, with the same bounded grid
   walk the growth rules use, so growth and scoring always agree. Distances
   detour around unbuildable land. Street-network distances were tried and
   removed: a new settlement's streets do not exist yet, so a network metric
   measures new and existing fabric on different terms.
4. **Arrange density.** Each new home was built at one of three density
   tiers, drawn at the configured shares during the run. Post-processing
   re-arranges the drawn values spatially so the highest tiers sit nearest
   the final centres. The tier mix, and therefore the population, is fixed by
   the run itself.

The chosen run's raw pre-processing state is written alongside the options, so
every edit the post-processing made stays visible, and the run report records
the coverage numbers for all of them. The raw layer is coloured by the density
tiers the run actually drew, exactly where it drew them (the winning ensemble
member is deterministically re-run to recover its drawn density grid); the
processed options show the arranged tiers instead.

## Why the best single run, not a consensus

Averaging many runs into a consensus destroys the run-level coherence that the
growth rules guarantee: contiguous green corridors and sensibly spaced centres
blur into a pattern no single future would produce. Scored on identical terms,
a repaired consensus never matched a good single run, so the pipeline
post-processes every run and keeps the best one.

## Known limitations

- Benefit is a threshold, not a gradient: 80 m and 780 m to a park score the
  same. A distance-decay score would be more faithful to the isobenefit idea
  of graded benefit.
- Any qualifying park serves its whole catchment regardless of quality.
- The pipeline has been exercised most heavily on Cambourne; the scenario
  library adds six further cases, but the strongest claims should still be
  read against that base.
