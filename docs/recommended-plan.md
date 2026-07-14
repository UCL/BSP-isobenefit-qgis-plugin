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
   - The green network is kept exactly as the run grew it. The growth rules
     already enforce the minimum green span, so the plan does not re-carve
     parks.
   - Centres start from the run's own grown centres. When centre optimisation
     is on (the default), each is re-positioned onto new land, central to the
     new homes it serves; centres are added where new development is
     under-served, redundant or tiny ones are culled, and each is grown to an
     area sized by the residents it serves. Existing centres and rail or tram
     station anchors are fixed and never culled, and every settlement keeps
     at least one attached centre.
3. **Select the best run.** Every candidate is scored on the same yardstick
   and the plan with the lowest mean walk to amenities wins. The score is
   threshold coverage: a home within the chosen walking distance of a centre,
   and of a qualifying park, counts as served. Centre and green walks are
   scored separately against their own distances. When a street layer is
   supplied, walking distances are measured along the network rather than
   across open ground.
4. **Arrange density.** Each new home was built at one of three density
   tiers, drawn at the configured shares during the run. Post-processing
   re-arranges the drawn values spatially so the highest tiers sit nearest
   the final centres. The tier mix, and therefore the population, is fixed by
   the run itself.

The chosen run's raw pre-processing state is written alongside the plan, so
every edit the post-processing made stays visible, and the run report records
the coverage numbers for both.

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
- The evaluation walks the street network when one is supplied, but the
  growth rules themselves still grow over open ground.
- The pipeline has been exercised most heavily on Cambourne; the scenario
  library adds six further cases, but the strongest claims should still be
  read against that base.
