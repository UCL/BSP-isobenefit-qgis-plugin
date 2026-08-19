# Prose guide (local companion)

The manuscript follows the portable prose guide at
`~/.claude/prose_guide.md`, which governs construction: the target register,
the gate, the fault catalogue (F1 to F23), and the revision pass. This file
holds only what is paper-specific, as that guide's section 12 prescribes:
the terminology table and the recorded exceptions.

Journal limits (Humanities and Social Sciences Communications, Article):
about 8,000 words excluding abstract, tables, figure legends, and
references; abstract up to about 300 words with no subheadings, references,
or first person; at most 12 display items; parenthetical author-year
citations; data availability, code availability, ethics, author
contributions, and competing interests statements required.

## Terminology

One term per concept, repeated without variation. Nearby quantities in this
paper have distinct definitions, and synonym rotation costs precision.

| Concept | Use | Do not use |
|---|---|---|
| A local centre of shops and services | centre | centrality (except when quoting D'Acci, defined once), hub, node |
| The two walking limits | the centre walk, the green walk | radius, buffer, threshold, catchment distance, T* (published model only) |
| The distance metric | the grid walk (bounded, blocked by unbuildable land) | network distance (only for the removed alternative), crow-fly |
| A home within both walks | served | covered, satisfied, within reach |
| The headline metric | served coverage | walkability score, accessibility |
| The stochastic model | the automaton, the growth rules | the CA (after first definition), the algorithm, the simulator (published model only) |
| One execution | a run | a simulation, an iteration (reserved for steps within a run) |
| The set of runs | the ensemble | the batch, the sweep (reserved for parameter sweeps) |
| The chosen run | the best single run | the winner, the consensus (that is the rejected alternative) |
| The unedited output | the raw grown state, raw | the pre-plan, the untouched run |
| The three centre arrangements | as grown, optimised placement, fewest centres | grown/placed/minimal (code names), variants |
| All four together | the four outputs | the four plans (raw is not a plan) |
| Pre-existing development | existing fabric | legacy fabric, the existing town (fine informally, once) |
| Land that cannot be built | unbuildable land, defined at first Methods use ("the areas treated as unavailable for development"); plain phrasing before the definition (abstract, introduction) | barriers (reserved for carved corridors), exclusions, bare "unbuildable" ahead of the definition |
| Density levels | the three tiers (high, medium, low) | bands, classes |
| Tier proportions | the shares | the mix (fine once, defined), weights |
| Cleanup of stranded growth | pruning and absorption | filtering, sanitising |
| The blended ensemble rasters | the likelihood layers (their per-cell value: the build likelihood) | likelihood surfaces, probability maps |
| The Freiburg availability variants | the pre-plan reading, the present-day reading | scenarios, versions |
| Where new people go | the target counts new residents only | population cap (published model only) |
| The tool's stage of use | the early stages of planning (or of the planning process) | plan-making, the earliest stage, the initial stage |
| The worked cases' status | hypothetical demonstration (a demonstration case, a demonstration setting) | validation, a test of the pipeline, showcase, pilot |
| The inviolable centre-walk rule | the walk constraint, described as enforced | hard constraint, hard walk constraint (code language) |

## Recorded exceptions

When a rule in the portable guide would obscure a claim, the claim wins
and the exception is recorded here with its date and reason.

- 2026-08-18, degree notation: slopes are written as "20 degrees" in prose
  and with the degree symbol in tables and captions. Display items keep
  the compact symbol; prose keeps the word. This is a deliberate
  convention, not synonym rotation.
