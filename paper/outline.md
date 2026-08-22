# Paper outline

Target: Article in Humanities and Social Sciences Communications, collection
"Isobenefit Urbanism" (guest editors L. S. D'Acci, T. Gabrieli). Limits: about
8,000 words excluding abstract, tables, figure legends and references; abstract
up to about 300 words, no subheadings or references, avoid first person; at most
12 display items; parenthetical (Harvard) citations; data availability,
code availability, competing interests and author contributions statements
required. Collection deadline 15 August 2026; an extension has been requested.

## Working title

Isobenefit Urbanism on real terrain: an open-source QGIS plugin for
simulating and planning walkable growth

Alternative: From abstract plain to OpenStreetMap: implementing Isobenefit
Urbanism as a reproducible GIS planning tool

Authors: Gareth Simons and Tommaso Gabrieli. Wider project team goes in the
acknowledgements; D'Acci and Gabrieli guest-editor roles declared under
competing interests.

## Contribution claims

1. A reworking of the published Isobenefit rules for real geography: frozen
   built fabric, protected green, water and infrastructure barriers, slope
   limits, and walkable (grid-walk) distances in place of straight lines.
2. A plan-derivation pipeline on top of the cellular automaton: ensembles with
   likelihood surfaces, selection of the best single run, and four outputs
   (raw, grown, placed, fewest) in which the walk constraint is hard.
3. An open, reproducible implementation: Rust engine on PyPI, thin QGIS
   plugin, seven worked scenarios on committed OSM data, deterministic under
   parallelism, verified end to end in CI.

## Section plan (word budgets sum to ~7,800)

### Abstract (~280 words)

Problem, approach, the four outputs, headline case-study numbers, validation
result, availability.

### 1. Introduction (~900 words)

- Isobenefit Urbanism in one paragraph: equal benefit of access to centres and
  green, growth guided by a few rules, form left to emerge. Source: prose in
  `website/src/pages/index.astro` (Concept and rationale).
- The gap: the published model grows on a uniform abstract plain; real places
  arrive with fragmented green, rivers, motorways and existing towns.
- What a planner needs beyond a simulator: a defensible plan drawing, not a
  raster of probabilities. States the two-products framing (likelihood layers
  for pattern questions, a recommended plan for the drawing) from
  `docs/recommended-plan.md`.
- Contributions list and paper structure.

### 2. Background (~800 words)

- D'Acci (2019, J. Environmental Management 246:128-140); D'Acci and Voto
  (2023, SoftwareX 22:101408) and the isobenefit-cities code, with its
  published defaults (build prob 0.5, T* = 5 cells ~ 1 km, cap 500,000,
  density tiers at 0.7/0.3/0).
- Adjacent literature: 15-minute city, walkability thresholds (WHO and
  Natural England 400 m everyday green; 800 m as a ten-minute walk), urban CA
  models, generative planning tools.
- Positioning: this work implements and extends the published model rather
  than proposing a rival one; every departure is tabulated.

### 3. Methods (~2,600 words)

3.1 Real inputs (~500). Nine OSM layers per scenario (built, green, centres,
unbuildable, streets, stops, stations, railways, industrial), Copernicus
GLO-30 slope bands, rasterisation to a working grid, corridor carving for
major roads, railways and rivers. Source: `plugin.md`, `scenarios/README.md`.

3.2 Rule adaptations (~700). Condense the 14-row table from
`website/src/pages/theory.md` into Display item 1, with each row tagged
unchanged, reparameterised, modified or extension. Text walks the
substantive modifications: walkable distance in place of straight lines,
local green-span rules in place of global connectivity, seeding guards,
shuffled visit order, new-residents-only population accounting.

3.3 Ensembles and selection (~350). Fifty runs, likelihood surfaces, why the
best single run and not a consensus (averaging blurs run-level coherence;
scored on identical terms a repaired consensus never matched a good single
run). The raw output is recovered by deterministic re-run of the winning
member at its own seed. Source: `docs/recommended-plan.md`, commit history.

3.4 Post-processing and the four outputs (~650). The boundary statement:
growth rules produce the raw grown state; everything after is post-processing
that improves presentation and centre arrangement without adding or removing
population. The three processed options share identical fabric and density;
only the centres differ. Walk constraint is hard in every option: moves that
strand a home are discarded, gap-fill adds centres ungated, removals proceed
only while coverage holds. Min-settlement pruning and per-cluster infill
absorption. Source: `isobenefit_qgis/sim_runner.py`, `grid.py`,
`docs/recommended-plan.md`.

3.5 Distance metric and transit (~400). One bounded grid-walk metric for both
growth and scoring. Street-network routing was implemented and then removed:
a new settlement's streets do not exist yet, so a network metric measures new
and existing fabric on different terms. Stations join the centre seeds and are
pinned in post-processing; stops and stations feed reported transit metrics
that do not influence selection.

3.6 Software and reproducibility (~300). Rust core (PyO3, abi3) on PyPI,
thin QGIS plugin (the plugin repository does not permit shipping binaries),
determinism independent of thread count via per-work-item seeding and
order-independent reductions, Rust/Python cross-check tests, single verify
script mirrored by CI, committed scenario data so every artefact rebuilds
offline from source at fixed seeds.

### 4. Case studies (~1,600 words)

Three cases, chosen for contrast in constraint type; the remaining library
scenarios (Dnipro, Celina, Kigali) are named once as available in the
repository and otherwise dropped from the paper.

- Cambourne (new settlement, the reference demo): the four-output report
  table (population, coverage, walks, centre counts) and the reading of it
  (placement cut the average centre walk 482 to 436 m; fewest gave most of it
  back for two fewer centres).
- Crews Hill, London (green-belt release, ~5,500 homes around a rail
  station): policy-live UK case; walkable extension versus car-led sprawl.
- Medellin Pajarito (hillside expansion, slopes over 20 degrees unbuildable
  across ~30% of the window, Metrocable anchors): topography as the binding
  constraint.
- Display item: three-scenario settings table (grid, target, tiers, shares,
  dispersal, slope) from `scenarios/`.
- Cross-scenario centre-walk sweep (400 / 800 / 1600 m) as the main
  comparative table, with the accounting caveat stated plainly: coverage
  counts every home including existing fabric.
- Numbers caveat: gallery metrics are computed at preview resolution with
  seed 42; regenerate at full scenario resolution before submission.

### 6. Discussion and limitations (~900 words)

- What the rule changes buy on real terrain, and what they cost in
  comparability with the published model.
- REMOVED 2026-08-22 at Gareth's direction: the Limitations subsection
  (threshold-not-gradient benefit, park quality, single-run spread, Cambourne
  emphasis) is out of the paper.
- The cleanup gap (Cambourne window: raw 82% of target versus 51% after
  cleanup) and what it says about min-settlement thresholds.
- Scope statement inherited from the README: research software for discussion
  with domain experts; the scenarios are not plans to build from.

### 7. Conclusion (~300 words)

### Statements

- Data availability: scenario inputs committed in the repository; OSM under
  ODbL; slope from Copernicus GLO-30.
- Code availability: AGPL-3.0-or-later; engine on PyPI (`isobenefit`), plugin
  on the QGIS repository; archive a tagged release on Zenodo for a DOI.
- Author contributions: project team per site credits (Gabrieli, D'Acci,
  Kwon, Marshall, Marin Maureira, Simons); to be settled with the team.
- Competing interests: D'Acci is a guest editor of the target collection and
  a project member; declare and rely on independent editorial handling.

## Display item budget (max 12)

1. Rule-comparison table (condensed from theory.md).
2. Pipeline diagram: inputs, CA, ensemble, selection, post-processing, four
   outputs.
3. Cambourne four-output map panel (existing, raw, grown, placed, fewest).
4. Cambourne report metrics table.
5. Seven-scenario summary table.
6. Centre-walk sweep chart across scenarios.
7. One or two additional scenario map panels.

Leaves two or three slots spare.

## Open items before submission

1. REMOVED 2026-08-19: the Freiburg comparison is out of the paper and the
   repository. A realised district embodies expert design and constraints the
   model never sees, so scoring the output against it implied a replication
   test the paper does not intend. The scenario, validation scripts, audit
   scripts, cached layers, and gallery entry were deleted (recoverable from
   git history at e9419eb and earlier).
2. RESOLVED 2026-08-18: the paper no longer cites gallery numbers.
   `scripts/paper_metrics.py` computes all case-study metrics at full
   scenario resolution (fifty-run ensembles).
3. RESOLVED 2026-08-19 (v0.13.0, tag not yet pushed): absorption removed,
   availability rule added at grid preparation (local width via a 3x3
   opening plus rook-connected region capacity), hamlet prune, and the
   infill provision exception. All numbers in the paper are regenerated
   under this behaviour. The old 51%-of-target question is superseded: the
   demonstration window now credits 84% raw against 62% in the plan
   options, with pruning of failed satellites as the only cause.
4. Push the v0.13.0 tag once Gareth signs off the paper state; the tag
   triggers CI to publish the core to PyPI and the plugin zip.
5. Decide the Cambourne scenario target: at baseline settings the window's
   developable land holds about 13,200 of the 30,000 target (recomputed
   2026-08-20 with true-area existing-centre seeds), which the paper
   currently reports as a capacity finding. Lowering the target is the
   alternative.
6. Gallery preview resolution: the availability width test is measured in
   cells, so coarse preview grids (Dnipro at 135 m) exclude much more land
   than the full-resolution runs. Decide whether previews should run finer
   or the width should be defined in metres.
7. Confirm authorship list and order with the UCL team.
8. Reference list: gather DOIs (two core sources are in theory.md; add
   15-minute-city, WHO and Natural England green-access sources).
9. APC funding or waiver.
10. Editor reply on the deadline extension.
