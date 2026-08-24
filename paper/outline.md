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
2. REVISED 2026-08-24 at Gareth's direction: enhancements beyond the
   published model, not the pipeline. Separated walking distances (centre
   and green, each with its own rules), transit-oriented development as an
   explicit heuristic (hubs anchor centres, corridor preference), and
   post-processing that optimises centre placement and grades densities.
   The ensemble is demoted to supporting machinery, not a claim. TOD is
   integrated through the text (2026-08-24): the introduction's gap
   paragraph, the contributions, the renamed Methods subsection
   (transit-oriented growth), the corridor demonstration's viability
   reading, the discussion, and the conclusion.
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
  (2023, SoftwareX 23:101408) and the isobenefit-cities code, with its
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

3.2 Rule adaptations (~700). REWRITTEN 2026-08-24 at Gareth's direction:
Display item 1 is a table of DEPARTURES ONLY, eight rows, each giving the
published rule, ours, and WHY, with the status column dropped. Rules we
kept are not listed. The eight: one radius to four walking distances;
seeding probabilities to a population per centre; straight lines to
bounded walks; global green connectivity to local rules; the population
cap counting new residents only; centres and densities arranged in
post-processing; ensembles with one run selected; transit settings.
Gareth rejected the earlier 16-row inventory as padding.

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
  table. As of the 2026-08-24 engine work all four outputs carry the SAME 26
  centre areas (20 existing, 6 earned) and the same population, so the
  options differ only in where the centres sit: 499 m as grown, 328 m
  placed, and fewest centres removes none because every centre was earned.
- Crews Hill, London (green-belt release, ~5,500 homes around a rail
  station): policy-live UK case; walkable extension versus car-led sprawl.
- Medellin Pajarito (hillside expansion, slopes over 20 degrees unbuildable
  across ~30% of the window, Metrocable anchors): topography as the binding
  constraint.
- Display item: three-scenario settings table (grid, target, tiers, shares,
  dispersed development, slope) from `scenarios/`.
- Cross-scenario centre-walk sweep (800 / 1,200 / 1,600 m). Every case
  houses close to its target across the sweep, so the table reports the mean
  centre walk and the text explains it through the number of centres
  founded, not through a population ceiling.

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

Items 1-4 and 8 below are settled and kept only as the record of why.
Everything still open is listed first.

### Open

1. Confirm the authorship list and order with the UCL team.
2. APC funding or a waiver.
3. Editor reply on the deadline extension.
4. `\pending` markers in the manuscript: the corresponding author, the
   Zenodo DOI for code availability, and the competing-interests wording.
5. Tag v0.14.0. Held 2026-08-24 at Gareth's direction until the paper
   settles, so the published version matches the manuscript's numbers.
6. Gallery preview resolution. The website renders previews at about 150
   cells a side while the paper runs full resolution, so the two are not
   comparable for the same preset. The scale-dependence that made this
   worse is gone (centres now follow population, not cell count), but the
   availability width test is still measured in cells.

### Settled, with the reasoning

7. REMOVED 2026-08-19: the Freiburg comparison is out of the paper and the
   repository. A realised district embodies expert design and constraints the
   model never sees, so scoring the output against it implied a replication
   test the paper does not intend (recoverable from git history at e9419eb).
8. RESOLVED 2026-08-18: the paper cites no gallery numbers.
   `scripts/paper_metrics.py` computes every case-study metric at full
   scenario resolution.
9. DONE 2026-08-22: v0.13.0 tagged and published.
10. RESOLVED 2026-08-24: the Cambourne target is 15,000, set by Gareth.
11. RESOLVED 2026-08-24: every reference verified against Crossref and
    DataCite; SoftwareX is volume 23, Barton et al. gained its Routledge
    DOI, WHO 2016 its IRIS URL, Natural England its NE265 number. Batty
    2005 and Calthorpe 1993 have no DOIs to find.

## Model changes of 2026-08-24, and where they left the paper

The day's engine work changed every reported number, so the manuscript was
regenerated throughout. In order: the 80% centre-seeding cap removed; centres
reformulated from a seeding probability to a population per centre; the green
span rule corrected so a settlement may build into a bay it encloses itself;
four defects in the earned-centres design fixed after an engine review (a
build-out test in place of a dry-iteration guess, credit that does not move
with reassigned population, one ledger per centre area rather than per cell,
and a canonical tie-break so layer order cannot change a plan); the plugin
brought into line with the scripts (a slope limit it never applied, barriers
carved by cell centre, walks that cannot cut a diagonal corner); and existing
centres stopped earning centres, so growth and post-processing now agree on
what an existing centre provides.

Where that leaves the case studies: all three windows house close to their
target, so the cases are separated by the setting each answers to rather than
by a population ceiling. Cambourne's four outputs now carry the same 26 centre
areas, and the options differ only in where the centres sit.
