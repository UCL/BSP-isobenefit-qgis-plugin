# Manuscript workflow

This directory holds the manuscript for the Humanities and Social Sciences
Communications collection "Isobenefit Urbanism". Every number in the text is
computed by a script in `scripts/` from data committed in this repository, so
the whole package rebuilds offline from source.

## Contents

- `main.tex` is the manuscript source.
- `refs.bib` holds the references.
- `figures/` holds the PNG panels the manuscript includes; each is written by
  one of the scripts below and is safe to regenerate in place.
- `outline.md` records the section plan, the word budgets, the display-item
  budget, and the open items before submission.
- `prose-guide.md` is the house style for the prose; run its revision pass on
  every section before calling it done.

## Building the PDF

Run `latexmk -pdf main.tex` from this directory. The build writes `main.pdf`
alongside the source. The `main.aux`, `main.bbl`, and related files are
latexmk working files; `latexmk -C` removes them if a clean rebuild is needed.

## Where the numbers come from

The manuscript cites two bodies of computed results, and each has one script
that regenerates it.

### Case-study metrics

The case-study tables (Cambourne, Crews Hill, Medellin Pajarito) and the
cross-scenario centre-walk sweep are read from
`temp/paper_metrics/metrics.json`, which `scripts/paper_metrics.py`
regenerates from the committed scenario inputs at full scenario resolution
(fifty-run ensembles, seed 42, about five minutes of compute). The same
script writes the full-resolution figure panels into `figures/`. The
website gallery (`scripts/render_scenario_gallery.py`) uses coarsened
preview grids and single runs; the paper no longer cites it.

### Freiburg comparison

The Freiburg regrow comparison (remove Rieselfeld and Vauban, regrow to their
combined population, compare against what was built) is produced by
`scripts/validate_freiburg.py`. Its metrics land in `temp/freiburg_validation*/`
as `metrics.json` plus PNG panels, and the paper panels are copied into
`figures/`. The variants map to output directories as follows.

- Default run (pre-plan substrate, present-day green removed) writes
  `temp/freiburg_validation/`.
- `--keep-green` keeps the present-day green layer and writes
  `temp/freiburg_validation_keepgreen/`.
- `--min-green-span 200|400|600` sweeps the green-span rule and writes
  `temp/freiburg_validation_span200/` and so on; the sweep behind the
  green-grain finding in the text uses 200, 400, and 600 m.

Two audit scripts back the comparison and write their findings into the same
directories. `scripts/check_freiburg_landclass.py` audits the OSM land
classification against the district boundaries, and
`scripts/check_freiburg_builtgap.py` audits gaps in the landuse-derived built
layer. These audits found the missing hospital campus and cemetery that the
`_reality.geojson` corrections now fix, so rerun them after any change to the
Freiburg inputs.

## Cached OSM inputs

`scenarios/freiburg_rieselfeld/` holds the standard committed scenario layers
plus three files that `validate_freiburg.py` fetches once and then caches.

- `_districts.geojson` holds the Rieselfeld and Vauban boundaries from
  Nominatim.
- `_protection.geojson` holds the protected-area geometry.
- `_reality.geojson` holds the land-classification corrections (hospital,
  university, railway, and construction land reclassified as built, cemeteries
  as unbuildable) that apply identically to both substrates.

Once these files exist every rerun is offline; deleting one triggers a single
refetch on the next run.

## Before submission

`outline.md` keeps the authoritative list under "Open items before
submission". As of 2026-08-18 it covers: fact-checking the Rieselfeld reserve
compensation claim, resolving the gallery anomalies (a zero population in one
Crews Hill preset, fewest identical to baseline in four scenarios) before
citing those numbers, deciding whether the Cambourne cleanup result stands or
the min-settlement threshold gets revisited, confirming the authorship list,
completing the reference DOIs, settling APC funding, and the editor's reply on
the deadline extension.
