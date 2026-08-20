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

## Before submission

`outline.md` keeps the authoritative list under "Open items before
submission". As of 2026-08-20 it covers: pushing the v0.13.0 release tag once
the paper state is signed off, deciding the Cambourne target against the
window's measured capacity, the gallery preview-resolution question for the
availability rule, confirming the authorship list, completing the reference
DOIs, settling APC funding, and the editor's reply on the deadline
extension.
