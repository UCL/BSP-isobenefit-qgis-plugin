# Backlog

Open design questions and deferred work, roughly ordered. Items move out of here
into releases once decided.

## 1. Centre provision for sub-threshold additions to existing fabric

Post-processing adds a mixed-use centre wherever new development lacks provision of
its own, and the addition threshold is small (a centre must reach a handful of
otherwise-unserved cells). A new patch grown against existing fabric can therefore
earn its own commercial area even when its population is below the minimum-settlement
threshold (default 1,000 people), the same threshold below which a *detached* patch
would be pruned as non-viable.

Question under debate: is that provision worth making? A sub-threshold addition could
not stand as a settlement, and the adjacent existing centre demonstrably serves it
(growth required a centre within the walk). The candidate rule: new development only
earns its own centre once its population reaches the minimum-settlement threshold,
unifying "viable as a settlement" and "warrants its own centre" into one number.

Before deciding: count, across the seven scenario galleries, how many added centres
serve sub-threshold catchments, and what served-coverage and centre-per-person
figures look like with and without the unified threshold.

## 2. Tabular summary report of the main outputs

User request (Dnipro testing, 2026-07-30). The ensemble run already writes
`<name>_report.txt`, but the statistics arrive as packed prose lines. The main
outputs should read as tables, in the style of the website's parameter tables:

- population accommodated vs the target, and the starting population;
- achieved densities: per-tier breakdown (people/km² and share of new homes in
  high/medium/low), plus the realised mean over new development;
- served coverage, centre and green walk averages, centre count, m² of centre
  and walkable green per person, with one row per plan option so the raw run
  and the two clustering options compare side by side.

Decisions to take: whether the table lives in the existing `report.txt` (fixed-
width), in a Markdown/HTML sidecar, or both; and whether single-run mode (which
currently writes no report at all) should produce the same summary. The log
panel already carries most figures; the report is the durable copy.

## 3. Load-parameters feedback should state what changed

Loading a params file works, but when the loaded values match the dialog (common,
since the defaults and the scenario presets are aligned) nothing visibly moves and
the load looks like a no-op. The feedback line should enumerate the changed fields
("3 fields updated: max iterations 100 to 400, ...") or state that every field
already matched. Old sidecars carrying the retired min_settlement_ha key are
skipped for that field without any notice; the feedback should say so.
