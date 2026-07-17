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

## 2. Load-parameters feedback should state what changed

Loading a params file works, but when the loaded values match the dialog (common,
since the defaults and the scenario presets are aligned) nothing visibly moves and
the load looks like a no-op. The feedback line should enumerate the changed fields
("3 fields updated: max iterations 100 to 400, ...") or state that every field
already matched. Old sidecars carrying the retired min_settlement_ha key are
silently skipped for that field; the feedback should say so.
