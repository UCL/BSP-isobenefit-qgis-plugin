# Backlog

Open design questions and deferred work, roughly ordered. Items move out of here
into releases once decided.

## 1. Load-parameters feedback should state what changed

Loading a params file works, but when the loaded values match the dialog (common,
since the defaults and the scenario presets are aligned) nothing visibly moves and
the load looks like a no-op. The feedback line should enumerate the changed fields
("3 fields updated: max iterations 100 to 400, ...") or state that every field
already matched. Old sidecars carrying the retired min_settlement_ha key are
skipped for that field without any notice; the feedback should say so.

## 2. Standard colour schema for downloaded data

The layers fetched by the plugin (green spaces, water, streets, transit stops,
existing development) currently take whatever styling QGIS assigns. Decide whether
to apply a fixed colour schema on load, so that, for example, green spaces are
always green and water is always blue. Should align with the unified colour
language already used on the website and in the scenario galleries.

## 3. Pre-burn hopeless pockets into existing fabric

An open pocket enclosed by existing development that is smaller than the minimum
settlement size can never host a viable new cluster, whatever the run does. Such
pockets could be burned into existing fabric before the simulation (not marked
unbuildable, which would wrongly block walk routing), so the CA does not spend
its population budget on cells the absorption step will remove anyway. Post-run
absorption stays as the general rule; this only handles the provably hopeless
pockets, and the practical effect is a few cells per town, which is why it is
parked.

## 4. Better colours for public transport stops on the website

The public transport stops on the website's maps need a better colour treatment.
Pick a colour that reads clearly against the tier ramps and the muted existing
fabric, and apply it consistently across the pages.
