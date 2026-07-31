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

## 3. Better colours for public transport stops on the website

The public transport stops on the website's maps need a better colour treatment.
Pick a colour that reads clearly against the tier ramps and the muted existing
fabric, and apply it consistently across the pages.
