# Backlog

Open design questions and deferred work, roughly ordered. Items move out of here
into releases once decided.

## 1. RESOLVED 0.13.0: unviable land is excluded up front as protected green

Superseded by the availability rule (2026-08-19). Grid preparation now marks
any open land that is not locally wide (a 3x3 morphological opening), or that
sits in a rook-connected wide region too small to hold the minimum settlement,
as protected green before the run: still walkable, still counted as green, but
never built, seeded, or provisioned. This replaced both the parked pre-burn
idea and post-run absorption, which was removed outright (attached infill on
viable land is legitimate development and is now counted). One open aspect:
the width test is measured in cells, so coarse preview grids exclude more land
in metres than fine grids do (see the gallery-resolution note in
`paper/outline.md`).
