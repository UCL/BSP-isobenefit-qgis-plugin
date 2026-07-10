"""Save and load run parameters as JSON, kept free of any QGIS/Qt import.

One schema serves two purposes: after every run the plugin writes a ``<output>_params.json``
sidecar next to the output raster (a cache of exactly what was run), and each scenario folder in
``scenarios/`` ships a ``params.json`` preset in the same format. Either file can be loaded back
into the run dialog to repopulate it.

Unknown keys are ignored on load and missing keys are simply not applied, so presets may be
partial (a scenario can pin only densities and walks) and old sidecars stay loadable as the
schema grows.
"""

from __future__ import annotations

import json
from pathlib import Path

SCHEMA = "isobenefit-params/1"

# keys that repopulate the dialog, with the type each value is coerced to on load
_FIELD_TYPES: dict[str, type] = {
    "crs": str,
    "grid_size_m": float,
    "max_iterations": int,
    "target_population": float,
    "build_prob": float,
    "dispersal": str,  # off | moderate | aggressive
    "random_seed": int,
    "centre_walk_m": float,
    "green_walk_m": float,
    "optimise_centres": bool,
    "centre_m2_per_person": float,
    "min_settlement_ha": float,
    "min_green_span_m": float,
    "ensemble": bool,
    "ensemble_runs": int,
}
_TIER_KEYS = ("high", "medium", "low")


def save_params(path: str | Path, params: dict) -> Path:
    """Write ``params`` (plus the schema marker) as pretty JSON; returns the path written."""
    path = Path(path)
    doc = {"schema": SCHEMA, **params}
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_params(path: str | Path) -> dict:
    """Read a params JSON and return only the recognised, type-coerced fields.

    Raises ``ValueError`` with a readable message if the file is not JSON, not an object, or has
    an unknown schema marker (a missing marker is tolerated for hand-written presets).
    """
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read parameters: {exc}") from exc
    if not isinstance(doc, dict):
        raise ValueError("Parameters file must hold a JSON object.")
    schema = doc.get("schema")
    if schema is not None and schema != SCHEMA:
        raise ValueError(f"Unsupported parameters schema: {schema!r} (expected {SCHEMA!r}).")
    out: dict = {}
    for key, typ in _FIELD_TYPES.items():
        if key in doc and doc[key] is not None:
            try:
                out[key] = typ(doc[key])
            except (TypeError, ValueError):
                continue  # a malformed single value never blocks the rest of the preset
    for group in ("densities_km2", "shares"):
        vals = doc.get(group)
        if isinstance(vals, dict):
            tiers = {}
            for tier in _TIER_KEYS:
                if tier in vals and vals[tier] is not None:
                    try:
                        tiers[tier] = float(vals[tier])
                    except (TypeError, ValueError):
                        continue
            if tiers:
                out[group] = tiers
    for meta in ("name", "notes"):
        if isinstance(doc.get(meta), str):
            out[meta] = doc[meta]
    return out


def sidecar_path(out_dir: str | Path, out_file_name: str) -> Path:
    """The conventional sidecar location for a run: ``<out_dir>/<out_file_name>_params.json``."""
    return Path(out_dir) / f"{out_file_name}_params.json"
