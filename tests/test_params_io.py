"""The params sidecar/preset format: save -> load roundtrip, partial presets, bad input."""

import pytest

from isobenefit_qgis import params_io


def test_roundtrip(tmp_path):
    params = {
        "name": "dnipro_A",
        "crs": "EPSG:32636",
        "grid_size_m": 25,
        "max_iterations": 200,
        "target_population": 153250,
        "build_prob": 0.25,
        "dispersal": "moderate",
        "random_seed": 42,
        "centre_walk_m": 400,
        "green_walk_m": 400,
        "optimise_centres": True,
        "centre_m2_per_person": 20,
        "min_settlement_ha": 2,
        "min_green_span_m": 400,
        "densities_km2": {"high": 40000, "medium": 25000, "low": 12000},
        "shares": {"high": 0.2, "medium": 0.6, "low": 0.2},
        "ensemble": True,
        "ensemble_runs": 50,
    }
    path = params_io.save_params(tmp_path / "run_params.json", params)
    loaded = params_io.load_params(path)
    assert loaded["crs"] == "EPSG:32636"
    assert loaded["grid_size_m"] == 25.0
    assert loaded["max_iterations"] == 200
    assert loaded["densities_km2"] == {"high": 40000.0, "medium": 25000.0, "low": 12000.0}
    assert loaded["shares"]["medium"] == 0.6
    assert loaded["ensemble"] is True and loaded["ensemble_runs"] == 50
    assert loaded["name"] == "dnipro_A"


def test_partial_preset_and_unknown_keys(tmp_path):
    # a scenario preset may pin only a few dials; unknown keys are ignored, malformed ones skipped
    p = tmp_path / "preset.json"
    p.write_text(
        '{"schema": "isobenefit-params/1", "densities_km2": {"high": 40000, "low": "x"},'
        ' "centre_walk_m": 800, "mystery": 1, "build_prob": "not-a-number"}',
        encoding="utf-8",
    )
    loaded = params_io.load_params(p)
    assert loaded == {"centre_walk_m": 800.0, "densities_km2": {"high": 40000.0}}


def test_rejects_bad_files(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError):
        params_io.load_params(bad)
    other = tmp_path / "other.json"
    other.write_text('{"schema": "something-else/9"}', encoding="utf-8")
    with pytest.raises(ValueError):
        params_io.load_params(other)
    with pytest.raises(ValueError):
        params_io.load_params(tmp_path / "missing.json")


def test_sidecar_path(tmp_path):
    assert params_io.sidecar_path(tmp_path, "run7").name == "run7_params.json"
