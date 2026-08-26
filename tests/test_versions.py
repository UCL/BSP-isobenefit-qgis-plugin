"""Static checks on the release version, which no runtime test exercises.

Plugin and engine release in strict lockstep, so one version string lives in four
files, and ``bootstrap.MAX_VERSION_EXCLUSIVE`` must sit above it: the pip spec is
``isobenefit>=<version>,<MAX``, which is unsatisfiable the moment the bound is
forgotten on a minor bump. That happened once (plugin 0.17.0 shipped with the
bound still at 0.17.0) and nothing caught it before the tag.
"""

from __future__ import annotations

import ast
import configparser
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]


def _metadata_version() -> str:
    parser = configparser.ConfigParser()
    parser.read(REPO / "isobenefit_qgis" / "metadata.txt")
    return parser.get("general", "version")


def _toml_version(path: pathlib.Path, section: str) -> str:
    in_section = False
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            in_section = stripped == f"[{section}]"
        elif in_section and stripped.startswith("version"):
            return stripped.split("=", 1)[1].strip().strip('"')
    raise AssertionError(f"no version in [{section}] of {path.name}")


def _max_version_exclusive() -> tuple[int, ...]:
    source = (REPO / "isobenefit_qgis" / "bootstrap.py").read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "MAX_VERSION_EXCLUSIVE" for t in node.targets
        ):
            return tuple(ast.literal_eval(node.value))
    raise AssertionError("MAX_VERSION_EXCLUSIVE not found in bootstrap.py")


def test_the_four_version_files_agree() -> None:
    version = _metadata_version()
    assert _toml_version(REPO / "pyproject.toml", "project") == version
    assert _toml_version(REPO / "core" / "Cargo.toml", "package") == version
    lock = REPO / "core" / "Cargo.lock"
    assert f'name = "isobenefit"\nversion = "{version}"' in lock.read_text(encoding="utf-8")


def test_engine_bound_sits_above_the_plugin_version() -> None:
    version = tuple(int(chunk) for chunk in _metadata_version().split("."))
    assert version < _max_version_exclusive(), (
        f"bootstrap.MAX_VERSION_EXCLUSIVE {_max_version_exclusive()} does not admit the "
        f"plugin's own version {version}: the pip spec cannot be satisfied"
    )
