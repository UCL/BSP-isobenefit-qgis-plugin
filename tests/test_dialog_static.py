"""Static checks on the dialog, which no test can construct without QGIS.

The plugin builds its dialog in ``Isobenefit.__init__``, so QGIS calls this code
at plugin load: an attribute used before it is assigned takes the whole plugin
down, not just the dialog. That happened once (a slope-limit widget added to the
layout forty lines above its own constructor) and neither ruff, the test suite
nor the verify script caught it, because none of them import a QGIS module.
"""

from __future__ import annotations

import ast
import pathlib

DIALOG = pathlib.Path(__file__).resolve().parents[1] / "isobenefit_qgis" / "isobenefit_dialog.py"


def _method(tree: ast.Module, class_name: str, method_name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    raise AssertionError(f"{class_name}.{method_name} not found in {DIALOG.name}")


def test_setupui_assigns_every_attribute_before_using_it() -> None:
    tree = ast.parse(DIALOG.read_text(encoding="utf-8"), str(DIALOG))
    setup = _method(tree, "IsobenefitDialog", "setupUi")

    # first assignment line for every attribute setupUi creates
    first_assign: dict[str, int] = {}
    for node in ast.walk(setup):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and isinstance(node.ctx, ast.Store)
        ):
            first_assign.setdefault(node.attr, node.lineno)

    # a widget this method builds must not be read before it is built. Attributes it
    # never assigns are methods or inherited Qt API, and are not this test's business.
    problems = [
        f"self.{node.attr} read at line {node.lineno}, assigned at line {first_assign[node.attr]}"
        for node in ast.walk(setup)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and isinstance(node.ctx, ast.Load)
        and node.attr in first_assign
        and node.lineno < first_assign[node.attr]
    ]
    assert not problems, "setupUi would raise at plugin load:\n  " + "\n  ".join(sorted(problems))


def test_every_committed_scenario_layer_declares_its_crs() -> None:
    """A GeoJSON without a ``crs`` member is longitude and latitude by definition.

    Every scenario layer holds projected metres, so a missing member puts the layer
    thousands of kilometres from where it belongs the moment someone drags it into
    QGIS, and the plugin then blames the CRS they chose.
    """
    import json

    root = pathlib.Path(__file__).resolve().parents[1] / "scenarios"
    missing = []
    for path in sorted(root.glob("*/*.geojson")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("type") != "FeatureCollection":
            continue
        name = ((doc.get("crs") or {}).get("properties") or {}).get("name", "")
        if not name.startswith("urn:ogc:def:crs:EPSG::"):
            missing.append(str(path.relative_to(root)))
    assert not missing, "scenario layers with no CRS member:\n  " + "\n  ".join(missing)
