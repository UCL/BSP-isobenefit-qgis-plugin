"""Headless tests for the pure OSM helpers (no QGIS/GDAL needed).

The QGIS/GDAL-coupled fetch (HTTP + OSM-driver parse + GeoPackage build) in
``osm_fetcher`` is exercised in QGIS manually; these cover the query/tag logic in
``osm_queries`` that is most prone to silent mistakes.

Run:
    uv run --no-project \
        --with core/dist/isobenefit-*.whl --with numpy --with shapely --with pytest \
        python -m pytest tests -q
"""

from __future__ import annotations

import pytest

from isobenefit_qgis.osm_queries import (
    DATASET_FIELDS,
    DATASET_ORDER,
    DATASETS,
    STATION_KINDS,
    bbox_to_overpass,
    build_query,
    feature_attributes,
    feature_matches,
    parse_hstore,
    point_is_pt_stop,
    pt_stop_kind,
)


def test_bbox_to_overpass_transposes_to_swne():
    # GIS (xmin, ymin, xmax, ymax) -> Overpass (south, west, north, east).
    assert bbox_to_overpass(-0.5, 51.0, 0.5, 52.0) == (51.0, -0.5, 52.0, 0.5)


def test_build_query_embeds_bbox_format_and_tags():
    q = build_query("built", 51.0, -0.5, 52.0, 0.5)
    assert "(51.0,-0.5,52.0,0.5)" in q  # s,w,n,e order, appended to each selector
    assert "landuse" in q and "residential" in q
    assert "[out:xml]" in q
    assert "(._;>;);" in q  # union with recursed members so nodes precede ways (GDAL-friendly order)
    assert q.strip().endswith("out body;")


def test_build_query_per_dataset_selectors():
    assert "highway" in build_query("streets", 0, 0, 1, 1)
    assert "bus_stop" in build_query("pt", 0, 0, 1, 1)
    assert "retail" in build_query("centres", 0, 0, 1, 1)
    q = build_query("unbuildable", 0, 0, 1, 1)
    assert "natural" in q and "aeroway" in q and "military" in q  # water + airports + military


def test_build_query_unknown_dataset_raises():
    with pytest.raises(KeyError):
        build_query("nope", 0, 0, 1, 1)


def test_parse_hstore_basic_and_comma_value():
    tags = parse_hstore('"railway"=>"station","name"=>"Smith, John Square","operator"=>"TfL"')
    assert tags["railway"] == "station"
    assert tags["name"] == "Smith, John Square"  # comma inside a quoted value preserved
    assert tags["operator"] == "TfL"


def test_parse_hstore_empty():
    assert parse_hstore(None) == {}
    assert parse_hstore("") == {}


def test_point_is_pt_stop():
    assert point_is_pt_stop({"highway": "bus_stop"})
    assert point_is_pt_stop({"railway": "tram_stop"})
    assert point_is_pt_stop({"public_transport": "platform"})
    assert not point_is_pt_stop({"amenity": "cafe"})


def test_feature_matches_polygons():
    assert feature_matches("built", {"landuse": "residential"})
    assert not feature_matches("built", {"landuse": "farmland"})
    assert feature_matches("green", {"leisure": "park"})
    assert feature_matches("green", {"natural": "wood"})
    assert feature_matches("centres", {"landuse": "retail"})
    assert not feature_matches("centres", {"landuse": "residential"})
    assert feature_matches("unbuildable", {"natural": "water"})
    assert feature_matches("unbuildable", {"waterway": "dock"})  # recovered from other_tags, merged in
    assert feature_matches("unbuildable", {"aeroway": "aerodrome"})  # airports
    assert feature_matches("unbuildable", {"landuse": "military"})
    assert feature_matches("unbuildable", {"military": "danger_area"})  # any military=* value
    assert not feature_matches("unbuildable", {"landuse": "residential"})


def test_feature_matches_streets_and_pt():
    assert feature_matches("streets", {"highway": "residential"})
    assert not feature_matches("streets", {"name": "x"})
    assert feature_matches("pt", {"railway": "halt"})


def test_pt_stop_kind_and_attributes():
    assert pt_stop_kind({"railway": "tram_stop"}) == "tram"
    assert pt_stop_kind({"railway": "station"}) == "rail"
    assert pt_stop_kind({"railway": "halt"}) == "rail"
    assert pt_stop_kind({"public_transport": "station"}) == "rail"
    assert pt_stop_kind({"highway": "bus_stop"}) == "bus"
    assert pt_stop_kind({"public_transport": "platform"}) == "bus"
    # rail/tram are the significant stops that anchor a centre; bus does not
    assert "rail" in STATION_KINDS and "tram" in STATION_KINDS and "bus" not in STATION_KINDS
    # only the pt dataset carries a persisted 'kind' field
    assert feature_attributes("pt", {"railway": "station"}) == {"kind": "rail"}
    assert feature_attributes("built", {"landuse": "residential"}) == {}
    assert DATASET_FIELDS.get("pt") == ["kind"] and DATASET_FIELDS.get("built", []) == []


def test_dataset_metadata_consistent():
    assert set(DATASET_ORDER) == set(DATASETS)
    for key in DATASET_ORDER:
        meta = DATASETS[key]
        assert meta["osm_layer"] in {"multipolygons", "lines", "points"}
        assert meta["geom_type"] in {"MultiPolygon", "MultiLineString", "Point"}
