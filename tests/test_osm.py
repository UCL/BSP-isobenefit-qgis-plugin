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
    bbox_to_overpass,
    build_combined_query,
    build_query,
    feature_attributes,
    feature_matches,
    is_barrier_line,
    parse_hstore,
    point_is_station,
    point_is_stop,
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
    assert "bus_stop" in build_query("stops", 0, 0, 1, 1)
    assert "station" in build_query("stations", 0, 0, 1, 1)
    assert "railway" in build_query("railways", 0, 0, 1, 1)
    assert "retail" in build_query("centres", 0, 0, 1, 1)
    q = build_query("unbuildable", 0, 0, 1, 1)
    assert "natural" in q and "aeroway" in q and "military" in q  # water + airports + military


def test_build_query_unknown_dataset_raises():
    with pytest.raises(KeyError):
        build_query("nope", 0, 0, 1, 1)


def test_build_combined_query_unions_datasets_and_dedupes():
    q = build_combined_query(["built", "centres", "streets", "stations"], 51.0, -0.5, 52.0, 0.5)
    assert "residential" in q  # built
    assert "retail" in q  # centres
    assert "highway" in q  # streets
    assert "station" in q  # stations
    assert "(51.0,-0.5,52.0,0.5)" in q  # one shared bbox on every selector
    assert "(._;>;);" in q and q.strip().endswith("out body;")  # nodes-before-ways recursion
    # selectors are de-duplicated: requesting a dataset twice doesn't repeat its selector
    assert build_combined_query(["streets", "streets"], 0, 0, 1, 1).count('["highway"]') == 1


def test_parse_hstore_basic_and_comma_value():
    tags = parse_hstore('"railway"=>"station","name"=>"Smith, John Square","operator"=>"TfL"')
    assert tags["railway"] == "station"
    assert tags["name"] == "Smith, John Square"  # comma inside a quoted value preserved
    assert tags["operator"] == "TfL"


def test_parse_hstore_empty():
    assert parse_hstore(None) == {}
    assert parse_hstore("") == {}


def test_point_is_station_and_stop():
    # significant rail/tram stations (anchor a centre)
    assert point_is_station({"railway": "station"})
    assert point_is_station({"railway": "tram_stop"})
    assert point_is_station({"public_transport": "station"})
    assert not point_is_station({"highway": "bus_stop"})
    # ordinary stops
    assert point_is_stop({"highway": "bus_stop"})
    assert point_is_stop({"public_transport": "platform"})
    assert not point_is_stop({"railway": "station"})  # a station is not an ordinary stop
    assert not point_is_stop({"amenity": "cafe"})


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
    # industrial is its own category — not residential built, and carved as unbuildable (no new housing)
    assert not feature_matches("built", {"landuse": "industrial"})
    assert feature_matches("industrial", {"landuse": "industrial"})
    assert not feature_matches("industrial", {"landuse": "residential"})
    assert feature_matches("unbuildable", {"landuse": "industrial"})


def test_is_barrier_line_and_unbuildable_query():
    # motorways, railways and rivers are carved into the unbuildable substrate (as line corridors)
    assert is_barrier_line({"highway": "motorway"})
    assert is_barrier_line({"highway": "trunk_link"})
    assert is_barrier_line({"railway": "rail"})
    assert is_barrier_line({"waterway": "river"})
    assert not is_barrier_line({"highway": "residential"})  # an ordinary street is not a barrier
    assert not is_barrier_line({"waterway": "riverbank"})  # a polygon, handled by the area filter
    # the unbuildable query fetches those barrier lines, but the polygon filter ignores them
    q = build_query("unbuildable", 0, 0, 1, 1)
    assert "motorway" in q and "railway" in q and "river" in q
    assert not feature_matches("unbuildable", {"highway": "motorway"})


def test_feature_matches_streets_and_pt():
    assert feature_matches("streets", {"highway": "residential"})
    assert not feature_matches("streets", {"name": "x"})
    assert feature_matches("stations", {"railway": "halt"})
    assert feature_matches("stops", {"highway": "bus_stop"})
    assert not feature_matches("stops", {"railway": "station"})  # stations are their own layer
    assert feature_matches("railways", {"railway": "rail"})
    assert not feature_matches("railways", {"railway": "tram_stop"})  # that's a station node, not a line


def test_streets_carry_highway_class():
    # the street network persists its highway class (for the routing graph); other datasets
    # carry no extra fields
    assert DATASET_FIELDS.get("streets") == ["highway"]
    assert DATASET_FIELDS.get("built", []) == []
    assert feature_attributes("streets", {"highway": "residential"}) == {"highway": "residential"}
    assert feature_attributes("streets", {"name": "x"}) == {}  # untagged street -> no value
    assert feature_attributes("stops", {"highway": "bus_stop"}) == {}


def test_dataset_metadata_consistent():
    assert set(DATASET_ORDER) == set(DATASETS)
    for key in DATASET_ORDER:
        meta = DATASETS[key]
        assert meta["osm_layer"] in {"multipolygons", "lines", "points"}
        assert meta["geom_type"] in {"MultiPolygon", "MultiLineString", "Point"}
