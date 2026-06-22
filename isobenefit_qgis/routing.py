"""QGIS-coupled street-network routing — true walking distances along the street graph.

``grid.py`` stays pure: by default it measures a straight-ish grid walk. When a
``NetworkRouter`` is injected (see ``grid.evaluate_plan``'s ``router`` argument), walking is
measured *along the street network* instead — built from the OSM ``streets`` layer with QGIS's
own network analysis (no extra dependency).

A router is a callable ``mask -> rows×cols metres`` (np.inf beyond ``max_distance_m``), the same
shape ``grid._walk_distance`` returns, so it drops straight into the scoring.

NOTE: this module touches the QGIS network-analysis API and the live street graph, so — like
``osm_fetcher`` — it is exercised in QGIS, not by the headless test suite. ``make_router`` never
raises: on any failure it logs and returns ``None`` so the simulation falls back to the grid walk.
"""

from __future__ import annotations

import numpy as np
from osgeo import gdal
from qgis.analysis import (
    QgsGraphAnalyzer,
    QgsGraphBuilder,
    QgsNetworkDistanceStrategy,
    QgsVectorLayerDirector,
)
from qgis.core import (
    Qgis,
    QgsFeature,
    QgsFeatureRequest,
    QgsGeometry,
    QgsMessageLog,
    QgsPointXY,
    QgsSpatialIndex,
)

LOG_TAG = "Isobenefit"

# Highway classes a pedestrian can walk. Motorway/trunk (and their links) are excluded — they are
# barriers, so the graph simply has no edge along them and the walk routes around / can't cross
# except where a walkable street bridges them. The 'highway' field is written by the OSM tool.
WALKABLE_HIGHWAY = frozenset(
    {
        "footway",
        "path",
        "pedestrian",
        "living_street",
        "residential",
        "service",
        "unclassified",
        "tertiary",
        "tertiary_link",
        "secondary",
        "secondary_link",
        "primary",
        "primary_link",
        "road",
        "track",
        "steps",
        "cycleway",
        "corridor",
        "crossing",
    }
)

# Cap on the number of distinct source vertices per routing query. Centres are few; transit stops
# can be many — beyond this we sample (and log), to bound the per-query Dijkstra count.
MAX_SOURCES = 256


def _log(message: str, level=Qgis.MessageLevel.Info) -> None:
    QgsMessageLog.logMessage(message, LOG_TAG, level)


class NetworkRouter:
    """Walking-distance router over a street graph. Call it with a bool target mask."""

    def __init__(self, graph, cell_vertex, rows, cols, max_distance_m):
        self._graph = graph
        self._cell_vertex = cell_vertex  # rows×cols int: nearest graph vertex per cell (-1 if none)
        self.rows, self.cols = rows, cols
        self.max_distance_m = float(max_distance_m)

    def __call__(self, mask) -> np.ndarray:
        field = np.full((self.rows, self.cols), np.inf)
        mask = np.asarray(mask, dtype=bool)
        # source vertices = the (deduped) nearest graph vertices of the target cells
        sources = sorted({int(self._cell_vertex[r, c]) for r, c in np.argwhere(mask)} - {-1})
        if not sources:
            return field
        if len(sources) > MAX_SOURCES:  # bound the work; sampling under-counts a few, never over
            step = len(sources) / MAX_SOURCES
            sources = [sources[int(i * step)] for i in range(MAX_SOURCES)]
            _log(f"routing: capped {len(sources)} of many target vertices (sampled).")
        # nearest-source network cost to every vertex = element-wise min over per-source Dijkstras
        best = None
        n_vertices = self._graph.vertexCount()
        for src in sources:
            _tree, cost = QgsGraphAnalyzer.dijkstra(self._graph, src, 0)
            arr = np.array(cost, dtype=float)
            arr[arr < 0] = np.inf  # QGIS marks unreachable vertices with -1
            best = arr if best is None else np.minimum(best, arr)
        # map each cell to its nearest vertex's cost, then bound by the walk limit
        flat_v = self._cell_vertex.reshape(-1)
        valid = (flat_v >= 0) & (flat_v < n_vertices)
        out = field.reshape(-1)
        out[valid] = best[flat_v[valid]]
        field = out.reshape(self.rows, self.cols)
        field[field > self.max_distance_m] = np.inf
        return field


def _walkable_streets_director(streets_layer):
    """A network director over only the walkable street features (by highway class).

    If the layer carries the ``highway`` field (OSM tool output) we keep only walkable classes;
    a layer without it (a user's own network) is used as-is.
    """
    if streets_layer.fields().indexFromName("highway") >= 0:
        walkable_ids = [f.id() for f in streets_layer.getFeatures() if f["highway"] in WALKABLE_HIGHWAY]
        sub = streets_layer.materialize(QgsFeatureRequest().setFilterFids(walkable_ids))
    else:
        sub = streets_layer
    director = QgsVectorLayerDirector(sub, -1, "", "", "", QgsVectorLayerDirector.Direction.DirectionBoth)
    director.addStrategy(QgsNetworkDistanceStrategy())  # edge cost = length (metres, in the projected CRS)
    return director


def make_router(streets_layer, target_crs, geotransform, rows, cols, granularity_m, max_distance_m):
    """Build a :class:`NetworkRouter` from a street layer, or ``None`` on any failure.

    Never raises: routing is an enhancement, so a missing/empty street layer or any
    network-analysis error degrades gracefully to the straight grid walk.
    """
    if streets_layer is None:
        return None
    try:
        director = _walkable_streets_director(streets_layer)
        builder = QgsGraphBuilder(target_crs)  # target_crs must be projected (metres)
        director.makeGraph(builder, [])
        graph = builder.graph()
        n = graph.vertexCount()
        if n == 0:
            _log("routing: street graph has no vertices — falling back to the grid walk.", Qgis.MessageLevel.Warning)
            return None
        # spatial index over graph vertices, to snap each grid cell to its nearest vertex
        index = QgsSpatialIndex()
        for i in range(n):
            p = graph.vertex(i).point()
            f = QgsFeature(i)
            f.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(p.x(), p.y())))
            index.addFeature(f)
        cell_vertex = np.full((rows, cols), -1, dtype=np.int64)
        for r in range(rows):
            for c in range(cols):
                x, y = gdal.ApplyGeoTransform(geotransform, c + 0.5, r + 0.5)  # cell centre
                nearest = index.nearestNeighbor(QgsPointXY(x, y), 1)
                if nearest:
                    cell_vertex[r, c] = nearest[0]
        _log(f"routing: street graph built ({n} vertices); walking measured along the network.")
        return NetworkRouter(graph, cell_vertex, rows, cols, max_distance_m)
    except Exception as exc:  # noqa: BLE001 — routing is optional; never break the run
        _log(f"routing: could not build the street graph ({exc}); using the grid walk.", Qgis.MessageLevel.Warning)
        return None
