"""QGIS-coupled street-network routing — true walking distances along the street graph.

The whole pipeline uses ONE distance model. By default that is the pure grid walk
(``grid._walk_distance``), so ``grid.py`` stays QGIS-free and headlessly testable. When a streets
layer is supplied, ``sim_runner`` injects a :class:`NetworkRouter` here and the SAME callable
drives centre placement, green-carving and scoring — there is no grid-vs-network hybrid and no
silent fallback (if the graph can't be built the run fails with a clear error).

"Solve once, portal through" (the efficient model): the input street network is fixed, so the
graph and each grid cell's on/off-ramp are precomputed ONCE and reused for every query and every
run. A query is then one bounded multi-source Dijkstra:

    effective walk(cell) = min( grid walk to the target,                       # local last-mile
                                access(cell) + network distance to the target ) # portal across streets

so a cell walks onto the network, traverses it, and walks off — whichever of the local grid walk
or the network portal is shorter. Everything is bounded at the walk distance, so we never compute
farther than we care about.

NOTE: like ``osm_fetcher`` this touches the QGIS network-analysis API and is exercised in QGIS,
not by the headless suite. ``make_router`` RAISES on failure (no silent fallback) — the caller
turns that into a clear run error.
"""

from __future__ import annotations

import heapq
import math

import numpy as np
from osgeo import gdal
from qgis.analysis import (
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

from .grid import _walk_distance

LOG_TAG = "Isobenefit"

# Highway classes a pedestrian can walk. Motorway/trunk (and their links) are excluded — they are
# barriers, so the graph has no edge along them and the walk routes around / can't cross except
# where a walkable street bridges them. The 'highway' field is written by the OSM tool.
WALKABLE_HIGHWAY = frozenset(
    {
        "footway", "path", "pedestrian", "living_street", "residential", "service",
        "unclassified", "tertiary", "tertiary_link", "secondary", "secondary_link",
        "primary", "primary_link", "road", "track", "steps", "cycleway", "corridor", "crossing",
    }
)


class RoutingError(RuntimeError):
    """Raised when a street-network graph cannot be built (no silent fallback)."""


def _log(message: str, level=Qgis.MessageLevel.Info) -> None:
    QgsMessageLog.logMessage(message, LOG_TAG, level)


def _dijkstra(graph, sources: dict, max_cost: float) -> list:
    """Bounded multi-source Dijkstra over a QgsGraph. ``sources`` maps vertex id -> initial cost
    (the on-ramp access). Returns the cost to every vertex (``inf`` beyond ``max_cost``)."""
    dist = [math.inf] * graph.vertexCount()
    heap = []
    for nid, cost in sources.items():
        if cost < dist[nid]:
            dist[nid] = cost
            heapq.heappush(heap, (cost, nid))
    while heap:
        cost, u = heapq.heappop(heap)
        if cost > dist[u]:
            continue
        vertex = graph.vertex(u)
        for edge_id in vertex.outgoingEdges():
            edge = graph.edge(edge_id)
            nxt = cost + edge.cost(0)
            if nxt > max_cost:
                continue
            w = edge.toVertex()
            if nxt < dist[w]:
                dist[w] = nxt
                heapq.heappush(heap, (nxt, w))
    return dist


class NetworkRouter:
    """Effective walking-distance field along the street network. Call with a bool target mask."""

    def __init__(self, graph, cell_node, cell_access, granularity_m, max_distance_m):
        self._graph = graph
        self._cell_node = cell_node  # rows×cols int: nearest graph vertex per cell (-1 if none in reach)
        self._cell_access = cell_access  # rows×cols float: on/off-ramp metres to that vertex
        self.granularity_m = float(granularity_m)
        self.max_distance_m = float(max_distance_m)
        self.rows, self.cols = cell_node.shape

    def __call__(self, mask) -> np.ndarray:
        mask = np.asarray(mask, dtype=bool)
        # local last-mile: the plain grid walk to the targets (handles adjacency / off-network)
        grid = _walk_distance(mask, self.granularity_m, self.max_distance_m)
        # network portal: seed every target's nearest vertex with its on-ramp access, Dijkstra once
        sources: dict = {}
        ys, xs = np.nonzero(mask)
        for y, x in zip(ys.tolist(), xs.tolist()):
            node = int(self._cell_node[y, x])
            if node < 0:
                continue
            access = float(self._cell_access[y, x])
            if access < sources.get(node, math.inf):
                sources[node] = access
        if sources:
            gnode = _dijkstra(self._graph, sources, self.max_distance_m)
            node = self._cell_node  # rows×cols
            portal = np.full((self.rows, self.cols), np.inf)
            reachable = node >= 0
            node_cost = np.array([gnode[i] if i >= 0 else math.inf for i in node.reshape(-1)])
            portal_flat = node_cost + self._cell_access.reshape(-1)
            portal = portal_flat.reshape(self.rows, self.cols)
            portal[~reachable] = np.inf
            effective = np.minimum(grid, portal)
        else:
            effective = grid
        effective[effective > self.max_distance_m] = np.inf
        return effective


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
    """Build a :class:`NetworkRouter` from a street layer (solve-once). Raises ``RoutingError`` on
    failure — there is no silent fallback; the caller turns this into a clear run error.

    ``target_crs`` must be projected (metres) so edge lengths and the on-ramp access are metric.
    """
    try:
        director = _walkable_streets_director(streets_layer)
        builder = QgsGraphBuilder(target_crs)  # otf reprojection on by default -> reprojects to target_crs
        director.makeGraph(builder, [])
        graph = builder.graph()
        n = graph.vertexCount()
        if n == 0:
            raise RoutingError("the street layer produced an empty walking graph (no walkable streets in range)")
        # snap each grid cell to its nearest graph vertex (the on/off-ramp), with the access metres
        index = QgsSpatialIndex()
        for i in range(n):
            p = graph.vertex(i).point()
            feat = QgsFeature(i)
            feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(p.x(), p.y())))
            index.addFeature(feat)
        cell_node = np.full((rows, cols), -1, dtype=np.int64)
        cell_access = np.full((rows, cols), np.inf)
        for r in range(rows):
            for c in range(cols):
                x, y = gdal.ApplyGeoTransform(geotransform, c + 0.5, r + 0.5)  # cell centre
                hit = index.nearestNeighbor(QgsPointXY(x, y), 1)
                if hit:
                    nid = hit[0]
                    cell_node[r, c] = nid
                    vp = graph.vertex(nid).point()
                    cell_access[r, c] = math.hypot(vp.x() - x, vp.y() - y)
        # cells whose nearest vertex is itself beyond a walk have no usable on-ramp
        cell_node[cell_access > max_distance_m] = -1
        _log(f"routing: street graph solved once ({n} vertices); walking measured along the network.")
        return NetworkRouter(graph, cell_node, cell_access, granularity_m, max_distance_m)
    except RoutingError:
        raise
    except Exception as exc:  # noqa: BLE001 — surface as a clear routing error, no silent fallback
        raise RoutingError(f"could not build the street-network graph: {exc}") from exc
