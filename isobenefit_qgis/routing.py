"""Street-network routing — true walking distances along a plain in-memory graph.

The whole pipeline uses ONE distance model. By default it is the pure grid walk
(``grid._walk_distance``); when a streets layer is supplied, ``sim_runner`` injects a
:class:`NetworkRouter` here and the SAME callable drives centre placement, green-carving and
scoring — no grid-vs-network hybrid, and no silent fallback (if the graph can't be built the run
fails with a clear error).

"Solve once, portal through": the input network is fixed, so the graph and each grid cell's
on/off-ramp (its nearest node + access metres) are built ONCE and reused for every query and run.
A query is one bounded multi-source Dijkstra over the graph; a cell's walk to the targets is

    access(cell) + network distance from its node to the nearest target's node

— you walk onto the network, traverse it, and the target sits at the far node. Everything is
bounded at the walk distance.

Why a hand-built graph and not QGIS network-analysis: ``QgsVectorLayerDirector.makeGraph`` is not
safe to drive from the background task thread and crashed on a materialised layer's lifetime. We
only READ geometries from the layer (worker-safe, like ``gis_io.burn_layer``); the graph, the
cell snapping and the Dijkstra are plain NumPy/heapq, so they're also unit-tested headlessly.
QGIS imports are deferred into the layer-reading functions so this module imports without QGIS.
"""

from __future__ import annotations

import heapq
import math

import numpy as np

LOG_TAG = "Isobenefit"

# Highway classes a pedestrian can walk. Motorway/trunk (and their links) are excluded as barriers,
# so the graph has no edge along them. The 'highway' field is written by the OSM tool.
WALKABLE_HIGHWAY = frozenset(
    {
        "footway", "path", "pedestrian", "living_street", "residential", "service",
        "unclassified", "tertiary", "tertiary_link", "secondary", "secondary_link",
        "primary", "primary_link", "road", "track", "steps", "cycleway", "corridor", "crossing",
    }
)


class RoutingError(RuntimeError):
    """Raised when a street-network graph cannot be built (no silent fallback)."""


def _log(message: str) -> None:
    try:
        from qgis.core import Qgis, QgsMessageLog

        QgsMessageLog.logMessage(message, LOG_TAG, Qgis.MessageLevel.Info)
    except Exception:  # noqa: BLE001 — logging is best-effort / absent off-QGIS
        pass


class NetworkRouter:
    """Walking-distance field along a fixed street graph. Call with a bool target mask.

    Pure data in, pure array out — constructed from plain ``nodes``/``adjacency`` and the
    precomputed per-cell ``cell_node``/``cell_access`` so it can be exercised without QGIS.
    """

    def __init__(self, nodes, adjacency, cell_node, cell_access, granularity_m, max_distance_m):
        self._nodes = nodes  # N×2 float (x, y) in the projected CRS (unused at query time, kept for tests)
        self._adj = adjacency  # node -> list of (neighbour, cost_m)
        self._cell_node = np.asarray(cell_node)  # rows×cols int: nearest node per cell (-1 if none in reach)
        self._cell_access = np.asarray(cell_access, dtype=float)  # rows×cols: on/off-ramp metres
        self.granularity_m = float(granularity_m)
        self.max_distance_m = float(max_distance_m)
        self.rows, self.cols = self._cell_node.shape
        self._cache: dict = {}  # source node -> bounded distance-to-all-nodes; "solve once" per node

    def _dijkstra(self, source_nodes) -> np.ndarray:
        """Bounded multi-source Dijkstra from ``source_nodes`` (the targets' nodes, seeded at 0)."""
        dist = np.full(len(self._adj), np.inf)
        heap = []
        for n in source_nodes:
            if dist[n] != 0.0:
                dist[n] = 0.0
                heapq.heappush(heap, (0.0, n))
        max_cost = self.max_distance_m
        while heap:
            cost, u = heapq.heappop(heap)
            if cost > dist[u]:
                continue
            for v, w in self._adj[u]:
                nc = cost + w
                if nc <= max_cost and nc < dist[v]:
                    dist[v] = nc
                    heapq.heappush(heap, (nc, v))
        return dist

    def _single_source(self, node) -> np.ndarray:
        """Bounded distances from one node to all nodes — computed once per node, then cached and
        reused across the whole ensemble (this is the "solve once" the hot path relies on)."""
        field = self._cache.get(node)
        if field is None:
            field = self._dijkstra((node,))
            self._cache[node] = field
        return field

    def __call__(self, mask) -> np.ndarray:
        mask = np.asarray(mask, dtype=bool)
        field = np.full((self.rows, self.cols), np.inf)
        # The targets' nearest network nodes are the Dijkstra sources. Extract them with VECTORISED
        # numpy — a Python loop over the mask (e.g. all green cells = tens of thousands) was the
        # single-core post-processing bottleneck.
        target_nodes = self._cell_node[mask]
        target_nodes = target_nodes[target_nodes >= 0]  # cells with a usable on-ramp
        if target_nodes.size == 0:
            return field
        source_nodes = np.unique(target_nodes)
        # Single-source queries (a lone centre) dominate — from centre placement, run per centre per
        # Lloyd iteration per run — so cache them: each node's paths are solved once and reused.
        # Multi-source queries (green / all centres) are far fewer, so run them directly.
        if source_nodes.size == 1:
            gnode = self._single_source(int(source_nodes[0]))
        else:
            gnode = self._dijkstra(source_nodes)
        flat_node = self._cell_node.reshape(-1)
        valid = flat_node >= 0
        out = field.reshape(-1)
        out[valid] = gnode[flat_node[valid]] + self._cell_access.reshape(-1)[valid]
        field = out.reshape(self.rows, self.cols)
        field[field > self.max_distance_m] = np.inf
        return field


def _build_network(streets_layer, target_crs):
    """Read walkable street segments and build a plain node/edge graph in ``target_crs`` metres.

    Reads geometries only (worker-thread safe, as ``gis_io`` does). Coincident endpoints are
    merged on a 0.1 m grid so connected segments share nodes. Returns ``(nodes N×2, adjacency)``.
    """
    from qgis.core import Qgis, QgsCoordinateTransform, QgsGeometry, QgsProject

    xform = QgsCoordinateTransform(streets_layer.crs(), target_crs, QgsProject.instance())
    has_highway = streets_layer.fields().indexFromName("highway") >= 0
    node_id: dict = {}
    nodes: list = []
    adjacency: list = []

    def node_of(x, y):
        key = (round(x, 1), round(y, 1))
        nid = node_id.get(key)
        if nid is None:
            nid = len(nodes)
            node_id[key] = nid
            nodes.append((x, y))
            adjacency.append([])
        return nid

    for feat in streets_layer.getFeatures():
        if has_highway and feat["highway"] not in WALKABLE_HIGHWAY:
            continue
        geom = QgsGeometry(feat.geometry())
        if geom.isEmpty():
            continue
        if geom.transform(xform) != Qgis.GeometryOperationResult.Success:
            continue  # a misplaced segment distorts every routed distance
        parts = geom.asMultiPolyline() if geom.isMultipart() else [geom.asPolyline()]
        for part in parts:
            prev = None
            for pt in part:
                nid = node_of(pt.x(), pt.y())
                if prev is not None and nid != prev:
                    (px, py), (nx, ny) = nodes[prev], nodes[nid]
                    cost = math.hypot(nx - px, ny - py)
                    adjacency[prev].append((nid, cost))
                    adjacency[nid].append((prev, cost))
                prev = nid
    return np.array(nodes, dtype=float) if nodes else np.zeros((0, 2)), adjacency


def _snap_cells(nodes, geotransform, rows, cols, max_distance_m):
    """Nearest node + access metres for each grid cell. ``-1``/``inf`` where none is within reach.

    Pure: cell centres come from the affine geotransform; nearest node via a coarse bucket grid.
    """
    cell_node = np.full((rows, cols), -1, dtype=np.int64)
    cell_access = np.full((rows, cols), np.inf)
    if len(nodes) == 0:
        return cell_node, cell_access
    gt = geotransform
    bucket = max(float(max_distance_m), 1.0)
    buckets: dict = {}
    for i, (x, y) in enumerate(nodes):
        buckets.setdefault((int(x // bucket), int(y // bucket)), []).append(i)
    for r in range(rows):
        for c in range(cols):
            px, py = c + 0.5, r + 0.5
            x = gt[0] + px * gt[1] + py * gt[2]
            y = gt[3] + px * gt[4] + py * gt[5]
            bx, by = int(x // bucket), int(y // bucket)
            best_i, best_d = -1, math.inf
            for ox in (-1, 0, 1):
                for oy in (-1, 0, 1):
                    for i in buckets.get((bx + ox, by + oy), ()):
                        d = math.hypot(nodes[i][0] - x, nodes[i][1] - y)
                        if d < best_d:
                            best_d, best_i = d, i
            if best_i >= 0 and best_d <= max_distance_m:
                cell_node[r, c] = best_i
                cell_access[r, c] = best_d
    return cell_node, cell_access


def make_router(streets_layer, target_crs, geotransform, rows, cols, granularity_m, max_distance_m):
    """Build a :class:`NetworkRouter` from a street layer (solve-once). Raises ``RoutingError`` on
    failure — no silent fallback; the caller turns this into a clear run error.

    ``target_crs`` must be projected (metres) so segment lengths and the access are metric.
    """
    try:
        nodes, adjacency = _build_network(streets_layer, target_crs)
        if len(nodes) == 0:
            raise RoutingError("no walkable streets in the supplied network")
        cell_node, cell_access = _snap_cells(nodes, geotransform, rows, cols, max_distance_m)
        _log(f"routing: street graph built once ({len(nodes)} nodes); walking measured along the network.")
        return NetworkRouter(nodes, adjacency, cell_node, cell_access, granularity_m, max_distance_m)
    except RoutingError:
        raise
    except Exception as exc:  # noqa: BLE001 — surface as a clear routing error, never crash/fallback
        raise RoutingError(f"could not build the street-network graph: {exc}") from exc
