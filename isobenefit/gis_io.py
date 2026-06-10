"""GIS IO for the Isobenefit plugin.

All coordinate-reference-system handling, rasterization and raster writing lives
here, using only QGIS-bundled libraries (numpy + GDAL/OGR/OSR) — no rasterio or
shapely. Crucially, every input layer is **reprojected to the target CRS** before
rasterization (the previous implementation did not, which silently misplaced any
layer whose CRS differed from the chosen one).

The simulation core never sees any of this: it receives plain numpy arrays.
"""

from __future__ import annotations

import math

import numpy as np
from osgeo import gdal, ogr, osr
from qgis.core import QgsCoordinateTransform, QgsGeometry, QgsPalettedRasterRenderer, QgsProject
from qgis.PyQt.QtGui import QColor

# Categorical class codes for the output raster.
NODATA = 255
NATURE = 0
NEW_LOW = 1
NEW_MED = 2
NEW_HIGH = 3
CENTRE = 4
EXIST_BUILT = 5
FIXED_GREEN = 6

# (class code, (r, g, b), legend label) — palette echoes the original NetLogo scheme.
_PALETTE = [
    (NATURE, (89, 176, 60), "Nature / green"),
    (NEW_LOW, (200, 136, 68), "New built — low density"),
    (NEW_MED, (197, 86, 17), "New built — medium density"),
    (NEW_HIGH, (101, 44, 7), "New built — high density"),
    (CENTRE, (255, 255, 255), "Centrality"),
    (EXIST_BUILT, (114, 114, 114), "Existing built"),
    (FIXED_GREEN, (54, 109, 35), "Existing green / park"),
]


def prepare_grid(extents_layer, target_crs, granularity_m):
    """Compute the simulation grid for the extents, padded to the granularity.

    Returns ``(rows, cols, geotransform, bounds)`` with the bounds and transform
    expressed in the target CRS. ``bounds`` is ``(x_min, y_min, x_max, y_max)``.
    """
    xform = QgsCoordinateTransform(extents_layer.crs(), target_crs, QgsProject.instance())
    bbox = xform.transformBoundingBox(extents_layer.extent())
    g = float(granularity_m)
    x_min = math.floor(bbox.xMinimum() / g) * g
    y_min = math.floor(bbox.yMinimum() / g) * g
    x_max = math.ceil(bbox.xMaximum() / g) * g
    y_max = math.ceil(bbox.yMaximum() / g) * g
    cols = int(round((x_max - x_min) / g))
    rows = int(round((y_max - y_min) / g))
    geotransform = (x_min, g, 0.0, y_max, 0.0, -g)
    return rows, cols, geotransform, (x_min, y_min, x_max, y_max)


def _srs_from_crs(target_crs) -> "osr.SpatialReference":
    srs = osr.SpatialReference()
    srs.ImportFromWkt(target_crs.toWkt())
    return srs


def burn_layer(arr, layer, target_crs, geotransform, burn_value):
    """Reproject ``layer`` to the target CRS and burn ``burn_value`` into ``arr``.

    Returns a new int16 array; the input is not mutated. Reprojection is the fix
    for the long-standing CRS bug — geometries are transformed before rasterizing.
    """
    rows, cols = arr.shape
    srs = _srs_from_crs(target_crs)
    mem_rast = gdal.GetDriverByName("MEM").Create("", cols, rows, 1, gdal.GDT_Int16)
    mem_rast.SetGeoTransform(geotransform)
    mem_rast.SetProjection(srs.ExportToWkt())
    mem_rast.GetRasterBand(1).WriteArray(arr.astype(np.int16))

    ogr_ds = ogr.GetDriverByName("Memory").CreateDataSource("burn")
    ogr_layer = ogr_ds.CreateLayer("g", srs, ogr.wkbPolygon)
    xform = QgsCoordinateTransform(layer.crs(), target_crs, QgsProject.instance())
    defn = ogr_layer.GetLayerDefn()
    for feat in layer.getFeatures():
        geom = QgsGeometry(feat.geometry())
        if geom.isEmpty():
            continue
        geom.transform(xform)
        og = ogr.CreateGeometryFromWkt(geom.asWkt())
        if og is None:
            continue
        ogr_feat = ogr.Feature(defn)
        ogr_feat.SetGeometry(og)
        ogr_layer.CreateFeature(ogr_feat)

    gdal.RasterizeLayer(mem_rast, [1], ogr_layer, burn_values=[burn_value], options=["ALL_TOUCHED=FALSE"])
    out = mem_rast.GetRasterBand(1).ReadAsArray()
    return out.astype(np.int16)


def point_cells(layer, target_crs, geotransform, rows, cols):
    """Reproject point features to the target CRS and return in-bounds (row, col)."""
    inv = gdal.InvGeoTransform(geotransform)
    xform = QgsCoordinateTransform(layer.crs(), target_crs, QgsProject.instance())
    seeds = []
    for feat in layer.getFeatures():
        geom = QgsGeometry(feat.geometry())
        if geom.isEmpty():
            continue
        geom.transform(xform)
        pt = geom.asPoint()
        px, py = gdal.ApplyGeoTransform(inv, pt.x(), pt.y())
        col = int(px)
        row = int(py)
        if 0 <= row < rows and 0 <= col < cols:
            seeds.append((row, col))
    return seeds


def classify(state, origin, density, per_block):
    """Map the simulation arrays to a uint8 categorical raster (see class codes).

    ``per_block`` is ``(high, med, low)`` persons-per-block; built cells carry one
    of these exact values, so density tiers can be matched directly.
    """
    high_pb, med_pb, low_pb = per_block
    cls = np.full(state.shape, NODATA, dtype=np.uint8)
    cls[state == 0] = NATURE
    built = state == 1
    cls[built & np.isclose(density, low_pb)] = NEW_LOW
    cls[built & np.isclose(density, med_pb)] = NEW_MED
    cls[built & np.isclose(density, high_pb)] = NEW_HIGH
    cls[state == 2] = CENTRE
    # existing (origin) features take visual precedence
    cls[origin == 1] = EXIST_BUILT
    cls[origin == 0] = FIXED_GREEN
    return cls


def write_class_raster(path, cls_arr, geotransform, target_crs):
    """Write a single-band Byte GeoTIFF of class codes (with nodata)."""
    rows, cols = cls_arr.shape
    srs = _srs_from_crs(target_crs)
    ds = gdal.GetDriverByName("GTiff").Create(
        path, cols, rows, 1, gdal.GDT_Byte, options=["COMPRESS=DEFLATE", "TILED=YES"]
    )
    ds.SetGeoTransform(geotransform)
    ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1)
    band.WriteArray(cls_arr.astype(np.uint8))
    band.SetNoDataValue(NODATA)
    band.FlushCache()


def apply_palette(rast_layer):
    """Apply the categorical colour palette to a loaded raster layer."""
    classes = [
        QgsPalettedRasterRenderer.Class(value, QColor(r, g, b), label)
        for value, (r, g, b), label in _PALETTE
    ]
    renderer = QgsPalettedRasterRenderer(rast_layer.dataProvider(), 1, classes)
    rast_layer.setRenderer(renderer)
