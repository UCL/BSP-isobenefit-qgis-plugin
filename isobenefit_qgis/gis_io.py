"""GIS IO for the Isobenefit plugin (QGIS + GDAL only; no rasterio/shapely).

The pure, QGIS-free logic (class codes, ``classify``, grid maths) lives in
``grid.py`` so it can be unit-tested in a plain venv. This module holds the
QGIS/GDAL-coupled parts: reading layers, **reprojection to the target CRS** before
rasterization (the previous implementation skipped this, silently misplacing any
layer whose CRS differed from the chosen one), raster writing, and renderers.
"""

from __future__ import annotations

import numpy as np
from osgeo import gdal, ogr, osr
from qgis.core import (
    QgsColorRampShader,
    QgsCoordinateTransform,
    QgsGeometry,
    QgsPalettedRasterRenderer,
    QgsProject,
    QgsRasterShader,
    QgsSingleBandPseudoColorRenderer,
)
from qgis.PyQt.QtGui import QColor

from .grid import NODATA, PALETTE, PLAN_PALETTE, align_bounds, classify  # noqa: F401  (re-exported)


def prepare_grid(extents_layer, target_crs, granularity_m):
    """Compute the simulation grid for the extents, reprojected to the target CRS.

    Returns ``(rows, cols, geotransform, bounds)``.
    """
    xform = QgsCoordinateTransform(extents_layer.crs(), target_crs, QgsProject.instance())
    bbox = xform.transformBoundingBox(extents_layer.extent())
    return align_bounds(bbox.xMinimum(), bbox.yMinimum(), bbox.xMaximum(), bbox.yMaximum(), float(granularity_m))


def _srs_from_crs(target_crs) -> "osr.SpatialReference":
    srs = osr.SpatialReference()
    srs.ImportFromWkt(target_crs.toWkt())
    return srs


def burn_layer(arr, layer, target_crs, geotransform, burn_value):
    """Reproject ``layer`` to the target CRS and burn ``burn_value`` into ``arr``.

    Returns a new int16 array; the input is not mutated. Geometries are transformed
    before rasterizing (the fix for the long-standing CRS bug).
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


def write_temporal_class_raster(path, frames, geotransform, target_crs):
    """Write one multi-band Byte GeoTIFF; band ``i+1`` holds step ``i``'s class codes."""
    if not frames:
        raise ValueError("no frames to write")
    rows, cols = frames[0].shape
    srs = _srs_from_crs(target_crs)
    ds = gdal.GetDriverByName("GTiff").Create(
        path, cols, rows, len(frames), gdal.GDT_Byte, options=["COMPRESS=DEFLATE", "TILED=YES"]
    )
    ds.SetGeoTransform(geotransform)
    ds.SetProjection(srs.ExportToWkt())
    for i, frame in enumerate(frames, start=1):
        band = ds.GetRasterBand(i)
        band.WriteArray(frame.astype(np.uint8))
        band.SetNoDataValue(NODATA)
        band.SetDescription(f"step {i - 1}")
    ds.FlushCache()
    ds = None


def write_probability_bands(path, bands, names, geotransform, target_crs):
    """Write a multi-band Float32 GeoTIFF — one probability surface per band."""
    rows, cols = bands[0].shape
    srs = _srs_from_crs(target_crs)
    ds = gdal.GetDriverByName("GTiff").Create(
        path, cols, rows, len(bands), gdal.GDT_Float32, options=["COMPRESS=DEFLATE", "TILED=YES"]
    )
    ds.SetGeoTransform(geotransform)
    ds.SetProjection(srs.ExportToWkt())
    for i, (arr, name) in enumerate(zip(bands, names), start=1):
        band = ds.GetRasterBand(i)
        band.WriteArray(arr.astype(np.float32))
        band.SetDescription(name)
    ds.FlushCache()
    ds = None


def apply_palette(rast_layer):
    """Apply the categorical colour palette to a loaded raster layer."""
    classes = [QgsPalettedRasterRenderer.Class(value, QColor(r, g, b), label) for value, (r, g, b), label in PALETTE]
    renderer = QgsPalettedRasterRenderer(rast_layer.dataProvider(), 1, classes)
    rast_layer.setRenderer(renderer)


def apply_probability_style(rast_layer, band=1):
    """Apply a 0–1 graduated colour ramp to ``band`` of a probability raster."""
    ramp = QgsColorRampShader(0.0, 1.0)
    ramp.setColorRampType(QgsColorRampShader.Type.Interpolated)
    ramp.setColorRampItemList(
        [
            QgsColorRampShader.ColorRampItem(0.0, QColor(255, 255, 255, 0), "0.0"),
            QgsColorRampShader.ColorRampItem(0.25, QColor(254, 224, 144), "0.25"),
            QgsColorRampShader.ColorRampItem(0.5, QColor(252, 141, 89), "0.5"),
            QgsColorRampShader.ColorRampItem(0.75, QColor(215, 48, 39), "0.75"),
            QgsColorRampShader.ColorRampItem(1.0, QColor(165, 0, 38), "1.0"),
        ]
    )
    shader = QgsRasterShader()
    shader.setRasterShaderFunction(ramp)
    renderer = QgsSingleBandPseudoColorRenderer(rast_layer.dataProvider(), band, shader)
    rast_layer.setRenderer(renderer)


def write_plan_raster(path, plan, geotransform, target_crs):
    """Write the recommended-plan categorical raster (single-band Byte; 0 = none)."""
    rows, cols = plan.shape
    srs = _srs_from_crs(target_crs)
    ds = gdal.GetDriverByName("GTiff").Create(
        path, cols, rows, 1, gdal.GDT_Byte, options=["COMPRESS=DEFLATE", "TILED=YES"]
    )
    ds.SetGeoTransform(geotransform)
    ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1)
    band.WriteArray(plan.astype(np.uint8))
    band.SetNoDataValue(0)
    ds.FlushCache()
    ds = None


def apply_plan_style(rast_layer):
    """Apply the categorical palette for the recommended-plan raster."""
    classes = [
        QgsPalettedRasterRenderer.Class(value, QColor(r, g, b), label) for value, (r, g, b), label in PLAN_PALETTE
    ]
    renderer = QgsPalettedRasterRenderer(rast_layer.dataProvider(), 1, classes)
    rast_layer.setRenderer(renderer)
