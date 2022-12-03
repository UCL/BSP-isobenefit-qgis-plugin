""" """
from pathlib import Path

from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsMesh,
    QgsMeshDatasetIndex,
    QgsMeshLayer,
    QgsPoint,
    QgsProject,
    QgsProviderRegistry,
    QgsVectorLayer,
)


def create_raster_base(
    output_path: Path, qgs_input_extents: QgsVectorLayer, crs: QgsCoordinateReferenceSystem, granularity_m: int
) -> QgsMeshLayer:
    """ """
    print(output_path, qgs_input_extents, granularity_m)
    # ref: https://gis.stackexchange.com/questions/427211/why-does-adding-a-face-to-qgsmeshlayer-produce-uniquesharedvertex-error
    # prepare extents
    x_min = qgs_input_extents.extent().xMinimum()
    x_max = qgs_input_extents.extent().xMaximum()
    y_min = qgs_input_extents.extent().yMinimum()
    y_max = qgs_input_extents.extent().yMaximum()
    x_min = int(x_min - x_min % granularity_m)
    x_max = int(x_max - x_max % granularity_m) + granularity_m
    y_min = int(y_min - y_min % granularity_m)
    y_max = int(y_max - y_max % granularity_m) + granularity_m
    # prepare points
    points: list[QgsPoint] = []
    face_idx = 0
    faces: dict[int, tuple[int, int, int, int]] = {}
    xs = list(range(x_min, x_max + 1, granularity_m))
    ys = list(range(y_min, y_max + 1, granularity_m))
    for row_idx, y in enumerate(ys):
        for col_idx, x in enumerate(xs):
            points.append(QgsPoint(x, y, 0))
            if row_idx > 0 and col_idx > 0:
                prev_row_col_idx = (row_idx - 1) * len(xs) + col_idx
                this_row_col_idx = row_idx * len(xs) + col_idx
                faces[face_idx] = (prev_row_col_idx - 1, prev_row_col_idx, this_row_col_idx, this_row_col_idx - 1)
                face_idx += 1
