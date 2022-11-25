""" """
from pathlib import Path

from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsMesh,
    QgsMeshLayer,
    QgsPoint,
    QgsProject,
    QgsProviderRegistry,
    QgsVectorLayer,
)


def create_mesh_layer(
    output_path: Path, qgs_input_extents: QgsVectorLayer, crs: QgsCoordinateReferenceSystem, granularity_m: int
) -> QgsMeshLayer:
    """ """
    print(output_path, qgs_input_extents, granularity_m)
    mesh_layer_path: str = str(output_path.absolute())
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
            points.append(QgsPoint(x, y, row_idx))
            if row_idx > 0 and col_idx > 0:
                prev_row_col_idx = (row_idx - 1) * len(xs) + col_idx
                this_row_col_idx = row_idx * len(xs) + col_idx
                faces[face_idx] = (prev_row_col_idx - 1, prev_row_col_idx, this_row_col_idx, this_row_col_idx - 1)
                face_idx += 1
    # Create mesh
    provider_meta = QgsProviderRegistry.instance().providerMetadata("mdal")
    mesh = QgsMesh()
    provider_meta.createMeshData(mesh, mesh_layer_path, "Ugrid", crs)
    layer_name = output_path.name.split(".")[0]
    mesh_layer = QgsMeshLayer(mesh_layer_path, layer_name, "mdal")
    mesh_layer.setCrs(crs)
    # add points to mesh
    crs_transform = QgsCoordinateTransform(crs, crs, QgsProject.instance())
    mesh_layer.startFrameEditing(crs_transform)
    # https://qgis.org/pyqgis/master/core/QgsMeshEditor.html#qgis.core.QgsMeshEditor
    editor = mesh_layer.meshEditor()
    editor.addPointsAsVertices(points, 1)
    for vertex_indices in faces.values():
        error = editor.addFace(vertex_indices)
        if error.errorType != 0:
            print(error.errorType)
            print(error.elementIndex)
    mesh_layer.commitFrameEditing(crs_transform, continueEditing=False)
    # https://qgis.org/pyqgis/master/core/QgsMeshDatasetGroup.html#module-QgsMeshDatasetGroup
    dsg = editor.createZValueDatasetGroup()
    print(dsg)
    print(dsg.datasetCount())
    print(dsg.maximum())
    dg = dsg.dataset(0)
    print(dg)
    print(dg.valuesCount())
    # QgsMeshDatasetSourceInterface() ?
    # see: https://docs.qgis.org/3.22/en/docs/user_manual/working_with_mesh/mesh_properties.html#datasets
    # TODO: understand datasets and datasetgroups and how to edit
    # mesh_layer.dataset...
    # https://qgis.org/pyqgis/master/core/QgsMeshDatasetSourceInterface.html#module-QgsMeshDatasetSourceInterface
    # "Datasets are grouped in the dataset groups. A dataset group represents a measured quantity (e.g. depth or wind speed), dataset represents values of the quantity in a particular time."
    return mesh_layer
