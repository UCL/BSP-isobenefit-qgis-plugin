""" """
from pathlib import Path

import meshio
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


def create_mesh_layer(
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
    # Create mesh
    mesh = QgsMesh()
    # uses MDAL for QgsProviderMetadata
    provider_meta = QgsProviderRegistry.instance().providerMetadata("mdal")
    mesh_layer_path: str = str(output_path.absolute())
    provider_meta.createMeshData(mesh, mesh_layer_path, "Ugrid", crs)  # see C++ docs for createMeshData (not in pyqgis)
    # above takes care of creating the layer, below takes care of handling the layer
    layer_name = output_path.name.split(".")[0] + " a layer name"
    mesh_layer = QgsMeshLayer(mesh_layer_path, layer_name, "mdal")
    mesh_layer.setCrs(crs)
    # add points to mesh
    crs_transform = QgsCoordinateTransform(crs, crs, QgsProject.instance())
    mesh_layer.startFrameEditing(crs_transform)
    mesh_editor = mesh_layer.meshEditor()  # QgsMeshEditor
    mesh_editor.addPointsAsVertices(points, 1)
    for vertex_indices in faces.values():
        mesh_editor.addFace(vertex_indices)
    mesh_layer.commitFrameEditing(crs_transform, continueEditing=False)
    # FYI: Z values can be manipulated: mesh_editor.changeZValues(list(range(100)), [20] * 100)
    # FYI: data provider = mesh_layer.dataProvider()
    # fetch Z values as DatasetGroup and edit as needed
    dsg = mesh_editor.createZValueDatasetGroup()  # QgsMeshDatasetGroup
    dsg.setName("First step")
    # FYI: dsg.dataset(0).datasetValue(1000).scalar()
    # FYI: dsg.dataset(0).datasetValues(True, 0, 2021).value(100).scalar()
    # this doesn't work
    print(dsg.dataset(0).datasetValues(True, 0, 100).values())  # all values are zero per Z
    dsg.dataset(0).datasetValues(True, 0, 100).setValues(list(range(100)))
    print(dsg.dataset(0).datasetValues(True, 0, 100).values())  # values are not updated
    # this doesn't work either
    print(dsg.dataset(0).datasetValue(1).scalar())  # 0
    dsg.dataset(0).datasetValue(1).set(10)
    print(dsg.dataset(0).datasetValue(1).scalar())  # 0 -> not updated
    # add dataset back to mesh layer
    mesh_layer.addDatasets(dsg)  # returns True
    dsg.setName("Another step")
    mesh_layer.addDatasets(dsg)
    print(dsg.dataset(0))
    print(dsg.dataset(1))
    mesh_layer.reload()
    # rendering settings
    mesh_renderer = mesh_layer.rendererSettings()
    mesh_renderer.setActiveScalarDatasetGroup(1)
    scalar_settings = mesh_renderer.scalarSettings(1)
    scalar_settings.setClassificationMinimumMaximum(1, 2021)
    mesh_renderer.setScalarSettings(1, scalar_settings)
    mesh_layer.setRendererSettings(mesh_renderer)

    # trying to edit values
    # mesh_layer.startFrameEditing(crs_transform)
    # editor.changeZValues([1, 2, 3], [42, 42, 42]) # crashes
    # mesh_layer.commitFrameEditing(crs_transform, continueEditing=False)
    # mesh_layer.saveDataset()
    # see: https://docs.qgis.org/3.22/en/docs/user_manual/working_with_mesh/mesh_properties.html#datasets
    # TODO: understand datasets and datasetgroups and how to edit
    # mesh_layer.dataset...
    # https://qgis.org/pyqgis/master/core/QgsMeshDatasetSourceInterface.html#module-QgsMeshDatasetSourceInterface
    # "Datasets are grouped in the dataset groups. A dataset group represents a measured quantity (e.g. depth or wind speed), dataset represents values of the quantity in a particular time."
    return mesh_layer
