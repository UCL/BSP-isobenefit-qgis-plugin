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
            points.append(QgsPoint(x, y, row_idx))
            if row_idx > 0 and col_idx > 0:
                prev_row_col_idx = (row_idx - 1) * len(xs) + col_idx
                this_row_col_idx = row_idx * len(xs) + col_idx
                faces[face_idx] = (prev_row_col_idx - 1, prev_row_col_idx, this_row_col_idx, this_row_col_idx - 1)
                face_idx += 1
    # Create mesh
    mesh = QgsMesh()
    print(mesh)  # QgsMesh
    # uses MDAL for QgsProviderMetadata
    provider_meta = QgsProviderRegistry.instance().providerMetadata("mdal")
    print(provider_meta)  # QgsProviderMetadata
    mesh_layer_path: str = str(output_path.absolute())
    # see C++ documentation for createMeshData - not in pyqgis
    provider_meta.createMeshData(mesh, mesh_layer_path, "Ugrid", crs)
    print(provider_meta.meshDriversMetadata())  # list of QgsMeshDriverMetadata
    for driver_meta in provider_meta.meshDriversMetadata():
        # e.g.
        print(driver_meta.name())  # Ugrid
        print(driver_meta.description())  # UGRID
    # above takes care of creating the layer
    # below takes care of handling the layer
    layer_name = output_path.name.split(".")[0] + " a layer name"
    mesh_layer = QgsMeshLayer(mesh_layer_path, layer_name, "mdal")
    print(mesh_layer)  # QgsMeshLayer: "name" (mdal)
    mesh_layer.setCrs(crs)
    mesh_layer.addDatasets()
    # add points to mesh
    crs_transform = QgsCoordinateTransform(crs, crs, QgsProject.instance())
    mesh_layer.startFrameEditing(crs_transform)
    # https://qgis.org/pyqgis/master/core/QgsMeshEditor.html#qgis.core.QgsMeshEditor
    mesh_editor = mesh_layer.meshEditor()
    print(mesh_editor)  # QgsMeshEditor
    mesh_editor.addPointsAsVertices(points, 1)
    for vertex_indices in faces.values():
        error = mesh_editor.addFace(vertex_indices)
        if error.errorType != 0:
            print(error.errorType)
            print(error.elementIndex)
    # Z values can be manipulated - but not sure how this relates to datasets and dataset groups
    mesh_editor.changeZValues(list(range(100)), [20] * 100)
    mesh_layer.commitFrameEditing(crs_transform, continueEditing=False)
    mesh_layer.reload()
    # rendering settings
    mesh_renderer = mesh_layer.rendererSettings()
    mesh_renderer.setActiveScalarDatasetGroup(0)
    scalar_settings = mesh_renderer.scalarSettings(0)
    scalar_settings.setClassificationMinimumMaximum(0, 42)
    mesh_renderer.setScalarSettings(0, scalar_settings)
    mesh_layer.setRendererSettings(mesh_renderer)
    # data provider
    data_provider = mesh_layer.dataProvider()
    data_provider.addDataset()
    print(data_provider)  # QgsMeshDataProvider
    print(data_provider.name())  # mdal
    print(data_provider.datasetGroupCount())  # 1
    print(data_provider.datasetCount(0))  # 1
    print(data_provider.subLayerCount())  # 0
    # todo: how to classify?
    # this does nothing:
    mesh_layer.datasetValues(QgsMeshDatasetIndex(0, 0), 0, 100).setValues([20] * 100)
    mesh_layer.reload()
    # retrieving and editing via createZValueDatasetGroup doesn't change the mesh in place, how to write back?
    dsg = mesh_editor.createZValueDatasetGroup()
    print(dsg)  # QgsMeshDatasetGroup
    print(dsg.datasetCount())  # 1
    print(dsg.maximum())  # 42
    print(dsg.name())  # "vertices Z value"
    dsg.setName("a name")
    print(dsg.name())  # "a name"
    dg = dsg.dataset(0)
    print(dg)  # QgsMeshDataset
    print(dg.datasetValue(0))  # QgsMeshDatasetValue
    print(dg.datasetValue(0).scalar())  # 0
    print(dg.datasetValue(100).scalar())  # 2
    print(dg.datasetValue(1000).scalar())  # 21
    print(dg.datasetValues(True, 0, 2000))  # QgsMeshDataBlock
    print(dg.datasetValues(True, 0, 2000).isValid())  # True
    print(dg.datasetValues(True, 0, 2000).count())  # 2000
    print(dg.datasetValues(True, 0, 100).values())  # first 100 values
    print(dg.datasetValues(True, 0, 2000).value(0))  # QgsMeshDatasetValue
    print(dg.datasetValues(True, 0, 2000).value(0).scalar())  # 0
    print(dg.datasetValues(True, 0, 2000).value(100).scalar())  # 2
    # dg.datasetValues(True, 0, 2000).setValues([20] * 2000) # does nothing to QgsMeshLayer

    # not sure
    print(dg.datasetValues(True, 2000, 1).value(0).scalar())  # 0
    print(dg.datasetValues(True, 2000, 100).value(99).scalar())  # miniscule number?
    print(dg.valuesCount())
    # trying to edit values
    mesh_layer.startFrameEditing(crs_transform)
    # editor.changeZValues([1, 2, 3], [42, 42, 42]) # crashes
    mesh_layer.commitFrameEditing(crs_transform, continueEditing=False)
    # mesh_layer.saveDataset()
    # see: https://docs.qgis.org/3.22/en/docs/user_manual/working_with_mesh/mesh_properties.html#datasets
    # TODO: understand datasets and datasetgroups and how to edit
    # mesh_layer.dataset...
    # https://qgis.org/pyqgis/master/core/QgsMeshDatasetSourceInterface.html#module-QgsMeshDatasetSourceInterface
    # "Datasets are grouped in the dataset groups. A dataset group represents a measured quantity (e.g. depth or wind speed), dataset represents values of the quantity in a particular time."
    return mesh_layer
