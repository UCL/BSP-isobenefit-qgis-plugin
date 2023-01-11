""" """
from __future__ import annotations

import logging
from pathlib import Path

from qgis.core import QgsCoordinateReferenceSystem, QgsMapLayerProxyModel, QgsVectorLayer, QgsWkbTypes
from qgis.gui import QgsFileWidget, QgsMapLayerComboBox, QgsProjectionSelectionWidget
from qgis.PyQt import QtCore, QtWidgets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LayerSpec:

    layer_key: str
    qgis_layer: QgsVectorLayer

    def __init__(self, layer_key: str, qgis_layer: QgsVectorLayer) -> None:
        """ """
        self.layer_key = layer_key
        self.qgis_layer = qgis_layer


class FuturbDialog(QtWidgets.QDialog):
    """ """

    out_dir_path: Path | None
    out_file_name: str | None
    extents_layer: QgsVectorLayer | None
    exist_built_areas: QgsVectorLayer | None
    exist_green_areas: QgsVectorLayer | None
    selected_crs: QgsCoordinateReferenceSystem | None

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        """ """
        super(FuturbDialog, self).__init__(parent)
        # paths state
        self.out_dir_path = None
        self.out_file_name = None
        # extents selection
        self.extents_layer = None
        # existing build areas
        self.exist_built_areas = None
        # existing green areas
        self.exist_green_areas = None
        # CRS selection
        self.selected_crs = None
        # prepare UI
        self.setupUi()

    def setupUi(self):
        """ """
        self.setObjectName("FuturbDialog")
        self.setWindowTitle("Future Urban Growth simulator")
        # overall grid layout
        self.grid = QtWidgets.QGridLayout(self)
        # model mode buttons
        self.model_mode_label = QtWidgets.QLabel("Model", self)
        self.grid.addWidget(self.model_mode_label, 0, 0, 1, 2, alignment=QtCore.Qt.AlignCenter)
        # grid size
        self.grid_size_m_label = QtWidgets.QLabel("Grid size in metres", self)
        self.grid.addWidget(self.grid_size_m_label, 1, 0, alignment=QtCore.Qt.AlignRight)
        self.grid_size_m = QtWidgets.QLineEdit("50", self)
        self.grid.addWidget(self.grid_size_m, 1, 1)
        # iterations
        self.n_iterations_label = QtWidgets.QLabel("Iterations", self)
        self.grid.addWidget(self.n_iterations_label, 2, 0, alignment=QtCore.Qt.AlignRight)
        self.n_iterations = QtWidgets.QLineEdit("20", self)
        self.grid.addWidget(self.n_iterations, 2, 1)
        # walking distance
        self.walk_dist_label = QtWidgets.QLabel("Walkable distance (m)", self)
        self.grid.addWidget(self.walk_dist_label, 3, 0, alignment=QtCore.Qt.AlignRight)
        self.walk_dist = QtWidgets.QLineEdit("1000", self)
        self.grid.addWidget(self.walk_dist, 3, 1)
        # max population in walking distance
        self.max_local_pop_label = QtWidgets.QLabel("Max population in walking distance", self)
        self.grid.addWidget(self.max_local_pop_label, 4, 0, alignment=QtCore.Qt.AlignRight)
        self.max_local_pop = QtWidgets.QLineEdit("10000", self)
        self.grid.addWidget(self.max_local_pop, 4, 1)
        # build prob
        self.build_prob_label = QtWidgets.QLabel("Build probability", self)
        self.grid.addWidget(self.build_prob_label, 5, 0, alignment=QtCore.Qt.AlignRight)
        self.build_prob = QtWidgets.QLineEdit("0.1", self)
        self.grid.addWidget(self.build_prob, 5, 1)
        # nb centrality prob
        self.cent_prob_nb_label = QtWidgets.QLabel("Neighbouring prob.", self)
        self.grid.addWidget(self.cent_prob_nb_label, 6, 0, alignment=QtCore.Qt.AlignRight)
        self.cent_prob_nb = QtWidgets.QLineEdit("0.01", self)
        self.grid.addWidget(self.cent_prob_nb, 6, 1)
        # isolated centrality prob
        self.cent_prob_isol_label = QtWidgets.QLabel("Isolated centrality prob.", self)
        self.grid.addWidget(self.cent_prob_isol_label, 7, 0, alignment=QtCore.Qt.AlignRight)
        self.cent_prob_isol = QtWidgets.QLineEdit("0", self)
        self.grid.addWidget(self.cent_prob_isol, 7, 1)
        # random seed
        self.random_seed_label = QtWidgets.QLabel("Random Seed", self)
        self.grid.addWidget(self.random_seed_label, 8, 0, alignment=QtCore.Qt.AlignRight)
        self.random_seed = QtWidgets.QLineEdit("42", self)
        self.grid.addWidget(self.random_seed, 8, 1)
        # spacer
        self.grid.addItem(
            QtWidgets.QSpacerItem(1, 20, hPolicy=QtWidgets.QSizePolicy.Expanding, vPolicy=QtWidgets.QSizePolicy.Fixed),
            9,
            0,
            1,
            2,
        )
        # file output
        self.file_output_label = QtWidgets.QLabel("File output path", self)
        self.grid.addWidget(self.file_output_label, 10, 0, 1, 2)
        self.file_output = QgsFileWidget(self)
        self.file_output.setStorageMode(QgsFileWidget.SaveFile)
        self.file_output.fileChanged.connect(self.handle_output_path)  # type: ignore (connect works)
        self.grid.addWidget(self.file_output, 11, 0, 1, 2)
        # feedback for file path
        self.file_path_feedback = QtWidgets.QLabel("Select an output file path", self)
        self.file_path_feedback.setWordWrap(True)
        self.grid.addWidget(self.file_path_feedback, 12, 0, 1, 2)
        # spacer
        self.grid.addItem(
            QtWidgets.QSpacerItem(1, 20, hPolicy=QtWidgets.QSizePolicy.Expanding, vPolicy=QtWidgets.QSizePolicy.Fixed),
            13,
            0,
            1,
            2,
        )
        # extents
        self.extents_layer_label = QtWidgets.QLabel("Input layer indicating extents for simulation", self)
        self.grid.addWidget(self.extents_layer_label, 14, 0, 1, 2)
        self.extents_layer_box = QgsMapLayerComboBox(self)
        self.extents_layer_box.setFilters(QgsMapLayerProxyModel.PolygonLayer)
        self.extents_layer_box.setShowCrs(True)
        self.extents_layer_box.layerChanged.connect(self.handle_extents_layer)  # type: ignore (connect works)
        self.grid.addWidget(self.extents_layer_box, 15, 0, 1, 2)
        # feedback for layers selection
        self.extents_layer_feedback = QtWidgets.QLabel("Select an extents layer", self)
        self.extents_layer_feedback.setWordWrap(True)
        self.grid.addWidget(self.extents_layer_feedback, 16, 0, 1, 2)
        # spacer
        self.grid.addItem(
            QtWidgets.QSpacerItem(1, 20, hPolicy=QtWidgets.QSizePolicy.Expanding, vPolicy=QtWidgets.QSizePolicy.Fixed),
            17,
            0,
            1,
            2,
        )
        # existing built areas
        self.built_layer_label = QtWidgets.QLabel("Optional layer indicating extents for existing urban areas", self)
        self.grid.addWidget(self.built_layer_label, 18, 0, 1, 2)
        self.built_layer_box = QgsMapLayerComboBox(self)
        self.built_layer_box.setAllowEmptyLayer(True)
        self.built_layer_box.setCurrentIndex(0)  # type: ignore
        self.built_layer_box.setFilters(QgsMapLayerProxyModel.PolygonLayer)
        self.built_layer_box.setShowCrs(True)
        self.grid.addWidget(self.built_layer_box, 19, 0, 1, 2)
        # spacer
        self.grid.addItem(
            QtWidgets.QSpacerItem(1, 20, hPolicy=QtWidgets.QSizePolicy.Expanding, vPolicy=QtWidgets.QSizePolicy.Fixed),
            20,
            0,
            1,
            2,
        )
        # unbuildable areas
        self.unbuildable_layer_label = QtWidgets.QLabel("Optional layer indicating extents for unbuildable areas", self)
        self.grid.addWidget(self.unbuildable_layer_label, 20, 0, 1, 2)
        self.unbuildable_layer_box = QgsMapLayerComboBox(self)
        self.unbuildable_layer_box.setAllowEmptyLayer(True)
        self.unbuildable_layer_box.setCurrentIndex(0)  # type: ignore
        self.unbuildable_layer_box.setFilters(QgsMapLayerProxyModel.PolygonLayer)
        self.unbuildable_layer_box.setShowCrs(True)
        self.grid.addWidget(self.unbuildable_layer_box, 21, 0, 1, 2)
        # spacer
        self.grid.addItem(
            QtWidgets.QSpacerItem(1, 20, hPolicy=QtWidgets.QSizePolicy.Expanding, vPolicy=QtWidgets.QSizePolicy.Fixed),
            22,
            0,
            1,
            2,
        )
        # projection
        self.crs_label = QtWidgets.QLabel("Coordinate reference system for simulation", self)
        self.grid.addWidget(self.crs_label, 23, 0, 1, 2)
        self.crs_selection = QgsProjectionSelectionWidget(self)
        # feedback for layers selection
        # crsChanged event fires immediately, so self.crs_feedback has to exist beforehand
        self.crs_feedback = QtWidgets.QLabel("Select a CRS", self)
        self.crs_feedback.setWordWrap(True)
        self.grid.addWidget(self.crs_feedback, 24, 0, 1, 2)
        self.crs_selection.crsChanged.connect(self.handle_crs)  # type: ignore (connect works)
        self.crs_selection.setOptionVisible(QgsProjectionSelectionWidget.CurrentCrs, False)
        self.crs_selection.setOptionVisible(QgsProjectionSelectionWidget.DefaultCrs, False)
        self.crs_selection.setOptionVisible(QgsProjectionSelectionWidget.LayerCrs, True)
        self.crs_selection.setOptionVisible(QgsProjectionSelectionWidget.ProjectCrs, True)
        self.crs_selection.setOptionVisible(QgsProjectionSelectionWidget.RecentCrs, False)
        self.grid.addWidget(self.crs_selection, 25, 0, 1, 2)
        # spacer
        self.grid.addItem(
            QtWidgets.QSpacerItem(1, 20, hPolicy=QtWidgets.QSizePolicy.Expanding, vPolicy=QtWidgets.QSizePolicy.Fixed),
            26,
            0,
            1,
            2,
        )
        # Cancel / OK buttons
        self.button_box = QtWidgets.QDialogButtonBox(self)
        self.grid.addWidget(self.button_box, 27, 0, 1, 2)
        self.button_box.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

    def show(self) -> None:
        """Primes layers logic when opening dialog."""
        # reset
        self.handle_extents_layer()
        self.handle_output_path()
        return super().show()

    def reset_state(self) -> None:
        """ """
        self.button_box.button(QtWidgets.QDialogButtonBox.Ok).setDisabled(True)

    def refresh_state(self) -> None:
        """ """
        if self.extents_layer is None:
            return
        if self.out_file_name is None:
            return
        if self.out_file_name is None:
            return
        if self.selected_crs is None:
            return
        self.button_box.button(QtWidgets.QDialogButtonBox.Ok).setDisabled(False)

    def handle_output_path(self) -> None:
        """ """
        # reset
        self.reset_state()
        self.out_dir_path = None
        self.out_file_name = None
        # bail if no path provided
        out_path_str: str = self.file_output.filePath().strip()
        if out_path_str == "":
            self.file_path_feedback.setText("Simulation requires an output filepath.")
            return None
        out_path: Path = Path(out_path_str)
        # bail if parent is not valid
        if not out_path.parent.exists():
            self.file_path_feedback.setText("Filepath's parent directory does not exist.")
            return None
        # bail if a directory
        if out_path.is_dir():
            self.file_path_feedback.setText("Requires an output file name.")
            return None
        # don't save in root
        if out_path.parent.absolute() == Path("/"):
            self.file_path_feedback.setText("Select an output directory other than root.")
            return None
        # check that file path ends with .tif
        if not out_path.name.endswith(".tif") and "." in out_path.name:
            self.file_path_feedback.setText("Output extension must be .tif")
            return None
        # success
        self.file_path_feedback.setText("")
        self.out_dir_path = out_path.parent.absolute()
        self.out_file_name = out_path.name.replace(".tif", "")
        self.refresh_state()

    def handle_extents_layer(self) -> None:
        """ """
        # reset
        self.reset_state()
        self.extents_layer = None
        # check geometry
        candidate_layer: QgsVectorLayer = self.extents_layer_box.currentLayer()
        if not isinstance(candidate_layer, QgsVectorLayer):
            self.extents_layer_feedback.setText("Geometry of type Polygon required.")
            return
        geom_type: QgsWkbTypes.GeometryType = candidate_layer.geometryType()  # type: ignore
        if geom_type != QgsWkbTypes.PolygonGeometry:
            self.extents_layer_feedback.setText("Geometry of type Polygon required.")
            return
        # success
        self.extents_layer_feedback.setText("")
        self.extents_layer = self.extents_layer_box.currentLayer()
        self.crs_selection.setLayerCrs(self.extents_layer.crs())
        self.refresh_state()

    def handle_crs(self) -> None:
        """ """
        # set project CRS as an option if projected
        if self.crs_selection.crs().isGeographic():
            self.crs_feedback.setText("Please select a projected CRS")
            self.selected_crs = None
        else:
            self.crs_feedback.setText("")
            self.selected_crs = self.crs_selection.crs()
        self.refresh_state()
