""" """
from __future__ import annotations

import logging
from pathlib import Path

from qgis.core import QgsCoordinateReferenceSystem, QgsFeature, QgsMapLayerProxyModel, QgsVectorLayer
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
    selected_layer: QgsVectorLayer | None
    selected_feature: QgsFeature | None
    selected_crs: QgsCoordinateReferenceSystem | None

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        """ """
        super(FuturbDialog, self).__init__(parent)
        # paths state
        self.out_dir_path = None
        self.out_file_name = None
        # layer selection
        self.selected_layer = None
        self.selected_feature = None
        # CRS selection
        self.selected_crs = None
        # prepare UI
        self.setupUi()

    def toggle_model_btns(self, mode: str) -> None:
        """Pressing one button should cancel the other."""
        if mode == "isobenefit" and self.classical_btn.isChecked():
            self.isobenefit_btn.setChecked(True)
            self.classical_btn.setChecked(False)
        elif mode == "classical" and self.isobenefit_btn.isChecked():
            self.isobenefit_btn.setChecked(False)
            self.classical_btn.setChecked(True)

    def setupUi(self):
        """ """
        self.setObjectName("FuturbDialog")
        self.setWindowTitle("Future Urban Growth simulator")
        # overall grid layout
        self.grid = QtWidgets.QGridLayout(self)
        # model mode buttons
        self.model_mode_label = QtWidgets.QLabel("Model", self)
        self.grid.addWidget(self.model_mode_label, 0, 0, 1, 2, alignment=QtCore.Qt.AlignCenter)
        # isobenefit
        self.isobenefit_btn = QtWidgets.QPushButton("Isobenefit", self)
        self.isobenefit_btn.setCheckable(True)
        self.isobenefit_btn.clicked.connect(lambda: self.toggle_model_btns("isobenefit"))
        self.isobenefit_btn.setFixedWidth(175)
        self.grid.addWidget(self.isobenefit_btn, 1, 0)
        # classical button
        self.classical_btn = QtWidgets.QPushButton("Classical", self)
        self.classical_btn.setCheckable(True)
        self.classical_btn.clicked.connect(lambda: self.toggle_model_btns("classical"))
        self.classical_btn.setFixedWidth(175)
        self.grid.addWidget(self.classical_btn, 1, 1)
        # prime to isoboenefit mode
        self.toggle_model_btns("isobenefit")
        # grid size
        self.grid_size_m_label = QtWidgets.QLabel("Grid size in metres", self)
        self.grid.addWidget(self.grid_size_m_label, 2, 0, alignment=QtCore.Qt.AlignRight)
        self.grid_size_m = QtWidgets.QLineEdit("100", self)
        self.grid.addWidget(self.grid_size_m, 2, 1)
        # iterations
        self.n_iterations_label = QtWidgets.QLabel("Iterations", self)
        self.grid.addWidget(self.n_iterations_label, 3, 0, alignment=QtCore.Qt.AlignRight)
        self.n_iterations = QtWidgets.QLineEdit("5", self)
        self.grid.addWidget(self.n_iterations, 3, 1)
        # max population
        self.max_population_label = QtWidgets.QLabel("Max Population", self)
        self.grid.addWidget(self.max_population_label, 4, 0, alignment=QtCore.Qt.AlignRight)
        self.max_population = QtWidgets.QLineEdit("500000", self)
        self.grid.addWidget(self.max_population, 4, 1)
        # max population in walking distance
        self.max_local_pop_label = QtWidgets.QLabel("Max population in walking distance", self)
        self.grid.addWidget(self.max_local_pop_label, 5, 0, alignment=QtCore.Qt.AlignRight)
        self.max_local_pop = QtWidgets.QLineEdit("10000", self)
        self.grid.addWidget(self.max_local_pop, 5, 1)
        # build prob
        self.build_prob_label = QtWidgets.QLabel("Build probability", self)
        self.grid.addWidget(self.build_prob_label, 6, 0, alignment=QtCore.Qt.AlignRight)
        self.build_prob = QtWidgets.QLineEdit("0.5", self)
        self.grid.addWidget(self.build_prob, 6, 1)
        # nb centrality prob
        self.nb_cent_label = QtWidgets.QLabel("Neighbouring prob.", self)
        self.grid.addWidget(self.nb_cent_label, 7, 0, alignment=QtCore.Qt.AlignRight)
        self.nb_cent = QtWidgets.QLineEdit("0.005", self)
        self.grid.addWidget(self.nb_cent, 7, 1)
        # isolated centrality prob
        self.isolated_cent_label = QtWidgets.QLabel("Isolated centrality prob.", self)
        self.grid.addWidget(self.isolated_cent_label, 8, 0, alignment=QtCore.Qt.AlignRight)
        self.isolated_cent = QtWidgets.QLineEdit("0.1", self)
        self.grid.addWidget(self.isolated_cent, 8, 1)
        # walking distance
        self.walk_dist_label = QtWidgets.QLabel("Walkable distance (m)", self)
        self.grid.addWidget(self.walk_dist_label, 9, 0, alignment=QtCore.Qt.AlignRight)
        self.walk_dist = QtWidgets.QLineEdit("1000", self)
        self.grid.addWidget(self.walk_dist, 9, 1)
        # random seed
        self.random_seed_label = QtWidgets.QLabel("Random Seed", self)
        self.grid.addWidget(self.random_seed_label, 10, 0, alignment=QtCore.Qt.AlignRight)
        self.random_seed = QtWidgets.QLineEdit("42", self)
        self.grid.addWidget(self.random_seed, 10, 1)
        # spacer
        self.grid.addItem(
            QtWidgets.QSpacerItem(1, 20, hPolicy=QtWidgets.QSizePolicy.Expanding, vPolicy=QtWidgets.QSizePolicy.Fixed),
            11,
            0,
            1,
            2,
        )
        # file output
        self.file_output_label = QtWidgets.QLabel("File output path", self)
        self.grid.addWidget(self.file_output_label, 12, 0, 1, 2)
        self.file_output = QgsFileWidget(self)
        self.file_output.setStorageMode(QgsFileWidget.SaveFile)
        self.file_output.fileChanged.connect(self.handle_output_path)  # type: ignore (connect works)
        self.grid.addWidget(self.file_output, 13, 0, 1, 2)
        # feedback for file path
        self.file_path_feedback = QtWidgets.QLabel("Select an output file path", self)
        self.file_path_feedback.setWordWrap(True)
        self.grid.addWidget(self.file_path_feedback, 14, 0, 1, 2)
        # spacer
        self.grid.addItem(
            QtWidgets.QSpacerItem(1, 20, hPolicy=QtWidgets.QSizePolicy.Expanding, vPolicy=QtWidgets.QSizePolicy.Fixed),
            15,
            0,
            1,
            2,
        )
        # layers list
        self.layers_list_label = QtWidgets.QLabel("Input layer indicating extents for simulation", self)
        self.grid.addWidget(self.layers_list_label, 16, 0, 1, 2)
        self.layer_box = QgsMapLayerComboBox(self)
        self.layer_box.setFilters(QgsMapLayerProxyModel.VectorLayer)
        self.layer_box.setShowCrs(True)
        self.layer_box.layerChanged.connect(self.handle_layer)  # type: ignore (connect works)
        self.grid.addWidget(self.layer_box, 17, 0, 1, 2)
        # feedback for layers selection
        self.layers_feedback = QtWidgets.QLabel("Select an input layer", self)
        self.layers_feedback.setWordWrap(True)
        self.grid.addWidget(self.layers_feedback, 18, 0, 1, 2)
        # spacer
        self.grid.addItem(
            QtWidgets.QSpacerItem(1, 20, hPolicy=QtWidgets.QSizePolicy.Expanding, vPolicy=QtWidgets.QSizePolicy.Fixed),
            19,
            0,
            1,
            2,
        )
        # projection
        self.crs_label = QtWidgets.QLabel("Coordinate reference system for simulation", self)
        self.grid.addWidget(self.crs_label, 21, 0, 1, 2)
        self.crs_selection = QgsProjectionSelectionWidget(self)
        # feedback for layers selection
        # crsChanged event fires immediately, so self.crs_feedback has to exist beforehand
        self.crs_feedback = QtWidgets.QLabel("Select a CRS", self)
        self.crs_feedback.setWordWrap(True)
        self.grid.addWidget(self.crs_feedback, 23, 0, 1, 2)
        self.crs_selection.crsChanged.connect(self.handle_crs)  # type: ignore (connect works)
        self.crs_selection.setOptionVisible(QgsProjectionSelectionWidget.CurrentCrs, False)
        self.crs_selection.setOptionVisible(QgsProjectionSelectionWidget.DefaultCrs, False)
        self.crs_selection.setOptionVisible(QgsProjectionSelectionWidget.LayerCrs, True)
        self.crs_selection.setOptionVisible(QgsProjectionSelectionWidget.ProjectCrs, True)
        self.crs_selection.setOptionVisible(QgsProjectionSelectionWidget.RecentCrs, False)
        self.grid.addWidget(self.crs_selection, 22, 0, 1, 2)
        # spacer
        self.grid.addItem(
            QtWidgets.QSpacerItem(1, 20, hPolicy=QtWidgets.QSizePolicy.Expanding, vPolicy=QtWidgets.QSizePolicy.Fixed),
            24,
            0,
            1,
            2,
        )
        # Cancel / OK buttons
        self.button_box = QtWidgets.QDialogButtonBox(self)
        self.grid.addWidget(self.button_box, 25, 0, 1, 2)
        self.button_box.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

    def show(self) -> None:
        """Primes layers logic when opening dialog."""
        # reset
        self.handle_layer()
        self.handle_output_path()
        return super().show()

    def reset_state(self) -> None:
        """ """
        self.button_box.button(QtWidgets.QDialogButtonBox.Ok).setDisabled(True)

    def refresh_state(self) -> None:
        """ """
        if self.selected_layer is None:
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

    def handle_layer(self) -> None:
        """ """
        # reset
        self.reset_state()
        self.selected_layer = None
        selected_layer: QgsVectorLayer = self.layer_box.currentLayer()
        # unpack the layer's features
        layer_features: list[QgsFeature] = [sl for sl in selected_layer.getFeatures()]  # type: ignore
        # bail if no features
        if not layer_features:
            self.layers_feedback.setText("No features available on the provided layer.")
            return None
        # check for selected features
        selected_features: list[QgsFeature] = selected_layer.selectedFeatures()  # type: ignore
        if selected_features:
            # bail if more than one selected
            if len(selected_features) > 1:
                self.layers_feedback.setText(
                    "Layer contains multiple selected features: try again with a single selected feature."
                )
                return None
            selected_feature: QgsFeature = selected_features[0]
        # otherwise, if nothing has been selected, take a look at the layers features
        else:
            # bail if more than one feature
            if len(layer_features) > 1:
                self.layers_feedback.setText(
                    "Layer contains multiple features but none have been selected: try again with a single selected feature."
                )
                return None
            selected_feature: QgsFeature = layer_features[0]
        # success
        self.layers_feedback.setText("")
        self.selected_layer = selected_layer
        self.selected_feature = selected_feature
        self.crs_selection.setLayerCrs(self.selected_layer.crs())
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
