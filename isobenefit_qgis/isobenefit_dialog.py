""" """

from __future__ import annotations

from pathlib import Path

from qgis.core import Qgis, QgsCoordinateReferenceSystem, QgsVectorLayer
from qgis.gui import QgsFileWidget, QgsMapLayerComboBox, QgsProjectionSelectionWidget
from qgis.PyQt import QtCore, QtWidgets


class IsobenefitDialog(QtWidgets.QDialog):
    """ """

    prob_sum: float | None
    out_dir_path: Path | None
    out_file_name: str | None
    extents_layer: QgsVectorLayer | None
    exist_built_areas: QgsVectorLayer | None
    exist_green_areas: QgsVectorLayer | None
    selected_crs: QgsCoordinateReferenceSystem | None

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        """ """
        super(IsobenefitDialog, self).__init__(parent)
        self.prob_sum = None
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
        self.setObjectName("IsobenefitDialog")
        self.setWindowTitle("Isobenefit Urbanism")
        # main layout
        main_layout = QtWidgets.QVBoxLayout(self)
        self.resize(550, 700)
        # scroll
        self.scroll = QtWidgets.QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        # add to main layout
        main_layout.addWidget(self.scroll)
        # wrap grid layout inside a generic widget (scroll doesn't accept grid layout directly)
        self.content = QtWidgets.QWidget()
        self.scroll.setWidget(self.content)
        # grid layout
        self.grid = QtWidgets.QGridLayout(self.content)
        # heading
        self.model_mode_label = QtWidgets.QLabel("Simulator parameters", self)
        self.grid.addWidget(self.model_mode_label, 0, 0, 1, 2, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        # left column container
        self.left_col = QtWidgets.QGridLayout(self)
        self.grid.addLayout(self.left_col, 1, 0)
        # iterations
        self.n_iterations_label = QtWidgets.QLabel("Max iterations", self)
        self.left_col.addWidget(self.n_iterations_label, 0, 0, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        self.n_iterations = QtWidgets.QLineEdit("100", self)
        self.left_col.addWidget(self.n_iterations, 0, 1)
        # grid size
        self.grid_size_m_label = QtWidgets.QLabel("Grid size in metres", self)
        self.left_col.addWidget(self.grid_size_m_label, 1, 0, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        self.grid_size_m = QtWidgets.QLineEdit("50", self)
        self.left_col.addWidget(self.grid_size_m, 1, 1)
        # walking distance
        self.walk_dist_label = QtWidgets.QLabel("Walkable distance (m)", self)
        self.left_col.addWidget(self.walk_dist_label, 2, 0, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        self.walk_dist = QtWidgets.QLineEdit("800", self)
        self.left_col.addWidget(self.walk_dist, 2, 1)
        # max population
        self.max_populat_label = QtWidgets.QLabel("Target population", self)
        self.left_col.addWidget(self.max_populat_label, 3, 0, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        self.max_populat = QtWidgets.QLineEdit("100000", self)
        self.left_col.addWidget(self.max_populat, 3, 1)
        # min green km2
        self.min_green_span_label = QtWidgets.QLabel("Min green span (m)", self)
        self.left_col.addWidget(self.min_green_span_label, 4, 0, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        self.min_green_span = QtWidgets.QLineEdit("800", self)
        self.left_col.addWidget(self.min_green_span, 4, 1)
        # ensemble likelihood map (on) vs a single growth animation (off)
        self.ensemble_check = QtWidgets.QCheckBox("Development likelihood (blend many runs)", self)
        self.ensemble_check.setChecked(True)
        self.ensemble_check.setToolTip(
            "On: blend many simulations into a probability-of-development map.\n"
            "Off: a single growth animation over time."
        )
        self.left_col.addWidget(self.ensemble_check, 5, 0, 1, 2)
        # detail = how many simulations to blend (precision vs time; noise ~ 1/sqrt(runs))
        self.detail_label = QtWidgets.QLabel("Detail", self)
        self.left_col.addWidget(self.detail_label, 6, 0, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        self.detail_mode = QtWidgets.QComboBox(self)
        # Network routing optimises every run, so the ensemble is kept small (the runs are similar);
        # a few is enough to surface a good layout.
        self.detail_mode.addItem("Quick (5 runs)", 5)
        self.detail_mode.addItem("Standard (10 runs)", 10)
        self.detail_mode.addItem("Thorough (25 runs)", 25)
        self.detail_mode.setCurrentIndex(1)
        self.left_col.addWidget(self.detail_mode, 6, 1)
        self.ensemble_check.toggled.connect(self.detail_mode.setEnabled)
        # recommended plan: optimise the simulation's grown centres (re-position, add where
        # under-served, cull redundant) vs keep them exactly as the simulation grew them
        self.optimise_centres_check = QtWidgets.QCheckBox("Optimise centre placement", self)
        self.optimise_centres_check.setChecked(True)
        self.optimise_centres_check.setToolTip(
            "On: re-position the recommended plan's centres central to their development, add "
            "centres where new development is under-served, and remove redundant ones.\n"
            "Off: keep the centres exactly where the simulation grew them."
        )
        self.left_col.addWidget(self.optimise_centres_check, 7, 0, 1, 2)
        self.ensemble_check.toggled.connect(self.optimise_centres_check.setEnabled)

        # right column container
        self.right_col = QtWidgets.QGridLayout(self)
        self.grid.addLayout(self.right_col, 1, 1)
        # build prob
        self.build_prob_label = QtWidgets.QLabel("Build probability", self)
        self.right_col.addWidget(self.build_prob_label, 0, 0, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        self.build_prob = QtWidgets.QLineEdit("0.25", self)
        self.right_col.addWidget(self.build_prob, 0, 1)
        # nb centrality prob
        self.cent_prob_nb_label = QtWidgets.QLabel("Neighbouring prob", self)
        self.right_col.addWidget(self.cent_prob_nb_label, 1, 0, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        self.cent_prob_nb = QtWidgets.QLineEdit("0.01", self)
        self.right_col.addWidget(self.cent_prob_nb, 1, 1)
        # isolated centrality prob
        self.cent_prob_isol_label = QtWidgets.QLabel("Isolated centrality prob", self)
        self.right_col.addWidget(self.cent_prob_isol_label, 2, 0, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        self.cent_prob_isol = QtWidgets.QLineEdit("0", self)
        self.right_col.addWidget(self.cent_prob_isol, 2, 1)
        # centrality permitted threshold
        self.pop_target_cent_threshold_label = QtWidgets.QLabel("Pop threshold for centres", self)
        self.right_col.addWidget(
            self.pop_target_cent_threshold_label, 3, 0, alignment=QtCore.Qt.AlignmentFlag.AlignRight
        )
        self.pop_target_cent_threshold = QtWidgets.QLineEdit("0.8", self)
        self.right_col.addWidget(self.pop_target_cent_threshold, 3, 1)
        # random seed
        self.random_seed_label = QtWidgets.QLabel("Random Seed", self)
        self.right_col.addWidget(self.random_seed_label, 4, 0, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        self.random_seed = QtWidgets.QLineEdit("42", self)
        self.right_col.addWidget(self.random_seed, 4, 1)

        # spacer
        self.grid.addItem(
            QtWidgets.QSpacerItem(
                1, 20, hPolicy=QtWidgets.QSizePolicy.Policy.Expanding, vPolicy=QtWidgets.QSizePolicy.Policy.Fixed
            ),
            2,
            0,
            1,
            2,
        )

        self.dens_block = QtWidgets.QGridLayout(self)
        self.grid.addLayout(self.dens_block, 3, 0, 1, 2)
        # low density
        self.low_density_label = QtWidgets.QLabel("Low density (km2)", self)
        self.dens_block.addWidget(self.low_density_label, 0, 0, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        self.low_density = QtWidgets.QLineEdit("1000", self)
        self.low_density.textChanged.connect(self.handle_densities)
        self.dens_block.addWidget(self.low_density, 0, 1)
        self.low_density_prob = QtWidgets.QLineEdit("0.2", self)
        self.low_density_prob.textChanged.connect(self.handle_densities)
        self.dens_block.addWidget(self.low_density_prob, 0, 2)
        # medium density
        self.med_density_label = QtWidgets.QLabel("Medium density (km2)", self)
        self.dens_block.addWidget(self.med_density_label, 1, 0, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        self.med_density = QtWidgets.QLineEdit("3000", self)
        self.med_density.textChanged.connect(self.handle_densities)
        self.dens_block.addWidget(self.med_density, 1, 1)
        self.med_density_prob = QtWidgets.QLineEdit("0.4", self)
        self.med_density_prob.textChanged.connect(self.handle_densities)
        self.dens_block.addWidget(self.med_density_prob, 1, 2)
        # high density
        self.high_density_label = QtWidgets.QLabel("High density (km2)", self)
        self.dens_block.addWidget(self.high_density_label, 2, 0, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        self.high_density = QtWidgets.QLineEdit("6000", self)
        self.high_density.textChanged.connect(self.handle_densities)
        self.dens_block.addWidget(self.high_density, 2, 1)
        self.high_density_prob = QtWidgets.QLineEdit("0.4", self)
        self.high_density_prob.textChanged.connect(self.handle_densities)
        self.dens_block.addWidget(self.high_density_prob, 2, 2)
        # built density
        self.built_density_label = QtWidgets.QLabel("Built density (km2)", self)
        self.dens_block.addWidget(self.built_density_label, 3, 0, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        self.built_density = QtWidgets.QLineEdit("2000", self)
        self.dens_block.addWidget(self.built_density, 3, 1, 1, 1)
        # densities and related probabilities
        self.density_text_feedback = QtWidgets.QLabel("Density probabilities must sum to 1", self)
        self.dens_block.addWidget(self.density_text_feedback, 4, 0, 1, 3, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        # spacer
        self.grid.addItem(
            QtWidgets.QSpacerItem(
                1, 20, hPolicy=QtWidgets.QSizePolicy.Policy.Expanding, vPolicy=QtWidgets.QSizePolicy.Policy.Fixed
            ),
            4,
            0,
            1,
            2,
        )

        # files and inputs
        self.inputs_outputs_block = QtWidgets.QGridLayout(self)
        self.grid.addLayout(self.inputs_outputs_block, 5, 0, 1, 2)
        # file output
        self.file_output_label = QtWidgets.QLabel("File output path", self)
        self.inputs_outputs_block.addWidget(self.file_output_label, 0, 0, 1, 2)
        self.file_output = QgsFileWidget(self)
        self.file_output.setStorageMode(QgsFileWidget.StorageMode.SaveFile)
        self.file_output.fileChanged.connect(self.handle_output_path)  # type: ignore (connect works)
        self.inputs_outputs_block.addWidget(self.file_output, 1, 0, 1, 2)
        # feedback for file path
        self.file_path_feedback = QtWidgets.QLabel("Select an output file path", self)
        self.file_path_feedback.setWordWrap(True)
        self.inputs_outputs_block.addWidget(self.file_path_feedback, 2, 0, 1, 2)
        # spacer
        self.inputs_outputs_block.addItem(
            QtWidgets.QSpacerItem(
                1, 20, hPolicy=QtWidgets.QSizePolicy.Policy.Expanding, vPolicy=QtWidgets.QSizePolicy.Policy.Fixed
            ),
            3,
            0,
            1,
            2,
        )
        # extents
        self.extents_layer_label = QtWidgets.QLabel("Input layer indicating extents for simulation", self)
        self.inputs_outputs_block.addWidget(self.extents_layer_label, 4, 0, 1, 2)
        self.extents_layer_box = QgsMapLayerComboBox(self)
        self.extents_layer_box.setFilters(Qgis.LayerFilter.PolygonLayer)
        self.extents_layer_box.setShowCrs(True)
        self.extents_layer_box.layerChanged.connect(self.handle_extents_layer)  # type: ignore (connect works)
        self.inputs_outputs_block.addWidget(self.extents_layer_box, 5, 0, 1, 2)
        # feedback for layers selection
        self.extents_layer_feedback = QtWidgets.QLabel("Select an extents layer", self)
        self.extents_layer_feedback.setWordWrap(True)
        self.inputs_outputs_block.addWidget(self.extents_layer_feedback, 6, 0, 1, 2)
        # spacer
        self.inputs_outputs_block.addItem(
            QtWidgets.QSpacerItem(
                1, 20, hPolicy=QtWidgets.QSizePolicy.Policy.Expanding, vPolicy=QtWidgets.QSizePolicy.Policy.Fixed
            ),
            7,
            0,
            1,
            2,
        )
        # existing built areas
        self.built_layer_label = QtWidgets.QLabel("Extents for existing urban areas [optional]", self)
        self.inputs_outputs_block.addWidget(self.built_layer_label, 8, 0, 1, 2)
        self.built_layer_box = QgsMapLayerComboBox(self)
        self.built_layer_box.setAllowEmptyLayer(True)
        self.built_layer_box.setCurrentIndex(0)  # type: ignore
        self.built_layer_box.setFilters(Qgis.LayerFilter.PolygonLayer)
        self.built_layer_box.setShowCrs(True)
        self.inputs_outputs_block.addWidget(self.built_layer_box, 9, 0, 1, 2)

        # green areas
        self.green_layer_label = QtWidgets.QLabel("Extents for existing green areas [optional]", self)
        self.inputs_outputs_block.addWidget(self.green_layer_label, 10, 0, 1, 2)
        self.green_layer_box = QgsMapLayerComboBox(self)
        self.green_layer_box.setAllowEmptyLayer(True)
        self.green_layer_box.setCurrentIndex(0)  # type: ignore
        self.green_layer_box.setFilters(Qgis.LayerFilter.PolygonLayer)
        self.green_layer_box.setShowCrs(True)
        self.inputs_outputs_block.addWidget(self.green_layer_box, 11, 0, 1, 2)
        # unbuildable areas
        self.unbuildable_layer_label = QtWidgets.QLabel("Extents for unbuildable areas [optional]", self)
        self.inputs_outputs_block.addWidget(self.unbuildable_layer_label, 12, 0, 1, 2)
        self.unbuildable_layer_box = QgsMapLayerComboBox(self)
        self.unbuildable_layer_box.setAllowEmptyLayer(True)
        self.unbuildable_layer_box.setCurrentIndex(0)  # type: ignore
        self.unbuildable_layer_box.setFilters(Qgis.LayerFilter.PolygonLayer)
        self.unbuildable_layer_box.setShowCrs(True)
        self.inputs_outputs_block.addWidget(self.unbuildable_layer_box, 13, 0, 1, 2)
        # centres — polygon areas (true-area centres) or point seeds
        self.centre_seeds_layer_label = QtWidgets.QLabel("Urban centres — areas or point seeds [optional]", self)
        self.inputs_outputs_block.addWidget(self.centre_seeds_layer_label, 14, 0, 1, 2)
        self.centre_seeds_layer_box = QgsMapLayerComboBox(self)
        self.centre_seeds_layer_box.setAllowEmptyLayer(True)
        self.centre_seeds_layer_box.setCurrentIndex(0)  # type: ignore
        self.centre_seeds_layer_box.setFilters(Qgis.LayerFilter.PolygonLayer | Qgis.LayerFilter.PointLayer)
        self.centre_seeds_layer_box.setShowCrs(True)
        self.inputs_outputs_block.addWidget(self.centre_seeds_layer_box, 15, 0, 1, 2)
        # public-transport stops — point layer; used for the plan's transit-access readout
        self.transit_stops_layer_label = QtWidgets.QLabel("Public-transport stops — points [optional]", self)
        self.inputs_outputs_block.addWidget(self.transit_stops_layer_label, 16, 0, 1, 2)
        self.transit_stops_layer_box = QgsMapLayerComboBox(self)
        self.transit_stops_layer_box.setAllowEmptyLayer(True)
        self.transit_stops_layer_box.setCurrentIndex(0)  # type: ignore
        self.transit_stops_layer_box.setFilters(Qgis.LayerFilter.PointLayer)
        self.transit_stops_layer_box.setShowCrs(True)
        self.inputs_outputs_block.addWidget(self.transit_stops_layer_box, 17, 0, 1, 2)
        # rail/tram stations — a separate point layer (edited/swapped on its own); the
        # significant stops that also anchor a centre in the recommended plan
        self.stations_layer_label = QtWidgets.QLabel("Rail / tram stations — points [optional]", self)
        self.inputs_outputs_block.addWidget(self.stations_layer_label, 18, 0, 1, 2)
        self.stations_layer_box = QgsMapLayerComboBox(self)
        self.stations_layer_box.setAllowEmptyLayer(True)
        self.stations_layer_box.setCurrentIndex(0)  # type: ignore
        self.stations_layer_box.setFilters(Qgis.LayerFilter.PointLayer)
        self.stations_layer_box.setShowCrs(True)
        self.inputs_outputs_block.addWidget(self.stations_layer_box, 19, 0, 1, 2)
        # street network — line layer; when supplied, walking distances are measured along the
        # network instead of straight across the grid
        self.streets_layer_label = QtWidgets.QLabel("Street network — lines [optional, enables routing]", self)
        self.inputs_outputs_block.addWidget(self.streets_layer_label, 20, 0, 1, 2)
        self.streets_layer_box = QgsMapLayerComboBox(self)
        self.streets_layer_box.setAllowEmptyLayer(True)
        self.streets_layer_box.setCurrentIndex(0)  # type: ignore
        self.streets_layer_box.setFilters(Qgis.LayerFilter.LineLayer)
        self.streets_layer_box.setShowCrs(True)
        self.inputs_outputs_block.addWidget(self.streets_layer_box, 21, 0, 1, 2)
        # spacer
        self.inputs_outputs_block.addItem(
            QtWidgets.QSpacerItem(
                1, 20, hPolicy=QtWidgets.QSizePolicy.Policy.Expanding, vPolicy=QtWidgets.QSizePolicy.Policy.Fixed
            ),
            22,
            0,
            1,
            2,
        )
        # projection
        self.crs_label = QtWidgets.QLabel("Coordinate reference system for simulation", self)
        self.inputs_outputs_block.addWidget(self.crs_label, 23, 0, 1, 2)
        self.crs_selection = QgsProjectionSelectionWidget(self)
        # feedback for layers selection
        # crsChanged event fires immediately, so self.crs_feedback has to exist beforehand
        self.crs_feedback = QtWidgets.QLabel("Select a CRS", self)
        self.crs_feedback.setWordWrap(True)
        self.inputs_outputs_block.addWidget(self.crs_feedback, 24, 0, 1, 2)
        self.crs_selection.crsChanged.connect(self.handle_crs)  # type: ignore (connect works)
        self.crs_selection.setOptionVisible(QgsProjectionSelectionWidget.CrsOption.CurrentCrs, False)
        self.crs_selection.setOptionVisible(QgsProjectionSelectionWidget.CrsOption.DefaultCrs, False)
        self.crs_selection.setOptionVisible(QgsProjectionSelectionWidget.CrsOption.LayerCrs, True)
        self.crs_selection.setOptionVisible(QgsProjectionSelectionWidget.CrsOption.ProjectCrs, True)
        self.crs_selection.setOptionVisible(QgsProjectionSelectionWidget.CrsOption.RecentCrs, False)
        self.inputs_outputs_block.addWidget(self.crs_selection, 25, 0, 1, 2)
        # spacer
        self.inputs_outputs_block.addItem(
            QtWidgets.QSpacerItem(
                1, 20, hPolicy=QtWidgets.QSizePolicy.Policy.Expanding, vPolicy=QtWidgets.QSizePolicy.Policy.Fixed
            ),
            26,
            0,
            1,
            2,
        )
        # Cancel / OK buttons
        self.button_box = QtWidgets.QDialogButtonBox(self)
        self.grid.addWidget(self.button_box, 6, 0, 1, 2)
        self.button_box.setStandardButtons(
            QtWidgets.QDialogButtonBox.StandardButton.Cancel | QtWidgets.QDialogButtonBox.StandardButton.Ok
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

    def show(self) -> None:
        """Primes layers logic when opening dialog."""
        # reset
        self.handle_extents_layer()
        self.handle_densities()
        self.handle_output_path()
        return super().show()

    def reset_state(self) -> None:
        """ """
        self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setDisabled(True)

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
        if self.prob_sum != 1:
            return
        self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setDisabled(False)

    def handle_densities(self) -> None:
        """ """
        try:
            high_prob = float(self.high_density_prob.text())
            med_prob = float(self.med_density_prob.text())
            low_prob = float(self.low_density_prob.text())
            self.prob_sum = round(high_prob + med_prob + low_prob, 2)
            if self.prob_sum != 1:
                self.reset_state()
                self.density_text_feedback.setText("Density probabilities must sum to 1")
                return
            if int(self.high_density.text()) <= int(self.med_density.text()):
                self.density_text_feedback.setText("High density must be greater than medium")
                self.reset_state()
                return
            if int(self.med_density.text()) <= int(self.low_density.text()):
                self.density_text_feedback.setText("Medium density must be greater than low")
                self.reset_state()
                return
            self.refresh_state()
            self.density_text_feedback.setText("")
        except Exception:
            self.reset_state()
            self.density_text_feedback.setText("Density probabilities must sum to 1")

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
        geom_type = candidate_layer.geometryType()
        if geom_type != Qgis.GeometryType.Polygon:
            self.extents_layer_feedback.setText("Geometry of type Polygon required.")
            return
        # success
        self.extents_layer_feedback.setText("")
        self.extents_layer = candidate_layer
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
