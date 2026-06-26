""" """

from __future__ import annotations

from pathlib import Path

from qgis.core import (
    Qgis,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsProject,
    QgsVectorLayer,
)
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
        """Grouped form layout: each section is a QGroupBox with a QFormLayout (label/field rows that
        fit the width, so there is no horizontal scrollbar); the OK/Cancel buttons stay pinned below
        the scroll area so they are always reachable."""
        self.setObjectName("IsobenefitDialog")
        self.setWindowTitle("Isobenefit Urbanism")
        self.resize(580, 800)

        main_layout = QtWidgets.QVBoxLayout(self)
        # Scrollable content — VERTICAL only. Horizontal scrolling is disabled and the form fields
        # grow to the viewport width, so wide combos elide instead of forcing a horizontal bar.
        self.scroll = QtWidgets.QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        main_layout.addWidget(self.scroll)
        self.content = QtWidgets.QWidget()
        self.scroll.setWidget(self.content)
        content_layout = QtWidgets.QVBoxLayout(self.content)
        content_layout.setSpacing(10)

        def _group(title: str) -> QtWidgets.QFormLayout:
            box = QtWidgets.QGroupBox(title, self)
            form = QtWidgets.QFormLayout(box)
            form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
            form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            content_layout.addWidget(box)
            return form

        def _layer_combo(layer_filter) -> QgsMapLayerComboBox:
            box = QgsMapLayerComboBox(self)
            box.setAllowEmptyLayer(True)
            box.setCurrentIndex(0)  # type: ignore
            box.setFilters(layer_filter)
            box.setShowCrs(True)
            box.setMinimumWidth(160)
            return box

        # --- Simulation ---------------------------------------------------------------
        sim = _group("Simulation")
        self.n_iterations = QtWidgets.QLineEdit("100", self)
        self.n_iterations.setToolTip("Max growth steps; a run stops early once it hits the target population.")
        sim.addRow("Max iterations", self.n_iterations)
        self.grid_size_m = QtWidgets.QLineEdit("50", self)
        sim.addRow("Grid size (m)", self.grid_size_m)
        self.max_populat = QtWidgets.QLineEdit("100000", self)
        sim.addRow("Target population", self.max_populat)
        self.build_prob = QtWidgets.QLineEdit("0.25", self)
        self.build_prob.setToolTip("Per-step probability that an eligible cell develops (the growth rate).")
        sim.addRow("Build probability", self.build_prob)
        self.random_seed = QtWidgets.QLineEdit("42", self)
        sim.addRow("Random seed", self.random_seed)

        # --- Walkable access ----------------------------------------------------------
        acc = _group("Walkable access")
        self.centre_walk_dist = QtWidgets.QLineEdit("400", self)
        self.centre_walk_dist.setToolTip("How far people walk to a centre (the CA grows by the larger walk).")
        acc.addRow("Centre walk (m)", self.centre_walk_dist)
        self.green_walk_dist = QtWidgets.QLineEdit("400", self)
        self.green_walk_dist.setToolTip("How far people will walk to a park.")
        acc.addRow("Green walk (m)", self.green_walk_dist)

        # --- Growth & centres ---------------------------------------------------------
        gc = _group("Growth && centres")
        self.dispersal_mode = QtWidgets.QComboBox(self)
        self.dispersal_mode.addItem("Off (compact)", 0.0)
        self.dispersal_mode.addItem("Low", 0.005)
        self.dispersal_mode.addItem("Medium", 0.02)
        self.dispersal_mode.addItem("High", 0.05)
        self.dispersal_mode.setCurrentIndex(2)  # Medium by default
        self.dispersal_mode.setToolTip(
            "How readily new settlements form away from existing development (satellite/leapfrog growth).\n"
            "Off: one compact, contiguous town. Higher: increasingly polycentric."
        )
        gc.addRow("Dispersed development", self.dispersal_mode)
        self.centre_pattern_mode = QtWidgets.QComboBox(self)
        self.centre_pattern_mode.addItem("Consolidated", 1.0)
        self.centre_pattern_mode.addItem("Balanced", 0.7)
        self.centre_pattern_mode.addItem("Dispersed", 0.45)
        self.centre_pattern_mode.setCurrentIndex(1)  # Balanced by default
        self.centre_pattern_mode.setToolTip(
            "Consolidated: the fewest, largest centres that still keep everyone within a walk.\n"
            "Dispersed: more, smaller, closer centres."
        )
        gc.addRow("Centre pattern", self.centre_pattern_mode)
        self.min_settlement = QtWidgets.QLineEdit("25", self)
        self.min_settlement.setToolTip(
            "Smallest viable new settlement, as an AREA in hectares (25 ha ≈ a 500×500 m block). A "
            "smaller detached cluster with no centre is pruned as a failed satellite (reverts to green)."
        )
        gc.addRow("Min settlement area (ha)", self.min_settlement)
        self.min_green_span = QtWidgets.QLineEdit("800", self)
        self.min_green_span.setToolTip("A green patch must span at least this distance to count as a usable park.")
        gc.addRow("Min green span (m)", self.min_green_span)
        self.optimise_centres_check = QtWidgets.QCheckBox("Optimise centre placement", self)
        self.optimise_centres_check.setChecked(True)
        self.optimise_centres_check.setToolTip(
            "On: re-position the recommended plan's centres central to their development, add "
            "centres where new development is under-served, and remove redundant ones.\n"
            "Off: keep the centres exactly where the simulation grew them."
        )
        gc.addRow(self.optimise_centres_check)

        # --- Density ------------------------------------------------------------------
        dens = _group("Density (people per km²)")
        self.min_density = QtWidgets.QLineEdit("1500", self)
        self.min_density.textChanged.connect(self.handle_densities)
        self.min_density.setToolTip("Density at a catchment edge (spread across min..max, denser near centres).")
        dens.addRow("Min density", self.min_density)
        self.max_density = QtWidgets.QLineEdit("6000", self)
        self.max_density.textChanged.connect(self.handle_densities)
        self.max_density.setToolTip("Density near a centre and the densification ceiling for the green budget.")
        dens.addRow("Max density", self.max_density)
        self.built_density = QtWidgets.QLineEdit("2000", self)
        self.built_density.setToolTip("Assumed density of the existing built fabric (people it already holds).")
        dens.addRow("Existing built density", self.built_density)
        self.density_text_feedback = QtWidgets.QLabel("", self)
        self.density_text_feedback.setWordWrap(True)
        dens.addRow(self.density_text_feedback)

        # --- Output -------------------------------------------------------------------
        out = _group("Output")
        self.ensemble_check = QtWidgets.QCheckBox("Development likelihood (blend many runs)", self)
        self.ensemble_check.setChecked(True)
        self.ensemble_check.setToolTip(
            "On: blend many simulations into a probability-of-development map.\n"
            "Off: a single growth animation over time."
        )
        out.addRow(self.ensemble_check)
        self.detail_mode = QtWidgets.QComboBox(self)
        self.detail_mode.addItem("Quick (10 runs)", 10)
        self.detail_mode.addItem("Standard (50 runs)", 50)
        self.detail_mode.addItem("Thorough (100 runs)", 100)
        self.detail_mode.setCurrentIndex(1)
        out.addRow("Detail", self.detail_mode)
        self.ensemble_check.toggled.connect(self.detail_mode.setEnabled)
        self.ensemble_check.toggled.connect(self.optimise_centres_check.setEnabled)
        self.file_output = QgsFileWidget(self)
        self.file_output.setStorageMode(QgsFileWidget.StorageMode.SaveFile)
        self.file_output.fileChanged.connect(self.handle_output_path)  # type: ignore (connect works)
        out.addRow("Output file (.tif)", self.file_output)
        self.file_path_feedback = QtWidgets.QLabel("Select an output file path", self)
        self.file_path_feedback.setWordWrap(True)
        out.addRow(self.file_path_feedback)
        self.crs_selection = QgsProjectionSelectionWidget(self)
        # crsChanged fires immediately, so crs_feedback must exist beforehand
        self.crs_feedback = QtWidgets.QLabel("Select a CRS", self)
        self.crs_feedback.setWordWrap(True)
        self.crs_selection.crsChanged.connect(self.handle_crs)  # type: ignore (connect works)
        self.crs_selection.setOptionVisible(QgsProjectionSelectionWidget.CrsOption.CurrentCrs, False)
        self.crs_selection.setOptionVisible(QgsProjectionSelectionWidget.CrsOption.DefaultCrs, False)
        self.crs_selection.setOptionVisible(QgsProjectionSelectionWidget.CrsOption.LayerCrs, True)
        self.crs_selection.setOptionVisible(QgsProjectionSelectionWidget.CrsOption.ProjectCrs, True)
        self.crs_selection.setOptionVisible(QgsProjectionSelectionWidget.CrsOption.RecentCrs, False)
        out.addRow("CRS", self.crs_selection)
        out.addRow(self.crs_feedback)

        # --- Input layers -------------------------------------------------------------
        inp = _group("Input layers")
        self.extents_layer_box = QgsMapLayerComboBox(self)
        self.extents_layer_box.setFilters(Qgis.LayerFilter.PolygonLayer)
        self.extents_layer_box.setShowCrs(True)
        self.extents_layer_box.setMinimumWidth(160)
        self.extents_layer_box.layerChanged.connect(self.handle_extents_layer)  # type: ignore (connect works)
        inp.addRow("Extents (required)", self.extents_layer_box)
        self.extents_layer_feedback = QtWidgets.QLabel("Select an extents layer", self)
        self.extents_layer_feedback.setWordWrap(True)
        inp.addRow(self.extents_layer_feedback)
        self.built_layer_box = _layer_combo(Qgis.LayerFilter.PolygonLayer)
        inp.addRow("Existing urban [opt]", self.built_layer_box)
        self.green_layer_box = _layer_combo(Qgis.LayerFilter.PolygonLayer)
        inp.addRow("Existing green [opt]", self.green_layer_box)
        self.unbuildable_layer_box = _layer_combo(Qgis.LayerFilter.PolygonLayer)
        inp.addRow("Unbuildable [opt]", self.unbuildable_layer_box)
        self.centre_seeds_layer_box = _layer_combo(Qgis.LayerFilter.PolygonLayer | Qgis.LayerFilter.PointLayer)
        inp.addRow("Urban centres [opt]", self.centre_seeds_layer_box)
        self.transit_stops_layer_box = _layer_combo(Qgis.LayerFilter.PointLayer)
        inp.addRow("PT stops [opt]", self.transit_stops_layer_box)
        self.stations_layer_box = _layer_combo(Qgis.LayerFilter.PointLayer)
        inp.addRow("Rail / tram stations [opt]", self.stations_layer_box)
        self.streets_layer_box = _layer_combo(Qgis.LayerFilter.LineLayer)
        self.streets_layer_box.setToolTip("Walking is measured along this street network (enables routing).")
        inp.addRow("Street network [opt]", self.streets_layer_box)

        content_layout.addStretch(1)

        # --- buttons (pinned below the scroll area, always visible) --------------------
        self.button_box = QtWidgets.QDialogButtonBox(self)
        self.button_box.setStandardButtons(
            QtWidgets.QDialogButtonBox.StandardButton.Cancel | QtWidgets.QDialogButtonBox.StandardButton.Ok
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        main_layout.addWidget(self.button_box)

    # OSM companion dataset key -> the simulation combo it should fill (industrial/railways have none)
    _OSM_COMBO_MAP = {
        "extents": "extents_layer_box",
        "built": "built_layer_box",
        "green": "green_layer_box",
        "unbuildable": "unbuildable_layer_box",
        "centres": "centre_seeds_layer_box",
        "stops": "transit_stops_layer_box",
        "stations": "stations_layer_box",
        "streets": "streets_layer_box",
    }

    def _prepopulate_from_osm(self) -> None:
        """If the OSM companion download has added layers (each tagged with the dataset key), pre-select
        the matching layer in every combo so a run is ready straight away. Optional combos are filled
        only when still empty (a deliberate selection is never overwritten); the extents combo is set
        to the OSM extents layer whenever one is present (it also drives the CRS suggestion)."""
        by_key: dict[str, QgsVectorLayer] = {}
        for layer in QgsProject.instance().mapLayers().values():
            key = layer.customProperty("isobenefit/osm_dataset")
            if key:
                by_key[str(key)] = layer
        if not by_key:
            return
        for key, attr in self._OSM_COMBO_MAP.items():
            layer = by_key.get(key)
            combo = getattr(self, attr, None)
            if layer is None or combo is None:
                continue
            if attr == "extents_layer_box" or combo.currentLayer() is None:
                combo.setLayer(layer)

    def show(self) -> None:
        """Primes layers logic when opening dialog."""
        # pre-fill the combos from any OSM companion download before validating
        self._prepopulate_from_osm()
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
        """Validate the density range. ``prob_sum`` is kept as the OK-enable flag used by
        ``refresh_state`` (1 = valid): the model derives its tiers from min/max, so the only check
        is that the max exceeds the min."""
        try:
            mn = float(self.min_density.text())
            mx = float(self.max_density.text())
            if mx <= mn:
                self.prob_sum = None
                self.reset_state()
                self.density_text_feedback.setText("Max density must be greater than min")
                return
            self.prob_sum = 1
            self.refresh_state()
            self.density_text_feedback.setText("")
        except Exception:
            self.prob_sum = None
            self.reset_state()
            self.density_text_feedback.setText("Enter valid min and max densities")

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
        # Steer to a LOCAL PROJECTED CRS: offer the layer's own CRS only if it is already projected,
        # and default the selection to the appropriate local UTM zone (never geographic).
        layer_crs = candidate_layer.crs()
        if layer_crs.isValid() and not layer_crs.isGeographic():
            self.crs_selection.setLayerCrs(layer_crs)
        utm = self._local_utm_crs(candidate_layer)
        if utm is not None:
            self.crs_selection.setCrs(utm)
        self.refresh_state()

    def _local_utm_crs(self, layer: QgsVectorLayer) -> QgsCoordinateReferenceSystem | None:
        """The appropriate local UTM CRS for a layer's extent, so the default is always a sensible
        local PROJECTED CRS (never geographic). Returns None if it can't be derived."""
        try:
            wgs84 = QgsCoordinateReferenceSystem("EPSG:4326")
            xform = QgsCoordinateTransform(layer.crs(), wgs84, QgsProject.instance())
            centre = xform.transform(layer.extent().center())
            lon, lat = centre.x(), centre.y()
            zone = int((lon + 180.0) / 6.0) % 60 + 1
            epsg = (32600 if lat >= 0 else 32700) + zone  # WGS84 / UTM north or south
            crs = QgsCoordinateReferenceSystem.fromEpsgId(epsg)
            return crs if crs.isValid() else None
        except Exception:  # noqa: BLE001 — CRS suggestion is best-effort; fall back to the picker
            return None

    def handle_crs(self) -> None:
        """Only LOCAL PROJECTED CRSs are accepted — geographic (lat/lon) CRSs are rejected so the
        simulation always runs in metres."""
        if not self.crs_selection.crs().isValid() or self.crs_selection.crs().isGeographic():
            self.crs_feedback.setText("Select a local projected CRS (e.g. the UTM zone) — not a geographic one")
            self.selected_crs = None
        else:
            self.crs_feedback.setText("")
            self.selected_crs = self.crs_selection.crs()
        self.refresh_state()
