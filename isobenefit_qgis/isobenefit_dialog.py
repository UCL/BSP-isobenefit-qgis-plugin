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

from .params_io import load_params_report
from .validation import check_density_tiers


class IsobenefitDialog(QtWidgets.QDialog):
    """ """

    prob_sum: float | None
    out_dir_path: Path | None
    out_file_name: str | None
    extents_layer: QgsVectorLayer | None
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
        # Status line explaining why Run is disabled — created up front so any handler can update it.
        self.run_status = QtWidgets.QLabel("", self)
        self.run_status.setWordWrap(True)
        self.run_status.setStyleSheet("color: #a00;")
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

        # --- Parameters preset / cache ---------------------------------------------------
        # Every run writes a <output>_params.json sidecar; this loads one back (or a scenario
        # preset from scenarios/<scenario>/params.json) to repopulate the dialog.
        pre = _group("Parameters")
        self.load_params_btn = QtWidgets.QPushButton("Load parameters from a previous run or scenario…", self)
        self.load_params_btn.clicked.connect(self.handle_load_params)
        pre.addRow(self.load_params_btn)
        self.params_feedback = QtWidgets.QLabel(
            "Each run saves its settings next to the output as *_params.json.", self
        )
        self.params_feedback.setWordWrap(True)
        pre.addRow(self.params_feedback)

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
        self.dispersal_mode = QtWidgets.QComboBox(self)
        self.dispersal_mode.addItem("Off (compact)", 0.0)
        self.dispersal_mode.addItem("Moderate", 0.0001)
        self.dispersal_mode.addItem("Aggressive", 0.04)
        self.dispersal_mode.setCurrentIndex(1)  # Moderate by default
        self.dispersal_mode.setToolTip(
            "How readily new settlements form away from existing development (satellite/leapfrog growth).\n"
            "Off: one compact, contiguous town. Moderate/Aggressive: increasingly polycentric."
        )
        sim.addRow("Dispersed development", self.dispersal_mode)
        self.random_seed = QtWidgets.QLineEdit("42", self)
        sim.addRow("Random seed", self.random_seed)

        # --- Walkable access ----------------------------------------------------------
        acc = _group("Walkable access")
        self.centre_walk_dist = QtWidgets.QLineEdit("1200", self)
        self.centre_walk_dist.setToolTip(
            "How far people walk to a centre. Growth uses the larger of the two walks; "
            "the finished plan is scored against each separately."
        )
        acc.addRow("Centre walk (m)", self.centre_walk_dist)
        self.green_walk_dist = QtWidgets.QLineEdit("400", self)
        self.green_walk_dist.setToolTip("How far people will walk to a park.")
        acc.addRow("Green walk (m)", self.green_walk_dist)
        self.stop_catchment_dist = QtWidgets.QLineEdit("400", self)
        self.stop_catchment_dist.setToolTip(
            "How far people walk to a bus stop. Sets the catchment around the transit "
            "corridor layer that the corridor preference acts on."
        )
        acc.addRow("Stop catchment (m)", self.stop_catchment_dist)
        self.hub_catchment_dist = QtWidgets.QLineEdit("1200", self)
        self.hub_catchment_dist.setToolTip(
            "How far people walk to a rail/tram station or designated hub. Sets the wider "
            "catchment around the transit hub layer; the default matches the centre walk."
        )
        acc.addRow("Hub catchment (m)", self.hub_catchment_dist)
        self.corridor_weight = QtWidgets.QLineEdit("0", self)
        self.corridor_weight.setToolTip(
            "Transit-oriented growth. 0 (default): the corridor layer is reported only and growth "
            "is unchanged. Higher values scale down development outside the stop catchment, "
            "concentrating growth along transit; 1 confines growth to the catchment entirely. "
            "Needs a transit corridor or hub layer."
        )
        acc.addRow("Corridor preference (0–1)", self.corridor_weight)

        # --- Post-processing ----------------------------------------------------------
        # Turns the raw CA result into scenario options. The plugin saves the existing fabric, the
        # raw (pre-processing) plan and up to three centre options (as grown / optimised placement /
        # fewest centres), all keeping every home within the centre walk: pick from the outputs.
        pp = _group("Post-processing")
        self.optimise_centres_check = QtWidgets.QCheckBox("Optimise centre placement", self)
        self.optimise_centres_check.setChecked(True)
        self.optimise_centres_check.setToolTip(
            "On: alongside the as-grown option, save two more — optimised placement (the run's centres "
            "re-positioned to cut walking distances, plus any the provision rule requires) and fewest "
            "centres (the smallest number that keeps every home within the centre walk).\n"
            "Off: save only the as-grown option (centres kept where the simulation grew them, each "
            "still sized to its catchment)."
        )
        pp.addRow(self.optimise_centres_check)
        self.centre_m2_person = QtWidgets.QLineEdit("20", self)
        self.centre_m2_person.setToolTip(
            "How much mixed-use centre land (shops, services, civic space over homes) to provide per "
            "resident served, as a "
            "rule-of-thumb provision. Each centre grows to this m² per person in its catchment, so "
            "denser or more populous catchments get bigger centres."
        )
        pp.addRow("Centre area (m² per person)", self.centre_m2_person)
        self.min_settlement = QtWidgets.QLineEdit("2000", self)
        self.min_settlement.setToolTip(
            "The service viability threshold: the population a settlement must reach to support a "
            "minimal local centre. A detached NEW cluster below it is not viable and reverts to "
            "green (converted to cells via the mean development density), and open land unable to "
            "hold it is set aside as protected green before the run. The raw plan (before this "
            "cleanup) is always saved too, so you can see exactly what was removed."
        )
        pp.addRow("Service viability (people)", self.min_settlement)
        self.min_green_span = QtWidgets.QLineEdit("400", self)
        self.min_green_span.setToolTip("No green corridor between developments may narrow below this span.")
        pp.addRow("Min green span (m)", self.min_green_span)
        self.min_park_area = QtWidgets.QLineEdit("2", self)
        self.min_park_area.setToolTip(
            "A contiguous green area must reach this size to qualify as a park, during growth "
            "and in the scores. The 2 ha default follows Natural England's accessible natural "
            "greenspace standard."
        )
        pp.addRow("Min park area (ha)", self.min_park_area)

        # --- Density ------------------------------------------------------------------
        # Three development-density tiers (people per km²), each drawn at a probability. Every new
        # block is built at one of the three densities; the three probabilities are the mix and must
        # sum to 1. Post-processing then arranges the drawn densities so the highest sit nearest the
        # mixed-use centres. Existing fabric carries no density and no population — it is assumed to be
        # served by its own centres, so it is context only and never counted here.
        dens = _group("Development density (people per km²)")
        intro = QtWidgets.QLabel(
            "Every new block is built at one of three densities. Set each density and the share of "
            "new blocks built at it; the three shares must sum to 1.",
            self,
        )
        intro.setWordWrap(True)
        dens.addRow(intro)

        def _tier_row(label: str, density_default: str, prob_default: str, tip: str):
            """A tier row: density (people/km²) on the left, its share (0–1) on the right."""
            density_edit = QtWidgets.QLineEdit(density_default, self)
            density_edit.setToolTip(f"People per km² for the {label.lower()}-density tier.")
            prob_edit = QtWidgets.QLineEdit(prob_default, self)
            prob_edit.setToolTip(tip)
            prob_edit.setMaximumWidth(70)
            row = QtWidgets.QWidget(self)
            hb = QtWidgets.QHBoxLayout(row)
            hb.setContentsMargins(0, 0, 0, 0)
            hb.addWidget(density_edit, 1)
            hb.addWidget(QtWidgets.QLabel("share", self))
            hb.addWidget(prob_edit)
            density_edit.textChanged.connect(self.handle_densities)
            prob_edit.textChanged.connect(self.handle_densities)
            dens.addRow(label, row)
            return density_edit, prob_edit

        share_tip = "Share of new blocks built at this density (0–1). The three shares must sum to 1."
        self.high_density, self.high_prob = _tier_row("High density", "6000", "0.2", share_tip)
        self.med_density, self.med_prob = _tier_row("Medium density", "3000", "0.3", share_tip)
        self.low_density, self.low_prob = _tier_row("Low density", "1500", "0.5", share_tip)
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
        # The run writes a family of files (plans, report, parameter sidecar), so the dialog asks
        # for a folder and a run name rather than pretending a single file is produced.
        self.file_output = QgsFileWidget(self)
        self.file_output.setStorageMode(QgsFileWidget.StorageMode.GetDirectory)
        self.file_output.fileChanged.connect(self.handle_output_path)  # type: ignore (connect works)
        out.addRow("Output folder", self.file_output)
        self.run_name = QtWidgets.QLineEdit("scenario", self)
        self.run_name.setToolTip("Names the run's output files (plans, report, parameters).")
        self.run_name.textChanged.connect(self.handle_output_path)
        out.addRow("Run name", self.run_name)
        self.file_path_feedback = QtWidgets.QLabel("Select an output folder", self)
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
        self.transit_stops_layer_box = _layer_combo(
            Qgis.LayerFilter.PointLayer | Qgis.LayerFilter.LineLayer | Qgis.LayerFilter.PolygonLayer
        )
        self.transit_stops_layer_box.setToolTip(
            "Growth anchors: bus stops (points) or a proposed corridor (a drawn line). Growth "
            "gravitates to their walkable catchment when the corridor preference is above 0."
        )
        inp.addRow("Transit corridors [opt]", self.transit_stops_layer_box)
        self.stations_layer_box = _layer_combo(Qgis.LayerFilter.PointLayer)
        self.stations_layer_box.setToolTip(
            "Centre anchors: rail/tram stations and any planner-designated hub. Each anchors a "
            "pinned mixed-use centre that seeds growth and survives post-processing."
        )
        inp.addRow("Transit hubs [opt]", self.stations_layer_box)

        content_layout.addStretch(1)

        # --- status + buttons (pinned below the scroll area, always visible) -----------
        log_hint = QtWidgets.QLabel(
            "A running simulation reports per-stage progress and warnings in the Log "
            "Messages panel (View ▸ Panels ▸ Log Messages, Isobenefit tab).",
            self,
        )
        log_hint.setWordWrap(True)
        log_hint.setStyleSheet("color: #6b6b6b;")
        main_layout.addWidget(log_hint)
        main_layout.addWidget(self.run_status)
        self.button_box = QtWidgets.QDialogButtonBox(self)
        self.button_box.setStandardButtons(
            QtWidgets.QDialogButtonBox.StandardButton.Cancel | QtWidgets.QDialogButtonBox.StandardButton.Ok
        )
        # the guide and the status line both speak of "Run"; the stock OK label does not
        self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText("Run")
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
        # Derive an output folder beside the detected data: <data dir>/scenarios, or the first
        # scenarios_N that does not already hold a previous run's files. Only when the user has
        # not chosen a folder; a deliberate choice is never overwritten.
        if not self.file_output.filePath():
            src = next(iter(by_key.values())).source().split("|")[0]
            base = Path(src).parent
            if base.is_dir():
                candidate = base / "scenarios"
                version = 1
                while candidate.exists() and any(candidate.iterdir()):
                    version += 1
                    candidate = base / f"scenarios_{version}"
                try:
                    candidate.mkdir(parents=True, exist_ok=True)
                except OSError:
                    return  # unwritable data folder: leave the field for the user
                self.file_output.setFilePath(str(candidate))

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
        if not hasattr(self, "button_box"):  # a widget signal can fire before __init__ builds the buttons
            return
        self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setDisabled(True)

    def refresh_state(self) -> None:
        """Enable Run only when every requirement is met, and spell out what is still missing so a
        greyed-out button is never a mystery."""
        # The CRS widget emits crsChanged mid-__init__ (see note at its creation), which reaches here
        # before the buttons exist. Skip until the dialog is fully built — show() re-validates anyway.
        if not hasattr(self, "button_box"):
            return
        missing = []
        if self.extents_layer is None:
            missing.append("an extents layer")
        if self.out_file_name is None:
            missing.append("an output folder and run name")
        if self.selected_crs is None:
            missing.append("a projected CRS")
        if self.prob_sum is None or abs(self.prob_sum - 1.0) > 1e-3:
            missing.append("valid densities and shares (summing to 1)")
        ok = not missing
        self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setDisabled(not ok)
        self.run_status.setText("" if ok else "To enable Run, set " + ", ".join(missing) + ".")

    def handle_load_params(self) -> None:
        """Pick a params JSON (a run's ``*_params.json`` sidecar or a scenario preset) and apply it."""
        path, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load run parameters", "", "Run parameters (*.json)"
        )
        if not path:
            return
        try:
            params, ignored = load_params_report(path)
        except ValueError as exc:
            self.params_feedback.setText(str(exc))
            return
        changes = self.apply_params(params)
        label = params.get("name") or Path(path).name
        if changes:
            noun = "field" if len(changes) == 1 else "fields"
            msg = f"Loaded {label}: {len(changes)} {noun} changed: {'; '.join(changes)}."
        else:
            msg = f"Loaded {label}: every field already matched the dialog."
        if ignored:
            msg += f" Ignored keys: {', '.join(ignored)}."
        self.params_feedback.setText(msg)

    def apply_params(self, params: dict) -> list[str]:
        """Repopulate the dialog from a loaded params dict (missing keys leave fields untouched).

        Returns a readable "label old to new" entry for every field the load actually moved, so
        the feedback line can enumerate the changes. A loaded preset often matches the dialog
        (defaults and scenario presets are aligned), and a load that visibly moves nothing would
        otherwise read as a no-op.
        """

        def fmt(v: float) -> str:
            return str(int(v)) if float(v).is_integer() else str(v)

        def differs(old: str, new: str) -> bool:
            try:
                return float(old) != float(new)
            except ValueError:
                return old.strip() != new.strip()

        changes: list[str] = []

        def set_text(widget, label: str, value) -> None:
            new = fmt(value)
            old = widget.text()
            widget.setText(new)
            if differs(old, new):
                changes.append(f"{label} {old.strip() or '(empty)'} to {new}")

        line_edits = {
            "grid_size_m": (self.grid_size_m, "grid size"),
            "max_iterations": (self.n_iterations, "max iterations"),
            "target_population": (self.max_populat, "target population"),
            "build_prob": (self.build_prob, "build probability"),
            "random_seed": (self.random_seed, "random seed"),
            "centre_walk_m": (self.centre_walk_dist, "centre walk"),
            "green_walk_m": (self.green_walk_dist, "green walk"),
            "min_green_span_m": (self.min_green_span, "min green span"),
            "min_park_area_ha": (self.min_park_area, "min park area"),
            "min_settlement_pop": (self.min_settlement, "service viability"),
            "centre_m2_per_person": (self.centre_m2_person, "centre m² per person"),
            "stop_catchment_m": (self.stop_catchment_dist, "stop catchment"),
            "hub_catchment_m": (self.hub_catchment_dist, "hub catchment"),
            "corridor_weight": (self.corridor_weight, "corridor preference"),
        }
        for key, (widget, label) in line_edits.items():
            if key in params:
                set_text(widget, label, params[key])
        tier_edits = {
            "densities_km2": (
                {"high": self.high_density, "medium": self.med_density, "low": self.low_density},
                "density",
            ),
            "shares": ({"high": self.high_prob, "medium": self.med_prob, "low": self.low_prob}, "share"),
        }
        for group, (widgets, noun) in tier_edits.items():
            for tier, widget in widgets.items():
                if tier in params.get(group, {}):
                    set_text(widget, f"{tier} {noun}", params[group][tier])
        if "dispersal" in params:
            wanted = str(params["dispersal"]).lower()
            for i in range(self.dispersal_mode.count()):
                if self.dispersal_mode.itemText(i).lower().startswith(wanted[:3]):
                    if i != self.dispersal_mode.currentIndex():
                        changes.append(
                            f"dispersal {self.dispersal_mode.currentText()} to {self.dispersal_mode.itemText(i)}"
                        )
                    self.dispersal_mode.setCurrentIndex(i)
                    break
        for key, widget, label in (
            ("optimise_centres", self.optimise_centres_check, "optimise centres"),
            ("ensemble", self.ensemble_check, "ensemble"),
        ):
            if key in params:
                new_state = bool(params[key])
                if new_state != widget.isChecked():
                    changes.append(f"{label} {'off to on' if new_state else 'on to off'}")
                widget.setChecked(new_state)
        if "ensemble_runs" in params:
            runs = int(params["ensemble_runs"])
            best = min(range(self.detail_mode.count()), key=lambda i: abs(self.detail_mode.itemData(i) - runs))
            if best != self.detail_mode.currentIndex():
                changes.append(f"ensemble runs {self.detail_mode.currentData()} to {self.detail_mode.itemData(best)}")
            self.detail_mode.setCurrentIndex(best)
        if "crs" in params:
            crs = QgsCoordinateReferenceSystem(str(params["crs"]))
            if crs.isValid():
                old_id = self.crs_selection.crs().authid()
                if crs.authid() != old_id:
                    changes.append(f"CRS {old_id or '(none)'} to {crs.authid()}")
                self.crs_selection.setCrs(crs)
        self.handle_densities()
        return changes

    def handle_densities(self) -> None:
        """Validate the three density tiers and their probabilities, and show live feedback.

        The rules live in :func:`validation.check_density_tiers` (pure, headless-tested): densities
        positive and strictly descending, shares each in [0, 1] and summing to 1. ``prob_sum`` holds
        the share total (``refresh_state`` enables Run only when it is 1); it is ``None`` whenever a
        field fails to parse or the densities are invalid."""
        result = check_density_tiers(
            self.high_density.text(), self.med_density.text(), self.low_density.text(),
            self.high_prob.text(), self.med_prob.text(), self.low_prob.text(),
        )
        self.prob_sum = result.total
        self.density_text_feedback.setStyleSheet("color: #060;" if result.ok else "color: #a00;")
        self.density_text_feedback.setText(result.message)
        if result.ok:
            self.refresh_state()
        else:
            self.reset_state()

    def handle_output_path(self) -> None:
        """Validate the output folder and run name; the run's files all derive from the pair."""
        self.reset_state()
        self.out_dir_path = None
        self.out_file_name = None
        folder = self.file_output.filePath().strip()
        if folder == "":
            self.file_path_feedback.setText("Select an output folder.")
            return
        out_dir = Path(folder)
        if not out_dir.is_dir():
            self.file_path_feedback.setText("The output folder does not exist.")
            return
        if out_dir.absolute() == Path("/"):
            self.file_path_feedback.setText("Select an output folder other than root.")
            return
        name = self.run_name.text().strip().removesuffix(".tif")
        if name == "" or any(ch in name for ch in '/\\:*?"<>|'):
            self.file_path_feedback.setText("Give the run a simple name (no slashes or special characters).")
            return
        self.file_path_feedback.setText("")
        self.out_dir_path = out_dir.absolute()
        self.out_file_name = name
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
        # Steer to a LOCAL PROJECTED CRS: if the layer is already projected use its own CRS, otherwise
        # suggest the appropriate local UTM zone — never geographic. setCrs + an explicit sync make
        # sure ``selected_crs`` is populated (so the Run button can enable without a manual CRS pick).
        layer_crs = candidate_layer.crs()
        if layer_crs.isValid() and not layer_crs.isGeographic():
            chosen = layer_crs
            self.crs_selection.setLayerCrs(layer_crs)
        else:
            chosen = self._local_utm_crs(candidate_layer)
        if chosen is not None and chosen.isValid():
            self.crs_selection.setCrs(chosen)
        self.handle_crs()  # sync selected_crs (and re-validate coverage) even if crsChanged didn't fire
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
        simulation always runs in metres — and the CRS must cover the extents layer, otherwise the
        run would fail later with a cryptic transform error."""
        crs = self.crs_selection.crs()
        if not crs.isValid() or crs.isGeographic():
            self.crs_feedback.setText("Select a local projected CRS (e.g. the UTM zone) — not a geographic one")
            self.selected_crs = None
        else:
            mismatch = self._crs_outside_aoi(crs)
            if mismatch:
                self.crs_feedback.setText(mismatch)
                self.selected_crs = None
            else:
                self.crs_feedback.setText("")
                self.selected_crs = crs
        self.refresh_state()

    def _crs_outside_aoi(self, crs: QgsCoordinateReferenceSystem) -> str | None:
        """A message when the chosen CRS's area of use does not cover the extents layer (e.g. a UTM
        zone from another continent left over from a previous session or a loaded preset). Returns
        None when the CRS covers the area, or when either side cannot be established (validation is
        best-effort; a genuinely broken transform still fails at run time)."""
        if self.extents_layer is None:
            return None
        try:
            wgs84 = QgsCoordinateReferenceSystem("EPSG:4326")
            xform = QgsCoordinateTransform(self.extents_layer.crs(), wgs84, QgsProject.instance())
            aoi = xform.transformBoundingBox(self.extents_layer.extent())
            bounds = crs.bounds()
        except Exception:  # noqa: BLE001 — best-effort check; never block on the check itself failing
            return None
        if aoi.isNull() or bounds.isNull():
            return None
        if bounds.intersects(aoi):
            return None
        suggestion = self._local_utm_crs(self.extents_layer)
        hint = f" (the local UTM zone is {suggestion.authid()})" if suggestion is not None else ""
        return f"{crs.authid()} does not cover the extents layer{hint}"
