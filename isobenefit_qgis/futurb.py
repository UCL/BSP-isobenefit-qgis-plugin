""" """

from __future__ import annotations

import os.path
from pathlib import Path
from typing import Any, Callable

from qgis.core import Qgis, QgsApplication, QgsMessageLog
from qgis.gui import QgisInterface
from qgis.PyQt.QtCore import QCoreApplication, QSettings, QTranslator, qVersion
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QMessageBox, QToolBar, QWidget

from . import bootstrap, osm_fetcher, sim_runner
from .isobenefit_dialog import IsobenefitDialog  # Import the code for the dialog
from .osm_dialog import OsmDialog


class Isobenefit:
    """QGIS Plugin Implementation."""

    iface: QgisInterface
    plugin_dir: str
    dlg: IsobenefitDialog
    actions: list[QAction]
    menu: str
    toolbar: QToolBar

    def __init__(self, iface: QgisInterface):
        """Constructor.

        :param iface: An interface instance that will be passed to this class
            which provides the hook by which you can manipulate the QGIS
            application at run time.
        :type iface: QgsInterface
        """
        # Save reference to the QGIS interface
        self.iface = iface
        # initialize plugin directory
        self.plugin_dir = os.path.dirname(__file__)
        # initialize locale
        locale = QSettings().value("locale/userLocale")[0:2]
        locale_path = os.path.join(self.plugin_dir, "i18n", "isobenefit_{}.qm".format(locale))
        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)
            if qVersion() > "4.3.3":
                QCoreApplication.installTranslator(self.translator)
        # Create the dialog (after translation) and keep reference
        self.dlg = IsobenefitDialog()
        # the OSM extraction dialog is created lazily (it needs the iface)
        self.osm_dlg: OsmDialog | None = None
        # Declare instance attributes
        self.actions = []
        self.menu = self.tr("Isobenefit")
        # TODO: We are going to let the user set this up in a future iteration
        self.toolbar = self.iface.addToolBar("Isobenefit")
        self.toolbar.setObjectName("Isobenefit")
        # keep a reference to the running tasks so they are not garbage-collected
        self._task: sim_runner.IsobenefitTask | None = None
        self._osm_task: osm_fetcher.OsmFetchTask | None = None

    def tr(self, message: str):
        """Get the translation for a string using Qt translation API.

        We implement this ourselves since we do not inherit QObject.

        :param message: String for translation.
        :type message: str, QString

        :returns: Translated version of message.
        :rtype: QString
        """
        return QCoreApplication.translate("Isobenefit", message)

    def add_action(
        self,
        icon_path: str,
        text: str,
        callback: Callable[[Any], Any],
        enabled_flag: bool = True,
        add_to_menu: bool = True,
        add_to_toolbar: bool = True,
        status_tip: str | None = None,
        whats_this: str | None = None,
        parent: QWidget | None = None,
    ):
        """Add a toolbar icon to the toolbar.

        :param icon_path: Path to the icon for this action. Can be a resource
            path (e.g. ':/plugins/foo/bar.png') or a normal file system path.
        :type icon_path: str

        :param text: Text that should be shown in menu items for this action.
        :type text: str

        :param callback: Function to be called when the action is triggered.
        :type callback: function

        :param enabled_flag: A flag indicating if the action should be enabled
            by default. Defaults to True.
        :type enabled_flag: bool

        :param add_to_menu: Flag indicating whether the action should also
            be added to the menu. Defaults to True.
        :type add_to_menu: bool

        :param add_to_toolbar: Flag indicating whether the action should also
            be added to the toolbar. Defaults to True.
        :type add_to_toolbar: bool

        :param status_tip: Optional text to show in a popup when mouse pointer
            hovers over the action.
        :type status_tip: str

        :param parent: Parent widget for the new action. Defaults None.
        :type parent: QWidget

        :param whats_this: Optional text to show in the status bar when the
            mouse pointer hovers over the action.

        :returns: The action that was created. Note that the action is also
            added to self.actions list.
        :rtype: QAction
        """

        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        if status_tip is not None:
            action.setStatusTip(status_tip)

        if whats_this is not None:
            action.setWhatsThis(whats_this)

        if add_to_toolbar:
            self.toolbar.addAction(action)

        if add_to_menu:
            # Customize the iface method if you want to add to a specific
            # menu (for example iface.addToVectorMenu):
            self.iface.addPluginToMenu(self.menu, action)

        self.actions.append(action)

        return action

    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""
        plugin_dir = Path(os.path.dirname(__file__))
        self.add_action(
            str(plugin_dir / "icon.png"),
            text=self.tr("Isobenefit Urbanism"),
            callback=self.run,
            parent=self.iface.mainWindow(),
        )
        self.add_action(
            str(plugin_dir / "osm_icon.png"),
            text=self.tr("Extract from OpenStreetMap"),
            callback=self.run_osm,
            parent=self.iface.mainWindow(),
        )

    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""
        for action in self.actions:
            self.iface.removePluginMenu(self.tr("Isobenefit Urbanism"), action)
            self.iface.removeToolBarIcon(action)
        # remove the toolbar
        del self.toolbar

    def run_osm(self):
        """Show the OpenStreetMap extraction dialog (modeless) so the canvas stays live.

        Fully independent of the simulation: it only writes editable layers to a
        GeoPackage and loads them into the project (no engine needed). The dialog is
        modeless so the user can draw the area of interest on the map; the fetch is
        queued from its ``accepted`` signal.
        """
        if self.osm_dlg is None:
            self.osm_dlg = OsmDialog(self.iface, self.iface.mainWindow())
            self.osm_dlg.accepted.connect(self._start_osm_fetch)
        self.osm_dlg.show()
        self.osm_dlg.raise_()
        self.osm_dlg.activateWindow()

    def _start_osm_fetch(self):
        """Build and queue the OSM download task from the dialog's current selection."""
        dlg = self.osm_dlg
        bbox = dlg.aoi_bbox_4326()
        datasets = dlg.selected_datasets()
        gpkg_path = dlg.output_path()
        if bbox is None or not datasets or gpkg_path is None:
            QMessageBox.warning(
                self.iface.mainWindow(),
                "Isobenefit",
                "Define an area of interest, at least one dataset, and an output GeoPackage.",
            )
            return
        task = osm_fetcher.OsmFetchTask(
            iface=self.iface,
            bbox=bbox,
            aoi_wkt=dlg.aoi_polygon_wkt_4326(),
            datasets=datasets,
            gpkg_path=gpkg_path,
            group_name=dlg.suggested_group_name(),
        )
        self._osm_task = task  # retain reference so the task is not garbage-collected
        QgsApplication.taskManager().addTask(task)
        QgsMessageLog.logMessage("OpenStreetMap download queued.", level=Qgis.MessageLevel.Info, notifyUser=True)

    def run(self):
        """Run method that performs all the real work"""
        # show the dialog
        self.dlg.show()
        result: int = self.dlg.exec()  # returns 1 if pressed
        if not result:
            QgsMessageLog.logMessage("No input for Isobenefit dialogue to process.", level=Qgis.MessageLevel.Warning)
            return
        # ensure the Rust simulation core is installed and compatible
        if not bootstrap.ensure_core(self.iface.mainWindow()):
            return
        # collect + validate the numeric parameters (a friendly message, not a traceback)
        try:
            total_iters = int(self.dlg.n_iterations.text())
            granularity_m = int(self.dlg.grid_size_m.text())
            max_distance_m = int(self.dlg.walk_dist.text())
            max_populat = int(self.dlg.max_populat.text())
            exist_built_density = int(self.dlg.built_density.text())
            min_green_span = int(self.dlg.min_green_span.text())
            random_seed = int(self.dlg.random_seed.text())
            build_prob = float(self.dlg.build_prob.text())
            cent_prob_nb = float(self.dlg.cent_prob_nb.text())
            cent_prob_isol = float(self.dlg.cent_prob_isol.text())
            pop_target_cent_threshold = float(self.dlg.pop_target_cent_threshold.text())
            prob_distribution = (
                float(self.dlg.high_density_prob.text()),
                float(self.dlg.med_density_prob.text()),
                float(self.dlg.low_density_prob.text()),
            )
            density_factors = (
                float(self.dlg.high_density.text()),
                float(self.dlg.med_density.text()),
                float(self.dlg.low_density.text()),
            )
        except ValueError:
            QMessageBox.warning(
                self.iface.mainWindow(),
                "Isobenefit",
                "Please enter valid numbers for all simulation parameters before running.",
            )
            return

        # build and queue the background task (the core is array-in / array-out;
        # all GIS IO is handled inside the task via gis_io).
        # likelihood map blends N runs (Detail picker); a single run = growth animation.
        n_ensemble = self.dlg.detail_mode.currentData() if self.dlg.ensemble_check.isChecked() else 1
        task = sim_runner.IsobenefitTask(
            iface=self.iface,
            out_dir_path=self.dlg.out_dir_path,
            out_file_name=self.dlg.out_file_name,
            target_crs=self.dlg.selected_crs,
            extents_layer=self.dlg.extents_layer,
            built_layer=self.dlg.built_layer_box.currentLayer(),
            green_layer=self.dlg.green_layer_box.currentLayer(),
            unbuildable_layer=self.dlg.unbuildable_layer_box.currentLayer(),
            centre_seeds_layer=self.dlg.centre_seeds_layer_box.currentLayer(),
            transit_stops_layer=self.dlg.transit_stops_layer_box.currentLayer(),
            stations_layer=self.dlg.stations_layer_box.currentLayer(),
            streets_layer=self.dlg.streets_layer_box.currentLayer(),
            total_iters=total_iters,
            granularity_m=granularity_m,
            max_distance_m=max_distance_m,
            max_populat=max_populat,
            exist_built_density=exist_built_density,
            min_green_span=min_green_span,
            build_prob=build_prob,
            cent_prob_nb=cent_prob_nb,
            cent_prob_isol=cent_prob_isol,
            pop_target_cent_threshold=pop_target_cent_threshold,
            prob_distribution=prob_distribution,
            density_factors=density_factors,
            random_seed=random_seed,
            n_ensemble=n_ensemble,
            optimise_centres=self.dlg.optimise_centres_check.isChecked(),
        )
        self._task = task  # retain reference so the task is not garbage-collected
        QgsApplication.taskManager().addTask(task)
        QgsMessageLog.logMessage("Isobenefit simulation queued.", level=Qgis.MessageLevel.Info, notifyUser=True)
