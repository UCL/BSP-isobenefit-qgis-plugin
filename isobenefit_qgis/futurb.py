""" """
from __future__ import annotations

import os.path
from pathlib import Path
from typing import Any, Callable

from qgis.core import Qgis, QgsApplication, QgsMessageLog
from qgis.gui import QgisInterface
from qgis.PyQt.QtCore import QCoreApplication, QSettings, QTranslator, qVersion
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QToolBar, QWidget

from . import bootstrap, sim_runner
from .isobenefit_dialog import IsobenefitDialog  # Import the code for the dialog


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
        # Declare instance attributes
        self.actions = []
        self.menu = self.tr("Isobenefit")
        # TODO: We are going to let the user set this up in a future iteration
        self.toolbar = self.iface.addToolBar("Isobenefit")
        self.toolbar.setObjectName("Isobenefit")
        # keep a reference to the running task so it is not garbage-collected
        self._task: sim_runner.IsobenefitTask | None = None

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
        icon_path = Path(os.path.dirname(__file__)) / "icon.png"
        self.add_action(
            str(icon_path), text=self.tr("Isobenefit Urbanism"), callback=self.run, parent=self.iface.mainWindow()
        )

    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""
        for action in self.actions:
            self.iface.removePluginMenu(self.tr("Isobenefit Urbanism"), action)
            self.iface.removeToolBarIcon(action)
        # remove the toolbar
        del self.toolbar

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
        # build and queue the background task (the core is array-in / array-out;
        # all GIS IO is handled inside the task via gis_io)
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
            total_iters=int(self.dlg.n_iterations.text()),
            granularity_m=int(self.dlg.grid_size_m.text()),
            max_distance_m=int(self.dlg.walk_dist.text()),
            max_populat=int(self.dlg.max_populat.text()),
            exist_built_density=int(self.dlg.built_density.text()),
            min_green_span=int(self.dlg.min_green_span.text()),
            build_prob=float(self.dlg.build_prob.text()),
            cent_prob_nb=float(self.dlg.cent_prob_nb.text()),
            cent_prob_isol=float(self.dlg.cent_prob_isol.text()),
            pop_target_cent_threshold=float(self.dlg.pop_target_cent_threshold.text()),
            prob_distribution=(
                float(self.dlg.high_density_prob.text()),
                float(self.dlg.med_density_prob.text()),
                float(self.dlg.low_density_prob.text()),
            ),
            density_factors=(
                float(self.dlg.high_density.text()),
                float(self.dlg.med_density.text()),
                float(self.dlg.low_density.text()),
            ),
            random_seed=int(self.dlg.random_seed.text()),
            n_ensemble=int(self.dlg.n_ensemble.text()),
        )
        self._task = task  # retain reference so the task is not garbage-collected
        QgsApplication.taskManager().addTask(task)
        QgsMessageLog.logMessage("Isobenefit simulation queued.", level=Qgis.MessageLevel.Info, notifyUser=True)
