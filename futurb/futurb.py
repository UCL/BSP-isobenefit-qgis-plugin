""" """
from __future__ import annotations

import os.path
from typing import Any, Callable

from qgis.core import (
    Qgis,
    QgsCoordinateReferenceSystem,
    QgsFeature,
    QgsGeometry,
    QgsMessageLog,
    QgsProject,
    QgsRasterBlock,
    QgsRasterLayer,
)
from qgis.gui import QgisInterface
from qgis.PyQt.QtCore import QCoreApplication, QSettings, QTranslator, qVersion
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QToolBar, QWidget
from shapely import geometry, wkt
from shapely.geometry.polygon import orient

# Import the code for the dialog
from .futurb_dialog import FuturbDialog

# Initialize Qt resources from file resources.py
from .resources import *


class Futurb:
    """QGIS Plugin Implementation."""

    iface: QgisInterface
    plugin_dir: str
    dlg: FuturbDialog
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
        locale_path = os.path.join(self.plugin_dir, "i18n", "Futurb_{}.qm".format(locale))
        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)
            if qVersion() > "4.3.3":
                QCoreApplication.installTranslator(self.translator)
        # Create the dialog (after translation) and keep reference
        self.dlg = FuturbDialog()
        # Declare instance attributes
        self.actions = []
        self.menu = self.tr("&Future Urban Growth test.")
        # TODO: We are going to let the user set this up in a future iteration
        self.toolbar = self.iface.addToolBar("Futurb")
        self.toolbar.setObjectName("Futurb")

    def tr(self, message: str):
        """Get the translation for a string using Qt translation API.

        We implement this ourselves since we do not inherit QObject.

        :param message: String for translation.
        :type message: str, QString

        :returns: Translated version of message.
        :rtype: QString
        """
        return QCoreApplication.translate("Futurb", message)

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

        icon_path = ":/plugins/futurb/icon.png"
        self.add_action(
            icon_path, text=self.tr("Future Urban Growth test."), callback=self.run, parent=self.iface.mainWindow()
        )

    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""
        for action in self.actions:
            self.iface.removePluginMenu(self.tr("&Future Urban Growth test."), action)
            self.iface.removeToolBarIcon(action)
        # remove the toolbar
        del self.toolbar

    def run(self):
        """Run method that performs all the real work"""
        # show the dialog
        self.dlg.show()
        result: int = self.dlg.exec_()  # returns 1 if pressed
        if result:
            if self.dlg.selected_layer is None:
                self.iface.messageBar().pushMessage(
                    "Error",
                    "No layer selected.",
                    level=Qgis.Critical,
                )
                return
            canvas_crs: QgsCoordinateReferenceSystem = self.iface.mapCanvas().mapSettings().destinationCrs()
            if canvas_crs.isGeographic():
                self.iface.messageBar().pushMessage(
                    "Error",
                    "Please use a projected Coordinate Reference System, e.g. EPSG 27700 for Britain.",
                    level=Qgis.Critical,
                )
                return
            # logic is already handled in the GUI set_layer method
            self.dlg.selected_layer.setCrs(canvas_crs)
            print(self.dlg.selected_layer.getFeatures())
            selected_feature: QgsFeature = self.dlg.selected_layer.getFeatures()[0]
            feature_geom: QgsGeometry = selected_feature.geometry()
            geom: geometry.Polygon = wkt.loads(feature_geom.asWkt())
            geom = orient(geom, -1)  # orient per QGS
            bounds: tuple[float, float, float, float] = geom.bounds
            width: float = int(abs(bounds[3] - bounds[1]))
            height: float = int(abs(bounds[2] - bounds[0]))
            base_raster: QgsRasterLayer = QgsRasterBlock(Qgis.Byte, width, height)
            print(base_raster)
            base_layer = QgsProject.instance().addMapLayer(base_raster, addToLegend=False)
            # add raster layer group
            layer_root = QgsProject.instance().layerTreeRoot()
            layer_group = layer_root.addGroup("Simulation Outputs")
            layer_group.addLayer(base_layer)
        else:
            QgsMessageLog.logMessage("no result")
