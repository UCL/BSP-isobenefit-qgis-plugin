""" """
from __future__ import annotations

import os.path
from pathlib import Path
from typing import Any, Callable

from qgis.core import Qgis, QgsCoordinateTransform, QgsFeature, QgsMessageLog, QgsProject, QgsVectorLayer
from qgis.gui import QgisInterface
from qgis.PyQt.QtCore import QCoreApplication, QSettings, QTranslator, qVersion
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QToolBar, QWidget
from shapely import geometry, wkt
from shapely.geometry.polygon import orient

from .futurb_dialog import FuturbDialog  # Import the code for the dialog
from .mesh import create_mesh_layer
from .resources import *  # Initialize Qt resources from file resources.py


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
            # prepare extents layer
            extents_layer = QgsVectorLayer(
                f"Polygon?crs={self.dlg.selected_crs.authid()}&field=id:integer&index=yes",
                "sim_input_extents",
                "memory",
            )
            #
            src_feature = self.dlg.selected_feature
            src_geom = src_feature.geometry()
            print(src_geom)
            target_feature = QgsFeature(id=1)
            crs_transform = QgsCoordinateTransform(
                self.dlg.selected_layer.crs(), self.dlg.selected_crs, QgsProject.instance()
            )
            src_geom.transform(crs_transform)
            target_feature.setGeometry(src_geom)
            extents_layer.dataProvider().addFeature(target_feature)
            extents_layer.dataProvider().updateExtents()
            QgsProject.instance().addMapLayer(extents_layer)
            print(extents_layer.extent())
            extents_layer.extent().xMinimum
            """
            # TODO:
            - explore temporal layers to understand outputs
            - create temporal layer
            - adapt simulation to work with mesh / temporal format logic
            - run simulation against crude (rectangular) extents
            - save to temporal layer
            """
            # None checking is handled by dialogue
            granularity_m: int = int(self.dlg.grid_size_m.text())  # type: ignore
            file_name = str(self.dlg.mesh_file_name)
            if not "." in file_name:
                file_name += ".nc"
            output_path: Path = self.dlg.mesh_dir / file_name  # type: ignore
            mesh_layer = create_mesh_layer(output_path, extents_layer, self.dlg.selected_crs, granularity_m)
            QgsProject.instance().addMapLayer(mesh_layer)
            # iterate and update
            # editor.changeZValues()

            # geom: geometry.Polygon = wkt.loads(feature_geom.asWkt())
            # geom = orient(geom, -1)  # orient per QGS
            # bounds: tuple[float, float, float, float] = geom.bounds
            # width: float = int(abs(bounds[3] - bounds[1]))
            # height: float = int(abs(bounds[2] - bounds[0]))
            # base_raster: QgsRasterLayer = QgsRasterBlock(Qgis.Byte, width, height)
            # print(base_raster)
            # base_layer = QgsProject.instance().addMapLayer(base_raster, addToLegend=False)
            # # add raster layer group
            # layer_root = QgsProject.instance().layerTreeRoot()
            # layer_group = layer_root.addGroup("Simulation Outputs")
            # layer_group.addLayer(base_layer)
        else:
            print("no result")
            QgsMessageLog.logMessage("no result", level=Qgis.Warning, notifyUser=True)
