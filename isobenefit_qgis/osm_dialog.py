"""Dialog for the OpenStreetMap extraction step.

The user defines an area of interest as a **polygon** — either drawn on the map or
taken from an existing polygon layer — then chooses which datasets to download and an
output GeoPackage. The polygon's bounding box drives the Overpass query (Overpass is
bbox-based); the downloaded features are then trimmed to the polygon itself
(``osm_fetcher``). The dialog is modeless so the map canvas stays interactive while
drawing. Downloading is deliberately separate from simulating: layers land on disk so
they can be edited or swapped before a run.
"""

from __future__ import annotations

from pathlib import Path

from qgis.core import (
    Qgis,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsGeometry,
    QgsPointXY,
    QgsProject,
    QgsWkbTypes,
)
from qgis.gui import QgsFileWidget, QgsMapLayerComboBox, QgsMapTool, QgsRubberBand
from qgis.PyQt import QtWidgets
from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QColor

from . import osm_queries

try:
    _POLYGON_FILTER = Qgis.LayerFilter.PolygonLayer
except Exception:  # pragma: no cover - fallback for older QGIS enum location
    from qgis.core import QgsMapLayerProxyModel

    _POLYGON_FILTER = QgsMapLayerProxyModel.Filter.PolygonLayers

WGS84 = QgsCoordinateReferenceSystem("EPSG:4326")


class PolygonMapTool(QgsMapTool):
    """A simple polygon-capture map tool: left-click adds vertices, right-click finishes.

    Emits ``completed`` with the polygon (in the canvas CRS) or ``cancelled`` (Esc /
    too few vertices). A rubber band previews the shape, including a live edge to the
    cursor.
    """

    completed = pyqtSignal(QgsGeometry)
    cancelled = pyqtSignal()

    def __init__(self, canvas):
        super().__init__(canvas)
        self.canvas = canvas
        self.points: list[QgsPointXY] = []
        self.rubber = QgsRubberBand(canvas, QgsWkbTypes.GeometryType.PolygonGeometry)
        self.rubber.setColor(QColor(220, 30, 30, 60))
        self.rubber.setStrokeColor(QColor(220, 30, 30, 200))
        self.rubber.setWidth(2)

    def canvasReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.points.append(self.toMapCoordinates(event.pos()))
            self._draw()
        elif event.button() == Qt.MouseButton.RightButton:
            self._finish()

    def canvasMoveEvent(self, event):
        if self.points:
            self._draw(self.toMapCoordinates(event.pos()))

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self._cleanup()
            self.cancelled.emit()

    def _draw(self, temp: QgsPointXY | None = None):
        self.rubber.reset(QgsWkbTypes.GeometryType.PolygonGeometry)
        pts = self.points + ([temp] if temp is not None else [])
        for i, p in enumerate(pts):
            self.rubber.addPoint(p, i == len(pts) - 1)

    def _finish(self):
        if len(self.points) >= 3:
            geom = QgsGeometry.fromPolygonXY([list(self.points)])
            self._cleanup()
            self.completed.emit(geom)
        else:
            self._cleanup()
            self.cancelled.emit()

    def _cleanup(self):
        self.rubber.reset(QgsWkbTypes.GeometryType.PolygonGeometry)
        self.points = []

    def deactivate(self):
        self._cleanup()
        super().deactivate()


class OsmDialog(QtWidgets.QDialog):
    """Collect AOI polygon, dataset selection and output path for an OSM download."""

    def __init__(self, iface, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.iface = iface
        self.dataset_checks: dict[str, QtWidgets.QCheckBox] = {}
        self._aoi_geom: QgsGeometry | None = None  # always stored in EPSG:4326
        self._draw_tool: PolygonMapTool | None = None
        self._prev_tool = None
        self.setupUi()

    def setupUi(self) -> None:
        self.setObjectName("OsmDialog")
        self.setWindowTitle("Extract from OpenStreetMap")
        self.resize(500, 520)
        layout = QtWidgets.QVBoxLayout(self)

        intro = QtWidgets.QLabel(
            "Download urban fabric, green space, centres, streets and transport stops "
            "from OpenStreetMap. Define the area as a polygon (drawn or from a layer); "
            "results are trimmed to it. Layers are saved to a GeoPackage and added to "
            "the project — edit or swap them, then run the simulation separately.",
            self,
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        # --- Area of interest (a polygon) ------------------------------------
        layout.addWidget(self._heading("Area of interest"))
        self.draw_button = QtWidgets.QPushButton("Draw area on map…", self)
        self.draw_button.clicked.connect(self._start_draw)
        layout.addWidget(self.draw_button)

        layout.addWidget(QtWidgets.QLabel("…or use an existing polygon layer:", self))
        self.aoi_layer_box = QgsMapLayerComboBox(self)
        self.aoi_layer_box.setAllowEmptyLayer(True, "— none —")
        self.aoi_layer_box.setFilters(_POLYGON_FILTER)
        self.aoi_layer_box.setShowCrs(True)
        self.aoi_layer_box.setCurrentIndex(0)  # type: ignore[arg-type]
        self.aoi_layer_box.layerChanged.connect(self._on_layer_chosen)  # type: ignore[attr-defined]
        layout.addWidget(self.aoi_layer_box)

        self.aoi_feedback = QtWidgets.QLabel("No area defined yet — draw one or pick a polygon layer.", self)
        self.aoi_feedback.setWordWrap(True)
        layout.addWidget(self.aoi_feedback)

        # --- Datasets --------------------------------------------------------
        layout.addWidget(self._heading("Datasets"))
        grid = QtWidgets.QGridLayout()
        layout.addLayout(grid)
        for i, key in enumerate(osm_queries.DATASET_ORDER):
            check = QtWidgets.QCheckBox(osm_queries.DATASETS[key]["label"], self)
            check.setChecked(True)
            check.toggled.connect(self._update_state)
            self.dataset_checks[key] = check
            grid.addWidget(check, i // 2, i % 2)

        # --- Output ----------------------------------------------------------
        layout.addWidget(self._heading("Output GeoPackage"))
        self.output_widget = QgsFileWidget(self)
        self.output_widget.setStorageMode(QgsFileWidget.StorageMode.SaveFile)
        self.output_widget.setFilter("GeoPackage (*.gpkg)")
        self.output_widget.fileChanged.connect(self._update_state)  # type: ignore[attr-defined]
        layout.addWidget(self.output_widget)
        self.output_feedback = QtWidgets.QLabel("Choose where to save the downloaded layers.", self)
        self.output_feedback.setWordWrap(True)
        layout.addWidget(self.output_feedback)

        layout.addStretch(1)

        self.button_box = QtWidgets.QDialogButtonBox(self)
        self.button_box.setStandardButtons(
            QtWidgets.QDialogButtonBox.StandardButton.Cancel | QtWidgets.QDialogButtonBox.StandardButton.Ok
        )
        self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText("Fetch")
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        self._update_state()

    def _heading(self, text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text, self)
        font = label.font()
        font.setBold(True)
        label.setFont(font)
        return label

    # --- Drawing -------------------------------------------------------------
    def _start_draw(self) -> None:
        canvas = self.iface.mapCanvas()
        self._prev_tool = canvas.mapTool()
        self._draw_tool = PolygonMapTool(canvas)
        self._draw_tool.completed.connect(self._on_drawn)
        self._draw_tool.cancelled.connect(self._restore_after_draw)
        canvas.setMapTool(self._draw_tool)
        self.hide()  # modeless: hiding returns interaction to the canvas
        self.iface.messageBar().pushMessage(
            "Isobenefit",
            "Draw the area: left-click to add corners, right-click to finish, Esc to cancel.",
            level=Qgis.MessageLevel.Info,
            duration=0,
        )

    def _on_drawn(self, geom: QgsGeometry) -> None:
        src_crs = self.iface.mapCanvas().mapSettings().destinationCrs()
        g = QgsGeometry(geom)
        if src_crs != WGS84:
            g.transform(QgsCoordinateTransform(src_crs, WGS84, QgsProject.instance()))
        self.aoi_layer_box.setCurrentIndex(0)  # a fresh drawing supersedes any chosen layer
        self._set_aoi(g, "drawn polygon")
        self._restore_after_draw()

    def _restore_after_draw(self) -> None:
        canvas = self.iface.mapCanvas()
        self.iface.messageBar().clearWidgets()
        if self._prev_tool is not None:
            canvas.setMapTool(self._prev_tool)
        self._draw_tool = None
        self.show()
        self.raise_()
        self.activateWindow()

    def _on_layer_chosen(self) -> None:
        layer = self.aoi_layer_box.currentLayer()
        if layer is None:
            return
        geoms = [f.geometry() for f in layer.getFeatures() if f.hasGeometry()]
        if not geoms:
            self._set_aoi(None, "")
            self.aoi_feedback.setText("The chosen layer has no geometry.")
            return
        g = QgsGeometry.unaryUnion(geoms)
        if layer.crs() != WGS84:
            g.transform(QgsCoordinateTransform(layer.crs(), WGS84, QgsProject.instance()))
        self._set_aoi(g, f"layer “{layer.name()}”")

    def _set_aoi(self, geom: QgsGeometry | None, source: str) -> None:
        self._aoi_geom = geom if (geom is not None and not geom.isEmpty()) else None
        if self._aoi_geom is None:
            self.aoi_feedback.setText("No area defined yet — draw one or pick a polygon layer.")
        else:
            b = self._aoi_geom.boundingBox()
            self.aoi_feedback.setText(
                f"Area: {source} — bbox (EPSG:4326) W {b.xMinimum():.4f}, S {b.yMinimum():.4f}, "
                f"E {b.xMaximum():.4f}, N {b.yMaximum():.4f}"
            )
        self._update_state()

    # --- Accessors for the caller -------------------------------------------
    def aoi_bbox_4326(self) -> tuple[float, float, float, float] | None:
        if self._aoi_geom is None:
            return None
        b = self._aoi_geom.boundingBox()
        return (b.xMinimum(), b.yMinimum(), b.xMaximum(), b.yMaximum())

    def aoi_polygon_wkt_4326(self) -> str | None:
        return None if self._aoi_geom is None else self._aoi_geom.asWkt()

    def selected_datasets(self) -> list[str]:
        return [key for key, check in self.dataset_checks.items() if check.isChecked()]

    def output_path(self) -> str | None:
        text = self.output_widget.filePath().strip()
        if not text:
            return None
        if not text.lower().endswith(".gpkg"):
            text += ".gpkg"
        return text

    def suggested_group_name(self) -> str:
        bbox = self.aoi_bbox_4326()
        if bbox is None:
            return "OSM"
        xmin, ymin, xmax, ymax = bbox
        return f"OSM — {ymin:.3f},{xmin:.3f} → {ymax:.3f},{xmax:.3f}"

    # --- Validation ----------------------------------------------------------
    def _update_state(self) -> None:
        ok_button = self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        valid = self._aoi_geom is not None and bool(self.selected_datasets())
        out = self.output_path()
        if out is None:
            self.output_feedback.setText("Choose where to save the downloaded layers.")
            valid = False
        elif not Path(out).parent.exists():
            self.output_feedback.setText("The output folder does not exist.")
            valid = False
        else:
            self.output_feedback.setText("")
        ok_button.setEnabled(valid)
