""" """
from __future__ import annotations

import logging

from qgis.gui import QgsFileWidget
from qgis.PyQt import QtCore, QtWidgets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FuturbDialog(QtWidgets.QDialog):
    """ """

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        """ """
        super(FuturbDialog, self).__init__(parent)
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
        self.grid_size_label = QtWidgets.QLabel("Grid Size", self)
        self.grid.addWidget(self.grid_size_label, 2, 0, alignment=QtCore.Qt.AlignRight)
        self.grid_size = QtWidgets.QLineEdit("1", self)
        self.grid.addWidget(self.grid_size, 2, 1)
        # iterations
        self.n_iterations_label = QtWidgets.QLabel("Iterations", self)
        self.grid.addWidget(self.n_iterations_label, 3, 0, alignment=QtCore.Qt.AlignRight)
        self.n_iterations = QtWidgets.QLineEdit("20", self)
        self.grid.addWidget(self.n_iterations, 3, 1)
        # max population
        self.max_population_label = QtWidgets.QLabel("Max Population", self)
        self.grid.addWidget(self.max_population_label, 4, 0, alignment=QtCore.Qt.AlignRight)
        self.max_population = QtWidgets.QLineEdit("100000", self)
        self.grid.addWidget(self.max_population, 4, 1)
        # max ab / km2
        self.max_ab_km2_label = QtWidgets.QLabel("Max ab/km2", self)
        self.grid.addWidget(self.max_ab_km2_label, 5, 0, alignment=QtCore.Qt.AlignRight)
        self.max_ab_km2 = QtWidgets.QLineEdit("10000", self)
        self.grid.addWidget(self.max_ab_km2, 5, 1)
        # build prob
        self.build_prob_label = QtWidgets.QLabel("Build probability", self)
        self.grid.addWidget(self.build_prob_label, 6, 0, alignment=QtCore.Qt.AlignRight)
        self.build_prob = QtWidgets.QLineEdit("0.3", self)
        self.grid.addWidget(self.build_prob, 6, 1)
        # cent P1
        self.new_cent_p1_label = QtWidgets.QLabel("New cent prob 1", self)
        self.grid.addWidget(self.new_cent_p1_label, 7, 0, alignment=QtCore.Qt.AlignRight)
        self.new_cent_p1 = QtWidgets.QLineEdit("0.1", self)
        self.grid.addWidget(self.new_cent_p1, 7, 1)
        # cent P2
        self.new_cent_p2_label = QtWidgets.QLabel("New cent prob 2", self)
        self.grid.addWidget(self.new_cent_p2_label, 8, 0, alignment=QtCore.Qt.AlignRight)
        self.new_cent_p2 = QtWidgets.QLineEdit("0.0", self)
        self.grid.addWidget(self.new_cent_p2, 8, 1)
        # T star
        self.t_star_label = QtWidgets.QLabel("T*", self)
        self.grid.addWidget(self.t_star_label, 9, 0, alignment=QtCore.Qt.AlignRight)
        self.t_star = QtWidgets.QLineEdit("5", self)
        self.grid.addWidget(self.t_star, 9, 1)
        # random seed
        self.random_seed_label = QtWidgets.QLabel("Random Seed", self)
        self.grid.addWidget(self.random_seed_label, 10, 0, alignment=QtCore.Qt.AlignRight)
        self.random_seed = QtWidgets.QLineEdit("42", self)
        self.grid.addWidget(self.random_seed, 10, 1)
        # layers list
        self.layers_list_label = QtWidgets.QLabel("Input layer", self)
        self.grid.addWidget(self.layers_list_label, 11, 0, 1, 2, alignment=QtCore.Qt.AlignLeft)
        self.layers_list = QtWidgets.QListWidget()
        self.grid.addWidget(self.layers_list, 12, 0, 2, 2)
        # file output
        self.file_output_label = QtWidgets.QLabel("File output path", self)
        self.grid.addWidget(self.file_output_label, 14, 0, 1, 2, alignment=QtCore.Qt.AlignLeft)
        self.file_output = QgsFileWidget(self)
        self.file_output.setStorageMode(QgsFileWidget.SaveFile)
        self.grid.addWidget(self.file_output, 15, 0, 1, 2)
        # Cancel / OK buttons
        self.button_box = QtWidgets.QDialogButtonBox(self)
        self.grid.addWidget(self.button_box, 16, 0, 1, 2)
        self.button_box.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)


if __name__ == "__main__":
    """ """
    Test = FuturbDialog()
    print(Test)
