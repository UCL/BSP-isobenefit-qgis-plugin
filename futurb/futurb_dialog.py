""" """
import logging

from qgis.PyQt import QtCore, QtWidgets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FuturbDialog(QtWidgets.QDialog):

    button_box: QtWidgets.QDialogButtonBox
    n_iterations: QtWidgets.QLineEdit
    isobenefit: QtWidgets.QPushButton
    classical: QtWidgets.QPushButton
    x_size: QtWidgets.QLineEdit
    y_size: QtWidgets.QLineEdit
    build_prob: QtWidgets.QSlider
    build_prob_val: QtWidgets.QLabel
    new_cent_p1: QtWidgets.QLineEdit
    new_cent_p2: QtWidgets.QLineEdit
    t_star: QtWidgets.QLineEdit
    random_seed: QtWidgets.QLineEdit
    layers_list: QtWidgets.QListWidget

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        """Constructor."""
        super(FuturbDialog, self).__init__(parent)
        self.setupUi()
        self.handle_build_prob()
        self.build_prob.valueChanged.connect(self.handle_build_prob)

    def handle_build_prob(self) -> None:
        """ """
        new_val: int = self.build_prob.value()
        dec_val: float = new_val / 100
        self.build_prob_val.setText(str(dec_val))

    def setupUi(self):
        """ """
        self.setObjectName("FuturbDialog")
        self.resize(398, 740)
        self.button_box = QtWidgets.QDialogButtonBox(self)
        self.button_box.setGeometry(QtCore.QRect(20, 690, 341, 32))
        self.button_box.setOrientation(QtCore.Qt.Horizontal)
        self.button_box.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok)
        self.button_box.setObjectName("button_box")
        self.isobenefit = QtWidgets.QPushButton(self)
        self.isobenefit.setGeometry(QtCore.QRect(80, 50, 100, 50))
        self.isobenefit.setObjectName("isobenefit")
        self.classical = QtWidgets.QPushButton(self)
        self.classical.setEnabled(True)
        self.classical.setGeometry(QtCore.QRect(220, 50, 100, 50))
        self.classical.setObjectName("classical")
        self.x_size = QtWidgets.QLineEdit(self)
        self.x_size.setGeometry(QtCore.QRect(110, 150, 80, 40))
        self.x_size.setText("")
        self.x_size.setObjectName("x_size")
        self.y_size = QtWidgets.QLineEdit(self)
        self.y_size.setGeometry(QtCore.QRect(210, 150, 80, 40))
        self.y_size.setText("")
        self.y_size.setObjectName("y_size")
        self.label_model_type = QtWidgets.QLabel(self)
        self.label_model_type.setGeometry(QtCore.QRect(140, 10, 100, 30))
        self.label_model_type.setObjectName("label_model_type")
        self.label_grid_size = QtWidgets.QLabel(self)
        self.label_grid_size.setGeometry(QtCore.QRect(150, 120, 100, 30))
        self.label_grid_size.setObjectName("label_grid_size")
        self.max_population = QtWidgets.QLineEdit(self)
        self.max_population.setGeometry(QtCore.QRect(210, 270, 121, 30))
        self.max_population.setObjectName("max_population")
        self.label_max_pop = QtWidgets.QLabel(self)
        self.label_max_pop.setGeometry(QtCore.QRect(90, 260, 100, 30))
        self.label_max_pop.setObjectName("label_max_pop")
        self.label_max_ab_km2 = QtWidgets.QLabel(self)
        self.label_max_ab_km2.setGeometry(QtCore.QRect(90, 330, 100, 30))
        self.label_max_ab_km2.setObjectName("label_max_ab_km2")
        self.max_ab_km2 = QtWidgets.QLineEdit(self)
        self.max_ab_km2.setGeometry(QtCore.QRect(210, 330, 121, 30))
        self.max_ab_km2.setObjectName("max_ab_km2")
        self.new_cent_p1 = QtWidgets.QLineEdit(self)
        self.new_cent_p1.setGeometry(QtCore.QRect(210, 450, 121, 30))
        self.new_cent_p1.setObjectName("new_cent_p1")
        self.label_new_cent_P1 = QtWidgets.QLabel(self)
        self.label_new_cent_P1.setGeometry(QtCore.QRect(90, 450, 100, 30))
        self.label_new_cent_P1.setObjectName("label_new_cent_P1")
        self.new_cent_p2 = QtWidgets.QLineEdit(self)
        self.new_cent_p2.setGeometry(QtCore.QRect(210, 510, 121, 30))
        self.new_cent_p2.setObjectName("new_cent_p2")
        self.label_new_cent_P2 = QtWidgets.QLabel(self)
        self.label_new_cent_P2.setGeometry(QtCore.QRect(90, 510, 100, 30))
        self.label_new_cent_P2.setObjectName("label_new_cent_P2")
        self.t_star = QtWidgets.QLineEdit(self)
        self.t_star.setGeometry(QtCore.QRect(210, 570, 121, 30))
        self.t_star.setObjectName("t_star")
        self.label_t_star = QtWidgets.QLabel(self)
        self.label_t_star.setGeometry(QtCore.QRect(90, 570, 100, 30))
        self.label_t_star.setObjectName("label_t_star")
        self.random_seed = QtWidgets.QLineEdit(self)
        self.random_seed.setGeometry(QtCore.QRect(210, 630, 121, 30))
        self.random_seed.setObjectName("random_seed")
        self.label_random_seed = QtWidgets.QLabel(self)
        self.label_random_seed.setGeometry(QtCore.QRect(90, 630, 100, 30))
        self.label_random_seed.setObjectName("label_random_seed")
        self.label_iters = QtWidgets.QLabel(self)
        self.label_iters.setGeometry(QtCore.QRect(60, 100, 56, 249))
        self.label_iters.setObjectName("label_iters")
        self.n_iterations = QtWidgets.QLineEdit(self)
        self.n_iterations.setGeometry(QtCore.QRect(200, 210, 319, 21))
        self.n_iterations.setObjectName("n_iterations")
        self.widget = QtWidgets.QWidget(self)
        self.widget.setGeometry(QtCore.QRect(40, 370, 311, 51))
        self.widget.setObjectName("widget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.horizontalLayout.setContentsMargins(5, 5, 5, 5)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.build_prob_label = QtWidgets.QLabel(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.build_prob_label.sizePolicy().hasHeightForWidth())
        self.build_prob_label.setSizePolicy(sizePolicy)
        self.build_prob_label.setObjectName("build_prob_label")
        self.horizontalLayout.addWidget(self.build_prob_label)
        self.build_prob = QtWidgets.QSlider(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.build_prob.sizePolicy().hasHeightForWidth())
        self.build_prob.setSizePolicy(sizePolicy)
        self.build_prob.setMinimumSize(QtCore.QSize(84, 0))
        self.build_prob.setMinimum(5)
        self.build_prob.setMaximum(95)
        self.build_prob.setSingleStep(5)
        self.build_prob.setPageStep(5)
        self.build_prob.setProperty("value", 30)
        self.build_prob.setOrientation(QtCore.Qt.Horizontal)
        self.build_prob.setInvertedControls(False)
        self.build_prob.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.build_prob.setTickInterval(10)
        self.build_prob.setObjectName("build_prob")
        self.horizontalLayout.addWidget(self.build_prob)
        self.build_prob_val = QtWidgets.QLabel(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.build_prob_val.sizePolicy().hasHeightForWidth())
        self.build_prob_val.setSizePolicy(sizePolicy)
        self.build_prob_val.setObjectName("build_prob_val")
        self.horizontalLayout.addWidget(self.build_prob_val)
        self.layers_list = QtWidgets.QListWidget()
        self.layers_list.setGeometry(QtCore.QRect(10, 460, 141, 192))
        self.layers_list.setObjectName("layers_list")

        self.retranslateUi()
        self.button_box.accepted.connect(self.accept)  # type: ignore
        self.button_box.rejected.connect(self.reject)  # type: ignore
        # QtCore.QMetaObject.connectSlotsByName()

    def retranslateUi(self):
        """ """
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("", "Future Urban Growth test."))
        self.isobenefit.setText(_translate("", "IsoBenefit"))
        self.classical.setText(_translate("", "Classical"))
        self.x_size.setPlaceholderText(_translate("", "x"))
        self.y_size.setPlaceholderText(_translate("", "y"))
        self.label_model_type.setText(
            _translate("", '<html><head/><body><p align="center">Model Type</p></body></html>')
        )
        self.label_grid_size.setText(_translate("", '<html><head/><body><p align="center">Grid Size</p></body></html>'))
        self.max_population.setText(_translate("", "100000"))
        self.max_population.setPlaceholderText(_translate("", "max population"))
        self.label_max_pop.setText(
            _translate("", '<html><head/><body><p align="right">Max Population</p></body></html>')
        )
        self.label_max_ab_km2.setText(
            _translate("", '<html><head/><body><p align="right">Max ab/km2</p></body></html>')
        )
        self.max_ab_km2.setText(_translate("", "10000"))
        self.max_ab_km2.setPlaceholderText(_translate("", "max ab/km2"))
        self.new_cent_p1.setText(_translate("", "0.1"))
        self.new_cent_p1.setPlaceholderText(_translate("", "new centrality P2"))
        self.label_new_cent_P1.setText(
            _translate("", '<html><head/><body><p align="right">New Centrality P1</p></body></html>')
        )
        self.new_cent_p2.setText(_translate("", "0"))
        self.new_cent_p2.setPlaceholderText(_translate("", "new centrality P2"))
        self.label_new_cent_P2.setText(
            _translate("", '<html><head/><body><p align="right">New Centrality P2</p></body></html>')
        )
        self.t_star.setText(_translate("", "5"))
        self.t_star.setPlaceholderText(_translate("", "T star"))
        self.label_t_star.setText(_translate("", '<html><head/><body><p align="right">T star</p></body></html>'))
        self.random_seed.setText(_translate("", "42"))
        self.random_seed.setPlaceholderText(_translate("", "random seed"))
        self.label_random_seed.setText(
            _translate("", '<html><head/><body><p align="right">Random Seed</p></body></html>')
        )
        self.label_iters.setText(_translate("", '<html><head/><body><p align="right">Iterations</p></body></html>'))
        self.n_iterations.setText(_translate("", "20"))
        self.n_iterations.setPlaceholderText(_translate("", "iterations"))
        self.build_prob_label.setText(
            _translate("", '<html><head/><body><p align="right">Build Probability</p></body></html>')
        )
        self.build_prob_val.setText(_translate("", '<html><head/><body><p align="center">0.0</p></body></html>'))


if __name__ == "__main__":
    """ """
    Test = FuturbDialog()
    print(Test)
