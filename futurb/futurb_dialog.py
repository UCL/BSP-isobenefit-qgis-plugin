# -*- coding: utf-8 -*-
"""
/***************************************************************************
 FuturbDialog
                                 A QGIS plugin
 Description
                             -------------------
        copyright            : (C) 2022 by Gareth Simons
        email                : garethsimons@me.com
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""

import logging
import os

from qgis.PyQt import QtWidgets, uic
from qgis.PyQt.QtWidgets import QDialogButtonBox, QLabel, QLineEdit, QListWidget, QPushButton, QSlider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FORM_CLASS, _ = uic.loadUiType(os.path.join(os.path.dirname(__file__), "futurb_dialog_base.ui"))
print(FORM_CLASS)


class FuturbDialog(QtWidgets.QDialog, FORM_CLASS):

    button_box: QDialogButtonBox
    n_iterations: QLineEdit
    isobenefit: QPushButton
    classical: QPushButton
    x_size: QLineEdit
    y_size: QLineEdit
    build_prob: QSlider
    build_prob_val: QLabel
    new_cent_p1: QLineEdit
    new_cent_p2: QLineEdit
    t_star: QLineEdit
    random_seed: QLineEdit
    layers_list: QListWidget

    def __init__(self, parent=None):
        """Constructor."""
        super(FuturbDialog, self).__init__(parent)
        # Set up the user interface from Designer.
        # After setupUI you can access any designer object by doing
        # self.<objectname>, and you can use autoconnect slots - see
        # http://qt-project.org/doc/qt-4.8/designer-using-a-ui-file.html
        # #widgets-and-dialogs-with-auto-connect
        self.setupUi(self)
        # prime
        self.widget.addWidget
        self.handle_build_prob()
        # handle changes
        self.build_prob.valueChanged.connect(self.handle_build_prob)

    def handle_build_prob(self) -> None:
        """ """
        new_val: int = self.build_prob.value()
        dec_val: float = new_val / 100
        self.build_prob_val.setText(str(dec_val))
