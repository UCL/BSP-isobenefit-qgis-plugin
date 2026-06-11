"""
This script initializes the plugin, making it known to QGIS.
"""
from qgis.gui import QgisInterface

from .futurb import Isobenefit


def classFactory(iface: QgisInterface):  # pylint: disable=invalid-name
    """ """
    return Isobenefit(iface)
