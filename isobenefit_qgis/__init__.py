"""This script initializes the plugin, making it known to QGIS.

Kept import-safe (no top-level QGIS import) so pure submodules such as ``grid``
can be imported in a plain venv for testing.
"""


def classFactory(iface):  # noqa: N802  (QGIS-mandated name)
    """Entry point used by QGIS to instantiate the plugin."""
    from .futurb import Isobenefit

    return Isobenefit(iface)
