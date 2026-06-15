"""Isobenefit Urbanism simulation core.

A pure compute engine (Rust extension) with no GIS dependencies: it takes numpy
arrays plus scalar parameters and returns numpy arrays / snapshots. All GIS IO
(reading layers, reprojection, rasterization, writing rasters) lives in the QGIS
plugin, not here.
"""

from ._core import (
    Simulation,
    ensemble_class_counts,
    ensemble_probability,
    run_ensemble,
    __version__,
)

__all__ = [
    "Simulation",
    "ensemble_class_counts",
    "ensemble_probability",
    "run_ensemble",
    "__version__",
]
