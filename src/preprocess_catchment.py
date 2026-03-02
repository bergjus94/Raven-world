"""Backward-compatible re-export shim.

preprocess_catchment.py was split into:
  preprocess_catchment_hru.py   – CatchmentProcessor, plot_raster, plot_map
  preprocess_connectivity.py    – HRUConnectivityCalculator,
                                   MultiSubbasinConnectivityCalculator
  preprocess_multisubbasin.py   – MultiSubbasinProcessor
"""
from preprocess_catchment_hru import CatchmentProcessor, plot_raster, plot_map
from preprocess_connectivity import (
    HRUConnectivityCalculator,
    MultiSubbasinConnectivityCalculator,
)
from preprocess_multisubbasin import MultiSubbasinProcessor

__all__ = [
    'CatchmentProcessor',
    'HRUConnectivityCalculator',
    'MultiSubbasinConnectivityCalculator',
    'MultiSubbasinProcessor',
    'plot_raster',
    'plot_map',
]
