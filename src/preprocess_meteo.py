#### Backward-compatible re-export shim.
####
#### preprocess_meteo.py was split into:
####   preprocess_meteo_base.py        – MeteoBase, normalize_coords
####   preprocess_meteo_era5.py        – ERA5LandAnalyzer, GridWeightsGenerator
####   preprocess_meteo_har.py         – HARAnalyzer, HARGridWeightsGenerator
####   preprocess_meteo_tphipr.py      – TPHiPrAnalyzer, TPHiPrGridWeightsGenerator,
####                                     process_tphipr_precipitation
####   preprocess_meteo_meteoswiss.py  – MeteoSwissAnalyzer, MeteoSwissGridWeightsGenerator,
####                                     process_meteoswiss_forcing
####
#### All existing callers (create_input_files.py, tests, …) import from
#### preprocess_meteo and continue to work unchanged.

from preprocess_meteo_base import MeteoBase, normalize_coords
from preprocess_meteo_era5 import ERA5LandAnalyzer, GridWeightsGenerator
from preprocess_meteo_har import HARAnalyzer, HARGridWeightsGenerator
from preprocess_meteo_tphipr import (
    TPHiPrAnalyzer,
    TPHiPrGridWeightsGenerator,
    process_tphipr_precipitation,
)
from preprocess_meteo_meteoswiss import (
    MeteoSwissAnalyzer,
    MeteoSwissGridWeightsGenerator,
    process_meteoswiss_forcing,
)

__all__ = [
    'MeteoBase', 'normalize_coords',
    'ERA5LandAnalyzer', 'GridWeightsGenerator',
    'HARAnalyzer', 'HARGridWeightsGenerator',
    'TPHiPrAnalyzer', 'TPHiPrGridWeightsGenerator', 'process_tphipr_precipitation',
    'MeteoSwissAnalyzer', 'MeteoSwissGridWeightsGenerator', 'process_meteoswiss_forcing',
]
