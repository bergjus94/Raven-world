"""pytest configuration and shared fixtures for Raven preprocessing pipeline tests.

Session-scoped fixtures generate synthetic test data once; function-scoped
fixtures create an isolated output directory and namelist for each test.
"""

# ── Force non-interactive matplotlib backend BEFORE any src imports ────────────
import matplotlib
matplotlib.use("Agg")   # prevents figure windows from popping up during tests

import sys
from pathlib import Path

import pytest
import yaml

# Add src/ to sys.path so tests can import preprocessing modules
SRC_DIR = Path(__file__).parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Path to the repo's default_params.yaml (needed by model processors)
PARAMS_DIR = SRC_DIR / "config" / "default_params.yaml"

# ── Custom markers ─────────────────────────────────────────────────────────────

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "raven: marks tests that require a Raven executable (deselect with -m 'not raven')",
    )
    config.addinivalue_line(
        "markers",
        "slow: marks tests that take a long time to run",
    )


# ── Session-scoped fixture root ────────────────────────────────────────────────

@pytest.fixture(scope="session")
def fixture_root(tmp_path_factory):
    """Generate all synthetic test data once per test session.

    Returns the root directory that acts as ``main_dir`` in namelists.
    """
    from tests.fixtures.generate_fixtures import generate_all_fixtures

    root = tmp_path_factory.mktemp("raven_fixtures")
    generate_all_fixtures(root)
    return root


# ── Namelist factory ───────────────────────────────────────────────────────────

def _build_namelist(
    *,
    fixture_root: Path,
    output_dir: Path,
    gauge_id: str,
    model_type: str,
    meteo_source: str,
    coupled: bool = False,
    multi_subbasin: bool = False,
    dem_file: str = "dem_test.tif",
    landuse_file: str = "test_landuse.tif",
) -> dict:
    """Build a namelist dict for one test combination."""

    nml = {
        "gauge_id": gauge_id,
        "start_date": "2000-01-01",
        "end_date": "2001-12-31",
        "warm_up_date": "1999-01-01",
        "cali_end_date": "2000-12-31",
        "model_type": model_type,
        "template": True,
        "main_dir": str(fixture_root),
        "config_dir": str(output_dir.relative_to(fixture_root))
                      if output_dir.is_relative_to(fixture_root)
                      else str(output_dir),
        "coupled": coupled,
        "debug": False,
        "meteo_source": meteo_source,
        "landuse_source": "ICIMOD",
        "criteria": ["elevation", "landuse"],
        "aspect_slope_threshold": 15.0,
        "future": False,
        "nconnect": "multiple",
        # Paths relative to main_dir
        "shape_dir": "01_data/topo/catchment_shapefile/catchment_shape_{gauge_id}.shp",
        "raster_dir": f"01_data/topo/catchment_dem/{dem_file}",
        "landuse_dir": f"01_data/topo/catchment_landuse/{landuse_file}",
        "stream_dir": "01_data/streamflow/streamflow_{gauge_id}.csv",
        "glacier_dir": "01_data/topo/glacier_outlines/rgi_test_glaciers.shp",
        "glogem_dir": "01_data/GloGEM/Test/GMB",
        "meteo_dir": "01_data/meteo/ERA5",
        "meteo_har_dir": "01_data/meteo/HAR",
        "params_dir": str(PARAMS_DIR),
        "swe_dir": "",          # not used in tests
        "raven_executable": "/home/jberg/RavenHydroFramework/build/Raven",
        # GloGEM settings (only used when coupled=True)
        "basin_name": "Test",
        "glogem_scenario": "ssp126",
        # Meteo forcing file lists (filled by HBV/HYMOD processor from data_obs)
        "forcing_files": [],
        "forcing_files_har": [],
        # Calibration block (minimal)
        "calibration": {
            "metrics": {"primary": "KGE", "secondary": ["NSE"]},
            "iterations": 1,
            "ngs": 1,
            "cali_end_date": "2000-12-31",
            "vali_end_date": "2001-12-31",
        },
    }

    if multi_subbasin:
        nml["subbasins"] = [
            {"id": 1, "gauge_id": "9900", "name": "TestMain",
             "gauged": 1, "reach_length": "_AUTO", "profile": "NONE"},
            {"id": 2, "gauge_id": "9901", "name": "TestSub1",
             "gauged": 0, "reach_length": "_AUTO", "profile": "NONE"},
            {"id": 3, "gauge_id": "9902", "name": "TestSub2",
             "gauged": 0, "reach_length": "_AUTO", "profile": "NONE"},
        ]

    return nml


@pytest.fixture()
def make_namelist(fixture_root, tmp_path):
    """Factory fixture: returns a callable that writes a namelist YAML.

    Usage in tests::

        def test_foo(make_namelist):
            nml_path = make_namelist(gauge_id='9999', model_type='HBV',
                                     meteo_source='ERA5')
    """
    created = []

    def _factory(
        gauge_id: str = "9999",
        model_type: str = "HBV",
        meteo_source: str = "ERA5",
        coupled: bool = False,
        multi_subbasin: bool = False,
        extra: dict | None = None,
    ) -> Path:
        tag = f"{model_type}_{meteo_source}{'_multi' if multi_subbasin else ''}{'_coupled' if coupled else ''}"
        output_dir = tmp_path / f"outputs_{tag}"
        output_dir.mkdir(parents=True, exist_ok=True)

        nml = _build_namelist(
            fixture_root=fixture_root,
            output_dir=output_dir,
            gauge_id=gauge_id,
            model_type=model_type,
            meteo_source=meteo_source,
            coupled=coupled,
            multi_subbasin=multi_subbasin,
        )
        if extra:
            nml.update(extra)

        nml_path = tmp_path / f"namelist_{tag}.yaml"
        with open(nml_path, "w") as f:
            yaml.dump(nml, f, default_flow_style=False, allow_unicode=True)
        created.append(nml_path)
        return nml_path

    return _factory


# ── Parametrize helpers ────────────────────────────────────────────────────────

#: Test matrix: (gauge_id, model_type, meteo_source, coupled, multi_subbasin)
#: Each entry is a pytest.param so you can mark individual combos.
PIPELINE_CASES = [
    pytest.param("9999", "HBV",   "ERA5", False, False, id="hbv_era5_single"),
    pytest.param("9999", "HBV",   "HAR",  False, False, id="hbv_har_single"),
    pytest.param("9999", "HYMOD", "ERA5", False, False, id="hymod_era5_single"),
    pytest.param("9999", "HYMOD", "HAR",  False, False, id="hymod_har_single"),
    pytest.param("9900", "HBV",   "ERA5", False, True,  id="hbv_era5_multi"),
    pytest.param("9900", "HBV",   "HAR",  False, True,  id="hbv_har_multi"),
    pytest.param("9900", "HYMOD", "ERA5", False, True,  id="hymod_era5_multi",
                 marks=pytest.mark.xfail(reason="HYMOD multi-subbasin not yet implemented",
                                         strict=True)),
    pytest.param("9900", "HYMOD", "HAR",  False, True,  id="hymod_har_multi",
                 marks=pytest.mark.xfail(reason="HYMOD multi-subbasin not yet implemented",
                                         strict=True)),
    pytest.param("9999", "HBV",   "HAR",  True,  False, id="hbv_har_coupled"),
]
