"""Integration tests for the full Raven preprocessing pipeline.

Each parametrized case runs ``create_input_files.main()`` end-to-end
(without actually executing Raven) and asserts the expected output files exist.

Tests marked ``@pytest.mark.raven`` additionally run the Raven executable and
check for valid output; they require the binary at the path given in the
namelist and are intentionally slow.

Run only fast tests:
    pytest tests/integration/ -m "not raven"

Run Raven end-to-end tests:
    pytest tests/integration/ -m raven
"""

import sys
import subprocess
from pathlib import Path

import pytest
import yaml


# ── Helpers ────────────────────────────────────────────────────────────────────

def _model_dir(nml_path: Path) -> Path:
    with open(nml_path) as f:
        nml = yaml.safe_load(f)
    return Path(nml["main_dir"]) / nml["config_dir"]


def _catchment_dir(nml_path: Path) -> Path:
    with open(nml_path) as f:
        nml = yaml.safe_load(f)
    return _model_dir(nml_path) / f"catchment_{nml['gauge_id']}"


def _hbv_dir(nml_path: Path) -> Path:
    with open(nml_path) as f:
        nml = yaml.safe_load(f)
    return _catchment_dir(nml_path) / nml["model_type"].upper()


def _assert_model_files_exist(nml_path: Path) -> None:
    """Check that all Raven .rvx input files were created."""
    with open(nml_path) as f:
        nml = yaml.safe_load(f)
    gauge_id = nml["gauge_id"]
    model_type = nml["model_type"].upper()
    model_dir = _hbv_dir(nml_path)

    for ext in (".rvh", ".rvt", ".rvp", ".rvi", ".rvc"):
        path = model_dir / f"{gauge_id}_{model_type}{ext}"
        assert path.exists(), f"Missing model file: {path}"


def _assert_data_obs_files_exist(nml_path: Path, meteo_source: str,
                                  coupled: bool = False) -> None:
    """Check that data_obs directory contains expected forcing files."""
    data_obs = _catchment_dir(nml_path) / "data_obs"
    assert data_obs.is_dir(), f"data_obs directory missing: {data_obs}"

    # Streamflow RVT
    with open(nml_path) as f:
        nml = yaml.safe_load(f)
    gauge_id = nml["gauge_id"]

    if nml.get("subbasins"):
        for sb in nml["subbasins"]:
            if sb.get("gauged", 0):
                assert (data_obs / f"Q_daily_{sb['gauge_id']}.rvt").exists()
    else:
        assert (data_obs / "Q_daily.rvt").exists()

    # Grid weights
    if meteo_source.upper() == "HAR":
        assert (data_obs / "GridWeights_HAR.txt").exists()
    else:
        assert (data_obs / "GridWeights.txt").exists()

    # GloGEM outputs
    if coupled:
        assert (data_obs / "irrigation.nc").exists()
        assert (data_obs / "GridWeights_Irrigation.txt").exists()


# ── Parametrized pipeline test ─────────────────────────────────────────────────

from tests.conftest import PIPELINE_CASES


@pytest.mark.parametrize(
    "gauge_id, model_type, meteo_source, coupled, multi_subbasin",
    PIPELINE_CASES,
)
def test_pipeline_produces_model_files(
    gauge_id, model_type, meteo_source, coupled, multi_subbasin,
    make_namelist, fixture_root
):
    """Full pipeline (no Raven execution): verify all output files are created."""
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
    from create_input_files import main as run_pipeline

    nml_path = make_namelist(
        gauge_id=gauge_id,
        model_type=model_type,
        meteo_source=meteo_source,
        coupled=coupled,
        multi_subbasin=multi_subbasin,
    )

    # If coupled, we need glacier_id_mapping.csv pre-created by catchment processor.
    # The pipeline creates it in STEP 3; run_pipeline handles this.
    run_pipeline(str(nml_path), force_reprocess=True)

    _assert_model_files_exist(nml_path)
    _assert_data_obs_files_exist(nml_path, meteo_source, coupled)


# ── Raven end-to-end tests (slow) ─────────────────────────────────────────────

RAVEN_CASES = [
    pytest.param("9999", "HBV", "ERA5", False, False, id="hbv_era5_single"),
    pytest.param("9999", "HBV", "HAR",  False, False, id="hbv_har_single"),
    pytest.param("9900", "HBV", "HAR",  False, True,  id="hbv_har_multi"),
    pytest.param("9999", "HBV", "HAR",  True,  False, id="hbv_har_coupled"),
]


@pytest.mark.raven
@pytest.mark.slow
@pytest.mark.parametrize(
    "gauge_id, model_type, meteo_source, coupled, multi_subbasin",
    RAVEN_CASES,
)
def test_raven_runs_without_error(
    gauge_id, model_type, meteo_source, coupled, multi_subbasin,
    make_namelist, fixture_root
):
    """Run Raven executable and assert it completes without fatal errors."""
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
    from create_input_files import main as run_pipeline

    nml_path = make_namelist(
        gauge_id=gauge_id,
        model_type=model_type,
        meteo_source=meteo_source,
        coupled=coupled,
        multi_subbasin=multi_subbasin,
    )
    run_pipeline(str(nml_path), force_reprocess=True)

    with open(nml_path) as f:
        nml = yaml.safe_load(f)

    raven_exe = nml.get("raven_executable", "")
    if not Path(raven_exe).exists():
        pytest.skip(f"Raven executable not found: {raven_exe}")

    model_type_upper = nml["model_type"].upper()
    gauge_id = nml["gauge_id"]
    hbv_dir = _hbv_dir(nml_path)
    rvi_path = hbv_dir / f"{gauge_id}_{model_type_upper}.rvi"

    log_path = hbv_dir / "raven_run.log"
    result = subprocess.run(
        [raven_exe, str(rvi_path.stem), "-o", str(hbv_dir)],
        capture_output=True, text=True, cwd=str(hbv_dir), timeout=300,
    )
    log_path.write_text(result.stdout + "\n" + result.stderr)

    assert result.returncode == 0, (
        f"Raven exited with code {result.returncode}.\n"
        f"Log:\n{result.stdout[-3000:]}"
    )
    # Raven writes errors to stderr; check for fatal keywords
    fatal_keywords = ["FATAL", "ERROR:", "ParseMainInputFile"]
    for kw in fatal_keywords:
        assert kw not in result.stdout, (
            f"Raven output contains '{kw}'.\nLog:\n{result.stdout[-3000:]}"
        )


# ── Additional structural checks ───────────────────────────────────────────────

@pytest.mark.parametrize("model_type,meteo_source,prefix", [
    ("HBV", "HAR", "har_"),
])
def test_multi_subbasin_per_subbasin_csvs(model_type, meteo_source, prefix, make_namelist):
    """Per-subbasin monthly T/PET CSVs must be created for all subbasins."""
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
    from create_input_files import main as run_pipeline

    nml_path = make_namelist(gauge_id="9900", model_type=model_type,
                             meteo_source=meteo_source, multi_subbasin=True)
    run_pipeline(str(nml_path), force_reprocess=True)

    with open(nml_path) as f:
        nml = yaml.safe_load(f)

    data_obs = _catchment_dir(nml_path) / "data_obs"
    for sb in nml["subbasins"]:
        sb_dir = data_obs / f"subbasin_{sb['id']}"
        assert (sb_dir / f"{prefix}monthly_temperature_averages.csv").exists(), (
            f"Missing {prefix}monthly_temperature_averages.csv for subbasin {sb['id']}"
        )
        assert (sb_dir / f"{prefix}monthly_pet_averages.csv").exists(), (
            f"Missing {prefix}monthly_pet_averages.csv for subbasin {sb['id']}"
        )


