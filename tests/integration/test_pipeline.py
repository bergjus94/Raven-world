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

def test_multi_subbasin_has_per_subbasin_monthly_csvs(make_namelist):
    """Per-subbasin monthly T/PET CSVs must be created for all subbasins."""
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
    from create_input_files import main as run_pipeline

    nml_path = make_namelist(gauge_id="9900", model_type="HBV",
                             meteo_source="HAR", multi_subbasin=True)
    run_pipeline(str(nml_path), force_reprocess=True)

    with open(nml_path) as f:
        nml = yaml.safe_load(f)

    data_obs = _catchment_dir(nml_path) / "data_obs"
    for sb in nml["subbasins"]:
        sb_dir = data_obs / f"subbasin_{sb['id']}"
        for fname in ("har_monthly_temperature_averages.csv",
                      "har_monthly_pet_averages.csv"):
            assert (sb_dir / fname).exists(), (
                f"Missing {fname} for subbasin {sb['id']}"
            )


def test_era5_single_data_obs_paths_use_relative(make_namelist):
    """Model .rvt file must reference ../data_obs/ not data_obs/."""
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
    from create_input_files import main as run_pipeline

    nml_path = make_namelist(gauge_id="9999", model_type="HBV",
                             meteo_source="ERA5")
    run_pipeline(str(nml_path), force_reprocess=True)

    with open(nml_path) as f:
        nml = yaml.safe_load(f)
    gauge_id = nml["gauge_id"]
    model_type = nml["model_type"].upper()
    rvt_path = _hbv_dir(nml_path) / f"{gauge_id}_{model_type}.rvt"
    content = rvt_path.read_text()

    # data_obs references must use parent path
    for line in content.splitlines():
        for kw in (":FileNameNC", ":RedirectToFile"):
            if kw in line and "data_obs" in line:
                assert "../data_obs" in line, (
                    f"Path in {rvt_path.name} must be ../data_obs, got: {line.strip()}"
                )


def test_hymod_pipeline_produces_model_files(make_namelist):
    """HYMOD processor must produce all .rvx files (sanity check)."""
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
    from create_input_files import main as run_pipeline

    nml_path = make_namelist(gauge_id="9999", model_type="HYMOD",
                             meteo_source="ERA5")
    run_pipeline(str(nml_path), force_reprocess=True)
    _assert_model_files_exist(nml_path)
