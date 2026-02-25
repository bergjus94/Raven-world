"""Unit tests for ERA5LandAnalyzer and HARAnalyzer."""

import pytest
import pandas as pd
from pathlib import Path


class TestERA5LandAnalyzer:
    """Tests for preprocess_meteo.ERA5LandAnalyzer."""

    def test_init_without_error(self, make_namelist):
        from preprocess_meteo import ERA5LandAnalyzer

        nml_path = make_namelist(gauge_id="9999", model_type="HBV",
                                 meteo_source="ERA5")
        analyzer = ERA5LandAnalyzer(nml_path, force_reprocess=True)
        assert analyzer.gauge_id == "9999"

    def test_processed_files_exist(self, make_namelist):
        from preprocess_meteo import ERA5LandAnalyzer

        nml_path = make_namelist(gauge_id="9999", model_type="HBV",
                                 meteo_source="ERA5")
        analyzer = ERA5LandAnalyzer(nml_path, force_reprocess=True)

        # At minimum temperature and precipitation must be produced
        expected = [
            "era5_land_temp_mean.nc",
            "era5_land_precip.nc",
        ]
        for fname in expected:
            path = analyzer.output_path / fname
            assert path.exists(), f"Expected processed file: {path}"

    def test_monthly_temperature_averages(self, make_namelist):
        from preprocess_meteo import ERA5LandAnalyzer

        nml_path = make_namelist(gauge_id="9999", model_type="HBV",
                                 meteo_source="ERA5")
        analyzer = ERA5LandAnalyzer(nml_path, force_reprocess=True)
        analyzer.calculate_monthly_temperature_averages()

        csv_path = analyzer.output_path / "monthly_temperature_averages.csv"
        assert csv_path.exists(), f"Expected {csv_path}"

        df = pd.read_csv(csv_path)
        # Should have 12 rows (one per month) or 12 values
        assert len(df) >= 12 or df.shape[1] >= 12, (
            "Monthly temperature CSV should cover all 12 months"
        )

    def test_monthly_pet_averages(self, make_namelist):
        from preprocess_meteo import ERA5LandAnalyzer

        nml_path = make_namelist(gauge_id="9999", model_type="HBV",
                                 meteo_source="ERA5")
        analyzer = ERA5LandAnalyzer(nml_path, force_reprocess=True)
        analyzer.calculate_monthly_pet_averages()

        csv_path = analyzer.output_path / "monthly_pet_averages.csv"
        assert csv_path.exists(), f"Expected {csv_path}"

    def test_grid_weights_file_created(self, make_namelist):
        from preprocess_meteo import ERA5LandAnalyzer, GridWeightsGenerator
        from preprocess_catchment import CatchmentProcessor
        import yaml

        nml_path = make_namelist(gauge_id="9999", model_type="HBV",
                                 meteo_source="ERA5")
        with open(nml_path) as f:
            nml = yaml.safe_load(f)

        # CatchmentProcessor must run first (creates HRU shapefile)
        CatchmentProcessor(nml_path).process_catchment()
        # ERA5LandAnalyzer must run first (creates era5_land_precip.nc)
        ERA5LandAnalyzer(nml_path, force_reprocess=True)

        gen = GridWeightsGenerator(nml_path)
        gen.generate()

        weights_path = (Path(nml["main_dir"]) / nml["config_dir"]
                        / f"catchment_{nml['gauge_id']}" / "data_obs"
                        / "GridWeights.txt")
        assert weights_path.exists(), f"Expected {weights_path}"


class TestHARAnalyzer:
    """Tests for preprocess_meteo.HARAnalyzer."""

    def test_init_without_error(self, make_namelist):
        from preprocess_meteo import HARAnalyzer

        nml_path = make_namelist(gauge_id="9999", model_type="HBV",
                                 meteo_source="HAR")
        analyzer = HARAnalyzer(nml_path, force_reprocess=True)
        assert analyzer.gauge_id == "9999"

    def test_processed_files_exist(self, make_namelist):
        from preprocess_meteo import HARAnalyzer

        nml_path = make_namelist(gauge_id="9999", model_type="HBV",
                                 meteo_source="HAR")
        analyzer = HARAnalyzer(nml_path, force_reprocess=True)

        expected = [
            "har_precip.nc",
            "har_temp_mean.nc",
        ]
        for fname in expected:
            path = analyzer.output_path / fname
            assert path.exists(), f"Expected processed file: {path}"

    def test_monthly_temperature_averages(self, make_namelist):
        from preprocess_meteo import HARAnalyzer

        nml_path = make_namelist(gauge_id="9999", model_type="HBV",
                                 meteo_source="HAR")
        analyzer = HARAnalyzer(nml_path, force_reprocess=True)
        analyzer.calculate_monthly_temperature_averages()

        csv_path = analyzer.output_path / "har_monthly_temperature_averages.csv"
        assert csv_path.exists(), f"Expected {csv_path}"

    def test_monthly_pet_averages(self, make_namelist):
        from preprocess_meteo import HARAnalyzer

        nml_path = make_namelist(gauge_id="9999", model_type="HBV",
                                 meteo_source="HAR")
        analyzer = HARAnalyzer(nml_path, force_reprocess=True)
        analyzer.calculate_monthly_pet_averages()

        csv_path = analyzer.output_path / "har_monthly_pet_averages.csv"
        assert csv_path.exists(), f"Expected {csv_path}"

    def test_har_grid_weights_total_cells(self, make_namelist):
        """NumberGridCells must match the dimensions of har_precip.nc (the file
        Raven actually reads), not just the subset with non-zero weights."""
        from preprocess_meteo import HARAnalyzer, HARGridWeightsGenerator
        from preprocess_catchment import CatchmentProcessor
        import xarray as xr
        import yaml

        nml_path = make_namelist(gauge_id="9999", model_type="HBV",
                                 meteo_source="HAR")
        with open(nml_path) as f:
            nml = yaml.safe_load(f)

        # Need HRU shapefile first
        CatchmentProcessor(nml_path).process_catchment()

        analyzer = HARAnalyzer(nml_path, force_reprocess=True)
        gen = HARGridWeightsGenerator(nml_path)
        gen.generate()

        weights_path = (
            Path(nml["main_dir"]) / nml["config_dir"]
            / f"catchment_{nml['gauge_id']}" / "data_obs" / "GridWeights_HAR.txt"
        )
        assert weights_path.exists(), f"Expected {weights_path}"

        # Determine actual grid size from the processed har_precip.nc file
        precip_nc = analyzer.output_path / "har_precip.nc"
        assert precip_nc.exists(), f"Expected har_precip.nc at {precip_nc}"
        ds = xr.open_dataset(precip_nc)
        ny = ds.dims["south_north"]
        nx = ds.dims["west_east"]
        expected_cells = ny * nx
        ds.close()

        # :NumberGridCells in the weights file must equal the full file dimensions
        content = weights_path.read_text()
        reported = None
        for line in content.splitlines():
            if ":NumberGridCells" in line or ":NumberStations" in line:
                reported = int(line.split()[-1])
                break

        assert reported is not None, "No :NumberGridCells line found in GridWeights_HAR.txt"
        assert reported == expected_cells, (
            f":NumberGridCells ({reported}) must equal har_precip.nc dimensions "
            f"south_north×west_east = {ny}×{nx} = {expected_cells}"
        )

    def test_compute_monthly_averages_for_subbasin(self, make_namelist, tmp_path):
        from preprocess_meteo import HARAnalyzer

        nml_path = make_namelist(gauge_id="9999", model_type="HBV",
                                 meteo_source="HAR")
        analyzer = HARAnalyzer(nml_path, force_reprocess=True)
        analyzer.calculate_monthly_temperature_averages()
        analyzer.calculate_monthly_pet_averages()

        sb_out = tmp_path / "subbasin_1"
        ok = analyzer.compute_monthly_averages_for_subbasin("9901", sb_out)
        # Returns False if shapefile not found (9901 does exist in fixtures)
        assert ok is True, "compute_monthly_averages_for_subbasin returned False"
        assert (sb_out / "har_monthly_temperature_averages.csv").exists()
        assert (sb_out / "har_monthly_pet_averages.csv").exists()
