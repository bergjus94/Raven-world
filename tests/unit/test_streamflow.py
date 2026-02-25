"""Unit tests for StreamflowProcessor."""

import pytest
import pandas as pd


class TestStreamflowProcessor:
    """Tests for preprocess_streamflow.StreamflowProcessor."""

    def test_single_basin_creates_rvt(self, make_namelist, tmp_path):
        from preprocess_streamflow import StreamflowProcessor

        nml_path = make_namelist(gauge_id="9999", model_type="HBV",
                                 meteo_source="ERA5")
        proc = StreamflowProcessor(nml_path)
        proc.process()

        rvt_path = proc.output_path / "Q_daily.rvt"
        assert rvt_path.exists(), f"Expected {rvt_path}"

    def test_single_basin_rvt_contains_hydrograph_1(self, make_namelist):
        from preprocess_streamflow import StreamflowProcessor

        nml_path = make_namelist(gauge_id="9999", model_type="HBV",
                                 meteo_source="ERA5")
        proc = StreamflowProcessor(nml_path)
        proc.process()

        content = (proc.output_path / "Q_daily.rvt").read_text()
        assert ":ObservationData\tHYDROGRAPH\t1\tm3/s" in content, (
            "Single-basin RVT must reference basin_id=1"
        )

    def test_single_basin_rvt_date_range(self, make_namelist):
        from preprocess_streamflow import StreamflowProcessor

        nml_path = make_namelist(gauge_id="9999", model_type="HBV",
                                 meteo_source="ERA5")
        proc = StreamflowProcessor(nml_path)
        proc.process()

        content = (proc.output_path / "Q_daily.rvt").read_text()
        # The RVT starts at warm_up_date (1999-01-01) with -1.2345 fill values
        # before real observations begin. The warm-up date must be present.
        assert "1999-01-01" in content

    def test_multi_subbasin_gauged_basin_writes_named_file(self, make_namelist):
        from preprocess_streamflow import StreamflowProcessor
        import yaml

        nml_path = make_namelist(gauge_id="9900", model_type="HBV",
                                 meteo_source="ERA5", multi_subbasin=True)
        with open(nml_path) as f:
            nml = yaml.safe_load(f)

        for sb in nml["subbasins"]:
            if sb.get("gauged", 0):
                proc = StreamflowProcessor(
                    nml_path,
                    basin_id=sb["id"],
                    output_filename=f"Q_daily_{sb['gauge_id']}.rvt",
                )
                proc.gauge_id = str(sb["gauge_id"])
                proc.stream_file = (
                    proc.main_dir
                    / proc.stream_dir_template.format(gauge_id=sb["gauge_id"])
                )
                proc.output_file = (
                    proc.output_path / f"Q_daily_{sb['gauge_id']}.rvt"
                )
                proc.process()

                rvt_path = proc.output_path / f"Q_daily_{sb['gauge_id']}.rvt"
                assert rvt_path.exists()

                content = rvt_path.read_text()
                assert (
                    f":ObservationData\tHYDROGRAPH\t{sb['id']}\tm3/s" in content
                ), f"Expected basin_id={sb['id']} in HYDROGRAPH line"

    def test_custom_output_filename(self, make_namelist):
        from preprocess_streamflow import StreamflowProcessor

        nml_path = make_namelist(gauge_id="9999", model_type="HBV",
                                 meteo_source="ERA5")
        proc = StreamflowProcessor(nml_path, basin_id=1,
                                   output_filename="Q_custom.rvt")
        proc.process()

        assert (proc.output_path / "Q_custom.rvt").exists()
        assert not (proc.output_path / "Q_daily.rvt").exists()
