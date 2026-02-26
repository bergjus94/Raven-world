"""Unit tests for HBVProcessor file generation."""

import pytest
import yaml
from pathlib import Path


def _get_model_dir(nml_path: Path) -> Path:
    with open(nml_path) as f:
        nml = yaml.safe_load(f)
    return Path(nml["main_dir"]) / nml["config_dir"]


class TestHBVProcessorSingleBasin:
    """Tests for HBVProcessor with a single-basin namelist."""

    @pytest.fixture()
    def prepared_hbv(self, make_namelist):
        """Run catchment + meteo preprocessing so HBVProcessor has inputs."""
        from preprocess_catchment import CatchmentProcessor
        from preprocess_meteo import HARAnalyzer, HARGridWeightsGenerator
        from preprocess_HBV import HBVProcessor

        nml_path = make_namelist(gauge_id="9999", model_type="HBV",
                                 meteo_source="HAR")
        CatchmentProcessor(nml_path).process_catchment()
        analyzer = HARAnalyzer(nml_path, force_reprocess=True)
        analyzer.calculate_monthly_temperature_averages()
        analyzer.calculate_monthly_pet_averages()
        HARGridWeightsGenerator(nml_path).generate()

        processor = HBVProcessor(nml_path)
        return processor, nml_path

    def test_create_rvt_file(self, prepared_hbv):
        processor, nml_path = prepared_hbv
        processor.create_rvt_file(template=False)

        with open(nml_path) as f:
            nml = yaml.safe_load(f)
        model_dir = Path(nml["main_dir"]) / nml["config_dir"]
        rvt_path = model_dir / f"catchment_{nml['gauge_id']}" / "HBV" / f"{nml['gauge_id']}_HBV.rvt"
        assert rvt_path.exists(), f"Expected {rvt_path}"

    def test_rvt_has_redirect_to_file(self, prepared_hbv):
        processor, nml_path = prepared_hbv
        processor.create_rvt_file(template=False)

        with open(nml_path) as f:
            nml = yaml.safe_load(f)
        model_dir = Path(nml["main_dir"]) / nml["config_dir"]
        rvt_path = model_dir / f"catchment_{nml['gauge_id']}" / "HBV" / f"{nml['gauge_id']}_HBV.rvt"
        content = rvt_path.read_text()
        # Paths must use ../data_obs/ (Raven runs from HBV/)
        assert "../data_obs/" in content, (
            "RVT paths must use ../data_obs/ (not data_obs/)"
        )

    def test_rvt_single_gauge_block(self, prepared_hbv):
        processor, nml_path = prepared_hbv
        processor.create_rvt_file(template=False)

        with open(nml_path) as f:
            nml = yaml.safe_load(f)
        model_dir = Path(nml["main_dir"]) / nml["config_dir"]
        rvt_path = model_dir / f"catchment_{nml['gauge_id']}" / "HBV" / f"{nml['gauge_id']}_HBV.rvt"
        content = rvt_path.read_text()
        assert content.count(":Gauge") == 1
        assert content.count(":EndGauge") == 1

    def test_create_rvp_file(self, prepared_hbv):
        processor, nml_path = prepared_hbv
        processor.create_rvp_file(template=False)

        with open(nml_path) as f:
            nml = yaml.safe_load(f)
        model_dir = Path(nml["main_dir"]) / nml["config_dir"]
        rvp_path = model_dir / f"catchment_{nml['gauge_id']}" / "HBV" / f"{nml['gauge_id']}_HBV.rvp"
        assert rvp_path.exists()

    def test_create_all_files(self, prepared_hbv):
        processor, nml_path = prepared_hbv
        processor.create_all_files(template=False)

        with open(nml_path) as f:
            nml = yaml.safe_load(f)
        model_dir = Path(nml["main_dir"]) / nml["config_dir"]
        hbv_dir = model_dir / f"catchment_{nml['gauge_id']}" / "HBV"
        extensions = [".rvh", ".rvt", ".rvp", ".rvi", ".rvc"]
        for ext in extensions:
            path = hbv_dir / f"{nml['gauge_id']}_HBV{ext}"
            assert path.exists(), f"Expected {path}"


class TestHBVProcessorMultiSubbasin:
    """Tests for HBVProcessor with multi-subbasin namelist."""

    @pytest.fixture()
    def prepared_multi_hbv(self, make_namelist):
        from preprocess_catchment import MultiSubbasinProcessor
        from preprocess_meteo import HARAnalyzer, HARGridWeightsGenerator
        from preprocess_HBV import HBVProcessor

        nml_path = make_namelist(gauge_id="9900", model_type="HBV",
                                 meteo_source="HAR", multi_subbasin=True)
        MultiSubbasinProcessor(nml_path).process_all_subbasins()
        analyzer = HARAnalyzer(nml_path, force_reprocess=True)
        analyzer.calculate_monthly_temperature_averages()
        analyzer.calculate_monthly_pet_averages()
        HARGridWeightsGenerator(nml_path).generate()

        # Per-subbasin monthly averages
        with open(nml_path) as f:
            nml = yaml.safe_load(f)
        shared_data = (
            Path(nml["main_dir"]) / nml["config_dir"]
            / f"catchment_{nml['gauge_id']}" / "data_obs"
        )
        for sb in nml["subbasins"]:
            sb_dir = shared_data / f"subbasin_{sb['id']}"
            analyzer.compute_monthly_averages_for_subbasin(
                str(sb["gauge_id"]), sb_dir
            )

        processor = HBVProcessor(nml_path)
        return processor, nml_path

    def test_rvt_has_three_gauge_blocks(self, prepared_multi_hbv):
        processor, nml_path = prepared_multi_hbv
        processor.create_rvt_file(template=False)

        with open(nml_path) as f:
            nml = yaml.safe_load(f)
        model_dir = Path(nml["main_dir"]) / nml["config_dir"]
        rvt_path = model_dir / f"catchment_{nml['gauge_id']}" / "HBV" / f"{nml['gauge_id']}_HBV.rvt"
        content = rvt_path.read_text()
        assert content.count(":Gauge") == 3, (
            "Multi-subbasin RVT must have one :Gauge block per subbasin"
        )

    def test_rvt_redirect_only_for_gauged(self, prepared_multi_hbv):
        processor, nml_path = prepared_multi_hbv
        processor.create_rvt_file(template=False)

        with open(nml_path) as f:
            nml = yaml.safe_load(f)
        model_dir = Path(nml["main_dir"]) / nml["config_dir"]
        rvt_path = model_dir / f"catchment_{nml['gauge_id']}" / "HBV" / f"{nml['gauge_id']}_HBV.rvt"
        content = rvt_path.read_text()

        # Only 1 gauged subbasin (9900) → only 1 redirect
        redirects = [l for l in content.splitlines() if ":RedirectToFile" in l
                     and "Q_daily" in l]
        assert len(redirects) == 1, (
            f"Expected exactly 1 Q_daily redirect (gauged only), found {len(redirects)}"
        )
        assert "9900" in redirects[0], "Redirect should reference gauge_id 9900"


class TestHBVSingleTPHiPr:
    """Tests for HBVProcessor with TPHiPr precipitation override (HAR for temperature)."""

    @pytest.fixture()
    def prepared_tphipr_hbv(self, make_namelist, fixture_root):
        """Run CatchmentProcessor + HAR (temp/PET) + TPHiPr (precip), then HBVProcessor."""
        from preprocess_catchment import CatchmentProcessor
        from preprocess_meteo import (
            HARAnalyzer,
            HARGridWeightsGenerator,
            TPHiPrAnalyzer,
            TPHiPrGridWeightsGenerator,
        )
        from preprocess_HBV import HBVProcessor

        tphipr_dir = str(fixture_root / "01_data" / "meteo" / "TPHiPr")
        nml_path = make_namelist(
            gauge_id="9999",
            model_type="HBV",
            meteo_source="HAR",
            extra={"precip_source": "TPHiPr", "tphipr_dir": tphipr_dir},
        )

        CatchmentProcessor(nml_path).process_catchment()
        analyzer = HARAnalyzer(nml_path, force_reprocess=True)
        analyzer.calculate_monthly_temperature_averages()
        analyzer.calculate_monthly_pet_averages()
        HARGridWeightsGenerator(nml_path).generate()
        TPHiPrAnalyzer(nml_path, force_reprocess=True).process()
        TPHiPrGridWeightsGenerator(nml_path).generate()

        return HBVProcessor(nml_path), nml_path

    def _rvt_content(self, processor, nml_path):
        processor.create_rvt_file(template=False)
        with open(nml_path) as f:
            nml = yaml.safe_load(f)
        model_dir = Path(nml["main_dir"]) / nml["config_dir"]
        rvt_path = (model_dir / f"catchment_{nml['gauge_id']}"
                    / "HBV" / f"{nml['gauge_id']}_HBV.rvt")
        return rvt_path.read_text()

    def test_rvt_precip_uses_tphipr(self, prepared_tphipr_hbv):
        """Rainfall block in .rvt must reference tphipr_precip.nc and GridWeights_TPHiPr.txt."""
        processor, nml_path = prepared_tphipr_hbv
        content = self._rvt_content(processor, nml_path)
        assert "tphipr_precip.nc" in content, "RVT must reference tphipr_precip.nc for rainfall"
        assert "GridWeights_TPHiPr.txt" in content, "RVT must reference GridWeights_TPHiPr.txt"

    def test_rvt_temperature_uses_har(self, prepared_tphipr_hbv):
        """Temperature block in .rvt must still reference HAR grid weights."""
        processor, nml_path = prepared_tphipr_hbv
        content = self._rvt_content(processor, nml_path)
        assert "GridWeights_HAR.txt" in content, (
            "RVT must still use GridWeights_HAR.txt for temperature when precip_source=TPHiPr"
        )
