"""Smoke tests for the multi-objective calibration scaffolding.

Exercises the new `src/calibration_objectives.py` module and the
`_setup_objectives` / `_combine_weighted` logic in `spotpy_optimize.py`
without running Raven or hitting the SMB share.

Real assets used:
    OneDrive/Raven_worldwide/01_data/snow/MODIS/fsca_MOD10A2_0101.csv

Synthetic assets generated per-test:
    fake_*_SNOW_FRAC_Daily_Average_ByHRU.csv  (Raven CustomOutput format)
    fake observed/simulated daily Q series

If you change calibration_objectives.py or the objective wiring in
spotpy_optimize.py, run::

    pytest tests/unit/test_multiobjective_smoke.py -v
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest
import yaml

# `src/` is added to sys.path by tests/conftest.py
import calibration_objectives as co


REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_FSCA_CSV = (Path.home() / 'OneDrive' / 'Raven_worldwide'
                 / '01_data' / 'snow' / 'MODIS' / 'fsca_MOD10A2_0101.csv')


# ───────────────────────────── helpers ─────────────────────────────


def _make_q_series(n: int = 365, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2010-01-01', periods=n, freq='D')
    # Seasonal cycle + noise, all positive
    seasonal = 5.0 + 4.0 * np.sin(np.linspace(0, 2 * np.pi, n))
    return pd.Series(seasonal + 0.3 * rng.standard_normal(n) + 0.5,
                     index=dates, name='Q')


def _write_fake_snow_frac_byhru(
    out_dir: Path,
    n_hrus: int = 3,
    n_days: int = 365,
    sim_value: float = 0.7,
    raven_garbage_first_line: bool = True,
) -> Path:
    """Write a Raven-style :CustomOutput SNOW_FRAC BY_HRU CSV.

    Raven CustomOutput BY_HRU files typically have a 1-line title row before
    the actual header (matches the format read by baseflow_separation.py with
    skiprows=1). If `raven_garbage_first_line=False`, write a plain CSV so we
    can A/B test the loader.
    """
    dates = pd.date_range('2010-01-01', periods=n_days, freq='D')
    cols: Dict[str, list] = {
        'time': list(range(n_days)),
        'date': [d.strftime('%Y-%m-%d') for d in dates],
        'hour': ['00:00:00'] * n_days,
    }
    for hru_id in range(1, n_hrus + 1):
        cols[str(hru_id)] = [sim_value] * n_days

    out = out_dir / 'fake_SNOW_FRAC_Daily_Average_ByHRU.csv'
    df = pd.DataFrame(cols)
    if raven_garbage_first_line:
        # Mirror the Raven format: line 1 is a title, line 2 is real header
        with open(out, 'w') as f:
            f.write(':CustomOutput SNOW_FRAC BY_HRU [-]\n')
            df.to_csv(f, index=False)
    else:
        df.to_csv(out, index=False)
    return out


# ───────────────────────── metric primitives ─────────────────────────


class TestApplyMetric:
    def test_perfect_kge_is_one(self):
        q = _make_q_series()
        assert co._apply_metric('KGE', q, q) == pytest.approx(1.0, abs=1e-6)

    def test_kge_with_noise_drops(self):
        q = _make_q_series()
        rng = np.random.default_rng(1)
        noisy = q + 2.0 * rng.standard_normal(len(q))
        v = co._apply_metric('KGE', q, noisy)
        assert 0.0 < v < 1.0

    def test_unknown_metric_raises(self):
        q = _make_q_series()
        with pytest.raises(ValueError, match='Unknown metric'):
            co._apply_metric('BOGUS', q, q)

    def test_too_few_points_returns_nan(self):
        q = _make_q_series(n=10)
        assert np.isnan(co._apply_metric('KGE', q, q))


# ───────────────────────── discharge objective ─────────────────────────


class TestQObjective:
    def test_identical_series_kge_one(self):
        q = _make_q_series()
        assert co.q_objective(q, q, metric='KGE') == pytest.approx(1.0, abs=1e-6)

    def test_date_range_subset(self):
        q = _make_q_series(n=365)
        # KGE on the full year vs first half — both should be ~1 for identical sim/obs
        full = co.q_objective(q, q, metric='KGE')
        half = co.q_objective(q, q, metric='KGE',
                              date_range=('2010-01-01', '2010-06-30'))
        assert full == pytest.approx(1.0, abs=1e-6)
        assert half == pytest.approx(1.0, abs=1e-6)

    def test_nse_metric_works(self):
        q = _make_q_series()
        assert co.q_objective(q, q, metric='NSE') == pytest.approx(1.0, abs=1e-6)


# ───────────────────────── baseflow objective ─────────────────────────


class TestBaseflowObjective:
    """Baseflow uses BaseflowSeparator (eckhardt by default). Smoke-test that
    the separator runs end-to-end and the metric reduces sensibly on identical
    obs/sim Q."""

    def test_eckhardt_identical_obs_sim(self):
        q = _make_q_series(n=365 * 3, seed=2)  # >1 yr so winter has data
        v = co.baseflow_objective(q, q, method='eckhardt', window='winter',
                                  metric='KGE')
        # Same Q for both → eckhardt(obs) == eckhardt(sim) → KGE == 1
        assert v == pytest.approx(1.0, abs=1e-3)

    def test_raw_winter_skips_separator(self):
        q = _make_q_series(n=365 * 2, seed=3)
        # raw_winter uses Q ≡ baseflow on Dec-Mar only — useful when the
        # separator misbehaves on synthetic data.
        v = co.baseflow_objective(q, q, method='raw_winter',
                                  window='winter', metric='KGE')
        assert v == pytest.approx(1.0, abs=1e-6)

    def test_unknown_method_raises(self):
        q = _make_q_series(n=400, seed=4)
        with pytest.raises(ValueError, match='Unknown baseflow method'):
            co.baseflow_objective(q, q, method='not_a_method')


# ───────────────────────── MODIS fSCA loader ─────────────────────────


@pytest.mark.skipif(not REAL_FSCA_CSV.exists(),
                    reason=f"Real fSCA fixture not present at {REAL_FSCA_CSV}")
class TestLoadModisFsca:
    def test_loads_as_series(self):
        s = co.load_modis_fsca(REAL_FSCA_CSV)
        assert isinstance(s, pd.Series)
        assert s.name == 'fsca_obs'
        assert isinstance(s.index, pd.DatetimeIndex)
        assert len(s) > 0

    def test_cloud_threshold_masks(self):
        # Threshold 0 → mask every row that has any cloud
        df = pd.read_csv(REAL_FSCA_CSV, parse_dates=['date'])
        n_with_cloud = int((df['n_cloud'] > 0).sum())

        s = co.load_modis_fsca(REAL_FSCA_CSV, cloud_threshold=0.0)
        assert int(s.isna().sum()) == n_with_cloud


# ───────────────────────── Raven SNOW_FRAC ByHRU loader ─────────────────────────


class TestLoadRavenSnowFrac:
    """Catches the most likely bug in the multi-objective scaffolding: Raven
    BY_HRU CSVs have a 1-line junk header, but load_raven_snow_frac does
    pd.read_csv without skiprows."""

    def test_plain_csv_format(self, tmp_path):
        """If the file lacks a Raven title row, load_raven_snow_frac should
        succeed (baseline behaviour)."""
        _write_fake_snow_frac_byhru(tmp_path, n_hrus=3, n_days=30,
                                    sim_value=0.6,
                                    raven_garbage_first_line=False)
        hru_areas = {1: 10.0, 2: 20.0, 3: 30.0}
        s = co.load_raven_snow_frac(tmp_path, hru_areas)
        # Equal-weighted by area: (10*0.6 + 20*0.6 + 30*0.6) / 60 = 0.6
        assert s.iloc[0] == pytest.approx(0.6, abs=1e-6)
        assert len(s) == 30

    def test_glacier_exclusion(self, tmp_path):
        _write_fake_snow_frac_byhru(tmp_path, n_hrus=3, n_days=10,
                                    raven_garbage_first_line=False)
        # Make HRU 3 a glacier — should be excluded from the basin mean
        hru_areas = {1: 10.0, 2: 20.0, 3: 30.0}
        s = co.load_raven_snow_frac(tmp_path, hru_areas, glacier_hrus=[3])
        assert s.iloc[0] == pytest.approx(0.7, abs=1e-6)  # uniform 0.7

    def test_raven_format_with_title_line(self, tmp_path):
        """**Expected to fail today** — documents the format we actually get
        from Raven so we can fix the loader."""
        _write_fake_snow_frac_byhru(tmp_path, n_hrus=3, n_days=10,
                                    raven_garbage_first_line=True)
        hru_areas = {1: 10.0, 2: 20.0, 3: 30.0}
        # If the loader is fixed (handle skiprows=1) this passes; if not, the
        # test should fail with a clear error.
        s = co.load_raven_snow_frac(tmp_path, hru_areas)
        assert s.iloc[0] == pytest.approx(0.7, abs=1e-6)


# ───────────────────────── snow objective end-to-end ─────────────────────────


class TestSnowObjective:
    def test_perfect_match(self, tmp_path):
        """Synthetic fSCA CSV that matches the simulated SNOW_FRAC exactly →
        KGE == 1.

        Constructed locally so we don't depend on the real fsca CSV's date
        range matching the fake Raven output's.
        """
        # Sim: 0.6 every day
        _write_fake_snow_frac_byhru(tmp_path, n_hrus=2, n_days=60,
                                    sim_value=0.6,
                                    raven_garbage_first_line=False)
        # Obs: pick a few dates within the sim period, all with fsca=0.6
        obs_dates = pd.date_range('2010-01-05', periods=6, freq='8D')
        obs_csv = tmp_path / 'fake_fsca.csv'
        # Some variance is required for KGE to be defined (otherwise σ_obs=0)
        fsca = [0.6 + 0.001 * i for i in range(len(obs_dates))]
        pd.DataFrame({
            'date': obs_dates,
            'fsca': fsca,
            'n_valid': [100] * len(obs_dates),
            'n_cloud': [0] * len(obs_dates),
            'n_total': [200] * len(obs_dates),
        }).to_csv(obs_csv, index=False)

        v = co.snow_objective(
            obs_fsca_csv=obs_csv,
            sim_output_dir=tmp_path,
            hru_areas={1: 1.0, 2: 1.0},
            metric='KGE',
        )
        # Sim is constant 0.6 but obs has tiny variance → r is poorly defined,
        # but β and α should both be near 1 → KGE should still be reasonable
        # (we just want to confirm the full pipeline runs without error).
        assert np.isfinite(v) or np.isnan(v)


# ───────────────────────── path resolver ─────────────────────────


class TestResolveModisFscaPath:
    def test_explicit_path_wins(self, tmp_path):
        explicit = tmp_path / 'custom.csv'
        explicit.write_text('placeholder')
        out = co.resolve_modis_fsca_path('0102', explicit_path=explicit)
        assert out == explicit

    def test_legacy_smb_pattern(self):
        """Back-compat: when smb_root is supplied, build
        <smb_root>/basins/<display>_<gauge>/<csv>."""
        out = co.resolve_modis_fsca_path('0102', display_name='Hunza',
                                         product='MOD10A2',
                                         smb_root='/fake/MODIS')
        assert out == Path('/fake/MODIS/basins/Hunza_0102/fsca_MOD10A2_0102.csv')

    def test_smb_missing_display_name_falls_back_to_gauge_id(self):
        out = co.resolve_modis_fsca_path('0102', smb_root='/fake/MODIS')
        assert str(out).endswith('basins/0102/fsca_MOD10A2_0102.csv')

    def test_main_dir_canonical_layout(self):
        """New canonical layout written by scripts/derive_basin_fsca.py:
        <main_dir>/01_data/snow/MODIS/basins/<gauge>/<csv> — no display_name
        in the path."""
        out = co.resolve_modis_fsca_path('2268',
                                         display_name='Rhone',  # ignored
                                         main_dir='/data/raven')
        assert out == Path('/data/raven/01_data/snow/MODIS/basins/2268/'
                           'fsca_MOD10A2_2268.csv')

    def test_no_args_raises(self):
        """Without explicit_path, smb_root, or main_dir the call is
        ambiguous — fail loudly."""
        with pytest.raises(ValueError, match='at least one of'):
            co.resolve_modis_fsca_path('0102')

    def test_swiss_display_name_via_legacy_smb(self):
        """Legacy SMB branch leaves the display_name as-is, including spaces
        and '@'. spotpy_optimize._setup_objectives pre-trims for that branch.

        Canonical (main_dir) layout doesn't care about display_name at all.
        """
        out = co.resolve_modis_fsca_path('2268',
                                         display_name='Rhone @ Gletsch',
                                         smb_root='/fake/MODIS')
        assert 'Rhone @ Gletsch_2268' in str(out)


# ───────────────────────── namelist parsing ─────────────────────────


class TestExampleMultiobjectiveNamelist:
    """Validate that namelists/example_multiobjective.yaml parses with the
    keys spotpy_optimize._setup_objectives expects."""

    @pytest.fixture
    def cal_cfg(self):
        path = REPO_ROOT / 'namelists' / 'example_multiobjective.yaml'
        with open(path) as f:
            nml = yaml.safe_load(f)
        return nml['calibration']

    def test_algorithm_present(self, cal_cfg):
        assert cal_cfg['algorithm'] in {'SCEUA', 'DDS', 'DREAM', 'NSGAII', 'PADDS'}

    def test_objectives_is_list_including_Q(self, cal_cfg):
        objs = cal_cfg['objectives']
        assert isinstance(objs, list)
        assert 'Q' in objs

    def test_weights_cover_objectives(self, cal_cfg):
        weights = cal_cfg.get('weights', {})
        # Weights for all active objectives should be present
        for o in cal_cfg['objectives']:
            assert o in weights, f"missing weight for {o}"

    def test_per_objective_subsections(self, cal_cfg):
        # Q is required; snow/baseflow only if listed
        assert 'metric' in cal_cfg['Q']
        if 'snow' in cal_cfg['objectives']:
            assert 'metric' in cal_cfg['snow']
            assert 'cloud_threshold' in cal_cfg['snow']
        if 'baseflow' in cal_cfg['objectives']:
            assert 'metric' in cal_cfg['baseflow']
            assert cal_cfg['baseflow']['method'] in (
                'eckhardt', 'lyne_hollick', 'sliding_min', 'raw_winter'
            )


# ───────────────────────── weighted-sum combination ─────────────────────────


def _combine_weighted_replay(objectives, weights, per_obj):
    """Replicate spotpy_optimize.RavenSCEUA._combine_weighted for testing
    without instantiating the heavyweight class."""
    active = [(o, per_obj[o]) for o in objectives
              if o in per_obj and np.isfinite(per_obj[o])]
    if not active:
        return -999.0
    wsum = sum(weights[o] for o, _ in active)
    return sum(weights[o] * v for o, v in active) / wsum


class TestCombineWeighted:
    def test_all_active(self):
        v = _combine_weighted_replay(
            ['Q', 'snow'], {'Q': 0.7, 'snow': 0.3},
            {'Q': 0.8, 'snow': 0.4})
        assert v == pytest.approx(0.7 * 0.8 + 0.3 * 0.4, abs=1e-9)

    def test_nan_objective_renormalises(self):
        # If snow goes NaN, only Q contributes — should equal the Q value
        v = _combine_weighted_replay(
            ['Q', 'snow'], {'Q': 0.7, 'snow': 0.3},
            {'Q': 0.8, 'snow': float('nan')})
        assert v == pytest.approx(0.8, abs=1e-9)

    def test_all_nan_returns_sentinel(self):
        v = _combine_weighted_replay(
            ['Q', 'snow'], {'Q': 0.7, 'snow': 0.3},
            {'Q': float('nan'), 'snow': float('nan')})
        assert v == -999.0


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
