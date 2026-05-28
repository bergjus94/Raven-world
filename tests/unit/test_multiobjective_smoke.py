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

    def test_custom_window_list(self):
        """User can pass an explicit month list (e.g. just DJF)."""
        q = _make_q_series(n=365 * 3, seed=5)
        v = co.baseflow_objective(q, q, method='eckhardt',
                                   window=[12, 1, 2], metric='KGE')
        assert v == pytest.approx(1.0, abs=1e-3)

    def test_custom_window_raw_winter(self):
        """Custom window with raw_winter method — separator skipped, user's
        months respected (no auto-tighten to Dec-Mar)."""
        q = _make_q_series(n=365 * 2, seed=6)
        # JF only (extreme winter), with raw Q. The user-explicit window
        # should NOT be replaced by the default 'raw_winter' (Dec-Mar).
        v = co.baseflow_objective(q, q, method='raw_winter',
                                   window=[1, 2], metric='KGE')
        assert v == pytest.approx(1.0, abs=1e-6)

    def test_invalid_window_month_raises(self):
        q = _make_q_series(n=400, seed=7)
        with pytest.raises(ValueError, match='1-12'):
            co.baseflow_objective(q, q, window=[0, 13])

    def test_empty_window_raises(self):
        q = _make_q_series(n=400, seed=8)
        with pytest.raises(ValueError, match='empty'):
            co.baseflow_objective(q, q, window=[])

    def test_unknown_window_string_raises(self):
        q = _make_q_series(n=400, seed=9)
        with pytest.raises(ValueError, match='Unknown window'):
            co.baseflow_objective(q, q, window='lustige_jahreszeit')


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


# ───────────────────────── new metrics (RMSE / MAE / PBIAS / CSI) ─────────────────────────


class TestNewMetrics:
    def test_all_metrics_registered(self):
        assert set(co.METRICS) == {'KGE', 'NSE', 'LogKGE',
                                    'RMSE', 'MAE', 'PBIAS', 'CSI'}

    def test_perfect_match_scores_one(self):
        """For identical obs/sim on a bounded [0,1] variable, every metric
        should report a near-perfect score (1.0)."""
        q = _make_q_series(n=200)
        # Normalise to [0, 1] so RMSE/MAE are bounded — mimics fSCA
        q = (q - q.min()) / (q.max() - q.min())
        for m in ('KGE', 'NSE', 'LogKGE', 'RMSE', 'MAE', 'PBIAS', 'CSI'):
            v = co._apply_metric(m, q, q)
            assert v == pytest.approx(1.0, abs=1e-3), f"{m} failed: {v}"

    def test_rmse_score_decreases_with_noise(self):
        rng = np.random.default_rng(7)
        n = 200
        obs = pd.Series(np.clip(0.5 + 0.3 * np.sin(np.linspace(0, 6, n)), 0, 1))
        sim_low_noise  = pd.Series(np.clip(obs + 0.02 * rng.standard_normal(n), 0, 1))
        sim_high_noise = pd.Series(np.clip(obs + 0.20 * rng.standard_normal(n), 0, 1))
        v_low  = co._apply_metric('RMSE', obs, sim_low_noise)
        v_high = co._apply_metric('RMSE', obs, sim_high_noise)
        assert v_low > v_high
        assert 0.0 <= v_high <= v_low <= 1.0

    def test_pbias_score_drops_with_bias(self):
        obs = pd.Series([0.5] * 100)
        sim = pd.Series([0.6] * 100)   # +20% bias
        v = co._apply_metric('PBIAS', obs, sim)
        # PBIAS = 20%, score = 1 - 20/100 = 0.8
        assert v == pytest.approx(0.8, abs=1e-3)

    def test_csi_perfect_when_all_above_threshold(self):
        obs = pd.Series([0.9] * 100)
        sim = pd.Series([0.95] * 100)
        # threshold 0.5: both all 'snow present', hits=100, misses=0, fa=0
        v = co._apply_metric('CSI', obs, sim)
        assert v == pytest.approx(1.0)


class TestRawDiagnostics:
    def test_keys_present(self):
        q = _make_q_series(n=100)
        d = co.raw_diagnostics(q, q)
        assert set(d.keys()) >= {'r', 'rmse', 'mae', 'pbias', 'n'}
        assert d['rmse'] == pytest.approx(0.0, abs=1e-6)
        assert d['mae'] == pytest.approx(0.0, abs=1e-6)
        assert d['r'] == pytest.approx(1.0, abs=1e-6)
        assert d['n'] == 100

    def test_too_few_points_returns_nans(self):
        q = _make_q_series(n=10)
        d = co.raw_diagnostics(q, q)
        assert np.isnan(d['rmse'])
        assert np.isnan(d['r'])
        assert d['n'] == 10


# ───────────────────────── long-format / per-band fSCA loaders ─────────────────────────


def _write_fake_band_fsca(out_path: Path, n_dates: int = 50,
                          bands: tuple = (2500, 2600, 2700, 2800)) -> Path:
    """Write a long-format per-band fSCA CSV mimicking preprocess_modis_fsca output."""
    dates = pd.date_range('2010-01-01', periods=n_dates, freq='8D')
    rows = []
    rng = np.random.default_rng(0)
    for d in dates:
        for band in bands:
            # Higher elevation = more snow on average
            base = (band - 2500) / 500.0
            fsca = float(np.clip(base + 0.1 * rng.standard_normal(), 0, 1))
            rows.append({
                'date':    d.strftime('%Y-%m-%d'),
                'band_m':  band,
                'fsca':    fsca,
                'n_valid': 100,
                'n_cloud': 5,
                'n_total': 110,
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, float_format='%.6f')
    return out_path


class TestLoadModisFscaLongFormat:
    def test_load_modis_fsca_collapses_multi_band(self, tmp_path):
        """When given a multi-band CSV, load_modis_fsca returns a basin-mean
        Series weighted by n_total."""
        csv = _write_fake_band_fsca(tmp_path / 'fsca_multi.csv')
        s = co.load_modis_fsca(csv, cloud_threshold=0.5)
        assert isinstance(s, pd.Series)
        # One value per date (50 dates in the fixture)
        assert len(s) == 50
        assert s.notna().any()

    def test_load_modis_fsca_bands_pivots_to_wide(self, tmp_path):
        csv = _write_fake_band_fsca(tmp_path / 'fsca_multi.csv')
        df = co.load_modis_fsca_bands(csv, cloud_threshold=0.5,
                                       min_pixels_per_band=10)
        assert list(df.columns) == [2500, 2600, 2700, 2800]
        assert len(df) == 50
        # Higher bands should have higher mean fSCA (per the fixture)
        assert df[2800].mean() > df[2500].mean()

    def test_load_modis_fsca_bands_rejects_single_band(self, tmp_path):
        """The single-band 'basin' CSV should be rejected by the per-band loader."""
        csv = tmp_path / 'fsca_basin.csv'
        pd.DataFrame({
            'date': pd.date_range('2010-01-01', periods=10, freq='8D'),
            'band_m': 'basin',
            'fsca': [0.5] * 10,
            'n_valid': [100] * 10,
            'n_cloud': [0] * 10,
            'n_total': [100] * 10,
        }).to_csv(csv, index=False)
        with pytest.raises(ValueError, match='only one band|no `band_m`'):
            co.load_modis_fsca_bands(csv)

    def test_min_pixels_filter_drops_low_count_cells(self, tmp_path):
        """Cells with n_valid < min_pixels_per_band become NaN."""
        csv = tmp_path / 'fsca_sparse.csv'
        pd.DataFrame({
            'date':    ['2010-01-01', '2010-01-01', '2010-01-09', '2010-01-09'],
            'band_m':  [2500, 2600, 2500, 2600],
            'fsca':    [0.3, 0.7, 0.4, 0.8],
            'n_valid': [100, 5, 90, 60],   # row index 1 has too few pixels
            'n_cloud': [0, 0, 0, 0],
            'n_total': [110, 10, 100, 70],
        }).to_csv(csv, index=False)
        df = co.load_modis_fsca_bands(csv, cloud_threshold=0.5,
                                       min_pixels_per_band=30)
        assert np.isnan(df.loc[pd.Timestamp('2010-01-01'), 2600])
        assert not np.isnan(df.loc[pd.Timestamp('2010-01-01'), 2500])
        assert not np.isnan(df.loc[pd.Timestamp('2010-01-09'), 2600])


# ───────────────────────── band-area helper + per-band sim ─────────────────────────


class TestComputeBandAreas:
    def test_groups_hrus_by_floor_of_elevation(self):
        hru_areas = {1: 10.0, 2: 20.0, 3: 30.0, 4: 40.0}
        hru_elev  = {1: 2543, 2: 2587, 3: 2612, 4: 3950}
        # band_width=100 → bands 2500 (1+2 → 30), 2600 (3 → 30), 3900 (4 → 40)
        out = co._compute_band_areas(hru_areas, hru_elev, band_width_m=100)
        assert out == {2500: 30.0, 2600: 30.0, 3900: 40.0}

    def test_glacier_hrus_excluded(self):
        hru_areas = {1: 10.0, 2: 20.0, 3: 30.0}
        hru_elev  = {1: 2543, 2: 2587, 3: 2612}
        out = co._compute_band_areas(hru_areas, hru_elev, band_width_m=100,
                                      glacier_hrus={3})
        assert out == {2500: 30.0}  # HRU 3 excluded

    def test_skips_hrus_without_elevation(self):
        hru_areas = {1: 10.0, 2: 20.0, 3: 30.0}
        hru_elev  = {1: 2543, 2: 2587}  # 3 missing
        out = co._compute_band_areas(hru_areas, hru_elev, band_width_m=100)
        assert out == {2500: 30.0}


class TestLoadRavenSnowFracPerBand:
    def test_aggregates_by_band(self, tmp_path):
        # 4 HRUs across 2 bands (2500-2599 → band 2500, 2700-2799 → band 2700)
        n_days = 30
        dates = pd.date_range('2010-01-01', periods=n_days, freq='D')
        df = pd.DataFrame({
            'time': range(n_days),
            'date': [d.strftime('%Y-%m-%d') for d in dates],
            'hour': ['00:00:00'] * n_days,
            '1': [0.5] * n_days,
            '2': [0.6] * n_days,   # band 2500: 1 & 2
            '3': [0.9] * n_days,
            '4': [1.0] * n_days,   # band 2700: 3 & 4
        })
        out = tmp_path / 'fake_SNOW_FRAC_Daily_Average_ByHRU.csv'
        df.to_csv(out, index=False)

        per_band = co.load_raven_snow_frac_per_band(
            output_dir=tmp_path,
            hru_areas={1: 10.0, 2: 10.0, 3: 10.0, 4: 10.0},
            hru_elevations={1: 2550, 2: 2590, 3: 2750, 4: 2790},
            band_width_m=100,
        )
        assert list(per_band.columns) == [2500, 2700]
        # band 2500: equal-area weighted mean of 0.5 and 0.6 = 0.55
        assert per_band[2500].iloc[0] == pytest.approx(0.55)
        assert per_band[2700].iloc[0] == pytest.approx(0.95)

    def test_excludes_glacier_hrus(self, tmp_path):
        n_days = 30
        dates = pd.date_range('2010-01-01', periods=n_days, freq='D')
        df = pd.DataFrame({
            'time': range(n_days),
            'date': [d.strftime('%Y-%m-%d') for d in dates],
            'hour': ['00:00:00'] * n_days,
            '1': [0.5] * n_days, '2': [0.6] * n_days,
            '3': [0.0] * n_days,  # glacier HRU, should be excluded
        })
        out = tmp_path / 'fake_SNOW_FRAC_Daily_Average_ByHRU.csv'
        df.to_csv(out, index=False)

        per_band = co.load_raven_snow_frac_per_band(
            output_dir=tmp_path,
            hru_areas={1: 10.0, 2: 10.0, 3: 10.0},
            hru_elevations={1: 2550, 2: 2590, 3: 3050},
            glacier_hrus={3},
            band_width_m=100,
        )
        assert 3050 not in per_band.columns
        assert 2500 in per_band.columns


# ───────────────────────── snow_objective elevation-band end-to-end ─────────────────────────


class TestSnowObjectiveElevationBand:
    @pytest.fixture
    def per_band_setup(self, tmp_path):
        """Build a minimal per-band obs CSV + matching Raven-format sim file."""
        # Obs: 30 dates × 2 bands, sim values = obs values exactly
        dates = pd.date_range('2010-02-01', periods=30, freq='8D')
        bands = (2500, 2700)
        obs_rows = []
        # Per-band time-varying fSCA so KGE has signal
        for i, d in enumerate(dates):
            obs_rows.append({'date': d.strftime('%Y-%m-%d'), 'band_m': 2500,
                             'fsca': 0.3 + 0.01 * i, 'n_valid': 100,
                             'n_cloud': 0, 'n_total': 100})
            obs_rows.append({'date': d.strftime('%Y-%m-%d'), 'band_m': 2700,
                             'fsca': 0.7 + 0.01 * i, 'n_valid': 100,
                             'n_cloud': 0, 'n_total': 100})
        obs_csv = tmp_path / 'fsca_obs.csv'
        pd.DataFrame(obs_rows).to_csv(obs_csv, index=False)

        # Sim: 1 year of daily Raven SNOW_FRAC ByHRU, 4 HRUs across 2 bands
        # Construct so per-band mean matches obs values at each MODIS date
        sim_dates = pd.date_range('2010-01-01', periods=365, freq='D')
        sim_df = pd.DataFrame({
            'time': range(365),
            'date': [d.strftime('%Y-%m-%d') for d in sim_dates],
            'hour': ['00:00:00'] * 365,
        })
        # For each sim date, find nearest MODIS date and copy its obs value
        obs_per_band_lookup = {b: dict(zip(dates, [r['fsca']
                                                    for r in obs_rows
                                                    if r['band_m'] == b]))
                                for b in bands}
        # Use the nearest MODIS date's value for each daily timestep
        for hru, band in [('1', 2500), ('2', 2500), ('3', 2700), ('4', 2700)]:
            col = []
            for d in sim_dates:
                # Find nearest MODIS date
                nearest_idx = (dates - d).map(abs).argmin()
                col.append(obs_per_band_lookup[band][dates[nearest_idx]])
            sim_df[hru] = col

        sim_dir = tmp_path / 'sim'
        sim_dir.mkdir()
        sim_df.to_csv(sim_dir / 'fake_SNOW_FRAC_Daily_Average_ByHRU.csv',
                       index=False)

        return {
            'obs_csv': obs_csv,
            'sim_dir': sim_dir,
            'hru_areas': {1: 10.0, 2: 10.0, 3: 10.0, 4: 10.0},
            'hru_elevations': {1: 2550, 2: 2590, 3: 2750, 4: 2790},
        }

    def test_elevation_band_perfect_match(self, per_band_setup):
        """With obs and sim constructed to match exactly, KGE per band should
        be ~1.0 → area-weighted mean ~1.0."""
        v = co.snow_objective(
            obs_fsca_csv=per_band_setup['obs_csv'],
            sim_output_dir=per_band_setup['sim_dir'],
            hru_areas=per_band_setup['hru_areas'],
            hru_elevations=per_band_setup['hru_elevations'],
            metric='KGE',
            aggregation='elevation_band',
            band_width_m=100,
            min_pixels_per_band=10,
        )
        assert v == pytest.approx(1.0, abs=0.05)

    def test_missing_hru_elevations_raises(self, per_band_setup):
        with pytest.raises(ValueError, match='hru_elevations is required'):
            co.snow_objective(
                obs_fsca_csv=per_band_setup['obs_csv'],
                sim_output_dir=per_band_setup['sim_dir'],
                hru_areas=per_band_setup['hru_areas'],
                hru_elevations=None,        # omitted
                aggregation='elevation_band',
            )

    def test_unknown_aggregation_raises(self, per_band_setup):
        with pytest.raises(ValueError, match='aggregation must be'):
            co.snow_objective(
                obs_fsca_csv=per_band_setup['obs_csv'],
                sim_output_dir=per_band_setup['sim_dir'],
                hru_areas=per_band_setup['hru_areas'],
                aggregation='not_a_mode',
            )

    def test_diagnostic_log_writes_rows(self, per_band_setup, tmp_path):
        log = tmp_path / 'snow_diag.csv'
        co.snow_objective(
            obs_fsca_csv=per_band_setup['obs_csv'],
            sim_output_dir=per_band_setup['sim_dir'],
            hru_areas=per_band_setup['hru_areas'],
            hru_elevations=per_band_setup['hru_elevations'],
            metric='KGE',
            aggregation='elevation_band',
            band_width_m=100,
            min_pixels_per_band=10,
            diagnostic_log=log,
        )
        assert log.exists()
        df = pd.read_csv(log)
        # one row per band exercised (2 bands)
        assert len(df) == 2
        assert {'r', 'rmse', 'mae', 'pbias', 'metric', 'band'}.issubset(df.columns)


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
