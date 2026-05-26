#### Baseflow separation methods for validating simulated baseflow.
####
#### Implements a 3-method ensemble for observed discharge:
####   - Eckhardt (2005) two-parameter recursive filter — primary method,
####     used for headline numbers + multi-objective calibration target.
####   - Lyne-Hollick (1979) one-parameter filter, 3-pass — classic baseline.
####   - Sliding-minimum (Pettyjohn & Henning 1979 / HYSEP) — non-recursive
####     sanity check.
####
#### Plus a Brutsaert-Nieber-style recession-constant estimator used to
#### parameterize Eckhardt from data instead of by lookup.
####
#### Usage:
####     sep = BaseflowSeparator(Q)              # Q is a pandas Series indexed by date
####     bf_eckhardt = sep.eckhardt()            # baseflow via Eckhardt
####     all_methods = sep.separate_all()        # dict method-name -> Series
####     bfi = sep.bfi('eckhardt')               # baseflow index
####
#### References
####   Eckhardt, K. (2005). How to construct recursive digital filters for
####     baseflow separation. Hydrol. Process. 19, 507–515.
####   Lyne, V., Hollick, M. (1979). Stochastic time-variable rainfall-runoff
####     modelling. Inst. of Engineers Australia Conf., 89–93.
####   Pettyjohn, W., Henning, R. (1979). Preliminary estimate of regional
####     effective ground-water recharge rates in Ohio. Ohio State U. Tech. Rep.
####   Brutsaert, W., Nieber, J. (1977). Regionalized drought flow hydrographs
####     from a mature glaciated plateau. Water Resour. Res. 13, 637–643.
####
#### Justine Berg, May 2026.

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd


#--------------------------------------------------------------------------------
########################### Lookup tables + defaults ############################
#--------------------------------------------------------------------------------

# Eckhardt's a-priori BFI_max guidance by catchment class.
# Source: Eckhardt (2005, 2008) and subsequent literature.
DEFAULT_BFI_MAX: Dict[str, float] = {
    'porous_perennial':   0.80,   # alluvial valleys, deep soils, perennial flow
    'hardrock_perennial': 0.50,   # bedrock-dominated, perennial flow
    'alpine':             0.50,   # default for Alpine glacier-fed catchments
    'ephemeral':          0.25,   # intermittent / arid streams
}


#--------------------------------------------------------------------------------
############################ BaseflowSeparator class ############################
#--------------------------------------------------------------------------------

class BaseflowSeparator:
    """
    Separate a daily discharge timeseries into baseflow + quickflow using a
    3-method ensemble.

    The input is intentionally generic — a 1-D pandas Series of daily streamflow
    (any unit — methods are scale-invariant; results inherit the input unit).
    """

    def __init__(self, Q: pd.Series, name: str = '') -> None:
        if not isinstance(Q, pd.Series):
            raise TypeError("Q must be a pandas Series indexed by date")
        # Coerce to float, keep NaNs (filters handle them by skipping)
        self.Q = Q.astype(float).copy()
        self.name = name or (Q.name or '')
        # Cache the recession constant once computed
        self._recession_a: Optional[float] = None

    #---------------------------------------------------------------------------------
    # Recession-constant estimation (Brutsaert-Nieber style)
    #---------------------------------------------------------------------------------

    def estimate_recession_constant(
        self,
        min_recession_length: int = 5,
        winter_only: bool = True,
    ) -> float:
        """
        Estimate the daily recession constant α from observed dry-spell recessions.

        For linear-reservoir recession with daily timestep, α = Q(t+1)/Q(t)
        during recession; the median of this ratio across all qualifying
        recession days is a robust estimator.

        Parameters
        ----------
        min_recession_length : int
            Drop recession segments shorter than this many days.
        winter_only : bool
            Use only Nov–Mar recessions (avoids snowmelt/ET-contaminated periods).

        Returns
        -------
        float in (0, 1), the daily α.
        """
        Q = self.Q.dropna()
        if len(Q) < 30:
            raise ValueError("Need at least 30 days of data to estimate α")

        # Find recession days: dQ/dt < 0 and Q(t-1), Q(t) > 0
        ratios = []
        in_seg = 0
        prev = None
        for t, q in Q.items():
            if winter_only and t.month not in (11, 12, 1, 2, 3):
                in_seg = 0
                prev = None
                continue
            if prev is None or q <= 0 or prev <= 0:
                in_seg = 0
                prev = q
                continue
            if q < prev:
                in_seg += 1
                if in_seg >= min_recession_length - 1:
                    ratios.append(q / prev)
            else:
                in_seg = 0
            prev = q

        if not ratios:
            # Fallback: any-season recessions with no min-length filter
            diffs = Q.diff()
            mask = (diffs < 0) & (Q.shift(1) > 0)
            ratios = (Q[mask] / Q.shift(1)[mask]).dropna().tolist()
        if not ratios:
            raise RuntimeError("Could not identify any recession segments")

        a = float(np.median(ratios))
        # Clamp to a sane range
        a = float(np.clip(a, 0.5, 0.999))
        self._recession_a = a
        return a

    #---------------------------------------------------------------------------------
    # Method 1: Eckhardt (2005)
    #---------------------------------------------------------------------------------

    def eckhardt(
        self,
        a: Optional[float] = None,
        BFI_max: float = 0.50,
        catchment_class: Optional[str] = None,
    ) -> pd.Series:
        """
        Eckhardt two-parameter recursive filter.

        Parameters
        ----------
        a : float, optional
            Recession constant. If None, estimated from data.
        BFI_max : float
            Maximum baseflow index, in (0, 1). Default 0.50 (alpine / hard rock).
            Ignored if `catchment_class` is set.
        catchment_class : str, optional
            One of {'porous_perennial', 'hardrock_perennial', 'alpine', 'ephemeral'}
            — looks up BFI_max from DEFAULT_BFI_MAX.

        Returns
        -------
        pd.Series of baseflow, same index as self.Q.
        """
        if a is None:
            a = self._recession_a if self._recession_a is not None else self.estimate_recession_constant()
        if catchment_class is not None:
            if catchment_class not in DEFAULT_BFI_MAX:
                raise ValueError(
                    f"Unknown catchment_class '{catchment_class}'. "
                    f"Options: {sorted(DEFAULT_BFI_MAX)}"
                )
            BFI_max = DEFAULT_BFI_MAX[catchment_class]
        if not (0 < BFI_max < 1):
            raise ValueError(f"BFI_max must be in (0, 1), got {BFI_max}")
        if not (0 < a < 1):
            raise ValueError(f"a must be in (0, 1), got {a}")

        y = self.Q.values
        b = np.full_like(y, np.nan, dtype=float)
        # Initialise: assume initial baseflow ~ first valid observation * BFI_max
        first_valid = np.argmax(~np.isnan(y))
        if np.isnan(y[first_valid]):
            return pd.Series(b, index=self.Q.index, name='baseflow_eckhardt')
        b[first_valid] = y[first_valid] * BFI_max

        denom = 1.0 - a * BFI_max
        for i in range(first_valid + 1, len(y)):
            if np.isnan(y[i]) or np.isnan(b[i-1]):
                # Carry forward the last known baseflow through the gap
                b[i] = b[i-1] if not np.isnan(b[i-1]) else np.nan
                continue
            b_i = ((1.0 - BFI_max) * a * b[i-1] + (1.0 - a) * BFI_max * y[i]) / denom
            # Physical constraint: baseflow can't exceed total flow
            b[i] = min(b_i, y[i])

        return pd.Series(b, index=self.Q.index, name='baseflow_eckhardt')

    #---------------------------------------------------------------------------------
    # Method 2: Lyne-Hollick (1979) 3-pass
    #---------------------------------------------------------------------------------

    def lyne_hollick(
        self,
        alpha: float = 0.925,
        n_passes: int = 3,
    ) -> pd.Series:
        """
        Lyne-Hollick recursive filter, multi-pass.

        Standard configuration is 3-pass (forward → backward → forward),
        α = 0.925 for daily data.
        """
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if n_passes < 1:
            raise ValueError("n_passes must be >= 1")

        y = self.Q.values.astype(float)
        valid = ~np.isnan(y)
        if not valid.any():
            return pd.Series(np.full_like(y, np.nan), index=self.Q.index,
                             name='baseflow_lyne_hollick')

        baseflow = y.copy()
        for k in range(n_passes):
            direction = 1 if k % 2 == 0 else -1
            baseflow = self._lh_one_pass(baseflow, alpha=alpha,
                                         direction=direction)

        return pd.Series(baseflow, index=self.Q.index, name='baseflow_lyne_hollick')

    @staticmethod
    def _lh_one_pass(y: np.ndarray, alpha: float, direction: int) -> np.ndarray:
        """One forward or backward Lyne-Hollick pass."""
        n = len(y)
        if direction == 1:
            rng = range(1, n)
        else:
            rng = range(n - 2, -1, -1)
        qf = np.zeros_like(y)
        b = y.copy()
        for i in rng:
            j = i - 1 if direction == 1 else i + 1
            if np.isnan(y[i]) or np.isnan(y[j]):
                qf[i] = qf[j] if not np.isnan(qf[j]) else 0.0
                continue
            qf[i] = alpha * qf[j] + (1.0 + alpha) / 2.0 * (y[i] - y[j])
            qf[i] = max(qf[i], 0.0)  # quickflow non-negative
            b[i] = y[i] - qf[i]
            b[i] = max(0.0, min(b[i], y[i]))  # 0 <= baseflow <= total
        return b

    #---------------------------------------------------------------------------------
    # Method 3: Sliding-minimum (HYSEP local-minimum style)
    #---------------------------------------------------------------------------------

    def sliding_minimum(self, window_days: int = 5) -> pd.Series:
        """
        Centred sliding minimum followed by linear interpolation between
        local minima — non-recursive sanity check.

        Parameters
        ----------
        window_days : int
            Width of the centred minimum window. HYSEP uses 2(0.2·A^0.2) days
            (A in mi²). For a typical small Alpine catchment use 5; for larger
            lowland catchments 7–11.
        """
        if window_days < 1 or window_days % 2 == 0:
            raise ValueError("window_days must be a positive odd integer")

        y = self.Q.copy()
        # Rolling min with centred window
        rmin = y.rolling(window_days, center=True, min_periods=1).min()
        # Identify local minima: cells where rolling-min equals current value
        is_local_min = (y == rmin) & y.notna()
        # Linear interpolation between local minima
        baseflow = y.where(is_local_min).interpolate(method='linear', limit_direction='both')
        # Cap at total flow
        baseflow = baseflow.where(baseflow.isna() | (baseflow <= y), y)
        baseflow.name = 'baseflow_sliding_min'
        return baseflow

    #---------------------------------------------------------------------------------
    # Ensemble wrapper + summaries
    #---------------------------------------------------------------------------------

    def separate_all(
        self,
        eckhardt_kwargs: Optional[dict] = None,
        lh_kwargs: Optional[dict] = None,
        sm_kwargs: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        Return a DataFrame with one column per method.

        Columns: ['eckhardt', 'lyne_hollick', 'sliding_min', 'Q'].
        """
        ekw = eckhardt_kwargs or {}
        lkw = lh_kwargs or {}
        skw = sm_kwargs or {}
        return pd.DataFrame({
            'Q':            self.Q,
            'eckhardt':     self.eckhardt(**ekw),
            'lyne_hollick': self.lyne_hollick(**lkw),
            'sliding_min':  self.sliding_minimum(**skw),
        })

    def bfi(self, method: Union[str, pd.Series] = 'eckhardt', **kwargs) -> float:
        """
        Long-term baseflow index = sum(baseflow) / sum(Q) over the full record.
        """
        if isinstance(method, pd.Series):
            b = method
        elif method == 'eckhardt':
            b = self.eckhardt(**kwargs)
        elif method == 'lyne_hollick':
            b = self.lyne_hollick(**kwargs)
        elif method in ('sliding_min', 'sliding_minimum'):
            b = self.sliding_minimum(**kwargs)
        else:
            raise ValueError(f"Unknown method '{method}'")
        valid = self.Q.notna() & b.notna()
        if not valid.any():
            return float('nan')
        return float(b[valid].sum() / self.Q[valid].sum())


    #---------------------------------------------------------------------------------
    # Plotting helper
    #---------------------------------------------------------------------------------

    def plot_diagnostics(
        self,
        save_path: Optional[Union[str, Path]] = None,
        eckhardt_kwargs: Optional[dict] = None,
        title_suffix: str = '',
    ) -> 'matplotlib.figure.Figure':
        """
        4-panel diagnostic: ensemble hydrograph (sample year), full-record
        time series (log-y), monthly BFI cycle, method-vs-method scatter.
        """
        import matplotlib.pyplot as plt

        ekw = eckhardt_kwargs or {}
        df = self.separate_all(eckhardt_kwargs=ekw)
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))

        # 1. Sample year — pick the year with the largest peak so the eye sees the methods diverge during storms
        year_of_max = df['Q'].idxmax().year
        sub = df[df.index.year == year_of_max]
        axes[0, 0].fill_between(sub.index, 0, sub['Q'], color='#cccccc', alpha=0.5, label='Total Q')
        axes[0, 0].plot(sub.index, sub['eckhardt'], color='#1f77b4', linewidth=1.4, label='Eckhardt')
        axes[0, 0].plot(sub.index, sub['lyne_hollick'], color='#d62728', linewidth=1.0, alpha=0.8, label='Lyne-Hollick')
        axes[0, 0].plot(sub.index, sub['sliding_min'], color='#2ca02c', linewidth=1.0, alpha=0.8, label='Sliding-min')
        axes[0, 0].set_title(f'Ensemble baseflow — sample year {year_of_max}')
        axes[0, 0].legend(fontsize=8, loc='upper right')
        axes[0, 0].set_ylabel('Q')
        axes[0, 0].grid(alpha=0.3)

        # 2. Full record log-y
        axes[0, 1].plot(df.index, df['Q'], color='#cccccc', linewidth=0.3, label='Total Q')
        axes[0, 1].plot(df.index, df['eckhardt'], color='#1f77b4', linewidth=0.5, alpha=0.8, label='Eckhardt')
        axes[0, 1].set_yscale('log')
        axes[0, 1].set_title('Full record (log-y, Eckhardt only)')
        axes[0, 1].set_ylabel('Q (log)')
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(alpha=0.3)

        # 3. Monthly BFI cycle, all methods
        month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        df_m = df.copy()
        df_m['month'] = df_m.index.month
        for col, color in (('eckhardt', '#1f77b4'),
                           ('lyne_hollick', '#d62728'),
                           ('sliding_min', '#2ca02c')):
            monthly_bfi = df_m.groupby('month').apply(
                lambda g, c=col: g[c].sum() / g['Q'].sum() if g['Q'].sum() > 0 else float('nan')
            )
            axes[1, 0].plot(range(1, 13), monthly_bfi.values, 'o-', label=col, color=color)
        axes[1, 0].set_xticks(range(1, 13))
        axes[1, 0].set_xticklabels(month_names)
        axes[1, 0].set_ylabel('BFI')
        axes[1, 0].set_title('Monthly BFI — all methods')
        axes[1, 0].legend(fontsize=8)
        axes[1, 0].grid(alpha=0.3)
        axes[1, 0].set_ylim(0, 1)

        # 4. Method scatter: Eckhardt vs Lyne-Hollick
        valid = df[['eckhardt', 'lyne_hollick']].dropna()
        axes[1, 1].scatter(valid['eckhardt'], valid['lyne_hollick'],
                           s=2, alpha=0.3, color='#444444')
        mx = float(valid.values.max())
        axes[1, 1].plot([0, mx], [0, mx], 'k--', linewidth=0.8, label='1:1')
        axes[1, 1].set_xlabel('Eckhardt baseflow')
        axes[1, 1].set_ylabel('Lyne-Hollick baseflow')
        axes[1, 1].set_title('Method consistency (Eckhardt vs LH)')
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].grid(alpha=0.3)

        suptitle = f'Baseflow separation diagnostics — {self.name}'
        if title_suffix:
            suptitle += f' — {title_suffix}'
        fig.suptitle(suptitle, fontsize=13, fontweight='bold')
        fig.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        return fig


#--------------------------------------------------------------------------------
############################# Validation against Raven ##########################
#--------------------------------------------------------------------------------

def load_raven_simulated_Q(
    output_dir: Union[str, Path],
    gauge_id: str,
) -> pd.Series:
    """
    Load simulated streamflow from a Raven Hydrographs.csv.

    The column name is "<gauge_id> [m3/s]" (Raven's convention).

    Parameters
    ----------
    output_dir : Path  — Raven output dir (contains *_Hydrographs.csv).
    gauge_id   : str   — the subbasin / gauge id used in the .rvh, e.g. '2268'.

    Returns
    -------
    pd.Series indexed by date, m³/s.
    """
    output_dir = Path(output_dir)
    matches = sorted(output_dir.glob('*_Hydrographs.csv'))
    if not matches:
        raise FileNotFoundError(f"No *_Hydrographs.csv in {output_dir}")
    df = pd.read_csv(matches[0])
    df['date'] = pd.to_datetime(df['date'])
    col_sim = f'{gauge_id} [m3/s]'
    if col_sim not in df.columns:
        raise KeyError(
            f"Column '{col_sim}' not found in {matches[0].name}; "
            f"available: {list(df.columns)}"
        )
    return df.set_index('date')[col_sim].astype(float).rename('Q_sim')


def load_raven_subsurface_fluxes(
    output_dir: Union[str, Path],
    area_km2: float,
    catchment_prefix: str = '',
) -> pd.DataFrame:
    """
    Load Raven's CustomOutput BETWEEN_*_AND_SURFACE_WATER CSV files and convert
    from mm/day at watershed scale to m³/s.

    Expects two files in `output_dir`:
        <prefix>_BETWEEN_SOIL[2]_AND_SURFACE_WATER_Daily_Average_ByWatershed.csv  (slow baseflow)
        <prefix>_BETWEEN_SOIL[1]_AND_SURFACE_WATER_Daily_Average_ByWatershed.csv  (interflow)

    For gw_1_layer, the single reservoir is SOIL[1] (no SOIL[2] file).

    Parameters
    ----------
    output_dir : Path
        Raven output directory (the one containing Hydrographs.csv etc).
    area_km2 : float
        Catchment area in km², used for mm/day -> m³/s conversion.
    catchment_prefix : str
        Optional file prefix, e.g. '2268_HBV'. If empty, auto-discovers.

    Returns
    -------
    DataFrame indexed by date with columns
    {slow_mm_day, fast_mm_day, slow_m3s, fast_m3s, total_subsurface_m3s}
    — slow_* are NaN for gw_1_layer (no separate slow reservoir).
    """
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Raven output dir not found: {output_dir}")

    # mm/day at watershed scale -> m³/s:
    #   mm/day × 1e-3 m/mm × A_km2 × 1e6 m²/km² / 86400 s/day = m³/s
    mmday_to_m3s = area_km2 * 1000.0 / 86400.0   # 39.22 km² -> 0.454

    def _load_one(soil_idx: int) -> Optional[pd.Series]:
        # Filename literally contains "SOIL[N]" — escape glob brackets by
        # listing all CSVs and filtering by substring instead.
        needle = f"BETWEEN_SOIL[{soil_idx}]_AND_SURFACE_WATER"
        matches = sorted(p for p in output_dir.glob('*.csv') if needle in p.name)
        if not matches:
            return None
        f = matches[0]
        # Raven CSV: row 1 = junk header, row 2 = column names, row 3+ = data
        df = pd.read_csv(f, skiprows=1)
        # Defensive: keep only time/day/mean columns
        df = df[['time', 'day', 'mean']]
        df['date'] = pd.to_datetime(df['day'])
        return df.set_index('date')['mean'].astype(float)

    slow_mm = _load_one(2)
    fast_mm = _load_one(1)
    if slow_mm is None and fast_mm is None:
        raise FileNotFoundError(
            f"No BETWEEN_*_AND_SURFACE_WATER files in {output_dir}. "
            "Did the .rvi include the baseflow :CustomOutput directives?"
        )

    out = pd.DataFrame({
        'slow_mm_day': slow_mm,
        'fast_mm_day': fast_mm,
    })
    out['slow_m3s'] = out['slow_mm_day'] * mmday_to_m3s
    out['fast_m3s'] = out['fast_mm_day'] * mmday_to_m3s
    out['total_subsurface_m3s'] = out[['slow_m3s', 'fast_m3s']].sum(axis=1, min_count=1)
    return out


#--------------------------------------------------------------------------------
########################### BaseflowValidator class #############################
#--------------------------------------------------------------------------------

class BaseflowValidator:
    """
    Compare baseflow components extracted from observed and simulated discharge.

    The same separation method is applied to both Q_obs and Q_sim, giving a
    consistent filter-method comparison. This validates whether the model
    produces a discharge signal whose *baseflow signature* matches obs — the
    standard approach in the literature.

    A secondary "raven_flux" mode also loads Raven's :CustomOutput Between-
    flux CSVs for inspection of internal HBV state, but the units / aggregation
    of those outputs can be quirky — prefer the filter-on-both approach.
    """

    def __init__(
        self,
        Q_obs: pd.Series,
        Q_sim: pd.Series,
        name: str = '',
    ) -> None:
        self.Q_obs = Q_obs.astype(float).copy()
        self.Q_sim = Q_sim.astype(float).copy()
        self.name = name
        self.obs_separator = BaseflowSeparator(self.Q_obs, name=f'{name} obs')
        self.sim_separator = BaseflowSeparator(self.Q_sim, name=f'{name} sim')

    @classmethod
    def from_raven_output(
        cls,
        Q_obs: pd.Series,
        raven_output_dir: Union[str, Path],
        gauge_id: str,
        name: str = '',
    ) -> 'BaseflowValidator':
        """Convenience: load Q_sim from Raven's Hydrographs.csv automatically."""
        Q_sim = load_raven_simulated_Q(raven_output_dir, gauge_id)
        return cls(Q_obs=Q_obs, Q_sim=Q_sim, name=name)

    #---------------------------------------------------------------------------------

    def aligned(
        self,
        method: str = 'eckhardt',
        eckhardt_kwargs: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        Return a date-aligned DataFrame for the overlap period:
        columns {Q_obs, Q_sim, baseflow_obs, baseflow_sim} all in m³/s.

        Both observed and simulated Q are passed through the SAME separator.
        Eckhardt parameters are auto-estimated separately for obs and sim
        (different recession constants are normal — the model may over/under
        damp).
        """
        ekw = eckhardt_kwargs or {}
        if method == 'eckhardt':
            obs_bf = self.obs_separator.eckhardt(**ekw)
            sim_bf = self.sim_separator.eckhardt(**ekw)
        elif method == 'lyne_hollick':
            obs_bf = self.obs_separator.lyne_hollick()
            sim_bf = self.sim_separator.lyne_hollick()
        elif method in ('sliding_min', 'sliding_minimum'):
            obs_bf = self.obs_separator.sliding_minimum()
            sim_bf = self.sim_separator.sliding_minimum()
        elif method == 'raw_winter':
            # No filter — Q_winter is taken to be 100% baseflow by mass balance.
            # Only valid for catchments where winter inputs (rain + melt) ≈ 0.
            obs_bf = self.Q_obs.copy()
            sim_bf = self.Q_sim.copy()
        else:
            raise ValueError(f"Unknown separation method '{method}'")

        df = pd.DataFrame({
            'Q_obs':         self.Q_obs,
            'Q_sim':         self.Q_sim,
            'baseflow_obs':  obs_bf,
            'baseflow_sim':  sim_bf,
        }).dropna()
        return df

    #---------------------------------------------------------------------------------

    # Winter months for filter-based metrics — Nov through March.  Restricting
    # to winter avoids melt-water being misclassified as baseflow by the filter.
    WINTER_MONTHS = (11, 12, 1, 2, 3)

    # Pure-baseflow window for raw_winter mode — Dec through March.  Skip Nov
    # to let autumn channel-storage and fast-reservoir residuals fully drain
    # before assuming Q == baseflow.  For high-altitude truly-frozen catchments
    # only — assumes no liquid inputs (rain or melt) reach the catchment.
    RAW_WINTER_MONTHS = (12, 1, 2, 3)

    def metrics(
        self,
        method: str = 'eckhardt',
        eckhardt_kwargs: Optional[dict] = None,
    ) -> dict:
        """
        Compute baseflow-comparison metrics on the **winter-only** overlap.

        Filters are run on the full record (so they have continuous data for
        the recursive math), but the metrics aggregate only over Nov–Mar to
        avoid melt-water being misclassified as baseflow.
        """
        df = self.aligned(method=method, eckhardt_kwargs=eckhardt_kwargs)
        # Restrict to the right winter window. raw_winter uses a tighter
        # Dec-Mar window to avoid autumn-drainage residuals; the filter-based
        # methods use the wider Nov-Mar window.
        win = self.RAW_WINTER_MONTHS if method == 'raw_winter' else self.WINTER_MONTHS
        df = df[df.index.month.isin(win)]
        o = df['baseflow_obs'].values
        s = df['baseflow_sim'].values
        Q = df['Q_obs'].values

        # Drop any remaining bad rows
        valid = np.isfinite(o) & np.isfinite(s) & np.isfinite(Q) & (Q > 0)
        o, s, Q = o[valid], s[valid], Q[valid]
        if len(o) < 30:
            return {}

        # Long-term BFI
        bfi_obs = float(o.sum() / Q.sum())
        bfi_sim = float(s.sum() / Q.sum())
        bfi_bias = bfi_sim - bfi_obs

        # NSE on baseflow
        nse_bf = 1.0 - float(((s - o) ** 2).sum() / ((o - o.mean()) ** 2).sum())

        # log-NSE on baseflow (focus on low flows)
        # Use a small offset to avoid log(0)
        eps = max(o[o > 0].min() * 0.01 if (o > 0).any() else 1e-6, 1e-9)
        log_o = np.log(o + eps)
        log_s = np.log(s + eps)
        nse_log = 1.0 - float(((log_s - log_o) ** 2).sum() / ((log_o - log_o.mean()) ** 2).sum())

        # KGE on baseflow (Kling-Gupta 2009)
        r = float(np.corrcoef(o, s)[0, 1])
        alpha = float(s.std() / o.std()) if o.std() > 0 else float('nan')
        beta = float(s.mean() / o.mean()) if o.mean() > 0 else float('nan')
        kge_bf = 1.0 - float(np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))

        return {
            'sep_method': method,
            'window':     ('winter Dec-Mar (Q ≡ baseflow assumption)'
                           if method == 'raw_winter' else 'winter Nov-Mar'),
            'n':         int(len(o)),
            'BFI_obs':   bfi_obs,
            'BFI_sim':   bfi_sim,
            'BFI_bias':  bfi_bias,
            'NSE_bf':    nse_bf,
            'logNSE_bf': nse_log,
            'KGE_bf':    kge_bf,
            'r':         r,
            'alpha':     alpha,
            'beta':      beta,
        }

    #---------------------------------------------------------------------------------

    def plot_validation(
        self,
        save_path: Optional[Union[str, Path]] = None,
        method: str = 'eckhardt',
        eckhardt_kwargs: Optional[dict] = None,
    ) -> 'matplotlib.figure.Figure':
        """
        4-panel comparison: timeseries overlay, scatter, monthly BFI cycle,
        metrics table.
        """
        import matplotlib.pyplot as plt
        df = self.aligned(method=method, eckhardt_kwargs=eckhardt_kwargs)
        m = self.metrics(method=method, eckhardt_kwargs=eckhardt_kwargs)
        if df.empty or not m:
            raise RuntimeError("No overlapping valid data between obs and sim")

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))

        # 1. Timeseries: Q + obs baseflow + sim baseflow (winter shaded)
        axes[0, 0].fill_between(df.index, 0, df['Q_obs'],
                                color='#cccccc', alpha=0.5, label='Q obs')
        # raw_winter mode: baseflow_obs == Q_obs and sim == Q_sim, so the
        # filter overlay would just duplicate the Q line — relabel for clarity.
        if method == 'raw_winter':
            obs_label = 'Q obs winter (≡ baseflow)'
            sim_label = 'Q sim winter (≡ baseflow)'
        else:
            obs_label = f'Baseflow obs ({method})'
            sim_label = f'Baseflow sim ({method})'
        axes[0, 0].plot(df.index, df['baseflow_obs'], color='#1f77b4',
                        linewidth=1.2, label=obs_label)
        axes[0, 0].plot(df.index, df['baseflow_sim'], color='#d62728',
                        linewidth=1.2, label=sim_label)
        # Shade the metric window (Dec-Mar for raw_winter, Nov-Mar otherwise)
        win = self.RAW_WINTER_MONTHS if method == 'raw_winter' else self.WINTER_MONTHS
        winter_mask = pd.Series(df.index.month.isin(win), index=df.index)
        # Find contiguous winter runs and shade them
        in_winter = False
        start = None
        for t, w in winter_mask.items():
            if w and not in_winter:
                start = t
                in_winter = True
            elif not w and in_winter:
                axes[0, 0].axvspan(start, t, color='#9ecae1', alpha=0.15, zorder=0)
                in_winter = False
        if in_winter:
            axes[0, 0].axvspan(start, winter_mask.index[-1],
                               color='#9ecae1', alpha=0.15, zorder=0)
        axes[0, 0].set_title('Obs vs sim baseflow (blue shading = winter metric window)')
        axes[0, 0].set_ylabel('Q (m³/s)')
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(alpha=0.3)

        # 2. Scatter
        axes[0, 1].scatter(df['baseflow_obs'], df['baseflow_sim'],
                           s=3, alpha=0.3, color='#444444')
        mx = max(df['baseflow_obs'].max(), df['baseflow_sim'].max())
        axes[0, 1].plot([0, mx], [0, mx], 'k--', linewidth=0.8, label='1:1')
        axes[0, 1].set_xlabel(obs_label.replace(' winter (≡ baseflow)', '') if method == 'raw_winter' else f'Baseflow obs ({method})')
        axes[0, 1].set_ylabel(sim_label.replace(' winter (≡ baseflow)', '') if method == 'raw_winter' else f'Baseflow sim ({method})')
        axes[0, 1].set_title('Scatter')
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(alpha=0.3)

        # 3. Monthly BFI cycle obs vs sim
        df_m = df.copy()
        df_m['month'] = df_m.index.month
        month_bfi_obs = df_m.groupby('month').apply(lambda g: g['baseflow_obs'].sum() / g['Q_obs'].sum())
        month_bfi_sim = df_m.groupby('month').apply(lambda g: g['baseflow_sim'].sum() / g['Q_obs'].sum())
        axes[1, 0].plot(range(1, 13), month_bfi_obs.values, 'o-',
                        color='#1f77b4', label='obs')
        axes[1, 0].plot(range(1, 13), month_bfi_sim.values, 's-',
                        color='#d62728', label='sim')
        # Shade the metric window in this panel too
        win_for_shading = self.RAW_WINTER_MONTHS if method == 'raw_winter' else self.WINTER_MONTHS
        for wm in win_for_shading:
            axes[1, 0].axvspan(wm - 0.5, wm + 0.5, color='#9ecae1', alpha=0.15, zorder=0)
        axes[1, 0].set_xticks(range(1, 13))
        axes[1, 0].set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
        axes[1, 0].set_ylabel('Monthly BFI')
        axes[1, 0].set_title('Monthly BFI — blue: filter trustworthy; white: melt-contaminated')
        axes[1, 0].legend(fontsize=8)
        axes[1, 0].grid(alpha=0.3)
        axes[1, 0].set_ylim(0, 1)

        # 4. Metrics table
        axes[1, 1].axis('off')
        rows = [
            ('Period (n days)',  f"{m['n']}"),
            ('BFI obs',          f"{m['BFI_obs']:.3f}"),
            ('BFI sim',          f"{m['BFI_sim']:.3f}"),
            ('BFI bias',         f"{m['BFI_bias']:+.3f}"),
            ('NSE baseflow',     f"{m['NSE_bf']:.3f}"),
            ('logNSE baseflow',  f"{m['logNSE_bf']:.3f}"),
            ('KGE baseflow',     f"{m['KGE_bf']:.3f}"),
            ('  r',              f"{m['r']:.3f}"),
            ('  α (sd ratio)',   f"{m['alpha']:.3f}"),
            ('  β (mean ratio)', f"{m['beta']:.3f}"),
        ]
        ytable = axes[1, 1].table(
            cellText=[[k, v] for k, v in rows],
            loc='center', cellLoc='left', colWidths=[0.55, 0.3],
        )
        ytable.auto_set_font_size(False)
        ytable.set_fontsize(10)
        ytable.scale(1, 1.4)
        win_label = 'Dec-Mar, Q ≡ baseflow' if method == 'raw_winter' else 'Nov-Mar'
        axes[1, 1].set_title(f'Metrics  ({method}, {win_label})')

        suptitle = f'Baseflow validation — {self.name}' if self.name else 'Baseflow validation'
        fig.suptitle(suptitle, fontsize=13, fontweight='bold')
        fig.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        return fig


#--------------------------------------------------------------------------------
############################# Convenience function ##############################
#--------------------------------------------------------------------------------

def separate_baseflow(
    Q: pd.Series,
    method: str = 'eckhardt',
    **kwargs,
) -> pd.Series:
    """
    One-shot baseflow separation. Returns a Series.

    Parameters
    ----------
    Q : pd.Series
        Daily discharge.
    method : str
        One of 'eckhardt', 'lyne_hollick', 'sliding_min'.
    **kwargs : passed to the underlying method.
    """
    sep = BaseflowSeparator(Q)
    if method == 'eckhardt':
        return sep.eckhardt(**kwargs)
    if method == 'lyne_hollick':
        return sep.lyne_hollick(**kwargs)
    if method in ('sliding_min', 'sliding_minimum'):
        return sep.sliding_minimum(**kwargs)
    raise ValueError(f"Unknown method '{method}'")
