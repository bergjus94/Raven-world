"""
Paper 5 — shared constants and helpers.

Single source of truth for:
  - Subsurface structure metadata (S1-S9 / LN-OF naming, config keys,
    calibrated parameter sets, axis labels)
  - Per-objective metric display labels (KGE_NP, nRMSE)
  - Catchment metadata (ID → name, region, area)
  - Hand-curated parameter descriptions (symbol, description, units)

Import this module from any paper-5 plotting / analysis script so naming
and metric labels stay consistent across the suite. Updating naming or
adding a structure happens here, once.

Naming convention [LOCKED 2026-06-03, paper_5_decisions.md §3]:
  Two-letter code [architecture][connection]:
    Architecture axis (A):  L=Lumped HBV-linear, T=Threshold (HBV-Light Q0+Q1),
                            O=Overland (SPHY-faithful direct routing)
    Connection axis (B):    N=None, S=Slow store, F=Fast store
"""
from __future__ import annotations

# ── Objective metric display labels ─────────────────────────────────────────
# Used for axis labels, plot titles, and table headers across all paper-5
# analysis scripts. Update here when metrics change.
METRIC_LABELS = {
    'obj_Q':        'Q (KGE_NP)',
    'obj_snow':     'fSCA (nRMSE)',
    'obj_baseflow': 'baseflow (KGE_NP)',
}

# Short forms for compact plot legends / table headers.
METRIC_SHORT = {
    'obj_Q':        'Q',
    'obj_snow':     'snow',
    'obj_baseflow': 'baseflow',
}


# ── Subsurface structure metadata ───────────────────────────────────────────
# Order is the canonical S1→S9 sequence used in the methodology doc.
# When iterating in display order, use this list — newer two-letter codes
# (LN/LS/LF, TN/TS/TF, ON/OS/OF) are also exposed via STRUCTURE_INFO.
STRUCTURE_ORDER = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9']

# Axis A (architecture) labels
ARCHITECTURE = {
    'L': ('Lumped',    'HBV-linear, single K1 release from FAST_RES'),
    'T': ('Threshold', 'HBV-Light Q0+Q1 (Seibert & Vis 2012) — adds BASE_THRESH_STOR above UZL'),
    'O': ('Overland',  'SPHY-style (Terink 2015) — direct overland routing + cascade percolation'),
}

# Axis B (glacier-GW connection) labels
CONNECTION = {
    'N': ('None',  'no :Split; glacier melt enters as PONDED_WATER'),
    'S': ('Slow',  ':Split target SLOW_RESERVOIR (deep GW)'),
    'F': ('Fast',  ':Split target FAST_RESERVOIR (upper store)'),
}

# Full structure registry. Each entry has:
#   s_code        : legacy 'S1'..'S9'
#   two_letter    : new 'LN', 'LS', ..., 'OF'
#   architecture  : 'L', 'T', or 'O'
#   connection    : 'N', 'S', or 'F'
#   config_key    : src/config/layers/configurations/<key>.yaml
#   params        : list of calibrated X-numbers (after gating)
#   description   : one-line text for plot titles
STRUCTURE_INFO = {
    'S1': {
        's_code': 'S1', 'two_letter': 'LN',
        'architecture': 'L', 'connection': 'N',
        'config_key': 'glogem_subdaily_opt1',
        'params': ['X01', 'X02', 'X03', 'X04', 'X05', 'X06', 'X07', 'X08',
                   'X09', 'X10', 'X11', 'X15'],
        'description': 'Lumped baseline (HBV-linear, no glacier-GW)',
    },
    'S2': {
        's_code': 'S2', 'two_letter': 'LS',
        'architecture': 'L', 'connection': 'S',
        'config_key': 'glogem_subdaily_opt1_glaciergw',
        'params': ['X01', 'X02', 'X03', 'X04', 'X05', 'X06', 'X07', 'X08',
                   'X09', 'X10', 'X11', 'X15', 'X16'],
        'description': 'Lumped + glacier→SLOW',
    },
    'S3': {
        's_code': 'S3', 'two_letter': 'TN',
        'architecture': 'T', 'connection': 'N',
        'config_key': 'glogem_subdaily_opt1_threshold',
        'params': ['X01', 'X02', 'X03', 'X04', 'X05', 'X06', 'X07', 'X08',
                   'X09', 'X10', 'X11', 'X15', 'X17', 'X18'],
        'description': 'Threshold (HBV-Light Q0+Q1), no glacier-GW',
    },
    'S4': {
        's_code': 'S4', 'two_letter': 'TS',
        'architecture': 'T', 'connection': 'S',
        'config_key': 'glogem_subdaily_opt1_threshold_glaciergw',
        'params': ['X01', 'X02', 'X03', 'X04', 'X05', 'X06', 'X07', 'X08',
                   'X09', 'X10', 'X11', 'X15', 'X16', 'X17', 'X18'],
        'description': 'Threshold + glacier→SLOW',
    },
    'S5': {
        's_code': 'S5', 'two_letter': 'ON',
        'architecture': 'O', 'connection': 'N',
        'config_key': 'glogem_subdaily_opt2_sphy_faithful',
        'params': ['X01', 'X02', 'X03', 'X04', 'X05', 'X06', 'X07', 'X08',
                   'X09', 'X10', 'X12', 'X13', 'X14', 'X15'],
        'description': 'Overland (SPHY-style cascade percolation), no glacier-GW',
    },
    'S6': {
        's_code': 'S6', 'two_letter': 'OS',
        'architecture': 'O', 'connection': 'S',
        'config_key': 'glogem_subdaily_opt2_sphy_faithful_glaciergw',
        'params': ['X01', 'X02', 'X03', 'X04', 'X05', 'X06', 'X07', 'X08',
                   'X09', 'X10', 'X12', 'X13', 'X14', 'X15', 'X16'],
        'description': 'Overland + glacier→SLOW',
    },
    'S7': {
        's_code': 'S7', 'two_letter': 'LF',
        'architecture': 'L', 'connection': 'F',
        'config_key': 'glogem_subdaily_opt1_glaciergw_fast',
        'params': ['X01', 'X02', 'X03', 'X04', 'X05', 'X06', 'X07', 'X08',
                   'X09', 'X10', 'X11', 'X15', 'X16'],
        'description': 'Lumped + glacier→FAST (S2 mirror)',
    },
    'S8': {
        's_code': 'S8', 'two_letter': 'TF',
        'architecture': 'T', 'connection': 'F',
        'config_key': 'glogem_subdaily_opt1_threshold_glaciergw_fast',
        'params': ['X01', 'X02', 'X03', 'X04', 'X05', 'X06', 'X07', 'X08',
                   'X09', 'X10', 'X11', 'X15', 'X16', 'X17', 'X18'],
        'description': 'Threshold + glacier→FAST (S4 mirror)',
    },
    'S9': {
        's_code': 'S9', 'two_letter': 'OF',
        'architecture': 'O', 'connection': 'F',
        'config_key': 'glogem_subdaily_opt2_sphy_faithful_glaciergw_fast',
        'params': ['X01', 'X02', 'X03', 'X04', 'X05', 'X06', 'X07', 'X08',
                   'X09', 'X10', 'X12', 'X13', 'X14', 'X15', 'X16'],
        'description': 'Overland + glacier→FAST (S6 mirror)',
    },
}

# Quick lookup: config_key → S-code (used by scripts that read NSGAII
# CSVs with directory names like 'glogem_subdaily_opt1_glaciergw').
CONFIG_TO_S_CODE = {v['config_key']: v['s_code'] for v in STRUCTURE_INFO.values()}

# Quick lookup: config_key → two-letter code
CONFIG_TO_LABEL = {v['config_key']: v['two_letter'] for v in STRUCTURE_INFO.values()}


def s_to_label(s_code: str) -> str:
    """Return the two-letter display label (e.g. 'S1' → 'LN')."""
    return STRUCTURE_INFO[s_code]['two_letter']


def label_to_s(two_letter: str) -> str:
    """Return the legacy S-code (e.g. 'LN' → 'S1')."""
    for s, info in STRUCTURE_INFO.items():
        if info['two_letter'] == two_letter:
            return s
    raise KeyError(f'Unknown structure label: {two_letter!r}')


def config_to_label(config_key: str) -> str:
    """Return two-letter label for a config-layer key. Falls back to the
    key itself if not recognised (defensive)."""
    return CONFIG_TO_LABEL.get(config_key, config_key)


# ── Hand-curated parameter descriptions ─────────────────────────────────────
# (symbol_latex, description, units) per X-number. Used by parameter-table
# figures and Morris/PAWN plots that label parameters by their physical role.
PARAM_INFO = {
    'X01': ('T_{rs}',     'Rain/snow transition temperature',                 '°C'),
    'X02': ('M_f',        'Snow melt factor (degree-day)',                    'mm·d⁻¹·°C⁻¹'),
    'X03': ('CFR',        'Refreeze coupling ratio',                          '–'),
    'X04': ('SWI',        'Snow water-holding fraction',                      '–'),
    'X05': ('β',          'HBV β infiltration exponent',                      '–'),
    'X06': ('FC',         'Field capacity (fraction of porosity)',            '–'),
    'X07': ('K_1',        'FAST_RES linear baseflow coefficient',             'd⁻¹'),
    'X08': ('K_2',        'SLOW_RES linear baseflow coefficient',             'd⁻¹'),
    'X09': ('T_c',        'Time of concentration',                            'd'),
    'X10': ('h_{topsoil}','Topsoil layer thickness',                          'm'),
    'X11': ('P_{fast}',   'PERC_CONSTANT FAST→SLOW peak rate',                'mm·d⁻¹'),
    'X12': ('P_{tops}',   'PERC_POWER_LAW TOPSOIL→FAST peak rate',            'mm·d⁻¹'),
    'X13': ('n',          'PERC_POWER_LAW exponent on TOPSOIL',               '–'),
    'X14': ('K_{perc}',   'PERC_LINEAR FAST→SLOW coefficient',                'd⁻¹'),
    'X15': ('CR',         'Capillary rise rate (FAST→TOPSOIL)',               'mm·d⁻¹'),
    'X16': ('GlacROF',    'Glacier-melt surface routing fraction',            '–'),
    'X17': ('UZL',        'FAST_RES storage threshold for Q0',                'mm'),
    'X18': ('K_0',        'FAST_RES above-threshold release rate',            'd⁻¹'),
}


# ── Catchment metadata (display) ─────────────────────────────────────────────
# ID → (display name, region, area km², forcing).
CATCHMENT_INFO = {
    '0102': ('Hunza @ Dainyor',                'UIB monsoon-influenced',         14000, 'TPHiPr'),
    '0130': ('Chenab @ Tandi',                 'UIB lower-elevation monsoon',    22000, 'TPHiPr'),
    '2161': ('Massa @ Blatten',                'Swiss Alpine — Valais',          412,   'MeteoSwiss'),
    '2200': ('Weisse Lütschine @ Zweilütschinen','Swiss Alpine — Bernese',       349,   'MeteoSwiss'),
    '2219': ('Simme @ Oberried-Lenk',          'Swiss Alpine — Bernese',         73,    'MeteoSwiss'),
    '2256': ('Rosegbach @ Pontresina',         'Swiss Alpine — Engadine',        140,   'MeteoSwiss'),
    '2268': ('Rhone @ Gletsch',                'Swiss Alpine — Valais',          84,    'MeteoSwiss'),
    '2269': ('Lonza @ Blatten',                'Swiss Alpine — Lötschental',     163,   'MeteoSwiss'),
}

# Display order: UIB first, then Swiss in ascending ID
CATCHMENT_ORDER = ['0102', '0130', '2161', '2200', '2219', '2256', '2268', '2269']


def catchment_label(catchment_id: str, fmt: str = 'short') -> str:
    """Return a display label for a catchment.

    fmt='short'    → 'Hunza' / 'Chenab' / 'Rhone' (first word of display name)
    fmt='medium'   → 'Hunza @ Dainyor'  (full display name)
    fmt='long'     → 'Hunza @ Dainyor (14000 km²)'
    """
    info = CATCHMENT_INFO.get(catchment_id)
    if not info:
        return catchment_id
    name, region, area, forcing = info
    if fmt == 'short':
        return name.split(' @ ')[0].split(' ')[0]
    if fmt == 'medium':
        return name
    if fmt == 'long':
        return f'{name} ({area:,} km²)'
    return name
