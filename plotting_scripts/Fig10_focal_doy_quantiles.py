"""
Fig10: Focal-event dynamics + FFMP performance distributions (3x2 grid).

Three rows pair operational dynamics (left) with FFMP performance outcome
distributions (right):

Row 1: NYC combined storage (%)         | Min daily NYC max-allowable diversion CDF
Row 2: NYC actual diversion (MCM/day)   | Event-total shortage / event demand CDF
Row 3: Montague gauge flow (log)        | Min 7-day Montague flow CDF

Trajectory column (left): all three focal-event ensembles overlaid as
10/50/90 quantile lines (no fill); historical 1964/1980/2002 droughts as
solid colored lines (drawn from a reconstructed-streamflow Pywr-DRB
simulation); FFMP rule curves on the storage row.

CDF column (right): four empirical CDFs per panel — three ensemble focal-
event distributions plus a historic-baseline distribution constructed from
all complete water years in the reconstruction (1901-2020). Triangle
markers on each ensemble CDF show where 1964/1980/2002 sit within the
ensemble's empirical distribution.

The narrative argument: focal-region drought operational dynamics are
climate-invariant across the three ensembles (visible as overlapping
quantile bands in the left column), but cumulative performance outcomes
(FFMP cap depth, NYC shortage fraction, Montague low flow) vary in their
focal-event distributions. The historic-baseline CDF and the three
historical-drought anchors locate the focal region within the larger
historical context.

NYC max-allowable diversion is derived from the FFMP storage zone
(res_level['nyc']) and a baseline 800 MGD delivery target with FFMP
drought-stage cap factors (zones 4/5/6/7 -> 0.85/0.70/0.65/0.65). This
fallback omits the FFMP running-average cap (which is not a recorded
parameter in the postprocessed HDF5 outputs) and is therefore an upper
bound on the daily delivered cap.

Usage:
    python Fig10_focal_doy_quantiles.py
"""

import sys
import os
import gc
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

import pywrdrb
from methods.config import (
    FIG_DIR, OUTPUT_DIR, N_YEARS, NYC_RESERVOIRS, NYC_TOTAL_CAPACITY,
    RECONSTRUCTION_OUTPUT_FNAME, DATASET_CONFIGS,
    GRID_N_BINS, FOCAL_FRAC_THRESH, FOCAL_RP_THRESH_YEARS,
    FOCAL_WORST_STORAGE_THRESH,
)
from methods.water_year import (
    MONTH_STARTS_WY, vectorized_water_year_doy,
)
from methods.load import (
    load_event_metrics, load_rank_subset_from_export, load_ffmp_boundaries,
)
from methods.return_period import compute_return_period_grid_exceedance as compute_return_period_grid
from methods.plotting.heatmap import (
    make_shared_edges_logmag, assign_grid_bins,
    compute_emergency_grid, compute_min_storage_grid,
    identify_focal_region, select_events_from_focal_region,
)
from methods.plotting.drought_dynamics import compute_fixed_extraction_window
from methods.plotting.percentile_bands import (
    format_xaxis_water_year, plot_quantile_lines,
)
from methods.plotting.styles import (
    DATASET_LABELS_SHORT, DATASET_COLORS,
    FFMP_ZONE_COLORS,
    DPI_PRINT, FONTSIZE_LABEL, FONTSIZE_TITLE, FONTSIZE_LEGEND, FONTSIZE_SMALL,
    apply_publication_style, label_panel, save_fig,
)
from methods.plotting.ensemble_summary import MGD_TO_MCM


# ── Configuration ────────────────────────────────────────────────────────

SSI_WINDOW = 3
MIN_COUNT = 1

DATASETS = list(DATASET_CONFIGS.keys())

# All three ensembles get trajectory + CDF panels; this expression kept
# for symmetry with prior versions.
TRAJECTORY_DATASETS = list(DATASETS)

# ibt_diversions/ibt_demands give NYC delivery and demand; res_level['nyc']
# provides the FFMP storage zone needed to derive the max-allowable cap.
# 'contribution' provides total NYC release to the Montague flow target.
RESULTS_SETS = ['res_storage', 'major_flow', 'ibt_diversions',
                'ibt_demands', 'res_level', 'contribution']

TEST_MODE = False

STRICT_THRESHOLDS = dict(
    rp_thresh_years=FOCAL_RP_THRESH_YEARS,
    frac_thresh=FOCAL_FRAC_THRESH,
    storage_thresh=FOCAL_WORST_STORAGE_THRESH,
)
RELAXED_THRESHOLDS = dict(
    rp_thresh_years=np.inf,
    frac_thresh=2.0,
    storage_thresh=200.0,
)

REFERENCE_WY_START = pd.Timestamp('2000-06-01')
REFERENCE_WY_END = pd.Timestamp('2001-05-31')

# Historical-drought overlays drawn from the reconstructed-streamflow
# Pywr-DRB simulation. The water year (FFMP convention, Jun 1 – May 31)
# selected is the one containing the storage minimum of each colloquially-
# named drought, taken from the SSI3 reconstruction event metrics
# (reconstruction_ssi3_event_metrics.csv):
#   1960s drought (start 1964-06; min storage 1965-11-15) -> WY 1965/66
#   1980 drought  (start 1980-01; min storage 1980-11-23) -> WY 1980/81
#   2002 drought  (start 2000-11; min storage 2002-01-24) -> WY 2001/02
# These are then filtered by SSI3 focal-region membership; only events
# whose (severity, magnitude) bin is in `focal_cells` are kept.
HISTORICAL_OVERLAYS_CONFIG = {
    'h1964': dict(label='1964', wy_start=pd.Timestamp('1965-06-01'),
                  enabled=True),
    'h1980': dict(label='1980', wy_start=pd.Timestamp('1980-06-01'),
                  enabled=True),
    'h2002': dict(label='2002', wy_start=pd.Timestamp('2001-06-01'),
                  enabled=True),
}
HISTORICAL_OVERLAY_KEYS = [k for k, c in HISTORICAL_OVERLAYS_CONFIG.items()
                           if c['enabled']]

# FFMP NYC delivery cap derivation. The postprocessed HDF5 outputs do not
# expose pywrdrb's recorded `max_flow_delivery_nyc` parameter, so we
# reconstruct the drought-stage component of the cap from `res_level['nyc']`
# (integer 1-7) and the FFMP zone factors. This omits the FFMP running-
# average cap; the resulting series is an upper bound on the actually
# binding daily cap.
NYC_BASELINE_DELIVERY_MGD = 800.0
NYC_ZONE_DELIVERY_FACTORS = {
    1: 1.0,   # Flood
    2: 1.0,   # Flood
    3: 1.0,   # Normal
    4: 0.85,  # Drought Watch
    5: 0.70,  # Drought Warning
    6: 0.65,  # Drought Emergency
    7: 0.65,  # Drought Emergency severe sub-zone
}
NYC_BASELINE_DELIVERY_MCM = NYC_BASELINE_DELIVERY_MGD * MGD_TO_MCM

# FFMP Montague flow targets (MCM/day) — horizontal reference lines on the
# Montague-row CDF. Pywr-DRB ships these as constants:
#   mrf_baseline_delMontague        = 1131.05 MGD (≈ 1750 cfs)  Level 1-2
#   level5_factor_mrf_delMontague   = 0.771   (most restrictive monthly
#                                              factor, Oct-Jan; the deepest
#                                              FFMP drought-emergency cap)
# Source: Pywr-DRB/src/pywrdrb/data/operational_constants/
#   constants.csv + ffmp_reservoir_operation_monthly_profiles.csv.
# Note: the previously hardcoded "1100 cfs ≈ 2.69 MCM/day" Drought Emergency
# value did not match the FFMP rule curves shipped with pywrdrb.
MONTAGUE_BASELINE_MGD = 1131.05
MONTAGUE_DROUGHT_EMERGENCY_FACTOR = 0.771
MONTAGUE_TARGET_NORMAL_MCM = MONTAGUE_BASELINE_MGD * MGD_TO_MCM
MONTAGUE_TARGET_EMERGENCY_MCM = (
    MONTAGUE_BASELINE_MGD * MONTAGUE_DROUGHT_EMERGENCY_FACTOR * MGD_TO_MCM
)

# When True, historical-drought timeseries (only the in-focal-region ones,
# after SSI3 filtering) are drawn on the left trajectory panels. The
# right-column CDF bars are always drawn for the in-focal historicals.
SHOW_HISTORICAL_TRAJECTORIES = True

# DOY ticks for the July-Nov operational band (FFMP water year starts Jun 1
# = DOY 1; July 1 = DOY 31; Dec 1 = DOY 184). Dashed black vertical lines
# at these DOYs visually band the late-summer/fall drawdown period on each
# left-column trajectory panel.
JULY_NOV_BAND_DOYS = (31, 184)

# (variable_key, trajectory y-label, y-scale, rolling-mean window in days,
#  cdf scalar key, cdf y-label, list of (value, text-label) horizontal
#  reference thresholds drawn on the CDF panel)
# NYC FFMP cap levels (MCM/day) labeled on the max-allowable CDF panel.
NYC_CAP_LEVELS_MCM = [
    (1.00 * NYC_BASELINE_DELIVERY_MCM,
     f'Normal cap ({1.00 * NYC_BASELINE_DELIVERY_MCM:.2f} MCM/day)'),
    (0.85 * NYC_BASELINE_DELIVERY_MCM,
     f'Watch cap ({0.85 * NYC_BASELINE_DELIVERY_MCM:.2f} MCM/day)'),
    (0.70 * NYC_BASELINE_DELIVERY_MCM,
     f'Warning cap ({0.70 * NYC_BASELINE_DELIVERY_MCM:.2f} MCM/day)'),
    (0.65 * NYC_BASELINE_DELIVERY_MCM,
     f'Emergency cap ({0.65 * NYC_BASELINE_DELIVERY_MCM:.2f} MCM/day)'),
]

VARIABLES = [
    ('nyc_storage_pct',
     'Combined NYC storage\n(% of capacity)',                    'linear', 1,
     'event_min_max_allowable_diversion_mcm',
     'Event-min daily NYC\nmax-allowable diversion (MCM/day)',
     NYC_CAP_LEVELS_MCM),
    ('nyc_release',
     'NYC release to\nMontague target (MCM/day)',                'linear', 7,
     'event_demand_satisfaction_pct',
     'Event-total NYC demand\nsatisfaction (% of demand)',
     [(100.0, 'Demand fully met')]),
    ('montague_flow',
     'Montague gauge flow\n(MCM/day, log scale)',                'log',    7,
     'event_min_montague_mcm',
     'Event-min 7-day\nMontague flow (MCM/day)',
     []),
]

# The dashed outer trajectory line shows the "stressful" 2% tail, whose
# direction is variable-specific: low NYC storage and low Montague flow are
# the adverse outcomes (2nd percentile), whereas a high NYC release to the
# Montague target is the adverse outcome (98th percentile).
STRESSFUL_OUTER_QUANTILE = {
    'nyc_storage_pct': 0.02,
    'nyc_release': 0.98,
    'montague_flow': 0.02,
}

PANEL_LETTERS = list('abcdef')   # 3 rows x 2 cols
XAXIS_SUFFIX_LABEL = 'Water Year (Jun 1 - May 31, FFMP convention)'

FIG_OUTPUT_DIR = os.path.join(FIG_DIR, 'Fig10')
FIG_NAME_STEM = f'Fig10_focal_doy_quantiles_ssi{SSI_WINDOW}'
if TEST_MODE:
    FIG_NAME_STEM += '_TESTMODE'

CACHE_DIR = os.path.join(OUTPUT_DIR, 'cache')
CACHE_VERSION = 'v5'  # bump: added nyc_release variable + demand-satisfaction scalar
REBUILD_CACHE = False

# Historical drought line styling (saturated, distinct from ensemble colors).
HIST_STYLES = {
    'h1964': dict(color='#8e1c5c', linewidth=1.5, linestyle='-',  alpha=0.95, zorder=7.5),
    'h1980': dict(color='#107070', linewidth=1.5, linestyle='-',  alpha=0.95, zorder=7.5),
    'h2002': dict(color='#5e35b1', linewidth=1.5, linestyle='-',  alpha=0.95, zorder=7.5),
}
HIST_LABELS = {
    'h1964': '1964 drought of record (reconstructed)',
    'h1980': '1980 drought (reconstructed)',
    'h2002': '2002 drought (reconstructed)',
}


# ── Helpers ──────────────────────────────────────────────────────────────

def _traces_to_doy_df(traces_by_event):
    """Stack a list of (event_id, pd.Series) into a DOY-indexed DataFrame."""
    frames = {}
    for event_id, s in traces_by_event:
        if s is None or len(s) == 0:
            continue
        doy = vectorized_water_year_doy(s.index)
        frames[event_id] = pd.Series(s.values, index=doy)
    if not frames:
        return pd.DataFrame()
    return pd.DataFrame(frames).sort_index()


def _derive_max_allowable_mcm(zone_series):
    """Daily NYC max-allowable diversion (MCM/day) from FFMP zone integers."""
    z = zone_series.astype(float).round().astype('Int64')
    factor = z.map(NYC_ZONE_DELIVERY_FACTORS).astype(float)
    factor = factor.fillna(1.0)
    return factor * NYC_BASELINE_DELIVERY_MCM


def _build_realization_cache(data, dataset_id, realization_ids):
    """Pre-aggregate per-realization time series for the plotted variables.

    Returns a dict {realization_id -> DataFrame} with columns:
      nyc_storage_pct, nyc_release, nyc_delivery, nyc_demand, nyc_shortage,
      nyc_max_allowable, montague_flow.
    """
    cache = {}
    for r in realization_ids:
        storage_raw = data.res_storage[dataset_id][r][NYC_RESERVOIRS].sum(axis=1)
        nyc_storage_pct = 100.0 * storage_raw / NYC_TOTAL_CAPACITY

        contribution = data.contribution[dataset_id][r]
        if isinstance(contribution, pd.DataFrame):
            nyc_release = contribution['mrf_montagueTrenton_nyc']
        else:
            nyc_release = contribution
        nyc_release = nyc_release * MGD_TO_MCM

        nyc_delivery = (
            data.ibt_diversions[dataset_id][r]['delivery_nyc'] * MGD_TO_MCM
        )
        nyc_demand = (
            data.ibt_demands[dataset_id][r]['demand_nyc'] * MGD_TO_MCM
        )
        nyc_shortage = (nyc_demand - nyc_delivery).clip(lower=0)
        zone = data.res_level[dataset_id][r]['nyc']
        nyc_max_allowable = _derive_max_allowable_mcm(zone)

        montague_flow = data.major_flow[dataset_id][r]['delMontague'] * MGD_TO_MCM

        cache[r] = pd.DataFrame({
            'nyc_storage_pct': nyc_storage_pct,
            'nyc_release': nyc_release,
            'nyc_delivery': nyc_delivery,
            'nyc_demand': nyc_demand,
            'nyc_shortage': nyc_shortage,
            'nyc_max_allowable': nyc_max_allowable,
            'montague_flow': montague_flow,
        })
    return cache


def _empirical_cdf(values):
    """Sorted values + midpoint plotting positions."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 2:
        return None, None
    sorted_v = np.sort(arr)
    n = sorted_v.size
    p = (np.arange(1, n + 1) - 0.5) / n
    return sorted_v, p


def _cache_key(thresholds):
    tag = (f"ssi{SSI_WINDOW}"
           f"_rp{thresholds['rp_thresh_years']}"
           f"_fr{thresholds['frac_thresh']}"
           f"_st{thresholds['storage_thresh']}")
    if TEST_MODE:
        tag += '_TESTMODE'
    return os.path.join(
        CACHE_DIR, f"fig10_focal_dynamics_{CACHE_VERSION}_{tag}.pkl")


def _load_reconstruction_full_df():
    """Daily DataFrame of the reconstructed-streamflow simulation.

    Returns a DataFrame indexed on the reconstruction date range with the
    same columns as the per-realization caches. Returns None if the
    reconstruction file is missing.
    """
    if not os.path.exists(RECONSTRUCTION_OUTPUT_FNAME):
        print(f"  Warning: reconstruction file missing at "
              f"{RECONSTRUCTION_OUTPUT_FNAME}; "
              f"historical / baseline distributions disabled.")
        return None

    data = pywrdrb.Data()
    data.load_output(
        output_filenames=[RECONSTRUCTION_OUTPUT_FNAME],
        results_sets=['res_storage', 'major_flow',
                      'ibt_diversions', 'ibt_demands', 'res_level',
                      'nyc_release_components'],
    )
    ds = list(data.res_storage.keys())[0]
    r = list(data.res_storage[ds].keys())[0]

    storage_pct = (
        100.0 * data.res_storage[ds][r][NYC_RESERVOIRS].sum(axis=1)
        / NYC_TOTAL_CAPACITY
    )
    contrib_cols = [f'mrf_montagueTrenton_{res}' for res in NYC_RESERVOIRS]
    nyc_release = (
        data.nyc_release_components[ds][r][contrib_cols].sum(axis=1)
        * MGD_TO_MCM
    )
    nyc_delivery = (
        data.ibt_diversions[ds][r]['delivery_nyc'] * MGD_TO_MCM
    )
    nyc_demand = (
        data.ibt_demands[ds][r]['demand_nyc'] * MGD_TO_MCM
    )
    nyc_shortage = (nyc_demand - nyc_delivery).clip(lower=0)
    zone = data.res_level[ds][r]['nyc']
    nyc_max_allowable = _derive_max_allowable_mcm(zone)
    montague_flow = data.major_flow[ds][r]['delMontague'] * MGD_TO_MCM

    return pd.DataFrame({
        'nyc_storage_pct': storage_pct,
        'nyc_release': nyc_release,
        'nyc_delivery': nyc_delivery,
        'nyc_demand': nyc_demand,
        'nyc_shortage': nyc_shortage,
        'nyc_max_allowable': nyc_max_allowable,
        'montague_flow': montague_flow,
    })


def _scalars_from_window(window):
    """Per-event scalars from a windowed daily DataFrame."""
    drawdown = float(
        window['nyc_storage_pct'].max() - window['nyc_storage_pct'].min()
    )
    min_storage = float(window['nyc_storage_pct'].min())
    min_max_allow = float(window['nyc_max_allowable'].min())
    total_demand = float(window['nyc_demand'].sum())
    total_shortage = float(window['nyc_shortage'].sum())
    shortage_frac_pct = (
        100.0 * total_shortage / total_demand if total_demand > 0 else 0.0
    )
    satisfaction_pct = 100.0 - shortage_frac_pct
    mont_smooth = window['montague_flow'].rolling(
        7, center=True, min_periods=1).mean()
    min_mont = float(mont_smooth.min())
    return dict(
        event_storage_drawdown_pct=drawdown,
        event_min_storage_pct=min_storage,
        event_min_max_allowable_diversion_mcm=min_max_allow,
        event_shortage_frac_of_demand_pct=shortage_frac_pct,
        event_demand_satisfaction_pct=satisfaction_pct,
        event_min_montague_mcm=min_mont,
    )


def _traces_from_window(window):
    """Per-variable DOY-indexed Series for plotting trajectories."""
    out = {}
    for var_name, _, _, smooth, _, _, _ in VARIABLES:
        s = window[var_name]
        if smooth and smooth > 1:
            s = s.rolling(smooth, center=True, min_periods=1).mean()
        doy = vectorized_water_year_doy(s.index)
        out[var_name] = pd.Series(s.values, index=doy).sort_index()
    return out


def load_historical_drought_overlays(full_df):
    """Per-WY traces and event scalars for 1964/1980/2002 droughts."""
    if full_df is None:
        return {}
    enabled = [k for k, c in HISTORICAL_OVERLAYS_CONFIG.items()
               if c['enabled']]
    overlays = {}
    for key in enabled:
        cfg = HISTORICAL_OVERLAYS_CONFIG[key]
        wy_start = cfg['wy_start']
        wy_end = wy_start + pd.DateOffset(years=1) - pd.Timedelta(days=1)
        window = full_df.loc[wy_start:wy_end]
        if window.empty or window['nyc_storage_pct'].dropna().empty:
            print(f"  Warning: WY {wy_start.date()} for {key} has no "
                  f"reconstruction data; overlay omitted.")
            continue
        scalars = _scalars_from_window(window)
        traces = _traces_from_window(window)
        overlays[key] = dict(
            traces=traces, scalars=scalars,
            label=cfg['label'], wy_start=wy_start,
        )
        print(f"  Loaded {cfg['label']} historical overlay "
              f"(min_storage={scalars['event_min_storage_pct']:.1f}%, "
              f"shortage={scalars['event_shortage_frac_of_demand_pct']:.2f}%, "
              f"min_montague={scalars['event_min_montague_mcm']:.2f} MCM/d).")
    return overlays


def filter_historical_overlays_to_focal_region(
    historical_overlays, focal_cells, sev_edges, mag_edges,
):
    """Keep only the historical droughts whose SSI3 reconstruction event
    falls inside the ensemble focal region.

    For each historical overlay key, find the SSI3 reconstruction drought
    event whose `min_storage_date` falls in that overlay's WY (Jun 1 –
    May 31), bin it on the same severity × magnitude grid as the
    ensembles, and check focal-cell membership. Drops historical events
    whose (severity, magnitude) is outside the focal region — and prints
    the per-event audit so the figure caption can cite which historicals
    are kept.
    """
    if not historical_overlays:
        return historical_overlays
    try:
        recon_events = load_event_metrics('reconstruction', SSI_WINDOW)
    except FileNotFoundError as e:
        print(f"  Warning: reconstruction event metrics not found ({e}); "
              f"focal-region filter disabled (keeping all historicals).")
        return historical_overlays

    recon_binned = assign_grid_bins(recon_events, sev_edges, mag_edges)
    recon_binned['min_storage_date'] = pd.to_datetime(
        recon_binned['min_storage_date']
    )

    kept = {}
    for hist_key, payload in historical_overlays.items():
        wy_start = payload['wy_start']
        wy_end = wy_start + pd.DateOffset(years=1) - pd.Timedelta(days=1)
        match = recon_binned[
            (recon_binned['min_storage_date'] >= wy_start) &
            (recon_binned['min_storage_date'] <= wy_end)
        ]
        if match.empty:
            print(f"  {hist_key}: no SSI{SSI_WINDOW} reconstruction event "
                  f"with peak in WY {wy_start.year}/{(wy_end.year)}; dropped.")
            continue
        # If multiple events share that WY (rare), take the worst
        # (lowest event_min_storage_pct).
        match = match.sort_values('event_min_storage_pct').iloc[0]
        sev_bin = int(match['sev_bin'])
        mag_bin = int(match['mag_bin'])
        in_focal = (sev_bin, mag_bin) in focal_cells
        status = 'IN focal region (kept)' if in_focal else 'OUT of focal region (dropped)'
        print(f"  {hist_key} ({payload['label']}): "
              f"SSI{SSI_WINDOW} event peak {match['min_storage_date'].date()} "
              f"sev={match['severity']:.2f} mag={match['magnitude']:.2f} "
              f"-> bin ({sev_bin},{mag_bin})  {status}")
        if in_focal:
            kept[hist_key] = payload
    return kept


def load_historical_baseline_per_wy_scalars(full_df, min_days=300):
    """Per-WY scalars across all complete water years in the reconstruction.

    A WY is considered complete if it has at least `min_days` non-NaN
    storage observations between Jun 1 and May 31. Returns a dict of
    {scalar_key: 1-D float ndarray}, one entry per qualifying WY.
    """
    if full_df is None:
        return {}, []
    storage = full_df['nyc_storage_pct']
    start_year = storage.index.min().year
    end_year = storage.index.max().year

    rows = []
    wy_starts = []
    for y in range(start_year, end_year + 1):
        wy_start = pd.Timestamp(f'{y}-06-01')
        wy_end = wy_start + pd.DateOffset(years=1) - pd.Timedelta(days=1)
        window = full_df.loc[wy_start:wy_end]
        if window['nyc_storage_pct'].dropna().size < min_days:
            continue
        rows.append(_scalars_from_window(window))
        wy_starts.append(wy_start)

    if not rows:
        return {}, []
    keys = rows[0].keys()
    arrays = {k: np.array([r[k] for r in rows], dtype=float) for k in keys}
    print(f"  Historic baseline: {len(rows)} complete water years "
          f"({wy_starts[0].year}-{wy_starts[-1].year}).")
    return arrays, wy_starts


def build_ffmp_by_wy_doy():
    """FFMP Watch/Warning/Emergency thresholds (%) indexed by water-year DOY."""
    fb = load_ffmp_boundaries().copy()
    fb['cal_doy'] = fb.index.dayofyear

    col_map = {}
    for candidate, zone in [('L3', 'Watch'), ('level3', 'Watch'),
                             ('L4', 'Warning'), ('level4', 'Warning'),
                             ('L5', 'Emergency'), ('level5', 'Emergency')]:
        if candidate in fb.columns:
            col_map[candidate] = zone
    if not col_map:
        return None
    fb = fb.rename(columns=col_map)
    zone_cols = [z for z in ['Watch', 'Warning', 'Emergency'] if z in fb.columns]
    seasonal = fb.groupby('cal_doy')[zone_cols].median()

    wy_doys = np.arange(1, 367)
    dates = pd.date_range(REFERENCE_WY_START, periods=366, freq='D')
    cal_doys = dates.dayofyear

    out = pd.DataFrame(index=wy_doys, columns=zone_cols, dtype=float)
    for wy_doy, cal_doy in zip(wy_doys, cal_doys):
        if cal_doy in seasonal.index:
            out.loc[wy_doy] = seasonal.loc[cal_doy].values
    return out


# ── Heavy extraction (cached) ────────────────────────────────────────────

def _compute_focal_dynamics(all_data, focal_cells, sev_edges, mag_edges):
    """Per-dataset HDF5 load + per-event window extraction.

    Returns:
      - dataset_traces[ds][var_name] -> DOY-indexed DataFrame (cols = events)
      - dataset_event_scalars[ds][skey] -> 1D float ndarray
      - dataset_n_events[ds] -> int
    """
    SCALAR_KEYS = (
        'event_storage_drawdown_pct',
        'event_min_storage_pct',
        'event_min_max_allowable_diversion_mcm',
        'event_shortage_frac_of_demand_pct',
        'event_demand_satisfaction_pct',
        'event_min_montague_mcm',
    )

    dataset_traces = {ds: {v[0]: pd.DataFrame() for v in VARIABLES}
                      for ds in DATASETS}
    dataset_event_scalars = {ds: {k: np.array([]) for k in SCALAR_KEYS}
                             for ds in DATASETS}
    dataset_n_events = {ds: 0 for ds in DATASETS}

    for ds in DATASETS:
        df_binned = assign_grid_bins(all_data[ds], sev_edges, mag_edges)
        selected = select_events_from_focal_region(
            df_binned, focal_cells,
            rank_col='event_min_storage_pct', ascending=True, n=None,
        )
        print(f"  {ds}: {len(selected)} focal events")
        dataset_n_events[ds] = len(selected)

        if len(selected) == 0:
            continue

        unique_reals = sorted(set(int(r) for r in selected['realization_id']))
        fname = os.path.join(OUTPUT_DIR, f'{ds}_with_postprocessing.hdf5')
        data = load_rank_subset_from_export(
            fname, unique_reals, RESULTS_SETS, rank=0, size=1,
        )
        realization_cache = _build_realization_cache(data, ds, unique_reals)
        del data
        gc.collect()

        per_var_traces = {v[0]: [] for v in VARIABLES}
        scalars = {k: [] for k in SCALAR_KEYS}

        for _, row in selected.iterrows():
            r_id = int(row['realization_id'])
            event_id = (f"R{r_id:04d}_"
                        f"{pd.Timestamp(row['start']).date()}")
            min_storage_date = pd.Timestamp(row['min_storage_date'])

            w_start, w_end = compute_fixed_extraction_window(
                min_storage_date, pad_before_wy=0, pad_after_wy=0,
            )
            window = realization_cache[r_id].loc[w_start:w_end]

            sc = _scalars_from_window(window)
            for k in SCALAR_KEYS:
                scalars[k].append(sc[k])

            for var_name, _, _, smooth, _, _, _ in VARIABLES:
                s = window[var_name]
                if smooth and smooth > 1:
                    s = s.rolling(smooth, center=True, min_periods=1).mean()
                per_var_traces[var_name].append((event_id, s))

        dataset_event_scalars[ds] = {
            k: np.asarray(v, dtype=float) for k, v in scalars.items()
        }
        dataset_traces[ds] = {
            var_name: _traces_to_doy_df(per_var_traces[var_name])
            for var_name, _, _, _, _, _, _ in VARIABLES
        }

        del realization_cache
        gc.collect()

    return dict(
        dataset_traces=dataset_traces,
        dataset_event_scalars=dataset_event_scalars,
        dataset_n_events=dataset_n_events,
    )


def _load_or_compute_focal_dynamics(
    all_data, focal_cells, sev_edges, mag_edges, thresholds,
):
    cache_path = _cache_key(thresholds)

    if (not REBUILD_CACHE) and os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
            print(f"  Loaded cached focal-event dynamics from {cache_path}")
            return cached
        except Exception as e:
            print(f"  Warning: cache load failed ({e}); recomputing.")

    print("\n  Computing focal-event dynamics (slow; HDF5 reads).")
    result = _compute_focal_dynamics(
        all_data, focal_cells, sev_edges, mag_edges,
    )

    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Cached to {cache_path}")
    except Exception as e:
        print(f"  Warning: cache save failed ({e}).")

    return result


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    apply_publication_style()
    os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

    print(f"Fig10: focal-event dynamics + FFMP performance distributions "
          f"(TEST_MODE={TEST_MODE}, SSI_WINDOW={SSI_WINDOW})")

    # 1. Event metrics per dataset
    all_data = {}
    for ds in DATASETS:
        all_data[ds] = load_event_metrics(ds, SSI_WINDOW)
        print(f"  {ds}: {len(all_data[ds])} events")

    # 2. Shared severity x magnitude grid
    sev_edges, mag_edges, _, _ = make_shared_edges_logmag(
        all_data, DATASETS, n_bins=GRID_N_BINS,
    )

    # 3. Per-dataset grids and focal-cell identification.
    T_W_grids, frac_grids, min_grids = {}, {}, {}
    for ds in DATASETS:
        _, _, T_W_grids[ds], _ = compute_return_period_grid(
            all_data[ds], sev_edges, mag_edges, N_YEARS, min_count=MIN_COUNT)
        frac_grids[ds], _ = compute_emergency_grid(
            all_data[ds], sev_edges, mag_edges, min_count=MIN_COUNT)
        min_grids[ds], _ = compute_min_storage_grid(
            all_data[ds], sev_edges, mag_edges, min_count=MIN_COUNT)

    thresholds = STRICT_THRESHOLDS
    focal_cells = identify_focal_region(
        T_W_grids, frac_grids, min_grids, DATASETS, **thresholds)
    print(f"  Strict focal region: {len(focal_cells)} cells")

    if len(focal_cells) == 0 and TEST_MODE:
        thresholds = RELAXED_THRESHOLDS
        focal_cells = identify_focal_region(
            T_W_grids, frac_grids, min_grids, DATASETS, **thresholds)
        print(f"  TEST_MODE relaxed focal region: {len(focal_cells)} cells")

    if len(focal_cells) == 0:
        raise RuntimeError(
            "No focal cells found (strict thresholds empty and TEST_MODE=False)."
        )

    # 4. Focal-event extraction (cached).
    dynamics = _load_or_compute_focal_dynamics(
        all_data, focal_cells, sev_edges, mag_edges, thresholds,
    )
    dataset_traces = dynamics['dataset_traces']
    dataset_event_scalars = dynamics['dataset_event_scalars']
    dataset_n_events = dynamics['dataset_n_events']

    # 5. FFMP zone thresholds by water-year DOY (storage row only).
    ffmp_by_wy = build_ffmp_by_wy_doy()

    # 6. Reconstruction-derived overlays + historic-baseline distribution.
    print("\n  Loading reconstruction (historical droughts + baseline)...")
    full_df = _load_reconstruction_full_df()
    historical_overlays = load_historical_drought_overlays(full_df)

    # Filter historical droughts to those whose SSI3 reconstruction event
    # falls inside the ensemble focal region. Drops droughts that are
    # extreme outliers of the focal distribution (e.g. the multi-year
    # 1960s drought, which often clips above the focal grid edges).
    print("\n  Filtering historical overlays by SSI"
          f"{SSI_WINDOW} focal-region membership:")
    historical_overlays = filter_historical_overlays_to_focal_region(
        historical_overlays, focal_cells, sev_edges, mag_edges,
    )
    active_hist_keys = list(historical_overlays.keys())

    baseline_arrays, _ = load_historical_baseline_per_wy_scalars(full_df)

    # 7. Figure — 3 rows x 2 cols.
    n_rows = len(VARIABLES)
    n_cols = 2

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(11.0, 12.0),
        gridspec_kw=dict(width_ratios=[1.45, 1.0]),
    )

    # Trajectory column: share x across rows; CDF column: share x across rows.
    for row in range(1, n_rows):
        axes[row, 0].sharex(axes[0, 0])
        axes[row, 1].sharex(axes[0, 1])

    ensemble_handles = {}      # ds -> Line2D from trajectory column (median)
    hist_handles = {}          # hist_key -> Line2D from CDF-column bar
    ffmp_handles = {}
    cdf_ref_handle = None
    july_nov_handle = None

    for row_idx, var_def in enumerate(VARIABLES):
        (var_name, var_label, yscale, _, scalar_key, cdf_ylabel,
         cdf_ref_lines) = var_def

        # ── Trajectory column (col 0) ────────────────────────────────
        ax_traj = axes[row_idx, 0]

        # July-Nov operational-band markers — two thin black dashed lines
        # banding the late-summer/fall drawdown period on every left-column
        # panel. Drawn before the data so they sit at the back.
        for x_doy in JULY_NOV_BAND_DOYS:
            ln = ax_traj.axvline(
                x_doy, color='black', linestyle='--', linewidth=0.9,
                alpha=0.75, zorder=2,
            )
            july_nov_handle = ln

        # FFMP rule curves on storage row only — drawn first so they sit
        # underneath the ensemble lines. The bottom-of-figure legend
        # describes the zones; no inline labels (kept declutter).
        if var_name == 'nyc_storage_pct' and ffmp_by_wy is not None:
            for zone in ['Watch', 'Warning', 'Emergency']:
                if zone not in ffmp_by_wy.columns:
                    continue
                zvals = ffmp_by_wy[zone].astype(float)
                ln, = ax_traj.plot(
                    zvals.index, zvals.values,
                    color=FFMP_ZONE_COLORS[zone], linestyle='--',
                    linewidth=1.1, alpha=0.95, zorder=4,
                )
                ffmp_handles.setdefault(zone, ln)

        for ds in DATASETS:
            traces_df = dataset_traces.get(ds, {}).get(var_name, pd.DataFrame())
            if traces_df.shape[1] < 2:
                continue
            traces_df = traces_df.reindex(range(1, 367))
            color = DATASET_COLORS[ds]
            # 50th (solid) + the variable-specific stressful 2% tail (thick
            # dashed): 2nd %ile for storage/Montague (low = adverse), 98th
            # %ile for NYC release (high = adverse). The 10th-percentile
            # line was dropped to declutter the left column; the outer line
            # is drawn thicker for visibility against the median.
            outer_q = STRESSFUL_OUTER_QUANTILE.get(var_name, 0.98)
            ln_med, _ = plot_quantile_lines(
                ax_traj, traces_df, color=color,
                median_q=0.50, outer_qs=(outer_q,),
                linewidth_med=1.8, linewidth_outer=1.7,
                alpha_med=0.95, alpha_outer=0.70,
                linestyle_outer='--',
            )
            if ln_med is not None:
                ensemble_handles.setdefault(ds, ln_med)

        # Historical drought lines (1964 / 1980 / 2002). Drawing is gated
        # by SHOW_HISTORICAL_TRAJECTORIES so the timeseries can be brought
        # back without code changes; the right-side CDF bars use the same
        # historical_overlays dict regardless.
        if SHOW_HISTORICAL_TRAJECTORIES:
            for hist_key in active_hist_keys:
                if hist_key not in historical_overlays:
                    continue
                s = historical_overlays[hist_key]['traces'].get(var_name)
                if s is None or s.dropna().size == 0:
                    continue
                s_sorted = s.sort_index()
                ln, = ax_traj.plot(
                    s_sorted.index, s_sorted.values, **HIST_STYLES[hist_key],
                )
                hist_handles.setdefault(hist_key, ln)

        if yscale == 'log':
            ax_traj.set_yscale('log')
        else:
            ax_traj.set_ylim(bottom=0)

        if row_idx == n_rows - 1:
            format_xaxis_water_year(ax_traj)
            ax_traj.set_xlabel(XAXIS_SUFFIX_LABEL, fontsize=FONTSIZE_LABEL)
        else:
            ax_traj.set_xticks(MONTH_STARTS_WY)
            ax_traj.set_xticklabels([])
        ax_traj.set_xlim(1, 366)
        ax_traj.set_ylabel(var_label, fontsize=FONTSIZE_LABEL)

        if row_idx == 0:
            ax_traj.set_title(
                'Focal-event drought-year trajectories\n'
                '(per-ensemble median + stressful 2nd/98th %ile)',
                fontsize=FONTSIZE_TITLE,
            )

        label_panel(
            ax_traj, PANEL_LETTERS[row_idx * n_cols + 0],
            fontsize=FONTSIZE_LABEL, fontweight='normal',
        )
        ax_traj.grid(False)
        for spine in ax_traj.spines.values():
            spine.set_edgecolor('#333333')

        # ── CDF column (col 1) ───────────────────────────────────────
        ax_cdf = axes[row_idx, 1]

        # Ensemble CDF curves only — historic baseline CDF (formerly drawn
        # as a gray dotted curve from all reconstruction WYs) is omitted by
        # request; the historical overlays appear as small bars instead.
        for ds in DATASETS:
            values = dataset_event_scalars.get(ds, {}).get(scalar_key, None)
            if values is None or values.size < 5:
                continue
            sorted_v, p_emp = _empirical_cdf(values)
            if sorted_v is None:
                continue
            color = DATASET_COLORS[ds]
            ax_cdf.plot(
                p_emp, sorted_v, color=color, linewidth=1.9, alpha=0.95,
                zorder=5,
            )

        # Reference horizontal lines from VARIABLES + Montague targets.
        # Annotation text is in MCM/day units.
        for tgt_y, tgt_lbl in cdf_ref_lines:
            ln = ax_cdf.axhline(
                tgt_y, color='#444444', linestyle=':', linewidth=1.4,
                alpha=0.9, zorder=3,
            )
            if cdf_ref_handle is None:
                cdf_ref_handle = ln
            ax_cdf.text(
                0.99, tgt_y, f' {tgt_lbl}',
                fontsize=FONTSIZE_SMALL - 1, color='#444444',
                va='bottom', ha='right',
                transform=ax_cdf.get_yaxis_transform(), zorder=3.5,
            )
        if var_name == 'montague_flow':
            for tgt, lbl in [
                (MONTAGUE_TARGET_NORMAL_MCM,
                 f'FFMP Montague baseline target '
                 f'({MONTAGUE_TARGET_NORMAL_MCM:.2f} MCM/day)'),
                (MONTAGUE_TARGET_EMERGENCY_MCM,
                 f'FFMP Drought Emergency target '
                 f'({MONTAGUE_TARGET_EMERGENCY_MCM:.2f} MCM/day)'),
            ]:
                ln = ax_cdf.axhline(
                    tgt, color='#444444', linestyle=':', linewidth=1.4,
                    alpha=0.9, zorder=3,
                )
                if cdf_ref_handle is None:
                    cdf_ref_handle = ln
                ax_cdf.text(
                    0.99, tgt, f' {lbl}',
                    fontsize=FONTSIZE_SMALL - 1, color='#444444',
                    va='bottom', ha='right',
                    transform=ax_cdf.get_yaxis_transform(), zorder=3.5,
                )

        # Historical drought small horizontal bars — drawn on the left
        # edge of the CDF panel at the historical scalar value, color-
        # coded and x-staggered so bars at identical y-values (e.g. two
        # droughts that both bottom out at the FFMP Emergency cap) remain
        # visually distinct. The bottom legend names each color; no inline
        # year labels are drawn (single-legend convention).
        bar_width = 0.04
        for i, hist_key in enumerate(active_hist_keys):
            if hist_key not in historical_overlays:
                continue
            hv = historical_overlays[hist_key]['scalars'].get(
                scalar_key, float('nan'))
            if not np.isfinite(hv):
                continue
            color = HIST_STYLES[hist_key]['color']
            xmin = i * bar_width
            xmax = xmin + bar_width
            ln = ax_cdf.axhline(
                hv, xmin=xmin, xmax=xmax, color=color,
                linewidth=2.8, alpha=0.95, zorder=6,
            )
            hist_handles.setdefault(hist_key, ln)

        ax_cdf.set_xlim(0, 1)
        cdf_xticks = np.round(np.arange(0.0, 1.01, 0.1), 2)
        ax_cdf.set_xticks(cdf_xticks)
        ax_cdf.set_xticklabels([f'{v:.1f}' for v in cdf_xticks],
                               fontsize=FONTSIZE_SMALL)
        # Force tick labels on every row even though sharex would otherwise
        # hide them on internal subplots.
        ax_cdf.tick_params(axis='x', labelbottom=True)
        if row_idx == n_rows - 1:
            ax_cdf.set_xlabel('Nonexceedance probability',
                              fontsize=FONTSIZE_LABEL)
        ax_cdf.set_ylabel(cdf_ylabel, fontsize=FONTSIZE_LABEL)
        if row_idx == 0:
            ax_cdf.set_title(
                'Focal-event empirical CDFs\n'
                '(left-edge bars: focal-region historical SSI3 droughts)',
                fontsize=FONTSIZE_TITLE,
            )

        label_panel(
            ax_cdf, PANEL_LETTERS[row_idx * n_cols + 1],
            fontsize=FONTSIZE_LABEL, fontweight='normal',
        )
        ax_cdf.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        for spine in ax_cdf.spines.values():
            spine.set_edgecolor('#333333')

    fig.subplots_adjust(
        left=0.08, right=0.97, top=0.91, bottom=0.18,
        hspace=0.50, wspace=0.32,
    )

    fig.align_ylabels(axes[:, 0])

    # ── Single figure-level legend (bottom) ─────────────────────────
    # All legend elements are centralized here. Order:
    #   1) Ensemble medians (3) — left-column trajectories + right-column CDFs
    #   2) Historic drought CDF bars (3) — right-column only
    #   3) FFMP storage rule curves (3) — left storage panel only
    #   4) CDF reference thresholds (1) — see panel-side annotations for values
    legend_handles, legend_labels = [], []
    for ds in DATASETS:
        if ds in ensemble_handles:
            legend_handles.append(ensemble_handles[ds])
            legend_labels.append(
                f"{DATASET_LABELS_SHORT.get(ds, ds)} focal events "
                f"(n={dataset_n_events[ds]}; median solid, "
                f"stressful 2nd/98th %ile dashed)"
            )
    for hist_key in active_hist_keys:
        if hist_key in hist_handles:
            legend_handles.append(hist_handles[hist_key])
            legend_labels.append(HIST_LABELS[hist_key])
    ffmp_level_map = {'Watch': 'L3', 'Warning': 'L4', 'Emergency': 'L5'}
    for zone in ['Watch', 'Warning', 'Emergency']:
        if zone in ffmp_handles:
            legend_handles.append(ffmp_handles[zone])
            legend_labels.append(
                f'FFMP {zone} threshold ({ffmp_level_map[zone]}, '
                f'storage rule curve)'
            )
    if july_nov_handle is not None:
        legend_handles.append(july_nov_handle)
        legend_labels.append(
            'July - November operational band (late-summer/fall drawdown)'
        )
    if cdf_ref_handle is not None:
        legend_handles.append(cdf_ref_handle)
        legend_labels.append(
            'CDF reference thresholds (FFMP / manuscript landmarks; '
            'see panel-side labels)'
        )

    if legend_handles:
        fig.legend(
            legend_handles, legend_labels,
            loc='lower center', bbox_to_anchor=(0.5, 0.01),
            ncol=3, fontsize=FONTSIZE_LEGEND,
            frameon=False,
        )

    out_stem = os.path.join(FIG_OUTPUT_DIR, FIG_NAME_STEM)
    save_fig(fig, out_stem, dpi=DPI_PRINT)
    plt.close(fig)
    print(f"\nSaved (png/svg/pdf): {out_stem}")


if __name__ == '__main__':
    main()
