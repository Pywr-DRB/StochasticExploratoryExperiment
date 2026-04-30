"""
Fig10: Focal-event dynamics + distributions (3x3 grid).

Two trajectory columns (WWDS, WWSS) show day-of-water-year quantile gradients
of focal-zone drought water-years with two reference trajectories overlaid
(10th-percentile and median minimum-storage focal events) plus optional
historical-drought overlays (1964 drought of record, 1980 drought) drawn
from a reconstructed-streamflow Pywr-DRB simulation. The right column shows
empirical CDFs of per-event scalars across all three ensembles, with
percentile callouts and (when enabled) horizontal anchors marking where the
historical droughts sit in each distribution.

The narrative argument: focal-region drought operational dynamics are
climate-invariant across WWDS and WWSS — the same operational signature
(rapid summer-fall drawdown, sustained low through winter, partial spring
recovery) emerges in both. Stationary-baseline trajectories are visually
indistinguishable from WWDS and are not shown in the trajectory columns,
but SSB stays in the right-column CDFs as a reference distribution. The
focal-region event set is therefore proposed as a stress-testing portfolio
for DRB operations — each scenario is known to be possible in each
climate trajectory.

Rows (top -> bottom):
  Storage  : NYC total storage (% of capacity)
  Release  : NYC release to Montague target (MCM/day)
  Montague : Montague streamflow (MCM/day, log scale)

Columns (left -> right):
  WWDS trajectories (climate_adjusted_low)
  WWSS trajectories (climate_adjusted_high)
  Focal-event CDFs (all three ensembles overlaid)

Focal-region cells are still identified jointly across all three ensembles
(unchanged from prior version). Stationary baseline drops out of the
trajectory loop but stays in the CDF column.

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
from methods.plotting.percentile_bands import format_xaxis_water_year
from methods.plotting.styles import (
    DATASET_LABELS, DATASET_LABELS_SHORT, DATASET_COLORS,
    FFMP_ZONE_COLORS, CMAP_SEQUENTIAL,
    DPI_PRINT, FONTSIZE_LABEL, FONTSIZE_TITLE, FONTSIZE_LEGEND, FONTSIZE_SMALL,
    apply_publication_style, label_panel, save_fig,
)
from methods.plotting.ensemble_summary import MGD_TO_MCM
from methods.plotting.doy_quantile_gradient import (
    compute_doy_quantile_grid, plot_doy_quantile_gradient,
)


# ── Configuration ────────────────────────────────────────────────────────

SSI_WINDOW = 3
MIN_COUNT = 1

# Datasets used for focal-region identification and CDF column.
DATASETS = list(DATASET_CONFIGS.keys())
# Datasets that get a trajectory column (Stationary baseline drops out;
# its contribution to the figure is the comparison line in the CDF column).
TRAJECTORY_DATASETS = ['climate_adjusted_low', 'climate_adjusted_high']

# Only the result sets we actually consume — inflow is unused, dropping it
# avoids loading a large DataFrame per realization from HDF5.
RESULTS_SETS = ['res_storage', 'contribution', 'major_flow']

# TEST_MODE relaxes focal thresholds when the strict Fig9/Fig10 region is empty
# on the local 5-realization dataset. Final-run default: TEST_MODE = False.
TEST_MODE = False

STRICT_THRESHOLDS = dict(
    rp_thresh_years=FOCAL_RP_THRESH_YEARS,
    frac_thresh=FOCAL_FRAC_THRESH,
    storage_thresh=FOCAL_WORST_STORAGE_THRESH,
)
RELAXED_THRESHOLDS = dict(
    rp_thresh_years=np.inf,  # accept any finite T_W
    frac_thresh=2.0,         # accept any non-NaN frac (>1 always true)
    storage_thresh=200.0,    # accept any non-NaN min storage
)

REFERENCE_WY_START = pd.Timestamp('2000-06-01')
REFERENCE_WY_END = pd.Timestamp('2001-05-31')
N_QUANTILE_LEVELS = 21    # 0, 5, 10, ..., 100 %

# Historical-drought overlays drawn from the reconstructed-streamflow
# Pywr-DRB simulation (RECONSTRUCTION_OUTPUT_FNAME). Each toggleable; set
# either flag to False to drop the corresponding overlay from both the
# trajectory columns and the CDF column. The water year is the FFMP-
# convention WY (Jun 1 – May 31) containing the storage minimum:
#   1964 drought of record  -> WY Jun 1964 – May 1965 (Feb-1965 storage low)
#   1980 drought            -> WY Jun 1980 – May 1981
SHOW_1964_OVERLAY = True
SHOW_1980_OVERLAY = True
HISTORICAL_OVERLAYS_CONFIG = {
    'h1964': dict(label='1964', wy_start=pd.Timestamp('1964-06-01'),
                  enabled=SHOW_1964_OVERLAY),
    'h1980': dict(label='1980', wy_start=pd.Timestamp('1980-06-01'),
                  enabled=SHOW_1980_OVERLAY),
}
HISTORICAL_OVERLAY_KEYS = [k for k, c in HISTORICAL_OVERLAYS_CONFIG.items()
                           if c['enabled']]

CMAP_N_BINS = 10          # discrete viridis bands (matches Fig9 %DE 10-pp ticks)
CMAP = plt.get_cmap(CMAP_SEQUENTIAL, CMAP_N_BINS)

# FFMP Montague flow targets (MCM/day) for vertical reference lines on the
# CDF column. Normal target = 1750 cfs; Drought Emergency target = 1100 cfs;
# converted via 1 cfs ≈ 0.002446575 MCM/day.
MONTAGUE_TARGET_NORMAL_MCM = 1750.0 * 0.002446575
MONTAGUE_TARGET_EMERGENCY_MCM = 1100.0 * 0.002446575

# (variable_key, trajectory y-label, y-scale, rolling-mean window in days,
#  cdf scalar key, cdf y-label, list of (value, text-label) horizontal
#  reference thresholds drawn on the CDF panel)
# Release and flow are converted from MGD to MCM/day (same convention as
# Fig4); storage stays as a percentage of capacity.
# CDF panels use SWAPPED axes — x = nonexceedance probability (0–1),
# y = the metric value — so the reference thresholds below are y-values.
VARIABLES = [
    ('nyc_storage_pct',
     'Combined NYC storage\n(% of capacity)',                    'linear', 1,
     'event_storage_drawdown_pct',
     'NYC storage drawdown\namplitude (% of capacity)',
     [(75.0, 'Manuscript reference: 75% drawdown')]),
    ('nyc_release',
     'Mandated NYC release to\nMontague target (MCM/day)',       'linear', 7,
     'event_release_ratio_vs_historical',
     'Cumulative focal-year release /\nhistorical median annual release',
     [(2.0, '2× historical median annual release'),
      (3.0, '3× historical median annual release')]),
    ('montague_flow',
     'Montague gauge flow\n(MCM/day, log scale)',                'log',    7,
     'event_min_montague_mcm',
     'Event minimum 7-day\nMontague flow (MCM/day)',
     []),  # FFMP target lines added explicitly so the legend handle is captured
]

PANEL_LETTERS = list('abcdefghi')
XAXIS_SUFFIX_LABEL = 'Water Year (Jun 1 - May 31, FFMP convention)'

FIG_OUTPUT_DIR = os.path.join(FIG_DIR, 'Fig10')
FIG_NAME_STEM = f'Fig10_focal_doy_quantiles_ssi{SSI_WINDOW}'
if TEST_MODE:
    FIG_NAME_STEM += '_TESTMODE'

# Cache for expensive per-event extraction (HDF5 read + per-realization
# aggregation + per-event window scalar computation). Set REBUILD_CACHE=True
# (or delete the .pkl file) to invalidate. Bump CACHE_VERSION when the
# cache payload schema changes.
CACHE_DIR = os.path.join(OUTPUT_DIR, 'cache')
CACHE_VERSION = 'v3'
REBUILD_CACHE = False

# CDF column percentile callouts: small filled circles + value labels at
# these nonexceedance probabilities, drawn on each ensemble's empirical CDF
# curve to make typical-event vs. bad-tail readouts directly legible.
CDF_PCT_CALLOUTS = (0.10, 0.50, 0.90)

# Reference trajectories overlaid on the trajectory columns. Events are
# sorted ascending by event_min_storage_pct (worst first), so:
#   p10    -> rank ~0.1*(n-1)   (10th-percentile min-storage value: severe
#                                 but not worst — characterizes the bad
#                                 tail without being the single extreme)
#   median -> rank n/2          (event at the median min-storage value)
# The worst-case event was dropped because it is too extreme to drive a
# stress-testing-portfolio narrative; the typical-but-bad-tail behavior
# (p10 + median) is what planners should design against. The DOY-wise
# mean was dropped because the 21-band gradient already conveys
# within-ensemble central tendency.
REF_KEYS = ['p10', 'median']
# Distinguishable from each other AND from the dashed FFMP zone lines.
# Historical-drought overlays use saturated colors that do not collide
# with p10 (orange) or median (black) or FFMP threshold dashes.
REF_STYLES = {
    'p10':    dict(color='#e67e22', linewidth=1.7, linestyle='--', alpha=0.95, zorder=7),
    'median': dict(color='#000000', linewidth=1.7, linestyle='-',  alpha=0.95, zorder=7),
    'h1964':  dict(color='#8e1c5c', linewidth=1.5, linestyle='-',  alpha=0.95, zorder=7.5),
    'h1980':  dict(color='#107070', linewidth=1.5, linestyle='-',  alpha=0.95, zorder=7.5),
}
REF_LABELS = {
    'p10':    '10th-percentile minimum-storage focal event (severe tail)',
    'median': 'Median minimum-storage focal event',
    'h1964':  '1964 drought of record (reconstructed historical)',
    'h1980':  '1980 drought (reconstructed historical)',
}


# ── Helpers ──────────────────────────────────────────────────────────────

def _traces_to_doy_df(traces_by_event):
    """Stack a list of (event_id, pd.Series) into a DOY-indexed DataFrame.

    DOY is invariant under whole-year date shifts, so we do not need to
    align every event's DatetimeIndex onto a common reference year before
    building the DOY frame — `vectorized_water_year_doy` yields identical
    values regardless of calendar year.
    """
    frames = {}
    for event_id, s in traces_by_event:
        if s is None or len(s) == 0:
            continue
        doy = vectorized_water_year_doy(s.index)
        frames[event_id] = pd.Series(s.values, index=doy)
    if not frames:
        return pd.DataFrame()
    return pd.DataFrame(frames).sort_index()


def _build_realization_cache(data, dataset_id, realization_ids):
    """Pre-aggregate per-realization time series for the three plotted variables.

    The per-event loop previously recomputed NYC-reservoir sums and column
    lookups for *every* focal event (often many events share a realization).
    Building one DataFrame per realization up front collapses that work
    from O(events) down to O(unique realizations).
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

        montague_flow = data.major_flow[dataset_id][r]['delMontague'] * MGD_TO_MCM

        cache[r] = pd.DataFrame({
            'nyc_storage_pct': nyc_storage_pct,
            'nyc_release': nyc_release,
            'montague_flow': montague_flow,
        })
    return cache


def _empirical_cdf(values):
    """Sorted values + midpoint plotting positions.

    Returns (sorted_values, p) where p[i] = (i + 0.5) / n is the
    nonexceedance probability for sorted_values[i]. n >= 2 events required.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 2:
        return None, None
    sorted_v = np.sort(arr)
    n = sorted_v.size
    p = (np.arange(1, n + 1) - 0.5) / n
    return sorted_v, p


def _cache_key(thresholds):
    """Filename for the per-event extraction cache."""
    tag = (f"ssi{SSI_WINDOW}"
           f"_rp{thresholds['rp_thresh_years']}"
           f"_fr{thresholds['frac_thresh']}"
           f"_st{thresholds['storage_thresh']}")
    if TEST_MODE:
        tag += '_TESTMODE'
    return os.path.join(
        CACHE_DIR, f"fig10_focal_dynamics_{CACHE_VERSION}_{tag}.pkl")


def load_historical_median_annual_release_mcm():
    """Median total annual NYC release-to-Montague-target, in MCM.

    Loads the reconstruction simulation, sums daily NYC release components
    by water year (Jun 1 – May 31), and returns the median across water
    years as a single scalar. Used as the denominator for the
    release-ratio CDF (panel f).

    Returns NaN if the reconstruction file is missing.
    """
    if not os.path.exists(RECONSTRUCTION_OUTPUT_FNAME):
        print(f"  Warning: reconstruction file missing at "
              f"{RECONSTRUCTION_OUTPUT_FNAME}; release ratio will be NaN.")
        return float('nan')
    data = pywrdrb.Data()
    data.load_output(
        output_filenames=[RECONSTRUCTION_OUTPUT_FNAME],
        results_sets=['nyc_release_components'],
    )
    ds = list(data.nyc_release_components.keys())[0]
    r = list(data.nyc_release_components[ds].keys())[0]
    contrib_cols = [f'mrf_montagueTrenton_{res}' for res in NYC_RESERVOIRS]
    daily_release_mcm = (
        data.nyc_release_components[ds][r][contrib_cols].sum(axis=1)
        * MGD_TO_MCM
    )
    # Group by water year (Jun 1 – May 31): a date with month >= 6 belongs
    # to the WY starting that calendar year; otherwise to the prior WY.
    wy = np.where(
        daily_release_mcm.index.month >= 6,
        daily_release_mcm.index.year,
        daily_release_mcm.index.year - 1,
    )
    annual_totals = daily_release_mcm.groupby(wy).sum()
    # Drop the partial first/last WYs if they have <= 200 days of data.
    counts = daily_release_mcm.groupby(wy).count()
    annual_totals = annual_totals[counts >= 300]
    return float(annual_totals.median())


def load_historical_drought_overlays():
    """Per-water-year traces and event scalars for the configured historical
    droughts (1964, 1980), drawn from the reconstructed-streamflow
    Pywr-DRB simulation at RECONSTRUCTION_OUTPUT_FNAME.

    Returns a dict keyed by HISTORICAL_OVERLAYS_CONFIG key (e.g. 'h1964')
    mapping to:
      - traces[var_name] -> DOY-indexed pd.Series (same DOY convention as
        focal-event traces; rolling-mean smoothing applied per VARIABLES).
      - scalars[scalar_key] -> float (same semantics as the focal-event
        scalars: drawdown amplitude, total release MCM, min 7-day Montague).

    Disabled overlays and any year whose WY does not fall inside the
    reconstruction date range are silently omitted; missing reconstruction
    file returns an empty dict (caller treats overlays as unavailable).
    """
    enabled_keys = [k for k, c in HISTORICAL_OVERLAYS_CONFIG.items()
                    if c['enabled']]
    if not enabled_keys:
        return {}
    if not os.path.exists(RECONSTRUCTION_OUTPUT_FNAME):
        print(f"  Warning: reconstruction file missing at "
              f"{RECONSTRUCTION_OUTPUT_FNAME}; "
              f"historical-drought overlays disabled.")
        return {}

    # Reconstruction outputs use 'nyc_release_components' (per-reservoir
    # mrf_montagueTrenton_{res}) rather than the post-processed
    # 'contribution' result set used by the focal-event simulation
    # ensembles. We sum across NYC reservoirs to recover the same total
    # mandated NYC release to the Montague target.
    data = pywrdrb.Data()
    data.load_output(
        output_filenames=[RECONSTRUCTION_OUTPUT_FNAME],
        results_sets=['res_storage', 'nyc_release_components', 'major_flow'],
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
    montague_flow = data.major_flow[ds][r]['delMontague'] * MGD_TO_MCM

    full = pd.DataFrame({
        'nyc_storage_pct': storage_pct,
        'nyc_release': nyc_release,
        'montague_flow': montague_flow,
    })

    overlays = {}
    for key in enabled_keys:
        cfg = HISTORICAL_OVERLAYS_CONFIG[key]
        wy_start = cfg['wy_start']
        wy_end = wy_start + pd.DateOffset(years=1) - pd.Timedelta(days=1)
        window = full.loc[wy_start:wy_end]
        if window.empty or window['nyc_storage_pct'].dropna().empty:
            print(f"  Warning: WY {wy_start.date()} for {key} has no "
                  f"reconstruction data; overlay omitted.")
            continue

        # Per-event scalars (same definitions as the focal-event extraction
        # loop in _compute_focal_dynamics).
        drawdown = float(
            window['nyc_storage_pct'].max() - window['nyc_storage_pct'].min()
        )
        total_release = float(window['nyc_release'].sum())
        mont_smooth = window['montague_flow'].rolling(
            7, center=True, min_periods=1).mean()
        min_montague = float(mont_smooth.min())

        # Project to DOY + apply same rolling-mean smoothing as the focal-
        # event traces so the historical overlay shares units and visual
        # smoothing with the gradient + p10 / median lines.
        var_traces = {}
        for var_name, _, _, smooth, _, _, _ in VARIABLES:
            s = window[var_name]
            if smooth and smooth > 1:
                s = s.rolling(smooth, center=True, min_periods=1).mean()
            doy = vectorized_water_year_doy(s.index)
            var_traces[var_name] = pd.Series(
                s.values, index=doy).sort_index()

        overlays[key] = dict(
            traces=var_traces,
            scalars=dict(
                event_storage_drawdown_pct=drawdown,
                event_total_release_mcm=total_release,
                event_min_montague_mcm=min_montague,
            ),
            label=cfg['label'],
            wy_start=wy_start,
        )
        print(f"  Loaded {cfg['label']} historical overlay "
              f"(drawdown={drawdown:.1f}%, "
              f"total release={total_release:.0f} MCM, "
              f"min Montague={min_montague:.2f} MCM/d).")
    return overlays


def build_ffmp_by_wy_doy():
    """FFMP Watch/Warning/Emergency thresholds (%) indexed by water-year DOY.

    Time-varies: reflects the seasonal FFMP rule curves, aligned so that
    DOY 1 = June 1 (water-year convention).
    """
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

    For TRAJECTORY_DATASETS: builds DOY-aligned trace DataFrames + four
    reference trajectories (worst, p10, median, mean) for the overlay lines.
    For all DATASETS: collects per-event scalars (used by the CDF column).

    Returns a dict keyed by:
      - dataset_traces[ds][var_name]      -> DOY-indexed DataFrame (cols = events).
      - dataset_refs[ds][var_name]        -> dict of {ref_key: Series indexed by DOY}.
      - dataset_ref_meta[ds]              -> dict of {ref_key: per-event metadata}.
      - dataset_event_scalars[ds][skey]   -> 1D float ndarray.
      - dataset_n_events[ds]              -> int (number of focal events).
    """
    # Scalars stored per focal event. Some are direct CDF inputs (drawdown,
    # min Montague); event_total_release_mcm is kept so the historical-ratio
    # CDF metric can be computed at plot time without rerunning the
    # extraction. event_min_storage_pct is kept for diagnostics / future use.
    SCALAR_KEYS = (
        'event_storage_drawdown_pct',
        'event_total_release_mcm',
        'event_min_montague_mcm',
        'event_min_storage_pct',
    )

    dataset_traces = {ds: {v[0]: pd.DataFrame() for v in VARIABLES}
                      for ds in DATASETS}
    dataset_refs = {ds: {v[0]: {} for v in VARIABLES} for ds in DATASETS}
    dataset_ref_meta = {ds: {} for ds in DATASETS}
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

        is_traj = ds in TRAJECTORY_DATASETS
        per_var_traces = {v[0]: [] for v in VARIABLES} if is_traj else None
        scalars = {k: [] for k in SCALAR_KEYS}

        # Reference event ranks within the sorted-ascending min-storage list:
        #   worst  = rank 0           (lowest min storage)
        #   p10    = rank ~0.1*(n-1)  (10th-percentile min storage; severe
        #                              tail but not the single extreme)
        #   median = rank n/2         (median min storage value)
        n_sel = len(selected)
        ref_ranks = {
            'worst':  0,
            'p10':    max(int(round(0.1 * (n_sel - 1))), 0),
            'median': n_sel // 2,
        }
        ref_event_ids = {}
        rows_list = list(selected.iterrows())
        for ref_key, rk in ref_ranks.items():
            _, rrow = rows_list[rk]
            r_id_ref = int(rrow['realization_id'])
            eid = (f"R{r_id_ref:04d}_"
                   f"{pd.Timestamp(rrow['start']).date()}")
            ref_event_ids[ref_key] = eid
            dataset_ref_meta[ds][ref_key] = {
                'event_id': eid,
                'realization_id': r_id_ref,
                'start': pd.Timestamp(rrow['start']),
                'min_storage_date': pd.Timestamp(rrow['min_storage_date']),
                'min_storage_pct': float(rrow['event_min_storage_pct']),
                'rank': rk,
            }

        for event_idx, (_, row) in enumerate(selected.iterrows()):
            r_id = int(row['realization_id'])
            event_id = (f"R{r_id:04d}_"
                        f"{pd.Timestamp(row['start']).date()}")
            min_storage_date = pd.Timestamp(row['min_storage_date'])

            w_start, w_end = compute_fixed_extraction_window(
                min_storage_date, pad_before_wy=0, pad_after_wy=0,
            )
            window = realization_cache[r_id].loc[w_start:w_end]

            # Per-event scalars. storage_drawdown_pct is already a column of
            # the event_metrics CSV (peak storage at event start minus min
            # storage during event); reading directly from the row avoids
            # recomputation.
            scalars['event_storage_drawdown_pct'].append(
                float(row['storage_drawdown_pct']))
            scalars['event_min_storage_pct'].append(
                float(row['event_min_storage_pct']))
            scalars['event_total_release_mcm'].append(
                float(window['nyc_release'].sum()))
            mont_smooth = window['montague_flow'].rolling(
                7, center=True, min_periods=1).mean()
            scalars['event_min_montague_mcm'].append(float(mont_smooth.min()))

            if is_traj:
                for var_name, _, _, smooth, _, _, _ in VARIABLES:
                    s = window[var_name]
                    if smooth and smooth > 1:
                        s = s.rolling(smooth, center=True, min_periods=1).mean()
                    per_var_traces[var_name].append((event_id, s))

        dataset_event_scalars[ds] = {
            k: np.asarray(v, dtype=float) for k, v in scalars.items()
        }

        if is_traj and per_var_traces is not None:
            dataset_traces[ds] = {
                var_name: _traces_to_doy_df(per_var_traces[var_name])
                for var_name, _, _, _, _, _, _ in VARIABLES
            }
            for var_name, _, _, _, _, _, _ in VARIABLES:
                traces_df = dataset_traces[ds][var_name]
                refs = {}
                # Specific events (worst / p10 / median)
                for ref_key in ('worst', 'p10', 'median'):
                    eid = ref_event_ids.get(ref_key)
                    if eid is not None and eid in traces_df.columns:
                        refs[ref_key] = traces_df[eid]
                # Mean: DOY-wise mean across all events in this ensemble
                if traces_df.shape[1] > 0:
                    refs['mean'] = traces_df.mean(axis=1)
                dataset_refs[ds][var_name] = refs

        if dataset_ref_meta[ds]:
            for ref_key in ('worst', 'p10', 'median'):
                m = dataset_ref_meta[ds].get(ref_key)
                if m is None:
                    continue
                print(f"    {ref_key:>6}: {m['event_id']} "
                      f"(min_storage={m['min_storage_pct']:.1f}%, "
                      f"rank={m['rank']}/{n_sel - 1})")

        del realization_cache
        gc.collect()

    return dict(
        dataset_traces=dataset_traces,
        dataset_refs=dataset_refs,
        dataset_ref_meta=dataset_ref_meta,
        dataset_event_scalars=dataset_event_scalars,
        dataset_n_events=dataset_n_events,
    )


def _load_or_compute_focal_dynamics(
    all_data, focal_cells, sev_edges, mag_edges, thresholds,
):
    """Cache wrapper around _compute_focal_dynamics().

    Cache file name encodes SSI_WINDOW and the focal thresholds. Set
    REBUILD_CACHE=True (or delete the .pkl) to force a recomputation.
    """
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

    print(f"Fig10: focal-event dynamics + distributions "
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

    # 3. Build per-dataset grids and identify focal cells.
    # Focal-region thresholding is on T_W (Bonaccorso-Shiau interarrival
    # time minus mean event duration); see methods/return_period.py.
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

    # 4. Per-dataset extraction (cached): traces + worst-case for trajectory
    # ensembles, per-event scalars for all ensembles.
    dynamics = _load_or_compute_focal_dynamics(
        all_data, focal_cells, sev_edges, mag_edges, thresholds,
    )
    dataset_traces = dynamics['dataset_traces']
    dataset_refs = dynamics['dataset_refs']
    dataset_ref_meta = dynamics['dataset_ref_meta']
    dataset_event_scalars = dynamics['dataset_event_scalars']
    dataset_n_events = dynamics['dataset_n_events']

    # 5. FFMP zone thresholds by water-year DOY (time-varying)
    ffmp_by_wy = build_ffmp_by_wy_doy()

    # 6. Figure — 3 rows x 3 cols.
    # Cols 0-1: trajectory columns for the two climate-adjusted ensembles
    # (DOY quantile gradient + worst-case red overlay; storage row also gets
    # FFMP threshold lines). These two columns are directly comparable —
    # similarity here visualizes the manuscript's "operational signature is
    # invariant" claim.
    # Col 2: focal-event CDFs across all three ensembles. The differences
    # between lines visualize the "frequency / distribution differs" claim.
    n_rows = len(VARIABLES)
    n_cols = len(TRAJECTORY_DATASETS) + 1  # 2 trajectory + 1 CDF = 3

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.4 * n_cols, 3.6 * n_rows),
    )

    # Axis sharing — trajectory cols use (DOY x, variable y); CDF col uses
    # (nonexceedance prob x, variable y). The CDF column's y-axis differs
    # by row so it is not shared across rows; its x-axis (probability 0–1)
    # is shared across rows so x-tick labels appear only on the bottom CDF.
    for row in range(n_rows):
        axes[row, 1].sharey(axes[row, 0])  # trajectory cols share y row-wise
    for row in range(1, n_rows):
        axes[row, 0].sharex(axes[0, 0])     # trajectory cols share x col-wise
        axes[row, 1].sharex(axes[0, 1])
        axes[row, 2].sharex(axes[0, 2])     # CDF panels share x (probability)

    # Historical median annual NYC release (MCM) — denominator for the
    # release-ratio CDF (panel f). Computed once at startup, used at plot
    # time so it does not invalidate the per-event scalar cache.
    print("\n  Loading historical median annual NYC release...")
    hist_median_release_mcm = load_historical_median_annual_release_mcm()
    print(f"    historical median annual NYC release = "
          f"{hist_median_release_mcm:.1f} MCM")

    # Historical-drought overlays (1964, 1980): per-WY traces + scalars
    # drawn from the same reconstruction simulation. Empty dict if the
    # reconstruction is missing or every overlay is disabled.
    print("\n  Loading historical-drought overlays...")
    historical_overlays = load_historical_drought_overlays()

    sm_for_colorbar = None
    ref_handles = {}           # ref_key -> Line2D (p10 / median / h1964 / h1980)
    ffmp_handles = {}
    ensemble_handles = {}      # ds -> Line2D from CDF column (CDF line)
    cdf_ref_handle = None      # any horizontal CDF reference line

    for row_idx, var_def in enumerate(VARIABLES):
        (var_name, var_label, yscale, _, scalar_key, cdf_ylabel,
         cdf_ref_lines) = var_def

        # ── Trajectory columns (cols 0, 1) ───────────────────────────
        for col_idx, ds in enumerate(TRAJECTORY_DATASETS):
            ax = axes[row_idx, col_idx]
            traces_df = dataset_traces[ds][var_name]

            if traces_df.shape[1] >= 2:
                traces_df = traces_df.reindex(range(1, 367))
                grid = compute_doy_quantile_grid(
                    traces_df, n_levels=N_QUANTILE_LEVELS,
                )
                sm = plot_doy_quantile_gradient(
                    ax, grid, cmap=CMAP, vmin=0.0, vmax=1.0,
                )
                sm_for_colorbar = sm

            # FFMP Watch/Warning/Emergency thresholds (storage row only)
            if var_name == 'nyc_storage_pct' and ffmp_by_wy is not None:
                for zone in ['Watch', 'Warning', 'Emergency']:
                    if zone not in ffmp_by_wy.columns:
                        continue
                    zvals = ffmp_by_wy[zone].astype(float)
                    ln, = ax.plot(
                        zvals.index, zvals.values,
                        color=FFMP_ZONE_COLORS[zone], linestyle='--',
                        linewidth=1.0, alpha=0.95, zorder=4,
                    )
                    ffmp_handles.setdefault(zone, ln)
                    if col_idx == 0:
                        y0 = zvals.dropna().iloc[0] if zvals.dropna().size else None
                        if y0 is not None:
                            ax.text(
                                2, y0, zone,
                                fontsize=FONTSIZE_SMALL - 1,
                                color=FFMP_ZONE_COLORS[zone],
                                va='center', ha='left',
                                zorder=4.5,
                            )

            # Reference trajectories from the focal-event distribution:
            # p10 (severe-tail) and median (typical focal event).
            refs = dataset_refs[ds].get(var_name, {})
            for ref_key in REF_KEYS:
                s = refs.get(ref_key)
                if s is None or s.dropna().size == 0:
                    continue
                s_sorted = s.sort_index()
                ln, = ax.plot(
                    s_sorted.index, s_sorted.values, **REF_STYLES[ref_key],
                )
                ref_handles.setdefault(ref_key, ln)

            # Historical-drought overlays (1964 / 1980) — same DOY axis,
            # drawn from the reconstructed-streamflow simulation. The same
            # line is repeated on every trajectory column because the
            # historical reference does not depend on the climate-adjusted
            # ensemble; visual repetition reinforces it as a fixed anchor.
            for hist_key in HISTORICAL_OVERLAY_KEYS:
                if hist_key not in historical_overlays:
                    continue
                s = historical_overlays[hist_key]['traces'].get(var_name)
                if s is None or s.dropna().size == 0:
                    continue
                s_sorted = s.sort_index()
                ln, = ax.plot(
                    s_sorted.index, s_sorted.values,
                    **REF_STYLES[hist_key],
                )
                ref_handles.setdefault(hist_key, ln)

            if yscale == 'log':
                ax.set_yscale('log')
            else:
                ax.set_ylim(bottom=0)

            # X-axis: water-year DOY
            if row_idx == n_rows - 1:
                format_xaxis_water_year(ax)
                ax.set_xlabel(XAXIS_SUFFIX_LABEL, fontsize=FONTSIZE_LABEL)
            else:
                ax.set_xticks(MONTH_STARTS_WY)
                ax.set_xticklabels([])
            ax.set_xlim(1, 366)

            if col_idx == 0:
                ax.set_ylabel(var_label, fontsize=FONTSIZE_LABEL)
            if row_idx == 0:
                n = dataset_n_events[ds]
                ax.set_title(
                    f"{DATASET_LABELS.get(ds, ds)}\n"
                    f"n = {n} drought events",
                    fontsize=FONTSIZE_TITLE,
                )

            label_panel(
                ax, PANEL_LETTERS[row_idx * n_cols + col_idx],
                fontsize=FONTSIZE_LABEL, fontweight='normal',
            )
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_edgecolor('#333333')

        # ── CDF column (col 2) — empirical CDFs ───────────────────────
        # Axes: x = nonexceedance probability (0–1), y = metric value.
        # At N ≈ 5K–10K events per ensemble the empirical CDF is exact
        # at the resolution that matters; bootstrap bands were dropped
        # because they were imperceptible and added visual noise.
        # Each ensemble gets one crisp CDF line; small filled circles mark
        # the 10/50/90 percentiles for direct readout. Manuscript reference
        # horizontals (75% drawdown; 2x/3x historical median release; FFMP
        # Montague targets) stay; historical-drought anchors (1964, 1980)
        # are added as colored dashed horizontals when the overlay flags
        # are enabled.
        ax = axes[row_idx, 2]
        # Track ensemble values (for historical-anchor nonexceedance lookup
        # against the focal distribution rather than against any single
        # ensemble — the historical scalar is one number; it sits at a
        # different probability inside each ensemble's CDF).
        ensemble_sorted_values = {}
        for ds in DATASETS:
            values = dataset_event_scalars.get(ds, {}).get(
                scalar_key, None)
            # The release-row CDF is a derived ratio; compute it at plot
            # time from the cached cumulative-release scalar so the cache
            # stays generic.
            if (scalar_key == 'event_release_ratio_vs_historical'
                    and (values is None or values.size == 0)
                    and not np.isnan(hist_median_release_mcm)
                    and hist_median_release_mcm > 0):
                cum = dataset_event_scalars.get(ds, {}).get(
                    'event_total_release_mcm', np.array([]))
                if cum.size > 0:
                    values = cum / hist_median_release_mcm
            if values is None or values.size < 5:
                continue
            sorted_v, p_emp = _empirical_cdf(values)
            if sorted_v is None:
                continue
            ensemble_sorted_values[ds] = sorted_v

            color = DATASET_COLORS[ds]
            ln, = ax.plot(
                p_emp, sorted_v, color=color, linewidth=1.9, alpha=0.95,
                zorder=5,
            )
            ensemble_handles.setdefault(ds, ln)

            # Percentile callouts: filled circles + small inline labels at
            # 10/50/90. Labels use the ensemble color so the eye binds
            # them to the right curve when multiple curves overlap.
            for p in CDF_PCT_CALLOUTS:
                v = float(np.quantile(sorted_v, p))
                ax.plot(
                    [p], [v], marker='o', color=color,
                    markersize=4.5, markeredgecolor='white',
                    markeredgewidth=0.6, zorder=6, linestyle='None',
                )

        # Horizontal reference thresholds — one per row; configured in
        # VARIABLES. These mirror the role of FFMP zone lines on the
        # storage trajectory panels (a manuscript-grounded landmark on the
        # variable axis).
        for tgt_y, tgt_lbl in cdf_ref_lines:
            ln = ax.axhline(
                tgt_y, color='#444444', linestyle=':', linewidth=1.4,
                alpha=0.9, zorder=3,
            )
            if cdf_ref_handle is None:
                cdf_ref_handle = ln
            ax.text(
                0.99, tgt_y, f' {tgt_lbl}',
                fontsize=FONTSIZE_SMALL - 1, color='#444444',
                va='bottom', ha='right',
                transform=ax.get_yaxis_transform(), zorder=3.5,
            )
        # FFMP Montague target lines on the Montague-row CDF (kept in the
        # rendering loop because they pull from script-level constants).
        if var_name == 'montague_flow':
            for tgt, lbl in [
                (MONTAGUE_TARGET_NORMAL_MCM,
                 'FFMP Montague target (1750 cfs)'),
                (MONTAGUE_TARGET_EMERGENCY_MCM,
                 'FFMP Drought Emergency target (1100 cfs)'),
            ]:
                ln = ax.axhline(
                    tgt, color='#444444', linestyle=':', linewidth=1.4,
                    alpha=0.9, zorder=3,
                )
                if cdf_ref_handle is None:
                    cdf_ref_handle = ln
                ax.text(
                    0.99, tgt, f' {lbl}',
                    fontsize=FONTSIZE_SMALL - 1, color='#444444',
                    va='bottom', ha='right',
                    transform=ax.get_yaxis_transform(), zorder=3.5,
                )

        # Historical-drought anchors (1964 / 1980) — colored dashed
        # horizontals at the historical scalar value, plus per-ensemble
        # triangle markers at the empirical nonexceedance probability of
        # that value within each ensemble's distribution. The triangles
        # let the reader see *where* the historical drought sits in each
        # focal-region CDF without requiring tedious y-line tracing.
        for hist_key in HISTORICAL_OVERLAY_KEYS:
            if hist_key not in historical_overlays:
                continue
            h_scalars = historical_overlays[hist_key]['scalars']
            label = historical_overlays[hist_key]['label']
            if scalar_key == 'event_release_ratio_vs_historical':
                if (np.isnan(hist_median_release_mcm)
                        or hist_median_release_mcm <= 0):
                    continue
                hv = (h_scalars.get('event_total_release_mcm', float('nan'))
                      / hist_median_release_mcm)
            else:
                hv = h_scalars.get(scalar_key, float('nan'))
            if not np.isfinite(hv):
                continue
            color = REF_STYLES[hist_key]['color']
            ax.axhline(
                hv, color=color, linestyle='--', linewidth=1.2,
                alpha=0.9, zorder=4,
            )
            ax.text(
                0.01, hv, f' {label}',
                fontsize=FONTSIZE_SMALL - 1, color=color,
                va='bottom', ha='left',
                transform=ax.get_yaxis_transform(), zorder=4.5,
                fontweight='semibold',
            )
            for ds, sorted_v in ensemble_sorted_values.items():
                n_ens = sorted_v.size
                p_anchor = float(np.searchsorted(sorted_v, hv) / n_ens)
                if not (0.0 <= p_anchor <= 1.0):
                    continue
                ax.plot(
                    [p_anchor], [hv], marker='^',
                    color=color, markeredgecolor='white',
                    markeredgewidth=0.6, markersize=7, zorder=6,
                    linestyle='None',
                )

        ax.set_xlim(0, 1)
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        if row_idx == n_rows - 1:
            ax.set_xlabel('Nonexceedance probability',
                          fontsize=FONTSIZE_LABEL)
        else:
            ax.set_xticklabels([])
        ax.set_ylabel(cdf_ylabel, fontsize=FONTSIZE_LABEL)
        if row_idx == 0:
            ax.set_title(
                'Focal-event empirical CDFs\n'
                '(◯ markers: 10 / 50 / 90 percentile)',
                fontsize=FONTSIZE_TITLE,
            )

        label_panel(
            ax, PANEL_LETTERS[row_idx * n_cols + 2],
            fontsize=FONTSIZE_LABEL, fontweight='normal',
        )
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')

    # In-panel ensemble legend: place on the top CDF panel (row 0, col 2).
    cdf_legend_handles, cdf_legend_labels = [], []
    for ds in DATASETS:
        if ds in ensemble_handles:
            cdf_legend_handles.append(ensemble_handles[ds])
            cdf_legend_labels.append(
                f"{DATASET_LABELS_SHORT.get(ds, ds)} "
                f"(n={dataset_n_events[ds]})"
            )
    if cdf_legend_handles:
        axes[0, 2].legend(
            cdf_legend_handles, cdf_legend_labels,
            loc='lower right', fontsize=FONTSIZE_SMALL,
            frameon=True, framealpha=0.92,
        )

    fig.subplots_adjust(
        left=0.07, right=0.98, top=0.92, bottom=0.30,
        hspace=0.55, wspace=0.30,
    )

    fig.align_ylabels(axes[:, 0])

    # Bottom legend covers focal-event reference trajectories (p10, median),
    # optional historical-drought overlays (1964, 1980), FFMP zone thresholds
    # on the storage row, and the dotted CDF-reference horizontal lines.
    # Ensemble legend is in-panel (above) so it lives next to the lines it
    # labels.
    legend_order = list(REF_KEYS) + list(HISTORICAL_OVERLAY_KEYS)
    legend_handles, legend_labels = [], []
    for ref_key in legend_order:
        if ref_key in ref_handles:
            legend_handles.append(ref_handles[ref_key])
            legend_labels.append(REF_LABELS[ref_key])
    ffmp_level_map = {'Watch': 'L3', 'Warning': 'L4', 'Emergency': 'L5'}
    for zone in ['Watch', 'Warning', 'Emergency']:
        if zone in ffmp_handles:
            legend_handles.append(ffmp_handles[zone])
            legend_labels.append(
                f'FFMP {zone} threshold ({ffmp_level_map[zone]}, '
                f'seasonal rule curve)'
            )
    if cdf_ref_handle is not None:
        legend_handles.append(cdf_ref_handle)
        legend_labels.append(
            'CDF reference thresholds '
            '(manuscript / FFMP landmarks; see panel-side labels)'
        )

    if legend_handles:
        fig.legend(
            legend_handles, legend_labels,
            loc='lower center', bbox_to_anchor=(0.5, 0.13),
            ncol=2, fontsize=FONTSIZE_LEGEND,
            frameon=False,
        )

    if sm_for_colorbar is not None:
        cbar_ax = fig.add_axes([0.30, 0.035, 0.40, 0.018])
        cbar = fig.colorbar(sm_for_colorbar, cax=cbar_ax,
                            orientation='horizontal')
        cbar.ax.xaxis.set_label_position('top')
        cbar.ax.xaxis.set_ticks_position('bottom')
        cbar.set_label('% of focal water-years below y-axis value',
                       fontsize=FONTSIZE_LEGEND, labelpad=6)
        bin_edges = np.linspace(0.0, 1.0, CMAP_N_BINS + 1)
        cbar.set_ticks(bin_edges)
        cbar.set_ticklabels([f'{int(round(v * 100))}' for v in bin_edges])

    out_stem = os.path.join(FIG_OUTPUT_DIR, FIG_NAME_STEM)
    save_fig(fig, out_stem, dpi=DPI_PRINT)
    plt.close(fig)
    print(f"\nSaved (png/svg/pdf): {out_stem}")


if __name__ == '__main__':
    main()
