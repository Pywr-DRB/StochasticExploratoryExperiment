"""
SI20: NYC aggregate inflow duration-severity curves (Gold et al. 2024 Fig 5b style).

Diagnostic figure that verifies SSI-based drought analysis against an
absolute-value (non-standardized) drought metric. For each duration
n in {3, 6, 12, 24, 36} months, plots the minimum n-month rolling
cumulative inflow as boxplots across each ensemble dataset, overlaid
with the observed reconstruction value as a point.

Observed record construction mirrors
methods/drought_analysis.py::calculate_historic_observed_droughts so that
the observed point uses the identical daily series as the observed SSI.

Outputs
-------
Cache CSV: {DROUGHT_METRICS_DIR}/duration_severity_curves.csv
Figure:    {FIG_DIR}/SI20_duration_severity_curves.{png,svg}
"""

import argparse
import gc
import os
import sys
import warnings

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from methods.config import (  # noqa: E402
    BASELINE_DATASET,
    DROUGHT_METRICS_DIR,
    FIG_DIR,
    NYC_RESERVOIRS,
)
from methods.ensemble_utils import ENSEMBLE_SETS  # noqa: E402
from methods.load import (  # noqa: E402
    load_baseline_historical_flow,
    load_wrf1960s_historical_flow,
)
from methods.plotting.styles import (  # noqa: E402
    DATASET_COLORS,
    DATASET_LABELS,
    DPI_HIGH,
    HISTORIC_COLOR,
    HISTORIC_LABEL,
    apply_publication_style,
    save_fig,
)

warnings.filterwarnings("ignore")

DURATIONS_MONTHS = [3, 6, 12, 24, 36]
DATASETS = ['stationary_ensemble', 'climate_adjusted_low', 'climate_adjusted_high']
RECONSTRUCTION_LABEL = 'reconstruction'

CACHE_FILE = os.path.join(DROUGHT_METRICS_DIR, "duration_severity_curves.csv")
FIG_PATH_STEM = os.path.join(FIG_DIR, "SI20_duration_severity_curves")


# ---------------------------------------------------------------------------
# Observed record (mirrors drought_analysis.calculate_historic_observed_droughts)
# ---------------------------------------------------------------------------

def load_observed_nyc_daily():
    """Construct the daily NYC-aggregate inflow series used by observed SSI.

    Mirrors methods/drought_analysis.py:104-127. Returns pd.Series indexed by date,
    values in MGD.
    """
    Q = load_baseline_historical_flow(
        gage_flow=False, period='full', flowtype=BASELINE_DATASET,
    )
    Q.replace(0, np.nan, inplace=True)
    # Defensive: drop_columns for nyc_aggregate is ['delTrenton'] (won't be in catchment_inflow CSV)
    for col in ['delTrenton']:
        if col in Q.columns:
            Q.drop(columns=[col], inplace=True)

    if BASELINE_DATASET == 'wrfaorc_withObsScaled':
        Q_1960s = load_wrf1960s_historical_flow(gage_flow=False)
        Q_1960s.replace(0, np.nan, inplace=True)
        for col in ['delTrenton']:
            if col in Q_1960s.columns:
                Q_1960s.drop(columns=[col], inplace=True)
        Q_full = pd.concat([Q_1960s, Q], axis=0).sort_index()
        Q_full.replace(0, np.nan, inplace=True)
        Q_full.dropna(axis=0, how='any', inplace=True)
    else:
        Q_full = Q.copy()

    missing = [c for c in NYC_RESERVOIRS if c not in Q_full.columns]
    if missing:
        raise RuntimeError(
            f"NYC reservoir columns missing from observed record: {missing}"
        )
    return Q_full[NYC_RESERVOIRS].sum(axis=1)


# ---------------------------------------------------------------------------
# Core metric: per-duration minimum rolling cumulative inflow
# ---------------------------------------------------------------------------

def _min_rolling_sum_and_start(monthly_series, n_months):
    """Return (min_value, window_start_date) for n-month rolling sum of a Series."""
    rolling = monthly_series.rolling(window=n_months, min_periods=n_months).sum()
    rolling = rolling.dropna()
    if rolling.empty:
        return np.nan, pd.NaT
    end_idx = rolling.idxmin()
    return float(rolling.loc[end_idx]), end_idx - pd.DateOffset(months=n_months - 1)


def compute_observed_rows():
    """Compute duration-severity rows for the observed reconstruction record."""
    daily = load_observed_nyc_daily()
    monthly = daily.resample('MS').sum()
    monthly.replace(0, np.nan, inplace=True)
    monthly = monthly.dropna()
    print(f"  Observed monthly series: {monthly.index.min().date()} to "
          f"{monthly.index.max().date()} ({len(monthly)} months)")

    rows = []
    for n in DURATIONS_MONTHS:
        val, start = _min_rolling_sum_and_start(monthly, n)
        rows.append({
            'dataset_id': RECONSTRUCTION_LABEL,
            'realization_id': 0,
            'duration_months': n,
            'min_cumulative_flow_mg': val,
            'window_start_date': start,
        })
        print(f"  n={n:2d} months: min={val:.3e} MG, window starts {start.date() if pd.notna(start) else 'NaT'}")
    return pd.DataFrame(rows)


def compute_ensemble_rows(dataset_id):
    """Compute duration-severity rows for all realizations in one ensemble dataset.

    Reads NYC reservoir catchment inflows directly from the per-set
    catchment_inflow_mgd.hdf5 input files (synthetic ensemble inputs fed to
    pywrdrb). This is numerically identical to the pywrdrb `inflow` results
    set (verified by comparison) but loads in seconds rather than minutes.
    """
    set_specs = ENSEMBLE_SETS[dataset_id]
    for spec in set_specs:
        if not os.path.exists(spec.files['catchment_inflow']):
            raise FileNotFoundError(spec.files['catchment_inflow'])

    print(f"  Reading dates from {os.path.basename(set_specs[0].files['catchment_inflow'])} ...")
    with h5py.File(set_specs[0].files['catchment_inflow'], 'r') as f:
        dates = pd.to_datetime([d.decode() for d in f['cannonsville']['date'][:]])

    n_real_total = sum(len(s.realization_ids) for s in set_specs)
    print(f"  Building daily NYC-aggregate panel ({len(dates)} days x "
          f"{n_real_total} realizations from {len(set_specs)} sets) ...")

    panel = pd.DataFrame(index=dates,
                         columns=range(n_real_total),
                         dtype=np.float64)
    for spec in set_specs:
        with h5py.File(spec.files['catchment_inflow'], 'r') as f:
            # HDF5 keys are global realization IDs (set 1 = '0'-'99', set 2 = '100'-'199', ...)
            for global_id in spec.realization_ids:
                key = str(global_id)
                nyc_sum = np.zeros(len(dates), dtype=np.float64)
                for res in NYC_RESERVOIRS:
                    nyc_sum += f[res][key][:]
                panel[global_id] = nyc_sum

    gc.collect()

    # Resample to monthly volume (MG/month); replace zeros with NaN defensively
    print("  Resampling to calendar-monthly volume ...")
    monthly_panel = panel.resample('MS').sum()
    monthly_panel.replace(0, np.nan, inplace=True)
    del panel
    gc.collect()

    rows = []
    for n in DURATIONS_MONTHS:
        print(f"  Rolling-{n}-month minima across {n_real_total} realizations ...")
        rolling = monthly_panel.rolling(window=n, min_periods=n).sum()
        val_min = rolling.min(axis=0)          # Series: realization_id -> min MG
        idx_min = rolling.idxmin(axis=0)       # Series: realization_id -> end Timestamp
        for r in monthly_panel.columns:
            end = idx_min[r]
            start = end - pd.DateOffset(months=n - 1) if pd.notna(end) else pd.NaT
            rows.append({
                'dataset_id': dataset_id,
                'realization_id': int(r),
                'duration_months': n,
                'min_cumulative_flow_mg': float(val_min[r]),
                'window_start_date': start,
            })
    del monthly_panel
    gc.collect()
    return pd.DataFrame(rows)


def compute_all():
    print("=" * 70)
    print("SI20: Computing NYC aggregate inflow duration-severity curves")
    print(f"BASELINE_DATASET = {BASELINE_DATASET}")
    print("=" * 70)

    parts = []
    print("\n[observed reconstruction]")
    parts.append(compute_observed_rows())

    for did in DATASETS:
        print(f"\n[{did}]")
        parts.append(compute_ensemble_rows(did))

    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot(df):
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(10, 6.5))

    offsets = {
        'stationary_ensemble':    -0.22,
        'climate_adjusted_low':    0.0,
        'climate_adjusted_high':  +0.22,
    }
    x_positions = np.arange(len(DURATIONS_MONTHS))

    for did in DATASETS:
        color = DATASET_COLORS[did]
        data_per_duration = [
            df[(df.dataset_id == did) & (df.duration_months == n)]
              ['min_cumulative_flow_mg'].dropna().values
            for n in DURATIONS_MONTHS
        ]
        positions = x_positions + offsets[did]
        ax.boxplot(
            data_per_duration,
            positions=positions,
            widths=0.18,
            whis=(5, 95),
            showfliers=True,
            patch_artist=True,
            boxprops=dict(facecolor=color, alpha=0.4, edgecolor=color, linewidth=1.2),
            medianprops=dict(color=color, linewidth=2),
            whiskerprops=dict(color=color, linewidth=1.2),
            capprops=dict(color=color, linewidth=1.2),
            flierprops=dict(marker='o', markerfacecolor=color, markeredgecolor=color,
                            markersize=3, alpha=0.35, linestyle='none'),
        )

    # Observed overlay — horizontally aligned with the stationary_ensemble box
    obs = df[df.dataset_id == RECONSTRUCTION_LABEL].set_index('duration_months')
    obs_vals = [obs.loc[n, 'min_cumulative_flow_mg'] for n in DURATIONS_MONTHS]
    obs_x = x_positions + offsets['stationary_ensemble']
    ax.scatter(
        obs_x, obs_vals,
        marker='^', s=130, color=HISTORIC_COLOR,
        edgecolor='white', linewidth=1.2, zorder=10,
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(n) for n in DURATIONS_MONTHS])
    ax.set_xlim(x_positions.min() - 0.5, x_positions.max() + 0.5)
    ax.set_xlabel('Duration (months)')
    ax.set_ylabel('Minimum cumulative inflow (MG, log scale)')
    ax.set_yscale('log')
    ax.set_title(
        'NYC aggregate inflow duration-severity curves\n'
        '(min n-month cumulative catchment inflow into Cannonsville + Pepacton + Neversink)'
    )

    handles = [
        Patch(facecolor=DATASET_COLORS[d], alpha=0.4, edgecolor=DATASET_COLORS[d],
              label=DATASET_LABELS[d])
        for d in DATASETS
    ]
    handles.append(
        Line2D([0], [0], marker='^', linestyle='none',
               markerfacecolor=HISTORIC_COLOR, markeredgecolor='white',
               markersize=11, label=f'{HISTORIC_LABEL} ({BASELINE_DATASET})')
    )
    ax.legend(handles=handles, loc='upper left', frameon=True)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.text(
        0.5, 0.02,
        'Box = IQR | Whiskers = 5-95% | Outliers = dots',
        ha='center', va='bottom', fontsize=9, style='italic', color='#555555',
    )
    save_fig(fig, FIG_PATH_STEM, dpi=DPI_HIGH)
    plt.close(fig)
    print(f"Saved figure: {FIG_PATH_STEM}.png")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--recompute', action='store_true',
                    help='Force recompute of CSV cache (slow ~15-30 min).')
    ap.add_argument('--observed-only', action='store_true',
                    help='Compute only observed reconstruction rows (for fast smoke-testing).')
    ap.add_argument('--no-plot', action='store_true',
                    help='Skip plotting (compute & cache only).')
    args = ap.parse_args()

    os.makedirs(DROUGHT_METRICS_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    if args.observed_only:
        print("Running observed-only smoke test (no cache write).")
        df = compute_observed_rows()
        print("\nObserved rows:")
        print(df.to_string(index=False))
        return

    if args.recompute or not os.path.exists(CACHE_FILE):
        df = compute_all()
        df.to_csv(CACHE_FILE, index=False)
        print(f"\nSaved cache: {CACHE_FILE}  ({len(df)} rows)")
    else:
        print(f"Loading cached results: {CACHE_FILE}")
        df = pd.read_csv(CACHE_FILE, parse_dates=['window_start_date'])

    if args.no_plot:
        print("Skipping plot (--no-plot).")
        return

    plot(df)
    print("Done.")


if __name__ == '__main__':
    main()
