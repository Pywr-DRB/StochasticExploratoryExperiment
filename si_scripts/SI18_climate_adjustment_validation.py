"""
Validate that climate-adjusted ensembles realize the target % monthly-flow
shifts specified in data/nyc_inflow_selected_scenarios_PRMS_2020_2059.csv.

For NYC inflow (cannonsville + pepacton + neversink, MGD) the script computes:
  - historical baseline mean monthly flow (1980-2019, BASELINE_DATASET)
  - mean monthly flow for climate_adjusted_low and climate_adjusted_high
  - actual % difference (ensemble vs historical baseline) for each month
  - target % difference from the PRMS scenarios CSV (already loaded in
    methods.config as monthly_shift_scenarios)

Monthly mean follows the same definition used by the CMIP6 sibling pipeline:
  daily flow -> sum by (year, month) -> mean across years (pooled across
  realizations for ensembles).

Results are cached to outputs/{CONFIG_NAME}/data/climate_validation/ so the
slow ensemble aggregation only runs once. Pass --recompute to rebuild caches.

Usage:
    python SI18_climate_adjustment_validation.py [--recompute]
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from methods.config import (
    DATASET_CONFIGS, BASELINE_DATASET, NYC_RESERVOIRS,
    CONFIG_DIR, FIG_DIR, monthly_shift_scenarios,
)
from methods.load import load_baseline_historical_flow, load_and_combine_ensemble_sets
from methods.ensemble_utils import ENSEMBLE_SETS


CLIMATE_SCENARIOS = ['climate_adjusted_low', 'climate_adjusted_high']
CACHE_DIR = f"{CONFIG_DIR}/data/climate_validation"
FIG_SUBDIR = f"{FIG_DIR}/SI18_climate_adjustment_validation"

MONTH_LABELS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def cache_path(label):
    return f"{CACHE_DIR}/{label}_nyc_monthly_mean.csv"


def _monthly_mean_from_daily(daily_series):
    """Monthly total per (year, month) then mean across years -> Series indexed 1..12."""
    s = daily_series.dropna()
    monthly_totals = s.groupby([s.index.year, s.index.month]).sum()
    return monthly_totals.groupby(level=1).mean()


def compute_historical_nyc_monthly_mean():
    print("  Loading historical baseline flow (1980-2019)...")
    Q = load_baseline_historical_flow(
        period='baseline', gage_flow=True, flowtype=BASELINE_DATASET,
    )
    nyc_daily = Q[NYC_RESERVOIRS].sum(axis=1)
    return _monthly_mean_from_daily(nyc_daily)


def compute_ensemble_nyc_monthly_mean(dataset_id):
    print(f"  Loading ensemble for {dataset_id} (this is the slow step)...")
    Q_syn = load_and_combine_ensemble_sets(ENSEMBLE_SETS[dataset_id], by_site=True)

    missing = [s for s in NYC_RESERVOIRS if s not in Q_syn]
    if missing:
        raise KeyError(f"NYC sites missing from ensemble {dataset_id}: {missing}")

    nyc_syn = Q_syn['cannonsville'] + Q_syn['pepacton'] + Q_syn['neversink']
    n_realizations = nyc_syn.shape[1]
    print(f"  Aggregating monthly totals across {n_realizations} realizations...")

    all_monthly_totals = []
    for r in nyc_syn.columns:
        s = nyc_syn[r].dropna()
        all_monthly_totals.append(s.groupby([s.index.year, s.index.month]).sum())
    pooled = pd.concat(all_monthly_totals)
    monthly_mean = pooled.groupby(level=1).mean()
    return monthly_mean, n_realizations


def load_or_compute(label, compute_fn, recompute):
    """Return a Series (month index 1..12) of monthly mean flow in MGD.

    compute_fn() must return either a Series or a (Series, metadata) tuple.
    """
    path = cache_path(label)
    if (not recompute) and os.path.exists(path):
        print(f"  Using cached: {path}")
        df = pd.read_csv(path)
        return df.set_index('month')['monthly_mean_mgd'], None

    result = compute_fn()
    if isinstance(result, tuple):
        series, meta = result
    else:
        series, meta = result, None

    series = series.reindex(range(1, 13))
    series.index.name = 'month'
    series.name = 'monthly_mean_mgd'
    series.to_frame().reset_index().to_csv(path, index=False)
    print(f"  Saved cache: {path}")
    return series, meta


def plot_pct_diff(actual_low, actual_high, target_low, target_high,
                  n_low, n_high, out_path):
    months = np.arange(1, 13)
    width = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)

    for ax, actual, target, scen_label, n in [
        (axes[0], actual_low, target_low, 'climate_adjusted_low', n_low),
        (axes[1], actual_high, target_high, 'climate_adjusted_high', n_high),
    ]:
        ax.bar(months - width / 2, target.values, width=width,
               color='#bbbbbb', edgecolor='black', linewidth=0.4,
               label='Target (PRMS CSV)')
        ax.bar(months + width / 2, actual.values, width=width,
               color='#1f77b4', edgecolor='black', linewidth=0.4,
               label='Actual (ensemble)')
        ax.axhline(0, color='black', linewidth=0.6)
        ax.set_xticks(months)
        ax.set_xticklabels(MONTH_LABELS)
        ax.set_xlabel('Month')
        n_str = f' (n={n})' if n is not None else ''
        ax.set_title(f"{scen_label}{n_str}")
        ax.grid(axis='y', linestyle=':', alpha=0.5)
        ax.legend(loc='best', frameon=False)

    axes[0].set_ylabel('% difference vs. 1980–2019 historical')
    fig.suptitle('NYC inflow: target vs. realized monthly % flow change',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved figure: {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recompute', action='store_true',
                        help='Recompute monthly means even if cache exists.')
    args = parser.parse_args()

    print("=" * 60)
    print("SI18: Climate-adjustment monthly-flow validation")
    print("=" * 60)

    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(FIG_SUBDIR, exist_ok=True)

    print("\n[1/3] Historical baseline (1980-2019)")
    historical, _ = load_or_compute(
        'historical_baseline',
        compute_historical_nyc_monthly_mean,
        args.recompute,
    )

    ensemble_means = {}
    ensemble_sizes = {}
    for i, dataset_id in enumerate(CLIMATE_SCENARIOS, start=2):
        print(f"\n[{i}/3] {dataset_id}")
        series, meta = load_or_compute(
            dataset_id,
            lambda did=dataset_id: compute_ensemble_nyc_monthly_mean(did),
            args.recompute,
        )
        ensemble_means[dataset_id] = series
        ensemble_sizes[dataset_id] = meta  # n_realizations or None when cached

    print("\nComputing % differences and plotting...")
    actual_low = (ensemble_means['climate_adjusted_low'] - historical) / historical * 100
    actual_high = (ensemble_means['climate_adjusted_high'] - historical) / historical * 100

    target_low = monthly_shift_scenarios['low']
    target_high = monthly_shift_scenarios['high']
    target_low.index = range(1, 13)
    target_high.index = range(1, 13)

    out_path = f"{FIG_SUBDIR}/nyc_monthly_pct_diff.png"
    plot_pct_diff(
        actual_low=actual_low,
        actual_high=actual_high,
        target_low=target_low,
        target_high=target_high,
        n_low=ensemble_sizes['climate_adjusted_low'],
        n_high=ensemble_sizes['climate_adjusted_high'],
        out_path=out_path,
    )

    # Also save the % difference table next to the cached means.
    pct_table = pd.DataFrame({
        'month': range(1, 13),
        'historical_mgd': historical.values,
        'low_mgd': ensemble_means['climate_adjusted_low'].values,
        'high_mgd': ensemble_means['climate_adjusted_high'].values,
        'low_pct_actual': actual_low.values,
        'high_pct_actual': actual_high.values,
        'low_pct_target': target_low.values,
        'high_pct_target': target_high.values,
    })
    pct_table_path = f"{CACHE_DIR}/nyc_monthly_pct_diff.csv"
    pct_table.to_csv(pct_table_path, index=False)
    print(f"  Saved % diff table: {pct_table_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
