"""
SI17: Flow co-occurrence during SSI3 droughts vs. NYC storage outcomes.

For each SSI3 drought event, computes metrics measuring whether NYC inflow
and non-NYC Montague flow are simultaneously depressed (basin-wide drought),
then relates these to minimum NYC storage during the event.

Usage:
    python SI17_plot_flow_cooccurrence_storage.py [dataset_id]

If dataset_id is omitted, all datasets in DATASET_CONFIGS are processed.
"""

import sys
import os
import gc

import numpy as np
import pandas as pd
from scipy import stats

import pywrdrb

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from methods.config import (
    OUTPUT_DIR, FIG_DIR, DATASET_CONFIGS,
    verify_dataset_id,
)
from methods.load import load_event_metrics
from methods.plotting.flow_cooccurrence import (
    plot_cooccurrence_scatter_grid,
    plot_cooccurrence_pooled,
)


# =============================================================================
# Co-occurrence metric computation
# =============================================================================

def _compute_doy_medians(series, half_window=7):
    """Compute smoothed day-of-year medians for a daily timeseries.

    Uses a centered rolling window around each DOY to reduce noise.
    Returns a Series indexed by DOY (1-366).
    """
    df = pd.DataFrame({'value': series, 'doy': series.index.dayofyear})
    raw_median = df.groupby('doy')['value'].median()

    # Pad edges for circular smoothing (DOY wraps around)
    padded = pd.concat([raw_median.iloc[-half_window:],
                        raw_median,
                        raw_median.iloc[:half_window]])
    smoothed = padded.rolling(2 * half_window + 1, center=True, min_periods=1).median()
    # Trim padding
    smoothed = smoothed.iloc[half_window: half_window + len(raw_median)]
    smoothed.index = raw_median.index
    return smoothed


def compute_cooccurrence_metrics(data, dataset_id, events_df, min_event_days=14):
    """Compute flow co-occurrence metrics for each drought event.

    Parameters
    ----------
    data : pywrdrb.Data
        Loaded with results_sets=['inflow', 'major_flow', 'contribution'].
    dataset_id : str
        Dataset identifier.
    events_df : pd.DataFrame
        Event metrics with columns: realization_id, start, end, event_min_storage_pct, etc.
    min_event_days : int
        Skip events shorter than this.

    Returns
    -------
    pd.DataFrame
        One row per qualifying event with co-occurrence metrics.
    """
    results = []
    realizations = events_df['realization_id'].unique()

    for r in realizations:
        # Extract timeseries for this realization
        try:
            nyc_inflow = data.inflow[dataset_id][r]['nyc']
            montague_total = data.major_flow[dataset_id][r]['delMontague']
            contribution = data.contribution[dataset_id][r]
            if isinstance(contribution, pd.DataFrame):
                contribution = contribution['mrf_montagueTrenton_nyc']
        except (KeyError, IndexError):
            print(f'  Warning: could not load timeseries for realization {r}, skipping')
            continue

        non_nyc_montague = (montague_total - contribution).clip(lower=0)

        # DOY medians for baseline comparison
        nyc_doy_med = _compute_doy_medians(nyc_inflow)
        nonnyc_doy_med = _compute_doy_medians(non_nyc_montague)

        r_events = events_df[events_df['realization_id'] == r]

        for idx, event in r_events.iterrows():
            start = pd.Timestamp(event['start'])
            end = pd.Timestamp(event['end'])
            duration = (end - start).days
            if duration < min_event_days:
                continue

            mask = (nyc_inflow.index >= start) & (nyc_inflow.index <= end)
            inflow_w = nyc_inflow[mask]
            nonnyc_w = non_nyc_montague[mask]

            if len(inflow_w) < min_event_days:
                continue

            # Map DOY medians to the event window dates
            doys = inflow_w.index.dayofyear
            nyc_med_vals = nyc_doy_med.reindex(doys).values
            nonnyc_med_vals = nonnyc_doy_med.reindex(doys).values

            # Metric 1: Co-drought fraction
            inflow_below = inflow_w.values < nyc_med_vals
            nonnyc_below = nonnyc_w.values < nonnyc_med_vals
            co_drought_frac = float(np.mean(inflow_below & nonnyc_below))

            # Metric 2: Spearman correlation
            if inflow_w.std() > 0 and nonnyc_w.std() > 0:
                spearman_r = stats.spearmanr(inflow_w.values, nonnyc_w.values).statistic
            else:
                spearman_r = np.nan

            # Metric 3: Mean non-NYC deficit ratio
            deficit = np.clip((nonnyc_med_vals - nonnyc_w.values) / (nonnyc_med_vals + 1e-9), 0, None)
            mean_nonnyc_deficit = float(np.nanmean(deficit))

            results.append({
                'realization_id': r,
                'start': start,
                'co_drought_frac': co_drought_frac,
                'spearman_r': spearman_r,
                'mean_nonnyc_deficit': mean_nonnyc_deficit,
            })

    return pd.DataFrame(results)


def _print_summary(merged_dfs):
    """Print summary statistics: mean min storage by co-drought fraction bins."""
    for dataset_id, df in merged_dfs.items():
        if len(df) == 0:
            continue
        print(f'\n=== {dataset_id} ({len(df)} events) ===')

        # Overall correlation
        rho, pval = stats.spearmanr(df['co_drought_frac'], df['event_min_storage_pct'])
        print(f'  Spearman rho(co_drought_frac, min_storage) = {rho:.3f}  (p = {pval:.2e})')

        # Bin by co-drought fraction
        bins = [0, 0.3, 0.5, 0.7, 1.01]
        labels = ['<0.3', '0.3-0.5', '0.5-0.7', '>0.7']
        df['cdf_bin'] = pd.cut(df['co_drought_frac'], bins=bins, labels=labels, right=False)
        summary = df.groupby('cdf_bin', observed=False).agg(
            n=('event_min_storage_pct', 'count'),
            mean_min_storage=('event_min_storage_pct', 'mean'),
            median_min_storage=('event_min_storage_pct', 'median'),
            pct_emergency=('ffmp_zone_at_min', lambda x: (x == 'Emergency').mean() * 100),
        )
        print(summary.to_string())
        df.drop(columns='cdf_bin', inplace=True)


# =============================================================================
# Main
# =============================================================================

def main():
    # Parse arguments
    if len(sys.argv) > 1:
        dataset_id = sys.argv[1]
        verify_dataset_id(dataset_id)
        datasets = [dataset_id]
    else:
        datasets = list(DATASET_CONFIGS.keys())
        print(f'Processing all datasets: {datasets}')

    fig_subdir = f'{FIG_DIR}/SI17_flow_cooccurrence'
    os.makedirs(fig_subdir, exist_ok=True)

    merged_dfs = {}

    for did in datasets:
        print(f'\n--- {did} ---')

        # 1. Load pre-computed event metrics
        events = load_event_metrics(did, ssi_window=3)
        events['start'] = pd.to_datetime(events['start'])

        # 2. Load raw timeseries from postprocessed HDF5
        fname = f'{OUTPUT_DIR}/{did}_with_postprocessing.hdf5'
        if not os.path.exists(fname):
            print(f'  HDF5 not found: {fname}, skipping')
            continue

        print(f'  Loading timeseries from {fname}...')
        data = pywrdrb.Data()
        data.load_from_export(fname, results_sets=['inflow', 'major_flow', 'contribution'])

        # 3. Compute co-occurrence metrics
        print('  Computing co-occurrence metrics...')
        cooccurrence = compute_cooccurrence_metrics(data, did, events)
        print(f'  Computed metrics for {len(cooccurrence)} events')

        del data
        gc.collect()

        if len(cooccurrence) == 0:
            print('  No qualifying events, skipping')
            continue

        # 4. Merge with event metrics on (realization_id, start)
        merged = events.merge(cooccurrence, on=['realization_id', 'start'], how='inner')
        print(f'  Merged: {len(merged)} events')
        merged_dfs[did] = merged

    if not merged_dfs:
        print('\nNo data to plot.')
        return

    # 5. Summary statistics
    _print_summary(merged_dfs)

    # 6. Plot
    print(f'\nGenerating figures in {fig_subdir}...')
    plot_cooccurrence_scatter_grid(merged_dfs, fig_subdir)
    plot_cooccurrence_pooled(merged_dfs, fig_subdir)

    print('\nDone.')


if __name__ == '__main__':
    main()
