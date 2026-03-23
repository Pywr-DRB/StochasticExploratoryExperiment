"""
07_select_focal_events.py

Identify extreme tail drought events from the conditional outcome analysis
and save their metadata for detailed timeseries visualization.

For a given SSI window, outcome metric, and magnitude bin, computes the
conditional exceedance probability P(outcome >= x | mag bin) for each event
and selects the event closest to each requested target probability.

The "longest consecutive shortage" event is always included by default.

Usage:
    python 07_select_focal_events.py [ssi_window]

Output:
    pywrdrb/focal_events/focal_events_ssi{window}.csv
"""

import sys
import os
import numpy as np
import pandas as pd

from methods.config import ROOT_DIR
from methods.load import load_drought_events

# ── Configuration ─────────────────────────────────────────────────────

DATASETS = ['stationary_ensemble']
MIN_DURATION = 30
MIN_SEVERITY = 1.0

# Magnitude bins (must match Fnew_conditional_outcomes.py)
MAG_BINS = [
    (0, 4, '< 4'),
    (4, 8, '4-8'),
    (8, np.inf, '>= 8'),
]

# Default focal event selection criteria
DEFAULT_OUTCOME_METRICS = [
    'max_consec_montague_days',
    'total_montague_shortage_mg',
]
DEFAULT_MAG_BIN_IDX = 2  # >= 8
DEFAULT_TARGET_PROBS = [0.05, 0.01]  # P(outcome >= x)

OUTPUT_DIR = os.path.join(ROOT_DIR, 'pywrdrb', 'focal_events')


def load_event_metrics(dataset_id, ssi_window):
    """Load event metrics CSV, filtering to analysis-relevant events."""
    fname = os.path.join(
        ROOT_DIR, 'pywrdrb', 'event_metrics',
        f'{dataset_id}_ssi{ssi_window}_event_metrics.csv'
    )
    df = pd.read_csv(fname)
    df['start'] = pd.to_datetime(df['start'])
    df['end'] = pd.to_datetime(df['end'])
    df = df[df['duration_days'] >= MIN_DURATION].copy()
    df['severity'] = df['severity'].abs()
    df['magnitude'] = df['magnitude'].abs()
    df = df[df['severity'] >= MIN_SEVERITY]
    return df


def select_focal_events(df, outcome_metric, mag_bin_idx, target_probs,
                         always_include_extreme=True):
    """
    Select focal drought events at specified exceedance probability levels.

    Parameters
    ----------
    df : pd.DataFrame
        Event metrics (already filtered to dataset).
    outcome_metric : str
        Column name for the outcome (e.g., 'max_consec_montague_days').
    mag_bin_idx : int
        Index into MAG_BINS for the magnitude bin to condition on.
    target_probs : list of float
        Target P(outcome >= x | mag bin) values to sample.
    always_include_extreme : bool
        If True, always include the most extreme event (rank 1).

    Returns
    -------
    pd.DataFrame
        Selected events with added columns: target_prob, actual_prob, rank.
    """
    lo, hi, label = MAG_BINS[mag_bin_idx]
    mask = (df['magnitude'] >= lo) & (df['magnitude'] < hi)
    bin_df = df[mask].copy()

    if len(bin_df) == 0:
        print(f"  WARNING: No events in magnitude bin {label}")
        return pd.DataFrame()

    # Sort by outcome metric descending (most extreme first)
    bin_df = bin_df.sort_values(outcome_metric, ascending=False).reset_index(drop=True)
    n = len(bin_df)

    # Compute rank-based exceedance probability for each event
    # P(outcome >= x_i) = rank_i / n  (rank 1 = most extreme)
    bin_df['rank'] = np.arange(1, n + 1)
    bin_df['exceedance_prob'] = bin_df['rank'] / n

    selected = []
    used_indices = set()

    # Always include the most extreme event
    if always_include_extreme:
        row = bin_df.iloc[0].copy()
        row['target_prob'] = 0.0
        row['actual_prob'] = row['exceedance_prob']
        row['selection_label'] = 'most_extreme'
        selected.append(row)
        used_indices.add(0)

    # Select event closest to each target probability
    for tp in target_probs:
        # Find event whose exceedance_prob is closest to target
        diffs = np.abs(bin_df['exceedance_prob'].values - tp)
        best_idx = np.argmin(diffs)

        if best_idx in used_indices:
            continue
        used_indices.add(best_idx)

        row = bin_df.iloc[best_idx].copy()
        row['target_prob'] = tp
        row['actual_prob'] = row['exceedance_prob']
        row['selection_label'] = f'P={tp}'
        selected.append(row)

    result = pd.DataFrame(selected)
    return result


def main():
    ssi_window = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    print(f"Selecting focal drought events (SSI-{ssi_window})")
    print(f"  Outcome metrics: {DEFAULT_OUTCOME_METRICS}")
    print(f"  Magnitude bin: {MAG_BINS[DEFAULT_MAG_BIN_IDX][2]}")
    print(f"  Target P(outcome >= x): {DEFAULT_TARGET_PROBS}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_focal = []
    for dataset_id in DATASETS:
        print(f"\n  Dataset: {dataset_id}")
        df = load_event_metrics(dataset_id, ssi_window)
        print(f"    Total events (filtered): {len(df)}")

        for outcome_metric in DEFAULT_OUTCOME_METRICS:
            print(f"\n    Metric: {outcome_metric}")
            focal = select_focal_events(
                df,
                outcome_metric=outcome_metric,
                mag_bin_idx=DEFAULT_MAG_BIN_IDX,
                target_probs=DEFAULT_TARGET_PROBS,
            )

            if len(focal) > 0:
                focal['dataset_id'] = dataset_id
                focal['ssi_window'] = ssi_window
                focal['outcome_metric'] = outcome_metric
                focal['mag_bin'] = MAG_BINS[DEFAULT_MAG_BIN_IDX][2]
                all_focal.append(focal)

                print(f"    Selected {len(focal)} focal events:")
                for _, row in focal.iterrows():
                    print(f"      {row['selection_label']:>14s}: "
                          f"R{int(row['realization_id']):04d}  "
                          f"{row['start'].strftime('%Y-%m-%d')} to "
                          f"{row['end'].strftime('%Y-%m-%d')}  "
                          f"mag={row['magnitude']:.1f}  "
                          f"{outcome_metric}={row[outcome_metric]:.1f}  "
                          f"P={row['actual_prob']:.4f}")

    if all_focal:
        combined = pd.concat(all_focal, ignore_index=True)
        out_fname = os.path.join(OUTPUT_DIR, f'focal_events_ssi{ssi_window}.csv')
        combined.to_csv(out_fname, index=False)
        print(f"\nSaved {len(combined)} focal events to: {out_fname}")
    else:
        print("\nNo focal events selected.")


if __name__ == '__main__':
    main()
