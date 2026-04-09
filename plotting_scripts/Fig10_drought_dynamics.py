"""
Fig10: Multi-Drought Dynamics Overlay

5-panel figure overlaying smoothed timeseries from multiple drought events
on a year-agnostic axis for cross-event comparison.

Panels:
  Top:  Drought duration bars
  (a)   NYC Aggregate Inflow (MGD)
  (b)   NYC Aggregate Storage (%) with FFMP zone boundaries
  (c)   NYC Releases to Montague (MGD)
  (d)   Montague Flow (MGD, log-scale)

Two modes:
  ENVELOPE_MODE = False  →  individual lines per selected event
  ENVELOPE_MODE = True   →  per-dataset min/max range + worst-case highlight

Drought events are selected from a specific cell in the severity x magnitude
grid (same discretization as the satisficing heatmap in Fig9), ranked by
worst-case storage.  Grid config is shared via methods.plotting.heatmap.

Usage:
    python Fig10_drought_dynamics.py
"""

import sys
import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from methods.config import (
    FIG_DIR, OUTPUT_DIR,
    DATASET_CONFIGS,
)
from methods.load import (
    load_event_metrics, load_rank_subset_from_export, load_ffmp_boundaries,
)
from methods.plotting.heatmap import (
    make_shared_edges_logmag, assign_grid_bins, select_from_grid_cell,
    GRID_N_BINS, GRID_TARGET_SEV_BIN, GRID_TARGET_MAG_BIN,
)
from methods.plotting.drought_dynamics import (
    extract_drought_timeseries,
    align_to_water_year,
    compute_reference_window_from_shifted,
    plot_drought_dynamics_overlay,
)

# ── Configuration ────────────────────────────────────────────────────────

SSI_WINDOW = 3
SMOOTHING_WINDOW = 7
MIN_COUNT = 1

# Envelope mode: True = show range across all events + worst-case highlight
#                False = individual lines for N_EVENTS_PER_DATASET events
ENVELOPE_MODE = True

# Number of events per dataset (only used when ENVELOPE_MODE = False)
N_EVENTS_PER_DATASET = 1

# Datasets to include (subset of DATASET_CONFIGS keys)
DATASETS = ['stationary_ensemble']
# DATASETS = list(DATASET_CONFIGS.keys())  # all datasets
RESULTS_SETS = ['inflow', 'res_storage', 'contribution', 'major_flow']

FIG_OUTPUT_DIR = os.path.join(FIG_DIR, 'Fig10_drought_dynamics')


# Data loading

def load_realization_data(dataset_id, realization_id):
    """Load timeseries data for a single realization."""
    fname = os.path.join(
        OUTPUT_DIR,
        f'{dataset_id}_with_postprocessing.hdf5'
    )
    data = load_rank_subset_from_export(
        fname, [realization_id], RESULTS_SETS, rank=0, size=1
    )
    return data


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print(f"Fig10: Multi-Drought Dynamics Overlay "
          f"({'envelope' if ENVELOPE_MODE else 'individual'} mode)")

    os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

    # 1. Load event metrics for all datasets
    all_data = {}
    for dataset_id in DATASETS:
        df = load_event_metrics(dataset_id, SSI_WINDOW)
        all_data[dataset_id] = df
        print(f"  {dataset_id}: {len(df)} events")

    # 2. Build shared severity x magnitude grid (log-spaced magnitude to match Fig9)
    sev_edges, mag_edges, sev_centers, mag_centers = make_shared_edges_logmag(
        all_data, DATASETS, n_bins=GRID_N_BINS,
    )

    print(f"\nGrid: {GRID_N_BINS} bins (log-magnitude)")
    print(f"  Severity range: [{sev_edges[0]:.2f}, {sev_edges[-1]:.2f}]")
    print(f"  Magnitude range: [{mag_edges[0]:.2f}, {mag_edges[-1]:.2f}] (log-spaced)")
    print(f"\nTarget cell: sev_bin={GRID_TARGET_SEV_BIN} "
          f"(~{sev_centers[GRID_TARGET_SEV_BIN]:.2f}), "
          f"mag_bin={GRID_TARGET_MAG_BIN} (~{mag_centers[GRID_TARGET_MAG_BIN]:.2f})")

    # 3. Select events from the target grid cell
    all_selected = []
    for dataset_id in DATASETS:
        df_binned = assign_grid_bins(all_data[dataset_id], sev_edges, mag_edges)
        if ENVELOPE_MODE:
            # Select ALL events in the cell, sorted by worst storage
            cell = df_binned[(df_binned['sev_bin'] == GRID_TARGET_SEV_BIN) &
                             (df_binned['mag_bin'] == GRID_TARGET_MAG_BIN)]
            selected = cell.sort_values('event_min_storage_pct', ascending=True)
        else:
            selected = select_from_grid_cell(
                df_binned, GRID_TARGET_SEV_BIN, GRID_TARGET_MAG_BIN,
                rank_col='event_min_storage_pct', ascending=True,
                n=N_EVENTS_PER_DATASET,
            )
        if len(selected) > 0:
            selected = selected.copy()
            selected['dataset_id'] = dataset_id
            all_selected.append(selected)
            print(f"  {dataset_id}: {len(selected)} events in cell")
            # Show the worst-case event
            worst = selected.iloc[0]
            print(f"    Worst: R{int(worst['realization_id']):04d} "
                  f"{worst['start']} to {worst['end']} "
                  f"(min_storage={worst['event_min_storage_pct']:.1f}%)")
        else:
            print(f"  {dataset_id}: no events in target cell")

    if not all_selected:
        print("No events found in target cell. Try different bin indices.")
        return

    all_events_df = pd.concat(all_selected, ignore_index=True)

    # 4. Build event descriptors and identify worst-case indices
    events = []
    for _, row in all_events_df.iterrows():
        events.append({
            'dataset_id': row['dataset_id'],
            'realization_id': int(row['realization_id']),
            'start': pd.Timestamp(row['start']),
            'end': pd.Timestamp(row['end']),
            'severity': row['severity'],
            'magnitude': row['magnitude'],
            'event_min_storage_pct': row['event_min_storage_pct'],
            'min_storage_date': pd.Timestamp(row['min_storage_date']),
        })

    # Find the worst-case event index per dataset (lowest min storage)
    highlight_indices = []
    if ENVELOPE_MODE:
        for did in DATASETS:
            did_indices = [i for i, ev in enumerate(events) if ev['dataset_id'] == did]
            if did_indices:
                worst_idx = min(did_indices,
                                key=lambda i: events[i]['event_min_storage_pct'])
                highlight_indices.append(worst_idx)
        print(f"\nHighlighted (worst per dataset): {highlight_indices}")

    # 5. Load data, extract timeseries, and align by water year of min storage
    #    Batch-load all realizations per dataset in one HDF5 open to avoid
    #    repeated file I/O (major speedup in envelope mode with many events).
    REFERENCE_WY_START = pd.Timestamp('2000-06-01')
    aligned_timeseries = [None] * len(events)

    ds_groups = {}
    for idx, ev in enumerate(events):
        ds_groups.setdefault(ev['dataset_id'], []).append(idx)

    for dataset_id, indices in ds_groups.items():
        unique_real_ids = sorted(set(events[i]['realization_id'] for i in indices))
        print(f"\nLoading: {dataset_id} — {len(unique_real_ids)} realization(s), "
              f"{len(indices)} event(s)...")
        fname = os.path.join(OUTPUT_DIR, f'{dataset_id}_with_postprocessing.hdf5')
        data = load_rank_subset_from_export(
            fname, unique_real_ids, RESULTS_SETS, rank=0, size=1
        )

        for idx in indices:
            ev = events[idx]
            # Extract only the actual drought period
            ts = extract_drought_timeseries(
                data, dataset_id, ev['realization_id'],
                ev['start'], ev['end'],
            )
            # Align by water year of min storage so months are preserved
            aligned, shifted_start, shifted_end = align_to_water_year(
                ts, ev['start'], ev['end'], ev['min_storage_date'],
                reference_wy_start=REFERENCE_WY_START,
            )
            aligned_timeseries[idx] = aligned
            # Store shifted dates for duration bars
            events[idx]['shifted_start'] = shifted_start
            events[idx]['shifted_end'] = shifted_end

        del data
        gc.collect()

    # 6. Compute reference window from shifted events (padded to month boundaries)
    reference_start, reference_end = compute_reference_window_from_shifted(events)
    print(f"\nReference window: {reference_start.date()} to {reference_end.date()}")
    print(f"  (events aligned so min-storage water year -> Jun 2000 - May 2001)")

    # 7. Load FFMP boundaries
    print("\nLoading FFMP boundaries...")
    ffmp = load_ffmp_boundaries()

    # 8. Plot
    mode_tag = 'envelope' if ENVELOPE_MODE else 'individual'
    fname = os.path.join(
        FIG_OUTPUT_DIR,
        f'Fig10_drought_dynamics_ssi{SSI_WINDOW}'
        f'_sev{GRID_TARGET_SEV_BIN}_mag{GRID_TARGET_MAG_BIN}_{mode_tag}.png'
    )
    fig = plot_drought_dynamics_overlay(
        events=events,
        aligned_timeseries=aligned_timeseries,
        reference_start=reference_start,
        reference_end=reference_end,
        smoothing_window=SMOOTHING_WINDOW,
        fname=fname,
        ffmp_boundaries=ffmp,
        envelope_mode=ENVELOPE_MODE,
        highlight_indices=highlight_indices,
    )
    plt.close(fig)

    print("\nDone.")


if __name__ == '__main__':
    main()
