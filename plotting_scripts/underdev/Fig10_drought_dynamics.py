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

Drought events are selected from a specific cell in the severity x magnitude
grid (same discretization as the satisficing heatmap in Fig9), ranked by
worst-case storage.

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from methods.config import (
    FIG_DIR, OUTPUT_DIR,
    DATASET_CONFIGS,
)
from methods.load import (
    load_event_metrics, load_rank_subset_from_export, load_ffmp_boundaries,
)
from methods.plotting.heatmap import (
    make_shared_edges, assign_grid_bins, select_from_grid_cell,
)
from methods.plotting.drought_dynamics import (
    get_plot_window,
    extract_drought_timeseries,
    align_to_reference,
    compute_reference_window,
    plot_drought_dynamics_overlay,
)

# ── Configuration ────────────────────────────────────────────────────────

SSI_WINDOW = 3
SMOOTHING_WINDOW = 7
N_BINS = 16
MIN_COUNT = 1           # for grid binning

# Grid cell to focus on (severity_bin, magnitude_bin) — 0-indexed
# Adjust these to explore different regions of the sev x mag space
TARGET_SEV_BIN = 3
TARGET_MAG_BIN = 1

# Number of events to select per dataset from the target cell
N_EVENTS_PER_DATASET = 1

DATASETS = list(DATASET_CONFIGS.keys())
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
    print("Fig10: Multi-Drought Dynamics Overlay (grid-based selection)")

    os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

    # 1. Load event metrics for all datasets
    all_data = {}
    for dataset_id in DATASETS:
        df = load_event_metrics(dataset_id, SSI_WINDOW)
        all_data[dataset_id] = df
        print(f"  {dataset_id}: {len(df)} events")

    # 2. Build shared severity x magnitude grid
    sev_edges, mag_edges, sev_centers, mag_centers = make_shared_edges(
        all_data, DATASETS, n_bins=N_BINS,
    )
    print(f"\nGrid: {N_BINS} bins")
    print(f"  Severity range: [{sev_edges[0]:.2f}, {sev_edges[-1]:.2f}]")
    print(f"  Magnitude range: [{mag_edges[0]:.2f}, {mag_edges[-1]:.2f}]")
    print(f"\nTarget cell: sev_bin={TARGET_SEV_BIN} "
          f"(~{sev_centers[TARGET_SEV_BIN]:.2f}), "
          f"mag_bin={TARGET_MAG_BIN} (~{mag_centers[TARGET_MAG_BIN]:.2f})")

    # 3. Select events from the target grid cell (worst storage first)
    all_selected = []
    for dataset_id in DATASETS:
        df_binned = assign_grid_bins(all_data[dataset_id], sev_edges, mag_edges)
        selected = select_from_grid_cell(
            df_binned, TARGET_SEV_BIN, TARGET_MAG_BIN,
            rank_col='event_min_storage_pct', ascending=True,
            n=N_EVENTS_PER_DATASET,
        )
        if len(selected) > 0:
            selected = selected.copy()
            selected['dataset_id'] = dataset_id
            all_selected.append(selected)
            for _, row in selected.iterrows():
                print(f"  {dataset_id} R{int(row['realization_id']):04d}: "
                      f"{row['start']} to {row['end']} "
                      f"(sev={row['severity']:.2f}, mag={row['magnitude']:.1f}, "
                      f"min_storage={row['event_min_storage_pct']:.1f}%)")
        else:
            print(f"  {dataset_id}: no events in target cell")

    if not all_selected:
        print("No events found in target cell. Try different bin indices.")
        return

    all_events_df = pd.concat(all_selected, ignore_index=True)

    # 4. Build event descriptors
    events = []
    for _, row in all_events_df.iterrows():
        events.append({
            'dataset_id': row['dataset_id'],
            'realization_id': int(row['realization_id']),
            'start': pd.Timestamp(row['start']),
            'end': pd.Timestamp(row['end']),
            'severity': row['severity'],
            'magnitude': row['magnitude'],
        })

    # 5. Compute reference window
    reference_start, reference_end = compute_reference_window(events)
    print(f"\nReference window: {reference_start.date()} to {reference_end.date()}")

    # 6. Load data and extract timeseries
    aligned_timeseries = [None] * len(events)

    groups = {}
    for idx, ev in enumerate(events):
        key = (ev['dataset_id'], ev['realization_id'])
        groups.setdefault(key, []).append(idx)

    for (dataset_id, realization_id), indices in groups.items():
        print(f"\nLoading: {dataset_id} R{realization_id:04d}...")
        data = load_realization_data(dataset_id, realization_id)

        for idx in indices:
            ev = events[idx]
            plot_start, plot_end = get_plot_window(ev['start'], ev['end'])
            ts = extract_drought_timeseries(
                data, dataset_id, realization_id, plot_start, plot_end
            )
            aligned = align_to_reference(ts, plot_start, reference_start)
            aligned_timeseries[idx] = aligned

            print(f"  Event {idx}: {ev['start'].date()} to {ev['end'].date()} "
                  f"(window: {plot_start.date()} to {plot_end.date()})")

        del data
        gc.collect()

    # 7. Load FFMP boundaries
    print("\nLoading FFMP boundaries...")
    ffmp = load_ffmp_boundaries()

    # 8. Plot
    fname = os.path.join(
        FIG_OUTPUT_DIR,
        f'Fig10_drought_dynamics_ssi{SSI_WINDOW}'
        f'_sev{TARGET_SEV_BIN}_mag{TARGET_MAG_BIN}.png'
    )
    fig = plot_drought_dynamics_overlay(
        events=events,
        aligned_timeseries=aligned_timeseries,
        reference_start=reference_start,
        reference_end=reference_end,
        smoothing_window=SMOOTHING_WINDOW,
        fname=fname,
        ffmp_boundaries=ffmp,
        all_event_data=all_data,
    )
    plt.close(fig)

    print("\nDone.")


if __name__ == '__main__':
    main()
