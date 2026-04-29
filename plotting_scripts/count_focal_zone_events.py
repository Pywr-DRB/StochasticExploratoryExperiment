"""Count drought events per ensemble falling within the Fig9 focal region.

Replicates the focal-region identification used by
plotting_scripts/Fig9_plot_drought_satisficing_heatmap_2col.py and reports
event counts (total and per focal cell) for each ensemble.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from methods.config import (
    N_YEARS,
    GRID_N_BINS, FOCAL_WORST_STORAGE_THRESH,
    FOCAL_FRAC_THRESH, FOCAL_RP_THRESH_YEARS,
)
from methods.load import load_event_metrics
from methods.return_period import compute_return_period_grid_exceedance as compute_return_period_grid
from methods.plotting.styles import DATASET_LABELS
from methods.plotting.heatmap import (
    make_shared_edges_logmag, compute_min_storage_grid, compute_emergency_grid,
    identify_focal_region, assign_grid_bins,
)

DATASETS = ['stationary_ensemble', 'climate_adjusted_low', 'climate_adjusted_high']
SSI_WINDOW_DEFAULT = 3
MIN_COUNT = 1  # matches Fig9 call


def main():
    ssi_window = int(sys.argv[1]) if len(sys.argv) > 1 else SSI_WINDOW_DEFAULT
    print(f"Counting focal-zone events (SSI-{ssi_window})\n")

    all_data = {}
    for did in DATASETS:
        all_data[did] = load_event_metrics(did, ssi_window)
        print(f"  {DATASET_LABELS.get(did, did)}: {len(all_data[did])} total events "
              f"(after min_duration filter)")

    sev_edges, mag_edges, _, _ = make_shared_edges_logmag(
        all_data, DATASETS, n_bins=GRID_N_BINS)

    T_W_grids, frac_grids, min_grids = {}, {}, {}
    for did in DATASETS:
        _, _, T_W_grids[did], _ = compute_return_period_grid(
            all_data[did], sev_edges, mag_edges, N_YEARS, min_count=MIN_COUNT)
        frac_grids[did], _ = compute_emergency_grid(
            all_data[did], sev_edges, mag_edges, min_count=MIN_COUNT)
        min_grids[did], _ = compute_min_storage_grid(
            all_data[did], sev_edges, mag_edges, min_count=MIN_COUNT)

    focal_cells = identify_focal_region(T_W_grids, frac_grids, min_grids, DATASETS)
    print(f"\nFocal region: {len(focal_cells)} cells")
    print(f"  thresholds: T_W (joint exc.) <= {FOCAL_RP_THRESH_YEARS} yr (all), "
          f"frac < {FOCAL_FRAC_THRESH:.2f} (all), "
          f"min sto < {FOCAL_WORST_STORAGE_THRESH:.0f}% (any)")
    print(f"  cells (sev_bin, mag_bin): {sorted(focal_cells)}\n")

    if not focal_cells:
        print("No focal cells found — nothing to count.")
        return

    print("=" * 72)
    print(f"{'Ensemble':<35}{'Events in focal zone':>20}{'% of total':>15}")
    print("=" * 72)

    per_dataset_results = {}
    for did in DATASETS:
        df_binned = assign_grid_bins(all_data[did], sev_edges, mag_edges)
        mask = False
        for (i, j) in focal_cells:
            mask = mask | ((df_binned['sev_bin'] == i) & (df_binned['mag_bin'] == j))
        in_focal = df_binned[mask]
        n_focal = len(in_focal)
        n_total = len(df_binned)
        pct = 100.0 * n_focal / n_total if n_total else 0.0
        per_dataset_results[did] = (df_binned, in_focal)
        label = DATASET_LABELS.get(did, did)
        print(f"{label:<35}{n_focal:>20d}{pct:>14.2f}%")

    print("=" * 72)

    # Per-cell breakdown
    print("\nPer-cell event counts:")
    header = f"\n  {'cell (sev,mag)':<16}" + "".join(
        f"{DATASET_LABELS.get(d, d):>25}" for d in DATASETS)
    print(header)
    print("  " + "-" * (16 + 25 * len(DATASETS)))
    for (i, j) in sorted(focal_cells):
        sev_lo, sev_hi = sev_edges[i], sev_edges[i + 1]
        mag_lo, mag_hi = mag_edges[j], mag_edges[j + 1]
        line = f"  ({i:>2},{j:>2})       "
        for did in DATASETS:
            df_binned, _ = per_dataset_results[did]
            cnt = ((df_binned['sev_bin'] == i) & (df_binned['mag_bin'] == j)).sum()
            line += f"{cnt:>25d}"
        print(line)
        print(f"    sev [{sev_lo:.2f}, {sev_hi:.2f}), "
              f"mag [{mag_lo:.2f}, {mag_hi:.2f})")


if __name__ == '__main__':
    main()
