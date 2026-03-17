"""
SI10: Montague & Trenton Shortage Analysis by NYC Storage Zone

Diagnostic script showing how Montague flow target shortages and Trenton
flow target shortages distribute across FFMP drought zone levels, with
an additional sub-zone splitting Drought Emergency into above/below a
configurable NYC storage threshold.

FFMP Zone Definitions:
  - Zone 7*: Drought Emergency with NYC storage < EMERGENCY_STORAGE_THRESHOLD_PCT
  - Zone 6:  Drought Emergency (storage >= threshold)
  - Zone 5:  Drought Warning
  - Zone 4:  Drought Watch
  - Zone 3:  Normal operations
  - Zones 1-2: Flood conditions (high storage)
  * Zone 7 is a synthetic sub-zone, not an official FFMP designation.

Usage:
    python SI10_plot_montague_trenton_shortage_by_zone.py [dataset_id]

Examples:
    python SI10_plot_montague_trenton_shortage_by_zone.py stationary_ensemble
    python SI10_plot_montague_trenton_shortage_by_zone.py climate_adjusted_low
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings("ignore")

import pywrdrb
from methods.config import (
    FIG_DIR, NYC_RESERVOIRS, NYC_TOTAL_CAPACITY,
    verify_dataset_id,
)
from methods.plotting.styles import (
    DATASET_LABELS,
    DPI_HIGH, FONTSIZE_SMALL, FONTSIZE_MEDIUM, FONTSIZE_LARGE,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Output directory
FIG_OUTPUT_DIR = f"{FIG_DIR}/SI10_montague_trenton_shortage_by_zone"
os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

# Storage threshold (%) for splitting the Emergency zone
EMERGENCY_STORAGE_THRESHOLD_PCT = 5.0

# Locations to analyse
LOCATIONS = ['delMontague', 'delTrenton']
LOCATION_LABELS = {
    'delMontague': 'Montague',
    'delTrenton': 'Trenton',
}

# Zone definitions (zone 7 is the synthetic Emergency sub-zone;
# zones 1-2 are merged into zone 2 during data loading)
ZONE_DEFINITIONS = {
    7: {'name': f'Emergency\n(<{EMERGENCY_STORAGE_THRESHOLD_PCT:.0f}%)',
        'color': '#4B0082',
        'label': f'Drought Emergency Storage <{EMERGENCY_STORAGE_THRESHOLD_PCT:.0f}%'},
    6: {'name': f'Emergency\n(\u2265{EMERGENCY_STORAGE_THRESHOLD_PCT:.0f}%)',
        'color': '#8B0000',
        'label': f'Emergency (\u2265{EMERGENCY_STORAGE_THRESHOLD_PCT:.0f}%)'},
    5: {'name': 'Warning', 'color': '#FF4500', 'label': 'Warning (Zone 5)'},
    4: {'name': 'Watch',   'color': '#FFA500', 'label': 'Watch (Zone 4)'},
    3: {'name': 'Normal',  'color': '#32CD32', 'label': 'Normal (Zone 3)'},
    2: {'name': 'Flood',   'color': '#4169E1', 'label': 'Flood (Zones 1-2)'},
}

# Ordered list of zones to plot (most severe first)
ZONES_TO_PLOT = [7, 6, 5, 4, 3, 2]

# Tolerance for negligible shortage (MGD)
SHORTAGE_TOLERANCE_MGD = 0.1


# ============================================================================
# DATA LOADING
# ============================================================================

def load_shortage_and_zone_data(dataset_id):
    """
    Load Montague/Trenton shortage, FFMP zone, and NYC storage data.

    Returns dict mapping realization_id to DataFrame with columns:
        delMontague, delTrenton, nyc_zone, nyc_storage_pct, zone
    where ``zone`` incorporates the Emergency sub-zone split.
    """
    fname = f'./pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5'

    if not os.path.exists(fname):
        raise FileNotFoundError(
            f"Postprocessed data not found: {fname}\n"
            f"Run S3_run_postprocessing.sh first!"
        )

    print(f"Loading data from: {fname}")
    data = pywrdrb.Data()
    data.load_from_export(
        fname,
        results_sets=['shortage', 'res_level', 'res_storage'],
    )

    realizations = sorted(data.shortage[dataset_id].keys())
    print(f"  Found {len(realizations)} realizations")

    shortage_zone_data = {}

    for r in realizations:
        montague_shortage = data.shortage[dataset_id][r]['delMontague'].copy()
        trenton_shortage = data.shortage[dataset_id][r]['delTrenton'].copy()

        # Apply tolerance filter
        montague_shortage[montague_shortage < SHORTAGE_TOLERANCE_MGD] = 0.0
        trenton_shortage[trenton_shortage < SHORTAGE_TOLERANCE_MGD] = 0.0

        # NYC zone level and storage
        nyc_zone = data.res_level[dataset_id][r]['nyc']
        nyc_storage = data.res_storage[dataset_id][r][NYC_RESERVOIRS].sum(axis=1)
        nyc_storage_pct = 100.0 * nyc_storage / NYC_TOTAL_CAPACITY

        # Build DataFrame on common index
        df = pd.DataFrame({
            'delMontague': montague_shortage,
            'delTrenton': trenton_shortage,
            'nyc_zone': nyc_zone,
            'nyc_storage_pct': nyc_storage_pct,
        }).dropna()

        # Assign zone with Emergency sub-split and Flood merge
        df['zone'] = df['nyc_zone'].astype(int)
        df.loc[df['zone'] <= 2, 'zone'] = 2  # Merge flood zones
        emergency_low = (
            (df['zone'] == 6) &
            (df['nyc_storage_pct'] < EMERGENCY_STORAGE_THRESHOLD_PCT)
        )
        df.loc[emergency_low, 'zone'] = 7

        shortage_zone_data[r] = df

    return shortage_zone_data


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_shortage_by_zone(shortage_zone_data, location):
    """
    Aggregate shortage statistics by zone for a given location.

    Returns (stats_df, all_shortages_df).
    """
    all_shortage_rows = []

    for r, df in shortage_zone_data.items():
        shortage_days = df[df[location] > 0].copy()
        shortage_days['realization_id'] = r
        all_shortage_rows.append(shortage_days)

    if not all_shortage_rows:
        print(f"  No shortage days found for {LOCATION_LABELS[location]}!")
        return pd.DataFrame(), pd.DataFrame()

    all_shortages = pd.concat(all_shortage_rows, ignore_index=True)

    total_days = len(all_shortages)
    total_volume = all_shortages[location].sum()

    print(f"\n  {LOCATION_LABELS[location]} — "
          f"total shortage days: {total_days:,}, "
          f"total volume: {total_volume:,.0f} MG")

    zone_stats = []
    for z in sorted(ZONE_DEFINITIONS.keys(), reverse=True):
        zdf = all_shortages[all_shortages['zone'] == z]
        n = len(zdf)
        vol = zdf[location].sum()
        zone_stats.append({
            'zone': z,
            'zone_name': ZONE_DEFINITIONS[z]['name'].replace('\n', ' '),
            'n_shortage_days': n,
            'total_shortage_mg': vol,
            'mean_shortage_mg': zdf[location].mean() if n > 0 else 0,
            'pct_of_all_shortage_days': 100.0 * n / total_days if total_days else 0,
            'pct_of_all_shortage_volume': 100.0 * vol / total_volume if total_volume else 0,
        })

    return pd.DataFrame(zone_stats), all_shortages


def print_summary_statistics(stats_df, dataset_id, location):
    """Print summary statistics to console."""
    loc_label = LOCATION_LABELS[location]
    print(f"\n{'='*80}")
    print(f"{loc_label.upper()} SHORTAGE ANALYSIS: "
          f"{DATASET_LABELS.get(dataset_id, dataset_id)}")
    print(f"{'='*80}")

    print(f"\n{'Zone':<25} {'Days':>10} {'% Days':>10} "
          f"{'Volume (MG)':>15} {'% Volume':>10}")
    print("-" * 72)

    for _, row in stats_df.iterrows():
        if row['n_shortage_days'] == 0:
            continue
        print(f"{row['zone_name']:<25} "
              f"{row['n_shortage_days']:>10,} "
              f"{row['pct_of_all_shortage_days']:>10.1f} "
              f"{row['total_shortage_mg']:>15,.0f} "
              f"{row['pct_of_all_shortage_volume']:>10.1f}")

    print("-" * 72)


# ============================================================================
# PLOTTING
# ============================================================================

def plot_shortage_by_zone_summary(stats_df, dataset_id, location, fname=None):
    """
    Bar chart: % of shortage days and % of shortage volume by zone.
    """
    # Filter to zones with data
    plot_df = stats_df[stats_df['zone'].isin(ZONES_TO_PLOT)].copy()

    fig, (ax_days, ax_vol) = plt.subplots(1, 2, figsize=(13, 5))

    zone_names = [ZONE_DEFINITIONS[z]['name'] for z in ZONES_TO_PLOT]
    colors = [ZONE_DEFINITIONS[z]['color'] for z in ZONES_TO_PLOT]
    x = np.arange(len(ZONES_TO_PLOT))

    # Look up values by zone (handles missing zones gracefully)
    pct_days = []
    pct_vol = []
    for z in ZONES_TO_PLOT:
        row = plot_df[plot_df['zone'] == z]
        pct_days.append(row['pct_of_all_shortage_days'].values[0] if len(row) else 0)
        pct_vol.append(row['pct_of_all_shortage_volume'].values[0] if len(row) else 0)

    pct_days = np.array(pct_days)
    pct_vol = np.array(pct_vol)

    # Left panel: shortage days
    bars1 = ax_days.bar(x, pct_days, color=colors, alpha=0.8, edgecolor='black')
    ax_days.set_xticks(x)
    ax_days.set_xticklabels(zone_names, fontsize=FONTSIZE_SMALL)
    ax_days.set_ylabel('% of All Shortage Days', fontsize=FONTSIZE_MEDIUM)
    ax_days.set_xlabel('NYC Storage Zone', fontsize=FONTSIZE_MEDIUM)
    ax_days.set_title(f'{LOCATION_LABELS[location]} Shortage Days by Zone',
                      fontsize=FONTSIZE_LARGE)
    ax_days.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax_days.set_axisbelow(True)

    for bar, pct in zip(bars1, pct_days):
        if pct > 0:
            ax_days.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                         f'{pct:.1f}%', ha='center', va='bottom',
                         fontsize=FONTSIZE_SMALL)

    # Right panel: shortage volume
    bars2 = ax_vol.bar(x, pct_vol, color=colors, alpha=0.8, edgecolor='black')
    ax_vol.set_xticks(x)
    ax_vol.set_xticklabels(zone_names, fontsize=FONTSIZE_SMALL)
    ax_vol.set_ylabel('% of Total Shortage Volume', fontsize=FONTSIZE_MEDIUM)
    ax_vol.set_xlabel('NYC Storage Zone', fontsize=FONTSIZE_MEDIUM)
    ax_vol.set_title(f'{LOCATION_LABELS[location]} Shortage Volume by Zone',
                     fontsize=FONTSIZE_LARGE)
    ax_vol.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax_vol.set_axisbelow(True)

    for bar, pct in zip(bars2, pct_vol):
        if pct > 0:
            ax_vol.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f'{pct:.1f}%', ha='center', va='bottom',
                        fontsize=FONTSIZE_SMALL)

    dataset_label = DATASET_LABELS.get(dataset_id, dataset_id)
    fig.suptitle(f'{LOCATION_LABELS[location]} Flow Target Shortage: {dataset_label}',
                 fontsize=FONTSIZE_LARGE, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if fname is None:
        fname = (f"{FIG_OUTPUT_DIR}/"
                 f"shortage_by_zone_{location}_{dataset_id}.png")
    plt.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close()


def plot_shortage_magnitude_distributions(all_shortages, dataset_id,
                                          location, fname=None):
    """
    Histogram + box plot of shortage magnitudes by zone.
    """
    fig, (ax_hist, ax_box) = plt.subplots(2, 1, figsize=(14, 10))

    zone_data_list = []
    zone_labels_list = []
    zone_colors_list = []
    box_data = []
    box_labels = []
    box_colors = []

    for z in ZONES_TO_PLOT:
        zdf = all_shortages[all_shortages['zone'] == z]
        vals = zdf[location]
        if len(vals) == 0:
            continue
        zone_info = ZONE_DEFINITIONS[z]
        label = f"{zone_info['name'].replace(chr(10), ' ')} (n={len(vals):,})"
        zone_data_list.append(vals)
        zone_labels_list.append(label)
        zone_colors_list.append(zone_info['color'])
        box_data.append(vals.values)
        box_labels.append(zone_info['name'].replace('\n', ' '))
        box_colors.append(zone_info['color'])

    if not zone_data_list:
        plt.close()
        return

    # Top: stacked histogram
    max_shortage = np.percentile(
        np.concatenate([v.values for v in zone_data_list]), 99
    )
    bins = np.linspace(0, max_shortage * 1.1, 30)

    ax_hist.hist(zone_data_list, bins=bins, stacked=True,
                 color=zone_colors_list, label=zone_labels_list,
                 edgecolor='black', linewidth=0.3, alpha=0.8)
    ax_hist.set_xlabel('Shortage Magnitude (MGD)', fontsize=FONTSIZE_MEDIUM)
    ax_hist.set_ylabel('Frequency (days)', fontsize=FONTSIZE_MEDIUM)
    ax_hist.set_title(f'{LOCATION_LABELS[location]} Shortage Magnitudes by Zone',
                      fontsize=FONTSIZE_LARGE)
    ax_hist.legend(fontsize=FONTSIZE_SMALL, loc='upper right')
    ax_hist.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax_hist.set_axisbelow(True)

    # Bottom: box plot
    bp = ax_box.boxplot(box_data, labels=box_labels, patch_artist=True,
                        showfliers=False, widths=0.6,
                        medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax_box.set_ylabel('Shortage Magnitude (MGD)', fontsize=FONTSIZE_MEDIUM)
    ax_box.set_xlabel('NYC Storage Zone', fontsize=FONTSIZE_MEDIUM)
    ax_box.set_title('Shortage Distribution by Zone', fontsize=FONTSIZE_MEDIUM)
    ax_box.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax_box.set_axisbelow(True)
    ax_box.tick_params(axis='x', labelsize=FONTSIZE_SMALL)

    dataset_label = DATASET_LABELS.get(dataset_id, dataset_id)
    fig.suptitle(f'{LOCATION_LABELS[location]} Shortage Magnitude: {dataset_label}',
                 fontsize=FONTSIZE_LARGE, y=0.995)
    plt.tight_layout()

    if fname is None:
        fname = (f"{FIG_OUTPUT_DIR}/"
                 f"shortage_magnitude_{location}_{dataset_id}.png")
    plt.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    if len(sys.argv) > 1:
        dataset_id = sys.argv[1]
    else:
        dataset_id = 'stationary_ensemble'
        print(f"No dataset specified, using default: {dataset_id}")

    verify_dataset_id(dataset_id)

    print(f"\n{'='*80}")
    print("SI10: MONTAGUE & TRENTON SHORTAGE BY NYC STORAGE ZONE")
    print(f"Dataset: {dataset_id}")
    print(f"Emergency sub-zone threshold: <{EMERGENCY_STORAGE_THRESHOLD_PCT}% NYC storage")
    print(f"{'='*80}\n")

    shortage_zone_data = load_shortage_and_zone_data(dataset_id)

    for location in LOCATIONS:
        loc_label = LOCATION_LABELS[location]
        print(f"\n--- Analysing {loc_label} ---")

        stats_df, all_shortages = analyze_shortage_by_zone(
            shortage_zone_data, location
        )

        if len(stats_df) == 0 or len(all_shortages) == 0:
            print(f"  No {loc_label} shortages found — skipping.")
            continue

        print_summary_statistics(stats_df, dataset_id, location)
        plot_shortage_by_zone_summary(stats_df, dataset_id, location)
        plot_shortage_magnitude_distributions(
            all_shortages, dataset_id, location
        )

        # Save CSV summary
        csv_fname = (f"{FIG_OUTPUT_DIR}/"
                     f"shortage_by_zone_{location}_{dataset_id}.csv")
        stats_df.to_csv(csv_fname, index=False)
        print(f"  Saved CSV: {csv_fname}")

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
