"""
SI: NYC Diversion Shortage Analysis by FFMP Zone

Diagnostic script to investigate when NYC diversion shortages occur
relative to FFMP drought zone levels.

Key question: Are NYC diversion shortages only during drought
(Watch/Warning/Emergency) periods, or do they also occur during "normal"
periods?

FFMP Zone Definitions:
  - Zone 6: Drought Emergency (most severe)
  - Zone 5: Drought Watch
  - Zone 4: Drought Warning
  - Zone 3: Normal operations
  - Zones 1-2: Flood conditions (high storage)

NYC Diversion Limit:
  - Base limit: 800 MGD (running average from June 1st)
  - Decreases when NYC storage enters drought zones
  - Shortage = Demand - Actual Diversion (when positive)

Usage:
    python SI_NYC_diversion_shortage_by_zone.py [dataset_id]

Examples:
    python SI_NYC_diversion_shortage_by_zone.py stationary_ensemble
    python SI_NYC_diversion_shortage_by_zone.py climate_adjusted_low
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import pywrdrb
from methods.config import (
    FIG_DIR, NYC_RESERVOIRS, NYC_TOTAL_CAPACITY,
    verify_dataset_id
)
from methods.plotting.styles import (
    DATASET_COLORS, DATASET_LABELS,
    DPI_HIGH, FONTSIZE_SMALL, FONTSIZE_MEDIUM, FONTSIZE_LARGE
)

# Output directory
FIG_OUTPUT_DIR = f"{FIG_DIR}/SI_NYC_shortage_by_zone"
os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

# Zone definitions
ZONE_DEFINITIONS = {
    6: {'name': 'Emergency', 'color': '#8B0000', 'label': 'Emergency (Zone 6)'},
    5: {'name': 'Watch', 'color': '#FF4500', 'label': 'Watch (Zone 5)'},
    4: {'name': 'Warning', 'color': '#FFA500', 'label': 'Warning (Zone 4)'},
    3: {'name': 'Normal', 'color': '#32CD32', 'label': 'Normal (Zone 3)'},
    2: {'name': 'Flood', 'color': '#4169E1', 'label': 'Flood (Zones 1-2)'},
    1: {'name': 'Flood', 'color': '#4169E1', 'label': 'Flood (Zones 1-2)'},
}

# Drought zones for analysis
DROUGHT_ZONES = [4, 5, 6]  # Warning, Watch, Emergency
NORMAL_ZONE = 3
FLOOD_ZONES = [1, 2]


def load_shortage_and_zone_data(dataset_id):
    """
    Load NYC diversion shortage and FFMP zone level data.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier

    Returns
    -------
    dict
        Dictionary mapping realization_id to DataFrame with columns:
        - nyc_diversion: actual NYC diversion (MGD)
        - nyc_demand: NYC demand (MGD)
        - nyc_shortage: shortage = max(0, demand - diversion)
        - nyc_zone: FFMP zone level (1-6)
        - nyc_storage_pct: NYC aggregate storage (%)
    """
    fname = f'./pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5'

    if not os.path.exists(fname):
        raise FileNotFoundError(
            f"Postprocessed data not found: {fname}\n"
            f"Run 04_postprocess_data.py first!"
        )

    print(f"Loading data from: {fname}")
    data = pywrdrb.Data()
    data.load_from_export(
        fname,
        results_sets=['ibt_diversions', 'ibt_demands', 'res_level', 'res_storage']
    )

    realizations = sorted(data.ibt_diversions[dataset_id].keys())
    print(f"  Found {len(realizations)} realizations")

    shortage_zone_data = {}

    for r in realizations:
        # NYC diversions and demands
        nyc_diversion = data.ibt_diversions[dataset_id][r]['delivery_nyc']
        nyc_demand = data.ibt_demands[dataset_id][r]['demand_nyc']
        nyc_shortage = (nyc_demand - nyc_diversion).clip(lower=0)
        nyc_shortage[nyc_shortage < 0.1] = 0.0  # Filter out negligible shortages

        # NYC zone level
        nyc_zone = data.res_level[dataset_id][r]['nyc']

        # NYC storage percentage
        nyc_storage = data.res_storage[dataset_id][r][NYC_RESERVOIRS].sum(axis=1)
        nyc_storage_pct = 100.0 * nyc_storage / NYC_TOTAL_CAPACITY

        # Combine into DataFrame
        df = pd.DataFrame({
            'nyc_diversion': nyc_diversion,
            'nyc_demand': nyc_demand,
            'nyc_shortage': nyc_shortage,
            'nyc_zone': nyc_zone,
            'nyc_storage_pct': nyc_storage_pct
        })

        shortage_zone_data[r] = df

    return shortage_zone_data


def analyze_shortage_by_zone(shortage_zone_data):
    """
    Analyze NYC diversion shortages by FFMP zone.

    Parameters
    ----------
    shortage_zone_data : dict
        Dictionary mapping realization_id to shortage/zone DataFrame

    Returns
    -------
    pd.DataFrame
        Summary statistics with columns:
        - zone: FFMP zone number
        - n_shortage_days: total number of days with shortage in this zone
        - total_shortage_mg: total shortage volume in this zone (MG)
        - mean_shortage_mg: mean daily shortage when shortage occurs (MG)
        - pct_of_all_shortage_days: percentage of all shortage days in this zone
        - pct_of_all_shortage_volume: percentage of total shortage volume in this zone
    """
    # Aggregate across all realizations
    all_shortage_days = []

    for r, df in shortage_zone_data.items():
        # Filter to days with shortage
        shortage_days = df[df['nyc_shortage'] > 0].copy()
        shortage_days['realization_id'] = r
        all_shortage_days.append(shortage_days)

    if len(all_shortage_days) == 0:
        print("  Warning: No shortage days found!")
        return pd.DataFrame()

    all_shortages = pd.concat(all_shortage_days, ignore_index=True)

    print(f"\nTotal shortage days across all realizations: {len(all_shortages):,}")

    # Calculate statistics by zone
    zone_stats = []

    total_shortage_days = len(all_shortages)
    total_shortage_volume = all_shortages['nyc_shortage'].sum()

    for zone_num in sorted(ZONE_DEFINITIONS.keys(), reverse=True):
        zone_shortages = all_shortages[all_shortages['nyc_zone'] == zone_num]

        n_days = len(zone_shortages)
        total_vol = zone_shortages['nyc_shortage'].sum()
        mean_shortage = zone_shortages['nyc_shortage'].mean() if n_days > 0 else 0
        pct_days = 100.0 * n_days / total_shortage_days if total_shortage_days > 0 else 0
        pct_vol = 100.0 * total_vol / total_shortage_volume if total_shortage_volume > 0 else 0

        zone_stats.append({
            'zone': zone_num,
            'zone_name': ZONE_DEFINITIONS[zone_num]['name'],
            'n_shortage_days': n_days,
            'total_shortage_mg': total_vol,
            'mean_shortage_mg': mean_shortage,
            'pct_of_all_shortage_days': pct_days,
            'pct_of_all_shortage_volume': pct_vol,
        })

    stats_df = pd.DataFrame(zone_stats)

    return stats_df, all_shortages


def calculate_realization_statistics(shortage_zone_data):
    """
    Calculate per-realization statistics on shortage by zone.

    Returns
    -------
    pd.DataFrame
        Per-realization statistics with columns for each zone's shortage frequency
    """
    realization_stats = []

    for r, df in shortage_zone_data.items():
        stats = {'realization_id': r}

        # Total days with shortage
        total_shortage_days = (df['nyc_shortage'] > 0).sum()
        stats['total_shortage_days'] = total_shortage_days

        if total_shortage_days > 0:
            # Shortage days by zone
            for zone_num in sorted(ZONE_DEFINITIONS.keys(), reverse=True):
                zone_shortage_days = ((df['nyc_shortage'] > 0) &
                                     (df['nyc_zone'] == zone_num)).sum()
                pct = 100.0 * zone_shortage_days / total_shortage_days
                stats[f'pct_shortage_days_zone{zone_num}'] = pct
        else:
            for zone_num in sorted(ZONE_DEFINITIONS.keys(), reverse=True):
                stats[f'pct_shortage_days_zone{zone_num}'] = 0

        realization_stats.append(stats)

    return pd.DataFrame(realization_stats)


def plot_shortage_by_zone_summary(stats_df, dataset_id, fname=None):
    """
    Create summary figure showing shortage distribution by zone.

    Layout:
      - Left: Bar chart of percentage of shortage days by zone
      - Right: Bar chart of percentage of shortage volume by zone
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_days, ax_volume = axes

    zones = stats_df['zone'].values
    zone_names = stats_df['zone_name'].values
    colors = [ZONE_DEFINITIONS[z]['color'] for z in zones]

    # Left panel: Percentage of shortage days
    pct_days = stats_df['pct_of_all_shortage_days'].values
    bars1 = ax_days.bar(zone_names, pct_days, color=colors, alpha=0.8, edgecolor='black')
    ax_days.set_ylabel('% of All Shortage Days', fontsize=FONTSIZE_MEDIUM)
    ax_days.set_xlabel('FFMP Zone', fontsize=FONTSIZE_MEDIUM)
    ax_days.set_title('NYC Diversion Shortage Days by Zone', fontsize=FONTSIZE_LARGE)
    ax_days.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax_days.set_axisbelow(True)

    # Add percentage labels on bars
    for bar, pct in zip(bars1, pct_days):
        height = bar.get_height()
        if pct > 0:
            ax_days.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{pct:.1f}%', ha='center', va='bottom',
                        fontsize=FONTSIZE_SMALL)

    # Right panel: Percentage of shortage volume
    pct_vol = stats_df['pct_of_all_shortage_volume'].values
    bars2 = ax_volume.bar(zone_names, pct_vol, color=colors, alpha=0.8, edgecolor='black')
    ax_volume.set_ylabel('% of Total Shortage Volume', fontsize=FONTSIZE_MEDIUM)
    ax_volume.set_xlabel('FFMP Zone', fontsize=FONTSIZE_MEDIUM)
    ax_volume.set_title('NYC Diversion Shortage Volume by Zone', fontsize=FONTSIZE_LARGE)
    ax_volume.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax_volume.set_axisbelow(True)

    # Add percentage labels on bars
    for bar, pct in zip(bars2, pct_vol):
        height = bar.get_height()
        if pct > 0:
            ax_volume.text(bar.get_x() + bar.get_width()/2., height + 1,
                          f'{pct:.1f}%', ha='center', va='bottom',
                          fontsize=FONTSIZE_SMALL)

    # Add dataset label
    dataset_label = DATASET_LABELS.get(dataset_id, dataset_id)
    fig.suptitle(f'NYC Diversion Shortage Analysis: {dataset_label}',
                 fontsize=FONTSIZE_LARGE, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if fname is None:
        fname = f"{FIG_OUTPUT_DIR}/shortage_by_zone_{dataset_id}.png"

    plt.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"\nSaved figure: {fname}")
    plt.close()


def plot_shortage_magnitude_distributions(all_shortages, dataset_id, fname=None):
    """
    Plot distributions of shortage magnitudes by zone.

    Creates 2-panel figure:
    - Top: Histogram of shortage magnitudes (0-500 MGD)
    - Bottom: Box plot showing distribution by zone
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    ax_hist, ax_box = axes

    # Get drought + normal zones (exclude flood zones for clarity)
    zones_to_plot = [6, 5, 4, 3]  # Emergency, Watch, Warning, Normal
    zone_labels = [ZONE_DEFINITIONS[z]['name'] for z in zones_to_plot]
    zone_colors = [ZONE_DEFINITIONS[z]['color'] for z in zones_to_plot]

    # Top panel: Stacked bar chart of shortage magnitudes (0-500 MGD)
    bins = np.linspace(0, 500, 26)  # 0-500 MGD in 20 MGD bins

    # Collect data for each zone
    zone_data_list = []
    zone_labels_list = []
    zone_colors_list = []

    for zone_num, color, label in zip(zones_to_plot, zone_colors, zone_labels):
        zone_data = all_shortages[all_shortages['nyc_zone'] == zone_num]['nyc_shortage']
        if len(zone_data) > 0:
            zone_data_list.append(zone_data)
            zone_labels_list.append(f'{label} (n={len(zone_data):,})')
            zone_colors_list.append(color)

    # Create stacked histogram
    ax_hist.hist(zone_data_list, bins=bins, stacked=True,
                color=zone_colors_list, label=zone_labels_list,
                edgecolor='black', linewidth=0.3, alpha=0.8)

    ax_hist.set_xlabel('Shortage Magnitude (MGD)', fontsize=FONTSIZE_MEDIUM)
    ax_hist.set_ylabel('Frequency (days)', fontsize=FONTSIZE_MEDIUM)
    ax_hist.set_title('Distribution of NYC Diversion Shortage Magnitudes by Zone',
                     fontsize=FONTSIZE_LARGE)
    ax_hist.legend(fontsize=FONTSIZE_SMALL, loc='upper right')
    ax_hist.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax_hist.set_xlim(0, 500)
    ax_hist.set_axisbelow(True)

    # Bottom panel: Box plot
    box_data = []
    box_labels = []
    box_colors = []

    for zone_num, color, label in zip(zones_to_plot, zone_colors, zone_labels):
        zone_data = all_shortages[all_shortages['nyc_zone'] == zone_num]['nyc_shortage']
        if len(zone_data) > 0:
            box_data.append(zone_data.values)
            box_labels.append(label)
            box_colors.append(color)

    # Create box plot with black median lines
    bp = ax_box.boxplot(box_data, labels=box_labels, patch_artist=True,
                        showfliers=False, widths=0.6,
                        medianprops=dict(color='black', linewidth=2))

    # Set box face colors
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax_box.set_ylabel('Shortage Magnitude (MGD)', fontsize=FONTSIZE_MEDIUM)
    ax_box.set_xlabel('FFMP Zone', fontsize=FONTSIZE_MEDIUM)
    ax_box.set_title('Box Plot: Shortage Distribution by Zone', fontsize=FONTSIZE_MEDIUM)
    ax_box.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax_box.set_axisbelow(True)
    ax_box.tick_params(axis='x', labelsize=FONTSIZE_SMALL)

    # Add dataset label
    dataset_label = DATASET_LABELS.get(dataset_id, dataset_id)
    fig.suptitle(f'NYC Diversion Shortage Magnitude Analysis: {dataset_label}',
                 fontsize=FONTSIZE_LARGE, y=0.995)

    plt.tight_layout()

    if fname is None:
        fname = f"{FIG_OUTPUT_DIR}/shortage_magnitude_distribution_{dataset_id}.png"

    plt.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"Saved figure: {fname}")
    plt.close()


def analyze_shortage_magnitude_distribution(all_shortages):
    """
    Analyze the distribution of shortage magnitudes by zone.

    Returns statistics on shortage magnitudes for each zone.
    """
    print("\n" + "="*80)
    print("SHORTAGE MAGNITUDE DISTRIBUTION BY ZONE")
    print("="*80)

    stats_by_zone = []

    for zone_num in sorted(ZONE_DEFINITIONS.keys(), reverse=True):
        zone_shortages = all_shortages[all_shortages['nyc_zone'] == zone_num]['nyc_shortage']

        if len(zone_shortages) == 0:
            continue

        zone_name = ZONE_DEFINITIONS[zone_num]['name']

        stats = {
            'zone': zone_num,
            'zone_name': zone_name,
            'n_days': len(zone_shortages),
            'mean': zone_shortages.mean(),
            'median': zone_shortages.median(),
            'p25': zone_shortages.quantile(0.25),
            'p75': zone_shortages.quantile(0.75),
            'p90': zone_shortages.quantile(0.90),
            'p95': zone_shortages.quantile(0.95),
            'p99': zone_shortages.quantile(0.99),
            'max': zone_shortages.max(),
            'pct_below_1mgd': 100.0 * (zone_shortages < 1.0).sum() / len(zone_shortages),
            'pct_below_5mgd': 100.0 * (zone_shortages < 5.0).sum() / len(zone_shortages),
            'pct_below_10mgd': 100.0 * (zone_shortages < 10.0).sum() / len(zone_shortages),
        }

        stats_by_zone.append(stats)

    # Print table
    print("\nShortage Magnitude Statistics (MGD):")
    print("-" * 120)
    print(f"{'Zone':<6} {'Name':<10} {'N':<8} {'Mean':<8} {'Median':<8} "
          f"{'P75':<8} {'P90':<8} {'P95':<8} {'P99':<8} {'Max':<8}")
    print("-" * 120)

    for s in stats_by_zone:
        print(f"{s['zone']:<6} {s['zone_name']:<10} {s['n_days']:<8,} "
              f"{s['mean']:<8.2f} {s['median']:<8.2f} "
              f"{s['p75']:<8.2f} {s['p90']:<8.2f} {s['p95']:<8.2f} "
              f"{s['p99']:<8.2f} {s['max']:<8.1f}")

    print("-" * 120)

    # Print percentage below thresholds
    print("\nPercentage of Shortage Days Below Threshold:")
    print("-" * 80)
    print(f"{'Zone':<6} {'Name':<10} {'< 1 MGD':<12} {'< 5 MGD':<12} {'< 10 MGD':<12}")
    print("-" * 80)

    for s in stats_by_zone:
        print(f"{s['zone']:<6} {s['zone_name']:<10} "
              f"{s['pct_below_1mgd']:<12.1f} "
              f"{s['pct_below_5mgd']:<12.1f} "
              f"{s['pct_below_10mgd']:<12.1f}")

    print("-" * 80)

    return pd.DataFrame(stats_by_zone)


def print_summary_statistics(stats_df, dataset_id):
    """Print summary statistics to console."""
    print(f"\n{'='*80}")
    print(f"NYC DIVERSION SHORTAGE ANALYSIS: {DATASET_LABELS.get(dataset_id, dataset_id)}")
    print(f"{'='*80}")

    print("\nShortage Distribution by FFMP Zone:")
    print("-" * 80)
    print(f"{'Zone':<12} {'Name':<12} {'Days':<12} {'% Days':<12} {'Volume (MG)':<15} {'% Volume':<12}")
    print("-" * 80)

    for _, row in stats_df.iterrows():
        zone_num = int(row['zone'])
        print(f"{zone_num:<12} {row['zone_name']:<12} "
              f"{row['n_shortage_days']:<12,} "
              f"{row['pct_of_all_shortage_days']:<12.1f} "
              f"{row['total_shortage_mg']:<15,.0f} "
              f"{row['pct_of_all_shortage_volume']:<12.1f}")

    print("-" * 80)

    # Summary statistics
    total_days = stats_df['n_shortage_days'].sum()
    total_volume = stats_df['total_shortage_mg'].sum()

    # Drought zones (Warning, Watch, Emergency)
    drought_days = stats_df[stats_df['zone'].isin(DROUGHT_ZONES)]['n_shortage_days'].sum()
    drought_volume = stats_df[stats_df['zone'].isin(DROUGHT_ZONES)]['total_shortage_mg'].sum()

    # Normal zone
    normal_days = stats_df[stats_df['zone'] == NORMAL_ZONE]['n_shortage_days'].sum()
    normal_volume = stats_df[stats_df['zone'] == NORMAL_ZONE]['total_shortage_mg'].sum()

    print(f"\nKEY FINDINGS:")
    print(f"  Total shortage days: {total_days:,}")
    print(f"  Total shortage volume: {total_volume:,.0f} MG")
    print(f"\n  Drought zones (Warning/Watch/Emergency):")
    print(f"    Days: {drought_days:,} ({100*drought_days/total_days:.1f}%)")
    print(f"    Volume: {drought_volume:,.0f} MG ({100*drought_volume/total_volume:.1f}%)")
    print(f"\n  Normal zone:")
    print(f"    Days: {normal_days:,} ({100*normal_days/total_days:.1f}%)")
    print(f"    Volume: {normal_volume:,.0f} MG ({100*normal_volume/total_volume:.1f}%)")

    if normal_days > 0:
        print(f"\n  ⚠ WARNING: {100*normal_days/total_days:.1f}% of shortage days occur during NORMAL conditions!")
        print(f"             This suggests shortages are not solely due to FFMP drought restrictions.")
    else:
        print(f"\n  ✓ All shortages occur during drought conditions (Warning/Watch/Emergency zones)")

    print("="*80)


def main():
    """Run NYC diversion shortage analysis."""
    # Get dataset from command line or use default
    if len(sys.argv) > 1:
        dataset_id = sys.argv[1]
    else:
        dataset_id = 'stationary_ensemble'
        print(f"No dataset specified, using default: {dataset_id}")

    verify_dataset_id(dataset_id)

    print(f"\n{'='*80}")
    print(f"NYC DIVERSION SHORTAGE BY FFMP ZONE ANALYSIS")
    print(f"Dataset: {dataset_id}")
    print(f"{'='*80}\n")

    # Load data
    shortage_zone_data = load_shortage_and_zone_data(dataset_id)

    # Analyze shortages by zone
    stats_df, all_shortages = analyze_shortage_by_zone(shortage_zone_data)

    if len(stats_df) == 0:
        print("No shortages found in this dataset!")
        return

    # Analyze shortage magnitude distributions
    magnitude_stats = analyze_shortage_magnitude_distribution(all_shortages)

    # Print summary statistics
    print_summary_statistics(stats_df, dataset_id)

    # Create visualizations
    plot_shortage_by_zone_summary(stats_df, dataset_id)
    plot_shortage_magnitude_distributions(all_shortages, dataset_id)

    # Calculate per-realization statistics
    print("\nCalculating per-realization statistics...")
    realization_stats = calculate_realization_statistics(shortage_zone_data)

    # Save detailed statistics to CSV
    csv_fname = f"{FIG_OUTPUT_DIR}/shortage_by_zone_{dataset_id}_summary.csv"
    stats_df.to_csv(csv_fname, index=False)
    print(f"Saved summary CSV: {csv_fname}")

    csv_fname_real = f"{FIG_OUTPUT_DIR}/shortage_by_zone_{dataset_id}_per_realization.csv"
    realization_stats.to_csv(csv_fname_real, index=False)
    print(f"Saved per-realization CSV: {csv_fname_real}")

    csv_fname_mag = f"{FIG_OUTPUT_DIR}/shortage_magnitude_stats_{dataset_id}.csv"
    magnitude_stats.to_csv(csv_fname_mag, index=False)
    print(f"Saved magnitude statistics CSV: {csv_fname_mag}")

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
