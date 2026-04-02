"""
SI: Shortage Analysis by FFMP Zone

Diagnostic script to investigate when shortages occur relative to
FFMP drought zone levels for NYC diversions, Montague flow target,
and Trenton flow target.

Key question: Are shortages only during drought
(Watch/Warning/Emergency) periods, or do they also occur during "normal"
periods?

FFMP Zone Definitions:
  - Zone 6: Drought Emergency (most severe)
  - Zone 5: Drought Watch
  - Zone 4: Drought Warning
  - Zone 3: Normal operations
  - Zones 1-2: Flood conditions (high storage)

Shortage types analyzed:
  - NYC Diversion: Demand - Actual Diversion (when positive)
  - Montague: MRF target - actual flow at Montague (when positive)
  - Trenton: MRF target - actual flow at Trenton (when positive)

Usage:
    python SI5_plot_nyc_diversion_shortage_by_zone.py [dataset_id]

Examples:
    python SI5_plot_nyc_diversion_shortage_by_zone.py stationary_ensemble
    python SI5_plot_nyc_diversion_shortage_by_zone.py climate_adjusted_low
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    FIG_DIR, OUTPUT_DIR, NYC_RESERVOIRS, NYC_TOTAL_CAPACITY,
    verify_dataset_id
)
from methods.plotting.styles import (
    DATASET_COLORS, DATASET_LABELS,
    FFMP_ZONE_COLORS_INT,
    DPI_HIGH, FONTSIZE_SMALL, FONTSIZE_MEDIUM, FONTSIZE_LARGE
)
from methods.plotting.shortage_by_zone import (
    print_summary_statistics,
    plot_shortage_by_zone_summary,
    plot_shortage_magnitude_distributions,
)

# Output directory
FIG_OUTPUT_DIR = f"{FIG_DIR}/SI5_NYC_shortage_by_zone"
os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

# Zone definitions (colors from centralized styles)
ZONE_DEFINITIONS = {
    6: {'name': 'Emergency', 'color': FFMP_ZONE_COLORS_INT[6], 'label': 'Emergency (Zone 6)'},
    5: {'name': 'Warning', 'color': FFMP_ZONE_COLORS_INT[5], 'label': 'Warning (Zone 5)'},
    4: {'name': 'Watch', 'color': FFMP_ZONE_COLORS_INT[4], 'label': 'Watch (Zone 4)'},
    3: {'name': 'Normal', 'color': FFMP_ZONE_COLORS_INT[3], 'label': 'Normal (Zone 3)'},
    2: {'name': 'Flood', 'color': FFMP_ZONE_COLORS_INT[2], 'label': 'Flood (Zones 1-2)'},
    1: {'name': 'Flood', 'color': FFMP_ZONE_COLORS_INT[1], 'label': 'Flood (Zones 1-2)'},
}

# Ordered list of zones to plot (most severe first, excluding flood)
ZONES_TO_PLOT = [6, 5, 4, 3]

# Drought zones for analysis
DROUGHT_ZONES = [4, 5, 6]  # Warning, Watch, Emergency
NORMAL_ZONE = 3
FLOOD_ZONES = [1, 2]

# Shortage analyses to run: (column_name, display_label)
SHORTAGE_ANALYSES = [
    ('nyc_shortage', 'NYC Diversion'),
    ('delMontague_shortage', 'Montague Flow Target'),
    ('delTrenton_shortage', 'Trenton Flow Target'),
]


def load_shortage_and_zone_data(dataset_id):
    """
    Load shortage and FFMP zone level data for NYC diversions,
    Montague flow target, and Trenton flow target.

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
        - delMontague_shortage: Montague flow target shortage (MGD)
        - delTrenton_shortage: Trenton flow target shortage (MGD)
        - nyc_zone: FFMP zone level (1-6)
        - nyc_storage_pct: NYC aggregate storage (%)
    """
    fname = f'{OUTPUT_DIR}/{dataset_id}_with_postprocessing.hdf5'

    if not os.path.exists(fname):
        raise FileNotFoundError(
            f"Postprocessed data not found: {fname}\n"
            f"Run 04_postprocess_data.py first!"
        )

    print(f"Loading data from: {fname}")
    data = pywrdrb.Data()
    data.load_from_export(
        fname,
        results_sets=['ibt_diversions', 'ibt_demands', 'res_level',
                      'res_storage', 'shortage']
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

        # Pre-calculated Montague and Trenton shortages from postprocessing
        delMontague_shortage = data.shortage[dataset_id][r]['delMontague']
        delTrenton_shortage = data.shortage[dataset_id][r]['delTrenton']

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
            'delMontague_shortage': delMontague_shortage,
            'delTrenton_shortage': delTrenton_shortage,
            'nyc_zone': nyc_zone,
            'nyc_storage_pct': nyc_storage_pct
        })

        shortage_zone_data[r] = df

    return shortage_zone_data


def analyze_shortage_by_zone(shortage_zone_data, shortage_col='nyc_shortage',
                             shortage_label='NYC Diversion'):
    """
    Analyze shortages by FFMP zone.

    Parameters
    ----------
    shortage_zone_data : dict
        Dictionary mapping realization_id to shortage/zone DataFrame
    shortage_col : str
        Column name for shortage values
    shortage_label : str
        Display label for the shortage type

    Returns
    -------
    pd.DataFrame
        Summary statistics by zone
    pd.DataFrame
        All shortage day records
    """
    # Aggregate across all realizations
    all_shortage_days = []

    for r, df in shortage_zone_data.items():
        # Filter to days with shortage
        shortage_days = df[df[shortage_col] > 0].copy()
        shortage_days['realization_id'] = r
        all_shortage_days.append(shortage_days)

    if len(all_shortage_days) == 0:
        print(f"  Warning: No {shortage_label} shortage days found!")
        return pd.DataFrame(), pd.DataFrame()

    all_shortages = pd.concat(all_shortage_days, ignore_index=True)

    print(f"\n{shortage_label} - Total shortage days across all realizations: {len(all_shortages):,}")

    # Calculate statistics by zone
    zone_stats = []

    total_shortage_days = len(all_shortages)
    total_shortage_volume = all_shortages[shortage_col].sum()

    for zone_num in sorted(ZONE_DEFINITIONS.keys(), reverse=True):
        zone_shortages = all_shortages[all_shortages['nyc_zone'] == zone_num]

        n_days = len(zone_shortages)
        total_vol = zone_shortages[shortage_col].sum()
        mean_shortage = zone_shortages[shortage_col].mean() if n_days > 0 else 0
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


def calculate_realization_statistics(shortage_zone_data, shortage_col='nyc_shortage'):
    """
    Calculate per-realization statistics on shortage by zone.

    Parameters
    ----------
    shortage_zone_data : dict
        Dictionary mapping realization_id to shortage/zone DataFrame
    shortage_col : str
        Column name for shortage values

    Returns
    -------
    pd.DataFrame
        Per-realization statistics with columns for each zone's shortage frequency
    """
    realization_stats = []

    for r, df in shortage_zone_data.items():
        stats = {'realization_id': r}

        # Total days with shortage
        total_shortage_days = (df[shortage_col] > 0).sum()
        stats['total_shortage_days'] = total_shortage_days

        if total_shortage_days > 0:
            # Shortage days by zone
            for zone_num in sorted(ZONE_DEFINITIONS.keys(), reverse=True):
                zone_shortage_days = ((df[shortage_col] > 0) &
                                     (df['nyc_zone'] == zone_num)).sum()
                pct = 100.0 * zone_shortage_days / total_shortage_days
                stats[f'pct_shortage_days_zone{zone_num}'] = pct
        else:
            for zone_num in sorted(ZONE_DEFINITIONS.keys(), reverse=True):
                stats[f'pct_shortage_days_zone{zone_num}'] = 0

        realization_stats.append(stats)

    return pd.DataFrame(realization_stats)




def analyze_shortage_magnitude_distribution(all_shortages, shortage_col='nyc_shortage',
                                             shortage_label='NYC Diversion'):
    """
    Analyze the distribution of shortage magnitudes by zone.

    Returns statistics on shortage magnitudes for each zone.
    """
    print("\n" + "="*80)
    print(f"{shortage_label.upper()} SHORTAGE MAGNITUDE DISTRIBUTION BY ZONE")
    print("="*80)

    stats_by_zone = []

    for zone_num in sorted(ZONE_DEFINITIONS.keys(), reverse=True):
        zone_shortages = all_shortages[all_shortages['nyc_zone'] == zone_num][shortage_col]

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




def main():
    """Run shortage-by-zone analysis for NYC diversions, Montague, and Trenton."""
    # Get dataset from command line or use default
    if len(sys.argv) > 1:
        dataset_id = sys.argv[1]
    else:
        dataset_id = 'stationary_ensemble'
        print(f"No dataset specified, using default: {dataset_id}")

    verify_dataset_id(dataset_id)

    print(f"\n{'='*80}")
    print(f"SHORTAGE BY FFMP ZONE ANALYSIS")
    print(f"Dataset: {dataset_id}")
    print(f"{'='*80}\n")

    # Load data (includes NYC, Montague, and Trenton shortages)
    shortage_zone_data = load_shortage_and_zone_data(dataset_id)

    # Run analysis for each shortage type
    for shortage_col, shortage_label in SHORTAGE_ANALYSES:
        print(f"\n{'#'*80}")
        print(f"# Analyzing: {shortage_label}")
        print(f"{'#'*80}")

        label_slug = shortage_label.replace(' ', '_').lower()

        # Analyze shortages by zone
        stats_df, all_shortages = analyze_shortage_by_zone(
            shortage_zone_data, shortage_col=shortage_col,
            shortage_label=shortage_label
        )

        if len(stats_df) == 0:
            print(f"No {shortage_label} shortages found in this dataset!")
            continue

        # Analyze shortage magnitude distributions
        magnitude_stats = analyze_shortage_magnitude_distribution(
            all_shortages, shortage_col=shortage_col,
            shortage_label=shortage_label
        )

        # Print summary statistics
        print_summary_statistics(stats_df, dataset_id,
                                title_label=shortage_label,
                                drought_zones=DROUGHT_ZONES,
                                normal_zone=NORMAL_ZONE)

        # Create visualizations
        plot_shortage_by_zone_summary(
            stats_df, dataset_id,
            title_label=shortage_label,
            zone_definitions=ZONE_DEFINITIONS,
            zones_to_plot=ZONES_TO_PLOT,
            fig_output_dir=FIG_OUTPUT_DIR,
        )
        plot_shortage_magnitude_distributions(
            all_shortages, dataset_id,
            shortage_col=shortage_col,
            title_label=shortage_label,
            zone_definitions=ZONE_DEFINITIONS,
            zones_to_plot=ZONES_TO_PLOT,
            zone_col='nyc_zone',
            fig_output_dir=FIG_OUTPUT_DIR,
        )

        # Calculate per-realization statistics
        print(f"\nCalculating per-realization statistics for {shortage_label}...")
        realization_stats = calculate_realization_statistics(
            shortage_zone_data, shortage_col=shortage_col
        )


    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
