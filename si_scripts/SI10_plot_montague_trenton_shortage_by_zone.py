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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import pywrdrb
from methods.config import (
    FIG_DIR, OUTPUT_DIR, NYC_RESERVOIRS, NYC_TOTAL_CAPACITY,
    verify_dataset_id,
)
from methods.plotting.styles import (
    DATASET_LABELS,
    FFMP_ZONE_COLORS_INT,
    DPI_HIGH, FONTSIZE_SMALL, FONTSIZE_MEDIUM, FONTSIZE_LARGE,
)
from methods.plotting.shortage_by_zone import (
    print_summary_statistics,
    plot_shortage_by_zone_summary,
    plot_shortage_magnitude_distributions,
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
        'color': FFMP_ZONE_COLORS_INT[7],
        'label': f'Drought Emergency Storage <{EMERGENCY_STORAGE_THRESHOLD_PCT:.0f}%'},
    6: {'name': f'Emergency\n(\u2265{EMERGENCY_STORAGE_THRESHOLD_PCT:.0f}%)',
        'color': FFMP_ZONE_COLORS_INT[6],
        'label': f'Emergency (\u2265{EMERGENCY_STORAGE_THRESHOLD_PCT:.0f}%)'},
    5: {'name': 'Warning', 'color': FFMP_ZONE_COLORS_INT[5], 'label': 'Warning (Zone 5)'},
    4: {'name': 'Watch',   'color': FFMP_ZONE_COLORS_INT[4], 'label': 'Watch (Zone 4)'},
    3: {'name': 'Normal',  'color': FFMP_ZONE_COLORS_INT[3], 'label': 'Normal (Zone 3)'},
    2: {'name': 'Flood',   'color': FFMP_ZONE_COLORS_INT[2], 'label': 'Flood (Zones 1-2)'},
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
    fname = f'{OUTPUT_DIR}/{dataset_id}_with_postprocessing.hdf5'

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



# ============================================================================
# PLOTTING (delegated to methods.plotting.shortage_by_zone)
# ============================================================================


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

        print_summary_statistics(stats_df, dataset_id, loc_label)
        plot_shortage_by_zone_summary(
            stats_df, dataset_id,
            title_label=loc_label,
            zone_definitions=ZONE_DEFINITIONS,
            zones_to_plot=ZONES_TO_PLOT,
            fig_output_dir=FIG_OUTPUT_DIR,
        )
        plot_shortage_magnitude_distributions(
            all_shortages, dataset_id,
            shortage_col=location,
            title_label=loc_label,
            zone_definitions=ZONE_DEFINITIONS,
            zones_to_plot=ZONES_TO_PLOT,
            zone_col='zone',
            fig_output_dir=FIG_OUTPUT_DIR,
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
