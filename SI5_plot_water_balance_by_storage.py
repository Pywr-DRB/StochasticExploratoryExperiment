"""
SI5: NYC Water Balance Decomposition by Storage Bins

This script creates a stacked horizontal bar chart showing how NYC water balance
components vary across different storage levels.

Features:
- Storage bins (0-10%, 10-20%, 20-30%, etc.) on Y-axis
- Water balance components shown as stacked bars (percentage of total)
- Configurable quantile (mean, median, 90th percentile, etc.)
- Bins with no data are greyed out

Water Balance Components:
- Inflow: Natural inflows to NYC reservoirs
- NYC Diversions: Water delivered to NYC for consumption
- Downstream Contributions: NYC releases to support downstream targets
- Evaporation & Spills: Losses from reservoirs

Usage:
    python SI5_plot_water_balance_by_storage.py <dataset_id>

Example:
    python SI5_plot_water_balance_by_storage.py stationary_ensemble
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import pywrdrb
from methods.config import *
from methods.plotting.styles import DPI_HIGH

# Output directory
FIG_DIR_WATER_BALANCE = f"{FIG_DIR}/water_balance_by_storage"
os.makedirs(FIG_DIR_WATER_BALANCE, exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Quantile to calculate for each storage bin (0.5 = median, 0.9 = 90th percentile)
QUANTILE = 0.5  # Use median by default

# Storage bins (in percentage)
STORAGE_BINS = [
    (0, 10),
    (10, 20),
    (20, 30),
    (30, 40),
    (40, 50),
    (50, 60),
    (60, 70),
    (70, 80),
    (80, 90),
    (90, 100),
]

# NYC reservoir parameters
NYC_RESERVOIRS = ['cannonsville', 'pepacton', 'neversink']
NYC_STORAGE_CAPACITIES = {
    'cannonsville': 95706,  # MG
    'pepacton': 140190,     # MG
    'neversink': 34941      # MG
}
NYC_TOTAL_CAPACITY = sum(NYC_STORAGE_CAPACITIES.values())  # 270,837 MG

# Water balance component colors
COMPONENT_COLORS = {
    'inflow': '#2E86AB',              # Blue
    'diversions': '#C73E1D',          # Red
    'contributions': '#F18F01',       # Orange
    'evap_spill': '#808080',          # Grey
}

# Component display names
COMPONENT_NAMES = {
    'inflow': 'Inflow',
    'diversions': 'NYC Diversions',
    'contributions': 'Downstream Contributions',
    'evap_spill': 'Evaporation & Spills',
}


def bin_data_by_storage(storage_pct, components_dict):
    """
    Bin daily water balance data by storage level.

    Parameters
    ----------
    storage_pct : pd.Series
        Daily storage percentage with datetime index
    components_dict : dict
        Dictionary with component names as keys and pd.Series as values
        Each series should have the same datetime index as storage_pct

    Returns
    -------
    binned_data : dict
        Dictionary with storage bins as keys, each containing a DataFrame
        with columns for each component
    """
    binned_data = {}

    for bin_min, bin_max in STORAGE_BINS:
        # Find days in this storage bin
        mask = (storage_pct >= bin_min) & (storage_pct < bin_max)

        if mask.sum() == 0:
            # No data in this bin
            binned_data[(bin_min, bin_max)] = None
            continue

        # Extract component values for days in this bin
        bin_df = pd.DataFrame({
            comp_name: comp_series[mask].values
            for comp_name, comp_series in components_dict.items()
        })

        binned_data[(bin_min, bin_max)] = bin_df

    return binned_data


def calculate_annual_totals(binned_data, dates):
    """
    Calculate annual totals for each storage bin and component.

    Parameters
    ----------
    binned_data : dict
        Output from bin_data_by_storage()
    dates : pd.DatetimeIndex
        DateTime index for the data

    Returns
    -------
    annual_totals : dict
        Dictionary with storage bins as keys, each containing a DataFrame
        with annual totals for each component
    """
    annual_totals = {}

    for storage_bin, bin_df in binned_data.items():
        if bin_df is None:
            annual_totals[storage_bin] = None
            continue

        # Add dates to dataframe
        bin_df_with_dates = bin_df.copy()
        bin_df_with_dates['date'] = dates[bin_df.index]
        bin_df_with_dates['year'] = bin_df_with_dates['date'].dt.year

        # Calculate annual totals for each component (exclude datetime columns)
        annual_df = bin_df_with_dates.groupby('year')[list(bin_df.columns)].sum()

        annual_totals[storage_bin] = annual_df

    return annual_totals


def calculate_quantile_fractions(annual_totals, quantile=0.5):
    """
    Calculate the quantile fractions for each storage bin.

    Parameters
    ----------
    annual_totals : dict
        Output from calculate_annual_totals()
    quantile : float
        Quantile to calculate (0.5 = median, 0.9 = 90th percentile)

    Returns
    -------
    fractions : dict
        Dictionary with storage bins as keys, each containing a dict of
        component: fraction pairs
    """
    fractions = {}

    for storage_bin, annual_df in annual_totals.items():
        if annual_df is None or len(annual_df) == 0:
            fractions[storage_bin] = None
            continue

        # Calculate quantile for each component
        quantile_values = annual_df.quantile(quantile)

        # Calculate total (sum of all components)
        total = quantile_values.sum()

        if total == 0:
            fractions[storage_bin] = None
            continue

        # Calculate fractions (as percentage)
        bin_fractions = {
            comp: 100.0 * quantile_values[comp] / total
            for comp in quantile_values.index
        }

        fractions[storage_bin] = bin_fractions

    return fractions


def extract_water_balance_components(data, dataset_id, realization_id):
    """
    Extract water balance components for a single realization.

    Parameters
    ----------
    data : pywrdrb.Data
        Data object from load_shortage_data()
    dataset_id : str
        Dataset identifier
    realization_id : int
        Realization identifier

    Returns
    -------
    storage_pct : pd.Series
        Daily storage percentage
    components : dict
        Dictionary with component names and their daily values
    """
    # Get NYC storage (percentage)
    storage_df = data.res_storage[dataset_id][realization_id]
    nyc_storage_mg = storage_df[NYC_RESERVOIRS].sum(axis=1)
    storage_pct = 100.0 * nyc_storage_mg / NYC_TOTAL_CAPACITY

    # Get inflows to NYC reservoirs
    # Sum of all inflows to Cannonsville, Pepacton, and Neversink
    inflow_df = data.inflow[dataset_id][realization_id]
    nyc_inflow = inflow_df[['cannonsville', 'pepacton', 'neversink']].sum(axis=1)

    # Get NYC diversions (deliveries)
    diversion_df = data.ibt_diversions[dataset_id][realization_id]
    nyc_diversions = diversion_df['delivery_nyc']

    # Get downstream contributions (releases to support flow targets)
    # This is the total NYC contribution to Montague-Trenton targets
    contribution_df = data.contribution[dataset_id][realization_id]
    downstream_contributions = contribution_df['mrf_montagueTrenton_nyc']

    # Calculate evaporation and spills (residual)
    # Water balance: Inflow = Diversions + Contributions + Evap/Spill + Storage Change
    # Evap/Spill = Inflow - Diversions - Contributions - Storage Change

    # Calculate storage change (already have storage in MG from earlier)
    storage_change = nyc_storage_mg.diff().fillna(0)

    evap_spill = nyc_inflow - nyc_diversions - downstream_contributions - storage_change
    evap_spill = evap_spill.clip(lower=0)  # Can't be negative

    components = {
        'inflow': nyc_inflow,
        'diversions': nyc_diversions,
        'contributions': downstream_contributions,
        'evap_spill': evap_spill,
    }

    return storage_pct, components


def aggregate_across_realizations(data, dataset_id, quantile=0.5):
    """
    Aggregate water balance data across all realizations.

    Parameters
    ----------
    data : pywrdrb.Data
        Data object
    dataset_id : str
        Dataset identifier
    quantile : float
        Quantile to calculate for aggregation

    Returns
    -------
    agg_fractions : dict
        Storage bin -> component fractions (averaged across realizations)
    """
    realization_ids = list(data.res_storage[dataset_id].keys())

    # Collect fractions from all realizations
    all_fractions = {bin: [] for bin in STORAGE_BINS}

    for real_id in realization_ids:
        # Extract components for this realization
        storage_pct, components = extract_water_balance_components(
            data, dataset_id, real_id
        )

        # Bin by storage
        binned = bin_data_by_storage(storage_pct, components)

        # Calculate annual totals
        annual = calculate_annual_totals(binned, storage_pct.index)

        # Calculate quantile fractions
        fractions = calculate_quantile_fractions(annual, quantile)

        # Store fractions
        for storage_bin, bin_fractions in fractions.items():
            if bin_fractions is not None:
                all_fractions[storage_bin].append(bin_fractions)

    # Average fractions across realizations
    agg_fractions = {}
    for storage_bin, fraction_list in all_fractions.items():
        if len(fraction_list) == 0:
            agg_fractions[storage_bin] = None
        else:
            # Average each component
            agg_fractions[storage_bin] = {
                comp: np.mean([f[comp] for f in fraction_list])
                for comp in fraction_list[0].keys()
            }

    return agg_fractions


def plot_water_balance_by_storage(agg_fractions, dataset_id, dataset_label, quantile):
    """
    Create stacked horizontal bar chart of water balance by storage bin.

    Parameters
    ----------
    agg_fractions : dict
        Output from aggregate_across_realizations()
    dataset_id : str
        Dataset identifier
    dataset_label : str
        Dataset display label
    quantile : float
        Quantile used for calculation
    """
    print("Creating water balance by storage plot...")

    fig, ax = plt.subplots(figsize=(12, 8))

    # Y-axis positions (one for each storage bin)
    y_positions = np.arange(len(STORAGE_BINS))
    bar_height = 0.8

    # Component order for stacking
    components_ordered = ['inflow', 'diversions', 'contributions', 'evap_spill']

    # Plot each storage bin
    for idx, storage_bin in enumerate(STORAGE_BINS):
        bin_min, bin_max = storage_bin
        bin_fractions = agg_fractions[storage_bin]

        if bin_fractions is None:
            # Grey out bins with no data
            ax.barh(y_positions[idx], 100, bar_height,
                   color='#E0E0E0', edgecolor='black', linewidth=0.5,
                   label='No Data' if idx == 0 else None)
            continue

        # Stack components
        left_position = 0
        for comp in components_ordered:
            if comp not in bin_fractions:
                continue

            fraction = bin_fractions[comp]
            ax.barh(y_positions[idx], fraction, bar_height,
                   left=left_position,
                   color=COMPONENT_COLORS[comp],
                   edgecolor='black', linewidth=0.5,
                   label=COMPONENT_NAMES[comp] if idx == 0 else None)

            left_position += fraction

    # Formatting
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"{bin_min}-{bin_max}%" for bin_min, bin_max in STORAGE_BINS],
                       fontsize=11)
    ax.set_ylabel('NYC Storage Level (%)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Percentage of Annual Water Balance (%)', fontsize=13, fontweight='bold')

    quantile_label = {0.5: 'Median', 0.9: '90th Percentile', 0.1: '10th Percentile'}.get(
        quantile, f'{int(quantile*100)}th Percentile'
    )
    ax.set_title(
        f'NYC Water Balance Decomposition by Storage Level\n{dataset_label} ({quantile_label})',
        fontsize=14, fontweight='bold', pad=20
    )

    ax.set_xlim(0, 100)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Legend
    ax.legend(loc='upper right', fontsize=10, frameon=True, fancybox=True)

    plt.tight_layout()

    # Save
    fname = f"{FIG_DIR_WATER_BALANCE}/{dataset_id}_water_balance_by_storage_q{int(quantile*100)}.png"
    plt.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close()


def main(dataset_id):
    """
    Main function to generate water balance by storage plot.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    """
    print("=" * 80)
    print(f"NYC WATER BALANCE BY STORAGE: {dataset_id}")
    print("=" * 80)

    # Verify dataset
    verify_dataset_id(dataset_id)
    dataset_config = DATASET_CONFIGS[dataset_id]
    dataset_label = f"{dataset_config['description']} ({dataset_config['type']})"

    # Load data
    print("\nLoading data...")
    fname = f'./pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5'

    if not os.path.exists(fname):
        raise FileNotFoundError(
            f"Postprocessed data not found: {fname}\n"
            "Run 04_postprocess_data.py first!"
        )

    print(f"  Loading from: {fname}")
    data = pywrdrb.Data()
    data.load_from_export(
        fname,
        results_sets=['res_storage', 'inflow', 'ibt_diversions', 'contribution']
    )
    print("  Data loaded successfully")

    # Aggregate across realizations
    print(f"\nAggregating data across realizations (quantile={QUANTILE})...")
    agg_fractions = aggregate_across_realizations(data, dataset_id, quantile=QUANTILE)

    # Print summary
    print("\nWater balance fractions by storage bin:")
    for storage_bin, fractions in agg_fractions.items():
        bin_min, bin_max = storage_bin
        if fractions is None:
            print(f"  {bin_min:3d}-{bin_max:3d}%: No data")
        else:
            print(f"  {bin_min:3d}-{bin_max:3d}%: " + ", ".join([
                f"{COMPONENT_NAMES[comp]}: {frac:.1f}%"
                for comp, frac in fractions.items()
            ]))

    # Create plot
    print("\nCreating plot...")
    plot_water_balance_by_storage(agg_fractions, dataset_id, dataset_label, QUANTILE)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        print(f"\nAvailable datasets: {list(DATASET_CONFIGS.keys())}")
        sys.exit(1)

    dataset_id = sys.argv[1]
    verify_dataset_id(dataset_id)

    main(dataset_id)
