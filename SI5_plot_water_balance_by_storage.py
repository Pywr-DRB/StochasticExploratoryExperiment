"""
SI5: NYC Water Balance Decomposition by Storage Bins

This script creates a stacked horizontal bar chart showing how NYC water balance
components vary across different storage levels.

Features:
- Storage bins (0-10%, 10-20%, 20-30%, etc.) on Y-axis
- Water balance components shown as stacked bars
- Configurable quantile (mean, median, 90th percentile, etc.)
- Configurable normalization: percentage of inflow or percentage of total
- Two aggregation methods:
  1. Since June 1: Aggregate from June 1 to the date of minimum storage
  2. N months prior: Aggregate for N months prior to minimum storage
- Bins with no data are greyed out

Water Balance Components:
- Inflow: Natural inflows to NYC reservoirs
- NYC Diversions: Water delivered to NYC for consumption
- Downstream Contributions: NYC releases to support downstream targets
- Evaporation & Spills: Losses from reservoirs

Configuration (edit constants in script):
- AGGREGATION_METHOD: 'since_june1' or 'n_months_prior' (default: 'since_june1')
- QUANTILE: Which percentile to plot (default: 0.50 for median)
- N_MONTHS_PRIOR: Window size in months (only used if method is 'n_months_prior')
- NORMALIZE_BY: 'inflow' (% of inflow) or 'total' (% of sum of all components)

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

# Aggregation method for water balance calculations
# 'since_june1': Aggregate from June 1 to minimum storage date
# 'n_months_prior': Aggregate for N months prior to minimum storage date
AGGREGATION_METHOD = 'since_june1'  # Options: 'since_june1' or 'n_months_prior'

# Quantile to calculate for each storage bin (0.5 = median, 0.9 = 90th percentile)
QUANTILE = 0.95  # Use median by default

# Number of months prior to minimum storage to analyze (only used if method is 'n_months_prior')
N_MONTHS_PRIOR = 6  # Analyze water balance for N months leading up to minimum storage

# Normalization method for water balance components
# 'inflow': Express all components as percentage of total inflow
# 'total': Express all components as percentage of sum of all components
NORMALIZE_BY = 'total'  # Options: 'inflow' or 'total'

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
    Bin water balance data by minimum storage level for each year.

    Each year is classified into a storage bin based on that year's minimum
    storage percentage. For each year, the water balance components are summed
    using the aggregation method specified by AGGREGATION_METHOD:
    - 'since_june1': Aggregate from June 1 to minimum storage date
    - 'n_months_prior': Aggregate for N months prior to minimum storage date

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
        with windowed totals for each component (index = year)
    """
    # Add year column to storage data
    storage_with_year = storage_pct.to_frame('storage_pct')
    storage_with_year['year'] = storage_pct.index.year

    # Find date and value of minimum storage for each year
    min_storage_dates = storage_with_year.groupby('year')['storage_pct'].idxmin()
    min_storage_values = storage_with_year.groupby('year')['storage_pct'].min()

    # Calculate windowed sums for each year
    windowed_totals = {comp_name: {} for comp_name in components_dict.keys()}

    for year, min_date in min_storage_dates.items():
        # Determine start date based on aggregation method
        if AGGREGATION_METHOD == 'since_june1':
            # Start from June 1 of the same year as min_date
            # Note: For minimum storage occurring Jan-May, this will use June 1 of previous year
            year_of_min_storage = min_date.year
            if min_date.month >= 6:
                # Use June 1 of same year
                start_date = pd.Timestamp(year=year_of_min_storage, month=6, day=1)
            else:
                # Use June 1 of previous year (for minimum storage in Jan-May)
                start_date = pd.Timestamp(year=year_of_min_storage - 1, month=6, day=1)

        elif AGGREGATION_METHOD == 'n_months_prior':
            # Calculate start date (N months before minimum)
            start_date = min_date - pd.DateOffset(months=N_MONTHS_PRIOR)

        else:
            raise ValueError(f"Invalid AGGREGATION_METHOD: {AGGREGATION_METHOD}. Must be 'since_june1' or 'n_months_prior'")

        # Sum components in window for this year
        for comp_name, comp_series in components_dict.items():
            # Filter to window
            mask = (comp_series.index >= start_date) & (comp_series.index <= min_date)
            windowed_totals[comp_name][year] = comp_series[mask].sum()

    # Convert to DataFrame
    windowed_df = pd.DataFrame(windowed_totals)
    windowed_df['min_storage'] = min_storage_values

    # Bin years by minimum storage
    binned_data = {}
    for bin_min, bin_max in STORAGE_BINS:
        # Find years where minimum storage falls in this bin
        mask = (windowed_df['min_storage'] >= bin_min) & (windowed_df['min_storage'] < bin_max)

        if mask.sum() == 0:
            # No years in this bin
            binned_data[(bin_min, bin_max)] = None
            continue

        # Extract windowed totals for years in this bin (exclude min_storage column)
        bin_df = windowed_df[mask].drop(columns=['min_storage'])

        binned_data[(bin_min, bin_max)] = bin_df

    return binned_data


def calculate_annual_totals(binned_data, dates):
    """
    Pass-through function for compatibility.

    The binned_data already contains windowed totals (after refactoring),
    so this function just returns the input unchanged.

    Parameters
    ----------
    binned_data : dict
        Output from bin_data_by_storage() - already contains windowed totals
    dates : pd.DatetimeIndex
        DateTime index for the data (unused, kept for compatibility)

    Returns
    -------
    windowed_totals : dict
        Same as binned_data (already contains windowed totals)
    """
    # Data is already in windowed form from bin_data_by_storage
    return binned_data


def calculate_quantile_fractions(annual_totals, quantile=0.5):
    """
    Calculate the quantile fractions for each storage bin.

    For each storage bin, calculates the specified quantile across all years
    in that bin, then converts to fractions showing the relative contribution
    of each water balance component.

    The normalization method is determined by NORMALIZE_BY:
    - 'inflow': Express all components as percentage of total inflow
    - 'total': Express all components as percentage of sum of all components

    Parameters
    ----------
    annual_totals : dict
        Output from calculate_annual_totals() - contains windowed totals
    quantile : float
        Quantile to calculate (0.5 = median, 0.9 = 90th percentile)

    Returns
    -------
    fractions : dict
        Dictionary with storage bins as keys, each containing a dict of
        component: fraction pairs (as percentages)
    """
    fractions = {}

    for storage_bin, annual_df in annual_totals.items():
        if annual_df is None or len(annual_df) == 0:
            fractions[storage_bin] = None
            continue

        # Calculate quantile for each component across all years in this bin
        quantile_values = annual_df.quantile(quantile)

        # Determine normalization denominator based on NORMALIZE_BY setting
        if NORMALIZE_BY == 'inflow':
            # Normalize by total inflow only
            if 'inflow' not in quantile_values.index:
                fractions[storage_bin] = None
                continue

            total = quantile_values['inflow']

            if total == 0:
                fractions[storage_bin] = None
                continue

        elif NORMALIZE_BY == 'total':
            # Normalize by sum of all components
            total = quantile_values.sum()

            if total == 0:
                fractions[storage_bin] = None
                continue
        else:
            raise ValueError(f"Invalid NORMALIZE_BY value: {NORMALIZE_BY}. Must be 'inflow' or 'total'")

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
    # When normalizing by inflow, exclude inflow from the plot (it's always 100%)
    if NORMALIZE_BY == 'inflow':
        components_ordered = ['diversions', 'contributions', 'evap_spill']
    else:
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

    # X-axis label depends on normalization method
    if NORMALIZE_BY == 'inflow':
        xlabel = 'Percentage of Total Inflow (%)'
    else:
        xlabel = 'Percentage of Annual Water Balance (%)'
    ax.set_xlabel(xlabel, fontsize=13, fontweight='bold')

    quantile_label = {0.5: 'Median', 0.9: '90th Percentile', 0.1: '10th Percentile'}.get(
        quantile, f'{int(quantile*100)}th Percentile'
    )

    # Title includes normalization method and aggregation method
    normalize_label = 'as % of Inflow' if NORMALIZE_BY == 'inflow' else 'as % of Total'
    if AGGREGATION_METHOD == 'since_june1':
        agg_label = 'June 1 to min storage'
    else:
        agg_label = f'{N_MONTHS_PRIOR}-month prior'

    ax.set_title(
        f'NYC Water Balance Decomposition by Storage Level\n{dataset_label} ({quantile_label}, {normalize_label}, {agg_label})',
        fontsize=14, fontweight='bold', pad=20
    )

    # X-axis limit depends on normalization method
    if NORMALIZE_BY == 'inflow':
        # When normalizing by inflow, components can sum to >100% (storage decrease)
        # or <100% (storage increase), so use wider range
        ax.set_xlim(0, 150)
    else:
        # When normalizing by total, always sums to 100%
        ax.set_xlim(0, 100)

    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Legend
    ax.legend(loc='upper right', fontsize=10, frameon=True, fancybox=True)

    plt.tight_layout()

    # Save - include normalization method in filename
    normalize_suffix = 'pct_inflow' if NORMALIZE_BY == 'inflow' else 'pct_total'
    fname = f"{FIG_DIR_WATER_BALANCE}/{dataset_id}_water_balance_by_storage_q{int(quantile*100)}_{normalize_suffix}.png"
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

    # Validate aggregation method
    if AGGREGATION_METHOD not in ['since_june1', 'n_months_prior']:
        raise ValueError(
            f"Invalid AGGREGATION_METHOD: '{AGGREGATION_METHOD}'\n"
            "Must be 'since_june1' or 'n_months_prior'"
        )

    # Print configuration
    print("\nConfiguration:")
    print(f"  Aggregation method: {AGGREGATION_METHOD}")
    if AGGREGATION_METHOD == 'since_june1':
        print(f"    -> Aggregating from June 1 to minimum storage date")
    else:
        print(f"    -> Aggregating {N_MONTHS_PRIOR} months prior to minimum storage date")
    print(f"  Quantile: {QUANTILE} ({int(QUANTILE*100)}th percentile)")
    print(f"  Normalization: {NORMALIZE_BY}")

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
