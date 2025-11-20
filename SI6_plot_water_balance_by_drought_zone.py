"""
SI6: NYC Water Balance Distribution by Drought Zone

This script analyzes the relationship between drought zones and water balance components.
For each year in the simulation, it identifies the minimum drought zone reached and
calculates water balance totals using one of two aggregation methods:
1. Since June 1: Aggregate from June 1 to the date of minimum zone
2. N months prior: Aggregate for N months prior to reaching minimum zone

Features:
- Classifies years by minimum drought zone reached
- Calculates aggregates of inflow and contributions prior to minimum zone
- Creates distribution plots (violin/box plots) for each drought zone category
- Shows how water balance varies across drought severity levels

Drought Zone Classification:
- Zone 6: Drought Emergency
- Zone 5: Drought Watch
- Zone 4: Drought Warning
- Zone 3: Normal
- Zones 1-2: Flood conditions
- "Other": Zones 1, 2, or 3 (Normal or above)

Configuration (edit constants in script):
- AGGREGATION_METHOD: 'since_june1' or 'n_months_prior' (default: 'since_june1')
- N_MONTHS_PRIOR: Window size in months (only used if method is 'n_months_prior')

Usage:
    python SI6_plot_water_balance_by_drought_zone.py <dataset_id>

Example:
    python SI6_plot_water_balance_by_drought_zone.py stationary_ensemble
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
FIG_DIR_DROUGHT_ZONE = f"{FIG_DIR}/water_balance_by_drought_zone"
os.makedirs(FIG_DIR_DROUGHT_ZONE, exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Aggregation method for water balance calculations
# 'since_june1': Aggregate from June 1 to minimum zone date
# 'n_months_prior': Aggregate for N months prior to minimum zone date
AGGREGATION_METHOD = 'n_months_prior'  # Options: 'since_june1' or 'n_months_prior'

# Number of months prior to minimum zone to analyze (only used if method is 'n_months_prior')
N_MONTHS_PRIOR = 9  # Analyze water balance for N months leading up to minimum zone

# Drop "Normal or Above" category from plots
DROP_NORMAL = False  # Set to False to include "Normal or Above" category

# Minimum inflow threshold for filtering (MG)
MIN_INFLOW_THRESHOLD = 1000  # Filter out years with total inflow below this value

# NYC reservoir parameters
NYC_RESERVOIRS = ['cannonsville', 'pepacton', 'neversink']

# Drought zone mapping
ZONE_NAMES = {
    6: 'Drought Emergency',
    5: 'Drought Watch',
    4: 'Drought Warning',
    3: 'Normal',
    2: 'Flood Watch',
    1: 'Flood Warning',
}

# Drought zone categories for plotting
DROUGHT_CATEGORIES = {
    'emergency': {'zones': [6], 'label': 'Drought Emergency', 'color': '#8B0000'},
    'watch': {'zones': [5], 'label': 'Drought Watch', 'color': '#FF4500'},
    'warning': {'zones': [4], 'label': 'Drought Warning', 'color': '#FFA500'},
    'other': {'zones': [1, 2, 3], 'label': 'Normal or Above', 'color': '#4682B4'},
}


def classify_years_by_min_zone(res_level_df):
    """
    Classify each year by the minimum drought zone reached.

    Parameters
    ----------
    res_level_df : pd.DataFrame
        Reservoir level DataFrame with 'nyc' column and datetime index

    Returns
    -------
    year_classifications : dict
        Dictionary mapping year -> (min_zone, min_zone_date)
    """
    # Add year column
    df = res_level_df.copy()
    df['year'] = df.index.year

    year_classifications = {}

    for year in df['year'].unique():
        year_data = df[df['year'] == year]

        # Find maximum zone value (higher zone = more severe drought)
        # Zone 6 is most severe drought, Zone 1 is flood
        max_zone = year_data['nyc'].max()

        # Find date when maximum zone occurred
        max_zone_date = year_data[year_data['nyc'] == max_zone].index[0]

        year_classifications[year] = {
            'min_zone': max_zone,
            'min_zone_date': max_zone_date
        }

    return year_classifications


def calculate_n_month_aggregates(year_classifications, inflow_series, contribution_series):
    """
    Calculate aggregates of inflow and contributions prior to minimum zone.

    Uses the aggregation method specified by AGGREGATION_METHOD:
    - 'since_june1': Aggregate from June 1 to minimum zone date
    - 'n_months_prior': Aggregate for N months prior to minimum zone date

    Parameters
    ----------
    year_classifications : dict
        Output from classify_years_by_min_zone()
    inflow_series : pd.Series
        Daily NYC inflow with datetime index
    contribution_series : pd.Series
        Daily NYC contributions with datetime index

    Returns
    -------
    aggregates : pd.DataFrame
        DataFrame with columns: year, min_zone, inflow_total, contribution_total
    """
    records = []

    for year, info in year_classifications.items():
        min_zone = info['min_zone']
        min_zone_date = info['min_zone_date']

        # Determine start date based on aggregation method
        if AGGREGATION_METHOD == 'since_june1':
            # Start from June 1 of the same year as min_zone_date
            # Note: For droughts occurring Jan-May, this will use June 1 of previous year
            year_of_min_zone = min_zone_date.year
            if min_zone_date.month >= 6:
                # Use June 1 of same year
                start_date = pd.Timestamp(year=year_of_min_zone, month=6, day=1)
            else:
                # Use June 1 of previous year (for droughts in Jan-May)
                start_date = pd.Timestamp(year=year_of_min_zone - 1, month=6, day=1)

        elif AGGREGATION_METHOD == 'n_months_prior':
            # Calculate start date (N months before minimum zone)
            start_date = min_zone_date - pd.DateOffset(months=N_MONTHS_PRIOR)

        else:
            raise ValueError(f"Invalid AGGREGATION_METHOD: {AGGREGATION_METHOD}. Must be 'since_june1' or 'n_months_prior'")

        # Sum inflow and contributions in window
        inflow_mask = (inflow_series.index >= start_date) & (inflow_series.index <= min_zone_date)
        contribution_mask = (contribution_series.index >= start_date) & (contribution_series.index <= min_zone_date)

        inflow_total = inflow_series[inflow_mask].sum()
        contribution_total = contribution_series[contribution_mask].sum()

        records.append({
            'year': year,
            'min_zone': min_zone,
            'inflow_total': inflow_total,
            'contribution_total': contribution_total
        })

    return pd.DataFrame(records)


def extract_water_balance_by_zone(data, dataset_id, realization_id):
    """
    Extract water balance components and classify by drought zone.

    Parameters
    ----------
    data : pywrdrb.Data
        Data object
    dataset_id : str
        Dataset identifier
    realization_id : int
        Realization identifier

    Returns
    -------
    aggregates_df : pd.DataFrame
        DataFrame with N-month aggregates classified by drought zone
    """
    # Get reservoir level data (drought zones)
    res_level_df = data.res_level[dataset_id][realization_id]

    # Get inflows to NYC reservoirs
    inflow_df = data.inflow[dataset_id][realization_id]
    nyc_inflow = inflow_df[NYC_RESERVOIRS].sum(axis=1)

    # Get NYC contributions to Montague
    contribution_df = data.contribution[dataset_id][realization_id]
    nyc_contributions = contribution_df['mrf_montagueTrenton_nyc']

    # Classify years by minimum drought zone
    year_classifications = classify_years_by_min_zone(res_level_df)

    # Calculate N-month aggregates
    aggregates_df = calculate_n_month_aggregates(
        year_classifications, nyc_inflow, nyc_contributions
    )

    return aggregates_df


def aggregate_across_realizations(data, dataset_id):
    """
    Aggregate water balance data across all realizations.

    Parameters
    ----------
    data : pywrdrb.Data
        Data object
    dataset_id : str
        Dataset identifier

    Returns
    -------
    all_aggregates : pd.DataFrame
        Combined DataFrame from all realizations
    """
    realization_ids = list(data.res_level[dataset_id].keys())

    all_aggregates = []

    for real_id in realization_ids:
        print(f"  Processing realization {real_id}...")

        # Extract aggregates for this realization
        aggregates_df = extract_water_balance_by_zone(data, dataset_id, real_id)
        aggregates_df['realization_id'] = real_id

        all_aggregates.append(aggregates_df)

    # Combine all realizations
    combined_df = pd.concat(all_aggregates, ignore_index=True)

    return combined_df


def categorize_by_drought_zone(aggregates_df):
    """
    Categorize aggregates by drought zone categories.

    Parameters
    ----------
    aggregates_df : pd.DataFrame
        Combined aggregates from all realizations

    Returns
    -------
    categorized_data : dict
        Dictionary mapping category name -> DataFrame subset
    """
    categorized_data = {}

    for cat_name, cat_info in DROUGHT_CATEGORIES.items():
        zones = cat_info['zones']
        mask = aggregates_df['min_zone'].isin(zones)
        categorized_data[cat_name] = aggregates_df[mask]

    return categorized_data


def plot_distributions_by_zone(categorized_data, dataset_id, dataset_label):
    """
    Create KDE plots showing distributions of water balance components by zone.

    Parameters
    ----------
    categorized_data : dict
        Output from categorize_by_drought_zone()
    dataset_id : str
        Dataset identifier
    dataset_label : str
        Dataset display label
    """
    print("Creating KDE distribution plots by drought zone...")

    # Create figure with two subplots stacked vertically (sharing x-axis)
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Prepare data for plotting
    # Drop "Normal or Above" category if DROP_NORMAL is True
    if DROP_NORMAL:
        categories = ['emergency', 'watch', 'warning']
    else:
        categories = ['emergency', 'watch', 'warning', 'other']

    # Plot inflow distributions
    ax = axes[0]

    for cat in categories:
        cat_info = DROUGHT_CATEGORIES[cat]
        df = categorized_data[cat].copy()

        if len(df) > 0:
            # Filter out low inflow values
            df_filtered = df[df['inflow_total'] > MIN_INFLOW_THRESHOLD]

            # Report filtering
            n_filtered = len(df) - len(df_filtered)
            if n_filtered > 0:
                print(f"  {cat_info['label']}: Filtered {n_filtered} years with inflow <= {MIN_INFLOW_THRESHOLD} MG (remaining: {len(df_filtered)})")

            if len(df_filtered) > 0:
                # Verify minimum value after filtering
                min_val = df_filtered['inflow_total'].min()
                print(f"  {cat_info['label']}: Min inflow after filtering = {min_val:.2f} MG")

                # Plot KDE
                df_filtered['inflow_total'].plot.kde(
                    ax=ax,
                    color=cat_info['color'],
                    linewidth=2.5,
                    alpha=0.8,
                    label=cat_info['label']
                )

                # Plot mean value line
                mean_val = df_filtered['inflow_total'].mean()
                ax.axvline(mean_val, color=cat_info['color'], linestyle='--',
                          linewidth=1.5, alpha=0.7)

    # X-axis label depends on aggregation method
    if AGGREGATION_METHOD == 'since_june1':
        xlabel = 'Total Inflow (June 1 to min zone, MG)'
    else:
        xlabel = f'Total Inflow ({N_MONTHS_PRIOR}-month prior to min zone, MG)'

    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title('NYC Inflow Distribution by Drought Zone',
                 fontsize=13, fontweight='bold', pad=15)
    ax.grid(axis='both', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Set x and y limits based on DROP_NORMAL setting
    # if DROP_NORMAL:
    #     ax.set_xlim(0, 200000)
    # else:
    #     ax.set_xlim(0, 400000)

    # Plot contribution distributions
    ax = axes[1]

    for cat in categories:
        cat_info = DROUGHT_CATEGORIES[cat]
        df = categorized_data[cat].copy()

        if len(df) > 0:
            # Filter out low inflow values (same filter as above)
            df_filtered = df[df['inflow_total'] > MIN_INFLOW_THRESHOLD]

            if len(df_filtered) > 0:
                # Plot KDE
                df_filtered['contribution_total'].plot.kde(
                    ax=ax,
                    color=cat_info['color'],
                    linewidth=2.5,
                    alpha=0.8,
                    label=cat_info['label']
                )

                # Plot mean value line
                mean_val = df_filtered['contribution_total'].mean()
                ax.axvline(mean_val, color=cat_info['color'], linestyle='--',
                          linewidth=1.5, alpha=0.7)

    # X-axis label depends on aggregation method
    if AGGREGATION_METHOD == 'since_june1':
        xlabel = 'Total NYC Contributions to Montague (June 1 to min zone, MG)'
    else:
        xlabel = f'Total NYC Contributions to Montague ({N_MONTHS_PRIOR}-month prior to min zone, MG)'

    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title('NYC Contributions Distribution by Drought Zone',
                 fontsize=13, fontweight='bold', pad=15)
    ax.grid(axis='both', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Set x and y limits based on DROP_NORMAL setting
    # if DROP_NORMAL:
    #     ax.set_xlim(0, 200000)
    # else:
    #     ax.set_xlim(0, 400000)

    # Single legend at the bottom with reordered zones
    handles, labels = axes[1].get_legend_handles_labels()

    # Reorder legend: Normal or Flood, Drought Warning, Drought Watch, Drought Emergency
    desired_order = ['Normal or Above', 'Drought Warning', 'Drought Watch', 'Drought Emergency']

    # Create ordered lists
    ordered_handles = []
    ordered_labels = []
    for desired_label in desired_order:
        if desired_label in labels:
            idx = labels.index(desired_label)
            ordered_handles.append(handles[idx])
            ordered_labels.append(labels[idx])

    fig.legend(ordered_handles, ordered_labels, loc='lower center', ncol=4,
               fontsize=11, frameon=True, fancybox=True, bbox_to_anchor=(0.5, -0.02))

    # Overall title
    fig.suptitle(f'Water Balance Distributions by Drought Zone\n{dataset_label}',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    # Save
    fname = f"{FIG_DIR_DROUGHT_ZONE}/{dataset_id}_water_balance_by_drought_zone_{N_MONTHS_PRIOR}M.png"
    plt.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close()


def plot_contribution_ratio_by_zone(categorized_data, dataset_id, dataset_label):
    """
    Create KDE plot showing distribution of NYC contributions / inflow ratio by zone.

    Parameters
    ----------
    categorized_data : dict
        Output from categorize_by_drought_zone()
    dataset_id : str
        Dataset identifier
    dataset_label : str
        Dataset display label
    """
    print("Creating contribution ratio KDE plot...")

    # Create single panel figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Prepare data for plotting
    # Drop "Normal or Above" category if DROP_NORMAL is True
    if DROP_NORMAL:
        categories = ['emergency', 'watch', 'warning']
    else:
        categories = ['emergency', 'watch', 'warning', 'other']

    # Plot contribution ratio distributions
    for cat in categories:
        cat_info = DROUGHT_CATEGORIES[cat]
        df = categorized_data[cat].copy()

        if len(df) > 0:
            # Filter out low inflow values
            df_filtered = df[df['inflow_total'] > MIN_INFLOW_THRESHOLD]

            if len(df_filtered) > 0:
                # Calculate contribution ratio (as percentage)
                contribution_ratio = 100.0 * df_filtered['contribution_total'] / df_filtered['inflow_total']

                # Plot KDE
                contribution_ratio.plot.kde(
                    ax=ax,
                    color=cat_info['color'],
                    linewidth=2.5,
                    alpha=0.8,
                    label=cat_info['label']
                )

                # Plot mean value line
                mean_val = contribution_ratio.mean()
                ax.axvline(mean_val, color=cat_info['color'], linestyle='--',
                          linewidth=1.5, alpha=0.7)

    # X-axis label depends on aggregation method
    if AGGREGATION_METHOD == 'since_june1':
        xlabel = 'NYC Contributions / Total Inflow (June 1 to min zone, %)'
    else:
        xlabel = f'NYC Contributions / Total Inflow ({N_MONTHS_PRIOR}-month prior to min zone, %)'

    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title(f'NYC Contribution Ratio Distribution by Drought Zone\n{dataset_label}',
                 fontsize=13, fontweight='bold', pad=15)
    ax.grid(axis='both', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.set_xlim(left=0, right=100)

    # Legend with reordered zones
    handles, labels = ax.get_legend_handles_labels()

    # Reorder legend: Normal or Flood, Drought Warning, Drought Watch, Drought Emergency
    desired_order = ['Normal or Above', 'Drought Warning', 'Drought Watch', 'Drought Emergency']

    # Create ordered lists
    ordered_handles = []
    ordered_labels = []
    for desired_label in desired_order:
        if desired_label in labels:
            idx = labels.index(desired_label)
            ordered_handles.append(handles[idx])
            ordered_labels.append(labels[idx])

    ax.legend(ordered_handles, ordered_labels, loc='best', fontsize=11, frameon=True, fancybox=True)

    plt.tight_layout()

    # Save
    fname = f"{FIG_DIR_DROUGHT_ZONE}/{dataset_id}_contribution_ratio_by_drought_zone_{N_MONTHS_PRIOR}M.png"
    plt.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close()


def main(dataset_id):
    """
    Main function to generate water balance by drought zone plots.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    """
    print("=" * 80)
    print(f"NYC WATER BALANCE BY DROUGHT ZONE: {dataset_id}")
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
        print(f"    -> Aggregating from June 1 to minimum zone date")
    else:
        print(f"    -> Aggregating {N_MONTHS_PRIOR} months prior to minimum zone date")
    print(f"  Drop Normal category: {DROP_NORMAL}")
    print(f"  Min inflow threshold: {MIN_INFLOW_THRESHOLD:,} MG")

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
        results_sets=['res_level', 'inflow', 'contribution']
    )
    print("  Data loaded successfully")

    # Aggregate across realizations
    print(f"\nAggregating data across realizations...")
    all_aggregates = aggregate_across_realizations(data, dataset_id)
    print(f"  Total years analyzed: {len(all_aggregates)}")

    # Categorize by drought zone
    print("\nCategorizing by drought zone...")
    categorized_data = categorize_by_drought_zone(all_aggregates)

    # Create plots
    print("\nCreating plots...")
    # plot_distributions_by_zone(categorized_data, dataset_id, dataset_label)
    plot_contribution_ratio_by_zone(categorized_data, dataset_id, dataset_label)

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
