"""
Plot performance metrics across ensembles with box plot distributions.

Shows:
- Left panel: Absolute performance distributions for all datasets (box plots)
- Right panel: Percentage change distributions relative to baseline (box plots)

This script uses pre-calculated metrics from postprocessing:
- shortage: Pre-calculated flow target violations
- mrf_target: Flow targets for calculating reliability
- res_storage: Reservoir storage for NYC system

Features:
- Box plots show full distribution of outcomes across realizations
- Flexible metric selection: Plot any list of performance metrics
- Dynamic dataset handling: Automatically adjusts based on config.py datasets
- Smart y-axis labeling: Detects metric types for appropriate labels

Usage:
  python 09_plot_performance_outcome_boxplots.py
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from methods.config import *


# Output directory
FIG_OUTPUT_DIR = f"{FIG_DIR}/performance_metrics"
os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

# Performance metrics directory
PERFORMANCE_METRICS_DIR = f"{ROOT_DIR}/pywrdrb/performance_metrics"

# ============================================================================
# CONFIGURABLE METRICS
# ============================================================================
# Specify which metrics to plot and in what order
# The order of this list determines the order of boxes in the plot
#
# Available metrics (from methods/postprocess.py calculate_performance_metrics):
#
# See PERFORMANCE_METRICS_DOCUMENTATION.md for comprehensive descriptions.
#
# CATEGORY 1: Flow Reliability (Montague & Trenton)
#   - years_reliable_montague: Years Montague flow target met >90% of days
#   - years_reliable_montague_95: Years Montague flow target met >95% of days
#   - mean_annual_montague_reliability: Average annual Montague reliability (0-1)
#   - min_annual_montague_reliability: Worst annual Montague reliability
#   - total_montague_shortage_mg: Total Montague shortage (MG)
#   - mean_annual_montague_shortage_mg: Mean annual Montague shortage (MG/year)
#   - years_reliable_trenton: Years Trenton flow target met >90% of days
#   - years_reliable_trenton_95: Years Trenton flow target met >95% of days
#   - mean_annual_trenton_reliability: Average annual Trenton reliability (0-1)
#   - total_trenton_shortage_mg: Total Trenton shortage (MG)
#   - mean_annual_trenton_shortage_mg: Mean annual Trenton shortage (MG/year)
#
# CATEGORY 2: NYC Reservoir Storage
#   - years_above_30pct: Years min storage stays >30%
#   - years_above_20pct: Years min storage stays >20%
#   - years_above_10pct: Years min storage stays >10%
#   - years_below_10pct: Years min storage drops ≤10%
#   - years_high_storage_june1: Years ≥95% storage on June 1
#   - years_high_storage_june1_90: Years ≥90% storage on June 1
#   - mean_june1_storage_pct: Average June 1 storage (%)
#   - mean_sept1_storage_pct: Average Sept 1 storage (%)
#   - years_low_carryover: Years <50% storage on Sept 1
#   - years_low_carryover_40: Years <40% storage on Sept 1
#   - mean_storage_pct: Long-term average storage (%)
#   - median_storage_pct: Median storage (%)
#   - min_storage_pct: Absolute minimum storage (%)
#   - max_storage_pct: Maximum storage (%)
#   - std_storage_pct: Storage standard deviation (%)
#   - pct_days_storage_below_30: % days storage <30%
#   - pct_days_storage_below_20: % days storage <20%
#   - mean_annual_storage_range: Average annual storage swing (%)
#
# CATEGORY 3: Water Supply Reliability
#   - pct_days_nyc_diversion_shortage: % days NYC diversion shortage
#   - total_nyc_diversion_shortage_mg: Total NYC diversion shortage (MG)
#   - mean_annual_nyc_diversion_shortage_mg: Mean annual NYC shortage (MG/year)
#   - max_daily_nyc_diversion_shortage_mg: Max daily NYC shortage (MGD)
#   - years_no_nyc_shortage: Years with zero NYC shortage
#   - years_minor_nyc_shortage: Years with ≤365 MG shortage
#
# CATEGORY 4: Drought Characteristics
#   - max_consecutive_drought_days: Longest Montague drought (days)
#   - mean_drought_duration_days: Average Montague drought duration (days)
#   - n_drought_events: Number of Montague drought events
#   - n_major_droughts: Number of ≥90-day droughts
#   - n_severe_droughts: Number of ≥180-day droughts
#   - worst_drought_max_daily_shortage_mg: Peak shortage in worst drought (MGD)
#   - max_consecutive_drought_days_trenton: Longest Trenton drought (days)
#   - n_drought_events_trenton: Number of Trenton drought events
#   - pct_days_combined_stress: % days with both NYC & Montague shortage
#
# CATEGORY 5: NYC Contributions
#   - mean_annual_nyc_contribution_mg: Mean annual NYC contribution (MG/year)
#   - max_annual_nyc_contribution_mg: Max annual NYC contribution (MG/year)
#   - min_annual_nyc_contribution_mg: Min annual NYC contribution (MG/year)
#   - std_annual_nyc_contribution_mg: Std dev annual NYC contribution (MG/year)
#   - total_nyc_contribution_mg: Total NYC contribution (MG)
#   - pct_days_nyc_contribution: % days with NYC contribution
#   - n_days_high_nyc_contribution: Days with >100 MGD contribution
#
# CATEGORY 6: System Balance
#   - nyc_contribution_to_shortage_ratio: NYC contribution / Montague shortage
#   - years_high_storage_and_reliable: Years with high storage AND reliability
#   - years_vulnerable: Years with low storage OR low reliability
#
# LEGACY (backward compatibility):
#   - years_reliable: Alias for years_reliable_montague
#   - years_high_storage: Alias for years_high_storage_june1
#   - years_trenton_reliable: Alias for years_reliable_trenton

METRICS_TO_PLOT = [
    'years_reliable',
    'years_high_storage',
    'years_above_20pct',
    'years_above_10pct',
    'years_low_carryover',
    'years_trenton_reliable',
    'max_consecutive_drought_days',
    'mean_annual_nyc_contribution_mg',
    'pct_days_nyc_contribution',
    'years_vulnerable',
]

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
# Specify which dataset to use as baseline (for calculating percentage changes)
BASELINE_DATASET = 'stationary_ensemble'

# Option to manually specify which datasets to plot
# Set to None to automatically use all datasets from config.py
# Set to list of dataset_ids to manually specify
DATASETS_TO_PLOT = None  # None = use all datasets from config.py

# ============================================================================
# DISPLAY CONFIGURATION
# ============================================================================
# Metric display names (for plot labels)
METRIC_DISPLAY_NAMES = {
    # Flow Reliability - Montague
    'years_reliable_montague': 'Years Montague\nReliable (>90%)',
    'years_reliable_montague_95': 'Years Montague\nReliable (>95%)',
    'mean_annual_montague_reliability': 'Mean Annual\nMontague Reliability',
    'min_annual_montague_reliability': 'Worst Annual\nMontague Reliability',
    'total_montague_shortage_mg': 'Total Montague\nShortage (MG)',
    'mean_annual_montague_shortage_mg': 'Mean Annual\nMontague Shortage',

    # Flow Reliability - Trenton
    'years_reliable_trenton': 'Years Trenton\nReliable (>90%)',
    'years_reliable_trenton_95': 'Years Trenton\nReliable (>95%)',
    'mean_annual_trenton_reliability': 'Mean Annual\nTrenton Reliability',
    'total_trenton_shortage_mg': 'Total Trenton\nShortage (MG)',
    'mean_annual_trenton_shortage_mg': 'Mean Annual\nTrenton Shortage',

    # NYC Storage - Thresholds
    'years_above_30pct': 'Years Min\nStorage >30%',
    'years_above_20pct': 'Years Min\nStorage >20%',
    'years_above_10pct': 'Years Min\nStorage >10%',
    'years_below_10pct': 'Years Min\nStorage ≤10%',

    # NYC Storage - Key Dates
    'years_high_storage_june1': 'Years June 1\nStorage ≥95%',
    'years_high_storage_june1_90': 'Years June 1\nStorage ≥90%',
    'mean_june1_storage_pct': 'Mean June 1\nStorage (%)',
    'mean_sept1_storage_pct': 'Mean Sept 1\nStorage (%)',
    'years_low_carryover': 'Years Sept 1\nStorage <50%',
    'years_low_carryover_40': 'Years Sept 1\nStorage <40%',

    # NYC Storage - Statistics
    'mean_storage_pct': 'Mean Storage (%)',
    'median_storage_pct': 'Median Storage (%)',
    'min_storage_pct': 'Min Storage (%)',
    'max_storage_pct': 'Max Storage (%)',
    'std_storage_pct': 'Storage Std Dev (%)',
    'pct_days_storage_below_30': '% Days\nStorage <30%',
    'pct_days_storage_below_20': '% Days\nStorage <20%',
    'mean_annual_storage_range': 'Mean Annual\nStorage Range (%)',

    # Water Supply Reliability
    'pct_days_nyc_diversion_shortage': '% Days NYC\nDiversion Shortage',
    'total_nyc_diversion_shortage_mg': 'Total NYC\nDiversion Shortage',
    'mean_annual_nyc_diversion_shortage_mg': 'Mean Annual NYC\nDiversion Shortage',
    'max_daily_nyc_diversion_shortage_mg': 'Max Daily NYC\nDiversion Shortage',
    'years_no_nyc_shortage': 'Years No\nNYC Shortage',
    'years_minor_nyc_shortage': 'Years Minor\nNYC Shortage',

    # Drought Characteristics
    'max_consecutive_drought_days': 'Max Consecutive\nDrought (days)',
    'mean_drought_duration_days': 'Mean Drought\nDuration (days)',
    'n_drought_events': 'Number of\nDrought Events',
    'n_major_droughts': 'Number of\nMajor Droughts',
    'n_severe_droughts': 'Number of\nSevere Droughts',
    'worst_drought_max_daily_shortage_mg': 'Worst Drought\nPeak Shortage',
    'max_consecutive_drought_days_trenton': 'Max Consecutive\nTrenton Drought',
    'n_drought_events_trenton': 'Number of\nTrenton Droughts',
    'pct_days_combined_stress': '% Days Combined\nSystem Stress',

    # NYC Contributions
    'mean_annual_nyc_contribution_mg': 'Mean Annual NYC\nContribution (MG)',
    'max_annual_nyc_contribution_mg': 'Max Annual NYC\nContribution (MG)',
    'min_annual_nyc_contribution_mg': 'Min Annual NYC\nContribution (MG)',
    'std_annual_nyc_contribution_mg': 'NYC Contribution\nStd Dev (MG)',
    'total_nyc_contribution_mg': 'Total NYC\nContribution (MG)',
    'pct_days_nyc_contribution': '% Days NYC\nContribution',
    'n_days_high_nyc_contribution': 'Days High NYC\nContribution (>100 MGD)',

    # System Balance
    'nyc_contribution_to_shortage_ratio': 'NYC Contribution /\nShortage Ratio',
    'years_high_storage_and_reliable': 'Years High Storage\n& Reliable',
    'years_vulnerable': 'Years\nVulnerable',

    # Legacy
    'years_reliable': 'Years Montague\nReliable',
    'years_high_storage': 'Years June 1\nStorage High',
}

# Color palette for datasets (will cycle if more datasets than colors)
DATASET_COLORS = {
    'stationary_ensemble': '#2E86AB',           # Blue
    'climate_adjusted_low': '#C73E1D',          # Red
    'climate_adjusted_high': '#06A77D',         # Teal
}

# Reconstruction scaling factor
# Reconstruction has 79 years, ensembles have 70 years
# Scale reconstruction metrics to be comparable
RECONSTRUCTION_YEARS = 79
ENSEMBLE_YEARS = 70
RECONSTRUCTION_SCALE_FACTOR = ENSEMBLE_YEARS / RECONSTRUCTION_YEARS  # 70/79 ≈ 0.886

# Metrics that should be scaled (count of years)
METRICS_TO_SCALE = [
    'years_reliable',
    'years_high_storage',
    'years_above_20pct',
    'years_above_10pct',
    'years_low_carryover',
    'years_trenton_reliable',
]

# ============================================================================
# METRIC METADATA - For smart y-axis labeling
# ============================================================================
# Categorize metrics by their units/types
METRIC_UNITS = {
    # Year count metrics
    'years_reliable': 'years',
    'years_high_storage': 'years',
    'years_above_20pct': 'years',
    'years_above_10pct': 'years',
    'years_low_carryover': 'years',
    'years_trenton_reliable': 'years',

    # Percentage metrics
    'mean_sept1_storage_pct': 'percent',
    'pct_days_nyc_diversion_shortage': 'percent',

    # Duration metrics (days)
    'max_consecutive_drought_days': 'days',

    # Volume metrics (million gallons)
    'mean_annual_nyc_contribution_mg': 'million_gallons',
    'max_annual_nyc_contribution_mg': 'million_gallons',
}

# Y-axis labels for different metric types
Y_AXIS_LABELS = {
    'years': 'Number of Years (out of 70)',
    'percent': 'Percentage (%)',
    'days': 'Days',
    'million_gallons': 'Million Gallons (MG)',
}

def get_ylabel_for_metrics(metric_list):
    """
    Determine appropriate y-axis label for a list of metrics.
    If all metrics have same units, return that unit's label.
    Otherwise, return a generic label.
    """
    units = set(METRIC_UNITS.get(m, 'value') for m in metric_list)

    if len(units) == 1:
        unit = units.pop()
        return Y_AXIS_LABELS.get(unit, 'Value')
    else:
        return 'Value'


def load_performance_metrics(dataset_id):
    """
    Load pre-calculated performance metrics from CSV.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier

    Returns
    -------
    metrics_df : pd.DataFrame
        DataFrame with performance metrics for all realizations
    """
    csv_file = f"{PERFORMANCE_METRICS_DIR}/{dataset_id}_performance_metrics.csv"

    if not os.path.exists(csv_file):
        raise FileNotFoundError(
            f"Performance metrics not found: {csv_file}\n"
            f"Run 04_postprocess_data.py first to calculate metrics!"
        )

    metrics_df = pd.read_csv(csv_file, index_col='realization_id')
    return metrics_df


def validate_metrics(metrics_df, dataset_id):
    """
    Validate that all requested metrics exist in the DataFrame.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame with performance metrics
    dataset_id : str
        Dataset identifier (for error messages)

    Raises
    ------
    ValueError
        If any requested metrics are missing
    """
    available_metrics = set(metrics_df.columns)
    requested_metrics = set(METRICS_TO_PLOT)
    missing_metrics = requested_metrics - available_metrics

    if missing_metrics:
        raise ValueError(
            f"ERROR: Dataset '{dataset_id}' is missing requested metrics: {missing_metrics}\n"
            f"Available metrics: {sorted(available_metrics)}\n"
            f"Requested metrics: {sorted(requested_metrics)}\n"
            f"Please update METRICS_TO_PLOT or regenerate metrics CSV."
        )


def get_datasets_from_config():
    """
    Get list of datasets from config.py.

    Returns
    -------
    datasets_to_plot : list
        List of dataset IDs to plot
    dataset_labels : dict
        Display labels for each dataset
    """
    from methods.config import DATASET_CONFIGS

    # Start with all datasets from config
    all_datasets = list(DATASET_CONFIGS.keys())

    # Determine which datasets to plot
    if DATASETS_TO_PLOT is None:
        # Use all datasets from config
        datasets_to_plot = all_datasets
    else:
        # Use manually specified datasets
        datasets_to_plot = DATASETS_TO_PLOT
        # Verify they exist
        for d in datasets_to_plot:
            if d not in all_datasets:
                raise ValueError(f"Dataset '{d}' not found in config.py!")

    # Verify baseline exists
    if BASELINE_DATASET not in datasets_to_plot:
        raise ValueError(
            f"Baseline dataset '{BASELINE_DATASET}' not in datasets to plot!\n"
            f"Datasets to plot: {datasets_to_plot}"
        )

    # Create display labels from descriptions
    dataset_labels = {}
    for dataset_id in datasets_to_plot:
        config = DATASET_CONFIGS[dataset_id]
        desc = config.get('description', dataset_id)
        dataset_labels[dataset_id] = desc

    return datasets_to_plot, dataset_labels


def prepare_data_for_boxplots(all_metrics_dfs, datasets_to_plot, dataset_labels):
    """
    Prepare data in long format for seaborn box plots.

    Parameters
    ----------
    all_metrics_dfs : dict
        Dictionary mapping dataset_id to metrics DataFrame
    datasets_to_plot : list
        List of dataset IDs to include
    dataset_labels : dict
        Display labels for datasets

    Returns
    -------
    df_absolute : pd.DataFrame
        Long-format DataFrame for absolute values
    df_pct_change : pd.DataFrame
        Long-format DataFrame for percentage changes
    """
    # Prepare absolute values
    dfs_absolute = []
    for dataset_id in datasets_to_plot:
        df = all_metrics_dfs[dataset_id][METRICS_TO_PLOT].copy()
        df['dataset'] = dataset_labels[dataset_id]
        df['dataset_id'] = dataset_id
        df['realization'] = df.index
        dfs_absolute.append(df)

    df_absolute = pd.concat(dfs_absolute, ignore_index=True)

    # Melt to long format for seaborn
    df_absolute = df_absolute.melt(
        id_vars=['dataset', 'dataset_id', 'realization'],
        value_vars=METRICS_TO_PLOT,
        var_name='metric',
        value_name='value'
    )

    # Prepare percentage changes (relative to baseline)
    baseline_df = all_metrics_dfs[BASELINE_DATASET][METRICS_TO_PLOT]

    dfs_pct_change = []
    for dataset_id in datasets_to_plot:
        if dataset_id == BASELINE_DATASET:
            continue  # Skip baseline for percentage change

        df = all_metrics_dfs[dataset_id][METRICS_TO_PLOT].copy()

        # Calculate percentage change for each metric
        for metric in METRICS_TO_PLOT:
            baseline_values = baseline_df[metric].values
            current_values = df[metric].values

            # Calculate pairwise percentage changes
            # Use epsilon to avoid division by zero
            eps = 1e-8
            pct_change = 100.0 * (current_values - baseline_values) / np.maximum(np.abs(baseline_values), eps)

            df[metric] = pct_change

        df['dataset'] = dataset_labels[dataset_id]
        df['dataset_id'] = dataset_id
        df['realization'] = df.index
        dfs_pct_change.append(df)

    if dfs_pct_change:
        df_pct_change = pd.concat(dfs_pct_change, ignore_index=True)

        # Melt to long format
        df_pct_change = df_pct_change.melt(
            id_vars=['dataset', 'dataset_id', 'realization'],
            value_vars=METRICS_TO_PLOT,
            var_name='metric',
            value_name='pct_change'
        )
    else:
        df_pct_change = None

    return df_absolute, df_pct_change


def identify_top_changing_metrics(all_metrics_dfs, baseline_dataset, comparison_datasets,
                                  all_metrics=None, n_top=10):
    """
    Identify metrics with largest mean percentage changes relative to baseline.

    This function calculates the mean percentage change for each metric across
    all comparison datasets and ranks them to identify the most sensitive metrics.

    Parameters
    ----------
    all_metrics_dfs : dict
        Dictionary mapping dataset_id to metrics DataFrame
    baseline_dataset : str
        Dataset ID to use as baseline for comparison
    comparison_datasets : list
        List of dataset IDs to compare against baseline
    all_metrics : list, optional
        List of all available metrics to consider. If None, uses all metrics
        from the baseline dataset.
    n_top : int, optional
        Number of top changing metrics to return. Default: 10

    Returns
    -------
    top_metrics_df : pd.DataFrame
        DataFrame with columns:
        - metric: Metric name
        - mean_abs_pct_change: Mean absolute percentage change across comparison datasets
        - mean_pct_change: Mean percentage change (signed) across comparison datasets
        - max_abs_pct_change: Maximum absolute percentage change
        - direction: 'increase', 'decrease', or 'mixed'
    """
    import numpy as np
    import pandas as pd

    # Get all available metrics if not specified
    if all_metrics is None:
        all_metrics = [col for col in all_metrics_dfs[baseline_dataset].columns]

    # Calculate percentage changes for each metric across all comparison datasets
    metric_changes = {}

    baseline_df = all_metrics_dfs[baseline_dataset]

    for metric in all_metrics:
        if metric not in baseline_df.columns:
            continue

        baseline_values = baseline_df[metric].values
        baseline_mean = baseline_values.mean()

        # Skip if baseline mean is essentially zero
        if abs(baseline_mean) < 1e-8:
            continue

        pct_changes = []

        for comp_dataset in comparison_datasets:
            comp_df = all_metrics_dfs[comp_dataset]

            if metric not in comp_df.columns:
                continue

            comp_values = comp_df[metric].values
            comp_mean = comp_values.mean()

            # Calculate percentage change
            pct_change = 100.0 * (comp_mean - baseline_mean) / abs(baseline_mean)
            pct_changes.append(pct_change)

        if pct_changes:
            mean_pct_change = np.mean(pct_changes)
            mean_abs_pct_change = np.mean([abs(pc) for pc in pct_changes])
            max_abs_pct_change = max([abs(pc) for pc in pct_changes])

            # Determine direction
            if all(pc >= 0 for pc in pct_changes):
                direction = 'increase'
            elif all(pc <= 0 for pc in pct_changes):
                direction = 'decrease'
            else:
                direction = 'mixed'

            metric_changes[metric] = {
                'mean_abs_pct_change': mean_abs_pct_change,
                'mean_pct_change': mean_pct_change,
                'max_abs_pct_change': max_abs_pct_change,
                'direction': direction
            }

    # Convert to DataFrame and sort by mean absolute percentage change
    changes_df = pd.DataFrame(metric_changes).T
    changes_df.index.name = 'metric'
    changes_df = changes_df.reset_index()
    changes_df = changes_df.sort_values('mean_abs_pct_change', ascending=False)

    # Return top N
    top_metrics_df = changes_df.head(n_top)

    return top_metrics_df


def print_top_changing_metrics(top_metrics_df, n_display=10):
    """
    Print a formatted table of top changing metrics.

    Parameters
    ----------
    top_metrics_df : pd.DataFrame
        DataFrame from identify_top_changing_metrics()
    n_display : int, optional
        Number of metrics to display. Default: 10
    """
    print(f"\n{'='*80}")
    print(f"TOP {min(n_display, len(top_metrics_df))} METRICS WITH LARGEST CHANGES")
    print(f"{'='*80}")
    print(f"{'Rank':<6}{'Metric':<45}{'Mean Δ%':<12}{'Max |Δ%|':<12}{'Direction':<12}")
    print(f"{'-'*80}")

    for i, row in enumerate(top_metrics_df.head(n_display).itertuples(index=False)):
        rank = i + 1
        metric = row.metric
        mean_change = row.mean_pct_change
        max_abs_change = row.max_abs_pct_change
        direction = row.direction

        # Format with appropriate sign and color indicator
        if direction == 'increase':
            sign = '↑'
        elif direction == 'decrease':
            sign = '↓'
        else:
            sign = '±'

        print(f"{rank:<6}{metric:<45}{mean_change:>+10.1f}%  {max_abs_change:>10.1f}%  {sign} {direction:<10}")

    print(f"{'='*80}\n")


def plot_boxplot_comparison():
    """
    Generate box plot comparison figure showing distributions.

    Layout:
    - Left panel: Absolute value distributions for all datasets
    - Right panel: Percentage change distributions (relative to baseline)
    """

    print("=" * 80)
    print("CREATING BOX PLOT PERFORMANCE METRICS VISUALIZATION")
    print("=" * 80)
    print(f"Metrics to plot ({len(METRICS_TO_PLOT)}): {METRICS_TO_PLOT}")
    print("=" * 80)

    # Get datasets from config
    datasets_to_plot, dataset_labels = get_datasets_from_config()

    print(f"\nDatasets to plot ({len(datasets_to_plot)}): {datasets_to_plot}")
    print(f"Baseline dataset: {BASELINE_DATASET}")
    print("=" * 80)

    # Load all metrics
    all_metrics_dfs = {}

    for dataset_id in datasets_to_plot:
        label = dataset_labels[dataset_id]
        print(f"\nLoading {dataset_id} ({label})...")

        # Load pre-calculated metrics from CSV
        try:
            metrics_df = load_performance_metrics(dataset_id)
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            return None

        # Validate that requested metrics exist
        try:
            validate_metrics(metrics_df, dataset_id)
        except ValueError as e:
            print(str(e))
            return None

        all_metrics_dfs[dataset_id] = metrics_df
        print(f"  Loaded {len(metrics_df)} realizations")

    # Load historic (reconstruction) metrics for comparison
    print(f"\nLoading historic (reconstruction) metrics...")
    print(f"  Note: Reconstruction has {RECONSTRUCTION_YEARS} years, ensembles have {ENSEMBLE_YEARS} years")
    print(f"  Scaling reconstruction year-count metrics by {RECONSTRUCTION_SCALE_FACTOR:.3f}")

    try:
        historic_metrics_df = load_performance_metrics('reconstruction')
        historic_values = {}
        for metric in METRICS_TO_PLOT:
            if metric in historic_metrics_df.columns:
                raw_value = historic_metrics_df[metric].iloc[0]

                # Scale metrics that count years to make them comparable
                if metric in METRICS_TO_SCALE:
                    scaled_value = raw_value * RECONSTRUCTION_SCALE_FACTOR
                    historic_values[metric] = scaled_value
                    print(f"  Historic {metric}: {raw_value:.1f} → {scaled_value:.1f} (scaled)")
                else:
                    historic_values[metric] = raw_value
                    print(f"  Historic {metric}: {raw_value:.1f}")
            else:
                print(f"  WARNING: Historic metric '{metric}' not found")

        if not historic_values:
            historic_values = None
    except FileNotFoundError as e:
        print(f"WARNING: {e}")
        print("Historic values will not be shown on plot.")
        historic_values = None

    # Identify top changing metrics (if we have comparison datasets)
    if len(datasets_to_plot) > 1:
        comparison_datasets_for_analysis = [d for d in datasets_to_plot if d != BASELINE_DATASET]

        if comparison_datasets_for_analysis:
            print(f"\n{'='*80}")
            print("IDENTIFYING TOP CHANGING METRICS")
            print(f"{'='*80}")

            # Get all available metrics
            all_available_metrics = list(all_metrics_dfs[BASELINE_DATASET].columns)

            # Identify top changing metrics
            top_metrics_df = identify_top_changing_metrics(
                all_metrics_dfs,
                BASELINE_DATASET,
                comparison_datasets_for_analysis,
                all_metrics=all_available_metrics,
                n_top=20  # Get top 20 for analysis
            )

            # Print top 15 changing metrics
            print_top_changing_metrics(top_metrics_df, n_display=15)

            # Save to CSV for further analysis
            top_metrics_csv = f"{FIG_OUTPUT_DIR}/top_changing_metrics.csv"
            top_metrics_df.to_csv(top_metrics_csv, index=False)
            print(f"Saved top changing metrics analysis to: {top_metrics_csv}\n")

    # Prepare data for box plots
    print(f"\n{'='*80}")
    print("Preparing data for box plots...")
    print(f"{'='*80}")

    df_absolute, df_pct_change = prepare_data_for_boxplots(
        all_metrics_dfs, datasets_to_plot, dataset_labels
    )

    print(f"  Absolute values: {len(df_absolute)} data points")
    if df_pct_change is not None:
        print(f"  Percentage changes: {len(df_pct_change)} data points")

    # Create figure
    print(f"\n{'='*60}")
    print("Creating box plot figure...")
    print(f"{'='*60}")

    # Determine layout based on whether we have percentage changes
    if df_pct_change is not None and len(datasets_to_plot) > 1:
        # Two-panel layout
        fig, (ax_absolute, ax_pct_change) = plt.subplots(1, 2, figsize=(16, 8))
        has_pct_panel = True
    else:
        # Single panel layout
        fig, ax_absolute = plt.subplots(1, 1, figsize=(10, 8))
        ax_pct_change = None
        has_pct_panel = False

    # Get metric display names
    metric_display_names = [METRIC_DISPLAY_NAMES.get(m, m) for m in METRICS_TO_PLOT]

    # Map metric names to display names in dataframes
    metric_name_map = {m: METRIC_DISPLAY_NAMES.get(m, m) for m in METRICS_TO_PLOT}
    df_absolute['metric_display'] = df_absolute['metric'].map(metric_name_map)
    if df_pct_change is not None:
        df_pct_change['metric_display'] = df_pct_change['metric'].map(metric_name_map)

    # Set up color palette
    palette = {}
    for dataset_id in datasets_to_plot:
        label = dataset_labels[dataset_id]
        if dataset_id in DATASET_COLORS:
            palette[label] = DATASET_COLORS[dataset_id]
        else:
            # Use default seaborn color if not specified
            palette[label] = None

    # Remove None values from palette (let seaborn handle defaults)
    palette = {k: v for k, v in palette.items() if v is not None}

    # Set style
    sns.set_style("whitegrid")

    # ====================================================================
    # LEFT PANEL: Absolute values
    # ====================================================================
    print("  Plotting absolute values...")

    sns.boxplot(
        data=df_absolute,
        x='metric_display',
        y='value',
        hue='dataset',
        ax=ax_absolute,
        palette=palette if palette else None,
        linewidth=1.5,
        fliersize=3
    )

    # Add historic values as scatter points (if available)
    if historic_values is not None:
        x_positions = np.arange(len(METRICS_TO_PLOT))
        historic_scatter_values = [historic_values.get(m, np.nan) for m in METRICS_TO_PLOT]

        ax_absolute.scatter(
            x_positions, historic_scatter_values,
            color='red', s=120, marker='D',
            edgecolors='darkred', linewidths=2.5,
            zorder=10, label='Historic'
        )

    # Formatting
    ax_absolute.set_xlabel('Performance Metric', fontsize=13, fontweight='bold')
    ylabel = get_ylabel_for_metrics(METRICS_TO_PLOT)
    ax_absolute.set_ylabel(ylabel, fontsize=13, fontweight='bold')
    ax_absolute.set_title('(a) Absolute Performance', fontsize=14, fontweight='bold', pad=15)
    ax_absolute.tick_params(axis='both', labelsize=10)
    ax_absolute.set_xticklabels(ax_absolute.get_xticklabels(), rotation=45, ha='right')
    ax_absolute.grid(axis='y', alpha=0.3, linestyle='--')
    ax_absolute.set_axisbelow(True)

    # Move legend to upper left
    ax_absolute.legend(loc='upper left', fontsize=10, frameon=True, fancybox=True)

    # ====================================================================
    # RIGHT PANEL: Percentage changes (if applicable)
    # ====================================================================
    if has_pct_panel and df_pct_change is not None:
        print("  Plotting percentage changes...")

        sns.boxplot(
            data=df_pct_change,
            x='metric_display',
            y='pct_change',
            hue='dataset',
            ax=ax_pct_change,
            palette=palette if palette else None,
            linewidth=1.5,
            fliersize=3
        )

        # Add horizontal line at 0
        ax_pct_change.axhline(0, color='black', linewidth=1.5, linestyle='-', alpha=0.7)

        # Formatting
        ax_pct_change.set_xlabel('Performance Metric', fontsize=13, fontweight='bold')
        ax_pct_change.set_ylabel('% Change from Baseline', fontsize=13, fontweight='bold')
        ax_pct_change.set_title(
            f'(b) % Change from {dataset_labels[BASELINE_DATASET]}',
            fontsize=14, fontweight='bold', pad=15
        )
        ax_pct_change.tick_params(axis='both', labelsize=10)
        ax_pct_change.set_xticklabels(ax_pct_change.get_xticklabels(), rotation=45, ha='right')
        ax_pct_change.grid(axis='y', alpha=0.3, linestyle='--')
        ax_pct_change.set_axisbelow(True)

        # Legend
        ax_pct_change.legend(loc='upper left', fontsize=10, frameon=True, fancybox=True)

    # Overall title
    fig.suptitle(
        'Water System Performance Metrics - Distribution Comparison',
        fontsize=16, fontweight='bold', y=0.98
    )

    plt.tight_layout()

    # Save
    fname = f"{FIG_OUTPUT_DIR}/performance_metrics_boxplot_comparison.png"
    plt.savefig(fname, dpi=400, bbox_inches='tight')
    print(f"\nSaved: {fname}")

    # Also save as SVG
    fname_svg = fname.replace('.png', '.svg')
    plt.savefig(fname_svg, bbox_inches='tight')
    print(f"Saved: {fname_svg}")

    return fig, (ax_absolute, ax_pct_change) if has_pct_panel else (ax_absolute,)


def main():
    """Main entry point."""
    plot_boxplot_comparison()

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
