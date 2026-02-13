"""
F12: Drought Comparison Timeseries Figure.

Compares drought events across multiple ensembles (stationary, climate low, climate high)
by selecting events at a specified exceedance rate.

Layout: 3 vertical panels showing overlaid timeseries
  - Panel 1: NYC Storage (%)
  - Panel 2: NYC Releases to Support Montague (MGD, linear scale)
  - Panel 3: Montague Satisfaction (%)

Features:
  - Year-agnostic time axis (generic month labels)
  - Exceedance-based event selection (similar to F2)
  - Multiple datasets overlaid on same axes
  - SSI_WINDOW = 3 by default (editable)

Usage:
    python F12_plot_drought_comparison_timeseries.py

Generates 1 figure comparing events at 0.1 exceedance rate.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import pywrdrb
from methods.config import FIG_DIR, NYC_RESERVOIRS, NYC_TOTAL_CAPACITY, N_YEARS
from methods.plotting.styles import (
    DPI_HIGH, FONTSIZE_SMALL, FONTSIZE_LABEL,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

FIG_OUTPUT_DIR = f"{FIG_DIR}/F12_drought_comparison"
os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

SATISFICING_DATA_DIR = "./pywrdrb/satisficing_analysis"
DROUGHT_METRICS_DIR = "./pywrdrb/drought_metrics"
SSI_WINDOW = 3  # Editable: 3, 6, or 12
DATASET_ID = 'stationary_ensemble'

# Buffer days before/after drought period
BUFFER_DAYS = 90

# Thresholds (for visualization)
STORAGE_THRESHOLD = 20.0  # %
COLOR_THRESHOLD = 'black'


# ============================================================================
# DATA LOADING
# ============================================================================

def load_drought_data_with_satisficing():
    """
    Load drought events with correct simulation dates and merge with satisficing.

    The drought_events.csv has actual simulation dates (2030-2099).
    The years_with_droughts.csv has satisficing outcomes by year.
    We merge them to get drought events with satisficing info.
    """
    # Load drought events (has correct simulation dates)
    events_fname = f"{DROUGHT_METRICS_DIR}/{DATASET_ID}_ssi{SSI_WINDOW}_drought_events.csv"
    events_df = pd.read_csv(events_fname)
    events_df['start'] = pd.to_datetime(events_df['start'])
    events_df['end'] = pd.to_datetime(events_df['end'])
    events_df['start_year'] = events_df['start'].dt.year

    # Load satisficing data (by year)
    satisficing_fname = f"{SATISFICING_DATA_DIR}/{DATASET_ID}_ssi{SSI_WINDOW}_years_with_droughts.csv"
    satisficing_df = pd.read_csv(satisficing_fname)

    # Rename columns for merge
    if 'realization' in satisficing_df.columns:
        satisficing_df = satisficing_df.rename(columns={'realization': 'realization_id'})

    # Merge on year and realization
    merged = events_df.merge(
        satisficing_df[['year', 'realization_id', 'min_storage_pct', 'max_violation_days', 'satisficing']],
        left_on=['start_year', 'realization_id'],
        right_on=['year', 'realization_id'],
        how='left'
    )

    # Drop rows without satisficing data
    merged = merged.dropna(subset=['min_storage_pct'])

    # Add satisficing category
    merged['storage_pass'] = merged['min_storage_pct'] >= STORAGE_THRESHOLD
    merged['montague_pass'] = merged['max_violation_days'] <= 3  # 3 days threshold

    # convert magnitude and severity to abs val
    merged['severity'] = np.abs(merged['severity'])
    merged['magnitude'] = np.abs(merged['magnitude'])

    print(f"  Loaded {len(merged)} drought events with satisficing data")
    print(f"  Date range: {merged['start'].min()} to {merged['end'].max()}")
    print(f"  Satisficing pass: {merged['satisficing'].sum()}, fail: {(~merged['satisficing']).sum()}")

    return merged


def compute_event_exceedances(df, metric='severity', n_years=N_YEARS):
    """
    Compute exceedance rates for each drought event in a dataset.

    For each realization, computes how many events have metric >= this event's value,
    then divides by n_years to get exceedance rate.

    Parameters
    ----------
    df : pd.DataFrame
        Drought events with 'realization_id' and metric columns
    metric : str
        Metric to use for exceedance calculation (e.g., 'severity', 'magnitude')
    n_years : int
        Number of years per realization for normalization

    Returns
    -------
    exceedances : np.ndarray
        Exceedance rate for each event in df
    """
    exceedances = np.zeros(len(df))

    for idx, row in df.iterrows():
        rid = row['realization_id']
        val = row[metric]

        # Get all events from this realization
        realization_events = df[df['realization_id'] == rid]

        # Count events with metric >= this value
        n_exceedances = np.sum(realization_events[metric].values >= val)

        # Normalize by years
        exceedances[idx] = n_exceedances / n_years

    return exceedances


def select_event_by_exceedance(dataset_id, target_exceedance=0.1, metric='severity',
                                ssi_window=SSI_WINDOW):
    """
    Select a drought event from a dataset based on target exceedance rate.

    Finds the event whose exceedance rate is closest to the target.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier (e.g., 'stationary_ensemble', 'climate_adjusted_low')
    target_exceedance : float
        Target exceedance rate (events per year)
    metric : str
        Metric to use for exceedance calculation (default: 'severity')
    ssi_window : int
        SSI window to use

    Returns
    -------
    selected_event : pd.Series
        The drought event closest to the target exceedance
    """
    # Load drought events for this dataset
    events_fname = f"{DROUGHT_METRICS_DIR}/{dataset_id}_ssi{ssi_window}_drought_events.csv"
    df = pd.read_csv(events_fname)
    df['start'] = pd.to_datetime(df['start'])
    df['end'] = pd.to_datetime(df['end'])
    
    # convert severity & magnitude to abs val
    df['severity'] = np.abs(df['severity'])
    df['magnitude'] = np.abs(df['magnitude'])
    
    # Compute exceedances for all events
    exceedances = compute_event_exceedances(df, metric=metric)

    # Find event closest to target exceedance
    differences = np.abs(exceedances - target_exceedance)
    best_idx = df.index[np.argmin(differences)]
    selected_event = df.loc[best_idx]
    actual_exceedance = exceedances[best_idx]

    print(f"  Dataset: {dataset_id}")
    print(f"  Target exceedance: {target_exceedance:.3f} yr^-1")
    print(f"  Selected event exceedance: {actual_exceedance:.3f} yr^-1")
    print(f"  Event severity: {selected_event['severity']:.2f}")
    print(f"  Event magnitude: {selected_event['magnitude']:.2f}")
    print(f"  Event: {selected_event['start'].strftime('%Y-%m-%d')} to {selected_event['end'].strftime('%Y-%m-%d')}")
    print(f"  Realization: {int(selected_event['realization_id'])}")

    return selected_event


def load_drought_timeseries(realization_id, start_date, end_date):
    """
    Load timeseries data for a drought period from HDF5.
    """
    fname = f'./pywrdrb/outputs/{DATASET_ID}_with_postprocessing.hdf5'

    data = pywrdrb.Data()
    data.load_from_export(fname, results_sets=[
        'res_storage', 'major_flow', 'contribution', 'shortage',
        'ibt_diversions', 'ibt_demands'
    ])

    r = realization_id

    # Calculate plot range with buffer
    plot_start = pd.Timestamp(start_date) - pd.Timedelta(days=BUFFER_DAYS)
    plot_end = pd.Timestamp(end_date) + pd.Timedelta(days=BUFFER_DAYS)

    # NYC aggregate storage (%)
    storage_df = data.res_storage[DATASET_ID][r]
    nyc_storage = storage_df[NYC_RESERVOIRS].sum(axis=1)
    nyc_storage_pct = 100 * nyc_storage / NYC_TOTAL_CAPACITY

    # Montague flow
    montague_flow = data.major_flow[DATASET_ID][r]['delMontague']

    # NYC contribution to Montague
    nyc_contribution = data.contribution[DATASET_ID][r]['mrf_montagueTrenton_nyc']

    # Contribution as % of Montague flow (avoid div by zero)
    contribution_pct = 100 * nyc_contribution / montague_flow.clip(lower=1)

    # Montague shortage (violation indicator)
    montague_shortage = data.shortage[DATASET_ID][r]['delMontague']

    # NYC diversions and demands
    nyc_diversion = data.ibt_diversions[DATASET_ID][r]['delivery_nyc']
    nyc_demand = data.ibt_demands[DATASET_ID][r]['demand_nyc']
    nyc_shortage = (nyc_demand - nyc_diversion).clip(lower=0)

    # Filter to plot range
    def filter_range(series):
        filtered = series.loc[plot_start:plot_end]
        return filtered

    result = {
        'storage_pct': filter_range(nyc_storage_pct),
        'montague_flow': filter_range(montague_flow),
        'contribution': filter_range(nyc_contribution),
        'contribution_pct': filter_range(contribution_pct),
        'montague_shortage': filter_range(montague_shortage),
        'nyc_diversion': filter_range(nyc_diversion),
        'nyc_demand': filter_range(nyc_demand),
        'nyc_shortage': filter_range(nyc_shortage),
        'drought_start': pd.Timestamp(start_date),
        'drought_end': pd.Timestamp(end_date),
    }

    print(f"    Loaded {len(result['storage_pct'])} days of data")

    return result


def calculate_consecutive_violations(shortage_series):
    """
    Calculate rolling count of consecutive days with shortage > 0.
    """
    is_violation = (shortage_series > 0).astype(int)
    groups = (~is_violation.astype(bool)).cumsum()
    consecutive = is_violation.groupby(groups).cumsum()
    return consecutive


def calculate_satisfaction_pct(shortage_series):
    """
    Calculate rolling 30-day satisfaction percentage (0-100%).

    Satisfaction % = (days without shortage / 30) * 100
    """
    # Rolling 30-day window: count non-shortage days
    is_satisfied = (shortage_series == 0).astype(int)
    satisfaction_pct = is_satisfied.rolling(window=3, min_periods=1).mean() * 100
    return satisfaction_pct


def create_yearless_month_axis(dates):
    """
    Convert datetime index to year-agnostic month positions.

    Returns positions (0-based from first month) and month labels.
    """
    if len(dates) == 0:
        return np.array([]), []

    # Get the first date to establish reference
    first_date = dates[0]
    first_month = first_date.month
    first_year = first_date.year

    # Calculate position for each date (months from start)
    positions = []
    for date in dates:
        years_diff = date.year - first_year
        months_diff = date.month - first_month
        position = years_diff * 12 + months_diff
        positions.append(position)

    positions = np.array(positions) / 30.0  # Convert to approximate months (using days)

    # Create month labels
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    return positions, month_names


# ============================================================================
# PLOTTING
# ============================================================================

def plot_three_panel_timeseries(axes, datasets_dict, colors, labels):
    """
    Create 3-panel subplot with year-agnostic time axis aligned by calendar month.

    Reusable function that plots multiple datasets on the same axes.
    Events are aligned by their actual calendar month/day-of-year.

    Parameters
    ----------
    axes : list of 3 matplotlib axes
        [storage_ax, release_ax, satisfaction_ax]
    datasets_dict : dict
        {label: data_dict} where data_dict has keys:
        'storage_pct', 'contribution', 'montague_shortage'
    colors : dict
        {label: color_string}
    labels : list
        Dataset labels in order to plot
    """
    ax_storage, ax_release, ax_satisfaction = axes

    # Month labels for x-axis
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Track x-axis range across all datasets
    all_x_min = float('inf')
    all_x_max = float('-inf')

    def get_day_of_year_positions(dates):
        """Convert dates to day-of-year positions (year-agnostic)."""
        # Use day of year (1-365/366) as x-position
        # For events spanning multiple years, continue counting past 365
        positions = []
        first_date = dates[0]
        first_year = first_date.year

        for date in dates:
            years_passed = date.year - first_year
            doy = date.dayofyear
            # Position = days since start of first year
            position = years_passed * 365 + doy
            positions.append(position)

        return np.array(positions)

    # ========================================================================
    # Panel 1: Storage
    # ========================================================================
    for label in labels:
        data = datasets_dict[label]
        color = colors[label]

        # Convert to year-agnostic x-axis based on day-of-year
        dates = data['storage_pct'].index
        x_positions = get_day_of_year_positions(dates)

        all_x_min = min(all_x_min, x_positions[0])
        all_x_max = max(all_x_max, x_positions[-1])

        # Plot storage
        ax_storage.plot(x_positions, data['storage_pct'].values,
                       color=color, linewidth=1.5, label=label, alpha=0.8)

    # Threshold line
    ax_storage.axhline(STORAGE_THRESHOLD, color=COLOR_THRESHOLD,
                      linestyle='--', linewidth=1, alpha=0.7, label='Threshold')

    ax_storage.set_ylabel('NYC Storage (%)', fontsize=FONTSIZE_LABEL)
    ax_storage.set_ylim(0, 100)
    ax_storage.grid(True, alpha=0.3, linestyle='--')
    ax_storage.set_axisbelow(True)
    ax_storage.legend(fontsize=FONTSIZE_SMALL, loc='best')

    # ========================================================================
    # Panel 2: NYC Releases to Support Montague (linear scale, single y-axis)
    # ========================================================================
    for label in labels:
        data = datasets_dict[label]
        color = colors[label]

        dates = data['contribution'].index
        x_positions = get_day_of_year_positions(dates)

        # Plot NYC contribution/release (non-dashed, linear scale)
        ax_release.plot(x_positions, data['contribution'].values,
                       color=color, linewidth=1.5, label=label, alpha=0.8)

    ax_release.set_ylabel('NYC Release to\nSupport Montague (MGD)', fontsize=FONTSIZE_LABEL)
    ax_release.grid(True, alpha=0.3, linestyle='--')
    ax_release.set_axisbelow(True)

    # ========================================================================
    # Panel 3: Satisfaction % (instead of violation days)
    # ========================================================================
    for label in labels:
        data = datasets_dict[label]
        color = colors[label]

        dates = data['montague_shortage'].index
        x_positions = get_day_of_year_positions(dates)

        # Calculate satisfaction %
        satisfaction_pct = calculate_satisfaction_pct(data['montague_shortage'])

        # Plot satisfaction %
        ax_satisfaction.plot(x_positions, satisfaction_pct.values,
                           color=color, linewidth=1.5, label=label, alpha=0.8)

    ax_satisfaction.set_ylabel('Montague\nSatisfaction (%)', fontsize=FONTSIZE_LABEL)
    ax_satisfaction.set_ylim(0, 120)
    ax_satisfaction.grid(True, alpha=0.3, linestyle='--')
    ax_satisfaction.set_axisbelow(True)

    # ========================================================================
    # Format x-axes: Generic month labels (year-agnostic, aligned by calendar)
    # ========================================================================
    # Only show x-axis labels on bottom panel
    for ax in [ax_storage, ax_release]:
        ax.tick_params(labelbottom=False)

    # Bottom panel gets month labels
    # Create tick positions at month boundaries (roughly every 30 days)
    # Starting from day 1 of the year
    month_starts_doy = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]  # Approximate day-of-year for each month

    # Generate ticks for the range of data
    tick_positions = []
    tick_labels = []
    year_offset = 0
    while True:
        for month_idx, doy in enumerate(month_starts_doy):
            pos = year_offset * 365 + doy
            if all_x_min <= pos <= all_x_max:
                tick_positions.append(pos)
                tick_labels.append(month_names[month_idx])
            elif pos > all_x_max:
                break
        if pos > all_x_max:
            break
        year_offset += 1

    ax_satisfaction.set_xticks(tick_positions)
    ax_satisfaction.set_xticklabels(tick_labels, fontsize=FONTSIZE_SMALL)
    ax_satisfaction.set_xlabel('Month', fontsize=FONTSIZE_LABEL)
    ax_satisfaction.set_xlim(all_x_min - 10, all_x_max + 10)  # Add small padding

    # Apply same x-limits to all panels
    ax_storage.set_xlim(all_x_min - 10, all_x_max + 10)
    ax_release.set_xlim(all_x_min - 10, all_x_max + 10)




# ============================================================================
# MAIN
# ============================================================================

def generate_comparison_figure(target_exceedance=0.1, metric='severity'):
    """
    Generate figure comparing drought events across 3 ensembles.

    Selects one event from each dataset at the specified exceedance rate.

    Parameters
    ----------
    target_exceedance : float
        Target exceedance rate (events per year)
    metric : str
        Metric to use for exceedance selection (default: 'severity')
    """
    # Define datasets and colors
    datasets = ['stationary_ensemble', 'climate_adjusted_low', 'climate_adjusted_high']
    dataset_labels = {
        'stationary_ensemble': 'Stationary',
        'climate_adjusted_low': 'Climate Low',
        'climate_adjusted_high': 'Climate High'
    }
    colors = {
        'stationary_ensemble': '#009E73',  # Green
        'climate_adjusted_low': '#D55E00',  # Orange
        'climate_adjusted_high': '#0072B2'   # Blue
    }

    print(f"\n{'=' * 70}")
    print(f"Selecting events at {target_exceedance} exceedance rate")
    print(f"Metric: {metric}, SSI Window: {SSI_WINDOW}")
    print(f"{'=' * 70}\n")

    # Select events from each dataset
    selected_events = {}
    datasets_dict = {}

    for dataset_id in datasets:
        print(f"\nSelecting event from {dataset_id}...")
        event = select_event_by_exceedance(
            dataset_id,
            target_exceedance=target_exceedance,
            metric=metric,
            ssi_window=SSI_WINDOW
        )
        selected_events[dataset_id] = event

        # Load timeseries data
        print(f"  Loading timeseries data...")
        data = load_drought_timeseries(
            int(event['realization_id']),
            event['start'],
            event['end']
        )
        datasets_dict[dataset_labels[dataset_id]] = data

    # Create figure with 3 panels (vertical stack)
    print("\nCreating figure...")
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    # Plot using the reusable function
    labels = [dataset_labels[d] for d in datasets]
    plot_three_panel_timeseries(
        axes,
        datasets_dict,
        colors={dataset_labels[d]: colors[d] for d in datasets},
        labels=labels
    )

    # Overall title
    fig.suptitle(
        f'Drought Comparison Across Ensembles\n'
        f'Exceedance Rate: {target_exceedance} yr$^{{-1}}$, SSI-{SSI_WINDOW}',
        fontsize=14, fontweight='bold', y=0.99
    )

    plt.tight_layout(rect=[0, 0.01, 1, 0.96])

    # Save
    fname = f"{FIG_OUTPUT_DIR}/F12_drought_comparison_ssi{SSI_WINDOW}_exceedance{target_exceedance:.3f}.png"
    plt.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"\nSaved: {fname}")

    plt.close()

    return fname


def main():
    """Main entry point."""
    print("=" * 70)
    print("F12: Drought Comparison Timeseries")
    print(f"SSI Window: {SSI_WINDOW}")
    print("=" * 70)

    # Generate figure at 0.1 exceedance rate
    fname = generate_comparison_figure(target_exceedance=0.1, metric='magnitude')

    # Summary
    print("\n" + "=" * 70)
    print("FIGURE GENERATED!")
    print("=" * 70)
    print(f"\nSaved: {fname}")
    print("\nDone!")


if __name__ == "__main__":
    main()
