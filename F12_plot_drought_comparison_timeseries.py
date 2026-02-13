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
from methods.config import FIG_DIR, NYC_RESERVOIRS, NYC_TOTAL_CAPACITY
from methods.plotting.styles import (
    DPI_HIGH, FONTSIZE_SMALL, FONTSIZE_LABEL,
    DATASET_COLORS, DATASET_LABELS,
)
from methods.load import load_drought_events, compute_event_exceedances

# ============================================================================
# CONFIGURATION
# ============================================================================

FIG_OUTPUT_DIR = f"{FIG_DIR}/F12_drought_comparison"
os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

SSI_WINDOW = 3  # Editable: 3, 6, or 12
DATASET_ID = 'stationary_ensemble'  # Used for loading timeseries data

# Buffer days before/after drought period
BUFFER_DAYS = 90

# Thresholds (for visualization)
STORAGE_THRESHOLD = 20.0  # %
COLOR_THRESHOLD = 'black'


# ============================================================================
# EVENT SELECTION
# ============================================================================

def select_event_by_exceedance(dataset_id, target_exceedance=0.1, metric='severity',
                                ssi_window=SSI_WINDOW, target_month=None, month_tolerance=1):
    """
    Select a drought event from a dataset based on target exceedance rate.

    Finds the event whose exceedance rate is closest to the target.
    If target_month is specified, only considers events starting within
    +/- month_tolerance months of that target.

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
    target_month : int, optional
        Target start month (1-12). If specified, only events starting within
        +/- month_tolerance months are considered.
    month_tolerance : int, optional
        Allowable deviation from target_month (default: 1)

    Returns
    -------
    selected_event : pd.Series
        The drought event closest to the target exceedance
    """
    # Load drought events using centralized function
    df = load_drought_events(dataset_id, ssi_window, observed=False, filter_extreme=False)

    # Compute exceedances for all events
    exceedances = compute_event_exceedances(df, metric=metric)

    # Add exceedances to dataframe for filtering
    df['exceedance'] = exceedances

    # Filter by month if specified
    if target_month is not None:
        df['start_month'] = df['start'].dt.month

        # Handle wraparound (e.g., target=1, tolerance=1 should include 12, 1, 2)
        valid_months = []
        for offset in range(-month_tolerance, month_tolerance + 1):
            month = ((target_month - 1 + offset) % 12) + 1
            valid_months.append(month)

        df_filtered = df[df['start_month'].isin(valid_months)].copy()

        if len(df_filtered) == 0:
            print(f"  WARNING: No events found starting in month {target_month} +/- {month_tolerance}")
            print(f"  Falling back to all events")
            df_filtered = df
    else:
        df_filtered = df

    # Find event closest to target exceedance
    differences = np.abs(df_filtered['exceedance'] - target_exceedance)
    best_idx = df_filtered.index[np.argmin(differences)]
    selected_event = df_filtered.loc[best_idx]
    actual_exceedance = selected_event['exceedance']

    print(f"  Dataset: {dataset_id}")
    print(f"  Target exceedance: {target_exceedance:.3f} yr^-1")
    print(f"  Selected event exceedance: {actual_exceedance:.3f} yr^-1")
    print(f"  Event severity: {selected_event['severity']:.2f}")
    print(f"  Event magnitude: {selected_event['magnitude']:.2f}")
    print(f"  Event: {selected_event['start'].strftime('%Y-%m-%d')} to {selected_event['end'].strftime('%Y-%m-%d')}")
    print(f"  Start month: {selected_event['start'].month}")
    print(f"  Realization: {int(selected_event['realization_id'])}")

    return selected_event


def load_drought_timeseries(realization_id, start_date, end_date):
    """
    Load timeseries data for a drought period from HDF5.

    Parameters
    ----------
    realization_id : int
        Realization ID to load
    start_date : pd.Timestamp or str
        Start date of drought event
    end_date : pd.Timestamp or str
        End date of drought event

    Returns
    -------
    dict
        Dictionary with timeseries data for the drought period
    """
    fname = f'./pywrdrb/outputs/{DATASET_ID}_with_postprocessing.hdf5'

    # Only load the specific realization needed (much faster!)
    data = pywrdrb.Data()
    data.load_from_export(
        fname,
        results_sets=[
            'res_storage', 'major_flow', 'contribution', 'shortage',
            'ibt_diversions', 'ibt_demands'
        ],
        realizations=[realization_id]
    )

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

def generate_comparison_figure(target_exceedance=0.1, metric='severity', month_tolerance=1):
    """
    Generate figure comparing drought events across 3 ensembles.

    Selects one event from each dataset at the specified exceedance rate.
    Events are selected to start in similar months (+/- month_tolerance).

    Parameters
    ----------
    target_exceedance : float
        Target exceedance rate (events per year)
    metric : str
        Metric to use for exceedance selection (default: 'severity')
    month_tolerance : int, optional
        Allowable deviation in start month between events (default: 1)
    """
    # Define datasets - use only the three available datasets
    datasets = ['stationary_ensemble', 'climate_adjusted_low', 'climate_adjusted_high']

    print(f"\n{'=' * 70}")
    print(f"Selecting events at {target_exceedance} exceedance rate")
    print(f"Metric: {metric}, SSI Window: {SSI_WINDOW}")
    print(f"Month tolerance: +/- {month_tolerance} months")
    print(f"{'=' * 70}\n")

    # Step 1: Select reference event from stationary ensemble (no month constraint)
    print(f"\nSelecting reference event from stationary_ensemble...")
    reference_event = select_event_by_exceedance(
        'stationary_ensemble',
        target_exceedance=target_exceedance,
        metric=metric,
        ssi_window=SSI_WINDOW
    )
    reference_month = reference_event['start'].month

    print(f"\n  -> Reference month: {reference_month} ({reference_event['start'].strftime('%B')})")

    # Step 2: Select events from other datasets matching the reference month
    selected_events = {'stationary_ensemble': reference_event}
    datasets_dict = {}

    for dataset_id in datasets:
        if dataset_id == 'stationary_ensemble':
            event = reference_event
        else:
            print(f"\nSelecting event from {dataset_id} (targeting month {reference_month})...")
            event = select_event_by_exceedance(
                dataset_id,
                target_exceedance=target_exceedance,
                metric=metric,
                ssi_window=SSI_WINDOW,
                target_month=reference_month,
                month_tolerance=month_tolerance
            )
            selected_events[dataset_id] = event

        # Load timeseries data
        print(f"  Loading timeseries data...")
        data = load_drought_timeseries(
            int(event['realization_id']),
            event['start'],
            event['end']
        )
        # Use standardized label from styles.py
        label = DATASET_LABELS[dataset_id]
        datasets_dict[label] = data

    # Create figure with 3 panels (vertical stack)
    print("\nCreating figure...")
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    # Plot using the reusable function
    labels = [DATASET_LABELS[d] for d in datasets]
    colors_dict = {DATASET_LABELS[d]: DATASET_COLORS[d] for d in datasets}

    plot_three_panel_timeseries(
        axes,
        datasets_dict,
        colors=colors_dict,
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
