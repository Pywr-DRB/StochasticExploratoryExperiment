"""
SI1: Plot Shortage Occurrence by Day of Year

This script creates histograms showing the number of days where shortages occur
at each location, aggregated by day of year (1-366).

Shortages are calculated as:
- Montague: Flow < target at Montague
- Trenton: Flow < target at Trenton
- NYC: Diversion delivery < demand

The script produces three figures, one for each location, showing the seasonal
pattern of shortage occurrence across all realizations.

Usage:
    python SI1_plot_shortage_occurrence_by_day.py <dataset_id>

Example:
    python SI1_plot_shortage_occurrence_by_day.py stationary_ensemble
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import pywrdrb
from methods.config import *
from methods.load import load_shortage_data
from methods.metrics.shortfall import calculate_shortage_by_day_of_year

# Output directory
FIG_DIR_SHORTAGE = f"{FIG_DIR}/SI1_shortage_occurrence"
os.makedirs(FIG_DIR_SHORTAGE, exist_ok=True)


def plot_shortage_occurrence(shortage_counts, location, dataset_id, dataset_label):
    """
    Create histogram of shortage occurrence by day of year.

    Parameters
    ----------
    shortage_counts : np.ndarray
        Array of shortage counts by day of year
    location : str
        Location identifier
    dataset_id : str
        Dataset identifier
    dataset_label : str
        Dataset label for plot title
    """
    # Location-specific formatting
    location_names = {
        'delMontague': 'Montague',
        'delTrenton': 'Trenton',
        'nyc': 'NYC Diversions'
    }

    location_name = location_names.get(location, location)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))

    # Day of year array
    days = np.arange(1, 367)

    # Plot histogram
    ax.bar(days, shortage_counts, width=1.0, color='#d62728', alpha=0.7,
           edgecolor='none', label='Shortage Days')

    # Add monthly grid lines and labels
    month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    for month_start in month_starts:
        ax.axvline(month_start, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)

    # Set month labels at midpoints
    month_midpoints = []
    for i in range(len(month_starts)):
        if i < len(month_starts) - 1:
            midpoint = (month_starts[i] + month_starts[i + 1]) / 2
        else:
            midpoint = (month_starts[i] + 366) / 2
        month_midpoints.append(midpoint)

    ax.set_xticks(month_midpoints)
    ax.set_xticklabels(month_labels, fontsize=11)

    # Labels and title
    ax.set_xlabel('Month', fontsize=13, fontweight='bold')
    ax.set_ylabel('Number of Shortage Days', fontsize=13, fontweight='bold')
    ax.set_title(f'Shortage Occurrence by Day of Year: {location_name}\n{dataset_label}',
                fontsize=15, fontweight='bold', pad=15)

    # Set x-axis limits
    ax.set_xlim(1, 366)

    # Grid
    ax.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # Summary statistics text box
    total_shortage_days = shortage_counts.sum()
    max_shortage_days = shortage_counts.max()
    max_shortage_doy = np.argmax(shortage_counts) + 1

    # Get month for max shortage day
    for i, month_start in enumerate(month_starts):
        if i < len(month_starts) - 1:
            if month_start <= max_shortage_doy < month_starts[i + 1]:
                max_month = month_labels[i]
                break
        else:
            max_month = month_labels[i]

    stats_text = (
        f'Total Shortage Days: {total_shortage_days:,}\n'
        f'Max for Single Day: {max_shortage_days:,}\n'
        f'Peak Day: DOY {max_shortage_doy} ({max_month})'
    )

    ax.text(0.98, 0.97, stats_text,
           transform=ax.transAxes,
           fontsize=10,
           verticalalignment='top',
           horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, pad=0.5))

    # Tight layout
    plt.tight_layout()

    # Save figure
    fname = f"{FIG_DIR_SHORTAGE}/{dataset_id}_shortage_doy_{location}.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"  Saved: {fname}")


    plt.close()


def plot_comparison_all_locations(shortage_data, dataset_id, dataset_label):
    """
    Create a 3-panel comparison figure showing all locations.

    Parameters
    ----------
    shortage_data : dict
        Dictionary mapping location to shortage_counts array
    dataset_id : str
        Dataset identifier
    dataset_label : str
        Dataset label for plot title
    """
    location_names = {
        'delMontague': 'Montague',
        'delTrenton': 'Trenton',
        'nyc': 'NYC Diversions'
    }

    locations = ['delMontague', 'delTrenton', 'nyc']
    colors = ['#d62728', '#ff7f0e', '#2ca02c']

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle(f'Shortage Occurrence by Day of Year: {dataset_label}',
                fontsize=16, fontweight='bold', y=0.995)

    # Month gridlines and labels
    month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    month_midpoints = []
    for i in range(len(month_starts)):
        if i < len(month_starts) - 1:
            midpoint = (month_starts[i] + month_starts[i + 1]) / 2
        else:
            midpoint = (month_starts[i] + 366) / 2
        month_midpoints.append(midpoint)

    days = np.arange(1, 367)

    for idx, (location, color) in enumerate(zip(locations, colors)):
        ax = axes[idx]
        shortage_counts = shortage_data[location]

        # Plot histogram
        ax.bar(days, shortage_counts, width=1.0, color=color, alpha=0.7,
              edgecolor='none')

        # Month gridlines
        for month_start in month_starts:
            ax.axvline(month_start, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)

        # Labels
        location_name = location_names[location]
        ax.set_ylabel('Shortage Days', fontsize=11, fontweight='bold')
        ax.set_xlim(1, 366)
        ax.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)

        # Panel label
        panel_label = chr(97 + idx)  # a, b, c
        ax.text(0.01, 0.97, f'({panel_label}) {location_name}',
               transform=ax.transAxes,
               fontsize=12, fontweight='bold',
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, pad=0.4))

        # Only show x-axis labels on bottom panel
        if idx == 2:
            ax.set_xlabel('Month', fontsize=12, fontweight='bold')
            ax.set_xticks(month_midpoints)
            ax.set_xticklabels(month_labels, fontsize=10)
        else:
            ax.set_xticks(month_midpoints)
            ax.set_xticklabels([])

    plt.tight_layout()

    # Save figure
    fname = f"{FIG_DIR_SHORTAGE}/{dataset_id}_shortage_doy_comparison.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"\n  Saved comparison: {fname}")
    

    plt.close()


def main(dataset_id):
    """
    Main function to generate shortage occurrence plots.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    """
    print("=" * 80)
    print(f"SHORTAGE OCCURRENCE BY DAY OF YEAR: {dataset_id}")
    print("=" * 80)

    # Verify dataset
    verify_dataset_id(dataset_id)
    dataset_config = DATASET_CONFIGS[dataset_id]
    dataset_label = f"{dataset_config['description']} ({dataset_config['type']})"

    # Load shortage data
    data = load_shortage_data(dataset_id)

    # Process each location
    locations = ['delMontague', 'delTrenton', 'nyc']
    shortage_data = {}

    print("\nCalculating shortage occurrence by day of year:")
    print("-" * 80)

    for location in locations:
        shortage_counts = calculate_shortage_by_day_of_year(data, dataset_id, location)
        shortage_data[location] = shortage_counts


    # Create comparison plot
    print("\nGenerating comparison plot:")
    print("-" * 80)
    plot_comparison_all_locations(shortage_data, dataset_id, dataset_label)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nFigures saved to: {FIG_DIR_SHORTAGE}/")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        print(f"\nAvailable datasets: {list(DATASET_CONFIGS.keys())}")
        sys.exit(1)

    dataset_id = sys.argv[1]
    verify_dataset_id(dataset_id)

    main(dataset_id)