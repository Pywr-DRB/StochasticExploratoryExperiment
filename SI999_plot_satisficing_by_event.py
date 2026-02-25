"""
SI2: Plot Satisficing Outcomes by Event Type

This script visualizes the comparison of satisficing outcomes across different
conditions:
- All Years (Jun-Dec baseline)
- SSI Drought Periods
- Non-Drought Years

The script loads results from 06_calculate_satisficing_by_drought.py and creates
comparison visualizations showing how performance differs during drought vs
non-drought conditions.

Usage:
    python SI2_plot_satisficing_by_event.py <dataset_id> <ssi_window>

Example:
    python SI2_plot_satisficing_by_event.py stationary_ensemble 6
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
from methods.load import load_satisficing_results

# Output directory
FIG_DIR_SATISFICING = f"{FIG_DIR}/SI2_satisficing_by_event"
os.makedirs(FIG_DIR_SATISFICING, exist_ok=True)


def plot_satisficing_percentages(results, dataset_id, ssi_window, dataset_label):
    """
    Create bar chart comparing satisficing percentages across conditions.

    Parameters
    ----------
    results : dict
        Dictionary of results DataFrames
    dataset_id : str
        Dataset identifier
    ssi_window : int
        SSI window
    dataset_label : str
        Dataset label for title
    """
    # Calculate percentages
    conditions = ['all_years', 'drought', 'non_drought']
    condition_labels = ['All Years\n(Jun-Dec)', 'SSI Drought\nPeriods', 'Non-Drought\nYears']

    percentages = []
    counts_sat = []
    counts_total = []

    for condition in conditions:
        df = results[condition]
        n_total = len(df)
        n_sat = df['satisficing'].sum()
        pct = 100 * n_sat / n_total if n_total > 0 else 0

        percentages.append(pct)
        counts_sat.append(n_sat)
        counts_total.append(n_total)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    # Define colors
    colors = ['#1f77b4', '#d62728', '#2ca02c']

    # Create bars
    x = np.arange(len(conditions))
    bars = ax.bar(x, percentages, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for i, (bar, pct, n_sat, n_total) in enumerate(zip(bars, percentages, counts_sat, counts_total)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
               f'{pct:.1f}%',
               ha='center', va='bottom', fontsize=13, fontweight='bold')

        # Add count labels below bars
        ax.text(bar.get_x() + bar.get_width()/2., -3,
               f'{n_sat:,}/{n_total:,}',
               ha='center', va='top', fontsize=9, style='italic')

    # Labels and formatting
    ax.set_ylabel('Satisficing Rate (%)', fontsize=13, fontweight='bold')
    ax.set_title(f'Satisficing Outcomes by Condition\n{dataset_label}, SSI-{ssi_window}',
                fontsize=15, fontweight='bold', pad=20)

    ax.set_xticks(x)
    ax.set_xticklabels(condition_labels, fontsize=12)
    ax.set_ylim(0, max(percentages) * 1.2)

    # Add horizontal line at 100%
    ax.axhline(100, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # Grid
    ax.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # Add criteria text
    criteria_text = (
        'Satisficing Criteria:\n'
        '• NYC storage ≥ 20% throughout period\n'
        '• Montague violations ≤ 3 consecutive days'
    )
    ax.text(0.02, 0.98, criteria_text,
           transform=ax.transAxes,
           fontsize=9,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, pad=0.5))

    plt.tight_layout()

    # Save figure
    fname = f"{FIG_DIR_SATISFICING}/{dataset_id}_ssi{ssi_window}_satisficing_comparison.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"  Saved: {fname}")


    plt.close()


def plot_failure_breakdown(results, dataset_id, ssi_window, dataset_label):
    """
    Create stacked bar chart showing failure types by condition.

    Parameters
    ----------
    results : dict
        Dictionary of results DataFrames
    dataset_id : str
        Dataset identifier
    ssi_window : int
        SSI window
    dataset_label : str
        Dataset label for title
    """
    conditions = ['all_years', 'drought', 'non_drought']
    condition_labels = ['All Years', 'Drought', 'Non-Drought']

    # Calculate failure breakdown
    failure_data = {
        'Satisficing': [],
        'Storage Only': [],
        'Violation Only': [],
        'Both Failures': []
    }

    for condition in conditions:
        df = results[condition]
        n_total = len(df)

        # Calculate failure categories
        storage_fail = df['min_storage_pct'] < 20
        montague_fail = df['max_violation_days'] > 3

        n_satisficing = df['satisficing'].sum()
        n_storage_only = ((storage_fail & ~montague_fail).sum())
        n_violation_only = ((montague_fail & ~storage_fail).sum())
        n_both = ((storage_fail & montague_fail).sum())

        # Convert to percentages
        failure_data['Satisficing'].append(100 * n_satisficing / n_total)
        failure_data['Storage Only'].append(100 * n_storage_only / n_total)
        failure_data['Violation Only'].append(100 * n_violation_only / n_total)
        failure_data['Both Failures'].append(100 * n_both / n_total)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    # Colors for each category
    colors = {
        'Satisficing': '#2ca02c',
        'Storage Only': '#ff7f0e',
        'Violation Only': '#d62728',
        'Both Failures': '#9467bd'
    }

    # Create stacked bars
    x = np.arange(len(conditions))
    width = 0.6

    bottom = np.zeros(len(conditions))

    for category in ['Satisficing', 'Storage Only', 'Violation Only', 'Both Failures']:
        values = failure_data[category]
        bars = ax.bar(x, values, width, label=category, color=colors[category],
                     alpha=0.8, edgecolor='black', linewidth=1, bottom=bottom)

        # Add percentage labels for segments > 5%
        for i, (val, b) in enumerate(zip(values, bottom)):
            if val > 5:  # Only label if segment is large enough
                ax.text(i, b + val/2, f'{val:.1f}%',
                       ha='center', va='center', fontsize=10, fontweight='bold', color='white')

        bottom += values

    # Labels and formatting
    ax.set_ylabel('Percentage (%)', fontsize=13, fontweight='bold')
    ax.set_title(f'Performance Outcome Breakdown by Condition\n{dataset_label}, SSI-{ssi_window}',
                fontsize=15, fontweight='bold', pad=20)

    ax.set_xticks(x)
    ax.set_xticklabels(condition_labels, fontsize=12)
    ax.set_ylim(0, 100)

    # Legend
    ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=10,
             framealpha=0.9, edgecolor='black')

    # Grid
    ax.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()

    # Save figure
    fname = f"{FIG_DIR_SATISFICING}/{dataset_id}_ssi{ssi_window}_failure_breakdown.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"  Saved: {fname}")


    plt.close()


def plot_metric_distributions(results, dataset_id, ssi_window, dataset_label):
    """
    Create violin plots comparing key metrics across conditions.

    Parameters
    ----------
    results : dict
        Dictionary of results DataFrames
    dataset_id : str
        Dataset identifier
    ssi_window : int
        SSI window
    dataset_label : str
        Dataset label for title
    """
    conditions = ['all_years', 'drought', 'non_drought']
    condition_labels = ['All Years', 'Drought', 'Non-Drought']

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Prepare data for plotting
    storage_data = []
    violation_data = []
    labels = []

    for condition, label in zip(conditions, condition_labels):
        df = results[condition]
        storage_data.extend(df['min_storage_pct'].values)
        violation_data.extend(df['max_violation_days'].values)
        labels.extend([label] * len(df))

    # Create DataFrame for seaborn
    plot_df = pd.DataFrame({
        'Condition': labels,
        'Min Storage (%)': storage_data,
        'Max Violation Days': violation_data
    })

    # Define colors
    palette = {'All Years': '#1f77b4', 'Drought': '#d62728', 'Non-Drought': '#2ca02c'}

    # Plot 1: Storage distributions
    ax = axes[0]
    sns.violinplot(data=plot_df, x='Condition', y='Min Storage (%)',
                  palette=palette, ax=ax, inner='quartile', linewidth=1.5)

    # Add threshold line
    ax.axhline(20, color='red', linestyle='--', linewidth=2, alpha=0.7,
              label='Satisficing Threshold (20%)')

    ax.set_ylabel('Minimum Storage (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('')
    ax.set_title('(a) Minimum NYC Storage', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)

    # Plot 2: Violation distributions
    ax = axes[1]
    sns.violinplot(data=plot_df, x='Condition', y='Max Violation Days',
                  palette=palette, ax=ax, inner='quartile', linewidth=1.5)

    # Add threshold line
    ax.axhline(3, color='red', linestyle='--', linewidth=2, alpha=0.7,
              label='Satisficing Threshold (3 days)')

    ax.set_ylabel('Maximum Consecutive Violation Days', fontsize=12, fontweight='bold')
    ax.set_xlabel('')
    ax.set_title('(b) Montague Flow Violations', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)

    # Overall title
    fig.suptitle(f'Performance Metric Distributions by Condition\n{dataset_label}, SSI-{ssi_window}',
                fontsize=15, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Save figure
    fname = f"{FIG_DIR_SATISFICING}/{dataset_id}_ssi{ssi_window}_metric_distributions.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"  Saved: {fname}")

    plt.close()


def plot_combined_summary(results, dataset_id, ssi_window, dataset_label):
    """
    Create a comprehensive 3-panel summary figure.

    Parameters
    ----------
    results : dict
        Dictionary of results DataFrames
    dataset_id : str
        Dataset identifier
    ssi_window : int
        SSI window
    dataset_label : str
        Dataset label for title
    """
    conditions = ['all_years', 'drought', 'non_drought']
    condition_labels = ['All Years', 'Drought', 'Non-Drought']

    fig = plt.figure(figsize=(16, 5))
    gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3)

    # Panel 1: Satisficing percentages
    ax1 = fig.add_subplot(gs[0, 0])

    percentages = []
    for condition in conditions:
        df = results[condition]
        pct = 100 * df['satisficing'].sum() / len(df)
        percentages.append(pct)

    colors = ['#1f77b4', '#d62728', '#2ca02c']
    x = np.arange(len(conditions))
    bars = ax1.bar(x, percentages, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax1.set_ylabel('Satisficing Rate (%)', fontsize=11, fontweight='bold')
    ax1.set_title('(a) Satisficing Outcomes', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(condition_labels, fontsize=10)
    ax1.set_ylim(0, max(percentages) * 1.15)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_axisbelow(True)

    # Panel 2: Mean storage levels
    ax2 = fig.add_subplot(gs[0, 1])

    mean_storage = []
    for condition in conditions:
        df = results[condition]
        mean_storage.append(df['min_storage_pct'].mean())

    bars = ax2.bar(x, mean_storage, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    for bar, val in zip(bars, mean_storage):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax2.axhline(20, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Threshold (20%)')
    ax2.set_ylabel('Mean Min Storage (%)', fontsize=11, fontweight='bold')
    ax2.set_title('(b) NYC Storage Levels', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(condition_labels, fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_axisbelow(True)

    # Panel 3: Mean violation days
    ax3 = fig.add_subplot(gs[0, 2])

    mean_violations = []
    for condition in conditions:
        df = results[condition]
        mean_violations.append(df['max_violation_days'].mean())

    bars = ax3.bar(x, mean_violations, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    for bar, val in zip(bars, mean_violations):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax3.axhline(3, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Threshold (3 days)')
    ax3.set_ylabel('Mean Max Violations (days)', fontsize=11, fontweight='bold')
    ax3.set_title('(c) Montague Violations', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(condition_labels, fontsize=10)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_axisbelow(True)

    # Overall title
    fig.suptitle(f'Comprehensive Performance Comparison: {dataset_label}, SSI-{ssi_window}',
                fontsize=15, fontweight='bold')

    plt.tight_layout()

    # Save figure
    fname = f"{FIG_DIR_SATISFICING}/{dataset_id}_ssi{ssi_window}_summary_3panel.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"  Saved: {fname}")

    plt.close()


def main(dataset_id, ssi_window):
    """
    Main function to generate satisficing comparison plots.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    ssi_window : int
        SSI window (3, 6, or 12 months)
    """
    print("=" * 80)
    print(f"SATISFICING COMPARISON PLOTS: {dataset_id}, SSI-{ssi_window}")
    print("=" * 80)

    # Verify dataset
    verify_dataset_id(dataset_id)
    dataset_config = DATASET_CONFIGS[dataset_id]
    dataset_label = f"{dataset_config['description']}"

    # Load results
    print("\nLoading satisficing results:")
    print("-" * 80)
    results = load_satisficing_results(dataset_id, ssi_window)

    # Print summary
    print("\nSummary:")
    print("-" * 80)
    for condition in ['all_years', 'drought', 'non_drought']:
        df = results[condition]
        n_total = len(df)
        n_sat = df['satisficing'].sum()
        pct = 100 * n_sat / n_total
        print(f"  {condition:15s}: {n_sat:>6,}/{n_total:>6,} ({pct:>5.1f}%) satisficing")

    # Generate plots
    print("\nGenerating plots:")
    print("-" * 80)

    print("  1. Satisficing percentage comparison...")
    plot_satisficing_percentages(results, dataset_id, ssi_window, dataset_label)

    print("  2. Failure breakdown by type...")
    plot_failure_breakdown(results, dataset_id, ssi_window, dataset_label)

    print("  3. Metric distributions (violin plots)...")
    plot_metric_distributions(results, dataset_id, ssi_window, dataset_label)

    print("  4. Combined 3-panel summary...")
    plot_combined_summary(results, dataset_id, ssi_window, dataset_label)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nFigures saved to: {FIG_DIR_SATISFICING}/")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        print(f"\nAvailable datasets: {list(DATASET_CONFIGS.keys())}")
        print(f"Available SSI windows: {SSI_WINDOWS}")
        sys.exit(1)

    dataset_id = sys.argv[1]
    ssi_window = int(sys.argv[2])

    # Validate inputs
    verify_dataset_id(dataset_id)
    if ssi_window not in SSI_WINDOWS:
        print(f"ERROR: Invalid SSI window. Must be one of {SSI_WINDOWS}")
        sys.exit(1)

    main(dataset_id, ssi_window)
