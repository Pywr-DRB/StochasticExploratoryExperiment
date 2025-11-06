"""
Analyze minimum NYC storage during drought events.

This script identifies the minimum NYC combined storage that occurs during each
drought event period and creates diagnostic plots showing the relationship between
storage levels and drought characteristics (severity, magnitude, duration).

The analysis:
1. Loads drought event classifications (SSI 3, 6, 12 months)
2. Loads Pywr-DRB simulation outputs (reservoir storage)
3. For each drought event, identifies minimum NYC storage during that period
4. Creates scatter plots of minimum storage vs drought metrics

Usage:
  python 10_plot_drought_storage_analysis.py <dataset_id> [ssi_window]

Examples:
  # Analyze stationary ensemble with SSI-12
  python 10_plot_drought_storage_analysis.py stationary_ensemble 12

  # Analyze climate-adjusted ensemble with SSI-6
  python 10_plot_drought_storage_analysis.py climate_adjusted_low 6

  # Analyze all SSI windows
  python 10_plot_drought_storage_analysis.py stationary_ensemble

Available datasets: stationary_ensemble, climate_adjusted_low,
                   climate_adjusted_medium, climate_adjusted_high
SSI windows: 3, 6, 12 months
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
from methods.plotting.styles import DATASET_COLORS, DATASET_LABELS_SHORT

# Output directory
FIG_OUTPUT_DIR = f"{FIG_DIR}/drought_storage_analysis"
os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

# Drought metrics directory
DROUGHT_METRICS_DIR = f"{ROOT_DIR}/pywrdrb/drought_metrics"

# NYC reservoir parameters
NYC_RESERVOIRS = ['cannonsville', 'pepacton', 'neversink']
NYC_STORAGE_CAPACITIES = {
    'cannonsville': 95706,  # MG
    'pepacton': 140190,     # MG
    'neversink': 34941      # MG
}
NYC_TOTAL_CAPACITY = sum(NYC_STORAGE_CAPACITIES.values())  # 270,837 MG


def load_drought_events(dataset_id, ssi_window):
    """
    Load drought events from CSV file.

    Parameters:
    -----------
    dataset_id : str
        Dataset identifier
    ssi_window : int
        SSI window size in months (3, 6, or 12)

    Returns:
    --------
    pd.DataFrame
        Drought events with columns: start, end, duration, severity, magnitude, realization_id
    """
    fname = f"{DROUGHT_METRICS_DIR}/{dataset_id}_ssi{ssi_window}_drought_events.csv"

    if not os.path.exists(fname):
        raise FileNotFoundError(
            f"Drought events file not found: {fname}\n"
            f"Run 05_calculate_ssi_drought_metrics.py first!"
        )

    droughts = pd.read_csv(fname)
    droughts['start'] = pd.to_datetime(droughts['start'])
    droughts['end'] = pd.to_datetime(droughts['end'])

    # Take absolute values for severity and magnitude
    for metric in ['severity', 'magnitude']:
        if metric in droughts.columns:
            droughts[metric] = droughts[metric].abs()

    # Remove infinite or NaN values
    droughts = droughts.replace([np.inf, -np.inf], np.nan).dropna(
        subset=['severity', 'magnitude', 'duration']
    )

    # Remove droughts with SSI abs(severity) > 6
    n_before = len(droughts)
    droughts = droughts[droughts['severity'] <= 6.0]
    n_after = len(droughts)
    if n_before > n_after:
        print(f"    Removed {n_before - n_after} droughts with |severity| > 6.0")

    print(f"  Loaded {len(droughts):,} drought events")
    return droughts


def load_reservoir_storage(dataset_id):
    """
    Load reservoir storage data from postprocessed HDF5 file.

    Parameters:
    -----------
    dataset_id : str
        Dataset identifier

    Returns:
    --------
    dict
        Dictionary mapping realization_id to storage DataFrame
    """
    fname = f"{OUTPUT_DIR}/{dataset_id}_with_postprocessing.hdf5"

    if not os.path.exists(fname):
        raise FileNotFoundError(
            f"Postprocessed data file not found: {fname}\n"
            f"Run 04_postprocess_data.py first!"
        )

    # Load only reservoir storage data
    data = pywrdrb.Data()
    data.load_from_export(fname, results_sets=['res_storage'])

    print(f"  Loaded storage data for {len(data.res_storage[dataset_id])} realizations")
    return data.res_storage[dataset_id]


def calculate_nyc_storage_pct(storage_df):
    """
    Calculate NYC combined storage as percentage of total capacity.

    Parameters:
    -----------
    storage_df : pd.DataFrame
        Storage data with columns for NYC reservoirs

    Returns:
    --------
    pd.Series
        NYC combined storage as percentage of capacity
    """
    # Sum storage across NYC reservoirs
    nyc_storage_mg = storage_df[NYC_RESERVOIRS].sum(axis=1)

    # Convert to percentage of total capacity
    nyc_storage_pct = 100.0 * nyc_storage_mg / NYC_TOTAL_CAPACITY

    return nyc_storage_pct


def find_minimum_storage_during_drought(storage_pct, start_date, end_date):
    """
    Find the minimum storage level during a drought period.

    Parameters:
    -----------
    storage_pct : pd.Series
        Time series of NYC storage as percentage
    start_date : pd.Timestamp
        Drought start date
    end_date : pd.Timestamp
        Drought end date

    Returns:
    --------
    float
        Minimum storage percentage during drought period
    """
    # Extract storage during drought period
    drought_storage = storage_pct.loc[start_date:end_date]

    if len(drought_storage) == 0:
        return np.nan

    return drought_storage.min()


def analyze_drought_storage(droughts, storage_data):
    """
    For each drought event, find the minimum NYC storage during that period.

    Parameters:
    -----------
    droughts : pd.DataFrame
        Drought events
    storage_data : dict
        Dictionary mapping realization_id to storage DataFrame

    Returns:
    --------
    pd.DataFrame
        Droughts DataFrame with added 'min_storage_pct' column
    """
    print("\n  Analyzing minimum storage for each drought event...")

    # Add column for minimum storage
    droughts = droughts.copy()
    droughts['min_storage_pct'] = np.nan

    # Group by realization for efficiency
    grouped = droughts.groupby('realization_id')
    n_realizations = len(grouped)

    for i, (real_id, real_droughts) in enumerate(grouped):
        if (i + 1) % 100 == 0:
            print(f"    Processed {i+1}/{n_realizations} realizations...")

        # Get storage data for this realization
        if real_id not in storage_data:
            print(f"    WARNING: Realization {real_id} not found in storage data")
            continue

        storage_df = storage_data[real_id]
        storage_pct = calculate_nyc_storage_pct(storage_df)

        # Find minimum storage for each drought in this realization
        for idx, drought in real_droughts.iterrows():
            min_storage = find_minimum_storage_during_drought(
                storage_pct,
                drought['start'],
                drought['end']
            )
            droughts.loc[idx, 'min_storage_pct'] = min_storage

    # Remove droughts where we couldn't find storage data
    n_before = len(droughts)
    droughts = droughts.dropna(subset=['min_storage_pct'])
    n_after = len(droughts)

    if n_before > n_after:
        print(f"    Removed {n_before - n_after} droughts with missing storage data")

    print(f"  Completed analysis for {len(droughts):,} drought events")
    return droughts


def create_storage_vs_metric_plots(droughts, dataset_id, ssi_window, metrics=['severity', 'magnitude']):
    """
    Create scatter plots of minimum storage vs drought metrics.

    Parameters:
    -----------
    droughts : pd.DataFrame
        Drought events with min_storage_pct column
    dataset_id : str
        Dataset identifier
    ssi_window : int
        SSI window size
    metrics : list
        Metrics to plot against storage

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure with subplots
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(7*n_metrics, 6))

    if n_metrics == 1:
        axes = [axes]

    # Get dataset color
    color = DATASET_COLORS.get(dataset_id, '#1f77b4')
    dataset_label = DATASET_LABELS_SHORT.get(dataset_id, dataset_id)

    metric_labels = {
        'severity': 'Drought Severity (min SSI)',
        'magnitude': 'Drought Magnitude (cumulative deficit)',
        'duration': 'Drought Duration (months)'
    }

    for ax, metric in zip(axes, metrics):
        # Create scatter plot
        ax.scatter(
            droughts[metric],
            droughts['min_storage_pct'],
            alpha=0.4,
            s=20,
            c=color,
            edgecolors='none',
            rasterized=True
        )

        # Calculate correlation
        corr = droughts[metric].corr(droughts['min_storage_pct'])

        # Add trend line
        z = np.polyfit(droughts[metric], droughts['min_storage_pct'], 1)
        p = np.poly1d(z)
        x_line = np.array([droughts[metric].min(), droughts[metric].max()])
        ax.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.8, label=f'r = {corr:.3f}')

        # Labels and formatting
        ax.set_xlabel(metric_labels.get(metric, metric.title()), fontsize=12, fontweight='bold')
        ax.set_ylabel('Minimum NYC Storage (%)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.legend(loc='best', fontsize=10)

        # Add reference lines for storage zones
        ax.axhline(y=100, color='green', linestyle=':', linewidth=1.5, alpha=0.5, label='Full')
        ax.axhline(y=75, color='yellow', linestyle=':', linewidth=1.5, alpha=0.5)
        ax.axhline(y=50, color='orange', linestyle=':', linewidth=1.5, alpha=0.5)
        ax.axhline(y=25, color='red', linestyle=':', linewidth=1.5, alpha=0.5, label='Critical')

        # Set y-axis limits
        ax.set_ylim(-5, 105)

    # Overall title
    title = f"Minimum NYC Storage During Drought Events: {dataset_label}\nSSI-{ssi_window} Droughts"
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.00)

    plt.tight_layout()

    return fig


def create_comprehensive_plot(droughts, dataset_id, ssi_window):
    """
    Create a comprehensive figure with multiple diagnostic plots.

    Parameters:
    -----------
    droughts : pd.DataFrame
        Drought events with min_storage_pct column
    dataset_id : str
        Dataset identifier
    ssi_window : int
        SSI window size

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Comprehensive diagnostic figure
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Get dataset color
    color = DATASET_COLORS.get(dataset_id, '#1f77b4')
    dataset_label = DATASET_LABELS_SHORT.get(dataset_id, dataset_id)

    # Plot 1: Storage vs Severity
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(droughts['severity'], droughts['min_storage_pct'],
                alpha=0.4, s=20, c=color, edgecolors='none', rasterized=True)
    z = np.polyfit(droughts['severity'], droughts['min_storage_pct'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(droughts['severity'].min(), droughts['severity'].max(), 100)
    corr = droughts['severity'].corr(droughts['min_storage_pct'])
    ax1.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Drought Severity (min SSI)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Minimum NYC Storage (%)', fontsize=11, fontweight='bold')
    ax1.set_title(f'Storage vs Severity (r={corr:.3f})', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-5, 105)

    # Plot 2: Storage vs Magnitude
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(droughts['magnitude'], droughts['min_storage_pct'],
                alpha=0.4, s=20, c=color, edgecolors='none', rasterized=True)
    z = np.polyfit(droughts['magnitude'], droughts['min_storage_pct'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(droughts['magnitude'].min(), droughts['magnitude'].max(), 100)
    corr = droughts['magnitude'].corr(droughts['min_storage_pct'])
    ax2.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.8)
    ax2.set_xlabel('Drought Magnitude (cumulative deficit)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Minimum NYC Storage (%)', fontsize=11, fontweight='bold')
    ax2.set_title(f'Storage vs Magnitude (r={corr:.3f})', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-5, 105)

    # Plot 3: Storage vs Duration
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(droughts['duration'], droughts['min_storage_pct'],
                alpha=0.4, s=20, c=color, edgecolors='none', rasterized=True)
    z = np.polyfit(droughts['duration'], droughts['min_storage_pct'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(droughts['duration'].min(), droughts['duration'].max(), 100)
    corr = droughts['duration'].corr(droughts['min_storage_pct'])
    ax3.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.8)
    ax3.set_xlabel('Drought Duration (months)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Minimum NYC Storage (%)', fontsize=11, fontweight='bold')
    ax3.set_title(f'Storage vs Duration (r={corr:.3f})', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-5, 105)

    # Plot 4: Distribution of minimum storage
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(droughts['min_storage_pct'], bins=50, color=color, alpha=0.7, edgecolor='black')
    ax4.axvline(droughts['min_storage_pct'].median(), color='red', linestyle='--',
                linewidth=2, label=f"Median: {droughts['min_storage_pct'].median():.1f}%")
    ax4.set_xlabel('Minimum NYC Storage (%)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax4.set_title('Distribution of Minimum Storage', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # Plot 5: 2D histogram of Severity vs Magnitude colored by storage
    ax5 = fig.add_subplot(gs[1, 1])
    scatter = ax5.scatter(droughts['severity'], droughts['magnitude'],
                         c=droughts['min_storage_pct'], s=30, alpha=0.6,
                         cmap='RdYlGn', vmin=0, vmax=100, edgecolors='none', rasterized=True)
    ax5.set_xlabel('Drought Severity (min SSI)', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Drought Magnitude (cumulative deficit)', fontsize=11, fontweight='bold')
    ax5.set_title('Drought Metrics Colored by Min Storage', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax5)
    cbar.set_label('Min Storage (%)', fontsize=10, fontweight='bold')

    # Plot 6: Summary statistics table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    # Calculate statistics
    stats_data = [
        ['Metric', 'Mean', 'Median', 'Std Dev', 'Min', 'Max'],
        ['Min Storage (%)',
         f"{droughts['min_storage_pct'].mean():.1f}",
         f"{droughts['min_storage_pct'].median():.1f}",
         f"{droughts['min_storage_pct'].std():.1f}",
         f"{droughts['min_storage_pct'].min():.1f}",
         f"{droughts['min_storage_pct'].max():.1f}"],
        ['Severity',
         f"{droughts['severity'].mean():.2f}",
         f"{droughts['severity'].median():.2f}",
         f"{droughts['severity'].std():.2f}",
         f"{droughts['severity'].min():.2f}",
         f"{droughts['severity'].max():.2f}"],
        ['Magnitude',
         f"{droughts['magnitude'].mean():.1f}",
         f"{droughts['magnitude'].median():.1f}",
         f"{droughts['magnitude'].std():.1f}",
         f"{droughts['magnitude'].min():.1f}",
         f"{droughts['magnitude'].max():.1f}"],
        ['Duration (mo)',
         f"{droughts['duration'].mean():.1f}",
         f"{droughts['duration'].median():.1f}",
         f"{droughts['duration'].std():.1f}",
         f"{droughts['duration'].min():.0f}",
         f"{droughts['duration'].max():.0f}"],
    ]

    # Create table
    table = ax6.table(cellText=stats_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header row
    for i in range(6):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(stats_data)):
        for j in range(6):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    ax6.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)

    # Overall title
    title = f"Drought-Storage Analysis: {dataset_label} (SSI-{ssi_window})\n{len(droughts):,} Drought Events"
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    return fig


def main():
    """Main execution function."""

    print("=" * 80)
    print("DROUGHT EVENT MINIMUM STORAGE ANALYSIS")
    print("=" * 80)

    # Parse command line arguments
    if len(sys.argv) < 2:
        print("\nUsage: python 10_plot_drought_storage_analysis.py <dataset_id> [ssi_window]")
        print(f"\nAvailable datasets: {list(DATASET_CONFIGS.keys())}")
        print(f"Available SSI windows: {SSI_WINDOWS}")
        print("\nExamples:")
        print("  # Analyze stationary ensemble with SSI-12")
        print("  python 10_plot_drought_storage_analysis.py stationary_ensemble 12")
        print("\n  # Analyze all SSI windows")
        print("  python 10_plot_drought_storage_analysis.py stationary_ensemble")
        sys.exit(1)

    dataset_id = sys.argv[1]
    verify_dataset_id(dataset_id)

    # Determine SSI windows to process
    if len(sys.argv) > 2:
        ssi_windows = [int(sys.argv[2])]
        if ssi_windows[0] not in SSI_WINDOWS:
            print(f"ERROR: Invalid SSI window: {ssi_windows[0]}")
            print(f"Must be one of: {SSI_WINDOWS}")
            sys.exit(1)
    else:
        ssi_windows = SSI_WINDOWS

    print(f"\nConfiguration:")
    print(f"  Dataset: {dataset_id}")
    print(f"  SSI Windows: {ssi_windows}")
    print()

    # Load reservoir storage data once
    print("Loading reservoir storage data...")
    storage_data = load_reservoir_storage(dataset_id)

    # Process each SSI window
    for ssi_window in ssi_windows:
        print(f"\n{'='*80}")
        print(f"PROCESSING SSI-{ssi_window}")
        print(f"{'='*80}")

        # Load drought events
        print(f"\nLoading drought events (SSI-{ssi_window})...")
        droughts = load_drought_events(dataset_id, ssi_window)

        # Analyze minimum storage during each drought
        droughts_with_storage = analyze_drought_storage(droughts, storage_data)

        # Create comprehensive diagnostic plot
        print(f"\nCreating comprehensive diagnostic plot...")
        fig = create_comprehensive_plot(droughts_with_storage, dataset_id, ssi_window)

        fname = f"{FIG_OUTPUT_DIR}/{dataset_id}_ssi{ssi_window}_drought_storage_comprehensive.png"
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        print(f"  Saved: {fname}")
        plt.close(fig)

        # Create simple scatter plots
        print(f"\nCreating scatter plots...")
        fig = create_storage_vs_metric_plots(
            droughts_with_storage,
            dataset_id,
            ssi_window,
            metrics=['severity', 'magnitude']
        )

        fname = f"{FIG_OUTPUT_DIR}/{dataset_id}_ssi{ssi_window}_drought_storage_scatter.png"
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        print(f"  Saved: {fname}")
        plt.close(fig)

        # Print summary statistics
        print(f"\nSummary Statistics (SSI-{ssi_window}):")
        print(f"  Total drought events: {len(droughts_with_storage):,}")
        print(f"  Min storage range: {droughts_with_storage['min_storage_pct'].min():.1f}% - {droughts_with_storage['min_storage_pct'].max():.1f}%")
        print(f"  Mean min storage: {droughts_with_storage['min_storage_pct'].mean():.1f}%")
        print(f"  Median min storage: {droughts_with_storage['min_storage_pct'].median():.1f}%")
        print(f"  Events with storage < 25%: {(droughts_with_storage['min_storage_pct'] < 25).sum():,} ({100*(droughts_with_storage['min_storage_pct'] < 25).sum()/len(droughts_with_storage):.1f}%)")
        print(f"  Correlation with severity: {droughts_with_storage['severity'].corr(droughts_with_storage['min_storage_pct']):.3f}")
        print(f"  Correlation with magnitude: {droughts_with_storage['magnitude'].corr(droughts_with_storage['min_storage_pct']):.3f}")
        print(f"  Correlation with duration: {droughts_with_storage['duration'].corr(droughts_with_storage['min_storage_pct']):.3f}")

    print("\n" + "=" * 80)
    print("COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nFigures saved to: {FIG_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
