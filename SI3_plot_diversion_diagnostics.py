"""
SI3: Plot Diversion Diagnostics

This script creates publication-quality diagnostic plots comparing observed and
ensemble-extrapolated diversion data for NYC and NJ water supplies.

The plots include:
- Distribution comparisons (KDE plots and box plots)
- Seasonal patterns (monthly box plots)
- Time series overview

These visualizations support the academic publication's supplementary information
by demonstrating the quality of the diversion extrapolation methodology.

Usage:
    python SI3_plot_diversion_diagnostics.py <dataset_id>

Example:
    python SI3_plot_diversion_diagnostics.py stationary_ensemble
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
from methods.load import load_observed_diversions, load_ensemble_diversions

# Matplotlib settings for publication-quality figures
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# Output directory
FIG_DIR_DIVERSIONS = f"{FIG_DIR}/diversion_diagnostics"
os.makedirs(FIG_DIR_DIVERSIONS, exist_ok=True)


def plot_distribution_comparison(observed, ensemble_dict, loc, dataset_id):
    """
    Create KDE and box plot comparing observed vs ensemble distributions.

    Parameters
    ----------
    observed : pd.Series
        Observed diversion data
    ensemble_dict : dict
        Dictionary of ensemble realizations
    loc : str
        Location identifier
    dataset_id : str
        Dataset identifier
    """
    location_names = {'nyc': 'NYC', 'nj': 'NJ'}
    location_name = location_names.get(loc, loc)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ### Left panel: KDE plot
    # Plot observed
    observed_clean = observed.dropna()
    if len(observed_clean) > 0:
        observed_clean.plot.kde(ax=ax1, color='#1f77b4', linewidth=2.5,
                                label='Observed', zorder=3)

    # Plot ensemble members (sample subset for clarity)
    n_to_plot = min(50, len(ensemble_dict))
    sample_keys = np.random.choice(list(ensemble_dict.keys()), n_to_plot, replace=False)

    for i, key in enumerate(sample_keys):
        ens_clean = ensemble_dict[key].dropna()
        if len(ens_clean) > 0:
            ens_clean.plot.kde(ax=ax1, color='#ff7f0e', alpha=0.15,
                              linewidth=0.8, label='Ensemble' if i == 0 else None)

    ax1.set_xlabel(f'{location_name} Diversion (MGD)')
    ax1.set_ylabel('Probability Density')
    ax1.set_title(f'{location_name} Diversion Distribution')
    ax1.legend(frameon=False)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(left=0)

    ### Right panel: Box plot
    # Prepare data for box plot
    box_data = []
    box_labels = []

    # Add observed
    box_data.append(observed_clean.values)
    box_labels.append('Observed')

    # Add ensemble statistics (all realizations combined)
    all_ensemble = pd.concat(ensemble_dict.values())
    box_data.append(all_ensemble.dropna().values)
    box_labels.append('Ensemble')

    bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True,
                     widths=0.6, showfliers=False)

    # Color the boxes
    colors = ['#1f77b4', '#ff7f0e']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax2.set_ylabel(f'{location_name} Diversion (MGD)')
    ax2.set_title(f'{location_name} Diversion Summary')
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')

    plt.tight_layout()
    fname = f"{FIG_DIR_DIVERSIONS}/{dataset_id}_{loc}_distribution.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


def plot_seasonal_patterns(observed, ensemble_dict, loc, dataset_id):
    """
    Create monthly box plots showing seasonal patterns.

    Parameters
    ----------
    observed : pd.Series
        Observed diversion data
    ensemble_dict : dict
        Dictionary of ensemble realizations
    loc : str
        Location identifier
    dataset_id : str
        Dataset identifier
    """
    location_names = {'nyc': 'NYC', 'nj': 'NJ'}
    location_name = location_names.get(loc, loc)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 5))

    # Prepare monthly data
    observed_monthly = observed.groupby(observed.index.month)
    ensemble_monthly = {k: v.groupby(v.index.month) for k, v in ensemble_dict.items()}

    # Prepare data for box plots
    months = range(1, 13)
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    positions_obs = np.arange(12) * 3 - 0.5
    positions_ens = np.arange(12) * 3 + 0.5

    # Observed data by month
    obs_data = [observed_monthly.get_group(m).dropna().values if m in observed_monthly.groups else []
                for m in months]

    # Ensemble data by month (combine all realizations)
    ens_data = []
    for m in months:
        month_vals = []
        for real_dict in ensemble_monthly.values():
            if m in real_dict.groups:
                month_vals.extend(real_dict.get_group(m).dropna().values)
        ens_data.append(month_vals if month_vals else [])

    # Plot observed
    bp1 = ax.boxplot(obs_data, positions=positions_obs, widths=0.8,
                     patch_artist=True, showfliers=False,
                     boxprops=dict(facecolor='#1f77b4', alpha=0.6),
                     medianprops=dict(color='black', linewidth=1.5))

    # Plot ensemble
    bp2 = ax.boxplot(ens_data, positions=positions_ens, widths=0.8,
                     patch_artist=True, showfliers=False,
                     boxprops=dict(facecolor='#ff7f0e', alpha=0.6),
                     medianprops=dict(color='black', linewidth=1.5))

    # Formatting
    ax.set_xticks(np.arange(12) * 3)
    ax.set_xticklabels(month_labels)
    ax.set_xlabel('Month')
    ax.set_ylabel(f'{location_name} Diversion (MGD)')
    ax.set_title(f'{location_name} Diversion Seasonal Patterns')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#1f77b4', alpha=0.6, label='Observed'),
                      Patch(facecolor='#ff7f0e', alpha=0.6, label='Ensemble')]
    ax.legend(handles=legend_elements, loc='upper right', frameon=False)

    plt.tight_layout()
    fname = f"{FIG_DIR_DIVERSIONS}/{dataset_id}_{loc}_seasonal.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


def plot_timeseries_overview(observed, ensemble_dict, loc, dataset_id):
    """
    Create time series overview plot.

    Parameters
    ----------
    observed : pd.Series
        Observed diversion data
    ensemble_dict : dict
        Dictionary of ensemble realizations
    loc : str
        Location identifier
    dataset_id : str
        Dataset identifier
    """
    location_names = {'nyc': 'NYC', 'nj': 'NJ'}
    location_name = location_names.get(loc, loc)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 4))

    # Plot ensemble percentiles
    # Combine all ensemble data
    all_dates = sorted(set().union(*[set(v.index) for v in ensemble_dict.values()]))
    ensemble_matrix = np.zeros((len(ensemble_dict), len(all_dates)))
    ensemble_matrix[:] = np.nan

    for i, (key, series) in enumerate(ensemble_dict.items()):
        for j, date in enumerate(all_dates):
            if date in series.index:
                ensemble_matrix[i, j] = series.loc[date]

    # Calculate percentiles
    p10 = np.nanpercentile(ensemble_matrix, 10, axis=0)
    p50 = np.nanpercentile(ensemble_matrix, 50, axis=0)
    p90 = np.nanpercentile(ensemble_matrix, 90, axis=0)

    # Plot ensemble range
    ax.fill_between(all_dates, p10, p90, color='#ff7f0e', alpha=0.3,
                    label='Ensemble 10-90%')
    ax.plot(all_dates, p50, color='#ff7f0e', linewidth=1.5,
           label='Ensemble Median', alpha=0.7)

    # Plot observed (resample to monthly for clarity)
    obs_monthly = observed.resample('MS').mean()
    ax.plot(obs_monthly.index, obs_monthly.values, color='#1f77b4',
           linewidth=2, label='Observed (Monthly Avg)', zorder=3)

    ax.set_xlabel('Date')
    ax.set_ylabel(f'{location_name} Diversion (MGD)')
    ax.set_title(f'{location_name} Diversion Time Series')
    ax.legend(frameon=False, loc='upper left')
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    fname = f"{FIG_DIR_DIVERSIONS}/{dataset_id}_{loc}_timeseries.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


def main(dataset_id):
    """Main function"""

    print("=" * 60)
    print(f"DIVERSION DIAGNOSTICS: {dataset_id}")
    print("=" * 60)

    # Verify dataset
    verify_dataset_id(dataset_id)

    # Loop through both NYC and NJ
    for loc in ['nyc', 'nj']:
        location_names = {'nyc': 'NYC', 'nj': 'NJ'}
        location_name = location_names.get(loc, loc)

        print(f"\n{location_name} Diversions:")
        print("-" * 40)

        # Load observed data
        print(f"  Loading observed {location_name} diversions...")
        try:
            observed = load_observed_diversions(loc=loc)
            print(f"    Loaded {len(observed)} days of observed data")
            print(f"    Date range: {observed.index.min()} to {observed.index.max()}")
        except Exception as e:
            print(f"  ERROR loading observed data: {e}")
            continue

        # Load ensemble data
        print(f"  Loading ensemble {location_name} diversions...")
        try:
            ensemble_dict = load_ensemble_diversions(dataset_id, loc=loc)
            print(f"    Loaded {len(ensemble_dict)} realizations")
            sample_key = list(ensemble_dict.keys())[0]
            print(f"    Each realization has {len(ensemble_dict[sample_key])} days")
        except Exception as e:
            print(f"  ERROR loading ensemble data: {e}")
            print(f"  This may be expected if diversion extrapolation hasn't been run yet.")
            continue

        # Create plots
        print(f"  Creating distribution comparison plot...")
        plot_distribution_comparison(observed, ensemble_dict, loc, dataset_id)

        print(f"  Creating seasonal pattern plot...")
        plot_seasonal_patterns(observed, ensemble_dict, loc, dataset_id)

        print(f"  Creating time series overview plot...")
        plot_timeseries_overview(observed, ensemble_dict, loc, dataset_id)

    print("\n" + "=" * 60)
    print("Diversion diagnostic plots completed!")
    print("=" * 60)


if __name__ == "__main__":

    # Get the dataset_id from command line arguments
    if len(sys.argv) != 2:
        print("Usage: python SI3_plot_diversion_diagnostics.py <dataset_id>")
        print(f"Available datasets: {list(DATASET_CONFIGS.keys())}")
        sys.exit(1)

    dataset_id = sys.argv[1]
    verify_dataset_id(dataset_id)

    main(dataset_id)
