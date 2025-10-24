"""
Plot comprehensive streamflow distribution comparison across scenarios.

This script creates a 3-panel publication-quality figure comparing streamflow
distributions for historic data and multiple ensemble scenarios.

Figure Layout:
--------------
- Top-left: Annual total flow distributions (KDE plots)
- Top-right: Overlapping annual flow duration curves with ensemble ranges
- Bottom: Weekly streamflow patterns across the year

The figure demonstrates:
1. The stationary ensemble envelopes the historic data with similar distributions
2. The shifts in climate-adjusted scenarios relative to historic and stationary

Data Requirements:
------------------
This script requires postprocessed data from Pywr-DRB simulations. Run the
following steps first:
1. Generate ensemble sets (01_generate_ensemble_sets.py)
2. Prepare Pywr-DRB inputs (02_prep_pywrdrb_inputs.py)
3. Run Pywr-DRB simulations (03_run_pywrdrb_simulations.py)
4. Postprocess data (04_postprocess_data.py)

The script loads data from: ./pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5

Usage:
------
python 10_plot_streamflow_scenario_comparison.py [site]

Arguments:
    site : str, optional (default='delMontague')
        Site name to analyze

Example:
    python 10_plot_streamflow_scenario_comparison.py delMontague
"""

import sys
import os

# Add SGLib to path if it exists (for HDF5Manager)
sglib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../SGLib/src'))
if os.path.exists(sglib_path) and sglib_path not in sys.path:
    sys.path.insert(0, sglib_path)

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

import pywrdrb
from methods.load import load_drb_reconstruction
from config import *


def calculate_annual_totals(flow_data):
    """Calculate annual total flows from daily data."""
    annual = flow_data.resample('YE').sum()
    # Only include complete years
    annual = annual[annual.index <= '2019-12-31']
    return annual.values.flatten()


def calculate_flow_duration_curve(flow_data):
    """
    Calculate flow duration curve from daily data.
    Returns sorted flows and their exceedance probabilities.
    """
    flows = flow_data.values.flatten()
    flows = flows[~np.isnan(flows)]
    flows_sorted = np.sort(flows)[::-1]  # Sort descending
    exceedance = np.arange(1, len(flows_sorted) + 1) / len(flows_sorted)
    return exceedance, flows_sorted


def calculate_weekly_statistics(flow_data):
    """
    Calculate weekly statistics (median, 10th, 90th percentiles) across all realizations.
    Returns week of year (1-52) and statistics.
    """
    # Add week of year column
    weekly = flow_data.copy()
    weekly['week'] = weekly.index.isocalendar().week

    # Group by week and calculate statistics across all realizations
    if isinstance(flow_data, pd.DataFrame):
        # Multiple realizations
        weekly_stats = weekly.groupby('week').agg({
            col: ['median', lambda x: np.percentile(x, 10), lambda x: np.percentile(x, 90)]
            for col in flow_data.columns
        })

        # Aggregate across all realizations
        medians = []
        p10s = []
        p90s = []
        for week in range(1, 53):
            if week in weekly_stats.index:
                week_data = weekly[weekly['week'] == week].drop(columns=['week']).values.flatten()
                week_data = week_data[~np.isnan(week_data)]
                medians.append(np.median(week_data))
                p10s.append(np.percentile(week_data, 10))
                p90s.append(np.percentile(week_data, 90))
            else:
                medians.append(np.nan)
                p10s.append(np.nan)
                p90s.append(np.nan)

        return np.arange(1, 53), np.array(p10s), np.array(medians), np.array(p90s)
    else:
        # Single time series (historic)
        weekly_stats = weekly.groupby('week').agg(['median',
                                                     lambda x: np.percentile(x, 10),
                                                     lambda x: np.percentile(x, 90)])
        weeks = weekly_stats.index.values
        p10 = weekly_stats.iloc[:, 1].values
        median = weekly_stats.iloc[:, 0].values
        p90 = weekly_stats.iloc[:, 2].values
        return weeks, p10, median, p90


def plot_streamflow_comparison(site='delMontague',
                                dataset_ids=None,
                                fname=None):
    """
    Create comprehensive 3-panel figure comparing streamflow distributions.

    Parameters:
    -----------
    site : str
        Site name to analyze (default: 'delMontague')
    dataset_ids : list
        List of dataset IDs to plot. If None, uses all available datasets.
    fname : str
        Output filename for the figure

    Returns:
    --------
    bool : True if successful, False otherwise
    """

    print(f"Creating streamflow comparison figure for {site}...")

    # Default to all datasets if not specified
    if dataset_ids is None:
        dataset_ids = ['stationary_ensemble',
                      'climate_adjusted_low',
                      'climate_adjusted_medium',
                      'climate_adjusted_high']

    # Verify all datasets exist in config
    for dataset_id in dataset_ids:
        verify_dataset_id(dataset_id)

    ### Load historic reconstruction data
    print("Loading historic reconstruction data...")
    Q_hist = load_drb_reconstruction()
    Q_hist.replace(0, np.nan, inplace=True)
    Q_hist.drop(columns=['delTrenton'], inplace=True, errors='ignore')

    if site not in Q_hist.columns:
        raise ValueError(f"Site {site} not found in historic data. Available sites: {list(Q_hist.columns)}")

    Q_hist_site = Q_hist[site]
    print(f"  Loaded {len(Q_hist_site)} days of historic data for {site}")

    ### Load ensemble data for each scenario using fast HDF5 loading
    print("\nLoading ensemble data from postprocessed HDF5 files...")
    ensemble_data = {}

    for dataset_id in dataset_ids:
        print(f"  Loading {dataset_id}...")

        # Check if postprocessed data exists
        data_file = f'./pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5'
        if not os.path.exists(data_file):
            print(f"    WARNING: Postprocessed data not found: {data_file}")
            print(f"    Skipping {dataset_id}...")
            continue

        # Load gage flow data using fast HDF5 loading
        data = pywrdrb.Data()
        data.load_from_export(data_file, results_sets=['gage_flow'])

        # Extract ensemble dict: {realization_id: DataFrame}
        syn_ensemble = data.gage_flow[dataset_id]
        realization_ids = list(syn_ensemble.keys())
        n_realizations = len(realization_ids)

        print(f"    Loaded {n_realizations} realizations from HDF5")

        # Convert to site-based structure: DataFrame with columns = realizations
        # Each realization DataFrame has columns = sites, we want site column across realizations
        site_data_list = []
        for real_id in realization_ids:
            if site not in syn_ensemble[real_id].columns:
                print(f"    WARNING: Site {site} not found in realization {real_id}, skipping dataset...")
                site_data_list = None
                break

            # Extract site column and rename to realization ID
            site_series = syn_ensemble[real_id][site].loc[:'2019-12-31']
            site_data_list.append(site_series)

        if site_data_list is None:
            continue

        # Concatenate into DataFrame with columns = realization IDs
        ensemble_data[dataset_id] = pd.concat(site_data_list, axis=1, keys=realization_ids)

        # Free memory
        del data, syn_ensemble

        print(f"    Converted to site-based structure: {ensemble_data[dataset_id].shape}")

    if len(ensemble_data) == 0:
        print("\nERROR: No ensemble data found!")
        print("Run postprocessing (04_postprocess_data.py) first!")
        return False

    print(f"\nSuccessfully loaded {len(ensemble_data)} ensemble datasets")

    ### Define publication-quality colors and labels
    colors = {
        'historic': '#000000',  # Black
        'stationary_ensemble': '#1f77b4',  # Blue
        'climate_adjusted_low': '#d62728',  # Red (Dry)
        'climate_adjusted_medium': '#9467bd',  # Purple (Medium)
        'climate_adjusted_high': '#2ca02c',  # Green (Wet)
    }

    labels = {
        'historic': 'Historic',
        'stationary_ensemble': 'Stationary',
        'climate_adjusted_low': 'Climate Low',
        'climate_adjusted_medium': 'Climate Med',
        'climate_adjusted_high': 'Climate High',
    }

    # Create figure with custom layout
    print("Creating figure...")
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig,
                          height_ratios=[1, 1.2],
                          hspace=0.35, wspace=0.3,
                          bottom=0.15)  # Leave space for legend

    ax1 = fig.add_subplot(gs[0, 0])  # Top-left: Annual distributions
    ax2 = fig.add_subplot(gs[0, 1])  # Top-right: FDCs
    ax3 = fig.add_subplot(gs[1, :])  # Bottom: Weekly flows

    ### PANEL 1: Annual Total Flow Distributions (KDE)
    print("Generating Panel 1: Annual flow distributions...")

    # Calculate annual totals for historic
    annual_hist = calculate_annual_totals(Q_hist_site)

    # Plot historic as scatter points
    kde_hist = stats.gaussian_kde(annual_hist)
    x_range = np.linspace(annual_hist.min() * 0.8, annual_hist.max() * 1.2, 200)
    ax1.plot(x_range, kde_hist(x_range),
             color=colors['historic'], linewidth=2.5,
             label=labels['historic'], zorder=10)

    # Plot ensemble distributions
    for dataset_id in ensemble_data.keys():
        annual_syn = calculate_annual_totals(ensemble_data[dataset_id])
        kde_syn = stats.gaussian_kde(annual_syn)
        ax1.fill_between(x_range, 0, kde_syn(x_range),
                        color=colors[dataset_id], alpha=0.4,
                        label=labels[dataset_id])
        ax1.plot(x_range, kde_syn(x_range),
                color=colors[dataset_id], linewidth=2, alpha=0.8)

    ax1.set_xlabel('Annual Flow (MGD)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax1.set_title('Annual Flow Distributions', fontsize=14, fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.tick_params(labelsize=11)

    ### PANEL 2: Flow Duration Curves
    print("Generating Panel 2: Flow duration curves...")

    # Calculate FDC for historic
    exc_hist, flow_hist = calculate_flow_duration_curve(Q_hist_site)
    ax2.scatter(exc_hist[::100], flow_hist[::100],
               color=colors['historic'], s=50, alpha=0.7,
               label=labels['historic'], zorder=10, marker='o')

    # Plot ensemble FDCs with ranges
    for dataset_id in ensemble_data.keys():
        Q_syn = ensemble_data[dataset_id]

        # Calculate FDC for each realization
        fdcs = []
        for col in Q_syn.columns:
            exc, flow = calculate_flow_duration_curve(Q_syn[col])
            # Interpolate to common exceedance values
            exc_common = np.linspace(0, 1, 1000)
            flow_interp = np.interp(exc_common, exc, flow)
            fdcs.append(flow_interp)

        fdcs = np.array(fdcs)
        p10 = np.percentile(fdcs, 10, axis=0)
        p50 = np.percentile(fdcs, 50, axis=0)
        p90 = np.percentile(fdcs, 90, axis=0)

        # Plot range
        ax2.fill_between(exc_common, p10, p90,
                        color=colors[dataset_id], alpha=0.3)
        ax2.plot(exc_common, p50,
                color=colors[dataset_id], linewidth=2,
                label=labels[dataset_id])

    ax2.set_xlabel('Exceedance Probability', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Flow (MGD)', fontsize=12, fontweight='bold')
    ax2.set_title('Annual Flow Duration Curves', fontsize=14, fontweight='bold', pad=10)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, linestyle='--', which='both')
    ax2.tick_params(labelsize=11)

    ### PANEL 3: Weekly Streamflow Ranges
    print("Generating Panel 3: Weekly streamflow ranges...")

    # Calculate weekly stats for historic
    weeks_hist, p10_hist, med_hist, p90_hist = calculate_weekly_statistics(Q_hist_site)
    ax3.fill_between(weeks_hist, p10_hist, p90_hist,
                     color=colors['historic'], alpha=0.3)
    ax3.plot(weeks_hist, med_hist,
            color=colors['historic'], linewidth=2.5,
            label=labels['historic'], zorder=10)

    # Plot ensemble weekly ranges
    for dataset_id in ensemble_data.keys():
        Q_syn = ensemble_data[dataset_id]
        weeks, p10, med, p90 = calculate_weekly_statistics(Q_syn)

        ax3.fill_between(weeks, p10, p90,
                        color=colors[dataset_id], alpha=0.25)
        ax3.plot(weeks, med,
                color=colors[dataset_id], linewidth=2,
                label=labels[dataset_id])

    ax3.set_xlabel('Week of Year', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Flow (MGD)', fontsize=12, fontweight='bold')
    ax3.set_title('Weekly Streamflow Patterns', fontsize=14, fontweight='bold', pad=10)
    ax3.set_xlim(1, 52)
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3, linestyle='--', which='both')
    ax3.tick_params(labelsize=11)

    # Add monthly tick marks on top axis
    month_starts = [1, 5, 9, 13, 18, 22, 27, 31, 35, 40, 44, 48]
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax3_top = ax3.twiny()
    ax3_top.set_xlim(1, 52)
    ax3_top.set_xticks(month_starts)
    ax3_top.set_xticklabels(month_labels, fontsize=10)
    ax3_top.tick_params(length=0)

    ### Create single legend at bottom
    handles, leg_labels = ax1.get_legend_handles_labels()
    fig.legend(handles, leg_labels,
              loc='lower center',
              bbox_to_anchor=(0.5, 0.02),
              ncol=5,
              fontsize=12,
              frameon=True,
              fancybox=True,
              shadow=True)

    # Add overall title
    fig.suptitle(f'Streamflow Distribution Comparison: {site}',
                fontsize=16, fontweight='bold', y=0.98)

    # Save figure
    if fname is None:
        fname = f"{FIG_DIR}/streamflow_scenario_comparison_{site}.png"

    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {fname}")

    plt.close()
    return True


def main(site='delMontague'):
    """
    Main entry point for generating streamflow scenario comparison figure.

    Parameters:
    -----------
    site : str
        Site name to analyze (default: 'delMontague')
    """

    print("=" * 80)
    print("STREAMFLOW SCENARIO COMPARISON FIGURE")
    print("=" * 80)
    print(f"Site: {site}")
    print("=" * 80)

    # Create output directory
    os.makedirs(FIG_DIR, exist_ok=True)

    # Generate figure
    success = plot_streamflow_comparison(site=site)

    if success:
        print("\n" + "=" * 80)
        print("Figure generated successfully!")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("ERROR: Figure generation failed!")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) > 2:
        print("Usage: python 10_plot_streamflow_scenario_comparison.py [site]")
        print("Example: python 10_plot_streamflow_scenario_comparison.py delMontague")
        sys.exit(1)

    site = sys.argv[1] if len(sys.argv) == 2 else 'delMontague'
    main(site)
