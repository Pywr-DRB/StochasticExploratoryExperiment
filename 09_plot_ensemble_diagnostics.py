import sys
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sglib import Ensemble
from sglib.plotting import plot_validation_panel, plot_spatial_correlation

from methods.plotting.gridded import plot_fdc_gridded, plot_autocorrelation_gridded
from methods.load import load_baseline_historical_flow, load_and_combine_ensemble_sets

from methods.config import *


def plot_ensemble_diagnostics(dataset_id):
    """
    Generate diagnostic plots for ensemble validation
    
    Parameters:
    -----------
    dataset_id : str
        Dataset identifier to analyze
    """
    
    # Verify dataset
    verify_dataset_id(dataset_id)
    dataset_config = DATASET_CONFIGS[dataset_id]
    
    print(f"Generating ensemble diagnostics for: {dataset_id}")
    print(f"Dataset type: {dataset_config['type']}")
    print(f"Description: {dataset_config['description']}")
    
    # Get ensemble set specs for this dataset
    ensemble_set_specs = ENSEMBLE_SETS[dataset_id]
    
    # Check if ensemble data exists
    missing_sets = []
    for spec in ensemble_set_specs:
        if not os.path.exists(spec.files['gage_flow']):
            missing_sets.append(spec.set_id + 1)
    
    if missing_sets:
        print(f"ERROR: Missing ensemble sets: {missing_sets}")
        print(f"Run ensemble generation (01_generate_ensemble_sets.py) first!")
        return False
    
    ### Loading data
    ## Historic reconstruction data
    # Total flow
    Q = load_baseline_historical_flow(period='full', gage_flow=True, flowtype=BASELINE_DATASET)
    Q.replace(0, np.nan, inplace=True)
    Q.drop(columns=['delTrenton'], inplace=True)  # Remove Trenton gage as it is not used in the ensemble
    Q_monthly = Q.resample('MS').sum()

    # Catchment inflows
    Q_inflows = load_baseline_historical_flow(gage_flow=False, period='full', flowtype=BASELINE_DATASET)
    Q_inflows.replace(0, np.nan, inplace=True)
    Q_inflows.drop(columns=['delTrenton'], inplace=True)

    print(f"Loaded reconstruction data with {Q.shape[0]// 365} years of daily data for {Q.shape[1]} sites.")

    ## Synthetic ensemble
    print("Loading ensemble data (by_site=True)...")
    Q_syn = load_and_combine_ensemble_sets(ensemble_set_specs, by_site=True)

    print("Loading ensemble data (by_site=False)...")
    syn_ensemble = load_and_combine_ensemble_sets(ensemble_set_specs, by_site=False)

    # Get realization IDs and count
    realization_ids = list(syn_ensemble.keys())
    n_realizations = len(realization_ids)
    print(f"Loaded synthetic ensemble with {n_realizations} realizations for {len(Q_syn)} sites.")

    # Lazy computation: only resample monthly data when needed (not all sites upfront)
    print("Preparing monthly resampling (lazy evaluation)...")
    Q_syn_monthly = {}  # Will be computed on-demand per site

    # Cache for Ensemble objects to avoid redundant creation
    ensemble_cache = {}

    # Create figure directories
    fig_subdirs = ['drought_metrics', 'fdc', 'autocorrelation', 'statistical_validation', 'spatial_correlation']
    for subdir in fig_subdirs:
        os.makedirs(f"{FIG_DIR}/{subdir}", exist_ok=True)

    ### Drought metric scatter plot (if drought metrics exist)
    # NOTE: Drought characteristics plotting requires ensemble data, not just metrics CSV
    # The new SGLib API calculates drought metrics from flow data directly
    # If you need this plot, create an Ensemble object and pass it to plot_drought_characteristics
    # along with the observed data. The function will compute metrics internally.
    print("Skipping drought metrics plot (requires Ensemble object input - see updated SGLib API)")

    ### Gridded FDCs and Autocorrelation plots
    for freq in ['daily', 'monthly']:

        print(f"\nGenerating {freq} diagnostic plots...")

        # Use daily or monthly flows
        if freq == 'daily':
            Qs = Q_syn
            Qh = Q
        else:
            # Compute monthly data on-demand only for sites we need
            if not Q_syn_monthly:
                print("  Computing monthly resampling for all sites...")
                Q_syn_monthly = {k: v.resample('MS').sum() for k, v in Q_syn.items()}
            Qs = Q_syn_monthly
            Qh = Q_monthly

        # Subsets of nodes based on generation methods
        for node_type in ['major', 'minor']:
            if node_type == 'major':
                nodes = pywrdrb_nodes_to_generate
            else:
                nodes = pywrdrb_nodes_to_regress
            
            print(f"  Plotting {freq} gridded FDCs for {node_type} nodes...")
            
            # Gridded FDC plot
            fn = f"{dataset_id}_{freq}_gage_flow_{node_type}_nodes.png"
            fname = f"{FIG_DIR}/fdc/{fn}"
            plot_fdc_gridded(Qh.loc[:, nodes], 
                            Qs=Qs,
                            timestep=freq,
                            fname=fname)

            print(f"  Plotting {freq} gridded ACFs for {node_type} nodes...")
            
            # Gridded autocorrelation plot
            fname = f"{FIG_DIR}/autocorrelation/{fn}"
            plot_autocorrelation_gridded(Qh.loc[:, nodes],
                                        Qs=Qs,
                                        timestep=freq,
                                        fname=fname)

    ### Statistical validation plots
    validate_nodes = ['delMontague', 'cannonsville', 'pepacton', 'delLordville']

    for site in validate_nodes:

        print(f"Plotting statistical validation for {site}...")

        if site == 'delTrenton':
            continue

        logscale = False

        fname = f"{dataset_id}_{site}_log.png" if logscale else f"{dataset_id}_{site}.png"
        fname = f"{FIG_DIR}/statistical_validation/{fname}"

        # Check cache first
        cache_key = f"validation_{site}"
        if cache_key not in ensemble_cache:
            # Convert synthetic data to Ensemble object
            # Q_syn[site] is a DataFrame with realizations as columns
            ensemble_dict = {}
            for col in Q_syn[site].columns:
                # Convert column to int if it's a string that represents an integer, otherwise use as-is
                if isinstance(col, str) and col.isdigit():
                    real_id = int(col)
                elif isinstance(col, (int, np.integer)):
                    real_id = int(col)
                else:
                    real_id = col
                ensemble_dict[real_id] = pd.DataFrame({site: Q_syn[site][col]})

            ensemble_cache[cache_key] = Ensemble(ensemble_dict)

        # Use new SGLib API
        plot_validation_panel(
            ensemble=ensemble_cache[cache_key],
            observed=Q.loc[:, site],
            site=site,
            timestep='monthly',
            log_space=logscale,
            filename=fname
        )

    ### Spatial correlation plots
    print("\nGenerating spatial correlation plots...")

    # Use first realization for correlation analysis
    Qs_df = syn_ensemble[realization_ids[0]].loc[:, Q.columns]

    # Create daily ensemble (cached)
    if 'spatial_daily' not in ensemble_cache:
        ensemble_dict_single = {0: Qs_df}
        ensemble_cache['spatial_daily'] = Ensemble(ensemble_dict_single)
    ensemble_single = ensemble_cache['spatial_daily']

    # Create monthly ensemble (cached, computed once)
    if 'spatial_monthly' not in ensemble_cache:
        Q_monthly_df = Q_monthly.loc[:, Qs_df.columns]
        Qs_monthly_df = Qs_df.resample('MS').sum()
        ensemble_dict_monthly = {0: Qs_monthly_df}
        ensemble_cache['spatial_monthly'] = Ensemble(ensemble_dict_monthly)
        ensemble_cache['Q_monthly_df'] = Q_monthly_df  # Cache for reuse
    ensemble_monthly = ensemble_cache['spatial_monthly']
    Q_monthly_df = ensemble_cache['Q_monthly_df']

    # Daily major nodes
    fname = f"{dataset_id}_daily_gage_flow_major_nodes.png"
    fname = f"{FIG_DIR}/spatial_correlation/{fname}"
    plot_spatial_correlation(
        ensemble_single,
        observed=Q.loc[:, pywrdrb_nodes_to_generate],
        realization=0,
        timestep='daily',
        method='pearson',
        show_difference=False,
        filename=fname
    )

    # Monthly major nodes
    fname = f"{dataset_id}_monthly_gage_flow_major_nodes.png"
    fname = f"{FIG_DIR}/spatial_correlation/{fname}"
    plot_spatial_correlation(
        ensemble_monthly,
        observed=Q_monthly_df.loc[:, pywrdrb_nodes_to_generate],
        realization=0,
        timestep='monthly',
        method='pearson',
        show_difference=False,
        filename=fname
    )

    # Daily minor nodes
    fname = f"{dataset_id}_daily_gage_flow_minor_nodes.png"
    fname = f"{FIG_DIR}/spatial_correlation/{fname}"
    plot_spatial_correlation(
        ensemble_single,
        observed=Q.loc[:, pywrdrb_nodes_to_regress],
        realization=0,
        timestep='daily',
        method='pearson',
        show_difference=False,
        filename=fname
    )

    # Monthly minor nodes
    fname = f"{dataset_id}_monthly_gage_flow_minor_nodes.png"
    fname = f"{FIG_DIR}/spatial_correlation/{fname}"
    plot_spatial_correlation(
        ensemble_monthly,
        observed=Q_monthly_df.loc[:, pywrdrb_nodes_to_regress],
        realization=0,
        timestep='monthly',
        method='pearson',
        show_difference=False,
        filename=fname
    )

    print(f"\nAll diagnostic plots saved for {dataset_id}!")
    return True


def main(dataset_id):
    """Main function"""
    
    print("=" * 60)
    print(f"ENSEMBLE DIAGNOSTIC PLOTS: {dataset_id}")
    print("=" * 60)
    
    # Generate all diagnostic plots
    success = plot_ensemble_diagnostics(dataset_id)
    
    if success:
        print("=" * 60)
        print("Diagnostic plots generated successfully!")
    else:
        print("=" * 60)
        print("ERROR: Diagnostic plot generation failed!")
        sys.exit(1)


if __name__ == "__main__":
    
    # Get the dataset_id from command line arguments
    if len(sys.argv) != 2:
        print("Usage: python 09_plot_ensemble_diagnostics.py <dataset_id>")
        print(f"Available datasets: {list(DATASET_CONFIGS.keys())}")
        sys.exit(1)
    
    dataset_id = sys.argv[1]
    verify_dataset_id(dataset_id)
    
    main(dataset_id)