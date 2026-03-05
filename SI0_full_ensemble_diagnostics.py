"""
Generate comprehensive diagnostic plots for ensemble validation (Supplemental).

This script creates detailed diagnostic plots for synthetic ensemble validation:
- Gridded FDC plots (daily/monthly, major/minor nodes)
- Gridded autocorrelation plots (daily/monthly, major/minor nodes)
- Statistical validation panels (selected sites)
- Spatial correlation plots (daily/monthly, major/minor nodes)
- Ensemble convergence diagnostics (mean/variance of annual flow vs. realization count)

For the main manuscript summary figure, see F1_plot_ensemble_diagnostics.py

Usage:
    python SI0_full_ensemble_diagnostics.py <dataset_id>
"""

import sys
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from synhydro import Ensemble
from synhydro.plotting import plot_validation_panel, plot_spatial_correlation

from methods.plotting.gridded import plot_fdc_gridded, plot_autocorrelation_gridded
from methods.plotting.ensemble_summary import plot_ensemble_convergence
from methods.load import load_baseline_historical_flow, load_and_combine_ensemble_sets
from methods.config import *

FIG_DIR = f"{FIG_DIR}/SI0_ensemble_diagnostics"


def plot_full_ensemble_diagnostics(dataset_id: str):
    """
    Generate comprehensive diagnostic plots for ensemble validation.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier to analyze

    Returns
    -------
    bool
        True if successful
    """
    # Verify dataset
    verify_dataset_id(dataset_id)
    dataset_config = DATASET_CONFIGS[dataset_id]

    print(f"Generating full ensemble diagnostics for: {dataset_id}")
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
        print("Run ensemble generation (01_generate_ensemble_sets.py) first!")
        return False

    ### Loading data
    ## Historic reconstruction data
    print("Loading historical data...")
    Q = load_baseline_historical_flow(period='baseline', gage_flow=True, flowtype=BASELINE_DATASET)
    Q.replace(0, np.nan, inplace=True)
    Q.drop(columns=['delTrenton'], inplace=True, errors='ignore')
    Q_monthly = Q.resample('MS').sum()

    print(f"Loaded reconstruction data: {Q.shape[0] // 365} years, {Q.shape[1]} sites")

    ## Synthetic ensemble
    print("Loading ensemble data (by_site=True)...")
    Q_syn = load_and_combine_ensemble_sets(ensemble_set_specs, by_site=True)

    print("Loading ensemble data (by_site=False)...")
    syn_ensemble = load_and_combine_ensemble_sets(ensemble_set_specs, by_site=False)

    # Get realization IDs and count
    realization_ids = list(syn_ensemble.keys())
    n_realizations = len(realization_ids)
    print(f"Loaded synthetic ensemble: {n_realizations} realizations, {len(Q_syn)} sites")

    # Lazy computation: only resample monthly data when needed
    Q_syn_monthly = {}

    # Cache for Ensemble objects to avoid redundant creation
    ensemble_cache = {}

    # Create figure directories
    fig_subdirs = ['fdc', 'autocorrelation', 'statistical_validation', 'spatial_correlation', 'convergence']
    for subdir in fig_subdirs:
        os.makedirs(f"{FIG_DIR}/{subdir}", exist_ok=True)

    ### Gridded FDCs and Autocorrelation plots
    for freq in ['daily', 'monthly']:
        print(f"\nGenerating {freq} diagnostic plots...")

        # Use daily or monthly flows
        if freq == 'daily':
            Qs = Q_syn
            Qh = Q
        else:
            # Compute monthly data on-demand
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
            plot_fdc_gridded(
                Qh.loc[:, nodes],
                Qs=Qs,
                timestep=freq,
                fname=fname
            )

            print(f"  Plotting {freq} gridded ACFs for {node_type} nodes...")

            # Gridded autocorrelation plot
            fname = f"{FIG_DIR}/autocorrelation/{fn}"
            plot_autocorrelation_gridded(
                Qh.loc[:, nodes],
                Qs=Qs,
                timestep=freq,
                fname=fname
            )

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
            ensemble_dict = {}
            for col in Q_syn[site].columns:
                if isinstance(col, str) and col.isdigit():
                    real_id = int(col)
                elif isinstance(col, (int, np.integer)):
                    real_id = int(col)
                else:
                    real_id = col
                ensemble_dict[real_id] = pd.DataFrame({site: Q_syn[site][col]})

            ensemble_cache[cache_key] = Ensemble(ensemble_dict)

        # Use SynHydro API
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
    Qs_df = syn_ensemble[realization_ids[0]]
    Qs_monthly_df = Qs_df.resample('MS').sum()

    for node_type in ['major', 'minor']:
        if node_type == 'major':
            nodes = pywrdrb_nodes_to_generate
        else:
            nodes = pywrdrb_nodes_to_regress

        # Filter synthetic data to only the relevant nodes
        ensemble_daily = Ensemble({0: Qs_df.loc[:, nodes]})
        ensemble_monthly_corr = Ensemble({0: Qs_monthly_df.loc[:, nodes]})

        for freq in ['daily', 'monthly']:
            print(f"  Plotting {freq} spatial correlation for {node_type} nodes...")
            fname = f"{FIG_DIR}/spatial_correlation/{dataset_id}_{freq}_gage_flow_{node_type}_nodes.png"

            if freq == 'daily':
                ens = ensemble_daily
                obs = Q.loc[:, nodes]
            else:
                ens = ensemble_monthly_corr
                obs = Q_monthly.loc[:, nodes]

            plot_spatial_correlation(
                ens,
                observed=obs,
                realization=0,
                timestep=freq,
                method='pearson',
                show_difference=False,
                filename=fname
            )

    ### Ensemble convergence diagnostics
    print("\nGenerating ensemble convergence plots...")

    for site in validate_nodes:
        if site not in Q_syn:
            print(f"  Skipping convergence for {site} (not in ensemble)")
            continue

        print(f"  Plotting convergence for {site}...")
        fname = f"{FIG_DIR}/convergence/{dataset_id}_{site}_convergence.png"
        plot_ensemble_convergence(
            Q_syn_site=Q_syn[site],
            realization_ids=realization_ids,
            site=site,
            fname=fname,
        )

    print(f"\nAll diagnostic plots saved for {dataset_id}!")
    return True


def main(dataset_id: str):
    """Main function."""
    print("=" * 60)
    print(f"FULL ENSEMBLE DIAGNOSTICS (SI0): {dataset_id}")
    print("=" * 60)

    success = plot_full_ensemble_diagnostics(dataset_id)

    if success:
        print("=" * 60)
        print("Diagnostic plots generated successfully!")
    else:
        print("=" * 60)
        print("ERROR: Diagnostic plot generation failed!")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python SI0_full_ensemble_diagnostics.py <dataset_id>")
        print(f"Available datasets: {list(DATASET_CONFIGS.keys())}")
        sys.exit(1)

    dataset_id = sys.argv[1]
    verify_dataset_id(dataset_id)

    main(dataset_id)
