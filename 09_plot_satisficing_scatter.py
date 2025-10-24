"""
Plot NYC inflow vs Montague contributions colored by satisficing conditions.

Satisficing conditions:
1. NYC storage >= 20% throughout June-Dec period
2. Montague flow target violations <= 3 continuous days

This script uses pre-calculated metrics from the postprocessing output:
- shortage: Pre-calculated flow target violations for each node
- contribution: Pre-calculated NYC downstream contributions to Montague
- inflow: Reservoir inflows with aggregated NYC values
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

import pywrdrb
from config import *


def calculate_satisficing_conditions(data, dataset_id, storage_threshold=20.0, violation_days=3):
    """
    Calculate satisficing conditions for each (year, realization) pair using pre-calculated metrics.
    
    Parameters:
    -----------
    data : pywrdrb.Data
        Data object with pre-calculated shortage, contribution, res_storage, and inflow
    dataset_id : str
        Dataset identifier
    storage_threshold : float
        Minimum acceptable NYC storage percentage (default 20%)
    violation_days : int
        Maximum acceptable continuous Montague violation days (default 3)
    
    Returns:
    --------
    pd.DataFrame : Results with satisficing status and aggregated metrics
    """
    
    nyc_reservoirs = ['cannonsville', 'pepacton', 'neversink']
    
    # Storage capacities for NYC reservoirs (MG)
    storage_capacities = {
        'cannonsville': 95706,
        'pepacton': 140190,
        'neversink': 34941
    }
    total_capacity = sum(storage_capacities.values())
    
    # Get realizations from shortage data (all dicts should have same realizations)
    realizations = list(data.shortage[dataset_id].keys())
    
    results = {
        'year': [],
        'realization': [],
        'nyc_inflow_jun_dec': [],
        'montague_contrib_jun_dec': [],
        'satisficing': [],
        'min_storage_pct': [],
        'max_violation_days': []
    }
    
    for r in realizations:
        # Use pre-calculated data directly from postprocessing
        nyc_storage = data.res_storage[dataset_id][r][nyc_reservoirs].sum(axis=1)
        nyc_storage_pct = 100.0 * nyc_storage / total_capacity
        
        # Use pre-calculated shortage (already filtered for violations >= 3 consecutive days)
        montague_shortage = data.shortage[dataset_id][r]['delMontague']
        
        # Use pre-calculated NYC inflow (aggregated in postprocessing)
        nyc_inflow = data.inflow[dataset_id][r]['nyc']
        
        # Use pre-calculated NYC contribution to Montague
        montague_contrib = data.contribution[dataset_id][r]['mrf_montagueTrenton_nyc']
        
        # Get years in data
        years = pd.DatetimeIndex(nyc_storage.index).year.unique()
        
        for year in years:
            # Filter June 1 - Dec 31
            mask = (nyc_storage.index >= f'{year}-06-01') & (nyc_storage.index <= f'{year}-12-31')
            
            if not mask.any():
                continue
            
            # Check storage condition
            min_storage = nyc_storage_pct[mask].min()
            storage_ok = min_storage >= storage_threshold
            
            # Check Montague violation condition using pre-calculated shortage
            violations = montague_shortage[mask] > 0
            if violations.any():
                # Calculate max consecutive violation days
                groups = (violations != violations.shift()).cumsum()
                max_consec = violations.groupby(groups).sum().max()
            else:
                max_consec = 0
            
            montague_ok = max_consec <= violation_days
            
            # Calculate aggregates for Jun-Dec period
            total_inflow = nyc_inflow[mask].sum()
            total_contrib = montague_contrib[mask].sum()
            
            # Store results
            results['year'].append(year)
            results['realization'].append(r)
            results['nyc_inflow_jun_dec'].append(total_inflow)
            results['montague_contrib_jun_dec'].append(total_contrib)
            results['satisficing'].append(storage_ok and montague_ok)
            results['min_storage_pct'].append(min_storage)
            results['max_violation_days'].append(max_consec)
    
    return pd.DataFrame(results)


def plot_satisficing_scatter(results_df, dataset_id, figsize=(10, 8), 
                             alpha_satisficing=0.6, alpha_nonsatisficing=0.8,
                             fname=None):
    """
    Create scatter plot of NYC inflow vs Montague contributions colored by satisficing.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Results from calculate_satisficing_conditions
    dataset_id : str
        Dataset identifier for title
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Separate satisficing and non-satisficing
    satisficing = results_df[results_df['satisficing']]
    non_satisficing = results_df[~results_df['satisficing']]
    
    # Plot satisficing points (blue)
    ax.scatter(satisficing['nyc_inflow_jun_dec'], 
              satisficing['montague_contrib_jun_dec'],
              c='#2E86AB', alpha=alpha_satisficing, s=20, 
              edgecolors='none', label=f'Satisficing (n={len(satisficing)})')
    
    # Plot non-satisficing points (red)
    ax.scatter(non_satisficing['nyc_inflow_jun_dec'], 
              non_satisficing['montague_contrib_jun_dec'],
              c='#A23B72', alpha=alpha_nonsatisficing, s=20,
              edgecolors='none', label=f'Non-satisficing (n={len(non_satisficing)})')
    
    # Labels and formatting
    ax.set_xlabel('NYC Reservoir Inflow (Jun-Dec) [MG]', fontsize=12)
    ax.set_ylabel('NYC → Montague Contributions (Jun-Dec) [MG]', fontsize=12)
    ax.set_title(f'{dataset_id}\nSatisficing Conditions Analysis', fontsize=14)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    
    # Add text box with criteria
    criteria_text = (
        'Satisficing Criteria:\n'
        '• NYC storage ≥ 20% (Jun-Dec)\n'
        '• Montague violations ≤ 3 consecutive days'
    )
    ax.text(0.02, 0.98, criteria_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Add statistics text
    pct_satisficing = 100 * len(satisficing) / len(results_df)
    stats_text = f'Satisficing: {pct_satisficing:.1f}%'
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
            fontsize=11, horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
    
    plt.tight_layout()
    
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        print(f"Saved: {fname}")
    
    return fig, ax


def plot_satisficing_comparison(all_results, dataset_ids, 
                                figsize=(15, 5), fname=None):
    """
    Create side-by-side comparison of multiple datasets.
    """
    
    n_datasets = len(dataset_ids)
    fig, axes = plt.subplots(1, n_datasets, figsize=figsize, sharey=True, sharex=True)
    
    if n_datasets == 1:
        axes = [axes]
    
    for idx, dataset_id in enumerate(dataset_ids):
        ax = axes[idx]
        results_df = all_results[dataset_id]
        
        # Separate satisficing and non-satisficing
        satisficing = results_df[results_df['satisficing']]
        non_satisficing = results_df[~results_df['satisficing']]
        
        # Plot
        ax.scatter(satisficing['nyc_inflow_jun_dec'], 
                  satisficing['montague_contrib_jun_dec'],
                  c='#2E86AB', alpha=0.5, s=15, edgecolors='none')
        
        ax.scatter(non_satisficing['nyc_inflow_jun_dec'], 
                  non_satisficing['montague_contrib_jun_dec'],
                  c='#A23B72', alpha=0.7, s=15, edgecolors='none')
        
        # Formatting
        ax.set_title(f'{dataset_id}\n({100*len(satisficing)/len(results_df):.1f}% satisficing)', 
                    fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        if idx == 0:
            ax.set_ylabel('NYC → Montague Contributions (Jun-Dec) [MG]', fontsize=11)
        ax.set_xlabel('NYC Inflow (Jun-Dec) [MG]', fontsize=11)
    
    # Create shared legend
    blue_patch = mpatches.Patch(color='#2E86AB', label='Satisficing')
    red_patch = mpatches.Patch(color='#A23B72', label='Non-satisficing')
    fig.legend(handles=[blue_patch, red_patch], 
              loc='upper center', bbox_to_anchor=(0.5, 0.98),
              ncol=2, frameon=True, fancybox=True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        print(f"Saved: {fname}")
    
    return fig, axes


def main(dataset_id):
    """Main function to generate satisficing condition plots."""
    
    print("=" * 60)
    print(f"SATISFICING CONDITIONS ANALYSIS: {dataset_id}")
    print("=" * 60)
    
    # Verify dataset
    verify_dataset_id(dataset_id)
    
    # Load pre-calculated data from postprocessing
    fname = f'./pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5'
    if not os.path.exists(fname):
        print(f"ERROR: Data file not found: {fname}")
        print("Run postprocessing (04_postprocess_data.py) first!")
        return
    
    print(f"Loading pre-calculated metrics from: {fname}")
    data = pywrdrb.Data()
    # Only load what we need: pre-calculated shortage, contribution, plus res_storage and inflow
    data.load_from_export(fname, results_sets=['res_storage', 'inflow', 'shortage', 'contribution'])
    
    # Calculate satisficing conditions using pre-calculated metrics
    print("Calculating satisficing conditions...")
    results = calculate_satisficing_conditions(data, dataset_id)
    
    # Print summary statistics
    n_total = len(results)
    n_satisficing = results['satisficing'].sum()
    pct_satisficing = 100 * n_satisficing / n_total
    
    print(f"\nResults summary:")
    print(f"  Total year-realization pairs: {n_total}")
    print(f"  Satisficing: {n_satisficing} ({pct_satisficing:.1f}%)")
    print(f"  Non-satisficing: {n_total - n_satisficing} ({100-pct_satisficing:.1f}%)")
    
    # Break down by failure type
    storage_fail = results['min_storage_pct'] < 20
    montague_fail = results['max_violation_days'] > 3
    both_fail = storage_fail & montague_fail
    
    print(f"\nFailure breakdown:")
    print(f"  Storage < 20%: {storage_fail.sum()} ({100*storage_fail.sum()/n_total:.1f}%)")
    print(f"  Montague > 3 days: {montague_fail.sum()} ({100*montague_fail.sum()/n_total:.1f}%)")
    print(f"  Both failures: {both_fail.sum()} ({100*both_fail.sum()/n_total:.1f}%)")
    
    # Create output directory
    output_dir = f"{FIG_DIR}/satisficing"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate plot
    print(f"\nGenerating scatter plot...")
    fname = f"{output_dir}/{dataset_id}_satisficing_scatter.png"
    fig, ax = plot_satisficing_scatter(results, dataset_id, fname=fname)
    
    # Save results to CSV for further analysis
    csv_fname = f"{output_dir}/{dataset_id}_satisficing_results.csv"
    results.to_csv(csv_fname, index=False)
    print(f"Saved results to: {csv_fname}")
    
    print("=" * 60)
    print("Analysis complete!")
    
    return results


def compare_all_datasets():
    """Compare satisficing conditions across all datasets using pre-calculated metrics."""
    
    print("=" * 60)
    print("COMPARING SATISFICING CONDITIONS ACROSS DATASETS")
    print("=" * 60)
    
    all_results = {}
    
    # Calculate for each dataset
    for dataset_id in DATASET_CONFIGS.keys():
        fname = f'./pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5'
        if not os.path.exists(fname):
            print(f"Skipping {dataset_id} (postprocessed data not found)")
            continue
        
        print(f"\nProcessing {dataset_id}...")
        data = pywrdrb.Data()
        # Load only pre-calculated metrics needed
        data.load_from_export(fname, results_sets=['res_storage', 'inflow', 'shortage', 'contribution'])
        
        results = calculate_satisficing_conditions(data, dataset_id)
        all_results[dataset_id] = results
        
        # Print summary
        pct_sat = 100 * results['satisficing'].sum() / len(results)
        print(f"  {dataset_id}: {pct_sat:.1f}% satisficing")
    
    # Create comparison plot
    if len(all_results) > 1:
        output_dir = f"{FIG_DIR}/satisficing"
        os.makedirs(output_dir, exist_ok=True)
        
        fname = f"{output_dir}/all_datasets_satisficing_comparison.png"
        plot_satisficing_comparison(all_results, list(all_results.keys()), fname=fname)
    
    return all_results


if __name__ == "__main__":
    # Get dataset_id from command line
    if len(sys.argv) == 2:
        dataset_id = sys.argv[1]
        if dataset_id == '--all':
            compare_all_datasets()
        else:
            verify_dataset_id(dataset_id)
            main(dataset_id)
    else:
        print("Usage: python plot_satisficing_conditions.py <dataset_id>")
        print("       python plot_satisficing_conditions.py --all")
        print(f"Available datasets: {list(DATASET_CONFIGS.keys())}")
        sys.exit(1)