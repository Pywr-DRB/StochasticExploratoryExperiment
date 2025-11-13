"""
Plot NYC aggregate storage from synthetic ensemble simulations.

This script loads Pywr simulation results and plots the aggregate NYC reservoir storage
across all realizations, showing ensemble spread and individual trajectories.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from methods.config import *
import pywrdrb
from pywrdrb import Data

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10


def load_nyc_storage_from_simulation(dataset_id, ensemble_set_specs):
    """
    Load NYC aggregate storage from Pywr simulation output using pywrdrb.Data().

    Parameters
    ----------
    dataset_id : str
        Dataset identifier (e.g., 'stationary_ensemble')
    ensemble_set_specs : list
        List of ensemble set specifications from config

    Returns
    -------
    dict
        Dictionary mapping realization IDs to NYC aggregate storage Series
    """
    # NYC reservoirs that contribute to aggregate storage
    nyc_reservoirs = ['cannonsville', 'pepacton', 'neversink']

    print("  Setting up path navigator...")
    # Setup pathnavigator (required for pywrdrb.Data)
    pn_config = pywrdrb.get_pn_config()
    for spec in ensemble_set_specs:
        dataset_dir = spec.directory
        dataset_name = spec.directory.split('/')[-1]
        pn_config[f"flows/{dataset_name}"] = os.path.abspath(dataset_dir)
    pywrdrb.load_pn_config(pn_config)

    print("  Loading simulation outputs...")
    # Collect output filenames from all ensemble sets
    output_filenames = [spec.output_file for spec in ensemble_set_specs]

    # Initialize Data object with required results sets
    results_sets = ['res_storage']
    data = Data(results_sets=results_sets, print_status=False)

    # Load all outputs at once
    data.load_output(output_filenames=output_filenames)

    # Combine all sets into single dictionary
    storage_dict = {}
    global_realization_id = 0

    for set_idx, spec in enumerate(ensemble_set_specs):
        set_name = f"{dataset_id}_set{set_idx + 1}"

        if set_name not in data.res_storage:
            print(f"    WARNING: {set_name} not found in res_storage")
            continue

        set_data = data.res_storage[set_name]
        print(f"\n  Processing ensemble set {set_idx + 1} ({len(set_data)} realizations)...")

        # Get local realization IDs
        local_ids = list(set_data.keys())
        if not local_ids:
            continue

        min_local_id = min(local_ids)

        # Process each realization
        for local_id, storage_df in set_data.items():
            try:
                # Extract NYC reservoir storage columns
                storage_cols = [col for col in storage_df.columns
                               if col in nyc_reservoirs]

                if not storage_cols:
                    # Try case-insensitive matching
                    storage_cols = [col for col in storage_df.columns
                                   if any(res.lower() in col.lower() for res in nyc_reservoirs)]

                if storage_cols:
                    # Sum across NYC reservoirs to get aggregate storage
                    nyc_storage = storage_df[storage_cols].sum(axis=1)
                    storage_dict[global_realization_id] = nyc_storage
                    global_realization_id += 1
                else:
                    print(f"    Warning: No NYC storage columns found for realization {local_id}")
                    if len(storage_df.columns) > 0:
                        print(f"    Available columns: {storage_df.columns.tolist()[:5]}...")

            except Exception as e:
                print(f"    Warning: Could not process realization {local_id}: {e}")
                continue

    if not storage_dict:
        raise ValueError(f"No valid storage data found for dataset: {dataset_id}")

    print(f"\n  Successfully loaded {len(storage_dict)} realizations")
    return storage_dict


def plot_nyc_aggregate_storage(storage_dict,
                                title="NYC Aggregate Storage - Synthetic Ensemble",
                                ylabel="Storage (% of Maximum)",
                                figsize=(14, 6),
                                save_path=None):
    """
    Plot NYC aggregate storage ensemble traces as percent of maximum capacity.

    Parameters
    ----------
    storage_dict : dict
        Dictionary mapping realization IDs to storage Series/DataFrames
    title : str
        Plot title
    ylabel : str
        Y-axis label
    figsize : tuple
        Figure size (width, height)
    save_path : str or Path, optional
        Path to save figure. If None, displays interactively.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Convert to DataFrame for easier manipulation
    storage_df = pd.DataFrame(storage_dict)

    # Convert to percent of maximum capacity
    # NYC total capacity from config
    storage_pct_df = (storage_df / NYC_TOTAL_CAPACITY) * 100.0

    # Calculate min/max for statistics
    ensemble_min = storage_pct_df.min(axis=1)
    ensemble_max = storage_pct_df.max(axis=1)

    # Plot all individual realizations
    for real_id in storage_pct_df.columns:
        ax.plot(storage_pct_df.index, storage_pct_df[real_id],
                color='steelblue', alpha=0.3, linewidth=0.8, zorder=1)

    # Formatting
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)  # Set y-axis from 0-100%
    ax.grid(True, alpha=0.3)

    # Add summary statistics as text
    stats_text = f"Realizations: {len(storage_dict)}\n"
    stats_text += f"Storage Range: {ensemble_min.min():.1f}% - {ensemble_max.max():.1f}%\n"
    stats_text += f"NYC Total Capacity: {NYC_TOTAL_CAPACITY:.0f} MG"

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
           fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
        plt.close(fig)
    else:
        plt.show()

    return fig, ax


def main(dataset_id='stationary_ensemble'):
    """
    Main function to load and plot NYC aggregate storage.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier (e.g., 'stationary_ensemble', 'climate_adjusted_low')
    """
    print("=" * 80)
    print("NYC AGGREGATE STORAGE PLOTTING")
    print("=" * 80)

    # Verify dataset
    if dataset_id not in DATASET_CONFIGS:
        print(f"ERROR: Unknown dataset ID: {dataset_id}")
        print(f"Available datasets: {list(DATASET_CONFIGS.keys())}")
        return False

    dataset_config = DATASET_CONFIGS[dataset_id]
    print(f"\nDataset: {dataset_id}")
    print(f"Type: {dataset_config['type']}")
    print(f"Description: {dataset_config['description']}")

    # Get ensemble set specifications
    ensemble_set_specs = ENSEMBLE_SETS[dataset_id]
    print(f"Number of ensemble sets: {len(ensemble_set_specs)}")

    try:
        # Load NYC storage data using pywrdrb.Data()
        print("\nLoading simulation outputs...")
        storage_dict = load_nyc_storage_from_simulation(dataset_id, ensemble_set_specs)
        print(f"\nSuccessfully loaded storage data for {len(storage_dict)} realizations")

        # Create output directory for figures
        base_dir = Path(__file__).parent
        fig_dir = base_dir / 'figures' / 'storage'
        fig_dir.mkdir(parents=True, exist_ok=True)

        # Generate plot
        save_path = fig_dir / f'{dataset_id}_nyc_aggregate_storage.png'

        print("\nGenerating plot...")
        plot_nyc_aggregate_storage(
            storage_dict,
            title=f"NYC Aggregate Storage - {dataset_config['description']}",
            save_path=save_path
        )

        print("\n" + "=" * 80)
        print("SUCCESS: Plot generated successfully!")
        print("=" * 80)

        return True

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Plot NYC aggregate storage from synthetic ensemble simulations'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='stationary_ensemble',
        help='Dataset ID to plot (default: stationary_ensemble)'
    )

    args = parser.parse_args()

    success = main(dataset_id=args.dataset)

    sys.exit(0 if success else 1)
