"""
Plot Montague shortage from synthetic ensemble simulations.

This script loads Pywr simulation results and plots Montague shortage
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


def load_montague_data_from_simulation(dataset_id, ensemble_set_specs):
    """
    Load Montague shortage, flow, and target from Pywr simulation output using pywrdrb.Data().

    Parameters
    ----------
    dataset_id : str
        Dataset identifier (e.g., 'stationary_ensemble')
    ensemble_set_specs : list
        List of ensemble set specifications from config

    Returns
    -------
    tuple of dicts
        (shortage_dict, flow_dict, target_dict) mapping realization IDs to Series
    """
    print("  Setting up path navigator...")
    # Setup pathnavigator (required for pywrdrb.Data)
    pn_config = pywrdrb.get_pn_config()
    for spec in ensemble_set_specs:
        dataset_dir = spec.directory
        dataset_name = spec.directory.split('/')[-1]
        pn_config[f"flows/{dataset_name}"] = os.path.abspath(dataset_dir)
    pywrdrb.load_pn_config(pn_config)

    print("  Loading postprocessed data with shortage calculations...")
    # Load the postprocessed data file that includes pre-calculated shortage, flow, and targets
    fname = f'./pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5'

    if not os.path.exists(fname):
        raise FileNotFoundError(
            f"Postprocessed data file not found: {fname}\n"
            f"Please run 04_postprocess_data.py first to calculate shortages."
        )

    # Load shortage, major_flow, and mrf_target data
    data = Data()
    data.load_from_export(fname, results_sets=['shortage', 'major_flow', 'mrf_target'])

    # Extract Montague data for all realizations
    shortage_dict = {}
    flow_dict = {}
    target_dict = {}

    if dataset_id not in data.shortage:
        raise ValueError(f"Dataset '{dataset_id}' not found in shortage data. "
                        f"Available: {list(data.shortage.keys())}")

    print(f"\n  Processing dataset: {dataset_id}")
    realizations = list(data.shortage[dataset_id].keys())
    print(f"  Found {len(realizations)} realizations")

    for r in realizations:
        # Get shortage
        shortage_df = data.shortage[dataset_id][r]
        if 'delMontague' not in shortage_df.columns:
            raise ValueError(f"'delMontague' column not found in shortage data. "
                           f"Available columns: {shortage_df.columns.tolist()}")
        shortage_dict[r] = shortage_df['delMontague']

        # Get flow
        flow_df = data.major_flow[dataset_id][r]
        if 'delMontague' not in flow_df.columns:
            raise ValueError(f"'delMontague' column not found in major_flow data. "
                           f"Available columns: {flow_df.columns.tolist()}")
        flow_dict[r] = flow_df['delMontague']

        # Get target
        target_df = data.mrf_target[dataset_id][r]
        if 'delMontague' not in target_df.columns:
            raise ValueError(f"'delMontague' column not found in mrf_target data. "
                           f"Available columns: {target_df.columns.tolist()}")
        target_dict[r] = target_df['delMontague']

    print(f"\n  Successfully loaded {len(shortage_dict)} realizations")
    return shortage_dict, flow_dict, target_dict


def plot_montague_shortage(shortage_dict, flow_dict, target_dict,
                           title="Montague Flow & Shortage - Synthetic Ensemble",
                           figsize=(14, 10),
                           save_path=None):
    """
    Plot Montague flow vs target and shortage ensemble traces in two panels.

    Parameters
    ----------
    shortage_dict : dict
        Dictionary mapping realization IDs to shortage Series
    flow_dict : dict
        Dictionary mapping realization IDs to flow Series
    target_dict : dict
        Dictionary mapping realization IDs to target Series
    title : str
        Plot title
    figsize : tuple
        Figure size (width, height)
    save_path : str or Path, optional
        Path to save figure. If None, displays interactively.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Convert to DataFrames for easier manipulation
    flow_df = pd.DataFrame(flow_dict)
    target_df = pd.DataFrame(target_dict)
    shortage_df = pd.DataFrame(shortage_dict)

    # === Panel 1: Flow vs Target ===
    # Plot all flow realizations
    for real_id in flow_df.columns:
        ax1.plot(flow_df.index, flow_df[real_id],
                color='steelblue', alpha=0.3, linewidth=0.8, zorder=1,
                label='Flow' if real_id == flow_df.columns[0] else '')

    # Plot all target realizations
    for real_id in target_df.columns:
        ax1.plot(target_df.index, target_df[real_id],
                color='darkred', alpha=0.3, linewidth=0.8, zorder=2,
                label='Target' if real_id == target_df.columns[0] else '')

    ax1.set_ylabel('Flow / Target (MGD)', fontsize=12)
    ax1.set_title('Montague Flow vs Target', fontsize=12, fontweight='bold')
    # ax1.set_ylim(bottom=1)
    ax1.grid(True, alpha=0.3)

    # Add legend (only showing one trace for each)
    handles, labels = ax1.get_legend_handles_labels()
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)
    ax1.legend(unique_handles, unique_labels, loc='upper right', fontsize=9)

    # Calculate flow statistics
    flow_min = flow_df.min(axis=1)
    flow_max = flow_df.max(axis=1)
    target_mean = target_df.mean(axis=1).mean()

    stats_text1 = f"Realizations: {len(flow_dict)}\n"
    stats_text1 += f"Flow Range: {flow_min.min():.0f} - {flow_max.max():.0f} MGD\n"
    stats_text1 += f"Mean Target: {target_mean:.0f} MGD"

    ax1.text(0.02, 0.98, stats_text1, transform=ax1.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=9)
    
    ax1.set_yscale('log')

    # === Panel 2: Shortage ===
    # Plot all individual realizations
    for real_id in shortage_df.columns:
        ax2.plot(shortage_df.index, shortage_df[real_id],
                color='firebrick', alpha=0.3, linewidth=0.8, zorder=1)

    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Shortage (MGD)', fontsize=12)
    ax2.set_title('Montague Shortage', fontsize=12, fontweight='bold')
    ax2.set_ylim(bottom=0)  # Shortage cannot be negative
    ax2.grid(True, alpha=0.3)

    # Calculate shortage statistics
    total_shortage_days = (shortage_df > 0).sum(axis=0)  # Days with shortage per realization
    pct_days_with_shortage = 100.0 * total_shortage_days / len(shortage_df)
    ensemble_max = shortage_df.max(axis=1)

    stats_text2 = f"Realizations: {len(shortage_dict)}\n"
    stats_text2 += f"Days with Shortage: {pct_days_with_shortage.mean():.1f}% (mean)\n"
    stats_text2 += f"Max Shortage: {ensemble_max.max():.0f} MGD\n"
    stats_text2 += f"Mean Shortage (when >0): {shortage_df[shortage_df > 0].mean().mean():.0f} MGD"

    ax2.text(0.02, 0.98, stats_text2, transform=ax2.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=9)

    # Overall title
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
        plt.close(fig)
    else:
        plt.show()

    return fig, (ax1, ax2)


def main(dataset_id='stationary_ensemble'):
    """
    Main function to load and plot Montague shortage.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier (e.g., 'stationary_ensemble', 'climate_adjusted_low')
    """
    print("=" * 80)
    print("MONTAGUE SHORTAGE PLOTTING")
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
        # Load Montague data from postprocessed file
        print("\nLoading postprocessed data...")
        shortage_dict, flow_dict, target_dict = load_montague_data_from_simulation(
            dataset_id, ensemble_set_specs
        )
        print(f"\nSuccessfully loaded data for {len(shortage_dict)} realizations")

        # Create output directory for figures
        base_dir = Path(__file__).parent
        fig_dir = base_dir / 'figures' / 'shortage'
        fig_dir.mkdir(parents=True, exist_ok=True)

        # Generate plot
        save_path = fig_dir / f'{dataset_id}_montague_shortage.png'

        print("\nGenerating plot...")
        plot_montague_shortage(
            shortage_dict, flow_dict, target_dict,
            title=f"Montague Flow & Shortage - {dataset_config['description']}",
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
        description='Plot Montague shortage from synthetic ensemble simulations'
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
