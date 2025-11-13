"""
Analyze metric sensitivity to climate change scenarios.

This standalone script identifies which performance metrics show the largest
changes relative to the stationary baseline, helping prioritize metrics for
analysis and visualization.

Usage:
    python analyze_metric_sensitivity.py
    python analyze_metric_sensitivity.py --n-top 20
    python analyze_metric_sensitivity.py --baseline stationary_ensemble
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from methods.config import *


# Performance metrics directory
PERFORMANCE_METRICS_DIR = f"{ROOT_DIR}/pywrdrb/performance_metrics"
OUTPUT_DIR = f"{ROOT_DIR}/figures/performance_metrics"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_performance_metrics(dataset_id):
    """Load pre-calculated performance metrics from CSV."""
    csv_file = f"{PERFORMANCE_METRICS_DIR}/{dataset_id}_performance_metrics.csv"

    if not os.path.exists(csv_file):
        raise FileNotFoundError(
            f"Performance metrics not found: {csv_file}\n"
            f"Run 04_postprocess_data.py first to calculate metrics!"
        )

    metrics_df = pd.read_csv(csv_file, index_col='realization_id')
    return metrics_df


def get_datasets_from_config():
    """Get list of datasets from config.py."""
    from methods.config import DATASET_CONFIGS
    all_datasets = list(DATASET_CONFIGS.keys())
    dataset_labels = {
        dataset_id: DATASET_CONFIGS[dataset_id].get('description', dataset_id)
        for dataset_id in all_datasets
    }
    return all_datasets, dataset_labels


def identify_top_changing_metrics(all_metrics_dfs, baseline_dataset, comparison_datasets,
                                  all_metrics=None, n_top=10):
    """
    Identify metrics with largest mean percentage changes relative to baseline.

    Parameters
    ----------
    all_metrics_dfs : dict
        Dictionary mapping dataset_id to metrics DataFrame
    baseline_dataset : str
        Dataset ID to use as baseline for comparison
    comparison_datasets : list
        List of dataset IDs to compare against baseline
    all_metrics : list, optional
        List of all available metrics to consider
    n_top : int, optional
        Number of top changing metrics to return

    Returns
    -------
    top_metrics_df : pd.DataFrame
        DataFrame with metric sensitivity analysis
    detailed_changes : dict
        Dictionary with per-dataset percentage changes for each metric
    """
    # Get all available metrics if not specified
    if all_metrics is None:
        all_metrics = [col for col in all_metrics_dfs[baseline_dataset].columns]

    # Calculate percentage changes for each metric
    metric_changes = {}
    detailed_changes = {}

    baseline_df = all_metrics_dfs[baseline_dataset]

    for metric in all_metrics:
        if metric not in baseline_df.columns:
            continue

        baseline_values = baseline_df[metric].values
        baseline_mean = baseline_values.mean()
        baseline_median = np.median(baseline_values)

        # Skip if baseline mean is essentially zero
        if abs(baseline_mean) < 1e-8:
            continue

        pct_changes = []
        dataset_changes = {}

        for comp_dataset in comparison_datasets:
            comp_df = all_metrics_dfs[comp_dataset]

            if metric not in comp_df.columns:
                continue

            comp_values = comp_df[metric].values
            comp_mean = comp_values.mean()
            comp_median = np.median(comp_values)

            # Calculate percentage change (mean-based)
            pct_change_mean = 100.0 * (comp_mean - baseline_mean) / abs(baseline_mean)
            pct_change_median = 100.0 * (comp_median - baseline_median) / abs(baseline_median)

            pct_changes.append(pct_change_mean)
            dataset_changes[comp_dataset] = {
                'mean_pct_change': pct_change_mean,
                'median_pct_change': pct_change_median
            }

        if pct_changes:
            mean_pct_change = np.mean(pct_changes)
            mean_abs_pct_change = np.mean([abs(pc) for pc in pct_changes])
            max_abs_pct_change = max([abs(pc) for pc in pct_changes])
            min_pct_change = min(pct_changes)
            max_pct_change = max(pct_changes)

            # Determine direction
            if all(pc >= 0 for pc in pct_changes):
                direction = 'increase'
            elif all(pc <= 0 for pc in pct_changes):
                direction = 'decrease'
            else:
                direction = 'mixed'

            metric_changes[metric] = {
                'mean_abs_pct_change': mean_abs_pct_change,
                'mean_pct_change': mean_pct_change,
                'min_pct_change': min_pct_change,
                'max_pct_change': max_pct_change,
                'max_abs_pct_change': max_abs_pct_change,
                'direction': direction,
                'baseline_mean': baseline_mean,
                'baseline_median': baseline_median
            }

            detailed_changes[metric] = dataset_changes

    # Convert to DataFrame and sort
    changes_df = pd.DataFrame(metric_changes).T
    changes_df.index.name = 'metric'
    changes_df = changes_df.reset_index()
    changes_df = changes_df.sort_values('mean_abs_pct_change', ascending=False)

    # Return top N
    top_metrics_df = changes_df.head(n_top)

    return top_metrics_df, detailed_changes


def print_top_changing_metrics(top_metrics_df, detailed_changes, dataset_labels, n_display=15):
    """Print formatted table of top changing metrics with details."""
    print(f"\n{'='*100}")
    print(f"TOP {min(n_display, len(top_metrics_df))} METRICS WITH LARGEST SENSITIVITY TO CLIMATE SCENARIOS")
    print(f"{'='*100}")
    print(f"{'Rank':<6}{'Metric':<40}{'Mean |Δ%|':<12}{'Range':<20}{'Dir':<8}{'Baseline':<12}")
    print(f"{'-'*100}")

    for i, row in enumerate(top_metrics_df.head(n_display).itertuples(index=False)):
        rank = i + 1
        metric = row.metric
        mean_abs_change = row.mean_abs_pct_change
        min_change = row.min_pct_change
        max_change = row.max_pct_change
        direction = row.direction
        baseline_mean = row.baseline_mean

        # Format direction indicator
        if direction == 'increase':
            sign = '↑'
        elif direction == 'decrease':
            sign = '↓'
        else:
            sign = '±'

        # Format range
        range_str = f"[{min_change:+.1f}%, {max_change:+.1f}%]"

        print(f"{rank:<6}{metric:<40}{mean_abs_change:>10.1f}%  {range_str:<20}{sign} {direction:<6}{baseline_mean:>10.2f}")

        # Print per-dataset details if available
        if metric in detailed_changes:
            for dataset_id, changes in detailed_changes[metric].items():
                label = dataset_labels.get(dataset_id, dataset_id)
                mean_pct = changes['mean_pct_change']
                print(f"{'':6}  └─ {label:<50}{mean_pct:>+10.1f}%")

    print(f"{'='*100}\n")


def create_sensitivity_heatmap(top_metrics_df, detailed_changes, dataset_labels, output_file):
    """
    Create a heatmap showing metric sensitivity across scenarios.

    Parameters
    ----------
    top_metrics_df : pd.DataFrame
        Top changing metrics
    detailed_changes : dict
        Detailed changes per metric per dataset
    dataset_labels : dict
        Display labels for datasets
    output_file : str
        Output file path for heatmap
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Prepare data for heatmap
    metrics = top_metrics_df['metric'].tolist()
    datasets = sorted([d for d in detailed_changes[metrics[0]].keys()])

    # Create matrix of percentage changes
    data_matrix = []
    for metric in metrics:
        row = []
        for dataset in datasets:
            if metric in detailed_changes and dataset in detailed_changes[metric]:
                pct_change = detailed_changes[metric][dataset]['mean_pct_change']
                row.append(pct_change)
            else:
                row.append(np.nan)
        data_matrix.append(row)

    # Create DataFrame
    heatmap_df = pd.DataFrame(
        data_matrix,
        index=metrics,
        columns=[dataset_labels.get(d, d) for d in datasets]
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(10, max(8, len(metrics) * 0.4)))

    # Create heatmap
    sns.heatmap(
        heatmap_df,
        cmap='RdBu_r',
        center=0,
        annot=True,
        fmt='.1f',
        cbar_kws={'label': '% Change from Baseline'},
        linewidths=0.5,
        ax=ax
    )

    ax.set_xlabel('Climate Scenario', fontsize=12, fontweight='bold')
    ax.set_ylabel('Performance Metric', fontsize=12, fontweight='bold')
    ax.set_title('Metric Sensitivity to Climate Change\n(% change from stationary baseline)',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved sensitivity heatmap: {output_file}")

    return fig, ax


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze metric sensitivity to climate change scenarios'
    )
    parser.add_argument(
        '--baseline',
        type=str,
        default='stationary_ensemble',
        help='Baseline dataset for comparison (default: stationary_ensemble)'
    )
    parser.add_argument(
        '--n-top',
        type=int,
        default=15,
        help='Number of top changing metrics to display (default: 15)'
    )
    parser.add_argument(
        '--heatmap',
        action='store_true',
        help='Create sensitivity heatmap visualization'
    )

    args = parser.parse_args()

    print("="*100)
    print("PERFORMANCE METRIC SENSITIVITY ANALYSIS")
    print("="*100)
    print(f"Baseline dataset: {args.baseline}")
    print(f"Top N metrics to display: {args.n_top}")
    print("="*100)

    # Get datasets from config
    all_datasets, dataset_labels = get_datasets_from_config()

    if args.baseline not in all_datasets:
        print(f"ERROR: Baseline dataset '{args.baseline}' not found in config.py!")
        print(f"Available datasets: {all_datasets}")
        return 1

    # Determine comparison datasets
    comparison_datasets = [d for d in all_datasets if d != args.baseline]

    print(f"\nComparison datasets ({len(comparison_datasets)}):")
    for dataset in comparison_datasets:
        print(f"  - {dataset}: {dataset_labels[dataset]}")

    # Load all metrics
    print(f"\n{'='*100}")
    print("LOADING PERFORMANCE METRICS")
    print(f"{'='*100}")

    all_metrics_dfs = {}

    for dataset_id in all_datasets:
        try:
            print(f"Loading {dataset_id}...")
            metrics_df = load_performance_metrics(dataset_id)
            all_metrics_dfs[dataset_id] = metrics_df
            print(f"  Loaded {len(metrics_df)} realizations × {len(metrics_df.columns)} metrics")
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")
            continue

    if args.baseline not in all_metrics_dfs:
        print(f"\nERROR: Could not load baseline dataset '{args.baseline}'")
        return 1

    # Identify top changing metrics
    print(f"\n{'='*100}")
    print("ANALYZING METRIC SENSITIVITY")
    print(f"{'='*100}")

    all_available_metrics = list(all_metrics_dfs[args.baseline].columns)

    top_metrics_df, detailed_changes = identify_top_changing_metrics(
        all_metrics_dfs,
        args.baseline,
        comparison_datasets,
        all_metrics=all_available_metrics,
        n_top=args.n_top
    )

    # Print results
    print_top_changing_metrics(
        top_metrics_df,
        detailed_changes,
        dataset_labels,
        n_display=args.n_top
    )

    # Save to CSV
    output_csv = f"{OUTPUT_DIR}/metric_sensitivity_analysis.csv"
    top_metrics_df.to_csv(output_csv, index=False)
    print(f"Saved detailed analysis to: {output_csv}")

    # Create detailed CSV with per-dataset changes
    detailed_rows = []
    for metric in top_metrics_df['metric']:
        if metric in detailed_changes:
            base_info = top_metrics_df[top_metrics_df['metric'] == metric].iloc[0]
            for dataset_id, changes in detailed_changes[metric].items():
                row = {
                    'metric': metric,
                    'dataset': dataset_id,
                    'dataset_label': dataset_labels.get(dataset_id, dataset_id),
                    'mean_pct_change': changes['mean_pct_change'],
                    'median_pct_change': changes['median_pct_change'],
                    'baseline_mean': base_info['baseline_mean'],
                    'direction': base_info['direction']
                }
                detailed_rows.append(row)

    detailed_df = pd.DataFrame(detailed_rows)
    detailed_csv = f"{OUTPUT_DIR}/metric_sensitivity_detailed.csv"
    detailed_df.to_csv(detailed_csv, index=False)
    print(f"Saved per-dataset details to: {detailed_csv}")

    # Create heatmap if requested
    if args.heatmap:
        print(f"\n{'='*100}")
        print("CREATING SENSITIVITY HEATMAP")
        print(f"{'='*100}")

        heatmap_file = f"{OUTPUT_DIR}/metric_sensitivity_heatmap.png"
        create_sensitivity_heatmap(
            top_metrics_df,
            detailed_changes,
            dataset_labels,
            heatmap_file
        )

    print(f"\n{'='*100}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*100}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
