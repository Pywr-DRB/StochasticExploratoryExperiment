#!/usr/bin/env python
"""
08_identify_episodes.py

Episode-Level Vulnerability Analysis for the Delaware River Basin.

This script identifies, characterizes, and classifies stress episodes
across the stochastic ensemble, determining which lead to cascading
failures vs. remaining contained.

Usage:
    python 08_identify_episodes.py [--dataset DATASET_ID]

Output:
    - Weekly time series with standardized variables
    - Episode database with features and cascade classification
    - Episode links (progression relationships)
    - Statistical comparison results
    - Visualization figures

Author: Auto-generated from episode_analysis_plan.md
"""

import argparse
import sys
from pathlib import Path

# Add methods to path
sys.path.insert(0, str(Path(__file__).parent))

from methods.config import DATASET_CONFIGS
from methods.postprocess import (
    load_episode_analysis_data,
    preprocess_to_weekly,
    compute_weekly_climatology,
    add_standardized_variables,
)
from methods.episode import (
    EpisodeAnalysisConfig,
    identify_all_episodes,
    characterize_episodes,
    link_episodes,
    compare_episode_populations,
    fit_cascade_model,
    save_episode_outputs,
)
from methods.episode.io import save_analysis_results
from methods.episode.linkage import compute_progression_rates
from methods.episode.characterization import compute_episode_summary_stats


def main():
    """Main entry point for episode analysis."""
    parser = argparse.ArgumentParser(
        description='Episode-level vulnerability analysis for Pywr-DRB'
    )
    parser.add_argument(
        '--dataset', type=str, default='stationary_ensemble',
        choices=list(DATASET_CONFIGS.keys()),
        help='Dataset to analyze'
    )
    parser.add_argument(
        '--output-dir', type=Path, default=None,
        help='Output directory (default: ./pywrdrb/episode_analysis)'
    )
    parser.add_argument(
        '--skip-plots', action='store_true',
        help='Skip generating plots'
    )
    args = parser.parse_args()

    # Initialize configuration
    config = EpisodeAnalysisConfig(dataset_id=args.dataset)
    if args.output_dir:
        config.output_dir = args.output_dir

    print("=" * 80)
    print("EPISODE-LEVEL VULNERABILITY ANALYSIS")
    print("=" * 80)
    print(f"Dataset: {config.dataset_id}")
    print(f"Output directory: {config.output_dir}")
    print()

    # =========================================================================
    # Step 1: Load Data
    # =========================================================================
    print("Step 1: Loading Pywr-DRB results...")
    try:
        data = load_episode_analysis_data(config.dataset_id)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("\nMake sure you have run 04_postprocess_data.py first!")
        return 1

    realizations = sorted(data.res_storage[config.dataset_id].keys())
    print(f"  Found {len(realizations)} realizations")
    print()

    # =========================================================================
    # Step 2: Preprocess to Weekly Resolution
    # =========================================================================
    print("Step 2: Preprocessing to weekly resolution...")
    weekly_ts = preprocess_to_weekly(data, config.dataset_id, config)
    print(f"  Weekly time series shape: {weekly_ts.shape}")
    print()

    # =========================================================================
    # Step 3: Compute Climatology and Standardize
    # =========================================================================
    print("Step 3: Computing climatology and standardizing variables...")
    climatology = compute_weekly_climatology(weekly_ts)
    weekly_ts = add_standardized_variables(weekly_ts, climatology)
    print()

    # =========================================================================
    # Step 4: Identify Episodes
    # =========================================================================
    print("Step 4: Identifying episodes...")
    episodes = identify_all_episodes(weekly_ts, config)
    print(f"  Total episodes identified: {len(episodes)}")

    # Print summary by type
    type_counts = episodes['episode_type'].value_counts()
    print("\n  Episodes by type:")
    for etype in ['E1', 'E1d', 'E1c', 'E2', 'E3', 'E4', 'E5']:
        count = type_counts.get(etype, 0)
        per_real = count / len(realizations)
        print(f"    {etype}: {count} ({per_real:.1f} per realization)")
    print()

    # =========================================================================
    # Step 5: Characterize Episodes
    # =========================================================================
    print("Step 5: Characterizing episodes...")
    episodes = characterize_episodes(episodes, weekly_ts, config)
    print()

    # =========================================================================
    # Step 6: Link Episodes and Classify Cascades
    # =========================================================================
    print("Step 6: Linking episodes and classifying cascades...")
    episode_links, episodes = link_episodes(episodes, config)

    # Print cascade summary
    stress_types = ['E1', 'E1d', 'E1c']
    stress_eps = episodes[episodes['episode_type'].isin(stress_types)]

    if len(stress_eps) > 0:
        cascade_counts = stress_eps['cascade_classification'].value_counts()
        print("\n  Cascade Classification Summary:")
        for cls in ['contained', 'partial_demand', 'partial_flow', 'cascade']:
            count = cascade_counts.get(cls, 0)
            pct = 100 * count / len(stress_eps)
            print(f"    {cls}: {count} ({pct:.1f}%)")

        # Progression rates by stress type
        print("\n  Progression Rates by Stress Type:")
        prog_rates = compute_progression_rates(episodes)
        for _, row in prog_rates.iterrows():
            print(f"    {row['stress_type']}: {row['n_episodes']} episodes, "
                  f"{row['rate_cascade']:.1f}% cascade, "
                  f"{row['rate_contained']:.1f}% contained")
    print()

    # =========================================================================
    # Step 7: Statistical Analysis
    # =========================================================================
    print("Step 7: Statistical analysis...")

    # Features to compare
    feature_cols = [
        'duration', 'storage_pct_onset', 'zone_onset', 'storage_trend_onset',
        'antecedent_deficit', 'inflow_severity', 'inflow_intensity',
        'demand_severity', 'demand_intensity', 'combined_stress_mean',
        'combined_stress_max', 'net_stress_cum', 'storage_pct_min',
        'zone_max', 'storage_drawdown', 'start_week_of_year'
    ]

    # Compare cascade vs. contained
    comparison_results = compare_episode_populations(
        episodes=stress_eps,
        group_col='cascade_classification',
        feature_cols=feature_cols,
        group_a='cascade',
        group_b='contained'
    )

    if len(comparison_results) > 0:
        # Print significant features
        sig_features = comparison_results[
            comparison_results['p_value_corrected'] < 0.05
        ].sort_values('effect_size_d', key=abs, ascending=False)

        print("\n  Significant discriminating features (p < 0.05):")
        for _, row in sig_features.head(10).iterrows():
            print(f"    {row['feature']}: d = {row['effect_size_d']:.2f}, "
                  f"p = {row['p_value_corrected']:.4f}")

    # Fit cascade probability model
    print("\n  Fitting cascade probability model...")
    model_features = [
        'storage_pct_onset', 'inflow_severity', 'demand_severity',
        'antecedent_deficit', 'storage_trend_onset', 'duration'
    ]
    cascade_model = fit_cascade_model(
        episodes=stress_eps,
        feature_cols=model_features,
        outcome_col='cascade_classification'
    )

    if 'error' in cascade_model:
        print(f"    Model fitting failed: {cascade_model['error']}")
    else:
        stats = cascade_model['model_fit_stats']
        print(f"    Model fit: pseudo-R2 = {stats['pseudo_r2']:.4f}, "
              f"AIC = {stats['aic']:.1f}")
        print(f"    N = {stats['n_observations']} "
              f"({stats['n_cascade']} cascade, {stats['n_non_cascade']} non-cascade)")
    print()

    # =========================================================================
    # Step 8: Save Outputs
    # =========================================================================
    print("Step 8: Saving outputs...")
    save_episode_outputs(weekly_ts, episodes, episode_links, climatology, config)
    save_analysis_results(comparison_results, cascade_model, config)
    print()

    # =========================================================================
    # Step 9: Generate Figures (optional)
    # =========================================================================
    if not args.skip_plots:
        print("Step 9: Generating figures...")
        try:
            from methods.plotting.episode import (
                create_sankey_diagram,
                create_feature_comparison_figure,
                create_episode_counts_by_type_figure,
                create_cascade_rate_histogram,
                create_stress_outcome_scatter,
                create_stress_outcome_heatmap,
            )

            fig_dir = config.output_dir / "figures"
            fig_dir.mkdir(exist_ok=True)

            # Sankey diagram
            create_sankey_diagram(
                episodes, config,
                save_path=fig_dir / f"sankey_progression.{config.figure_format}"
            )

            # Episode counts by type
            create_episode_counts_by_type_figure(
                episodes, config,
                save_path=fig_dir / f"episode_counts.{config.figure_format}"
            )

            # Feature comparison
            top_features = comparison_results.head(6)['feature'].tolist() if len(comparison_results) > 0 else feature_cols[:6]
            create_feature_comparison_figure(
                episodes, top_features, comparison_results, config,
                save_path=fig_dir / f"feature_comparison.{config.figure_format}"
            )

            # Cascade rate histogram
            create_cascade_rate_histogram(
                episodes, config,
                save_path=fig_dir / f"cascade_rate_histogram.{config.figure_format}"
            )

            # Stress-outcome scatter plot (inflow vs demand severity)
            create_stress_outcome_scatter(
                episodes, config,
                x_var='inflow_severity',
                y_var='demand_severity',
                size_var='duration',
                save_path=fig_dir / f"stress_outcome_scatter.{config.figure_format}"
            )

            # Stress-outcome heatmap (cascade rate across stress space)
            create_stress_outcome_heatmap(
                episodes, config,
                x_var='inflow_severity',
                y_var='demand_severity',
                save_path=fig_dir / f"stress_outcome_heatmap.{config.figure_format}"
            )

            # Additional scatter: storage at onset vs combined stress
            create_stress_outcome_scatter(
                episodes, config,
                x_var='storage_pct_onset',
                y_var='combined_stress_mean',
                size_var='duration',
                save_path=fig_dir / f"storage_stress_scatter.{config.figure_format}"
            )

            print(f"  Figures saved to: {fig_dir}")

        except ImportError as e:
            print(f"  Warning: Could not generate some plots ({e})")
        except Exception as e:
            print(f"  Warning: Error generating plots ({e})")
        print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 80)
    print("EPISODE ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  - {len(realizations)} realizations analyzed")
    print(f"  - {len(episodes)} total episodes identified")
    print(f"  - {len(stress_eps)} stress episodes (E1/E1d/E1c)")
    if len(stress_eps) > 0:
        n_cascade = (stress_eps['cascade_classification'] == 'cascade').sum()
        print(f"  - {n_cascade} cascade episodes ({100*n_cascade/len(stress_eps):.1f}%)")
    print(f"\nOutputs saved to: {config.output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
