"""
Episode definition validation and coverage analysis.

This module provides functions to validate that episode definitions properly
capture consequential outcomes in the ensemble simulations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import EpisodeAnalysisConfig


def identify_uncaptured_outcomes(
    weekly_ts: pd.DataFrame,
    episodes: pd.DataFrame,
    config: 'EpisodeAnalysisConfig'
) -> Dict[str, pd.DataFrame]:
    """
    Identify outcome events not captured by any episode definition.

    This function checks for:
    1. Demand shortages (satisfaction < tolerance) not within any E3/E5 episode
    2. Flow shortages (satisfaction < tolerance) not within any E4/E5 episode
    3. Low storage periods (storage_pct below thresholds) not within any E2 episode
    4. Zone transitions not linked to any stress episode

    Parameters
    ----------
    weekly_ts : pd.DataFrame
        Weekly time series with satisfaction and storage variables
    episodes : pd.DataFrame
        Identified episodes with episode_type and timing
    config : EpisodeAnalysisConfig
        Configuration with thresholds

    Returns
    -------
    uncaptured : Dict[str, pd.DataFrame]
        Dictionary with keys 'demand_shortages', 'flow_shortages', 'low_storage',
        'orphan_zones' containing DataFrames of uncaptured events
    """
    results = {}

    # Get tolerance threshold
    tol = config.satisfaction_tolerance

    # =========================================================================
    # 1. Uncaptured Demand Shortages
    # =========================================================================
    # Find all weeks with demand shortage
    demand_shortage_weeks = weekly_ts[
        weekly_ts['demand_satisfaction'] < tol
    ][['realization_id', 'week', 'date', 'demand_satisfaction']].copy()

    if len(demand_shortage_weeks) > 0:
        # Get E3 and E5 episodes (those that track demand shortages)
        demand_episodes = episodes[episodes['episode_type'].isin(['E3', 'E5'])]

        # Mark weeks covered by episodes
        demand_shortage_weeks['captured'] = False

        for _, ep in demand_episodes.iterrows():
            mask = (
                (demand_shortage_weeks['realization_id'] == ep['realization_id']) &
                (demand_shortage_weeks['week'] >= ep['start_week']) &
                (demand_shortage_weeks['week'] <= ep['end_week'])
            )
            demand_shortage_weeks.loc[mask, 'captured'] = True

        uncaptured_demand = demand_shortage_weeks[~demand_shortage_weeks['captured']].copy()
        uncaptured_demand['shortage_magnitude'] = 1.0 - uncaptured_demand['demand_satisfaction']
    else:
        uncaptured_demand = pd.DataFrame()

    results['demand_shortages'] = uncaptured_demand

    # =========================================================================
    # 2. Uncaptured Flow Shortages
    # =========================================================================
    flow_shortage_weeks = weekly_ts[
        weekly_ts['flow_satisfaction'] < tol
    ][['realization_id', 'week', 'date', 'flow_satisfaction']].copy()

    if len(flow_shortage_weeks) > 0:
        flow_episodes = episodes[episodes['episode_type'].isin(['E4', 'E5'])]

        flow_shortage_weeks['captured'] = False

        for _, ep in flow_episodes.iterrows():
            mask = (
                (flow_shortage_weeks['realization_id'] == ep['realization_id']) &
                (flow_shortage_weeks['week'] >= ep['start_week']) &
                (flow_shortage_weeks['week'] <= ep['end_week'])
            )
            flow_shortage_weeks.loc[mask, 'captured'] = True

        uncaptured_flow = flow_shortage_weeks[~flow_shortage_weeks['captured']].copy()
        uncaptured_flow['shortage_magnitude'] = 1.0 - uncaptured_flow['flow_satisfaction']
    else:
        uncaptured_flow = pd.DataFrame()

    results['flow_shortages'] = uncaptured_flow

    # =========================================================================
    # 3. Uncaptured Low Storage Periods
    # =========================================================================
    # Define low storage as below 50% capacity (significant drawdown)
    low_storage_threshold = 0.50

    low_storage_weeks = weekly_ts[
        weekly_ts['storage_pct'] < low_storage_threshold
    ][['realization_id', 'week', 'date', 'storage_pct', 'ffmp_zone']].copy()

    if len(low_storage_weeks) > 0:
        # Check if covered by E2 (zone transition) episodes
        zone_episodes = episodes[episodes['episode_type'] == 'E2']

        low_storage_weeks['captured_by_E2'] = False

        for _, ep in zone_episodes.iterrows():
            mask = (
                (low_storage_weeks['realization_id'] == ep['realization_id']) &
                (low_storage_weeks['week'] >= ep['start_week']) &
                (low_storage_weeks['week'] <= ep['end_week'])
            )
            low_storage_weeks.loc[mask, 'captured_by_E2'] = True

        uncaptured_storage = low_storage_weeks[~low_storage_weeks['captured_by_E2']].copy()
    else:
        uncaptured_storage = pd.DataFrame()

    results['low_storage'] = uncaptured_storage

    # =========================================================================
    # 4. Orphan Zone Transitions (E2 not linked to any stress E1/E1d/E1c)
    # =========================================================================
    zone_episodes = episodes[episodes['episode_type'] == 'E2'].copy()
    stress_episodes = episodes[episodes['episode_type'].isin(['E1', 'E1d', 'E1c'])]

    if len(zone_episodes) > 0 and len(stress_episodes) > 0:
        zone_episodes['has_linked_stress'] = False
        lag_window = config.progression_lag_window_weeks

        for idx, zone_ep in zone_episodes.iterrows():
            r = zone_ep['realization_id']
            zone_start = zone_ep['start_week']

            # Check for stress episodes that could have caused this zone transition
            linked = stress_episodes[
                (stress_episodes['realization_id'] == r) &
                (stress_episodes['start_week'] <= zone_start) &
                (stress_episodes['end_week'] + lag_window >= zone_start)
            ]

            if len(linked) > 0:
                zone_episodes.loc[idx, 'has_linked_stress'] = True

        orphan_zones = zone_episodes[~zone_episodes['has_linked_stress']]
    else:
        orphan_zones = zone_episodes.copy() if len(zone_episodes) > 0 else pd.DataFrame()

    results['orphan_zones'] = orphan_zones

    return results


def compute_coverage_statistics(
    weekly_ts: pd.DataFrame,
    episodes: pd.DataFrame,
    uncaptured: Dict[str, pd.DataFrame],
    config: 'EpisodeAnalysisConfig'
) -> pd.DataFrame:
    """
    Compute coverage statistics for episode definitions.

    Parameters
    ----------
    weekly_ts : pd.DataFrame
        Weekly time series
    episodes : pd.DataFrame
        Identified episodes
    uncaptured : Dict[str, pd.DataFrame]
        Output from identify_uncaptured_outcomes()
    config : EpisodeAnalysisConfig
        Configuration

    Returns
    -------
    stats : pd.DataFrame
        Coverage statistics
    """
    tol = config.satisfaction_tolerance
    n_realizations = weekly_ts['realization_id'].nunique()

    stats = []

    # Demand shortage coverage
    total_demand_shortage_weeks = (weekly_ts['demand_satisfaction'] < tol).sum()
    uncaptured_demand_weeks = len(uncaptured['demand_shortages'])
    captured_demand_weeks = total_demand_shortage_weeks - uncaptured_demand_weeks

    stats.append({
        'outcome_type': 'Demand Shortage',
        'total_events': total_demand_shortage_weeks,
        'captured_events': captured_demand_weeks,
        'uncaptured_events': uncaptured_demand_weeks,
        'capture_rate': 100 * captured_demand_weeks / max(total_demand_shortage_weeks, 1),
        'events_per_realization': total_demand_shortage_weeks / n_realizations,
        'uncaptured_per_realization': uncaptured_demand_weeks / n_realizations
    })

    # Flow shortage coverage
    total_flow_shortage_weeks = (weekly_ts['flow_satisfaction'] < tol).sum()
    uncaptured_flow_weeks = len(uncaptured['flow_shortages'])
    captured_flow_weeks = total_flow_shortage_weeks - uncaptured_flow_weeks

    stats.append({
        'outcome_type': 'Flow Shortage',
        'total_events': total_flow_shortage_weeks,
        'captured_events': captured_flow_weeks,
        'uncaptured_events': uncaptured_flow_weeks,
        'capture_rate': 100 * captured_flow_weeks / max(total_flow_shortage_weeks, 1),
        'events_per_realization': total_flow_shortage_weeks / n_realizations,
        'uncaptured_per_realization': uncaptured_flow_weeks / n_realizations
    })

    # Low storage coverage
    total_low_storage_weeks = (weekly_ts['storage_pct'] < 0.50).sum()
    uncaptured_storage_weeks = len(uncaptured['low_storage'])
    captured_storage_weeks = total_low_storage_weeks - uncaptured_storage_weeks

    stats.append({
        'outcome_type': 'Low Storage (<50%)',
        'total_events': total_low_storage_weeks,
        'captured_events': captured_storage_weeks,
        'uncaptured_events': uncaptured_storage_weeks,
        'capture_rate': 100 * captured_storage_weeks / max(total_low_storage_weeks, 1),
        'events_per_realization': total_low_storage_weeks / n_realizations,
        'uncaptured_per_realization': uncaptured_storage_weeks / n_realizations
    })

    # Zone transition linkage
    total_zone_episodes = len(episodes[episodes['episode_type'] == 'E2'])
    orphan_zone_episodes = len(uncaptured['orphan_zones'])
    linked_zone_episodes = total_zone_episodes - orphan_zone_episodes

    stats.append({
        'outcome_type': 'Zone Transitions (E2)',
        'total_events': total_zone_episodes,
        'captured_events': linked_zone_episodes,
        'uncaptured_events': orphan_zone_episodes,
        'capture_rate': 100 * linked_zone_episodes / max(total_zone_episodes, 1),
        'events_per_realization': total_zone_episodes / n_realizations,
        'uncaptured_per_realization': orphan_zone_episodes / n_realizations
    })

    return pd.DataFrame(stats)


def analyze_uncaptured_patterns(
    uncaptured: Dict[str, pd.DataFrame],
    weekly_ts: pd.DataFrame,
    config: 'EpisodeAnalysisConfig'
) -> Dict[str, pd.DataFrame]:
    """
    Analyze patterns in uncaptured outcomes to identify potential gaps.

    Parameters
    ----------
    uncaptured : Dict[str, pd.DataFrame]
        Output from identify_uncaptured_outcomes()
    weekly_ts : pd.DataFrame
        Weekly time series
    config : EpisodeAnalysisConfig
        Configuration

    Returns
    -------
    patterns : Dict[str, pd.DataFrame]
        Analysis of uncaptured event patterns
    """
    patterns = {}

    # Analyze uncaptured demand shortages
    if len(uncaptured['demand_shortages']) > 0:
        df = uncaptured['demand_shortages'].copy()

        # Merge with weekly_ts to get context
        df = df.merge(
            weekly_ts[['realization_id', 'week', 'inflow_std', 'demand_std',
                       'storage_pct', 'ffmp_zone']],
            on=['realization_id', 'week'],
            how='left'
        )

        # Categorize by context
        df['context'] = 'unknown'
        df.loc[df['inflow_std'] < config.inflow_stress_threshold, 'context'] = 'during_inflow_stress'
        df.loc[df['demand_std'] > config.demand_stress_threshold, 'context'] = 'during_demand_stress'
        df.loc[df['storage_pct'] < 0.5, 'context'] = 'low_storage'
        df.loc[df['ffmp_zone'] > config.baseline_ffmp_zone, 'context'] = 'elevated_zone'

        # Summary by context
        context_summary = df.groupby('context').agg({
            'realization_id': 'count',
            'shortage_magnitude': ['mean', 'max']
        }).round(4)
        context_summary.columns = ['n_events', 'mean_magnitude', 'max_magnitude']

        patterns['demand_shortage_contexts'] = context_summary.reset_index()

    # Analyze uncaptured flow shortages
    if len(uncaptured['flow_shortages']) > 0:
        df = uncaptured['flow_shortages'].copy()

        df = df.merge(
            weekly_ts[['realization_id', 'week', 'inflow_std', 'demand_std',
                       'storage_pct', 'ffmp_zone']],
            on=['realization_id', 'week'],
            how='left'
        )

        df['context'] = 'unknown'
        df.loc[df['inflow_std'] < config.inflow_stress_threshold, 'context'] = 'during_inflow_stress'
        df.loc[df['storage_pct'] < 0.5, 'context'] = 'low_storage'
        df.loc[df['ffmp_zone'] > config.baseline_ffmp_zone, 'context'] = 'elevated_zone'

        context_summary = df.groupby('context').agg({
            'realization_id': 'count',
            'shortage_magnitude': ['mean', 'max']
        }).round(4)
        context_summary.columns = ['n_events', 'mean_magnitude', 'max_magnitude']

        patterns['flow_shortage_contexts'] = context_summary.reset_index()

    return patterns


def validate_episode_definitions(
    weekly_ts: pd.DataFrame,
    episodes: pd.DataFrame,
    config: 'EpisodeAnalysisConfig',
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Run complete validation of episode definitions.

    This is the main validation entry point that:
    1. Identifies uncaptured outcomes
    2. Computes coverage statistics
    3. Analyzes patterns in gaps
    4. Prints summary report

    Parameters
    ----------
    weekly_ts : pd.DataFrame
        Weekly time series
    episodes : pd.DataFrame
        Identified episodes
    config : EpisodeAnalysisConfig
        Configuration
    verbose : bool
        Whether to print summary report

    Returns
    -------
    coverage_stats : pd.DataFrame
        Coverage statistics
    uncaptured : Dict[str, pd.DataFrame]
        Uncaptured events
    patterns : Dict[str, pd.DataFrame]
        Pattern analysis
    """
    # Run validation
    uncaptured = identify_uncaptured_outcomes(weekly_ts, episodes, config)
    coverage_stats = compute_coverage_statistics(weekly_ts, episodes, uncaptured, config)
    patterns = analyze_uncaptured_patterns(uncaptured, weekly_ts, config)

    if verbose:
        print("\n" + "=" * 80)
        print("EPISODE DEFINITION VALIDATION REPORT")
        print("=" * 80)

        print("\n--- Coverage Statistics ---\n")
        for _, row in coverage_stats.iterrows():
            print(f"{row['outcome_type']}:")
            print(f"  Total events: {int(row['total_events'])} ({row['events_per_realization']:.1f} per realization)")
            print(f"  Captured: {int(row['captured_events'])} ({row['capture_rate']:.1f}%)")
            print(f"  Uncaptured: {int(row['uncaptured_events'])} ({row['uncaptured_per_realization']:.2f} per realization)")
            print()

        # Warnings for low capture rates
        low_capture = coverage_stats[coverage_stats['capture_rate'] < 90]
        if len(low_capture) > 0:
            print("--- WARNINGS ---")
            for _, row in low_capture.iterrows():
                print(f"  LOW CAPTURE RATE: {row['outcome_type']} at {row['capture_rate']:.1f}%")
                print(f"    Consider adjusting episode thresholds or duration requirements")
            print()

        # Pattern analysis
        if patterns:
            print("--- Uncaptured Event Contexts ---\n")
            for key, df in patterns.items():
                if len(df) > 0:
                    print(f"{key}:")
                    print(df.to_string(index=False))
                    print()

        print("=" * 80)

    return coverage_stats, uncaptured, patterns


def suggest_threshold_adjustments(
    coverage_stats: pd.DataFrame,
    patterns: Dict[str, pd.DataFrame],
    config: 'EpisodeAnalysisConfig'
) -> List[str]:
    """
    Suggest potential threshold adjustments based on validation results.

    Parameters
    ----------
    coverage_stats : pd.DataFrame
        Coverage statistics
    patterns : Dict[str, pd.DataFrame]
        Pattern analysis
    config : EpisodeAnalysisConfig
        Current configuration

    Returns
    -------
    suggestions : List[str]
        List of suggested adjustments
    """
    suggestions = []

    # Check demand shortage coverage
    demand_row = coverage_stats[coverage_stats['outcome_type'] == 'Demand Shortage']
    if len(demand_row) > 0 and demand_row.iloc[0]['capture_rate'] < 95:
        if 'demand_shortage_contexts' in patterns:
            contexts = patterns['demand_shortage_contexts']
            if 'during_inflow_stress' in contexts['context'].values:
                suggestions.append(
                    f"Consider extending progression_lag_window_weeks beyond {config.progression_lag_window_weeks} "
                    "to capture delayed demand impacts from inflow stress"
                )
            if 'unknown' in contexts['context'].values:
                unknown_row = contexts[contexts['context'] == 'unknown']
                if len(unknown_row) > 0 and unknown_row.iloc[0]['n_events'] > 10:
                    suggestions.append(
                        "Some demand shortages occur outside identified stress contexts. "
                        "Consider adding additional stress indicators or reducing min_episode_duration_weeks "
                        f"(currently {config.min_episode_duration_weeks})"
                    )

    # Check flow shortage coverage
    flow_row = coverage_stats[coverage_stats['outcome_type'] == 'Flow Shortage']
    if len(flow_row) > 0 and flow_row.iloc[0]['capture_rate'] < 95:
        suggestions.append(
            "Flow shortage capture rate below 95%. "
            "Consider whether flow shortages at Montague/Trenton have different timing "
            "than NYC-focused stress episodes due to travel time lags (3-5 days)."
        )

    # Check zone transition linkage
    zone_row = coverage_stats[coverage_stats['outcome_type'] == 'Zone Transitions (E2)']
    if len(zone_row) > 0 and zone_row.iloc[0]['capture_rate'] < 90:
        suggestions.append(
            f"Many zone transitions (E2) are not linked to stress episodes. "
            f"Consider increasing progression_lag_window_weeks beyond {config.progression_lag_window_weeks} "
            "or investigating whether zone transitions are driven by factors other than "
            "inflow/demand stress (e.g., seasonal storage patterns)."
        )

    return suggestions
