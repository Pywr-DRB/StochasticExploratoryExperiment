"""
Episode linkage and cascade classification.

This module determines progression relationships between episodes and
classifies stress episodes as contained, partial, or cascade based on
whether they lead to outcome impacts.
"""

import numpy as np
import pandas as pd
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import EpisodeAnalysisConfig


def link_episodes(
    episodes: pd.DataFrame,
    config: 'EpisodeAnalysisConfig'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Determine progression relationships between episodes.

    For each stress episode (E1/E1d/E1c), finds outcome episodes (E2/E3/E4/E5)
    that start during or shortly after the stress episode.

    Parameters
    ----------
    episodes : pd.DataFrame
        All identified episodes with features
    config : EpisodeAnalysisConfig
        Configuration with progression_lag_window_weeks

    Returns
    -------
    episode_links : pd.DataFrame
        Parent-child relationships between episodes
    episodes : pd.DataFrame
        Episodes with added progression flags and cascade classification
    """
    if len(episodes) == 0:
        episodes_out = episodes.copy()
        for col in ['progressed_to_E2', 'progressed_to_E3', 'progressed_to_E4',
                    'progressed_to_E5', 'cascade_classification',
                    'lag_to_first_impact', 'lag_to_cascade']:
            episodes_out[col] = np.nan
        return pd.DataFrame(), episodes_out

    stress_types = ['E1', 'E1d', 'E1c']
    outcome_types = ['E2', 'E3', 'E4', 'E5']
    lag_window = config.progression_lag_window_weeks

    # Initialize progression flags
    episodes = episodes.copy()
    for ot in outcome_types:
        episodes[f'progressed_to_{ot}'] = False
    episodes['cascade_classification'] = 'contained'
    episodes['lag_to_first_impact'] = np.nan
    episodes['lag_to_cascade'] = np.nan

    # Build links list
    links = []

    # Get stress and outcome episodes
    stress_eps = episodes[episodes['episode_type'].isin(stress_types)]
    outcome_eps = episodes[episodes['episode_type'].isin(outcome_types)]

    if len(stress_eps) == 0:
        return pd.DataFrame(), episodes

    print(f"  Linking {len(stress_eps)} stress episodes to {len(outcome_eps)} outcome episodes...")

    for _, stress_ep in stress_eps.iterrows():
        r = stress_ep['realization_id']
        stress_start = stress_ep['start_week']
        stress_end = stress_ep['end_week']
        stress_id = stress_ep['episode_id']

        # Find outcome episodes in same realization that start during stress
        # or within lag_window after stress start
        candidate_outcomes = outcome_eps[
            (outcome_eps['realization_id'] == r) &
            (outcome_eps['start_week'] >= stress_start) &
            (outcome_eps['start_week'] <= stress_end + lag_window)
        ]

        found_e2 = False
        found_e3 = False
        found_e4 = False
        found_e5 = False
        first_impact_lag = np.inf
        cascade_lag = np.nan

        for _, out_ep in candidate_outcomes.iterrows():
            out_type = out_ep['episode_type']
            out_id = out_ep['episode_id']
            lag = out_ep['start_week'] - stress_start

            # Record link
            links.append({
                'parent_episode_id': stress_id,
                'child_episode_id': out_id,
                'parent_type': stress_ep['episode_type'],
                'child_type': out_type,
                'lag_weeks': lag,
                'realization_id': r
            })

            # Update progression flags
            episodes.loc[episodes['episode_id'] == stress_id, f'progressed_to_{out_type}'] = True

            # Track which outcomes were found
            if out_type == 'E2':
                found_e2 = True
            elif out_type == 'E3':
                found_e3 = True
                first_impact_lag = min(first_impact_lag, lag)
            elif out_type == 'E4':
                found_e4 = True
                first_impact_lag = min(first_impact_lag, lag)
            elif out_type == 'E5':
                found_e5 = True
                cascade_lag = lag

        # Set lag values
        if first_impact_lag < np.inf:
            episodes.loc[episodes['episode_id'] == stress_id, 'lag_to_first_impact'] = first_impact_lag
        if not np.isnan(cascade_lag):
            episodes.loc[episodes['episode_id'] == stress_id, 'lag_to_cascade'] = cascade_lag

        # Classify cascade status
        if found_e5 or (found_e3 and found_e4):
            classification = 'cascade'
        elif found_e3:
            classification = 'partial_demand'
        elif found_e4:
            classification = 'partial_flow'
        else:
            classification = 'contained'

        episodes.loc[episodes['episode_id'] == stress_id, 'cascade_classification'] = classification

    # Create episode links DataFrame
    episode_links = pd.DataFrame(links)

    # Print summary
    if len(stress_eps) > 0:
        cascade_counts = episodes[episodes['episode_type'].isin(stress_types)]['cascade_classification'].value_counts()
        print("    Cascade classification summary:")
        for cls in ['contained', 'partial_demand', 'partial_flow', 'cascade']:
            count = cascade_counts.get(cls, 0)
            pct = 100 * count / len(stress_eps)
            print(f"      {cls}: {count} ({pct:.1f}%)")

    return episode_links, episodes


def get_cascade_episodes(episodes: pd.DataFrame) -> pd.DataFrame:
    """
    Get stress episodes that resulted in cascades.

    Parameters
    ----------
    episodes : pd.DataFrame
        Episodes with cascade_classification column

    Returns
    -------
    cascade_eps : pd.DataFrame
        Stress episodes classified as cascades
    """
    stress_types = ['E1', 'E1d', 'E1c']
    return episodes[
        (episodes['episode_type'].isin(stress_types)) &
        (episodes['cascade_classification'] == 'cascade')
    ]


def get_contained_episodes(episodes: pd.DataFrame) -> pd.DataFrame:
    """
    Get stress episodes that remained contained (no outcome impacts).

    Parameters
    ----------
    episodes : pd.DataFrame
        Episodes with cascade_classification column

    Returns
    -------
    contained_eps : pd.DataFrame
        Stress episodes classified as contained
    """
    stress_types = ['E1', 'E1d', 'E1c']
    return episodes[
        (episodes['episode_type'].isin(stress_types)) &
        (episodes['cascade_classification'] == 'contained')
    ]


def compute_progression_rates(episodes: pd.DataFrame) -> pd.DataFrame:
    """
    Compute progression rates from stress to outcome episodes.

    Parameters
    ----------
    episodes : pd.DataFrame
        Episodes with progression flags

    Returns
    -------
    rates : pd.DataFrame
        Progression rates by stress episode type
    """
    stress_types = ['E1', 'E1d', 'E1c']
    results = []

    for stress_type in stress_types:
        type_eps = episodes[episodes['episode_type'] == stress_type]
        n_total = len(type_eps)

        if n_total == 0:
            continue

        # Count progressions to each outcome type
        results.append({
            'stress_type': stress_type,
            'n_episodes': n_total,
            'rate_to_E2': type_eps['progressed_to_E2'].mean() * 100,
            'rate_to_E3': type_eps['progressed_to_E3'].mean() * 100,
            'rate_to_E4': type_eps['progressed_to_E4'].mean() * 100,
            'rate_to_E5': type_eps['progressed_to_E5'].mean() * 100,
            'rate_cascade': (type_eps['cascade_classification'] == 'cascade').mean() * 100,
            'rate_partial': type_eps['cascade_classification'].isin(['partial_demand', 'partial_flow']).mean() * 100,
            'rate_contained': (type_eps['cascade_classification'] == 'contained').mean() * 100,
        })

    return pd.DataFrame(results)
