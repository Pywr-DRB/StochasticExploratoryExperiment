"""
Core episode identification algorithms.

This module provides the main episode detection logic using a state machine
approach with configurable thresholds and exit criteria.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional

# Try to import numba for JIT compilation, fall back to pure Python if unavailable
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def jit(*args, **kwargs):
        """No-op decorator when numba is not available."""
        def decorator(func):
            return func
        return decorator


@jit(nopython=True)
def _identify_episodes_numba(
    values: np.ndarray,
    threshold: float,
    threshold_direction: int,
    exit_persistence: int,
    min_duration: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    JIT-compiled episode identification for a single time series.

    Parameters
    ----------
    values : np.ndarray
        Time series values to analyze
    threshold : float
        Threshold value for condition
    threshold_direction : int
        -1 for "below threshold", +1 for "above threshold"
    exit_persistence : int
        Number of consecutive timesteps not meeting condition to end episode
    min_duration : int
        Minimum episode duration to include

    Returns
    -------
    starts : np.ndarray
        Episode start indices
    ends : np.ndarray
        Episode end indices
    durations : np.ndarray
        Episode durations
    """
    n = len(values)
    max_episodes = n // 2 + 1

    starts = np.zeros(max_episodes, dtype=np.int64)
    ends = np.zeros(max_episodes, dtype=np.int64)
    durations = np.zeros(max_episodes, dtype=np.int64)

    n_episodes = 0
    in_episode = False
    episode_start = 0
    exit_counter = 0

    for i in range(n):
        # Check if condition is met
        if threshold_direction < 0:
            condition_met = values[i] < threshold
        else:
            condition_met = values[i] > threshold

        if not in_episode:
            if condition_met:
                # Start new episode
                in_episode = True
                episode_start = i
                exit_counter = 0
        else:
            # In episode
            if condition_met:
                exit_counter = 0
            else:
                exit_counter += 1

            # Check for exit
            if exit_counter >= exit_persistence:
                # End episode (at last timestep where condition was met)
                episode_end = i - exit_persistence
                episode_duration = episode_end - episode_start + 1

                if episode_duration >= min_duration:
                    starts[n_episodes] = episode_start
                    ends[n_episodes] = episode_end
                    durations[n_episodes] = episode_duration
                    n_episodes += 1

                in_episode = False
                exit_counter = 0

    # Handle episode still active at end of series
    if in_episode:
        episode_end = n - 1
        episode_duration = episode_end - episode_start + 1
        if episode_duration >= min_duration:
            starts[n_episodes] = episode_start
            ends[n_episodes] = episode_end
            durations[n_episodes] = episode_duration
            n_episodes += 1

    return starts[:n_episodes], ends[:n_episodes], durations[:n_episodes]


def _identify_episodes_python(
    values: np.ndarray,
    threshold: float,
    threshold_direction: int,
    exit_persistence: int,
    min_duration: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pure Python episode identification (fallback when numba unavailable).

    Same interface and logic as _identify_episodes_numba.
    """
    n = len(values)
    starts = []
    ends = []
    durations = []

    in_episode = False
    episode_start = 0
    exit_counter = 0

    for i in range(n):
        if threshold_direction < 0:
            condition_met = values[i] < threshold
        else:
            condition_met = values[i] > threshold

        if not in_episode:
            if condition_met:
                in_episode = True
                episode_start = i
                exit_counter = 0
        else:
            if condition_met:
                exit_counter = 0
            else:
                exit_counter += 1

            if exit_counter >= exit_persistence:
                episode_end = i - exit_persistence
                episode_duration = episode_end - episode_start + 1

                if episode_duration >= min_duration:
                    starts.append(episode_start)
                    ends.append(episode_end)
                    durations.append(episode_duration)

                in_episode = False
                exit_counter = 0

    if in_episode:
        episode_end = n - 1
        episode_duration = episode_end - episode_start + 1
        if episode_duration >= min_duration:
            starts.append(episode_start)
            ends.append(episode_end)
            durations.append(episode_duration)

    return (
        np.array(starts, dtype=np.int64),
        np.array(ends, dtype=np.int64),
        np.array(durations, dtype=np.int64)
    )


def identify_episodes(
    ts: pd.DataFrame,
    condition_col: str,
    threshold: float,
    threshold_direction: str,
    exit_persistence: int,
    min_duration: int,
    episode_type: str,
    episode_id_start: int = 0
) -> pd.DataFrame:
    """
    Identify episodes across all realizations.

    Parameters
    ----------
    ts : pd.DataFrame
        Weekly time series with columns: realization_id, week, date, {condition_col}
    condition_col : str
        Column name containing the condition variable
    threshold : float
        Threshold value for condition
    threshold_direction : str
        "below" for E1 (inflow deficit), "above" for E1d/E1c (demand/combined stress)
    exit_persistence : int
        Weeks above/below threshold to end episode
    min_duration : int
        Minimum episode duration in weeks
    episode_type : str
        Episode type code (E1, E1d, E1c, E2, E3, E4, E5)
    episode_id_start : int
        Starting episode ID (for combining multiple episode types)

    Returns
    -------
    episodes_df : pd.DataFrame
        Episode records with columns: episode_id, realization_id, episode_type,
        start_week, end_week, start_date, end_date, duration
    """
    direction = -1 if threshold_direction == "below" else 1

    # Select implementation based on numba availability
    if HAS_NUMBA:
        identify_fn = _identify_episodes_numba
    else:
        identify_fn = _identify_episodes_python

    all_episodes = []
    episode_id = episode_id_start

    # Process each realization
    for r, group in ts.groupby('realization_id'):
        group = group.sort_values('week')
        values = group[condition_col].values.astype(np.float64)
        weeks = group['week'].values
        dates = group['date'].values

        # Handle NaN values by replacing with neutral value
        nan_mask = np.isnan(values)
        if nan_mask.any():
            # Use a neutral value that won't trigger episode detection
            neutral = threshold + 1 if direction < 0 else threshold - 1
            values = np.where(nan_mask, neutral, values)

        starts, ends, durations = identify_fn(
            values, threshold, direction, exit_persistence, min_duration
        )

        for i in range(len(starts)):
            all_episodes.append({
                'episode_id': episode_id,
                'realization_id': r,
                'episode_type': episode_type,
                'start_week': weeks[starts[i]],
                'end_week': weeks[ends[i]],
                'start_date': dates[starts[i]],
                'end_date': dates[ends[i]],
                'duration': durations[i]
            })
            episode_id += 1

    if len(all_episodes) == 0:
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=[
            'episode_id', 'realization_id', 'episode_type',
            'start_week', 'end_week', 'start_date', 'end_date', 'duration'
        ])

    return pd.DataFrame(all_episodes)


def identify_compound_episodes(
    ts: pd.DataFrame,
    episode_type: str,
    exit_persistence: int,
    min_duration: int,
    satisfaction_tolerance: float,
    episode_id_start: int = 0
) -> pd.DataFrame:
    """
    Identify compound impact episodes (E5) where both E3 and E4 are active.

    E5 episodes occur when both demand_satisfaction < tolerance AND
    flow_satisfaction < tolerance simultaneously.

    Parameters
    ----------
    ts : pd.DataFrame
        Weekly time series with demand_satisfaction and flow_satisfaction
    episode_type : str
        Should be 'E5'
    exit_persistence : int
        Weeks with either condition satisfied to end episode
    min_duration : int
        Minimum episode duration
    satisfaction_tolerance : float
        Threshold below which satisfaction indicates shortage
    episode_id_start : int
        Starting episode ID

    Returns
    -------
    episodes_df : pd.DataFrame
        E5 episode records
    """
    # Create compound condition: both satisfactions below tolerance
    ts = ts.copy()
    ts['compound_shortage'] = (
        (ts['demand_satisfaction'] < satisfaction_tolerance) &
        (ts['flow_satisfaction'] < satisfaction_tolerance)
    ).astype(float)

    # Identify episodes where compound_shortage == 1 (True)
    return identify_episodes(
        ts=ts,
        condition_col='compound_shortage',
        threshold=0.5,  # > 0.5 means True
        threshold_direction='above',
        exit_persistence=exit_persistence,
        min_duration=min_duration,
        episode_type=episode_type,
        episode_id_start=episode_id_start
    )


def identify_all_episodes(
    weekly_ts: pd.DataFrame,
    config: 'EpisodeAnalysisConfig'
) -> pd.DataFrame:
    """
    Identify all episode types across the weekly time series.

    This is the main entry point for episode identification, calling
    the appropriate detection functions for each episode type.

    Parameters
    ----------
    weekly_ts : pd.DataFrame
        Weekly time series with standardized variables
    config : EpisodeAnalysisConfig
        Configuration with thresholds and parameters

    Returns
    -------
    episodes : pd.DataFrame
        All identified episodes across all types
    """
    all_episodes = []
    episode_id = 0

    # E1: Inflow Stress (inflow_std < threshold)
    print("    Identifying E1 (Inflow Stress) episodes...")
    e1_episodes = identify_episodes(
        ts=weekly_ts,
        condition_col='inflow_std',
        threshold=config.inflow_stress_threshold,
        threshold_direction='below',
        exit_persistence=config.exit_persistence_weeks,
        min_duration=config.min_episode_duration_weeks,
        episode_type='E1',
        episode_id_start=episode_id
    )
    all_episodes.append(e1_episodes)
    episode_id += len(e1_episodes)
    print(f"      Found {len(e1_episodes)} E1 episodes")

    # E1d: Demand Stress (demand_std > threshold)
    print("    Identifying E1d (Demand Stress) episodes...")
    e1d_episodes = identify_episodes(
        ts=weekly_ts,
        condition_col='demand_std',
        threshold=config.demand_stress_threshold,
        threshold_direction='above',
        exit_persistence=config.exit_persistence_weeks,
        min_duration=config.min_episode_duration_weeks,
        episode_type='E1d',
        episode_id_start=episode_id
    )
    all_episodes.append(e1d_episodes)
    episode_id += len(e1d_episodes)
    print(f"      Found {len(e1d_episodes)} E1d episodes")

    # E1c: Combined Stress (combined_stress_std > threshold)
    print("    Identifying E1c (Combined Stress) episodes...")
    e1c_episodes = identify_episodes(
        ts=weekly_ts,
        condition_col='combined_stress_std',
        threshold=config.combined_stress_threshold,
        threshold_direction='above',
        exit_persistence=config.exit_persistence_weeks,
        min_duration=config.min_episode_duration_weeks,
        episode_type='E1c',
        episode_id_start=episode_id
    )
    all_episodes.append(e1c_episodes)
    episode_id += len(e1c_episodes)
    print(f"      Found {len(e1c_episodes)} E1c episodes")

    # E2: Zone Transition (ffmp_zone > baseline)
    print("    Identifying E2 (Zone Transition) episodes...")
    e2_episodes = identify_episodes(
        ts=weekly_ts,
        condition_col='ffmp_zone',
        threshold=config.baseline_ffmp_zone,
        threshold_direction='above',
        exit_persistence=config.exit_persistence_weeks,
        min_duration=config.min_episode_duration_weeks,
        episode_type='E2',
        episode_id_start=episode_id
    )
    all_episodes.append(e2_episodes)
    episode_id += len(e2_episodes)
    print(f"      Found {len(e2_episodes)} E2 episodes")

    # E3: Demand Impact (demand_satisfaction < tolerance)
    print("    Identifying E3 (Demand Impact) episodes...")
    # Invert satisfaction for "below threshold" detection
    weekly_ts_temp = weekly_ts.copy()
    weekly_ts_temp['demand_shortage'] = 1.0 - weekly_ts_temp['demand_satisfaction']
    e3_episodes = identify_episodes(
        ts=weekly_ts_temp,
        condition_col='demand_shortage',
        threshold=1.0 - config.satisfaction_tolerance,
        threshold_direction='above',
        exit_persistence=config.exit_persistence_weeks,
        min_duration=config.min_episode_duration_weeks,
        episode_type='E3',
        episode_id_start=episode_id
    )
    all_episodes.append(e3_episodes)
    episode_id += len(e3_episodes)
    print(f"      Found {len(e3_episodes)} E3 episodes")

    # E4: Flow Impact (flow_satisfaction < tolerance)
    print("    Identifying E4 (Flow Impact) episodes...")
    weekly_ts_temp['flow_shortage'] = 1.0 - weekly_ts_temp['flow_satisfaction']
    e4_episodes = identify_episodes(
        ts=weekly_ts_temp,
        condition_col='flow_shortage',
        threshold=1.0 - config.satisfaction_tolerance,
        threshold_direction='above',
        exit_persistence=config.exit_persistence_weeks,
        min_duration=config.min_episode_duration_weeks,
        episode_type='E4',
        episode_id_start=episode_id
    )
    all_episodes.append(e4_episodes)
    episode_id += len(e4_episodes)
    print(f"      Found {len(e4_episodes)} E4 episodes")

    # E5: Compound Impact (both E3 and E4 active)
    print("    Identifying E5 (Compound Impact) episodes...")
    e5_episodes = identify_compound_episodes(
        ts=weekly_ts,
        episode_type='E5',
        exit_persistence=config.exit_persistence_weeks,
        min_duration=config.min_episode_duration_weeks,
        satisfaction_tolerance=config.satisfaction_tolerance,
        episode_id_start=episode_id
    )
    all_episodes.append(e5_episodes)
    print(f"      Found {len(e5_episodes)} E5 episodes")

    # Combine all episodes
    episodes = pd.concat(all_episodes, ignore_index=True)

    # Ensure proper dtypes
    episodes['episode_id'] = episodes['episode_id'].astype(int)
    episodes['realization_id'] = episodes['realization_id'].astype(int)
    episodes['start_week'] = episodes['start_week'].astype(int)
    episodes['end_week'] = episodes['end_week'].astype(int)
    episodes['duration'] = episodes['duration'].astype(int)

    return episodes
