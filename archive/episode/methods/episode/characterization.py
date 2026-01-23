"""
Episode characterization and feature extraction.

This module computes feature vectors for each identified episode,
including stress metrics, state variables, and outcome measures.
"""

import numpy as np
import pandas as pd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import EpisodeAnalysisConfig

# Import from parent package
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import calculate_water_year_period_index


def characterize_episodes(
    episodes: pd.DataFrame,
    weekly_ts: pd.DataFrame,
    config: 'EpisodeAnalysisConfig'
) -> pd.DataFrame:
    """
    Compute feature vectors for each episode.

    Features are organized into categories:
    - TEMPORAL: duration, timing, sequence position
    - STRESS: inflow/demand severity and intensity during episode
    - STATE: storage levels and FFMP zones at onset and during episode
    - OUTCOME: shortage severity for impact episodes (E3, E4, E5)

    Parameters
    ----------
    episodes : pd.DataFrame
        Episode records from identify_all_episodes()
    weekly_ts : pd.DataFrame
        Weekly time series with all variables
    config : EpisodeAnalysisConfig
        Configuration object

    Returns
    -------
    episodes : pd.DataFrame
        Episodes with added feature columns
    """
    if len(episodes) == 0:
        return episodes

    print(f"  Characterizing {len(episodes)} episodes...")

    # Pre-index weekly_ts for efficient lookups
    ts_indexed = weekly_ts.set_index(['realization_id', 'week'])

    features = []

    for idx, ep in episodes.iterrows():
        r = ep['realization_id']
        start_w = int(ep['start_week'])
        end_w = int(ep['end_week'])
        ep_type = ep['episode_type']

        # Get episode time series
        ep_ts = weekly_ts[
            (weekly_ts['realization_id'] == r) &
            (weekly_ts['week'] >= start_w) &
            (weekly_ts['week'] <= end_w)
        ].copy()

        # Get antecedent time series (12 weeks prior)
        ante_ts = weekly_ts[
            (weekly_ts['realization_id'] == r) &
            (weekly_ts['week'] >= start_w - 12) &
            (weekly_ts['week'] < start_w)
        ]

        # Get trend time series (4 weeks prior)
        trend_ts = weekly_ts[
            (weekly_ts['realization_id'] == r) &
            (weekly_ts['week'] >= start_w - 4) &
            (weekly_ts['week'] < start_w)
        ]

        # =====================================================================
        # TEMPORAL FEATURES
        # =====================================================================
        start_date = pd.to_datetime(ep['start_date'])
        start_week_of_year = calculate_water_year_period_index(
            pd.DatetimeIndex([start_date]), 'weekly', config.period_origin
        )[0]
        start_year = start_date.year

        # =====================================================================
        # STRESS FEATURES (during episode)
        # =====================================================================
        if len(ep_ts) > 0 and 'inflow_std' in ep_ts.columns:
            inflow_std = ep_ts['inflow_std'].dropna()
            demand_std = ep_ts['demand_std'].dropna()

            # Inflow severity: sum of negative anomalies (more negative = more severe)
            inflow_severity = inflow_std[inflow_std < 0].sum() if len(inflow_std) > 0 else 0.0
            # Inflow intensity: most extreme negative anomaly
            inflow_intensity = inflow_std.min() if len(inflow_std) > 0 else np.nan

            # Demand severity: sum of positive anomalies
            demand_severity = demand_std[demand_std > 0].sum() if len(demand_std) > 0 else 0.0
            # Demand intensity: most extreme positive anomaly
            demand_intensity = demand_std.max() if len(demand_std) > 0 else np.nan

            # Combined stress metrics
            combined_stress = ep_ts['combined_stress_std'].dropna()
            combined_stress_mean = combined_stress.mean() if len(combined_stress) > 0 else np.nan
            combined_stress_max = combined_stress.max() if len(combined_stress) > 0 else np.nan

            # Net stress cumulative (MG, positive = drawing storage)
            net_stress = ep_ts['net_stress'].dropna()
            net_stress_cum = net_stress.sum() * 7 if len(net_stress) > 0 else 0.0  # Convert MGD to MG
        else:
            inflow_severity = np.nan
            inflow_intensity = np.nan
            demand_severity = np.nan
            demand_intensity = np.nan
            combined_stress_mean = np.nan
            combined_stress_max = np.nan
            net_stress_cum = np.nan

        # =====================================================================
        # STATE FEATURES (at onset)
        # =====================================================================
        if len(ep_ts) > 0:
            onset_row = ep_ts.iloc[0]
            storage_pct_onset = onset_row['storage_pct']
            zone_onset = onset_row['ffmp_zone']
        else:
            storage_pct_onset = np.nan
            zone_onset = np.nan

        # Storage trend (linear slope over 4 weeks prior)
        if len(trend_ts) >= 2:
            x = np.arange(len(trend_ts))
            y = trend_ts['storage_pct'].values
            try:
                storage_trend_onset = np.polyfit(x, y, 1)[0]  # Slope (pct/week)
            except:
                storage_trend_onset = np.nan
        else:
            storage_trend_onset = np.nan

        # Antecedent deficit (sum of negative inflow anomalies over 12 weeks prior)
        if len(ante_ts) > 0 and 'inflow_std' in ante_ts.columns:
            ante_inflow = ante_ts['inflow_std'].dropna()
            antecedent_deficit = ante_inflow[ante_inflow < 0].sum()
        else:
            antecedent_deficit = 0.0

        # =====================================================================
        # STATE FEATURES (during episode)
        # =====================================================================
        if len(ep_ts) > 0:
            storage_pct_min = ep_ts['storage_pct'].min()
            zone_max = ep_ts['ffmp_zone'].max()  # Higher = more severe
            storage_drawdown = storage_pct_onset - storage_pct_min
        else:
            storage_pct_min = np.nan
            zone_max = np.nan
            storage_drawdown = np.nan

        # =====================================================================
        # OUTCOME FEATURES (for E3, E4, E5)
        # =====================================================================
        # Demand shortage metrics (E3, E5)
        if ep_type in ['E3', 'E5'] and len(ep_ts) > 0:
            demand_sat = ep_ts['demand_satisfaction']
            demand_shortage_severity = (1 - demand_sat).sum()
            demand_shortage_max = (1 - demand_sat).max()
            demand_shortage_duration = (demand_sat < config.satisfaction_tolerance).sum()
        else:
            demand_shortage_severity = np.nan
            demand_shortage_max = np.nan
            demand_shortage_duration = np.nan

        # Flow shortage metrics (E4, E5)
        if ep_type in ['E4', 'E5'] and len(ep_ts) > 0:
            flow_sat = ep_ts['flow_satisfaction']
            flow_shortage_severity = (1 - flow_sat).sum()
            flow_shortage_max = (1 - flow_sat).max()
            flow_shortage_duration = (flow_sat < config.satisfaction_tolerance).sum()
        else:
            flow_shortage_severity = np.nan
            flow_shortage_max = np.nan
            flow_shortage_duration = np.nan

        # =====================================================================
        # STORE FEATURES
        # =====================================================================
        features.append({
            'episode_id': ep['episode_id'],
            # Temporal
            'start_week_of_year': start_week_of_year,
            'start_year': start_year,
            # Stress
            'inflow_severity': inflow_severity,
            'inflow_intensity': inflow_intensity,
            'demand_severity': demand_severity,
            'demand_intensity': demand_intensity,
            'combined_stress_mean': combined_stress_mean,
            'combined_stress_max': combined_stress_max,
            'net_stress_cum': net_stress_cum,
            # State at onset
            'storage_pct_onset': storage_pct_onset,
            'zone_onset': zone_onset,
            'storage_trend_onset': storage_trend_onset,
            'antecedent_deficit': antecedent_deficit,
            # State during
            'storage_pct_min': storage_pct_min,
            'zone_max': zone_max,
            'storage_drawdown': storage_drawdown,
            # Outcome
            'demand_shortage_severity': demand_shortage_severity,
            'demand_shortage_max': demand_shortage_max,
            'demand_shortage_duration': demand_shortage_duration,
            'flow_shortage_severity': flow_shortage_severity,
            'flow_shortage_max': flow_shortage_max,
            'flow_shortage_duration': flow_shortage_duration,
        })

    # Create features DataFrame and merge with episodes
    features_df = pd.DataFrame(features)
    episodes_with_features = episodes.merge(features_df, on='episode_id')

    print(f"  Added {len(features_df.columns) - 1} features to episodes")

    return episodes_with_features


def compute_episode_summary_stats(episodes: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics for episodes by type.

    Parameters
    ----------
    episodes : pd.DataFrame
        Episodes with features

    Returns
    -------
    summary : pd.DataFrame
        Summary statistics by episode type
    """
    summary_rows = []

    for ep_type in episodes['episode_type'].unique():
        type_eps = episodes[episodes['episode_type'] == ep_type]

        summary_rows.append({
            'episode_type': ep_type,
            'count': len(type_eps),
            'n_realizations': type_eps['realization_id'].nunique(),
            'mean_per_realization': len(type_eps) / type_eps['realization_id'].nunique(),
            'mean_duration': type_eps['duration'].mean(),
            'median_duration': type_eps['duration'].median(),
            'max_duration': type_eps['duration'].max(),
            'mean_storage_onset': type_eps['storage_pct_onset'].mean() if 'storage_pct_onset' in type_eps.columns else np.nan,
            'mean_storage_min': type_eps['storage_pct_min'].mean() if 'storage_pct_min' in type_eps.columns else np.nan,
        })

    return pd.DataFrame(summary_rows)
