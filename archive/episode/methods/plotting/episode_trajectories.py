"""
Publication-quality trajectory visualizations for episode dynamics.

This module provides functions to visualize how episodes evolve through
the stress-state-outcome phase space, suitable for academic publications.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import Optional, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..episode.config import EpisodeAnalysisConfig

# Publication-quality color palette
TRAJECTORY_COLORS = {
    'cascade': '#c0392b',      # Dark red
    'contained': '#27ae60',    # Green
    'partial_demand': '#f39c12',  # Orange
    'partial_flow': '#3498db',    # Blue
}

# Phase space variable labels (publication style)
VARIABLE_LABELS = {
    'inflow_std': 'Standardized Inflow Anomaly ($\\sigma$)',
    'demand_std': 'Standardized Demand Anomaly ($\\sigma$)',
    'combined_stress_std': 'Combined Stress Index ($\\sigma$)',
    'storage_pct': 'NYC Reservoir Storage (%)',
    'ffmp_zone': 'FFMP Drought Zone',
    'demand_satisfaction': 'Demand Satisfaction Ratio',
    'flow_satisfaction': 'Flow Target Satisfaction Ratio',
    'net_stress': 'Net Stress (Demand - Inflow, MGD)',
}


def get_episode_trajectory(
    episode: pd.Series,
    weekly_ts: pd.DataFrame,
    variables: List[str],
    window_before: int = 8,
    window_after: int = 4
) -> pd.DataFrame:
    """
    Extract time series trajectory for a single episode.

    Parameters
    ----------
    episode : pd.Series
        Episode record with realization_id, start_week, end_week
    weekly_ts : pd.DataFrame
        Weekly time series
    variables : List[str]
        Variables to extract
    window_before : int
        Weeks before episode start to include
    window_after : int
        Weeks after episode end to include

    Returns
    -------
    trajectory : pd.DataFrame
        Time series for the episode window with relative_week column
    """
    r = episode['realization_id']
    start_w = episode['start_week']
    end_w = episode['end_week']

    window_start = max(0, start_w - window_before)
    window_end = end_w + window_after

    traj = weekly_ts[
        (weekly_ts['realization_id'] == r) &
        (weekly_ts['week'] >= window_start) &
        (weekly_ts['week'] <= window_end)
    ].copy()

    # Add relative week (0 = episode start)
    traj['relative_week'] = traj['week'] - start_w

    # Mark episode period
    traj['in_episode'] = (traj['week'] >= start_w) & (traj['week'] <= end_w)

    return traj[['relative_week', 'in_episode', 'week'] + variables]


def create_phase_space_trajectory_figure(
    episodes: pd.DataFrame,
    weekly_ts: pd.DataFrame,
    config: 'EpisodeAnalysisConfig',
    x_var: str = 'storage_pct',
    y_var: str = 'combined_stress_std',
    n_examples: int = 20,
    save_path: Optional[Path] = None,
    figsize: tuple = (10, 8),
    random_seed: int = 42
) -> plt.Figure:
    """
    Create 2D phase space plot with episode trajectories.

    Shows how episodes evolve through the stress-state space, with
    trajectories colored by outcome and arrows indicating direction.

    Parameters
    ----------
    episodes : pd.DataFrame
        Episodes with cascade_classification
    weekly_ts : pd.DataFrame
        Weekly time series
    config : EpisodeAnalysisConfig
        Configuration
    x_var : str
        Variable for x-axis
    y_var : str
        Variable for y-axis
    n_examples : int
        Number of example trajectories per classification
    save_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size
    random_seed : int
        Random seed for reproducibility

    Returns
    -------
    fig : matplotlib.Figure
    """
    np.random.seed(random_seed)

    # Filter to stress episodes with cascade classification
    stress_types = ['E1', 'E1d', 'E1c']
    stress_eps = episodes[
        (episodes['episode_type'].isin(stress_types)) &
        (episodes['cascade_classification'].notna())
    ].copy()

    if len(stress_eps) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No stress episodes with classification", ha='center', va='center')
        return fig

    fig, ax = plt.subplots(figsize=figsize)

    # Sample episodes from each classification
    classifications = ['contained', 'cascade']

    for cls in classifications:
        cls_eps = stress_eps[stress_eps['cascade_classification'] == cls]
        if len(cls_eps) == 0:
            continue

        # Sample episodes
        n_sample = min(n_examples, len(cls_eps))
        sampled = cls_eps.sample(n=n_sample, random_state=random_seed)

        color = TRAJECTORY_COLORS[cls]

        for _, ep in sampled.iterrows():
            traj = get_episode_trajectory(
                ep, weekly_ts, [x_var, y_var],
                window_before=8, window_after=4
            )

            if len(traj) < 3:
                continue

            x = traj[x_var].values
            y = traj[y_var].values
            in_ep = traj['in_episode'].values

            # Plot full trajectory with thin line
            ax.plot(x, y, color=color, alpha=0.15, linewidth=0.8, zorder=1)

            # Highlight episode portion with thicker line
            ep_mask = np.where(in_ep)[0]
            if len(ep_mask) > 1:
                ax.plot(x[ep_mask], y[ep_mask], color=color, alpha=0.4,
                       linewidth=1.5, zorder=2)

            # Mark start and end of episode
            if len(ep_mask) > 0:
                ax.scatter(x[ep_mask[0]], y[ep_mask[0]], color=color,
                          s=20, marker='o', alpha=0.5, zorder=3)
                ax.scatter(x[ep_mask[-1]], y[ep_mask[-1]], color=color,
                          s=20, marker='s', alpha=0.5, zorder=3)

    # Add mean trajectories for each classification
    for cls in classifications:
        cls_eps = stress_eps[stress_eps['cascade_classification'] == cls]
        if len(cls_eps) < 5:
            continue

        # Compute mean trajectory
        all_trajs = []
        for _, ep in cls_eps.iterrows():
            traj = get_episode_trajectory(
                ep, weekly_ts, [x_var, y_var],
                window_before=8, window_after=4
            )
            if len(traj) > 0:
                traj = traj.set_index('relative_week')
                all_trajs.append(traj[[x_var, y_var]])

        if len(all_trajs) > 0:
            combined = pd.concat(all_trajs, keys=range(len(all_trajs)))
            mean_traj = combined.groupby('relative_week').mean()

            x_mean = mean_traj[x_var].values
            y_mean = mean_traj[y_var].values

            # Plot mean trajectory with thick line
            ax.plot(x_mean, y_mean, color=TRAJECTORY_COLORS[cls],
                   linewidth=3, alpha=0.9, zorder=10,
                   label=f'{cls.title()} (mean, n={len(cls_eps)})')

            # Add arrow to show direction
            if len(x_mean) > 2:
                mid = len(x_mean) // 2
                ax.annotate('', xy=(x_mean[mid+1], y_mean[mid+1]),
                           xytext=(x_mean[mid], y_mean[mid]),
                           arrowprops=dict(arrowstyle='->', color=TRAJECTORY_COLORS[cls],
                                          lw=2), zorder=11)

    # Add threshold lines
    if y_var == 'combined_stress_std':
        ax.axhline(config.combined_stress_threshold, color='gray',
                   linestyle='--', alpha=0.7, label='Stress threshold')
    if x_var == 'storage_pct':
        ax.axvline(50, color='gray', linestyle=':', alpha=0.5)  # 50% storage

    # Labels
    ax.set_xlabel(VARIABLE_LABELS.get(x_var, x_var), fontsize=12)
    ax.set_ylabel(VARIABLE_LABELS.get(y_var, y_var), fontsize=12)
    ax.set_title('Episode Trajectories in Phase Space', fontsize=14, fontweight='bold')

    # Legend
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

    # Add text annotation
    ax.text(0.98, 0.02, 'Trajectories: 8 weeks before to 4 weeks after episode',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
            style='italic', alpha=0.7)

    plt.tight_layout()

    if save_path:
        dpi = config.figure_dpi if config else 300
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"  Saved phase space trajectories: {save_path}")

    return fig


def create_temporal_trajectory_figure(
    episodes: pd.DataFrame,
    weekly_ts: pd.DataFrame,
    config: 'EpisodeAnalysisConfig',
    variables: List[str] = None,
    n_examples: int = 5,
    save_path: Optional[Path] = None,
    figsize: tuple = (14, 10),
    random_seed: int = 42
) -> plt.Figure:
    """
    Create multi-panel temporal trajectory comparison.

    Shows how key variables evolve over time for cascade vs contained
    episodes, aligned by episode onset.

    Parameters
    ----------
    episodes : pd.DataFrame
        Episodes with cascade_classification
    weekly_ts : pd.DataFrame
        Weekly time series
    config : EpisodeAnalysisConfig
        Configuration
    variables : List[str], optional
        Variables to plot (default: standard set)
    n_examples : int
        Number of individual trajectories to show
    save_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size
    random_seed : int
        Random seed

    Returns
    -------
    fig : matplotlib.Figure
    """
    np.random.seed(random_seed)

    if variables is None:
        variables = ['inflow_std', 'storage_pct', 'demand_satisfaction', 'flow_satisfaction']

    # Filter to stress episodes
    stress_types = ['E1', 'E1d', 'E1c']
    stress_eps = episodes[
        (episodes['episode_type'].isin(stress_types)) &
        (episodes['cascade_classification'].isin(['contained', 'cascade']))
    ]

    n_vars = len(variables)
    fig, axes = plt.subplots(n_vars, 2, figsize=figsize, sharex=True)

    classifications = ['contained', 'cascade']
    titles = ['Contained Episodes', 'Cascade Episodes']

    for col, (cls, title) in enumerate(zip(classifications, titles)):
        cls_eps = stress_eps[stress_eps['cascade_classification'] == cls]

        if len(cls_eps) == 0:
            for row in range(n_vars):
                axes[row, col].text(0.5, 0.5, f'No {cls} episodes', ha='center', va='center')
            continue

        # Compute mean and percentile bands
        all_trajs = {var: [] for var in variables}

        for _, ep in cls_eps.iterrows():
            traj = get_episode_trajectory(
                ep, weekly_ts, variables,
                window_before=8, window_after=4
            )
            if len(traj) > 0:
                traj = traj.set_index('relative_week')
                for var in variables:
                    if var in traj.columns:
                        all_trajs[var].append(traj[var])

        # Plot each variable
        for row, var in enumerate(variables):
            ax = axes[row, col]

            if len(all_trajs[var]) > 0:
                combined = pd.concat(all_trajs[var], axis=1)

                # Compute statistics
                mean_vals = combined.mean(axis=1)
                p25 = combined.quantile(0.25, axis=1)
                p75 = combined.quantile(0.75, axis=1)
                p10 = combined.quantile(0.10, axis=1)
                p90 = combined.quantile(0.90, axis=1)

                weeks = mean_vals.index.values

                # Plot bands
                ax.fill_between(weeks, p10, p90, alpha=0.15,
                               color=TRAJECTORY_COLORS[cls])
                ax.fill_between(weeks, p25, p75, alpha=0.3,
                               color=TRAJECTORY_COLORS[cls])

                # Plot mean
                ax.plot(weeks, mean_vals, color=TRAJECTORY_COLORS[cls],
                       linewidth=2.5, label='Mean')

                # Plot individual examples (faint)
                sample_eps = cls_eps.sample(n=min(n_examples, len(cls_eps)),
                                           random_state=random_seed)
                for _, ep in sample_eps.iterrows():
                    traj = get_episode_trajectory(ep, weekly_ts, [var],
                                                 window_before=8, window_after=4)
                    if len(traj) > 0:
                        ax.plot(traj['relative_week'], traj[var],
                               color=TRAJECTORY_COLORS[cls], alpha=0.15, linewidth=0.8)

            # Shade episode period
            ax.axvspan(0, cls_eps['duration'].median(), alpha=0.1, color='gray')

            # Add threshold lines
            if var == 'inflow_std':
                ax.axhline(config.inflow_stress_threshold, color='red',
                          linestyle='--', alpha=0.7)
            elif var in ['demand_satisfaction', 'flow_satisfaction']:
                ax.axhline(config.satisfaction_tolerance, color='red',
                          linestyle='--', alpha=0.7)

            # Labels
            ax.set_ylabel(VARIABLE_LABELS.get(var, var), fontsize=10)
            ax.axvline(0, color='black', linestyle='-', alpha=0.3)

            if row == 0:
                ax.set_title(f'{title}\n(n={len(cls_eps)})', fontsize=12, fontweight='bold')

            if row == n_vars - 1:
                ax.set_xlabel('Weeks Relative to Episode Onset', fontsize=11)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', alpha=0.3, label='25-75th percentile'),
        Patch(facecolor='gray', alpha=0.15, label='10-90th percentile'),
        Line2D([0], [0], color='gray', linewidth=2.5, label='Mean'),
        Line2D([0], [0], color='red', linestyle='--', label='Threshold'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=4,
               fontsize=9, bbox_to_anchor=(0.5, 0.02))

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    if save_path:
        dpi = config.figure_dpi if config else 300
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"  Saved temporal trajectories: {save_path}")

    return fig


def create_3d_trajectory_figure(
    episodes: pd.DataFrame,
    weekly_ts: pd.DataFrame,
    config: 'EpisodeAnalysisConfig',
    x_var: str = 'inflow_std',
    y_var: str = 'storage_pct',
    z_var: str = 'demand_satisfaction',
    n_examples: int = 30,
    save_path: Optional[Path] = None,
    figsize: tuple = (12, 10),
    random_seed: int = 42
) -> plt.Figure:
    """
    Create 3D trajectory visualization through stress-state-outcome space.

    Parameters
    ----------
    episodes : pd.DataFrame
        Episodes with cascade_classification
    weekly_ts : pd.DataFrame
        Weekly time series
    config : EpisodeAnalysisConfig
        Configuration
    x_var, y_var, z_var : str
        Variables for 3D axes
    n_examples : int
        Number of trajectories per classification
    save_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size
    random_seed : int
        Random seed

    Returns
    -------
    fig : matplotlib.Figure
    """
    np.random.seed(random_seed)

    stress_types = ['E1', 'E1d', 'E1c']
    stress_eps = episodes[
        (episodes['episode_type'].isin(stress_types)) &
        (episodes['cascade_classification'].isin(['contained', 'cascade']))
    ]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    for cls in ['contained', 'cascade']:
        cls_eps = stress_eps[stress_eps['cascade_classification'] == cls]
        if len(cls_eps) == 0:
            continue

        n_sample = min(n_examples, len(cls_eps))
        sampled = cls_eps.sample(n=n_sample, random_state=random_seed)

        color = TRAJECTORY_COLORS[cls]

        for _, ep in sampled.iterrows():
            traj = get_episode_trajectory(
                ep, weekly_ts, [x_var, y_var, z_var],
                window_before=6, window_after=3
            )

            if len(traj) < 3:
                continue

            x = traj[x_var].values
            y = traj[y_var].values
            z = traj[z_var].values
            in_ep = traj['in_episode'].values

            # Plot trajectory
            ax.plot3D(x, y, z, color=color, alpha=0.3, linewidth=1)

            # Mark episode start
            ep_mask = np.where(in_ep)[0]
            if len(ep_mask) > 0:
                ax.scatter3D(x[ep_mask[0]], y[ep_mask[0]], z[ep_mask[0]],
                            color=color, s=30, marker='o', alpha=0.6)

    # Labels
    ax.set_xlabel(VARIABLE_LABELS.get(x_var, x_var), fontsize=10, labelpad=10)
    ax.set_ylabel(VARIABLE_LABELS.get(y_var, y_var), fontsize=10, labelpad=10)
    ax.set_zlabel(VARIABLE_LABELS.get(z_var, z_var), fontsize=10, labelpad=10)

    ax.set_title('Episode Trajectories in 3D Phase Space\n(Stress → State → Outcome)',
                fontsize=12, fontweight='bold')

    # Legend
    legend_elements = [
        Line2D([0], [0], color=TRAJECTORY_COLORS['contained'], linewidth=2,
               label=f"Contained (n={len(stress_eps[stress_eps['cascade_classification']=='contained'])})"),
        Line2D([0], [0], color=TRAJECTORY_COLORS['cascade'], linewidth=2,
               label=f"Cascade (n={len(stress_eps[stress_eps['cascade_classification']=='cascade'])})")
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    if save_path:
        dpi = config.figure_dpi if config else 300
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"  Saved 3D trajectories: {save_path}")

    return fig


def create_divergence_point_figure(
    episodes: pd.DataFrame,
    weekly_ts: pd.DataFrame,
    config: 'EpisodeAnalysisConfig',
    variable: str = 'storage_pct',
    save_path: Optional[Path] = None,
    figsize: tuple = (12, 6)
) -> plt.Figure:
    """
    Identify and visualize the divergence point between cascade and contained trajectories.

    Shows when/where cascade episodes diverge from contained episodes in terms
    of a key state variable.

    Parameters
    ----------
    episodes : pd.DataFrame
        Episodes with cascade_classification
    weekly_ts : pd.DataFrame
        Weekly time series
    config : EpisodeAnalysisConfig
        Configuration
    variable : str
        Variable to analyze for divergence
    save_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib.Figure
    """
    stress_types = ['E1', 'E1d', 'E1c']
    stress_eps = episodes[
        (episodes['episode_type'].isin(stress_types)) &
        (episodes['cascade_classification'].isin(['contained', 'cascade']))
    ]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Collect trajectories
    trajs = {'contained': [], 'cascade': []}

    for cls in ['contained', 'cascade']:
        cls_eps = stress_eps[stress_eps['cascade_classification'] == cls]
        for _, ep in cls_eps.iterrows():
            traj = get_episode_trajectory(
                ep, weekly_ts, [variable],
                window_before=8, window_after=4
            )
            if len(traj) > 0:
                traj = traj.set_index('relative_week')
                trajs[cls].append(traj[variable])

    # Compute mean trajectories and confidence intervals
    means = {}
    stds = {}

    for cls in ['contained', 'cascade']:
        if len(trajs[cls]) > 0:
            combined = pd.concat(trajs[cls], axis=1)
            means[cls] = combined.mean(axis=1)
            stds[cls] = combined.std(axis=1)

    # Left panel: Mean trajectories
    ax1 = axes[0]

    for cls in ['contained', 'cascade']:
        if cls in means:
            weeks = means[cls].index.values
            ax1.plot(weeks, means[cls], color=TRAJECTORY_COLORS[cls],
                    linewidth=2.5, label=f'{cls.title()}')
            ax1.fill_between(weeks,
                            means[cls] - stds[cls],
                            means[cls] + stds[cls],
                            color=TRAJECTORY_COLORS[cls], alpha=0.2)

    ax1.axvline(0, color='black', linestyle='-', alpha=0.3)
    ax1.axvspan(0, stress_eps['duration'].median(), alpha=0.1, color='gray')
    ax1.set_xlabel('Weeks Relative to Episode Onset', fontsize=11)
    ax1.set_ylabel(VARIABLE_LABELS.get(variable, variable), fontsize=11)
    ax1.set_title('Mean Trajectories with $\\pm$1 SD', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)

    # Right panel: Difference and significance
    ax2 = axes[1]

    if 'contained' in means and 'cascade' in means:
        # Align indices
        common_weeks = means['contained'].index.intersection(means['cascade'].index)
        diff = means['cascade'].loc[common_weeks] - means['contained'].loc[common_weeks]

        # Pooled standard error (approximate)
        se = np.sqrt(stds['cascade'].loc[common_weeks]**2 / len(trajs['cascade']) +
                     stds['contained'].loc[common_weeks]**2 / len(trajs['contained']))

        ax2.fill_between(common_weeks, diff - 1.96*se, diff + 1.96*se,
                        color='purple', alpha=0.2, label='95% CI')
        ax2.plot(common_weeks, diff, color='purple', linewidth=2,
                label='Cascade - Contained')
        ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax2.axvline(0, color='black', linestyle='-', alpha=0.3)

        # Mark significant divergence points
        significant = np.abs(diff) > 1.96 * se
        if significant.any():
            first_sig = common_weeks[significant].min()
            ax2.axvline(first_sig, color='red', linestyle=':', alpha=0.7,
                       label=f'First significant divergence (week {first_sig})')

        ax2.set_xlabel('Weeks Relative to Episode Onset', fontsize=11)
        ax2.set_ylabel(f'$\\Delta$ {VARIABLE_LABELS.get(variable, variable)}', fontsize=11)
        ax2.set_title('Trajectory Divergence (Cascade - Contained)', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)

    plt.tight_layout()

    if save_path:
        dpi = config.figure_dpi if config else 300
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"  Saved divergence analysis: {save_path}")

    return fig


def create_publication_figure_panel(
    episodes: pd.DataFrame,
    weekly_ts: pd.DataFrame,
    config: 'EpisodeAnalysisConfig',
    save_path: Optional[Path] = None,
    figsize: tuple = (16, 12)
) -> plt.Figure:
    """
    Create a comprehensive publication-ready multi-panel figure.

    Panel A: Phase space trajectories (storage vs stress)
    Panel B: Temporal evolution of key variables
    Panel C: Divergence analysis
    Panel D: Outcome distribution in stress space

    Parameters
    ----------
    episodes : pd.DataFrame
        Episodes with all features and classifications
    weekly_ts : pd.DataFrame
        Weekly time series
    config : EpisodeAnalysisConfig
        Configuration
    save_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib.Figure
    """
    fig = plt.figure(figsize=figsize)

    # Create grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    stress_types = ['E1', 'E1d', 'E1c']
    stress_eps = episodes[
        (episodes['episode_type'].isin(stress_types)) &
        (episodes['cascade_classification'].isin(['contained', 'cascade']))
    ]

    # Panel A: Phase space trajectories
    ax_a = fig.add_subplot(gs[0, 0])

    for cls in ['contained', 'cascade']:
        cls_eps = stress_eps[stress_eps['cascade_classification'] == cls]
        if len(cls_eps) < 5:
            continue

        # Compute mean trajectory
        all_trajs = []
        for _, ep in cls_eps.sample(n=min(50, len(cls_eps)), random_state=42).iterrows():
            traj = get_episode_trajectory(
                ep, weekly_ts, ['storage_pct', 'combined_stress_std'],
                window_before=6, window_after=3
            )
            if len(traj) > 0:
                all_trajs.append(traj)

        # Plot individual trajectories (faint)
        for traj in all_trajs[:20]:
            ax_a.plot(traj['storage_pct'], traj['combined_stress_std'],
                     color=TRAJECTORY_COLORS[cls], alpha=0.1, linewidth=0.5)

        # Compute and plot mean
        if all_trajs:
            combined = pd.concat([t.set_index('relative_week')[['storage_pct', 'combined_stress_std']]
                                 for t in all_trajs], keys=range(len(all_trajs)))
            mean_traj = combined.groupby('relative_week').mean()

            ax_a.plot(mean_traj['storage_pct'], mean_traj['combined_stress_std'],
                     color=TRAJECTORY_COLORS[cls], linewidth=2.5, alpha=0.9,
                     label=f'{cls.title()} (n={len(cls_eps)})')

    ax_a.axhline(config.combined_stress_threshold, color='gray', linestyle='--', alpha=0.5)
    ax_a.set_xlabel('Storage (%)', fontsize=11)
    ax_a.set_ylabel('Combined Stress ($\\sigma$)', fontsize=11)
    ax_a.set_title('(A) Phase Space Trajectories', fontsize=12, fontweight='bold')
    ax_a.legend(fontsize=9)

    # Panel B: Temporal evolution
    ax_b = fig.add_subplot(gs[0, 1])

    variables = ['storage_pct', 'demand_satisfaction']

    for cls in ['contained', 'cascade']:
        cls_eps = stress_eps[stress_eps['cascade_classification'] == cls]
        if len(cls_eps) < 5:
            continue

        trajs = {v: [] for v in variables}
        for _, ep in cls_eps.iterrows():
            traj = get_episode_trajectory(ep, weekly_ts, variables,
                                         window_before=6, window_after=3)
            if len(traj) > 0:
                traj = traj.set_index('relative_week')
                for v in variables:
                    trajs[v].append(traj[v])

        for i, v in enumerate(variables):
            if trajs[v]:
                combined = pd.concat(trajs[v], axis=1)
                mean_vals = combined.mean(axis=1)
                linestyle = '-' if i == 0 else '--'
                label = f'{cls.title()} - {"Storage" if i==0 else "Demand Sat."}'
                ax_b.plot(mean_vals.index, mean_vals, color=TRAJECTORY_COLORS[cls],
                         linewidth=2, linestyle=linestyle, label=label)

    ax_b.axvline(0, color='black', linestyle='-', alpha=0.3)
    ax_b.set_xlabel('Weeks from Onset', fontsize=11)
    ax_b.set_ylabel('Value', fontsize=11)
    ax_b.set_title('(B) Temporal Evolution', fontsize=12, fontweight='bold')
    ax_b.legend(fontsize=8, ncol=2)

    # Panel C: Storage divergence
    ax_c = fig.add_subplot(gs[1, 0])

    trajs = {'contained': [], 'cascade': []}
    for cls in ['contained', 'cascade']:
        cls_eps = stress_eps[stress_eps['cascade_classification'] == cls]
        for _, ep in cls_eps.iterrows():
            traj = get_episode_trajectory(ep, weekly_ts, ['storage_pct'],
                                         window_before=6, window_after=4)
            if len(traj) > 0:
                trajs[cls].append(traj.set_index('relative_week')['storage_pct'])

    for cls in ['contained', 'cascade']:
        if trajs[cls]:
            combined = pd.concat(trajs[cls], axis=1)
            mean_vals = combined.mean(axis=1)
            p25 = combined.quantile(0.25, axis=1)
            p75 = combined.quantile(0.75, axis=1)

            ax_c.fill_between(mean_vals.index, p25, p75,
                             color=TRAJECTORY_COLORS[cls], alpha=0.2)
            ax_c.plot(mean_vals.index, mean_vals, color=TRAJECTORY_COLORS[cls],
                     linewidth=2.5, label=f'{cls.title()}')

    ax_c.axvline(0, color='black', linestyle='-', alpha=0.3)
    ax_c.axhline(50, color='gray', linestyle=':', alpha=0.5)
    ax_c.set_xlabel('Weeks from Onset', fontsize=11)
    ax_c.set_ylabel('Storage (%)', fontsize=11)
    ax_c.set_title('(C) Storage Trajectory Divergence', fontsize=12, fontweight='bold')
    ax_c.legend(fontsize=9)

    # Panel D: Outcome in stress space
    ax_d = fig.add_subplot(gs[1, 1])

    if 'inflow_severity' in stress_eps.columns and 'demand_severity' in stress_eps.columns:
        for cls in ['contained', 'cascade']:
            mask = stress_eps['cascade_classification'] == cls
            if mask.sum() > 0:
                ax_d.scatter(stress_eps.loc[mask, 'inflow_severity'],
                            stress_eps.loc[mask, 'demand_severity'],
                            c=TRAJECTORY_COLORS[cls], alpha=0.5, s=30,
                            label=f'{cls.title()} (n={mask.sum()})')

        ax_d.set_xlabel('Inflow Severity', fontsize=11)
        ax_d.set_ylabel('Demand Severity', fontsize=11)
        ax_d.set_title('(D) Outcomes in Stress Space', fontsize=12, fontweight='bold')
        ax_d.legend(fontsize=9)
    else:
        ax_d.text(0.5, 0.5, 'Severity features not available',
                 ha='center', va='center', transform=ax_d.transAxes)

    # Overall title
    fig.suptitle('Episode Dynamics: Stress to Outcome Trajectories',
                fontsize=14, fontweight='bold', y=1.02)

    if save_path:
        dpi = config.figure_dpi if config else 300
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"  Saved publication figure: {save_path}")

    return fig
