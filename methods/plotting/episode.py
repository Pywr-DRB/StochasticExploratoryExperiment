"""
Visualization functions for episode-level vulnerability analysis.

This module provides plotting functions for:
- Sankey diagrams of episode progression
- Feature comparison boxplots
- Episode trajectory visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..episode.config import EpisodeAnalysisConfig

# Try to import plotly for Sankey diagrams
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# Episode type colors
EPISODE_COLORS = {
    'E1': '#3498db',      # Blue - inflow stress
    'E1d': '#9b59b6',     # Purple - demand stress
    'E1c': '#e74c3c',     # Red - combined stress
    'E2': '#f39c12',      # Orange - zone transition
    'E3': '#e67e22',      # Dark orange - demand impact
    'E4': '#d35400',      # Darker orange - flow impact
    'E5': '#c0392b',      # Dark red - compound impact
}

CASCADE_COLORS = {
    'contained': '#27ae60',     # Green
    'partial_demand': '#f1c40f', # Yellow
    'partial_flow': '#e67e22',   # Orange
    'cascade': '#e74c3c',        # Red
}


def create_sankey_diagram(
    episodes: pd.DataFrame,
    config: 'EpisodeAnalysisConfig',
    save_path: Optional[Path] = None,
    title: str = "Episode Progression: Stress to Outcomes"
) -> Optional['go.Figure']:
    """
    Create Sankey diagram showing episode progression flows.

    Parameters
    ----------
    episodes : pd.DataFrame
        Episodes with cascade_classification
    config : EpisodeAnalysisConfig
        Configuration object
    save_path : Path, optional
        Path to save figure
    title : str
        Figure title

    Returns
    -------
    fig : plotly.graph_objects.Figure or None
        Sankey diagram figure (None if plotly not available)
    """
    if not HAS_PLOTLY:
        print("Warning: plotly not installed, cannot create Sankey diagram")
        print("Install with: pip install plotly kaleido")
        return None

    # Filter to stress episodes
    stress_types = ['E1', 'E1d', 'E1c']
    stress_eps = episodes[episodes['episode_type'].isin(stress_types)]

    if len(stress_eps) == 0:
        print("No stress episodes found for Sankey diagram")
        return None

    # Count classifications
    counts = stress_eps['cascade_classification'].value_counts()

    # Define nodes
    labels = [
        'Stress Episodes',
        'Contained',
        'Partial (Demand)',
        'Partial (Flow)',
        'Cascade'
    ]

    # Define colors
    node_colors = [
        '#3498db',  # Blue - all stress
        CASCADE_COLORS['contained'],
        CASCADE_COLORS['partial_demand'],
        CASCADE_COLORS['partial_flow'],
        CASCADE_COLORS['cascade'],
    ]

    # Define links (source, target, value)
    links = [
        (0, 1, counts.get('contained', 0)),
        (0, 2, counts.get('partial_demand', 0)),
        (0, 3, counts.get('partial_flow', 0)),
        (0, 4, counts.get('cascade', 0)),
    ]

    # Add percentages to labels
    total = len(stress_eps)
    labels_with_pct = [
        f'Stress Episodes\n(n={total})',
        f"Contained\n({counts.get('contained', 0)}, {100*counts.get('contained', 0)/total:.1f}%)",
        f"Partial (Demand)\n({counts.get('partial_demand', 0)}, {100*counts.get('partial_demand', 0)/total:.1f}%)",
        f"Partial (Flow)\n({counts.get('partial_flow', 0)}, {100*counts.get('partial_flow', 0)/total:.1f}%)",
        f"Cascade\n({counts.get('cascade', 0)}, {100*counts.get('cascade', 0)/total:.1f}%)",
    ]

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color='black', width=0.5),
            label=labels_with_pct,
            color=node_colors
        ),
        link=dict(
            source=[l[0] for l in links],
            target=[l[1] for l in links],
            value=[l[2] for l in links],
            color=['rgba(128,128,128,0.4)'] * len(links)
        )
    )])

    fig.update_layout(
        title_text=title,
        font_size=12,
        width=800,
        height=500
    )

    if save_path:
        save_path = Path(save_path)
        if save_path.suffix == '.html':
            fig.write_html(str(save_path))
        else:
            try:
                fig.write_image(str(save_path), scale=2)
            except Exception as e:
                print(f"Warning: Could not save image ({e}). Saving as HTML instead.")
                fig.write_html(str(save_path.with_suffix('.html')))
        print(f"  Saved Sankey diagram: {save_path}")

    return fig


def create_feature_comparison_figure(
    episodes: pd.DataFrame,
    features: List[str],
    stats_df: Optional[pd.DataFrame] = None,
    config: Optional['EpisodeAnalysisConfig'] = None,
    save_path: Optional[Path] = None,
    figsize: tuple = (14, 10)
) -> plt.Figure:
    """
    Create multi-panel boxplot figure comparing features across cascade groups.

    Parameters
    ----------
    episodes : pd.DataFrame
        Episodes with features and cascade_classification
    features : List[str]
        Features to compare
    stats_df : pd.DataFrame, optional
        Results from compare_episode_populations() for annotations
    config : EpisodeAnalysisConfig, optional
        Configuration object
    save_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib.Figure
        Comparison figure
    """
    # Filter to stress episodes
    stress_types = ['E1', 'E1d', 'E1c']
    stress_eps = episodes[episodes['episode_type'].isin(stress_types)]

    if len(stress_eps) == 0:
        print("No stress episodes found for feature comparison")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No stress episodes", ha='center', va='center')
        return fig

    # Filter features to those that exist
    available_features = [f for f in features if f in stress_eps.columns]
    if len(available_features) == 0:
        print("No valid features found for comparison")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No valid features", ha='center', va='center')
        return fig

    n_features = len(available_features)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_features == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    groups = ['contained', 'cascade']
    colors = [CASCADE_COLORS['contained'], CASCADE_COLORS['cascade']]

    for i, feat in enumerate(available_features):
        ax = axes[i]

        data = []
        for g in groups:
            group_data = stress_eps[stress_eps['cascade_classification'] == g][feat].dropna()
            data.append(group_data)

        # Create boxplot
        bp = ax.boxplot(data, labels=['Contained', 'Cascade'], patch_artist=True)

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Add p-value annotation if available
        if stats_df is not None and len(stats_df) > 0:
            stat_row = stats_df[stats_df['feature'] == feat]
            if len(stat_row) > 0:
                p_col = 'p_value_corrected' if 'p_value_corrected' in stat_row.columns else 'p_value'
                p = stat_row[p_col].values[0]
                d = stat_row['effect_size_d'].values[0]
                sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
                ax.set_title(f'{feat}\n(d={d:.2f}, {sig})', fontsize=10)
            else:
                ax.set_title(feat, fontsize=10)
        else:
            ax.set_title(feat, fontsize=10)

        ax.set_ylabel('Value')
        ax.tick_params(axis='x', rotation=0)

    # Hide unused axes
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if save_path:
        dpi = config.figure_dpi if config else 300
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"  Saved feature comparison: {save_path}")

    return fig


def create_trajectory_comparison_figure(
    cascade_episode_id: int,
    contained_episode_id: int,
    episodes: pd.DataFrame,
    weekly_ts: pd.DataFrame,
    config: 'EpisodeAnalysisConfig',
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create side-by-side trajectory comparison for selected episodes.

    Parameters
    ----------
    cascade_episode_id : int
        Episode ID of cascade example
    contained_episode_id : int
        Episode ID of contained example
    episodes : pd.DataFrame
        Episodes DataFrame
    weekly_ts : pd.DataFrame
        Weekly time series
    config : EpisodeAnalysisConfig
        Configuration object
    save_path : Path, optional
        Path to save figure

    Returns
    -------
    fig : matplotlib.Figure
        Trajectory comparison figure
    """
    variables = ['inflow_std', 'storage_pct', 'demand_satisfaction', 'flow_satisfaction']
    ylabels = ['Inflow (std)', 'Storage (%)', 'Demand Sat.', 'Flow Sat.']
    thresholds = [config.inflow_stress_threshold, None, config.satisfaction_tolerance, config.satisfaction_tolerance]

    fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex='col')

    episode_ids = [cascade_episode_id, contained_episode_id]
    titles = ['Cascade Episode', 'Contained Episode']

    for col, (ep_id, title) in enumerate(zip(episode_ids, titles)):
        # Get episode info
        ep_data = episodes[episodes['episode_id'] == ep_id]
        if len(ep_data) == 0:
            print(f"Warning: Episode {ep_id} not found")
            continue

        ep = ep_data.iloc[0]
        r = ep['realization_id']
        start_w = int(ep['start_week'])
        end_w = int(ep['end_week'])

        # Get extended window (12 weeks before, 4 after)
        window_start = max(0, start_w - 12)
        window_end = end_w + 4

        ts = weekly_ts[
            (weekly_ts['realization_id'] == r) &
            (weekly_ts['week'] >= window_start) &
            (weekly_ts['week'] <= window_end)
        ].sort_values('week')

        if len(ts) == 0:
            continue

        weeks = ts['week'].values

        for row, (var, ylabel, thresh) in enumerate(zip(variables, ylabels, thresholds)):
            ax = axes[row, col]

            if var in ts.columns:
                ax.plot(weeks, ts[var].values, 'b-', linewidth=1.5)

            # Shade episode period
            ax.axvspan(start_w, end_w, alpha=0.2, color='gray', label='Episode')

            # Add threshold lines
            if thresh is not None:
                ax.axhline(thresh, color='r', linestyle='--', alpha=0.7, label='Threshold')

            ax.set_ylabel(ylabel)

            if row == 0:
                ax.set_title(f'{title}\n(ID: {ep_id}, Duration: {ep["duration"]} weeks)')

            if row == 3:
                ax.set_xlabel('Week')

            if row == 0 and col == 0:
                ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=config.figure_dpi, bbox_inches='tight')
        print(f"  Saved trajectory comparison: {save_path}")

    return fig


def create_episode_counts_by_type_figure(
    episodes: pd.DataFrame,
    config: Optional['EpisodeAnalysisConfig'] = None,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create bar chart showing episode counts by type.

    Parameters
    ----------
    episodes : pd.DataFrame
        All episodes
    config : EpisodeAnalysisConfig, optional
        Configuration object
    save_path : Path, optional
        Path to save figure

    Returns
    -------
    fig : matplotlib.Figure
        Bar chart figure
    """
    type_counts = episodes['episode_type'].value_counts()
    type_order = ['E1', 'E1d', 'E1c', 'E2', 'E3', 'E4', 'E5']
    type_counts = type_counts.reindex(type_order).fillna(0)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [EPISODE_COLORS.get(t, '#888888') for t in type_order]
    bars = ax.bar(type_order, type_counts.values, color=colors, edgecolor='black')

    # Add value labels on bars
    for bar, count in zip(bars, type_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{int(count)}', ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Episode Type')
    ax.set_ylabel('Count')
    ax.set_title('Episode Counts by Type')

    # Add legend with episode descriptions
    type_descriptions = {
        'E1': 'Inflow Stress',
        'E1d': 'Demand Stress',
        'E1c': 'Combined Stress',
        'E2': 'Zone Transition',
        'E3': 'Demand Impact',
        'E4': 'Flow Impact',
        'E5': 'Compound Impact'
    }
    legend_elements = [Patch(facecolor=EPISODE_COLORS.get(t, '#888888'),
                            edgecolor='black', label=f'{t}: {type_descriptions.get(t, "")}')
                      for t in type_order]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    plt.tight_layout()

    if save_path:
        dpi = config.figure_dpi if config else 300
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"  Saved episode counts: {save_path}")

    return fig


def create_cascade_rate_histogram(
    episodes: pd.DataFrame,
    config: Optional['EpisodeAnalysisConfig'] = None,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create histogram of cascade rates across realizations.

    Parameters
    ----------
    episodes : pd.DataFrame
        Episodes with cascade_classification
    config : EpisodeAnalysisConfig, optional
        Configuration object
    save_path : Path, optional
        Path to save figure

    Returns
    -------
    fig : matplotlib.Figure
        Histogram figure
    """
    # Compute cascade rate per realization
    stress_types = ['E1', 'E1d', 'E1c']
    stress_eps = episodes[episodes['episode_type'].isin(stress_types)]

    rates = []
    for r in stress_eps['realization_id'].unique():
        r_eps = stress_eps[stress_eps['realization_id'] == r]
        n_cascade = (r_eps['cascade_classification'] == 'cascade').sum()
        rate = 100 * n_cascade / len(r_eps) if len(r_eps) > 0 else 0
        rates.append(rate)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(rates, bins=20, color=CASCADE_COLORS['cascade'], edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(rates), color='black', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(rates):.1f}%')
    ax.axvline(np.median(rates), color='blue', linestyle=':', linewidth=2,
               label=f'Median: {np.median(rates):.1f}%')

    ax.set_xlabel('Cascade Rate (%)')
    ax.set_ylabel('Number of Realizations')
    ax.set_title('Distribution of Cascade Rates Across Realizations')
    ax.legend()

    plt.tight_layout()

    if save_path:
        dpi = config.figure_dpi if config else 300
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"  Saved cascade rate histogram: {save_path}")

    return fig


def create_stress_outcome_scatter(
    episodes: pd.DataFrame,
    config: Optional['EpisodeAnalysisConfig'] = None,
    x_var: str = 'inflow_severity',
    y_var: str = 'demand_severity',
    size_var: str = 'duration',
    save_path: Optional[Path] = None,
    figsize: tuple = (12, 10),
    alpha: float = 0.6,
    add_contours: bool = True,
    add_marginals: bool = True
) -> plt.Figure:
    """
    Create 2D scatter plot of stress variables with outcome encoding.

    Shows how combinations of inflow and demand stress lead to different
    cascade outcomes. Color indicates cascade classification, size indicates
    a third variable (e.g., duration or severity).

    Parameters
    ----------
    episodes : pd.DataFrame
        Episodes with features and cascade_classification
    config : EpisodeAnalysisConfig, optional
        Configuration object
    x_var : str
        Variable for x-axis (default: 'inflow_severity')
    y_var : str
        Variable for y-axis (default: 'demand_severity')
    size_var : str
        Variable for point size (default: 'duration')
    save_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size
    alpha : float
        Point transparency (0-1)
    add_contours : bool
        Whether to add density contours for cascade/contained groups
    add_marginals : bool
        Whether to add marginal histograms

    Returns
    -------
    fig : matplotlib.Figure
        Scatter plot figure
    """
    from matplotlib.lines import Line2D
    from scipy import stats as scipy_stats

    # Filter to stress episodes
    stress_types = ['E1', 'E1d', 'E1c']
    stress_eps = episodes[episodes['episode_type'].isin(stress_types)].copy()

    if len(stress_eps) == 0:
        print("No stress episodes found for scatter plot")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No stress episodes", ha='center', va='center')
        return fig

    # Check required columns exist
    required_cols = [x_var, y_var, 'cascade_classification']
    missing = [c for c in required_cols if c not in stress_eps.columns]
    if missing:
        print(f"Missing columns for scatter plot: {missing}")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"Missing columns: {missing}", ha='center', va='center')
        return fig

    # Drop rows with NaN in key variables
    plot_data = stress_eps.dropna(subset=[x_var, y_var])

    if len(plot_data) == 0:
        print("No valid data after dropping NaN")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No valid data", ha='center', va='center')
        return fig

    # Compute point sizes
    if size_var and size_var in plot_data.columns:
        sizes = plot_data[size_var].fillna(plot_data[size_var].median())
        # Normalize sizes to reasonable range (20-200)
        size_min, size_max = sizes.min(), sizes.max()
        if size_max > size_min:
            sizes_norm = 20 + 180 * (sizes - size_min) / (size_max - size_min)
        else:
            sizes_norm = 50
    else:
        sizes_norm = 50
        size_var = None

    # Create figure with or without marginals
    if add_marginals:
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(4, 4, hspace=0.05, wspace=0.05)
        ax_main = fig.add_subplot(gs[1:4, 0:3])
        ax_top = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
        ax_right = fig.add_subplot(gs[1:4, 3], sharey=ax_main)
    else:
        fig, ax_main = plt.subplots(figsize=figsize)
        ax_top = None
        ax_right = None

    # Plot each classification group
    classifications = ['contained', 'partial_demand', 'partial_flow', 'cascade']
    markers = {'contained': 'o', 'partial_demand': 's', 'partial_flow': '^', 'cascade': 'D'}

    for cls in classifications:
        mask = plot_data['cascade_classification'] == cls
        if mask.sum() == 0:
            continue

        x_data = plot_data.loc[mask, x_var]
        y_data = plot_data.loc[mask, y_var]
        s_data = sizes_norm[mask] if hasattr(sizes_norm, '__iter__') else sizes_norm

        ax_main.scatter(
            x_data, y_data,
            c=CASCADE_COLORS[cls],
            s=s_data,
            marker=markers[cls],
            alpha=alpha,
            edgecolors='white',
            linewidth=0.5,
            label=f'{cls.replace("_", " ").title()} (n={mask.sum()})'
        )

    # Add density contours for cascade vs contained
    if add_contours:
        for cls, color, linestyle in [('contained', CASCADE_COLORS['contained'], '-'),
                                       ('cascade', CASCADE_COLORS['cascade'], '--')]:
            mask = plot_data['cascade_classification'] == cls
            if mask.sum() < 10:
                continue

            x_data = plot_data.loc[mask, x_var].values
            y_data = plot_data.loc[mask, y_var].values

            try:
                # Compute 2D kernel density estimate
                xmin, xmax = plot_data[x_var].min(), plot_data[x_var].max()
                ymin, ymax = plot_data[y_var].min(), plot_data[y_var].max()

                # Add padding
                x_pad = (xmax - xmin) * 0.1
                y_pad = (ymax - ymin) * 0.1

                xx, yy = np.mgrid[xmin-x_pad:xmax+x_pad:100j, ymin-y_pad:ymax+y_pad:100j]
                positions = np.vstack([xx.ravel(), yy.ravel()])
                values = np.vstack([x_data, y_data])
                kernel = scipy_stats.gaussian_kde(values)
                density = np.reshape(kernel(positions).T, xx.shape)

                # Draw contours at 25%, 50%, 75% of max density
                levels = np.percentile(density[density > 0], [25, 50, 75])
                ax_main.contour(xx, yy, density, levels=levels, colors=color,
                               linestyles=linestyle, linewidths=1.5, alpha=0.8)
            except Exception:
                # KDE may fail with too few points or colinear data
                pass

    # Add marginal histograms
    if add_marginals and ax_top is not None and ax_right is not None:
        bins = 30

        for cls in ['contained', 'cascade']:
            mask = plot_data['cascade_classification'] == cls
            if mask.sum() < 2:
                continue

            # Top histogram (x variable)
            ax_top.hist(plot_data.loc[mask, x_var], bins=bins, alpha=0.5,
                       color=CASCADE_COLORS[cls], edgecolor='none', density=True)

            # Right histogram (y variable)
            ax_right.hist(plot_data.loc[mask, y_var], bins=bins, alpha=0.5,
                         color=CASCADE_COLORS[cls], edgecolor='none', density=True,
                         orientation='horizontal')

        ax_top.set_ylabel('Density')
        ax_right.set_xlabel('Density')
        plt.setp(ax_top.get_xticklabels(), visible=False)
        plt.setp(ax_right.get_yticklabels(), visible=False)

    # Labels and styling
    x_label = x_var.replace('_', ' ').title()
    y_label = y_var.replace('_', ' ').title()
    ax_main.set_xlabel(x_label, fontsize=12)
    ax_main.set_ylabel(y_label, fontsize=12)

    # Add reference lines at zero if data spans zero
    if plot_data[x_var].min() < 0 < plot_data[x_var].max():
        ax_main.axvline(0, color='gray', linestyle=':', alpha=0.5, zorder=0)
    if plot_data[y_var].min() < 0 < plot_data[y_var].max():
        ax_main.axhline(0, color='gray', linestyle=':', alpha=0.5, zorder=0)

    # Create legend
    legend_elements = [
        Line2D([0], [0], marker=markers[cls], color='w', markerfacecolor=CASCADE_COLORS[cls],
               markersize=10, label=f'{cls.replace("_", " ").title()}')
        for cls in classifications if (plot_data['cascade_classification'] == cls).sum() > 0
    ]

    # Add size legend if applicable
    if size_var:
        size_label = size_var.replace('_', ' ').title()
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                   markersize=5, label=f'Size = {size_label} (small)')
        )
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                   markersize=12, label=f'Size = {size_label} (large)')
        )

    ax_main.legend(handles=legend_elements, loc='upper right', fontsize=9,
                   framealpha=0.9, edgecolor='gray')

    # Title
    title = f'Stress-Outcome Space: {x_label} vs {y_label}'
    if add_marginals:
        ax_top.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax_main.set_title(title, fontsize=14, fontweight='bold')

    # Add summary statistics as text
    n_total = len(plot_data)
    n_cascade = (plot_data['cascade_classification'] == 'cascade').sum()
    n_contained = (plot_data['cascade_classification'] == 'contained').sum()
    summary_text = (f'Total: {n_total} episodes\n'
                   f'Cascade: {n_cascade} ({100*n_cascade/n_total:.1f}%)\n'
                   f'Contained: {n_contained} ({100*n_contained/n_total:.1f}%)')
    ax_main.text(0.02, 0.98, summary_text, transform=ax_main.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    if not add_marginals:
        plt.tight_layout()

    if save_path:
        dpi = config.figure_dpi if config else 300
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"  Saved stress-outcome scatter: {save_path}")

    return fig


def create_stress_outcome_heatmap(
    episodes: pd.DataFrame,
    config: Optional['EpisodeAnalysisConfig'] = None,
    x_var: str = 'inflow_severity',
    y_var: str = 'demand_severity',
    n_bins: int = 10,
    save_path: Optional[Path] = None,
    figsize: tuple = (12, 10)
) -> plt.Figure:
    """
    Create 2D heatmap showing cascade rate across stress space.

    Bins episodes by x and y stress variables and computes cascade rate
    within each bin. Useful for identifying stress thresholds.

    Parameters
    ----------
    episodes : pd.DataFrame
        Episodes with features and cascade_classification
    config : EpisodeAnalysisConfig, optional
        Configuration object
    x_var : str
        Variable for x-axis binning
    y_var : str
        Variable for y-axis binning
    n_bins : int
        Number of bins in each dimension
    save_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib.Figure
        Heatmap figure
    """
    # Filter to stress episodes
    stress_types = ['E1', 'E1d', 'E1c']
    stress_eps = episodes[episodes['episode_type'].isin(stress_types)].copy()

    if len(stress_eps) == 0:
        print("No stress episodes found for heatmap")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No stress episodes", ha='center', va='center')
        return fig

    # Drop NaN
    plot_data = stress_eps.dropna(subset=[x_var, y_var])

    if len(plot_data) < 10:
        print("Insufficient data for heatmap")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Insufficient data", ha='center', va='center')
        return fig

    # Create bins
    x_bins = np.linspace(plot_data[x_var].min(), plot_data[x_var].max(), n_bins + 1)
    y_bins = np.linspace(plot_data[y_var].min(), plot_data[y_var].max(), n_bins + 1)

    # Compute cascade rate in each bin
    cascade_rate = np.zeros((n_bins, n_bins))
    counts = np.zeros((n_bins, n_bins))

    plot_data['x_bin'] = np.digitize(plot_data[x_var], x_bins) - 1
    plot_data['y_bin'] = np.digitize(plot_data[y_var], y_bins) - 1

    # Clip to valid range
    plot_data['x_bin'] = plot_data['x_bin'].clip(0, n_bins - 1)
    plot_data['y_bin'] = plot_data['y_bin'].clip(0, n_bins - 1)

    for (xb, yb), group in plot_data.groupby(['x_bin', 'y_bin']):
        xb, yb = int(xb), int(yb)
        n_total = len(group)
        n_cascade = (group['cascade_classification'] == 'cascade').sum()
        cascade_rate[yb, xb] = 100 * n_cascade / n_total if n_total > 0 else np.nan
        counts[yb, xb] = n_total

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Cascade rate heatmap
    ax1 = axes[0]
    im1 = ax1.imshow(cascade_rate, origin='lower', aspect='auto',
                     extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]],
                     cmap='RdYlGn_r', vmin=0, vmax=100)
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Cascade Rate (%)')
    ax1.set_xlabel(x_var.replace('_', ' ').title())
    ax1.set_ylabel(y_var.replace('_', ' ').title())
    ax1.set_title('Cascade Rate by Stress Level')

    # Add contour lines at key thresholds
    contour_levels = [25, 50, 75]
    cs = ax1.contour(
        (x_bins[:-1] + x_bins[1:]) / 2,
        (y_bins[:-1] + y_bins[1:]) / 2,
        cascade_rate, levels=contour_levels,
        colors='black', linewidths=1, linestyles=[':', '--', '-']
    )
    ax1.clabel(cs, inline=True, fontsize=8, fmt='%d%%')

    # Episode count heatmap
    ax2 = axes[1]
    im2 = ax2.imshow(counts, origin='lower', aspect='auto',
                     extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]],
                     cmap='Blues')
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Episode Count')
    ax2.set_xlabel(x_var.replace('_', ' ').title())
    ax2.set_ylabel(y_var.replace('_', ' ').title())
    ax2.set_title('Episode Frequency by Stress Level')

    plt.suptitle('Stress Space Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        dpi = config.figure_dpi if config else 300
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"  Saved stress-outcome heatmap: {save_path}")

    return fig
