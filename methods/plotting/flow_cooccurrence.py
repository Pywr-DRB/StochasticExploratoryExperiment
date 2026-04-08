"""
Plotting functions for flow co-occurrence during SSI3 droughts.

Visualizes the relationship between basin-wide drought co-occurrence
(NYC inflow + non-NYC Montague flow both below normal) and NYC storage outcomes.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from methods.plotting.styles import (
    FFMP_ZONE_COLORS, DATASET_COLORS, DATASET_LABELS,
    DATASET_MARKERS, ALPHA_SCATTER,
    FIGSIZE_TRIPLE, FIGSIZE_SINGLE,
    FONTSIZE_LABEL, FONTSIZE_TITLE, FONTSIZE_LEGEND, FONTSIZE_MEDIUM,
    DPI_PRINT,
    apply_publication_style, label_panel, get_dataset_label,
)


# Zone ordering for consistent legend
_ZONE_ORDER = ['Normal', 'Watch', 'Warning', 'Emergency']


def plot_cooccurrence_scatter_grid(merged_dfs, output_dir):
    """
    Multi-panel figure: co-drought fraction vs min storage, one column per dataset.

    Top row: scatter colored by FFMP zone at min storage, sized by severity.
    Bottom row: box plots of co-drought fraction by classification.

    Parameters
    ----------
    merged_dfs : dict
        {dataset_id: DataFrame} with columns including co_drought_frac,
        event_min_storage_pct, ffmp_zone_at_min, severity, classification.
    output_dir : str
        Directory to save figure.
    """
    apply_publication_style()

    datasets = [d for d in merged_dfs if len(merged_dfs[d]) > 0]
    n = len(datasets)
    if n == 0:
        return

    fig, axes = plt.subplots(2, n, figsize=(6 * n, 10),
                             gridspec_kw={'height_ratios': [3, 1.2]})
    if n == 1:
        axes = axes.reshape(-1, 1)

    letters = 'abcdefghij'

    for col, dataset_id in enumerate(datasets):
        df = merged_dfs[dataset_id]
        ax_scatter = axes[0, col]
        ax_box = axes[1, col]

        # --- Top: scatter ---
        _plot_scatter_panel(ax_scatter, df, dataset_id)
        label_panel(ax_scatter, letters[col], dataset_id=dataset_id)

        # --- Bottom: box plot of co-drought fraction by classification ---
        _plot_classification_boxes(ax_box, df)
        label_panel(ax_box, letters[n + col])

    fig.tight_layout()
    fname = f'{output_dir}/cooccurrence_scatter_grid.png'
    fig.savefig(fname, dpi=DPI_PRINT, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {fname}')


def plot_cooccurrence_pooled(merged_dfs, output_dir):
    """
    Single-panel scatter pooling all datasets.

    Datasets distinguished by marker shape, colored by severity.

    Parameters
    ----------
    merged_dfs : dict
        {dataset_id: DataFrame} with co-occurrence metrics and event metrics.
    output_dir : str
        Directory to save figure.
    """
    apply_publication_style()

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    for dataset_id, df in merged_dfs.items():
        if len(df) == 0:
            continue
        marker = DATASET_MARKERS.get(dataset_id, 'o')
        color = DATASET_COLORS.get(dataset_id, '#808080')
        label = get_dataset_label(dataset_id, style='short')

        ax.scatter(
            df['co_drought_frac'], df['event_min_storage_pct'],
            c=color, marker=marker, alpha=ALPHA_SCATTER,
            s=30, label=label, edgecolors='white', linewidths=0.3,
        )

    # Overall trend
    all_df = _concat_all(merged_dfs)
    if len(all_df) > 10:
        _add_lowess_line(ax, all_df['co_drought_frac'], all_df['event_min_storage_pct'],
                         color='black', label='LOWESS trend')
        rho, pval = stats.spearmanr(all_df['co_drought_frac'], all_df['event_min_storage_pct'])
        ax.text(0.98, 0.02, f'Spearman $\\rho$ = {rho:.2f} (p = {pval:.1e})',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=FONTSIZE_MEDIUM, bbox=dict(boxstyle='round,pad=0.3',
                                                     facecolor='white', alpha=0.8))

    ax.set_xlabel('Co-drought fraction', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('Min NYC storage during drought (%)', fontsize=FONTSIZE_LABEL)
    ax.set_title('Basin-wide drought co-occurrence vs. storage outcomes')
    ax.legend(fontsize=FONTSIZE_LEGEND)

    fig.tight_layout()
    fname = f'{output_dir}/cooccurrence_pooled.png'
    fig.savefig(fname, dpi=DPI_PRINT, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {fname}')


# =============================================================================
# Internal helpers
# =============================================================================

def _plot_scatter_panel(ax, df, dataset_id):
    """Scatter of co-drought fraction vs min storage, colored by FFMP zone."""
    # Size by severity (normalize to reasonable marker sizes)
    sev = df['severity'].values
    sizes = 15 + 80 * (sev - sev.min()) / (sev.max() - sev.min() + 1e-9)

    # Plot by zone for legend ordering
    for zone in _ZONE_ORDER:
        mask = df['ffmp_zone_at_min'] == zone
        if not mask.any():
            continue
        ax.scatter(
            df.loc[mask, 'co_drought_frac'],
            df.loc[mask, 'event_min_storage_pct'],
            c=FFMP_ZONE_COLORS[zone], s=sizes[mask.values],
            alpha=ALPHA_SCATTER, label=zone,
            edgecolors='white', linewidths=0.3,
        )

    # Trend line + annotation
    if len(df) > 10:
        _add_lowess_line(ax, df['co_drought_frac'], df['event_min_storage_pct'],
                         color='black')
        rho, pval = stats.spearmanr(df['co_drought_frac'], df['event_min_storage_pct'])
        ax.text(0.98, 0.02, f'$\\rho$ = {rho:.2f}\np = {pval:.1e}',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=FONTSIZE_MEDIUM,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.set_xlabel('Co-drought fraction', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('Min NYC storage (%)', fontsize=FONTSIZE_LABEL)
    ax.legend(fontsize=FONTSIZE_LEGEND - 1, loc='upper right',
              title='Zone at min', title_fontsize=FONTSIZE_LEGEND - 1)


def _plot_classification_boxes(ax, df):
    """Box plots of co-drought fraction grouped by satisficing classification."""
    class_order = ['all_pass', 'montague_fail', 'storage_fail', 'both_fail']
    class_labels = {
        'all_pass': 'Pass',
        'montague_fail': 'Mont. fail',
        'storage_fail': 'Stor. fail',
        'both_fail': 'Both fail',
    }
    class_colors = {
        'all_pass': '#A8D84E',
        'montague_fail': '#f9a825',
        'storage_fail': '#ef6c00',
        'both_fail': '#d32f2f',
    }

    box_data = []
    labels = []
    colors = []
    for cls in class_order:
        vals = df.loc[df['classification'] == cls, 'co_drought_frac']
        if len(vals) > 0:
            box_data.append(vals.values)
            labels.append(f'{class_labels[cls]}\n(n={len(vals)})')
            colors.append(class_colors[cls])

    if not box_data:
        return

    bp = ax.boxplot(box_data, labels=labels, patch_artist=True, widths=0.6)
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    for median in bp['medians']:
        median.set_color('black')

    ax.set_ylabel('Co-drought frac.', fontsize=FONTSIZE_LABEL)
    ax.set_xlabel('Satisficing classification', fontsize=FONTSIZE_LABEL)


def _add_lowess_line(ax, x, y, color='black', label=None, frac=0.6):
    """Add a LOWESS trend line to an axes."""
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 10:
            return
        result = lowess(y[valid], x[valid], frac=frac, it=3)
        ax.plot(result[:, 0], result[:, 1], color=color, linewidth=2,
                linestyle='--', alpha=0.8, label=label)
    except ImportError:
        # Fall back to simple linear fit
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 5:
            return
        slope, intercept, _, _, _ = stats.linregress(x[valid], y[valid])
        x_line = np.linspace(x[valid].min(), x[valid].max(), 100)
        ax.plot(x_line, slope * x_line + intercept, color=color,
                linewidth=2, linestyle='--', alpha=0.8, label=label)


def _concat_all(merged_dfs):
    """Concatenate all datasets into a single DataFrame."""
    import pandas as pd
    frames = []
    for did, df in merged_dfs.items():
        tmp = df.copy()
        tmp['dataset_id'] = did
        frames.append(tmp)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
