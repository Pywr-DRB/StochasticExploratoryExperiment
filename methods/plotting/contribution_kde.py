"""
Reusable KDE plotting function for NYC contribution/inflow ratio by drought zone.

Extracted from F3_plot_drought_contribution_composite.py so it can be
used in multiple figures.
"""

import numpy as np

from methods.plotting.styles import (
    FFMP_ZONE_COLORS_INT, FONTSIZE_LABEL, ALPHA_LINE,
)
from methods.plotting.water_balance_by_drought_zone import (
    DROUGHT_CATEGORIES,
    calculate_reconstruction_contribution_ratio,
    MIN_INFLOW_THRESHOLD,
    XLIM_MAX_MANUAL,
    N_MONTHS_PRIOR,
)

# Override DROUGHT_CATEGORIES colors to match FFMP zone colors from styles
DROUGHT_CATEGORIES['emergency']['color'] = FFMP_ZONE_COLORS_INT[6]   # '#d32f2f'
DROUGHT_CATEGORIES['watch']['color'] = FFMP_ZONE_COLORS_INT[5]       # '#ef6c00'
DROUGHT_CATEGORIES['warning']['color'] = FFMP_ZONE_COLORS_INT[4]     # '#f9a825'
DROUGHT_CATEGORIES['other']['color'] = 'limegreen'

# KDE categories to plot
KDE_CATEGORIES = ['emergency', 'watch', 'warning', 'other']


def plot_kde_panel(ax, categorized_data, n_months_prior=None, panel_label='a)'):
    """
    KDE of NYC contributions / inflow ratio by drought zone (stationary ensemble).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    categorized_data : dict
        Output from categorize_by_drought_zone(), keyed by drought category.
    n_months_prior : int, optional
        Window length in months. Defaults to N_MONTHS_PRIOR from F4 config.
    panel_label : str
        Panel label text (default 'a)').

    Returns
    -------
    tuple
        (handles, labels) from ax.get_legend_handles_labels()
    """
    if n_months_prior is None:
        n_months_prior = N_MONTHS_PRIOR
    categories = KDE_CATEGORIES
    category_data = {}

    for cat in categories:
        cat_info = DROUGHT_CATEGORIES[cat]
        df = categorized_data[cat].copy()
        if len(df) == 0:
            continue
        df_filtered = df[df['inflow_total'] > MIN_INFLOW_THRESHOLD]
        if len(df_filtered) == 0:
            continue
        ratio = 100.0 * df_filtered['contribution_total'] / df_filtered['inflow_total']
        category_data[cat] = {'ratio': ratio, 'n': len(df_filtered)}

    # Determine x-axis max
    xlim_max = XLIM_MAX_MANUAL if XLIM_MAX_MANUAL is not None else 100

    # Plot KDEs
    for cat in categories:
        if cat not in category_data:
            continue
        cat_info = DROUGHT_CATEGORIES[cat]
        ratio = category_data[cat]['ratio']
        n = category_data[cat]['n']

        label = f"{cat_info['label']} (n={n})" if cat != 'other' else f"Normal or Above (n={n})"
        ratio.plot.kde(ax=ax, color=cat_info['color'], linewidth=2.5, alpha=ALPHA_LINE, label=label)
        mean_val = ratio.mean()
        ax.axvline(mean_val, color=cat_info['color'], linestyle='--', linewidth=1.5, alpha=0.7)

    # Dummy for mean legend entry
    ax.axvline(np.nan, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Mean')

    # 1964 reconstruction
    reconstruction_ratio = calculate_reconstruction_contribution_ratio()
    if reconstruction_ratio is not None:
        ax.axvline(reconstruction_ratio, color='black', linestyle='-', linewidth=2.5, alpha=0.9, label='1964 Drought')
        if reconstruction_ratio > xlim_max:
            xlim_max = reconstruction_ratio * 1.1

    xlabel = f'NYC contributions / total inflow\n({n_months_prior}-mo prior to min zone, %)'
    ax.set_xlabel(xlabel, fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('Density', fontsize=FONTSIZE_LABEL)
    ax.set_xlim(left=0, right=xlim_max)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Panel label
    ax.text(0.02, 0.97, panel_label, transform=ax.transAxes, fontsize=12, va='top', ha='left')

    return ax.get_legend_handles_labels()
