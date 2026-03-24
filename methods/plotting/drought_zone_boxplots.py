"""
Reusable boxplot functions for drought zone frequency and duration analysis.

Provides grouped boxplots comparing drought zone metrics across ensemble
scenarios, with consistent styling and formatting.
"""

import os
import numpy as np
import pandas as pd

from methods.load import load_annual_metrics
from methods.config import N_YEARS, RECONSTRUCTION_N_YEARS
from methods.plotting.styles import (
    DATASET_COLORS,
    FONTSIZE_SMALL, FONTSIZE_LABEL,
    label_panel,
)

# Default scenarios
DEFAULT_SCENARIOS = ['stationary_ensemble', 'climate_adjusted_low', 'climate_adjusted_high']


def style_boxplot(bp, colors_all, data_all, ax):
    """
    Style boxplot elements: color whiskers/caps/fliers by dataset,
    keep medians black, and add circle markers for means.
    """
    for i, color in enumerate(colors_all):
        # Box face
        bp['boxes'][i].set_facecolor(color)
        bp['boxes'][i].set_alpha(0.7)

        # Whiskers (2 per box: lower, upper)
        bp['whiskers'][2 * i].set_color(color)
        bp['whiskers'][2 * i + 1].set_color(color)

        # Caps (2 per box)
        bp['caps'][2 * i].set_color(color)
        bp['caps'][2 * i + 1].set_color(color)

        # Fliers / outliers
        bp['fliers'][i].set_markeredgecolor(color)
        bp['fliers'][i].set_markerfacecolor(color)
        bp['fliers'][i].set_alpha(0.5)

        # Mean circle marker
        vals = data_all[i]
        if len(vals) > 0:
            mean_val = np.mean(vals)
            ax.scatter(bp['medians'][i].get_xdata().mean(), mean_val,
                       marker='o', s=25, color=color, edgecolors='white',
                       linewidths=0.8, zorder=10)


def plot_frequency_boxplot(ax, panel_label='b)', show_historic=True,
                           scenarios=None):
    """
    Plot frequency boxplot showing distribution of years in each drought zone.

    X-axis groups: Watch, Warning, Emergency (3 groups)
    Per group: one boxplot per scenario showing the count of years
    each realization spent in that zone.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    panel_label : str
    show_historic : bool
    scenarios : list, optional
        Dataset IDs to plot. Defaults to DEFAULT_SCENARIOS.
    """
    if scenarios is None:
        scenarios = DEFAULT_SCENARIOS

    # Drought zones and order (left to right on x-axis)
    zone_values = [5, 4, 6]  # Watch=5, Warning=4, Emergency=6
    zone_labels = ['Watch', 'Warning', 'Emergency']

    # Load annual metrics for all datasets and compute zone frequency per realization
    all_freq_data = {}
    for dataset_id in scenarios:
        annual = load_annual_metrics(dataset_id)
        df_all = annual[annual['period'] == 'all'].copy()

        freq_data = {}
        for zv, label in zip(zone_values, zone_labels):
            # Count water years where max_zone == this zone, per realization
            counts = df_all.groupby('realization_id').apply(
                lambda g: (g['max_zone'] == zv).sum()
            )
            freq_data[label] = counts.values
        all_freq_data[dataset_id] = freq_data

    # Set up grouped box plot positions
    # When show_historic, reserve a slot for the historic marker so the
    # full group (historic + boxplots) is centered within each zone.
    n_zones = len(zone_labels)
    n_datasets = len(scenarios)
    n_slots = n_datasets + (1 if show_historic else 0)

    group_width = 0.8
    box_width = group_width / (n_slots + 0.5)
    slot_center = (n_slots - 1) / 2
    ds_start = 1 if show_historic else 0  # scenario slots start after historic

    positions_all = []
    colors_all = []
    data_all = []

    for zone_idx, label in enumerate(zone_labels):
        for ds_idx, dataset_id in enumerate(scenarios):
            x_pos = zone_idx + (ds_start + ds_idx - slot_center) * box_width
            positions_all.append(x_pos)
            colors_all.append(DATASET_COLORS[dataset_id])
            data_all.append(all_freq_data[dataset_id][label])

    # Create box plots
    bp = ax.boxplot(data_all, positions=positions_all, widths=box_width * 0.8,
                    patch_artist=True, showfliers=True,
                    boxprops=dict(linewidth=1.2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    medianprops=dict(linewidth=1.5, color='black'),
                    flierprops=dict(marker='o', markersize=3))

    style_boxplot(bp, colors_all, data_all, ax)

    # Format axes
    ax.set_xticks(range(n_zones))
    ax.set_xticklabels(zone_labels, fontsize=FONTSIZE_SMALL)
    ax.tick_params(axis='x', length=0)
    ax.set_ylabel(f'Number of Realization Years\nZone Experienced (out of {N_YEARS})', fontsize=FONTSIZE_LABEL)
    ax.set_xlim(-0.5, n_zones - 0.5)
    ax.set_ylim(bottom=0)
    ax.grid(False)
    ax.set_axisbelow(True)

    # Historic markers (scaled to ensemble year count for fair comparison)
    if show_historic:
        scale = N_YEARS / RECONSTRUCTION_N_YEARS
        hist_slot_x = (0 - slot_center) * box_width + box_width * 0.15
        try:
            hist_annual = load_annual_metrics('reconstruction')
            hist_all = hist_annual[hist_annual['period'] == 'all'].copy()
            for zone_idx, (zv, label) in enumerate(zip(zone_values, zone_labels)):
                hist_count = (hist_all['max_zone'] == zv).sum() * scale
                ax.scatter(zone_idx + hist_slot_x, hist_count, marker='^', s=60,
                           color='black', edgecolors='white', linewidths=0.5, zorder=10)
        except FileNotFoundError:
            hist_csv = './pywrdrb/performance_metrics/reconstruction_performance_metrics.csv'
            if os.path.exists(hist_csv):
                hist_perf = pd.read_csv(hist_csv)
                hist_col_map = {
                    'Watch': 'years_exactly_watch',
                    'Warning': 'years_exactly_warning',
                    'Emergency': 'years_exactly_emergency',
                }
                for zone_idx, label in enumerate(zone_labels):
                    col = hist_col_map[label]
                    hist_count = hist_perf[col].values[0] * scale
                    ax.scatter(zone_idx + hist_slot_x, hist_count, marker='^', s=60,
                               color='black', edgecolors='white', linewidths=0.5, zorder=10)

    # Vertical separators between zone groups
    for sep_x in [0.5, 1.5]:
        ax.axvline(sep_x, color='grey', linewidth=0.8, alpha=0.4, zorder=0)

    label_panel(ax, panel_label.rstrip(')'))


def plot_duration_boxplot(ax, panel_label='c)', show_historic=True,
                          scenarios=None):
    """
    Plot duration boxplot showing distribution of drought episode durations.

    X-axis groups: Watch, Warning, Emergency (3 groups)
    Per group: one boxplot per scenario showing individual episode durations
    (each episode is one data point, not a per-realization mean).

    Requires pre-computed zone duration events from
    pywrdrb/performance_metrics/{dataset_id}_zone_duration_events.csv
    (generated by S3_run_postprocessing.sh).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    panel_label : str
    show_historic : bool
    scenarios : list, optional
        Dataset IDs to plot. Defaults to DEFAULT_SCENARIOS.
    """
    if scenarios is None:
        scenarios = DEFAULT_SCENARIOS

    # Drought zones and order (left to right on x-axis)
    zone_order = [5, 4, 6]  # Watch, Warning, Emergency
    zone_labels_map = {4: 'Warning', 5: 'Watch', 6: 'Emergency'}

    # Load pre-computed episode records for all datasets
    all_duration_data = {}
    for dataset_id in scenarios:
        csv_fname = f'./pywrdrb/performance_metrics/{dataset_id}_zone_duration_events.csv'

        if not os.path.exists(csv_fname):
            raise FileNotFoundError(
                f"Zone duration events not found: {csv_fname}\n"
                f"Run S3_run_postprocessing.sh first to generate this file."
            )

        events_df = pd.read_csv(csv_fname)

        dataset_durations = {}
        for zone_num in zone_order:
            zone_events = events_df[events_df['max_zone'] == zone_num]
            dataset_durations[zone_num] = (zone_events['duration_days'] / 30.44).tolist()

        all_duration_data[dataset_id] = dataset_durations

    # Set up grouped box plot positions
    # When show_historic, reserve a slot for the historic markers so the
    # full group (historic + boxplots) is centered within each zone.
    n_zones = len(zone_order)
    n_datasets = len(scenarios)
    n_slots = n_datasets + (1 if show_historic else 0)

    group_width = 0.8
    box_width = group_width / (n_slots + 0.5)
    slot_center = (n_slots - 1) / 2
    ds_start = 1 if show_historic else 0

    positions_all = []
    colors_all = []
    data_all = []

    for zone_idx, zone_num in enumerate(zone_order):
        for ds_idx, dataset_id in enumerate(scenarios):
            x_pos = zone_idx + (ds_start + ds_idx - slot_center) * box_width
            positions_all.append(x_pos)
            colors_all.append(DATASET_COLORS[dataset_id])

            zone_durations = all_duration_data[dataset_id][zone_num]
            data_all.append(zone_durations)

    # Create box plots
    bp = ax.boxplot(data_all, positions=positions_all, widths=box_width * 0.8,
                    patch_artist=True, showfliers=True,
                    boxprops=dict(linewidth=1.2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    medianprops=dict(linewidth=1.5, color='black'),
                    flierprops=dict(marker='o', markersize=3))

    style_boxplot(bp, colors_all, data_all, ax)

    # Historic markers
    if show_historic:
        hist_slot_x = (0 - slot_center) * box_width + box_width * 0.15
        hist_csv = './pywrdrb/performance_metrics/reconstruction_zone_duration_events.csv'
        if os.path.exists(hist_csv):
            hist_events = pd.read_csv(hist_csv)
            for zone_idx, zone_num in enumerate(zone_order):
                hist_durations = (hist_events[hist_events['max_zone'] == zone_num]['duration_days'] / 30.44).values
                for d in hist_durations:
                    ax.scatter(zone_idx + hist_slot_x, d, marker='^', s=60,
                               color='black', edgecolors='white', linewidths=0.5, zorder=10)

    # Format axes
    ax.set_xticks(range(n_zones))
    ax.set_xticklabels([zone_labels_map[z] for z in zone_order], fontsize=FONTSIZE_SMALL)
    ax.tick_params(axis='x', length=0)
    ax.set_ylabel('Time Spent in Zone\nBefore Recovery (months)', fontsize=FONTSIZE_LABEL)
    ax.set_xlim(-0.5, n_zones - 0.5)
    ax.set_ylim(bottom=0, top=24)
    ax.grid(False)
    ax.set_axisbelow(True)

    # Vertical separators between zone groups
    for sep_x in [0.5, 1.5]:
        ax.axvline(sep_x, color='grey', linewidth=0.8, alpha=0.4, zorder=0)

    label_panel(ax, panel_label.rstrip(')'))
