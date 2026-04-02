"""
Shared plotting and summary functions for shortage-by-zone analyses.

Used by:
  - si_scripts/SI5_plot_shortages_by_zone.py
  - si_scripts/SI10_plot_montague_trenton_shortage_by_zone.py

These functions visualise how shortage days and volumes distribute across
FFMP drought zones (or custom zone definitions that include sub-zones).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from methods.plotting.styles import (
    DATASET_LABELS,
    DPI_HIGH, FONTSIZE_SMALL, FONTSIZE_MEDIUM, FONTSIZE_LARGE,
)


# ============================================================================
# Summary printing
# ============================================================================

def print_summary_statistics(stats_df, dataset_id, title_label,
                             drought_zones=None, normal_zone=None):
    """
    Print shortage-by-zone summary statistics to console.

    Parameters
    ----------
    stats_df : pd.DataFrame
        Zone-level summary with columns ``zone``, ``zone_name``,
        ``n_shortage_days``, ``total_shortage_mg``,
        ``pct_of_all_shortage_days``, ``pct_of_all_shortage_volume``.
    dataset_id : str
        Dataset identifier (used for display label).
    title_label : str
        Human-readable label for the shortage type, e.g.
        ``'NYC Diversion'`` or ``'Montague'``.
    drought_zones : list[int] or None
        Zone numbers considered drought (for detailed breakdown).
        If *None*, the detailed drought-vs-normal breakdown is skipped.
    normal_zone : int or None
        Zone number for normal operations (for detailed breakdown).
    """
    print(f"\n{'='*80}")
    print(f"{title_label.upper()} SHORTAGE ANALYSIS: "
          f"{DATASET_LABELS.get(dataset_id, dataset_id)}")
    print(f"{'='*80}")

    print(f"\n{'Zone':<25} {'Days':>10} {'% Days':>10} "
          f"{'Volume (MG)':>15} {'% Volume':>10}")
    print("-" * 72)

    for _, row in stats_df.iterrows():
        if row['n_shortage_days'] == 0:
            continue
        print(f"{row['zone_name']:<25} "
              f"{row['n_shortage_days']:>10,} "
              f"{row['pct_of_all_shortage_days']:>10.1f} "
              f"{row['total_shortage_mg']:>15,.0f} "
              f"{row['pct_of_all_shortage_volume']:>10.1f}")

    print("-" * 72)

    # Detailed drought-vs-normal breakdown (opt-in)
    if drought_zones is not None and normal_zone is not None:
        total_days = stats_df['n_shortage_days'].sum()
        total_volume = stats_df['total_shortage_mg'].sum()

        if total_days > 0 and total_volume > 0:
            drought_days = stats_df[
                stats_df['zone'].isin(drought_zones)
            ]['n_shortage_days'].sum()
            drought_volume = stats_df[
                stats_df['zone'].isin(drought_zones)
            ]['total_shortage_mg'].sum()

            normal_days = stats_df[
                stats_df['zone'] == normal_zone
            ]['n_shortage_days'].sum()
            normal_volume = stats_df[
                stats_df['zone'] == normal_zone
            ]['total_shortage_mg'].sum()

            print(f"\nKEY FINDINGS:")
            print(f"  Total shortage days: {total_days:,}")
            print(f"  Total shortage volume: {total_volume:,.0f} MG")
            print(f"\n  Drought zones:")
            print(f"    Days: {drought_days:,} "
                  f"({100*drought_days/total_days:.1f}%)")
            print(f"    Volume: {drought_volume:,.0f} MG "
                  f"({100*drought_volume/total_volume:.1f}%)")
            print(f"\n  Normal zone:")
            print(f"    Days: {normal_days:,} "
                  f"({100*normal_days/total_days:.1f}%)")
            print(f"    Volume: {normal_volume:,.0f} MG "
                  f"({100*normal_volume/total_volume:.1f}%)")

    print("=" * 80)


# ============================================================================
# Bar chart: shortage days & volume by zone
# ============================================================================

def plot_shortage_by_zone_summary(stats_df, dataset_id, title_label,
                                  zone_definitions, zones_to_plot,
                                  fig_output_dir, fname=None):
    """
    Two-panel bar chart showing % of shortage days and % of shortage
    volume by zone.

    Parameters
    ----------
    stats_df : pd.DataFrame
        Zone-level summary (see ``print_summary_statistics`` for schema).
    dataset_id : str
        Dataset identifier.
    title_label : str
        Display label for the shortage type.
    zone_definitions : dict
        ``{zone_num: {'name': ..., 'color': ...}}``
    zones_to_plot : list[int]
        Ordered list of zone numbers to include (most severe first).
    fig_output_dir : str
        Directory for saving the figure.
    fname : str or None
        Explicit output path.  If *None* a default is generated.
    """
    fig, (ax_days, ax_vol) = plt.subplots(1, 2, figsize=(13, 5))

    zone_names = [zone_definitions[z]['name'] for z in zones_to_plot]
    colors = [zone_definitions[z]['color'] for z in zones_to_plot]
    x = np.arange(len(zones_to_plot))

    # Look up values by zone (handles missing zones gracefully)
    pct_days = np.array([
        stats_df.loc[stats_df['zone'] == z, 'pct_of_all_shortage_days'].values[0]
        if len(stats_df[stats_df['zone'] == z]) else 0
        for z in zones_to_plot
    ])
    pct_vol = np.array([
        stats_df.loc[stats_df['zone'] == z, 'pct_of_all_shortage_volume'].values[0]
        if len(stats_df[stats_df['zone'] == z]) else 0
        for z in zones_to_plot
    ])

    # Left panel: shortage days
    bars1 = ax_days.bar(x, pct_days, color=colors, alpha=0.8,
                        edgecolor='black')
    ax_days.set_xticks(x)
    ax_days.set_xticklabels(zone_names, fontsize=FONTSIZE_SMALL)
    ax_days.set_ylabel('% of All Shortage Days', fontsize=FONTSIZE_MEDIUM)
    ax_days.set_xlabel('FFMP Zone', fontsize=FONTSIZE_MEDIUM)
    ax_days.set_title(f'{title_label} Shortage Days by Zone',
                      fontsize=FONTSIZE_LARGE)
    ax_days.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax_days.set_axisbelow(True)

    for bar, pct in zip(bars1, pct_days):
        if pct > 0:
            ax_days.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 1,
                         f'{pct:.1f}%', ha='center', va='bottom',
                         fontsize=FONTSIZE_SMALL)

    # Right panel: shortage volume
    bars2 = ax_vol.bar(x, pct_vol, color=colors, alpha=0.8,
                       edgecolor='black')
    ax_vol.set_xticks(x)
    ax_vol.set_xticklabels(zone_names, fontsize=FONTSIZE_SMALL)
    ax_vol.set_ylabel('% of Total Shortage Volume', fontsize=FONTSIZE_MEDIUM)
    ax_vol.set_xlabel('FFMP Zone', fontsize=FONTSIZE_MEDIUM)
    ax_vol.set_title(f'{title_label} Shortage Volume by Zone',
                     fontsize=FONTSIZE_LARGE)
    ax_vol.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax_vol.set_axisbelow(True)

    for bar, pct in zip(bars2, pct_vol):
        if pct > 0:
            ax_vol.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 1,
                        f'{pct:.1f}%', ha='center', va='bottom',
                        fontsize=FONTSIZE_SMALL)

    dataset_label = DATASET_LABELS.get(dataset_id, dataset_id)
    fig.suptitle(f'{title_label} Shortage Analysis: {dataset_label}',
                 fontsize=FONTSIZE_LARGE, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if fname is None:
        label_slug = title_label.replace(' ', '_').lower()
        fname = f"{fig_output_dir}/shortage_by_zone_{label_slug}_{dataset_id}.png"

    plt.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close()


# ============================================================================
# Histogram + box plot: shortage magnitude distributions
# ============================================================================

def plot_shortage_magnitude_distributions(all_shortages, dataset_id,
                                          shortage_col, title_label,
                                          zone_definitions, zones_to_plot,
                                          zone_col, fig_output_dir,
                                          fname=None):
    """
    Two-panel figure: stacked histogram and box plot of shortage
    magnitudes coloured by zone.

    Parameters
    ----------
    all_shortages : pd.DataFrame
        All shortage-day records (one row per day with shortage > 0).
    dataset_id : str
        Dataset identifier.
    shortage_col : str
        Column name holding shortage values.
    title_label : str
        Display label for the shortage type.
    zone_definitions : dict
        ``{zone_num: {'name': ..., 'color': ...}}``
    zones_to_plot : list[int]
        Ordered list of zone numbers to include.
    zone_col : str
        Column in *all_shortages* holding the zone assignment.
    fig_output_dir : str
        Directory for saving the figure.
    fname : str or None
        Explicit output path.
    """
    fig, (ax_hist, ax_box) = plt.subplots(2, 1, figsize=(14, 10))

    zone_data_list = []
    zone_labels_list = []
    zone_colors_list = []
    box_data = []
    box_labels = []
    box_colors = []

    for z in zones_to_plot:
        vals = all_shortages.loc[all_shortages[zone_col] == z, shortage_col]
        if len(vals) == 0:
            continue
        zone_info = zone_definitions[z]
        clean_name = zone_info['name'].replace('\n', ' ')
        zone_data_list.append(vals)
        zone_labels_list.append(f"{clean_name} (n={len(vals):,})")
        zone_colors_list.append(zone_info['color'])
        box_data.append(vals.values)
        box_labels.append(clean_name)
        box_colors.append(zone_info['color'])

    if not zone_data_list:
        plt.close()
        return

    # Adaptive bins based on the 99th percentile
    max_shortage = np.percentile(
        np.concatenate([v.values for v in zone_data_list]), 99
    )
    bins = np.linspace(0, max_shortage * 1.1, 30)

    # Top: stacked histogram
    ax_hist.hist(zone_data_list, bins=bins, stacked=True,
                 color=zone_colors_list, label=zone_labels_list,
                 edgecolor='black', linewidth=0.3, alpha=0.8)
    ax_hist.set_xlabel('Shortage Magnitude (MGD)', fontsize=FONTSIZE_MEDIUM)
    ax_hist.set_ylabel('Frequency (days)', fontsize=FONTSIZE_MEDIUM)
    ax_hist.set_title(f'Distribution of {title_label} Shortage Magnitudes by Zone',
                      fontsize=FONTSIZE_LARGE)
    ax_hist.legend(fontsize=FONTSIZE_SMALL, loc='upper right')
    ax_hist.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax_hist.set_axisbelow(True)

    # Bottom: box plot
    bp = ax_box.boxplot(box_data, labels=box_labels, patch_artist=True,
                        showfliers=False, widths=0.6,
                        medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax_box.set_ylabel('Shortage Magnitude (MGD)', fontsize=FONTSIZE_MEDIUM)
    ax_box.set_xlabel('FFMP Zone', fontsize=FONTSIZE_MEDIUM)
    ax_box.set_title('Shortage Distribution by Zone', fontsize=FONTSIZE_MEDIUM)
    ax_box.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax_box.set_axisbelow(True)
    ax_box.tick_params(axis='x', labelsize=FONTSIZE_SMALL)

    dataset_label = DATASET_LABELS.get(dataset_id, dataset_id)
    fig.suptitle(f'{title_label} Shortage Magnitude Analysis: {dataset_label}',
                 fontsize=FONTSIZE_LARGE, y=0.995)
    plt.tight_layout()

    if fname is None:
        label_slug = title_label.replace(' ', '_').lower()
        fname = (f"{fig_output_dir}/"
                 f"shortage_magnitude_distribution_{label_slug}_{dataset_id}.png")

    plt.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close()
