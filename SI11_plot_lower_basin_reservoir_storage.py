"""
SI Figure: Blue Marsh and Beltzville reservoir storage across ensemble scenarios.

Shows median and 1st-99th percentile range of storage (% capacity) by water year week.

Usage:
    python SI_plot_lower_basin_reservoir_storage.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore")

import pywrdrb
from methods.config import ROOT_DIR, FIG_DIR, PERIOD_ORIGIN
from methods.utils import calculate_water_year_period_index
from methods.plotting.styles import (
    DATASET_COLORS, DATASET_LABELS, DATASET_LINESTYLES,
    FONTSIZE_LABEL, FONTSIZE_MEDIUM, DPI_HIGH, apply_publication_style,
)

FIG_OUTPUT_DIR = f"{FIG_DIR}/SI_lower_basin_storage"
os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

DATASETS = ['stationary_ensemble', 'climate_adjusted_low', 'climate_adjusted_high']

RESERVOIRS = {
    'blueMarsh': 'Blue Marsh',
    'beltzvilleCombined': 'Beltzville',
}

WY_MONTH_STARTS = [1, 5, 9, 14, 18, 23, 27, 32, 36, 40, 45, 49]
WY_MONTH_LABELS = ['Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov',
                    'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May']


def compute_weekly_percentiles(dataset_id, res_col):
    """Compute weekly storage percentiles for a single reservoir across all realizations."""
    fname = f'{ROOT_DIR}/pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5'
    data = pywrdrb.Data()
    data.load_from_export(fname, results_sets=['res_storage'])

    realizations = sorted(data.res_storage[dataset_id].keys())
    sample_idx = data.res_storage[dataset_id][realizations[0]].index
    p_idx = calculate_water_year_period_index(sample_idx, period='weekly', origin=PERIOD_ORIGIN)
    periods = np.sort(np.unique(p_idx))

    # Get capacity from max observed storage across all realizations
    capacity = max(data.res_storage[dataset_id][r][res_col].max() for r in realizations)

    storage_by_period = {p: [] for p in periods}
    for r in realizations:
        stor_pct = 100.0 * data.res_storage[dataset_id][r][res_col] / capacity
        p_idx_r = calculate_water_year_period_index(stor_pct.index, period='weekly', origin=PERIOD_ORIGIN)
        for p in periods:
            storage_by_period[p].extend(stor_pct[p_idx_r == p].values)

    rows = []
    for p in periods:
        v = np.array(storage_by_period[p])
        rows.append({
            'period': p,
            'p1': np.percentile(v, 1),
            'p50': np.percentile(v, 50),
            'p99': np.percentile(v, 99),
        })
    return pd.DataFrame(rows).set_index('period')


def plot_figure():
    apply_publication_style()
    plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12})

    res_keys = list(RESERVOIRS.keys())
    fig, axes = plt.subplots(len(res_keys), 1, figsize=(10, 6.5),
                             sharex=True, gridspec_kw={'hspace': 0.08})

    for i, (res_col, res_name) in enumerate(RESERVOIRS.items()):
        ax = axes[i]
        print(f"Processing {res_name}...")

        for did in DATASETS:
            pct = compute_weekly_percentiles(did, res_col)
            w = pct.index.values
            color = DATASET_COLORS[did]
            ls = DATASET_LINESTYLES.get(did, '-')

            ax.fill_between(w, pct['p1'], pct['p99'],
                            color=color, alpha=0.07, linewidth=0)
            ax.plot(w, pct['p50'], color=color, linewidth=2.5,
                    linestyle=ls, alpha=0.95, zorder=4)
            ax.plot(w, pct['p1'], color=color, linewidth=1.2,
                    linestyle='--', alpha=0.5, zorder=3)

        ax.set_ylim(0, 105)
        ax.set_ylabel(f'{res_name}\nStorage (% capacity)', fontsize=FONTSIZE_LABEL)
        ax.grid(True, alpha=0.12, linestyle='--')
        ax.set_axisbelow(True)
        ax.text(0.015, 0.95, f'({chr(97+i)})', transform=ax.transAxes,
                fontsize=14, va='top', fontweight='bold')

    # X-axis on bottom panel only
    axes[-1].set_xticks(WY_MONTH_STARTS)
    axes[-1].set_xticklabels(WY_MONTH_LABELS, fontsize=FONTSIZE_MEDIUM)
    axes[-1].set_xlim(0.5, 52.5)
    axes[-1].set_xlabel('Month (Water Year)', fontsize=FONTSIZE_LABEL)

    # Legend
    handles = []
    for did in DATASETS:
        handles.append(Line2D([0], [0], color=DATASET_COLORS[did], linewidth=2.5,
                              linestyle=DATASET_LINESTYLES.get(did, '-'),
                              label=DATASET_LABELS[did]))
    handles.append(Line2D([0], [0], color='grey', linewidth=2.5, linestyle='-',
                          alpha=0.9, label='Median'))
    handles.append(Line2D([0], [0], color='grey', linewidth=1.2, linestyle='--',
                          alpha=0.5, label='1st Percentile'))

    fig.legend(handles=handles, loc='lower center', ncol=5,
               fontsize=10, frameon=False, bbox_to_anchor=(0.54, -0.02))

    fname = f"{FIG_OUTPUT_DIR}/lower_basin_reservoir_storage.png"
    fig.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"Saved: {fname}")
    plt.close(fig)


if __name__ == '__main__':
    plot_figure()
