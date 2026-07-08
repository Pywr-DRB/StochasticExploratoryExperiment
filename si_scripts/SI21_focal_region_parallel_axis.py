"""
Creates a parallel axis plot showing multiple dimensions of the drought periods defined in the 'focal region' in Fig9 and Fig10.

Dimensions (left-to-right), grouped by metric category:
  HAZARD / EXPOSURE:
    - Drought severity (peak |SSI|)
    - Drought magnitude (cumulative deficit)
    - Drought duration
    - Onset (intensification) rate  [expanded metric]
    - Recovery rate                 [expanded metric]
    - Peak SSI month
    - Antecedent wetness, prior 3 mo [expanded metric]
  OPERATIONAL:
    - Average weekly NYC diversion
    - Cumulative NYC directed releases to Montague
  OUTCOME:
    - NYC demand shortage (%)
    - NYC storage at drought start (%)
    - Minimum NYC reservoir storage (%)

Each line is a single SSI-3 focal-region drought; line colour encodes NYC
reservoir drawdown % during the event. Dynamics/antecedent metrics (onset
rate, recovery rate, prior-3mo wetness) are derived from the drought_events
SSI definitions and merged in. Brushing highlights high-consequence events.

"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from methods.config import (
    FIG_DIR, N_YEARS, GRID_N_BINS, DROUGHT_METRICS_DIR,
    MIN_COUNT_PER_BIN as MIN_COUNT,
    FOCAL_FRAC_THRESH, FOCAL_RP_THRESH_YEARS, FOCAL_WORST_STORAGE_THRESH,
    DATASET_CONFIGS,
)
from methods.load import load_event_metrics
from methods.return_period import (
    compute_return_period_grid_exceedance as compute_return_period_grid,
)
from methods.plotting.heatmap import (
    make_shared_edges_logmag, assign_grid_bins,
    compute_emergency_grid, compute_min_storage_grid,
    identify_focal_region, select_events_from_focal_region,
)
from methods.plotting.styles import (
    DATASET_LABELS, apply_publication_style, save_fig, DPI_HIGH,
)
from methods.plotting.parallel_axis import (
    custom_parallel_coordinates, apply_brush,
)


# -- configuration -----------------------------------------------------------

SSI_WINDOW = 3
DATASETS = list(DATASET_CONFIGS.keys())

FIG_OUTPUT_DIR = os.path.join(FIG_DIR, 'SI21_focal_region_parallel_axis')

# Parallel-axis dimensions (left -> right): (column, axis label, decimals)
# Grouped by category: HAZARD/EXPOSURE | OPERATIONAL | OUTCOME.
AXIS_SPEC = [
    # --- hazard / exposure ---
    ('severity',                    'Drought\nseverity',                  1),
    ('magnitude',                   'Drought\nmagnitude',                 0),
    ('duration_days',               'Duration\n(days)',                   0),
    ('onset_rate',                  'Onset rate\n(|SSI|/mo)',             2),
    ('recovery_rate',               'Recovery rate\n(|SSI|/mo)',          2),
    ('peak_severity_month',         'Peak SSI\nmonth',                    0),
    ('prior_3m_surplus',            'Antecedent\nwetness (3mo)',          1),
    # --- operational ---
    ('avg_weekly_nyc_diversion_mg', 'Avg weekly NYC\ndiversion (MG/wk)',  1),
    ('total_nyc_contribution_mg',   'Cum. NYC release\nto Montague (MG)', 0),
    # --- outcome ---
    ('nyc_shortage_pct',            'NYC demand\nshortage (%)',           1),
    ('storage_at_start_pct',        'NYC storage at\ndrought start (%)',  1),
    ('event_min_storage_pct',       'Min. NYC\nstorage (%)',              1),
]
AXIS_COLS = [c for c, _, _ in AXIS_SPEC]
AXIS_LABELS = [lbl for _, lbl, _ in AXIS_SPEC]

COLOR_COL = 'storage_drawdown_pct'
CMAP = 'plasma'

# Brushing: highlight high-consequence drought events; fade the rest to grey.
# AND-combined list of (column, operator, threshold). Edit freely / extend with
# additional conditions (e.g. ('nyc_shortage_pct', '>', 5.0)) as the storyline
# definition evolves. Set to None to disable brushing.
BRUSH_CONDITIONS = [
    ('nyc_shortage_pct', '>', 10.0),
    # ('event_min_storage_pct', '<', 25.0),
]


def load_hazard_dynamics(ds):
    """Derive expanded hazard-dynamics metrics from the drought_events CSV.

    Returns a frame keyed by (realization_id, start) with onset_rate,
    recovery_rate, peakedness, and antecedent prior_3m_surplus. These come
    from the SSI drought definition (recovery_period, max_severity_date), so
    no pipeline rerun is needed. See si_scripts/clustering/ for validation.
    """
    f = os.path.join(DROUGHT_METRICS_DIR,
                     f"{ds}_ssi{SSI_WINDOW}_drought_events.csv")
    de = pd.read_csv(f, parse_dates=['start', 'end', 'max_severity_date'])
    de['realization_id'] = de['realization_id'].astype(int)
    sev_abs = de['severity'].abs()
    # time-to-peak (months) = duration - recovery_period; floor denominators
    # at 0.5 month for the ~13% of events that peak in their first month.
    time_to_peak = (de['duration'] - de['recovery_period']).clip(lower=0.5)
    return pd.DataFrame({
        'realization_id': de['realization_id'],
        'start': de['start'],
        'onset_rate': sev_abs / time_to_peak,
        'recovery_rate': sev_abs / de['recovery_period'].clip(lower=0.5),
        'peakedness': sev_abs / de['avg_severity'].abs().replace(0, np.nan),
        'prior_3m_surplus': de['prior_3m_surplus'],
    })


def compute_focal_region(all_data):
    """Identify the focal-region grid cells (same procedure as Fig9/Fig10)."""
    sev_edges, mag_edges, _, _ = make_shared_edges_logmag(
        all_data, DATASETS, n_bins=GRID_N_BINS,
    )

    T_W_grids, frac_grids, min_grids = {}, {}, {}
    for ds in DATASETS:
        _, _, T_W_grids[ds], _ = compute_return_period_grid(
            all_data[ds], sev_edges, mag_edges, N_YEARS, min_count=MIN_COUNT)
        frac_grids[ds], _ = compute_emergency_grid(
            all_data[ds], sev_edges, mag_edges, min_count=MIN_COUNT)
        min_grids[ds], _ = compute_min_storage_grid(
            all_data[ds], sev_edges, mag_edges, min_count=MIN_COUNT)

    focal_cells = identify_focal_region(
        T_W_grids, frac_grids, min_grids, DATASETS,
        rp_thresh_years=FOCAL_RP_THRESH_YEARS,
        frac_thresh=FOCAL_FRAC_THRESH,
        storage_thresh=FOCAL_WORST_STORAGE_THRESH,
    )
    return sev_edges, mag_edges, focal_cells


def main():
    apply_publication_style()
    os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

    print(f"SI21: Focal-region drought parallel-axis plot (SSI-{SSI_WINDOW})")

    # 1. Load per-event metrics for each ensemble
    all_data = {ds: load_event_metrics(ds, SSI_WINDOW) for ds in DATASETS}

    # 2. Identify the focal region (jointly defined across all ensembles)
    sev_edges, mag_edges, focal_cells = compute_focal_region(all_data)
    print(f"  Focal region: {len(focal_cells)} cells")
    if len(focal_cells) == 0:
        print("  No focal cells found; nothing to plot.")
        return

    # 3. Select all focal-region events per ensemble
    focal_events = {}
    for ds in DATASETS:
        df_binned = assign_grid_bins(all_data[ds], sev_edges, mag_edges)
        sel = select_events_from_focal_region(df_binned, focal_cells, n=None)
        # Average weekly NYC diversion (MG/week) over the event window.
        # min_duration filtering in load_event_metrics guarantees duration > 0.
        sel = sel.assign(
            avg_weekly_nyc_diversion_mg=(
                sel['total_nyc_diversion_mg'] * 7.0 / sel['duration_days']),
        )
        # Merge expanded hazard-dynamics metrics (onset/recovery rate, etc.).
        sel['realization_id'] = sel['realization_id'].astype(int)
        sel['start'] = pd.to_datetime(sel['start'])
        dyn = load_hazard_dynamics(ds)
        sel = sel.merge(dyn, on=['realization_id', 'start'], how='left')
        n_missing = sel['onset_rate'].isna().sum()
        focal_events[ds] = sel
        print(f"  {ds}: {len(sel)} focal-region events "
              f"({n_missing} without dynamics match)")

    # 4. Shared per-axis bounds and shared colour range (pool all ensembles)
    pooled = np.concatenate(
        [focal_events[ds][AXIS_COLS].values for ds in DATASETS], axis=0)
    tops = np.nanmax(pooled, axis=0)
    bottoms = np.nanmin(pooled, axis=0)

    pooled_color = np.concatenate(
        [focal_events[ds][COLOR_COL].values for ds in DATASETS])
    vmin = float(np.nanmin(pooled_color))
    vmax = float(np.nanmax(pooled_color))

    # 5. Draw one panel per ensemble
    fig, axes = plt.subplots(
        len(DATASETS), 1, figsize=(18, 3.4 * len(DATASETS)))
    if len(DATASETS) == 1:
        axes = [axes]

    mappable = None
    for ax, ds in zip(axes, DATASETS):
        m = custom_parallel_coordinates(
            ax, focal_events[ds], AXIS_COLS,
            axis_labels=AXIS_LABELS,
            tops=tops, bottoms=bottoms,
            color_by_continuous=COLOR_COL,
            color_palette_continuous=CMAP,
            vmin=vmin, vmax=vmax,
            alpha_base=0.35, lw_base=0.7, fontsize=11,
            brush_conditions=BRUSH_CONDITIONS,
            alpha_brush=0.04, lw_brush=0.5,
        )
        mappable = mappable or m
        n_total = len(focal_events[ds])
        n_brush = int(apply_brush(focal_events[ds], BRUSH_CONDITIONS).sum())
        ax.set_title(
            f"{DATASET_LABELS[ds]}  "
            f"(n = {n_total:,} events; {n_brush:,} highlighted)",
            fontsize=13, fontweight='bold', pad=18)

    # 6. Shared colourbar
    if mappable is not None:
        mappable.set_array([])
        cbar = fig.colorbar(
            mappable, ax=axes, orientation='vertical',
            fraction=0.025, pad=0.02)
        cbar.set_label('NYC storage drawdown (%)', fontsize=12)

    brush_txt = (' AND '.join(f'{c} {op} {t:g}'
                              for c, op, t in BRUSH_CONDITIONS)
                 if BRUSH_CONDITIONS else 'none')
    fig.suptitle(
        f'Focal-region drought events (SSI-{SSI_WINDOW})\n'
        f'highlighted: {brush_txt}',
        fontsize=15, fontweight='bold', y=0.995)

    # 7. Save
    out_stem = os.path.join(FIG_OUTPUT_DIR, 'SI21_focal_region_parallel_axis')
    save_fig(fig, out_stem, dpi=DPI_HIGH)
    plt.close(fig)


if __name__ == '__main__':
    main()
