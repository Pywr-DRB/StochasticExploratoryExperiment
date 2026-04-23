"""
Fig10 alternative: Focal-event DOY quantile gradient (3x3 grid).

Shows the distribution of focal-zone drought water-years as a BrBG gradient
background at each day-of-water-year, with the single worst-case trajectory
overlaid, for three variables x three climate scenarios.

Rows (top -> bottom):
  (a) NYC total storage (%)
  (b) NYC release to Montague (MGD)
  (c) Montague streamflow (MGD)

Columns (left -> right):
  stationary_ensemble, climate_adjusted_low, climate_adjusted_high

Multi-metric focal-region criteria match Fig9/Fig10 exactly. Strict thresholds
are preserved as the default for the final run; TEST_MODE relaxes them so the
script yields a populated figure on the local 5-realization dataset.

Usage:
    python Fig10alt_focal_doy_quantiles.py
"""

import sys
import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

import pywrdrb
from methods.config import (
    FIG_DIR, OUTPUT_DIR, N_YEARS, NYC_RESERVOIRS, NYC_TOTAL_CAPACITY,
    RECONSTRUCTION_OUTPUT_FNAME, DATASET_CONFIGS,
)
from methods.water_year import (
    MONTH_STARTS_WY, vectorized_water_year_doy,
)
from methods.load import load_event_metrics, load_rank_subset_from_export
from methods.plotting.heatmap import (
    make_shared_edges_logmag, assign_grid_bins,
    compute_exceedance_rate_grid, compute_emergency_grid,
    compute_min_storage_grid, identify_focal_region,
    select_events_from_focal_region,
    GRID_N_BINS, FOCAL_FRAC_THRESH, FOCAL_RATE_THRESH, WORST_STORAGE_THRESH,
)
from methods.plotting.drought_dynamics import (
    extract_drought_timeseries,
    align_to_water_year,
    compute_fixed_extraction_window,
)
from methods.plotting.percentile_bands import format_xaxis_water_year
from methods.plotting.styles import (
    DATASET_LABELS, HISTORIC_COLOR, CMAP_DIVERGING,
    DPI_PRINT, FONTSIZE_LABEL, FONTSIZE_TITLE, FONTSIZE_LEGEND,
    apply_publication_style,
)
from methods.plotting.doy_quantile_gradient import (
    compute_doy_quantile_grid, plot_doy_quantile_gradient,
)


# ── Configuration ────────────────────────────────────────────────────────

SSI_WINDOW = 12
MIN_COUNT = 1

DATASETS = list(DATASET_CONFIGS.keys())
RESULTS_SETS = ['inflow', 'res_storage', 'contribution', 'major_flow']

# TEST_MODE relaxes focal thresholds when the strict Fig9/Fig10 region is empty
# on the local 5-realization dataset. Final-run default: TEST_MODE = False.
TEST_MODE = False

STRICT_THRESHOLDS = dict(
    rate_thresh=FOCAL_RATE_THRESH,
    frac_thresh=FOCAL_FRAC_THRESH,
    storage_thresh=WORST_STORAGE_THRESH,
)
RELAXED_THRESHOLDS = dict(
    rate_thresh=0.0,
    frac_thresh=2.0,        # accept any non-NaN frac (>1 always true)
    storage_thresh=200.0,   # accept any non-NaN min storage
)

REFERENCE_WY_START = pd.Timestamp('2000-06-01')
REFERENCE_WY_END = pd.Timestamp('2001-05-31')
N_QUANTILE_LEVELS = 21    # 0, 5, 10, ..., 100 %

CMAP = CMAP_DIVERGING     # BrBG

# (variable_key, y-label, y-scale, rolling-mean window in days)
VARIABLES = [
    ('nyc_storage_pct', 'NYC Total\nStorage (%)',          'linear', 1),
    ('nyc_release',     'NYC Releases to\nMontague (MGD)', 'linear', 7),
    ('montague_flow',   'Montague Flow\n(MGD)',            'log',    7),
]

FIG_OUTPUT_DIR = os.path.join(FIG_DIR, 'Fig10')
FIG_NAME = f'Fig10alt_focal_doy_quantiles_ssi{SSI_WINDOW}.png'
if TEST_MODE:
    FIG_NAME = FIG_NAME.replace('.png', '_TESTMODE.png')


# ── Helpers ──────────────────────────────────────────────────────────────

def _traces_to_doy_df(traces_by_event):
    """Stack a list of (event_id, pd.Series) into a DOY-indexed DataFrame."""
    frames = {}
    for event_id, s in traces_by_event:
        if s is None or len(s) == 0:
            continue
        doy = vectorized_water_year_doy(s.index)
        frames[event_id] = pd.Series(s.values, index=doy)
    if not frames:
        return pd.DataFrame()
    return pd.DataFrame(frames).sort_index()


def load_reconstruction_annual_cycle():
    """Mean annual cycle (by DOY) for the three plotted variables."""
    if not os.path.exists(RECONSTRUCTION_OUTPUT_FNAME):
        print(f"  Warning: reconstruction file not found at "
              f"{RECONSTRUCTION_OUTPUT_FNAME}")
        return None
    try:
        data = pywrdrb.Data()
        data.load_output(
            output_filenames=[RECONSTRUCTION_OUTPUT_FNAME],
            results_sets=['res_storage', 'major_flow', 'nyc_release_components'],
        )
        ds = list(data.res_storage.keys())[0]
        r = list(data.res_storage[ds].keys())[0]

        storage_raw = data.res_storage[ds][r][NYC_RESERVOIRS].sum(axis=1)
        nyc_storage_pct = 100.0 * storage_raw / NYC_TOTAL_CAPACITY

        contrib_cols = [f'mrf_montagueTrenton_{res}' for res in NYC_RESERVOIRS]
        nyc_release = data.nyc_release_components[ds][r][contrib_cols].sum(axis=1)

        montague_flow = data.major_flow[ds][r]['delMontague']
    except Exception as e:
        print(f"  Warning: Error loading reconstruction: {e}")
        return None

    out = {}
    for name, series in [
        ('nyc_storage_pct', nyc_storage_pct),
        ('nyc_release', nyc_release),
        ('montague_flow', montague_flow),
    ]:
        doy = vectorized_water_year_doy(series.index)
        df = pd.DataFrame({'doy': doy, 'value': series.values})
        out[name] = df.groupby('doy')['value'].mean().sort_index()
    return out


def _slice_variable_trace(aligned, var_name, smooth_window=1):
    s = aligned[var_name]
    if smooth_window and smooth_window > 1:
        s = s.rolling(smooth_window, center=True, min_periods=1).mean()
    mask = (s.index >= REFERENCE_WY_START) & (s.index <= REFERENCE_WY_END)
    return s[mask]


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    apply_publication_style()
    os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

    print(f"Fig10alt: DOY quantile gradient "
          f"(TEST_MODE={TEST_MODE}, SSI_WINDOW={SSI_WINDOW})")

    # 1. Event metrics per dataset
    all_data = {}
    for ds in DATASETS:
        all_data[ds] = load_event_metrics(ds, SSI_WINDOW)
        print(f"  {ds}: {len(all_data[ds])} events")

    # 2. Shared severity x magnitude grid
    sev_edges, mag_edges, _, _ = make_shared_edges_logmag(
        all_data, DATASETS, n_bins=GRID_N_BINS,
    )

    # 3. Build per-dataset grids and identify focal cells
    rate_grids, frac_grids, min_grids = {}, {}, {}
    for ds in DATASETS:
        rate_grids[ds], _ = compute_exceedance_rate_grid(
            all_data[ds], sev_edges, mag_edges, N_YEARS, min_count=MIN_COUNT)
        frac_grids[ds], _ = compute_emergency_grid(
            all_data[ds], sev_edges, mag_edges, min_count=MIN_COUNT)
        min_grids[ds], _ = compute_min_storage_grid(
            all_data[ds], sev_edges, mag_edges, min_count=MIN_COUNT)

    thresholds = STRICT_THRESHOLDS
    focal_cells = identify_focal_region(
        rate_grids, frac_grids, min_grids, DATASETS, **thresholds)
    print(f"  Strict focal region: {len(focal_cells)} cells")

    if len(focal_cells) == 0 and TEST_MODE:
        thresholds = RELAXED_THRESHOLDS
        focal_cells = identify_focal_region(
            rate_grids, frac_grids, min_grids, DATASETS, **thresholds)
        print(f"  TEST_MODE relaxed focal region: {len(focal_cells)} cells")

    if len(focal_cells) == 0:
        raise RuntimeError(
            "No focal cells found (strict thresholds empty and TEST_MODE=False)."
        )

    # 4. Per-dataset: select focal events + load + extract 1-WY traces
    dataset_traces = {}       # ds -> {var: DOY-indexed DataFrame}
    dataset_worst = {}        # ds -> {var: pd.Series indexed by DOY}
    dataset_worst_meta = {}   # ds -> dict (for print/reference)

    for ds in DATASETS:
        df_binned = assign_grid_bins(all_data[ds], sev_edges, mag_edges)
        selected = select_events_from_focal_region(
            df_binned, focal_cells,
            rank_col='event_min_storage_pct', ascending=True, n=None,
        )
        print(f"  {ds}: {len(selected)} focal events")

        if len(selected) == 0:
            dataset_traces[ds] = {v[0]: pd.DataFrame() for v in VARIABLES}
            dataset_worst[ds] = {v[0]: None for v in VARIABLES}
            dataset_worst_meta[ds] = None
            continue

        unique_reals = sorted(set(int(r) for r in selected['realization_id']))
        fname = os.path.join(OUTPUT_DIR, f'{ds}_with_postprocessing.hdf5')
        data = load_rank_subset_from_export(
            fname, unique_reals, RESULTS_SETS, rank=0, size=1,
        )

        per_var_traces = {v[0]: [] for v in VARIABLES}
        worst_event_id = None

        for event_idx, (_, row) in enumerate(selected.iterrows()):
            event_id = (f"R{int(row['realization_id']):04d}_"
                        f"{pd.Timestamp(row['start']).date()}")
            min_storage_date = pd.Timestamp(row['min_storage_date'])

            w_start, w_end = compute_fixed_extraction_window(
                min_storage_date, pad_before_wy=0, pad_after_wy=0,
            )
            ts = extract_drought_timeseries(
                data, ds, int(row['realization_id']), w_start, w_end,
            )
            aligned, _, _ = align_to_water_year(
                ts, row['start'], row['end'], min_storage_date,
                reference_wy_start=REFERENCE_WY_START,
            )
            for var_name, _, _, smooth in VARIABLES:
                per_var_traces[var_name].append(
                    (event_id, _slice_variable_trace(
                        aligned, var_name, smooth_window=smooth))
                )

            if event_idx == 0:
                worst_event_id = event_id
                dataset_worst_meta[ds] = {
                    'event_id': event_id,
                    'realization_id': int(row['realization_id']),
                    'start': pd.Timestamp(row['start']),
                    'min_storage_date': min_storage_date,
                    'min_storage_pct': float(row['event_min_storage_pct']),
                }

        dataset_traces[ds] = {
            var_name: _traces_to_doy_df(per_var_traces[var_name])
            for var_name, _, _, _ in VARIABLES
        }
        dataset_worst[ds] = {}
        for var_name, _, _, _ in VARIABLES:
            traces_df = dataset_traces[ds][var_name]
            if (worst_event_id is not None
                    and worst_event_id in traces_df.columns):
                dataset_worst[ds][var_name] = traces_df[worst_event_id]
            else:
                dataset_worst[ds][var_name] = None

        if dataset_worst_meta[ds]:
            meta = dataset_worst_meta[ds]
            print(f"    worst-case: {meta['event_id']} "
                  f"(min_storage={meta['min_storage_pct']:.1f}%)")

        del data
        gc.collect()

    # 5. Reconstruction mean annual cycle (shared reference across columns)
    print("\nLoading reconstruction mean annual cycle...")
    recon = load_reconstruction_annual_cycle()

    # 6. Figure
    fig, axes = plt.subplots(
        len(VARIABLES), len(DATASETS),
        figsize=(3.8 * len(DATASETS), 2.8 * len(VARIABLES)),
        sharex='col', sharey='row',
    )
    if len(VARIABLES) == 1:
        axes = np.array([axes])
    if len(DATASETS) == 1:
        axes = axes[:, None]

    sm_for_colorbar = None
    worst_handle = None
    recon_handle = None

    for row_idx, (var_name, var_label, yscale, _) in enumerate(VARIABLES):
        for col_idx, ds in enumerate(DATASETS):
            ax = axes[row_idx, col_idx]
            traces_df = dataset_traces[ds][var_name]

            if traces_df.shape[1] >= 2:
                traces_df = traces_df.reindex(range(1, 367))
                grid = compute_doy_quantile_grid(
                    traces_df, n_levels=N_QUANTILE_LEVELS,
                )
                sm = plot_doy_quantile_gradient(
                    ax, grid, cmap=CMAP, vmin=0.0, vmax=1.0,
                )
                sm_for_colorbar = sm

            # Reconstruction mean (dashed black)
            if recon is not None and var_name in recon:
                rs = recon[var_name]
                ln, = ax.plot(
                    rs.index, rs.values,
                    color=HISTORIC_COLOR, linestyle='--', linewidth=1.5,
                    alpha=0.85, zorder=5,
                )
                if recon_handle is None:
                    recon_handle = ln

            # Worst-case focal trajectory (solid red)
            worst = dataset_worst[ds][var_name]
            if worst is not None and worst.dropna().size > 0:
                w = worst.sort_index()
                ln, = ax.plot(
                    w.index, w.values,
                    color='#c0392b', linewidth=2.0, alpha=0.95, zorder=6,
                )
                if worst_handle is None:
                    worst_handle = ln

            if yscale == 'log':
                ax.set_yscale('log')

            if row_idx == len(VARIABLES) - 1:
                format_xaxis_water_year(ax)
            else:
                ax.set_xticks(MONTH_STARTS_WY)
                ax.set_xticklabels([])
                ax.set_xlim(1, 366)

            if col_idx == 0:
                ax.set_ylabel(var_label, fontsize=FONTSIZE_LABEL)
            if row_idx == 0:
                ax.set_title(DATASET_LABELS.get(ds, ds),
                             fontsize=FONTSIZE_TITLE)

            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_edgecolor('#333333')

    fig.subplots_adjust(
        left=0.09, right=0.98, top=0.92, bottom=0.22,
        hspace=0.12, wspace=0.08,
    )

    # Shared legend for overlay lines (placed above the colorbar)
    legend_handles, legend_labels = [], []
    if worst_handle is not None:
        legend_handles.append(worst_handle)
        legend_labels.append('Worst-case focal event')
    if recon_handle is not None:
        legend_handles.append(recon_handle)
        legend_labels.append('Reconstruction mean')

    if legend_handles:
        fig.legend(
            legend_handles, legend_labels,
            loc='lower center', bbox_to_anchor=(0.5, 0.095),
            ncol=len(legend_handles), fontsize=FONTSIZE_LEGEND,
            frameon=False,
        )

    if sm_for_colorbar is not None:
        cbar_ax = fig.add_axes([0.30, 0.04, 0.40, 0.018])
        cbar = fig.colorbar(sm_for_colorbar, cax=cbar_ax,
                            orientation='horizontal')
        cbar.set_label('% of focal water-years below y-axis value',
                       fontsize=FONTSIZE_LEGEND)
        cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        cbar.set_ticklabels(['0', '20', '40', '60', '80', '100'])

    out_path = os.path.join(FIG_OUTPUT_DIR, FIG_NAME)
    fig.savefig(out_path, dpi=DPI_PRINT, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
