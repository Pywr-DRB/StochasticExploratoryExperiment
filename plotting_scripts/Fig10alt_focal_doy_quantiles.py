"""
Fig10 alternative: Focal-event DOY quantile gradient (3x3 grid).

Shows the distribution of focal-zone drought water-years as a viridis
sequential gradient background at each day-of-water-year, with the single
worst-case trajectory overlaid, for three variables x three climate scenarios.
The storage row additionally overlays the time-varying FFMP
Watch/Warning/Emergency zone thresholds as dashed lines.

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
    RECONSTRUCTION_OUTPUT_FNAME, RECONSTRUCTION_START_DATE,
    RECONSTRUCTION_END_DATE, DATASET_CONFIGS,
    GRID_N_BINS, FOCAL_FRAC_THRESH, FOCAL_RP_THRESH_YEARS,
    FOCAL_WORST_STORAGE_THRESH,
)
from methods.water_year import (
    MONTH_STARTS_WY, vectorized_water_year_doy,
)
from methods.load import (
    load_event_metrics, load_rank_subset_from_export, load_ffmp_boundaries,
)
from methods.return_period import compute_return_period_grid
from methods.plotting.heatmap import (
    make_shared_edges_logmag, assign_grid_bins,
    compute_emergency_grid, compute_min_storage_grid,
    identify_focal_region, select_events_from_focal_region,
)
from methods.plotting.drought_dynamics import compute_fixed_extraction_window
from methods.plotting.percentile_bands import format_xaxis_water_year
from methods.plotting.styles import (
    DATASET_LABELS, FFMP_ZONE_COLORS, CMAP_SEQUENTIAL,
    DPI_PRINT, FONTSIZE_LABEL, FONTSIZE_TITLE, FONTSIZE_LEGEND, FONTSIZE_SMALL,
    apply_publication_style, label_panel, save_fig,
)
from methods.plotting.ensemble_summary import MGD_TO_MCM
from methods.plotting.doy_quantile_gradient import (
    compute_doy_quantile_grid, plot_doy_quantile_gradient,
)


# ── Configuration ────────────────────────────────────────────────────────

SSI_WINDOW = 3
MIN_COUNT = 1

DATASETS = list(DATASET_CONFIGS.keys())
# Only the result sets we actually consume — inflow is unused, dropping it
# avoids loading a large DataFrame per realization from HDF5.
RESULTS_SETS = ['res_storage', 'contribution', 'major_flow']

# TEST_MODE relaxes focal thresholds when the strict Fig9/Fig10 region is empty
# on the local 5-realization dataset. Final-run default: TEST_MODE = False.
TEST_MODE = False

STRICT_THRESHOLDS = dict(
    rp_thresh_years=FOCAL_RP_THRESH_YEARS,
    frac_thresh=FOCAL_FRAC_THRESH,
    storage_thresh=FOCAL_WORST_STORAGE_THRESH,
)
RELAXED_THRESHOLDS = dict(
    rp_thresh_years=np.inf,  # accept any finite T_W
    frac_thresh=2.0,         # accept any non-NaN frac (>1 always true)
    storage_thresh=200.0,    # accept any non-NaN min storage
)

REFERENCE_WY_START = pd.Timestamp('2000-06-01')
REFERENCE_WY_END = pd.Timestamp('2001-05-31')
N_QUANTILE_LEVELS = 21    # 0, 5, 10, ..., 100 %

CMAP_N_BINS = 10          # discrete viridis bands (matches Fig9 %DE 10-pp ticks)
CMAP = plt.get_cmap(CMAP_SEQUENTIAL, CMAP_N_BINS)

# (variable_key, y-label, y-scale, rolling-mean window in days)
# Release and flow are converted from MGD to MCM/day (same convention as
# Fig4); storage stays as a percentage of capacity.
VARIABLES = [
    ('nyc_storage_pct',
     'Combined NYC storage\n(% of capacity)',                    'linear', 1),
    ('nyc_release',
     'Mandated NYC release to\nMontague target (MCM/day)',       'linear', 7),
    ('montague_flow',
     'Montague gauge flow\n(MCM/day, log scale)',                'log',    7),
]

PANEL_LETTERS = list('abcdefghi')
XAXIS_SUFFIX_LABEL = 'Water Year (Jun 1 - May 31, FFMP convention)'

FIG_OUTPUT_DIR = os.path.join(FIG_DIR, 'Fig10')
FIG_NAME_STEM = f'Fig10alt_focal_doy_quantiles_ssi{SSI_WINDOW}'
if TEST_MODE:
    FIG_NAME_STEM += '_TESTMODE'


# ── Helpers ──────────────────────────────────────────────────────────────

def _traces_to_doy_df(traces_by_event):
    """Stack a list of (event_id, pd.Series) into a DOY-indexed DataFrame.

    DOY is invariant under whole-year date shifts, so we do not need to
    align every event's DatetimeIndex onto a common reference year before
    building the DOY frame — `vectorized_water_year_doy` yields identical
    values regardless of calendar year.
    """
    frames = {}
    for event_id, s in traces_by_event:
        if s is None or len(s) == 0:
            continue
        doy = vectorized_water_year_doy(s.index)
        frames[event_id] = pd.Series(s.values, index=doy)
    if not frames:
        return pd.DataFrame()
    return pd.DataFrame(frames).sort_index()


def _build_realization_cache(data, dataset_id, realization_ids):
    """Pre-aggregate per-realization time series for the three plotted variables.

    The per-event loop previously recomputed NYC-reservoir sums and column
    lookups for *every* focal event (often many events share a realization).
    Building one DataFrame per realization up front collapses that work
    from O(events) down to O(unique realizations).
    """
    cache = {}
    for r in realization_ids:
        storage_raw = data.res_storage[dataset_id][r][NYC_RESERVOIRS].sum(axis=1)
        nyc_storage_pct = 100.0 * storage_raw / NYC_TOTAL_CAPACITY

        contribution = data.contribution[dataset_id][r]
        if isinstance(contribution, pd.DataFrame):
            nyc_release = contribution['mrf_montagueTrenton_nyc']
        else:
            nyc_release = contribution
        nyc_release = nyc_release * MGD_TO_MCM

        montague_flow = data.major_flow[dataset_id][r]['delMontague'] * MGD_TO_MCM

        cache[r] = pd.DataFrame({
            'nyc_storage_pct': nyc_storage_pct,
            'nyc_release': nyc_release,
            'montague_flow': montague_flow,
        })
    return cache


def load_reconstruction_annual_cycle():
    """Median daily annual cycle (by DOY) for the three plotted variables.

    Release and flow are converted from MGD to MCM/day to match the figure.
    """
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
        nyc_release = (
            data.nyc_release_components[ds][r][contrib_cols].sum(axis=1)
            * MGD_TO_MCM
        )

        montague_flow = data.major_flow[ds][r]['delMontague'] * MGD_TO_MCM
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
        out[name] = df.groupby('doy')['value'].median().sort_index()
    return out


def build_ffmp_by_wy_doy():
    """FFMP Watch/Warning/Emergency thresholds (%) indexed by water-year DOY.

    Time-varies: reflects the seasonal FFMP rule curves, aligned so that
    DOY 1 = June 1 (water-year convention).
    """
    fb = load_ffmp_boundaries().copy()
    fb['cal_doy'] = fb.index.dayofyear

    col_map = {}
    for candidate, zone in [('L3', 'Watch'), ('level3', 'Watch'),
                             ('L4', 'Warning'), ('level4', 'Warning'),
                             ('L5', 'Emergency'), ('level5', 'Emergency')]:
        if candidate in fb.columns:
            col_map[candidate] = zone
    if not col_map:
        return None
    fb = fb.rename(columns=col_map)
    zone_cols = [z for z in ['Watch', 'Warning', 'Emergency'] if z in fb.columns]
    seasonal = fb.groupby('cal_doy')[zone_cols].median()

    wy_doys = np.arange(1, 367)
    dates = pd.date_range(REFERENCE_WY_START, periods=366, freq='D')
    cal_doys = dates.dayofyear

    out = pd.DataFrame(index=wy_doys, columns=zone_cols, dtype=float)
    for wy_doy, cal_doy in zip(wy_doys, cal_doys):
        if cal_doy in seasonal.index:
            out.loc[wy_doy] = seasonal.loc[cal_doy].values
    return out


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

    # 3. Build per-dataset grids and identify focal cells.
    # Focal-region thresholding is on T_W (Bonaccorso-Shiau interarrival
    # time minus mean event duration); see methods/return_period.py.
    T_W_grids, frac_grids, min_grids = {}, {}, {}
    for ds in DATASETS:
        _, _, T_W_grids[ds], _ = compute_return_period_grid(
            all_data[ds], sev_edges, mag_edges, N_YEARS, min_count=MIN_COUNT)
        frac_grids[ds], _ = compute_emergency_grid(
            all_data[ds], sev_edges, mag_edges, min_count=MIN_COUNT)
        min_grids[ds], _ = compute_min_storage_grid(
            all_data[ds], sev_edges, mag_edges, min_count=MIN_COUNT)

    thresholds = STRICT_THRESHOLDS
    focal_cells = identify_focal_region(
        T_W_grids, frac_grids, min_grids, DATASETS, **thresholds)
    print(f"  Strict focal region: {len(focal_cells)} cells")

    if len(focal_cells) == 0 and TEST_MODE:
        thresholds = RELAXED_THRESHOLDS
        focal_cells = identify_focal_region(
            T_W_grids, frac_grids, min_grids, DATASETS, **thresholds)
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

        # Aggregate each realization's three variables once, reuse across
        # every focal event that lands in that realization.
        realization_cache = _build_realization_cache(data, ds, unique_reals)
        del data
        gc.collect()

        per_var_traces = {v[0]: [] for v in VARIABLES}
        worst_event_id = None

        for event_idx, (_, row) in enumerate(selected.iterrows()):
            r_id = int(row['realization_id'])
            event_id = (f"R{r_id:04d}_"
                        f"{pd.Timestamp(row['start']).date()}")
            min_storage_date = pd.Timestamp(row['min_storage_date'])

            w_start, w_end = compute_fixed_extraction_window(
                min_storage_date, pad_before_wy=0, pad_after_wy=0,
            )
            # Slice the pre-aggregated per-realization frame by date range
            # (.loc[start:end] is cheap on a DatetimeIndex).
            window = realization_cache[r_id].loc[w_start:w_end]

            for var_name, _, _, smooth in VARIABLES:
                s = window[var_name]
                if smooth and smooth > 1:
                    s = s.rolling(smooth, center=True, min_periods=1).mean()
                per_var_traces[var_name].append((event_id, s))

            if event_idx == 0:
                worst_event_id = event_id
                dataset_worst_meta[ds] = {
                    'event_id': event_id,
                    'realization_id': r_id,
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

        del realization_cache
        gc.collect()

    # 5. Reconstruction mean annual cycle (shared reference across columns)
    print("\nLoading reconstruction mean annual cycle...")
    recon = load_reconstruction_annual_cycle()

    # 5b. FFMP zone thresholds by water-year DOY (time-varying)
    ffmp_by_wy = build_ffmp_by_wy_doy()

    # 5c. Per-dataset event counts (for column annotations)
    dataset_n_events = {
        ds: int(dataset_traces[ds][VARIABLES[0][0]].shape[1])
        for ds in DATASETS
    }

    # 6. Figure — sized for readable 2-line titles, aligned y-labels, a
    # 5-entry legend block, and a horizontal discrete colorbar without
    # any overlap between those elements.
    fig, axes = plt.subplots(
        len(VARIABLES), len(DATASETS),
        figsize=(4.4 * len(DATASETS), 3.2 * len(VARIABLES)),
        sharex='col', sharey='row',
    )
    if len(VARIABLES) == 1:
        axes = np.array([axes])
    if len(DATASETS) == 1:
        axes = axes[:, None]

    sm_for_colorbar = None
    worst_handle = None
    recon_handle = None
    ffmp_handles = {}

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

            # FFMP Watch/Warning/Emergency thresholds (storage row only)
            if var_name == 'nyc_storage_pct' and ffmp_by_wy is not None:
                for zone in ['Watch', 'Warning', 'Emergency']:
                    if zone not in ffmp_by_wy.columns:
                        continue
                    zvals = ffmp_by_wy[zone].astype(float)
                    ln, = ax.plot(
                        zvals.index, zvals.values,
                        color=FFMP_ZONE_COLORS[zone], linestyle='--',
                        linewidth=1.0, alpha=0.95, zorder=4,
                    )
                    ffmp_handles.setdefault(zone, ln)
                    # Thin left-edge label
                    if col_idx == 0:
                        y0 = zvals.dropna().iloc[0] if zvals.dropna().size else None
                        if y0 is not None:
                            ax.text(
                                2, y0, zone,
                                fontsize=FONTSIZE_SMALL - 1,
                                color=FFMP_ZONE_COLORS[zone],
                                va='center', ha='left',
                                zorder=4.5,
                            )

            # Historical reconstruction daily median (dashed black, 1.8 pt)
            if recon is not None and var_name in recon:
                rs = recon[var_name]
                ln, = ax.plot(
                    rs.index, rs.values,
                    color='#000000', linestyle='--', linewidth=1.8,
                    alpha=1.0, zorder=5,
                )
                if recon_handle is None:
                    recon_handle = ln

            # Worst-case focal trajectory (solid red, 2.5 pt)
            worst = dataset_worst[ds][var_name]
            if worst is not None and worst.dropna().size > 0:
                w = worst.sort_index()
                ln, = ax.plot(
                    w.index, w.values,
                    color='#c0392b', linewidth=2.5, alpha=0.98, zorder=6,
                )
                if worst_handle is None:
                    worst_handle = ln

            if yscale == 'log':
                ax.set_yscale('log')

            # Keep water-year tick marks on every subplot; only the bottom
            # row shows tick labels, and the axis label is shared via
            # fig.supxlabel() below so it only appears once.
            if row_idx == len(VARIABLES) - 1:
                format_xaxis_water_year(ax)
            else:
                ax.set_xticks(MONTH_STARTS_WY)
                ax.set_xticklabels([])
                ax.set_xlim(1, 366)

            if col_idx == 0:
                ax.set_ylabel(var_label, fontsize=FONTSIZE_LABEL)
            if row_idx == 0:
                n = dataset_n_events[ds]
                ax.set_title(
                    f"{DATASET_LABELS.get(ds, ds)}\n"
                    f"n = {n} drought events",
                    fontsize=FONTSIZE_TITLE,
                )

            # Panel letter (top-left), matches label_panel() style used
            # by Fig4/Fig7/Fig8/Fig9.
            label_panel(
                ax, PANEL_LETTERS[row_idx * len(DATASETS) + col_idx],
                fontsize=FONTSIZE_LABEL, fontweight='normal',
            )

            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_edgecolor('#333333')

    fig.subplots_adjust(
        left=0.11, right=0.98, top=0.90, bottom=0.28,
        hspace=0.22, wspace=0.10,
    )

    # Align all left-column y-axis labels so the text begins at the same
    # x coordinate across rows (avoids the staggered look from variable-length
    # first lines).
    fig.align_ylabels(axes[:, 0])

    # Single shared x-axis label — tick marks stay on every subplot, but
    # only one "Water Year (Jun 1 - May 31, FFMP convention)" caption is
    # rendered, centered under the bottom row.
    fig.supxlabel(XAXIS_SUFFIX_LABEL, fontsize=FONTSIZE_LABEL, y=0.235)

    # Shared legend for overlay lines (placed above the colorbar)
    recon_years = (
        f"{pd.Timestamp(RECONSTRUCTION_START_DATE).year}"
        f"–{pd.Timestamp(RECONSTRUCTION_END_DATE).year}"
    )
    legend_handles, legend_labels = [], []
    if worst_handle is not None:
        legend_handles.append(worst_handle)
        legend_labels.append(
            'Worst-case focal drought trajectory '
            '(lowest minimum NYC storage in focal region)'
        )
    if recon_handle is not None:
        legend_handles.append(recon_handle)
        legend_labels.append(
            f'Historical reconstruction daily median ({recon_years})'
        )
    ffmp_level_map = {'Watch': 'L3', 'Warning': 'L4', 'Emergency': 'L5'}
    for zone in ['Watch', 'Warning', 'Emergency']:
        if zone in ffmp_handles:
            legend_handles.append(ffmp_handles[zone])
            legend_labels.append(
                f'FFMP {zone} threshold ({ffmp_level_map[zone]}, '
                f'seasonal rule curve)'
            )

    if legend_handles:
        fig.legend(
            legend_handles, legend_labels,
            loc='lower center', bbox_to_anchor=(0.5, 0.14),
            ncol=2, fontsize=FONTSIZE_LEGEND,
            frameon=False,
        )

    if sm_for_colorbar is not None:
        cbar_ax = fig.add_axes([0.30, 0.045, 0.40, 0.020])
        cbar = fig.colorbar(sm_for_colorbar, cax=cbar_ax,
                            orientation='horizontal')
        # Label above the bar, tick marks and tick labels below.
        cbar.ax.xaxis.set_label_position('top')
        cbar.ax.xaxis.set_ticks_position('bottom')
        cbar.set_label('% of focal water-years below y-axis value',
                       fontsize=FONTSIZE_LEGEND, labelpad=6)
        # Tick at every discrete bin edge (0, 10, 20, ..., 100 %).
        bin_edges = np.linspace(0.0, 1.0, CMAP_N_BINS + 1)
        cbar.set_ticks(bin_edges)
        cbar.set_ticklabels([f'{int(round(v * 100))}' for v in bin_edges])

    out_stem = os.path.join(FIG_OUTPUT_DIR, FIG_NAME_STEM)
    save_fig(fig, out_stem, dpi=DPI_PRINT)
    plt.close(fig)
    print(f"\nSaved (png/svg/pdf): {out_stem}")


if __name__ == '__main__':
    main()
