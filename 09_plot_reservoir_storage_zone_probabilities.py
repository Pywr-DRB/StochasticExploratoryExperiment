"""
This script is used to plot patterns in reservoir storage zone probabilities.

The final plot should have time on the horizontal axis, corresponding to a single year. 

The Y-axis will be total simulated NYC reservoir storage as a percentage. 

The storage will be broken-up based on the FFMP level boundaries which are the same for each year. 

The plot should show a heatmap indicating the probability of being inside each of the FFMP sotrage zones 
during a particular period of the year. 

The plot should support daily, weekly, and monthly time aggregations.

Ultimately, for each time period, there should be N colors vertically, representing the N storage zones. 
"""


# add near other imports
from matplotlib.colors import TwoSlopeNorm

import pywrdrb
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors


from config import RECONSTRUCTION_OUTPUT_FNAME
from config import verify_ensemble_type

### Plotting
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors

def _period_index(dts: pd.DatetimeIndex, period: str = 'daily', origin: str = 'june1') -> np.ndarray:
    """
    Map dates to a 'generic-year' period index, shared across all years.
    period: 'daily' (1..365/366 starting at June 1), 'weekly' (1..~53), or 'monthly' (1..12).
    origin='june1' defines the water-year origin at June 1.
    """
    dts = pd.DatetimeIndex(dts)
    if period not in ('daily', 'weekly', 'monthly'):
        raise ValueError("period must be one of {'daily','weekly','monthly'}")

    if origin != 'june1':
        raise NotImplementedError("Only origin='june1' supported here.")

    june1_this = pd.to_datetime(dts.year.astype(str) + '-06-01')
    is_after = dts >= june1_this
    june1_prev = pd.to_datetime((dts.year - 1).astype(str) + '-06-01')

    # day-of-water-year (1..365/366)
    doy_wy = np.where(is_after,
                      (dts - june1_this).days + 1,
                      (dts - june1_prev).days + 1)

    if period == 'daily':
        return doy_wy

    if period == 'weekly':
        # Weeks since June 1; integer in 1..ceil(366/7)
        return ((doy_wy - 1) // 7) + 1

    # monthly: month-of-water-year with June=1,...,May=12
    wy_month = ((dts.month - 6) % 12) + 1
    return wy_month


def _ordered_threshold_columns(ffmp_level_boundaries: pd.DataFrame, zone_cols=None):
    """Choose and order threshold columns ascending by their global median value."""
    if zone_cols is None:
        num_cols = ffmp_level_boundaries.select_dtypes(include=[np.number]).columns.tolist()
    else:
        num_cols = list(zone_cols)
    # Order by global median to ensure ascending thresholds
    med = ffmp_level_boundaries[num_cols].median(axis=0).sort_values()
    return list(med.index)


def _build_y_edges_grid(boundaries: pd.DataFrame,
                        period_idx: np.ndarray,
                        ordered_cols: list[str],
                        pct_extents=(0.0, 100.0)) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert time-dynamic FFMP thresholds into a 2D grid of zone edges per 'generic-year' period.
    Returns:
      y_edges_grid : array (Z+1, P+1) for pcolormesh
      periods_sorted : sorted unique period labels (length P)
    """
    # Median threshold for each period to define representative edges
    df_b = boundaries.copy()
    df_b['__p__'] = period_idx
    grouped = df_b.groupby('__p__')[ordered_cols].median()
    periods_sorted = grouped.index.to_numpy()

    # Z+1 edges per period: [0, t1, t2, ..., tK, 100]
    lo, hi = pct_extents
    edges_mat = np.column_stack([
        np.full((grouped.shape[0], 1), lo),
        grouped.to_numpy(copy=False),
        np.full((grouped.shape[0], 1), np.nextafter(hi, np.inf)),  # include right edge
    ])  # shape: (P, Z+1)

    # Transpose to Z+1 x P and append last column to reach P+1 for pcolormesh
    y_edges = edges_mat.T  # (Z+1, P)
    y_edges_grid = np.concatenate([y_edges, y_edges[:, [-1]]], axis=1)  # (Z+1, P+1)
    return y_edges_grid, periods_sorted


def plot_ffmp_zone_probabilities(
    df_ts: pd.DataFrame,
    ffmp_level_boundaries: pd.DataFrame,
    period: str = 'daily',             # 'daily' | 'weekly' | 'monthly'
    zone_cols: list[str] | None = None,
    pct_extents=(0.0, 100.0),
    cmap='magma',
    vmin=None,                         # if None: smallest positive prob
    vmax=1.0,
    log_floor=None,                    # e.g., 1e-4 to floor zeros for LogNorm; None -> leave zeros as NaN
    title: str | None = None,
    period_label='Period of year',
    storage_label='Total NYC storage (% of capacity)',
    figsize=(14, 6),
    fname: str | None = None,
):
    """
    Plot P(storage ∈ zone z | period) as a heatmap with:
      - X: generic-year period (daily/weekly/monthly) relative to June 1,
      - Y: storage % with time-dynamic FFMP zone edges (median per period),
      - Color: probability (log scale) of being in each zone at that period.

    Parameters
    ----------
    df_ts : DataFrame
        Wide matrix with datetime index and simulation columns (storage % in [0,100]).
    ffmp_level_boundaries : DataFrame
        Daily (or higher frequency) thresholds with the same datetime index,
        columns are level thresholds in % of capacity (internal boundaries).
    period : str
        'daily', 'weekly', or 'monthly' aggregation along the generic year (June 1 origin).
    zone_cols : list[str] or None
        Which columns in ffmp_level_boundaries define the internal thresholds; if None, use all numeric.
    pct_extents : (float, float)
        Lower and upper bounds for the storage axis (typically (0,100)).
    log_floor : float or None
        If given, replace 0 probabilities by this floor before LogNorm.
    """
    if not isinstance(df_ts.index, pd.DatetimeIndex):
        raise ValueError("df_ts must have a DatetimeIndex.")
    if not isinstance(ffmp_level_boundaries.index, pd.DatetimeIndex):
        raise ValueError("ffmp_level_boundaries must have a DatetimeIndex.")

    # Align to common dates
    common_idx = df_ts.index.intersection(ffmp_level_boundaries.index)
    if common_idx.empty:
        raise ValueError("No overlapping dates between df_ts and ffmp_level_boundaries.")
    S = df_ts.loc[common_idx]  # (N_dates, N_sims)
    B = ffmp_level_boundaries.loc[common_idx]  # (N_dates, K)

    # Determine ordered threshold columns and zone count
    thr_cols = _ordered_threshold_columns(B, zone_cols)
    Z = len(thr_cols) + 1
    lo, hi = pct_extents
    hi_inc = np.nextafter(hi, np.inf)  # ensure inclusion of right edge

    # Period indices (generic-year) for both S and B (must match)
    p_idx = _period_index(common_idx, period=period, origin='june1')
    periods_sorted = np.sort(np.unique(p_idx))
    P = periods_sorted.size
    period_to_pos = {p: i for i, p in enumerate(periods_sorted)}

    # Build 2D Y-edges grid for visualization (median thresholds per period)
    y_edges_grid, periods_sorted = _build_y_edges_grid(B, p_idx, thr_cols, pct_extents=pct_extents)

    # Compute zone probabilities by aggregating counts with the *daily* dynamic edges
    sim_cols = S.columns.tolist()
    counts = np.zeros((P, Z), dtype=np.int64)  # (P, Z)
    # Iterate per date: histogram across simulations using that day's edges
    # This avoids exploding to long format.
    S_vals = S.to_numpy()  # (N_dates, N_sims)
    B_vals = B[thr_cols].to_numpy()  # (N_dates, K)

    for row, p in enumerate(p_idx):
        edges = np.concatenate(([lo], np.sort(B_vals[row]), [hi_inc]))  # (Z+1,)
        # Histogram over all sims for this date
        h, _ = np.histogram(S_vals[row, :], bins=edges)
        counts[period_to_pos[p], :] += h

    totals = counts.sum(axis=1, keepdims=True)  # (P,1)
    with np.errstate(invalid='ignore', divide='ignore'):
        probs_PZ = np.where(totals > 0, counts / totals, 0.0)  # (P, Z)
    probs = probs_PZ.T  # (Z, P) for plotting

    # Prepare X edges for pcolormesh
    x_centers = periods_sorted.astype(float)  # 1..P
    if x_centers.size == 1:
        x_edges_1d = np.array([x_centers[0]-0.5, x_centers[0]+0.5], dtype=float)
    else:
        mid = 0.5 * (x_centers[:-1] + x_centers[1:])
        x_edges_1d = np.empty(x_centers.size + 1, dtype=float)
        x_edges_1d[1:-1] = mid
        x_edges_1d[0] = x_centers[0] - (mid[0] - x_centers[0])
        x_edges_1d[-1] = x_centers[-1] + (x_centers[-1] - mid[-1])

    # X/Y grids for pcolormesh (curvilinear grid: varying Y edges per period)
    X = np.tile(x_edges_1d, (Z + 1, 1))                  # (Z+1, P+1)
    Y = y_edges_grid                                     # (Z+1, P+1)

    # Handle zeros for LogNorm
    M = probs.copy()
    if log_floor is not None:
        M = np.where(M > 0, M, log_floor)
    else:
        # mask zeros to show as 'bad' color on log scale
        M = np.where(M > 0, M, np.nan)
        
    # Convert prob to 0-100
    M *= 100

    # Color normalization (log scale)
    if vmin is None:
        finite_pos = M[np.isfinite(M) & (M > 0)]
        vmin = finite_pos.min() if finite_pos.size else 1e-6
    norm = colors.LogNorm(vmin=vmin, vmax=vmax)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color='#f0f0f0')

    quad = ax.pcolormesh(X, Y, M, cmap=cmap_obj, norm=norm, shading='flat')

    # Zone boundary lines (use representative median edges per period)
    for j in range(Y.shape[1]):
        ax.plot(X[:, j], Y[:, j], color='white', linewidth=0.6, alpha=0.7)

    # Labels, limits, colorbar
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(lo, hi)
    ax.set_xlabel(period_label)
    ax.set_ylabel(storage_label)
    if title:
        ax.set_title(title)

    cbar = plt.colorbar(quad, ax=ax, pad=0.02)
    cbar.set_label('Probability (log scale)')

    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches='tight')
    return fig, ax, probs, x_edges_1d, y_edges_grid


if __name__ == "__main__":
    
    # Load a single FFMP level boundary data
    ffmp_level_data = pywrdrb.Data(results_sets=["ffmp_level_boundaries"])
    ffmp_level_data.load_output(output_filenames = [RECONSTRUCTION_OUTPUT_FNAME])
    
    # The FFMP level boundaries indicate different storage zones
    # These are the same for all years and simulations
    # Here, I want to keep just one copy of the pd.DataFrame
    # This will be a daily timeseries data
    ffmp_level_boundaries = ffmp_level_data.ffmp_level_boundaries['reconstruction'][0]

    # Convert ffmp level boundaries to percentage (currently fraction)
    ffmp_level_boundaries = ffmp_level_boundaries * 100


    # Save storage probabilities in this dict, to compare in the difference plot
    storage_probs = {}

    # Get ensemble type from command line arguments
    for ensemble_type in ['stationary', 'climate_adjusted']:
        inflow_type = f'{ensemble_type}_ensemble'
        verify_ensemble_type(ensemble_type)

        # Load ensemble data from processed HDF5 - should have everything we need inside
        fname = f'./pywrdrb/outputs/{ensemble_type}_ensemble_with_postprocessing.hdf5'
        data = pywrdrb.Data()
        data.load_from_export(fname)

        realization_ids = list(data.res_storage[inflow_type].keys())

        nyc_reservoirs = ['cannonsville', 'pepacton', 'neversink']

        # Make a df with nyc_agg storage for all realizations
        period = 'weekly'
        if period == 'daily':
            agg_period = 'D'
        elif period == 'weekly':
            agg_period = 'W'
        elif period == 'annual':
            agg_period = 'YS'

        # If you can restructure to get all realizations at once
        print('Making DF of all data')
        all_data = {rid: data.res_storage[inflow_type][rid][nyc_reservoirs].sum(axis=1).resample(agg_period).min() 
                    for rid in realization_ids}
        df_nyc_storage = pd.DataFrame(all_data)

        df = df_nyc_storage.copy()
        df = df / df.max().max() * 100 # Percentage of max storage


        # Relabel all columns 1, 2, ..., N
        df.columns = [f'{i+1}' for i in range(df.shape[1])]

        # Period of year should be relative to June 1st
        if period == 'daily':
            # Calculate period_of_year where June 1 = 1, July 31 next year = 365/366
            june_1 = pd.to_datetime(df.index.year.astype(str) + '-06-01')

            # For dates >= June 1: days since June 1 + 1
            # For dates < June 1: days since June 1 of previous year + 1
            mask_after_june = df.index >= june_1

            df['period_of_year'] = np.where(
                mask_after_june,
                (df.index - june_1).days + 1,
                (df.index - pd.to_datetime((df.index.year - 1).astype(str) + '-06-01')).days + 1
            )
        elif period == 'weekly':
            df['period_of_year'] = pd.to_datetime(df.index).isocalendar().week

        # Plot results
        print('Plotting FFMP zone probabilities...')

        fname = f'{inflow_type}_storage_zone_probabilities.png'
        _, _, prob, x_edges_1d, y_edges_grid = plot_ffmp_zone_probabilities(df, ffmp_level_boundaries,
                                    period=period,
                                    vmin=0.01,
                                    vmax=100,
                                    log_floor=0.01,
                                    fname=None)
        
        storage_probs[ensemble_type] = prob
        
        if 'x_edges_1d_ref' not in locals():
            x_edges_1d_ref, y_edges_grid_ref = x_edges_1d, y_edges_grid
    
    ### Now plot the difference between the two. 
    # This plot should match the same x, y format as the prior
    # period on the horizontal and storage on the y
    # However, the cmap should be used to show the percent change in storage prob
    # for the climate_adjusted relative to the stationary
    print("Plotting difference in storage probs...")
    
    # percent change relative to stationary (in %), with small floor to avoid 0-division
    eps = 1e-8
    p_s = storage_probs['stationary']
    p_c = storage_probs['climate_adjusted']
    prob_diff = (p_c - p_s)
    
    # Print min/max of prob_diff 
    print(f"Max prob_diff: {prob_diff.max().max()} | Min prob diff: {prob_diff.min().min()}")
    
    prob_diff_perc = 100.0 * (prob_diff / np.maximum(p_s, eps))       
    
    for use_percent_change in [False, True]:
        
        prob_vals = prob_diff_perc if use_percent_change else prob_diff
        if use_percent_change:
            cbar_lab = 'Δ Probability (% relative to stationary)'
            norm = TwoSlopeNorm(vmin=-100, vcenter=0, vmax=100)
            fname='storage_zone_prob_diff_percent.png'
        else:
            cbar_lab = 'Δ Probability (absolute)'
            vmin = prob_vals.min().min()
            vmax = prob_vals.max().max()
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
            fname = 'storage_zone_prob_diff_absolute.png'
        
        # plot difference using same grids
        fig, ax = plt.subplots(figsize=(14, 6))
        X = np.tile(x_edges_1d_ref, (y_edges_grid_ref.shape[0], 1))
        Y = y_edges_grid_ref
        pcm = ax.pcolormesh(X, Y, prob_vals, 
                            cmap='BrBG', norm=norm, shading='flat')
        for j in range(Y.shape[1]):
            ax.plot(X[:, j], Y[:, j], color='white', linewidth=0.6, alpha=0.7)
        ax.set_xlabel('Period of year')
        ax.set_ylabel('Total NYC storage (% of capacity)')
        cbar = plt.colorbar(pcm, ax=ax, pad=0.02, extend='both')
        cbar.set_label(cbar_lab)
        plt.tight_layout()
        plt.savefig(fname, dpi=300, bbox_inches='tight')
