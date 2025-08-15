import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors

from matplotlib import colors, ticker
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from collections import defaultdict
from config import verify_ensemble_type, FIG_DIR
FIG_DIR = f"{FIG_DIR}/drought_return_period/"


def calculate_drought_frequency(
    drought_df,
    x1_metric='severity',
    x2_metric='magnitude',
    x1_range=None,
    x2_range=None,
    ngrid=50,
    n_realizations=1000,
    n_years=70,
):
    """
    Efficient joint frequency calculation:
    P(X1 >= x1 & X2 >= x2) per year-realization,
    using annual at-least-one-event rule and a difference-array update.
    """
    # Validate
    required_cols = ['start', 'end', 'realization_id', x1_metric, x2_metric]
    missing_cols = [col for col in required_cols if col not in drought_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    if x1_range is None or x2_range is None:
        raise ValueError("x1_range and x2_range must be provided.")

    df = drought_df.copy()
    df['start'] = pd.to_datetime(df['start'])
    df['end'] = pd.to_datetime(df['end'])

    # Realizations
    rids = np.asarray(pd.unique(df['realization_id']))
    rids.sort()
    R = len(rids)
    rid_to_idx = {rid: i for i, rid in enumerate(rids)}

    # Shared calendar years
    min_year = df['start'].dt.year.min()
    max_year = df['end'].dt.year.max()
    years = np.arange(min_year, min_year + n_years, dtype=int)
    Y = len(years)
    year_to_idx = {y: i for i, y in enumerate(years)}

    # Grids
    x1_grid = np.linspace(x1_range[0], x1_range[1], ngrid)
    x2_grid = np.linspace(x2_range[0], x2_range[1], ngrid)
    I, J = ngrid, ngrid

    # Map events to grid indices
    x1_vals = df[x1_metric].to_numpy(float)
    x2_vals = df[x2_metric].to_numpy(float)
    i_e = np.searchsorted(x1_grid, x1_vals, side='right') - 1
    j_e = np.searchsorted(x2_grid, x2_vals, side='right') - 1

    # (realization, year) -> list of (i,j)
    groups = defaultdict(list)
    rid_idx = np.fromiter((rid_to_idx[r] for r in df['realization_id']),
                          count=len(df), dtype=int)
    starts_y = df['start'].dt.year.to_numpy()
    ends_y = df['end'].dt.year.to_numpy()

    for k in range(len(df)):
        ii, jj = i_e[k], j_e[k]
        if ii < 0 or jj < 0:
            continue
        rs = rid_idx[k]
        ys, ye = int(starts_y[k]), int(ends_y[k])
        ys = max(ys, years[0])
        ye = min(ye, years[-1])
        base = rs * Y
        for y in range(ys, ye + 1):
            yi = year_to_idx.get(y)
            if yi is not None:
                groups[base + yi].append((ii, jj))

    # Difference-array accumulation
    C_diff = np.zeros((I, J + 1), dtype=np.int32)
    for pairs in groups.values():
        if not pairs:
            continue
        h = np.full(I, -1, dtype=np.int16)
        for ii, jj in pairs:
            if jj > h[ii]:
                h[ii] = jj
        f = np.maximum.accumulate(h[::-1])[::-1]
        valid = np.where(f >= 0)[0]
        if valid.size:
            C_diff[valid, 0] += 1
            C_diff[valid, f[valid] + 1] -= 1

    counts = np.cumsum(C_diff, axis=1)[:, :J]
    denom = float(n_realizations * n_years)
    freq = counts / denom

    eps = 1 / denom
    freq[freq < eps] = eps  # avoid inf in return period calc
    return_period = 1.0 / freq
    

    return {
        'frequency_matrix': freq,
        'return_period_matrix': return_period,
        'x1_grid': x1_grid,
        'x2_grid': x2_grid,
        'x1_metric': x1_metric,
        'x2_metric': x2_metric,
        'total_years': denom
    }


def _grid_edges_from_centers(grid: np.ndarray):
    """Assumes monotone, roughly uniform spacing; returns len(grid)+1 edges."""
    d = np.diff(grid)
    d0, d1 = d[0], d[-1]
    edges = np.empty(grid.size + 1, dtype=float)
    edges[1:-1] = 0.5*(grid[:-1] + grid[1:])
    edges[0]  = grid[0] - 0.5*d0
    edges[-1] = grid[-1] + 0.5*d1
    return edges

def plot_drought_frequency_heatmap(freq_result, 
                                   syn_droughts=None,
                                   obs_droughts=None,
                                   return_period=True,
                                   figsize=(10, 8), 
                                   cmap='magma',          # better for log scales
                                   log_cmap=True,
                                   vmin=None,
                                   vmax=None,
                                   fname=None,
                                   title=None,
                                   show_contours=True,
                                   contour_levels=None,
                                   difference=False):
    # Select matrix
    use_key = 'return_period_matrix' if return_period else 'frequency_matrix'
    M = np.array(freq_result[use_key], dtype=float)

    # Mask invalids: p==0 (=> T=inf) or NaNs
    if return_period:
        M = np.where(np.isfinite(M), M, np.nan)
    else:
        M = np.where(np.isfinite(M) & (M >= 0), M, np.nan)

    # Norm
    if return_period and not difference:
        # sensible defaults if not provided
        if vmin is None: vmin = np.nanpercentile(M, 1) if np.isfinite(M).any() else 1.0
        if vmax is None: vmax = np.nanpercentile(M, 99) if np.isfinite(M).any() else 1000.0
        vmin = max(vmin, 1e-6)  # must be >0 for LogNorm
        norm = colors.LogNorm(vmin=vmin, vmax=vmax) if log_cmap else colors.Normalize(vmin=vmin, vmax=vmax)
        cbar_label = "Return period T (years)"
        default_contours = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    else:
        if vmin is None: vmin = 0.0
        if vmax is None: vmax = float(np.nanmax(M)) if np.isfinite(M).any() else 0.5
        norm = colors.Normalize(vmin=vmin, vmax=vmax) if not log_cmap else colors.LogNorm(vmin=max(vmin,1e-6), vmax=vmax)
        cbar_label = "Frequency | P(X1 ≥ x1 & X2 ≥ x2)"
        default_contours = None  # contouring p is fine too; set if desired

    if difference:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        cbar_label = "Change in Return Period (%)"

    # Grid edges for pcolormesh
    x_edges = _grid_edges_from_centers(freq_result['x2_grid'])  # x-axis = X2
    y_edges = _grid_edges_from_centers(freq_result['x1_grid'])  # y-axis = X1

    # Build figure
    fig, ax = plt.subplots(figsize=figsize)

    # Colormap with a defined 'bad' color for masked/NaN cells
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color='#f0f0f0')  # light grey for no data

    # Heatmap
    pm = ax.pcolormesh(x_edges, y_edges, M, cmap=cmap_obj, norm=norm, shading='auto')

    # Optional isolines
    if show_contours and (contour_levels is not None or default_contours is not None):
        levels = contour_levels if contour_levels is not None else default_contours
        # Only draw contours within finite region
        with np.errstate(invalid='ignore'):
            cs = ax.contour(freq_result['x2_grid'], freq_result['x1_grid'], M, 
                            levels=levels, colors='white', linewidths=0.8)
            ax.clabel(cs, fmt=lambda v: f"{int(v)}-yr" if return_period else f"{v:.2f}", 
                      fontsize=8, inline=True, inline_spacing=4)

    # Overlays
    xcol, ycol = freq_result['x2_metric'], freq_result['x1_metric']
    if syn_droughts is not None:
        s = syn_droughts[[xcol, ycol]].dropna()
        ax.scatter(s[xcol], s[ycol], s=10, facecolors='none', edgecolors='k', alpha=0.25, lw=0.5, label='Synthetic')
    if obs_droughts is not None:
        o = obs_droughts[[xcol, ycol]].dropna()
        ax.scatter(o[xcol], o[ycol], s=30, marker='^', c='k', alpha=0.9, label='Observed')

    # Axes labels & title
    ax.set_xlabel(f"{xcol.title()}")
    ax.set_ylabel(f"{ycol.title()}")
    if title:
        ax.set_title(title)

    # Colorbar with clean ticks
    extend = 'max' if not difference else 'both'
    cbar = plt.colorbar(pm, ax=ax, pad=0.02, extend=extend)
    cbar.set_label(cbar_label)
    if return_period and isinstance(norm, colors.LogNorm):
        # decade-ish ticks
        ticks = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])
        ticks = ticks[(ticks >= vmin) & (ticks <= vmax)]
        if ticks.size:
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f"{int(t)}" for t in ticks])

    # Minor gridlines for readability
    ax.grid(which='both', color='white', alpha=0.2, linewidth=0.5)
    ax.set_axisbelow(True)

    # Legend outside
    if (syn_droughts is not None) or (obs_droughts is not None):
        ax.legend(frameon=False, loc='lower center', bbox_to_anchor=(1.02, 1.0))

    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname, dpi=400, bbox_inches='tight')
        # optionally also save vector:
        base = fname.rsplit('.', 1)[0]
        plt.savefig(f"{base}.svg", bbox_inches='tight')
    return fig, ax


if __name__ == "__main__":
    SSI_WINDOW = 12

    obs_droughts = pd.read_csv(f"./pywrdrb/drought_metrics/observed_ssi{SSI_WINDOW}_drought_events.csv")
    obs_droughts['severity'] = obs_droughts['severity'].abs()
    obs_droughts['magnitude'] = obs_droughts['magnitude'].abs()

    all_results = {}

    for ensemble_type in ['stationary', 'climate_adjusted']:
        inflow_type = f"{ensemble_type}_ensemble"
        verify_ensemble_type(ensemble_type)

        syn_droughts = pd.read_csv(
            f"./pywrdrb/drought_metrics/{ensemble_type}_ensemble_ssi{SSI_WINDOW}_drought_events.csv"
        )
        syn_droughts = syn_droughts[~syn_droughts.isin([-np.inf, np.inf]).any(axis=1)]


        # Convert severity and magnitude to positive values
        syn_droughts['severity'] = syn_droughts['severity'].abs()
        syn_droughts['magnitude'] = syn_droughts['magnitude'].abs()

        severity_min, severity_max = np.nanmin(syn_droughts['severity']), np.nanmax(syn_droughts['severity'])
        magnitude_min, magnitude_max = np.nanmin(syn_droughts['magnitude']), np.nanmax(syn_droughts['magnitude'])

        print(f"Severity - Min: {severity_min}, Max: {severity_max}")
        print(f"Magnitude - Min: {magnitude_min}, Max: {magnitude_max}")

        result = calculate_drought_frequency(
            syn_droughts,
            x1_metric='magnitude',
            x2_metric='severity',
            x1_range=[0, 120],
            x2_range=[1, 4],
            ngrid=100,
            n_realizations=1000,
            n_years=70
        )

        all_results[ensemble_type] = result
        
        print(f"Frequency matrix shape: {result['frequency_matrix'].shape}")
        print(f"Max frequency: {result['frequency_matrix'].max():.4f}")
        print(f"Max return period: {result['return_period_matrix'].max():.4f}")
        print(f"Total years analyzed: {result['total_years']}")

        plot_return_period = True
        if plot_return_period:
            fname = f"{FIG_DIR}/{ensemble_type}_ssi{SSI_WINDOW}_drought_return_period.png"
            vmin, vmax = 1, 1000
        else:
            fname = f"{FIG_DIR}/{ensemble_type}_ssi{SSI_WINDOW}_drought_frequency.png"
            vmin, vmax = 0, 0.5

        plot_drought_frequency_heatmap(result,
                                       syn_droughts=None,
                                       obs_droughts=obs_droughts,
                                    return_period=plot_return_period,
                                    vmin=vmin,
                                    vmax=vmax,
                                    fname=fname)


    ## Now, calculate the difference in return period for stationary and climate adjusted
    # we can calculate this using the frequency values to avoid inf
    eps = 1 / (70000)
    p_s = all_results['stationary']['frequency_matrix']
    p_s[p_s < eps] = eps
    p_c = all_results['climate_adjusted']['frequency_matrix']
    p_c[p_c < eps] = eps
    
    return_period_diff = (p_s/p_c) - 1.0
    return_period_diff *= 100
    
    

    # Print the fraction of return_period_diff values that are real number values
    is_real = np.isfinite(return_period_diff)
    print(f"Fraction of real values: {np.sum(is_real) / is_real.size:.4f}")

    diff_results = {
        "return_period_matrix" : return_period_diff,
        "frequency_matrix" : all_results['climate_adjusted']['frequency_matrix'] - all_results['stationary']['frequency_matrix'],
        "total_years" : all_results['stationary']['total_years'],
        "x1_grid": all_results['stationary']['x1_grid'],
        "x2_grid": all_results['stationary']['x2_grid'],
        "x1_metric": all_results['stationary']['x1_metric'],
        "x2_metric": all_results['stationary']['x2_metric'],
    }
    
    plot_return_period = True
    fname = f"{FIG_DIR}/ssi{SSI_WINDOW}_drought_return_period_percentage_difference.png"
    vmin, vmax = -50, 50
    plot_drought_frequency_heatmap(diff_results,
                                   obs_droughts=obs_droughts,
                                    return_period=plot_return_period,
                                    log_cmap=False,
                                    vmin=vmin,
                                    vmax=vmax,
                                    cmap='BrBG',
                                    fname=fname,
                                    difference=True)
