import sys
from scipy import stats
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
    # Validate
    required_cols = ['start', 'end', 'realization_id', x1_metric, x2_metric]
    missing_cols = [c for c in required_cols if c not in drought_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    if x1_range is None or x2_range is None:
        raise ValueError("x1_range and x2_range must be provided.")

    df = drought_df.copy()
    df['start'] = pd.to_datetime(df['start'])
    df['end']   = pd.to_datetime(df['end'])

    # --- Fit marginals on original scale (positive support) ---
    def fit_marginal(varname, data):
        data = np.asarray(data, dtype=float)
        data = data[np.isfinite(data) & (data > 0)]
        if varname.lower() == 'severity':
            dist = stats.genexpon
            params = dist.fit(data)
        elif varname.lower() == 'magnitude':
            dist = stats.norm
            params = dist.fit(data)
        else:
            raise ValueError("Unsupported metric for copula-based calculation "
                             "(use 'severity' or 'magnitude').")
        return dist, params

    dist_x1, pars_x1 = fit_marginal(x1_metric, df[x1_metric])
    dist_x2, pars_x2 = fit_marginal(x2_metric, df[x2_metric])

    # --- Gaussian copula parameter via normal-scores correlation ---
    eps = 1e-12
    u1_data = np.clip(dist_x1.cdf(df[x1_metric].to_numpy(float), *pars_x1), eps, 1 - eps)
    u2_data = np.clip(dist_x2.cdf(df[x2_metric].to_numpy(float), *pars_x2), eps, 1 - eps)
    z1 = stats.norm.ppf(u1_data)
    z2 = stats.norm.ppf(u2_data)
    rho = float(np.corrcoef(z1, z2)[0, 1])
    rho = float(np.clip(rho, -0.999, 0.999))
    cov = np.array([[1.0, rho], [rho, 1.0]])

    # --- Expected interarrival time (years), start-to-start by realization ---
    df_sorted = df.sort_values(['realization_id', 'start'])
    starts = (
        df_sorted.groupby('realization_id')['start']
        .apply(lambda s: s.sort_values().diff().dt.days.dropna())
        .to_numpy()
    )
    if starts.size == 0:
        # Fallback: average years per event using counts
        counts = df_sorted.groupby('realization_id').size().to_numpy()
        counts = counts[counts > 0]
        if counts.size == 0:
            raise ValueError("No events to estimate interarrival time.")
        E_L_years = float(np.mean(n_years / counts))
    else:
        E_L_years = float(np.mean(starts) / 365.25)
    print(f"Estimated mean interarrival time E[L]: {E_L_years:.2f} years")
    lam_all = 1.0 / E_L_years  # events/year (all droughts)

    # --- Grids (keep shapes consistent with plotting: rows=x1, cols=x2) ---
    x1_grid = np.linspace(x1_range[0], x1_range[1], ngrid)
    x2_grid = np.linspace(x2_range[0], x2_range[1], ngrid)

    U1 = np.clip(dist_x1.cdf(x1_grid, *pars_x1), eps, 1 - eps)  # F_X1(x1)
    U2 = np.clip(dist_x2.cdf(x2_grid, *pars_x2), eps, 1 - eps)  # F_X2(x2)

    # Build full (ngrid x ngrid) grid of (z1,z2) pairs and evaluate C(u,v)
    Z1v = stats.norm.ppf(U1)            # shape (ngrid,)
    Z2v = stats.norm.ppf(U2)            # shape (ngrid,)
    Z1g, Z2g = np.meshgrid(Z1v, Z2v, indexing='ij')
    pts = np.column_stack([Z1g.ravel(), Z2g.ravel()])

    mvn = stats.multivariate_normal(mean=[0.0, 0.0], cov=cov)
    C_uv = np.array([mvn.cdf(p) for p in pts], dtype=float).reshape(ngrid, ngrid)
    
    # Joint exceedance p = 1 - u - v + C(u,v)
    U1m = U1[:, None]
    U2m = U2[None, :]
    p_joint = 1.0 - U1m - U2m + C_uv
    p_joint = np.clip(p_joint, 1e-15, 1.0)

    # Return period and annual probability (Poisson at-least-one in a year)
    # T (years) = E[L]/p_joint ; P_year = 1 - exp(-lambda * p_joint)
    return_period = E_L_years / p_joint
    freq = 1.0 - np.exp(-lam_all * p_joint)

    # small floor for plotting stability
    denom_years = float(n_realizations * n_years)
    eps_plot = 1.0 / denom_years
    freq = np.maximum(freq, eps_plot)

    return {
        'frequency_matrix': freq,
        'return_period_matrix': return_period,
        'x1_grid': x1_grid,
        'x2_grid': x2_grid,
        'x1_metric': x1_metric,
        'x2_metric': x2_metric,
        'total_years': denom_years,
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
    obs_droughts['severity'] = np.log(obs_droughts['severity'].abs())
    obs_droughts['magnitude'] = np.log(obs_droughts['magnitude'].abs())

    all_results = {}

    for ensemble_type in ['stationary', 'climate_adjusted']:
        inflow_type = f"{ensemble_type}_ensemble"
        verify_ensemble_type(ensemble_type)

        syn_droughts = pd.read_csv(
            f"./pywrdrb/drought_metrics/{ensemble_type}_ensemble_ssi{SSI_WINDOW}_drought_events.csv"
        )
        syn_droughts = syn_droughts[~syn_droughts.isin([-np.inf, np.inf]).any(axis=1)]


        # Convert severity and magnitude to positive values
        syn_droughts['severity'] = np.log(syn_droughts['severity'].abs())
        syn_droughts['magnitude'] = np.log(syn_droughts['magnitude'].abs())

        severity_min, severity_max = np.nanmin(syn_droughts['severity']), np.nanmax(syn_droughts['severity'])
        magnitude_min, magnitude_max = np.nanmin(syn_droughts['magnitude']), np.nanmax(syn_droughts['magnitude'])

        print(f"Severity - Min: {severity_min}, Max: {severity_max}")
        print(f"Magnitude - Min: {magnitude_min}, Max: {magnitude_max}")

        result = calculate_drought_frequency(
            syn_droughts,
            x1_metric='magnitude',
            x2_metric='severity',
            x1_range=[0, 5.0],
            x2_range=[0, 2.0],
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
    T_s = all_results['stationary']['return_period_matrix']
    T_c = all_results['climate_adjusted']['return_period_matrix']
    return_period_diff = 100.0 * (T_c - T_s) / T_s    

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
    
    fname = f"{FIG_DIR}/ssi{SSI_WINDOW}_drought_return_period_percentage_difference.png"
    vmin, vmax = -50, 50
    plot_drought_frequency_heatmap(diff_results,
                                   obs_droughts=obs_droughts,
                                    return_period=True,
                                    log_cmap=False,
                                    vmin=vmin,
                                    vmax=vmax,
                                    cmap='BrBG',
                                    fname=fname,
                                    difference=True)
