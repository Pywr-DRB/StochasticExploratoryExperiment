import sys
import os
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors, ticker
from matplotlib.colors import TwoSlopeNorm
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

from config import *


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
    Efficient joint frequency calculation using copula-based approach.
    P(X1 >= x1 & X2 >= x2) per year, with Gaussian copula and marginal fits.
    """
    # Validate
    required_cols = ['start', 'end', 'realization_id', x1_metric, x2_metric]
    missing_cols = [c for c in required_cols if c not in drought_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    if x1_range is None or x2_range is None:
        raise ValueError("x1_range and x2_range must be provided.")

    df = drought_df.copy()
    df['start'] = pd.to_datetime(df['start'])
    df['end'] = pd.to_datetime(df['end'])

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
    lam_all = 1.0 / E_L_years  # events/year (all droughts)
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
        'interarrival_years': E_L_years,
        'copula_rho': rho,
        'severity_params': pars_x2,  # x2_metric is severity
        'magnitude_params': pars_x1,  # x1_metric is magnitude
    }


def _grid_edges_from_centers(grid: np.ndarray):
    """Assumes monotone, roughly uniform spacing; returns len(grid)+1 edges."""
    d = np.diff(grid)
    d0, d1 = d[0], d[-1]
    edges = np.empty(grid.size + 1, dtype=float)
    edges[1:-1] = 0.5*(grid[:-1] + grid[1:])
    edges[0] = grid[0] - 0.5*d0
    edges[-1] = grid[-1] + 0.5*d1
    return edges


def _load_observed_droughts(ssi_window=12):
    """
    Load and preprocess observed drought events.

    Parameters:
    -----------
    ssi_window : int
        SSI window size (months)

    Returns:
    --------
    pd.DataFrame
        Observed droughts with log-transformed severity and magnitude
    """
    obs_fname = f"./pywrdrb/drought_metrics/observed_ssi{ssi_window}_drought_events.csv"
    obs_droughts = pd.read_csv(obs_fname)
    obs_droughts['severity'] = np.log(obs_droughts['severity'].abs())
    obs_droughts['magnitude'] = np.log(obs_droughts['magnitude'].abs())
    return obs_droughts


def _plot_single_heatmap_panel(ax,
                               data_matrix,
                               x_grid,
                               y_grid,
                               cmap,
                               norm,
                               contour_levels=None,
                               contour_colors='white',
                               contour_fmt=None,
                               obs_droughts=None,
                               x_metric='severity',
                               y_metric='magnitude',
                               scatter_size=80,
                               show_grid=True):
    """
    Plot a single heatmap panel with optional contours and observed drought overlay.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    data_matrix : np.ndarray
        2D array of values to plot
    x_grid, y_grid : np.ndarray
        Grid centers for x and y axes
    cmap : matplotlib colormap
        Colormap to use
    norm : matplotlib normalization
        Color normalization
    contour_levels : list, optional
        Levels for contour lines
    contour_colors : str or list
        Colors for contour lines
    contour_fmt : callable, optional
        Format function for contour labels
    obs_droughts : pd.DataFrame, optional
        Observed drought events to overlay
    x_metric, y_metric : str
        Column names for scatter plot
    scatter_size : float
        Size of scatter points
    show_grid : bool
        Whether to show grid lines

    Returns:
    --------
    pcolormesh
        The pcolormesh object (for creating colorbars)
    """
    # Ensure valid data
    M = np.where(np.isfinite(data_matrix), data_matrix, np.nan)

    # Grid edges for pcolormesh
    x_edges = _grid_edges_from_centers(x_grid)
    y_edges = _grid_edges_from_centers(y_grid)

    # Heatmap
    pm = ax.pcolormesh(x_edges, y_edges, M, cmap=cmap, norm=norm, shading='auto')

    # Contours
    if contour_levels is not None:
        with np.errstate(invalid='ignore'):
            cs = ax.contour(x_grid, y_grid, M,
                           levels=contour_levels,
                           colors=contour_colors,
                           linewidths=1.0,
                           alpha=0.8)
            if contour_fmt is not None:
                ax.clabel(cs, fmt=contour_fmt, fontsize=9, inline=True, inline_spacing=5)

    # Observed droughts overlay
    if obs_droughts is not None:
        o = obs_droughts[[x_metric, y_metric]].dropna()
        ax.scatter(o[x_metric], o[y_metric],
                  s=scatter_size, marker='^', c='black',
                  edgecolors='white', linewidths=0.5,
                  alpha=0.95, label='Observed', zorder=10)

    # Grid
    if show_grid:
        ax.grid(which='both', color='white', alpha=0.15, linewidth=0.5)
        ax.set_axisbelow(True)

    return pm


def plot_drought_frequency_heatmap(freq_result, 
                                   syn_droughts=None,
                                   obs_droughts=None,
                                   return_period=True,
                                   figsize=(10, 8), 
                                   cmap='magma',
                                   log_cmap=True,
                                   vmin=None,
                                   vmax=None,
                                   fname=None,
                                   title=None,
                                   show_contours=True,
                                   contour_levels=None,
                                   difference=False):
    """Plot drought frequency or return period heatmap with optional overlays."""
    
    # Select matrix
    use_key = 'return_period_matrix' if return_period else 'frequency_matrix'
    M = np.array(freq_result[use_key], dtype=float)

    # Mask invalids
    if return_period:
        M = np.where(np.isfinite(M), M, np.nan)
    else:
        M = np.where(np.isfinite(M) & (M >= 0), M, np.nan)

    # Norm
    if return_period and not difference:
        if vmin is None: vmin = np.nanpercentile(M, 1) if np.isfinite(M).any() else 1.0
        if vmax is None: vmax = np.nanpercentile(M, 99) if np.isfinite(M).any() else 1000.0
        vmin = max(vmin, 1e-6)
        norm = colors.LogNorm(vmin=vmin, vmax=vmax) if log_cmap else colors.Normalize(vmin=vmin, vmax=vmax)
        cbar_label = "Return period T (years)"
        default_contours = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    else:
        if vmin is None: vmin = 0.0
        if vmax is None: vmax = float(np.nanmax(M)) if np.isfinite(M).any() else 0.5
        norm = colors.Normalize(vmin=vmin, vmax=vmax) if not log_cmap else colors.LogNorm(vmin=max(vmin,1e-6), vmax=vmax)
        cbar_label = "Frequency | P(X1 ≥ x1 & X2 ≥ x2)"
        default_contours = None

    if difference:
        if vmin<0 and vmax>0:
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        elif vmin<1 and vmax>1:
            norm = TwoSlopeNorm(vmin=vmin, vcenter=1, vmax=vmax)
        else:
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cbar_label = "Change in Return Period (%)"

    # Build figure
    fig, ax = plt.subplots(figsize=figsize)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color='#f0f0f0')

    # Determine contour settings
    levels = None
    contour_fmt = None
    if show_contours:
        levels = contour_levels if contour_levels is not None else default_contours
        if levels is not None:
            contour_fmt = lambda v: f"{int(v)}-yr" if return_period else f"{v:.2f}"

    # Use helper to plot heatmap
    pm = _plot_single_heatmap_panel(
        ax=ax,
        data_matrix=M,
        x_grid=freq_result['x2_grid'],
        y_grid=freq_result['x1_grid'],
        cmap=cmap_obj,
        norm=norm,
        contour_levels=levels,
        contour_colors='white',
        contour_fmt=contour_fmt,
        obs_droughts=obs_droughts,
        x_metric=freq_result['x2_metric'],
        y_metric=freq_result['x1_metric'],
        scatter_size=80,  # Updated from 30 to 80 for consistency
        show_grid=False  # Will add custom grid below
    )

    # Add synthetic droughts if provided (not in helper since it's unique to this function)
    xcol, ycol = freq_result['x2_metric'], freq_result['x1_metric']
    if syn_droughts is not None:
        s = syn_droughts[[xcol, ycol]].dropna()
        ax.scatter(s[xcol], s[ycol], s=10, facecolors='none', edgecolors='k', alpha=0.25, lw=0.5, label='Synthetic')

    # Axes labels & title
    ax.set_xlabel(f"{xcol.title()}")
    ax.set_ylabel(f"{ycol.title()}")
    if title:
        ax.set_title(title)

    # Colorbar
    extend = 'max' if not difference else 'both'
    cbar = plt.colorbar(pm, ax=ax, pad=0.02, extend=extend)
    cbar.set_label(cbar_label)
    if return_period and isinstance(norm, colors.LogNorm):
        ticks = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])
        ticks = ticks[(ticks >= vmin) & (ticks <= vmax)]
        if ticks.size:
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f"{int(t)}" for t in ticks])

    # Minor gridlines
    ax.grid(which='both', color='white', alpha=0.2, linewidth=0.5)
    ax.set_axisbelow(True)

    # Legend
    if (syn_droughts is not None) or (obs_droughts is not None):
        ax.legend(frameon=False, loc='lower center', bbox_to_anchor=(1.02, 1.0))

    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname, dpi=400, bbox_inches='tight')
        # Also save vector
        base = fname.rsplit('.', 1)[0]
        plt.savefig(f"{base}.svg", bbox_inches='tight')
    return fig, ax


def analyze_drought_frequency(dataset_id, ssi_window=12):
    """
    Analyze drought frequency for a dataset
    
    Parameters:
    -----------
    dataset_id : str
        Dataset identifier to analyze
    ssi_window : int
        SSI window size (months)
    """
    
    # Verify dataset
    verify_dataset_id(dataset_id)
    dataset_config = DATASET_CONFIGS[dataset_id]
    
    print(f"Analyzing drought frequency for: {dataset_id}")
    print(f"Dataset type: {dataset_config['type']}")
    print(f"SSI window: {ssi_window} months")
    
    # Load drought events
    syn_fname = f"./pywrdrb/drought_metrics/{dataset_id}_ssi{ssi_window}_drought_events.csv"
    obs_fname = f"./pywrdrb/drought_metrics/observed_ssi{ssi_window}_drought_events.csv"
    
    if not os.path.exists(syn_fname):
        print(f"ERROR: Drought events file not found: {syn_fname}")
        print(f"Run 05_calculate_ssi_drought_metrics.py first for {dataset_id}!")
        return None
    
    syn_droughts = pd.read_csv(syn_fname)
    obs_droughts = pd.read_csv(obs_fname)
    
    # Remove infinite values
    syn_droughts = syn_droughts[~syn_droughts.isin([-np.inf, np.inf]).any(axis=1)]
    
    # Convert severity and magnitude to log scale (positive values)
    syn_droughts['severity'] = np.log(syn_droughts['severity'].abs())
    syn_droughts['magnitude'] = np.log(syn_droughts['magnitude'].abs())
    obs_droughts['severity'] = np.log(obs_droughts['severity'].abs())
    obs_droughts['magnitude'] = np.log(obs_droughts['magnitude'].abs())
    
    # Get data ranges
    severity_min, severity_max = np.nanmin(syn_droughts['severity']), np.nanmax(syn_droughts['severity'])
    magnitude_min, magnitude_max = np.nanmin(syn_droughts['magnitude']), np.nanmax(syn_droughts['magnitude'])
    
    print(f"  Severity range (log): [{severity_min:.2f}, {severity_max:.2f}]")
    print(f"  Magnitude range (log): [{magnitude_min:.2f}, {magnitude_max:.2f}]")
    
    # Calculate frequency
    result = calculate_drought_frequency(
        syn_droughts,
        x1_metric='magnitude',
        x2_metric='severity',
        x1_range=[0, 5.0],
        x2_range=[0, 2.0],
        ngrid=100,
        n_realizations=TOTAL_REALIZATIONS,
        n_years=N_YEARS
    )
    
    print(f"  Max frequency: {result['frequency_matrix'].max():.4f}")
    print(f"  Max return period: {result['return_period_matrix'].max():.1f} years")
    print(f"  Mean interarrival time: {result['interarrival_years']:.2f} years")
    print(f"  Copula correlation (ρ): {result['copula_rho']:.4f}")
    print(f"  Severity params (genexpon): {result['severity_params']}")
    print(f"  Magnitude params (norm): μ={result['magnitude_params'][0]:.3f}, σ={result['magnitude_params'][1]:.3f}")

    return result, syn_droughts, obs_droughts


def plot_drought_frequency_analysis(dataset_id, ssi_window=12):
    """
    Generate drought frequency plots for a dataset
    
    Parameters:
    -----------
    dataset_id : str
        Dataset identifier to analyze
    ssi_window : int
        SSI window size (months)
    """
    
    # Analyze drought frequency
    result = analyze_drought_frequency(dataset_id, ssi_window)
    
    if result is None:
        return None
    
    result, syn_droughts, obs_droughts = result
    
    # Create output directory
    output_dir = f"{FIG_DIR}/drought_return_period"
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot return period
    print(f"Plotting return period heatmap...")
    fname = f"{output_dir}/{dataset_id}_ssi{ssi_window}_drought_return_period.png"
    
    plot_drought_frequency_heatmap(
        result,
        syn_droughts=None,  # Can add syn_droughts for overlay
        obs_droughts=obs_droughts,
        return_period=True,
        vmin=1,
        vmax=1000,
        title=f"{dataset_id} - Drought Return Period (SSI-{ssi_window})",
        fname=fname
    )
    print(f"  Saved: {fname}")
    
    return result


def plot_4panel_comparison(ssi_window=12,
                           figsize=(14, 10),
                           vmin_abs=1,
                           vmax_abs=1000,
                           vmin_diff=-1.0,
                           vmax_diff=1.0,
                           fname=None):
    """
    Create a 4-panel comparison figure showing return periods for all scenarios.

    Layout:
    - Left panel: Stationary ensemble (absolute return period)
    - Right panels (stacked): Low, Medium, High climate scenarios (relative change from stationary)

    Parameters:
    -----------
    ssi_window : int
        SSI window size (months)
    figsize : tuple
        Figure size in inches
    vmin_abs, vmax_abs : float
        Color scale limits for absolute return period (left panel)
    vmin_diff, vmax_diff : float
        Color scale limits for log ratio difference (right panels)
    fname : str
        Output filename (if None, will auto-generate)
    """

    print(f"\n{'='*60}")
    print("Creating 4-Panel Comparison Figure")
    print(f"{'='*60}")

    # Define datasets to plot
    datasets = {
        'stationary_ensemble': 'Stationary',
        'climate_adjusted_low': 'Low',
        'climate_adjusted_medium': 'Medium',
        'climate_adjusted_high': 'High'
    }

    # Calculate frequency for all datasets
    all_results = {}

    for dataset_id, label in datasets.items():
        print(f"\nAnalyzing {dataset_id} ({label})...")
        result = analyze_drought_frequency(dataset_id, ssi_window)
        if result is None:
            print(f"ERROR: Could not analyze {dataset_id}")
            return None
        all_results[dataset_id] = result[0]  # Just the frequency result

    # Load observed droughts once
    obs_droughts = _load_observed_droughts(ssi_window)

    # Print comparison summary of copula parameters
    print(f"\n{'='*60}")
    print("COPULA PARAMETER COMPARISON ACROSS DATASETS")
    print(f"{'='*60}")
    print(f"{'Dataset':<25} {'ρ':>8} {'E[L] (yr)':>12} {'μ_mag':>10} {'σ_mag':>10}")
    print("-" * 75)
    for dataset_id, label in datasets.items():
        res = all_results[dataset_id]
        print(f"{label:<25} {res['copula_rho']:>8.4f} {res['interarrival_years']:>12.2f} "
              f"{res['magnitude_params'][0]:>10.3f} {res['magnitude_params'][1]:>10.3f}")
    print("=" * 75)
    print("Note: ρ = copula correlation, E[L] = interarrival time,")
    print("      μ_mag/σ_mag = magnitude distribution parameters (log-normal)")
    print("")

    # Calculate relative changes (log ratio) for climate scenarios
    print(f"{'='*60}")
    print("Calculating relative changes from stationary...")
    print(f"{'='*60}")

    T_ref = all_results['stationary_ensemble']['return_period_matrix']
    eps = 1e-8

    diff_results = {}
    for dataset_id in ['climate_adjusted_low', 'climate_adjusted_medium', 'climate_adjusted_high']:
        T_comp = all_results[dataset_id]['return_period_matrix']
        log_ratio = np.log10(np.maximum(T_comp, eps) / np.maximum(T_ref, eps))

        diff_results[dataset_id] = {
            'log_ratio_matrix': log_ratio,
            'x1_grid': all_results['stationary_ensemble']['x1_grid'],
            'x2_grid': all_results['stationary_ensemble']['x2_grid'],
            'x1_metric': all_results['stationary_ensemble']['x1_metric'],
            'x2_metric': all_results['stationary_ensemble']['x2_metric'],
        }

    print(f"\n{'='*60}")
    print("Creating multi-panel figure...")
    print(f"{'='*60}")

    # Set up figure with GridSpec for flexible layout
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1],
                          hspace=0.15, wspace=0.25,
                          left=0.08, right=0.95, top=0.95, bottom=0.12)

    # Create axes
    ax_stat = fig.add_subplot(gs[:, 0])  # Left panel spans all rows
    ax_low = fig.add_subplot(gs[0, 1])   # Top right
    ax_med = fig.add_subplot(gs[1, 1])   # Middle right
    ax_high = fig.add_subplot(gs[2, 1])  # Bottom right

    axes = [ax_stat, ax_low, ax_med, ax_high]
    dataset_list = list(datasets.keys())
    panel_labels = ['(a)', '(b)', '(c)', '(d)']

    # Set up colormaps and norms
    # Left panel: absolute return period
    cmap_abs = plt.get_cmap('magma').copy()
    cmap_abs.set_bad(color='#f0f0f0')
    norm_abs = colors.LogNorm(vmin=vmin_abs, vmax=vmax_abs)

    # Right panels: relative change (diverging colormap)
    cmap_diff = plt.get_cmap('BrBG').copy()
    cmap_diff.set_bad(color='#f0f0f0')
    norm_diff = TwoSlopeNorm(vmin=vmin_diff, vcenter=0, vmax=vmax_diff)

    # Contour levels
    contour_levels_abs = [2, 5, 10, 20, 50, 100, 200, 500]

    # Storage for colormeshes (to create colorbars later)
    pm_abs = None
    pm_diff = None

    # Plot each panel
    for idx, (ax, dataset_id, panel_label) in enumerate(zip(axes, dataset_list, panel_labels)):

        if idx == 0:  # Stationary panel (absolute values)
            result = all_results[dataset_id]

            pm_abs = _plot_single_heatmap_panel(
                ax=ax,
                data_matrix=result['return_period_matrix'],
                x_grid=result['x2_grid'],
                y_grid=result['x1_grid'],
                cmap=cmap_abs,
                norm=norm_abs,
                contour_levels=contour_levels_abs,
                contour_colors='white',
                contour_fmt=lambda v: f"{int(v)}",
                obs_droughts=obs_droughts,
                x_metric='severity',
                y_metric='magnitude',
                scatter_size=80,
                show_grid=True
            )

        else:  # Climate scenario panels (relative change)
            diff_res = diff_results[dataset_id]

            pm_diff = _plot_single_heatmap_panel(
                ax=ax,
                data_matrix=diff_res['log_ratio_matrix'],
                x_grid=diff_res['x2_grid'],
                y_grid=diff_res['x1_grid'],
                cmap=cmap_diff,
                norm=norm_diff,
                contour_levels=[0],  # Zero contour line
                contour_colors='black',
                contour_fmt=None,  # No labels on zero contour
                obs_droughts=obs_droughts,
                x_metric='severity',
                y_metric='magnitude',
                scatter_size=80,
                show_grid=True
            )

        # Panel title
        title_text = datasets[dataset_id]
        ax.text(0.02, 0.98, f"{panel_label} {title_text}",
               transform=ax.transAxes, fontsize=13, fontweight='bold',
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, pad=0.3))

        # Axis labels
        if idx == 0:  # Stationary (left panel)
            ax.set_xlabel('Severity (log)', fontsize=12)
            ax.set_ylabel('Magnitude (log)', fontsize=12)
        elif idx == 3:  # Bottom right panel
            ax.set_xlabel('Severity (log)', fontsize=12)
        else:  # Other right panels
            ax.set_xticklabels([])

        # Y-axis labels only for left panel
        if idx != 0:
            ax.set_ylabel('')

        # Tick label sizes
        ax.tick_params(labelsize=10)

    # Add two colorbars at bottom
    # Left colorbar for absolute return period
    cbar_abs_ax = fig.add_axes([0.08, 0.04, 0.35, 0.02])  # [left, bottom, width, height]
    cbar_abs = fig.colorbar(pm_abs, cax=cbar_abs_ax, orientation='horizontal', extend='max')
    cbar_abs.set_label('Return Period (years)', fontsize=11, fontweight='bold')

    # Set colorbar ticks for absolute
    ticks = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])
    ticks = ticks[(ticks >= vmin_abs) & (ticks <= vmax_abs)]
    cbar_abs.set_ticks(ticks)
    cbar_abs.set_ticklabels([f"{int(t)}" for t in ticks], fontsize=9)

    # Right colorbar for relative change
    cbar_diff_ax = fig.add_axes([0.56, 0.04, 0.35, 0.02])  # [left, bottom, width, height]
    cbar_diff = fig.colorbar(pm_diff, cax=cbar_diff_ax, orientation='horizontal', extend='both')
    cbar_diff.set_label('Change in Return Period (log₁₀ ratio)', fontsize=11, fontweight='bold')

    # Set colorbar ticks for difference
    diff_ticks = np.linspace(vmin_diff, vmax_diff, 9)
    cbar_diff.set_ticks(diff_ticks)
    cbar_diff.set_ticklabels([f"{t:.1f}" for t in diff_ticks], fontsize=9)

    # Add legend for observed droughts
    handles = [plt.Line2D([0], [0], marker='^', color='w',
                         markerfacecolor='black', markeredgecolor='white',
                         markersize=10, label='Observed', linewidth=0)]
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.97, 0.98),
              fontsize=11, frameon=True, fancybox=True, shadow=True)

    # Save figure
    if fname is None:
        output_dir = f"{FIG_DIR}/drought_return_period"
        os.makedirs(output_dir, exist_ok=True)
        fname = f"{output_dir}/comparison_4panel_ssi{ssi_window}_drought_return_period.png"

    plt.savefig(fname, dpi=400, bbox_inches='tight')
    # Also save vector version
    base = fname.rsplit('.', 1)[0]
    plt.savefig(f"{base}.svg", bbox_inches='tight')

    print(f"\nSaved: {fname}")
    print(f"Saved: {base}.svg")

    return fig, axes


def compare_drought_frequencies(dataset_ids, 
                                ssi_window=12,
                                vmax=None, 
                                vmin=None):
    """
    Compare drought frequencies between multiple datasets
    
    Parameters:
    -----------
    dataset_ids : list
        List of dataset identifiers to compare
    ssi_window : int
        SSI window size (months)
    """
    
    print(f"\nComparing drought frequencies between datasets...")
    
    all_results = {}
    
    # Calculate for each dataset
    for dataset_id in dataset_ids:
        result = analyze_drought_frequency(dataset_id, ssi_window)
        if result is not None:
            all_results[dataset_id] = result[0]  # Just the frequency result
    
    if len(all_results) < 2:
        print("Need at least 2 datasets for comparison")
        return
    
    # If we have stationary as reference, compare others against it
    if 'stationary_ensemble' in all_results:
        print("\nGenerating comparison plots...")
        
        T_ref = all_results['stationary_ensemble']['return_period_matrix']
        
        for dataset_id in all_results:
            if dataset_id == 'stationary_ensemble':
                continue
            
            print(f"  Comparing {dataset_id} vs stationary...")
            
            T_comp = all_results[dataset_id]['return_period_matrix']
            
            # Calculate percentage difference
            eps = 1e-8
            return_period_diff_perc = 100.0 * (T_comp - T_ref) / np.maximum(T_ref, eps)
            log_ratio = np.log10(np.maximum(T_comp, eps) / np.maximum(T_ref, eps))
            
            # Create difference results dict
            diff_results = {
                "return_period_matrix": log_ratio,
                "frequency_matrix": all_results[dataset_id]['frequency_matrix'] - 
                                   all_results['stationary_ensemble']['frequency_matrix'],
                "total_years": all_results['stationary_ensemble']['total_years'],
                "x1_grid": all_results['stationary_ensemble']['x1_grid'],
                "x2_grid": all_results['stationary_ensemble']['x2_grid'],
                "x1_metric": all_results['stationary_ensemble']['x1_metric'],
                "x2_metric": all_results['stationary_ensemble']['x2_metric'],
            }
            
            # Plot percentage difference
            output_dir = f"{FIG_DIR}/drought_return_period"
            fname = f"{output_dir}/{dataset_id}_vs_stationary_ssi{ssi_window}_return_period_diff.png"
            
            # Load observed droughts for overlay
            obs_fname = f"./pywrdrb/drought_metrics/observed_ssi{ssi_window}_drought_events.csv"
            obs_droughts = pd.read_csv(obs_fname)
            obs_droughts['severity'] = np.log(obs_droughts['severity'].abs())
            obs_droughts['magnitude'] = np.log(obs_droughts['magnitude'].abs())
            
            plot_drought_frequency_heatmap(
                diff_results,
                obs_droughts=obs_droughts,
                return_period=True,
                log_cmap=False,
                vmin=vmin,
                vmax=vmax,
                cmap='BrBG',
                title=f"Return Period Change: {dataset_id} vs stationary (%)",
                fname=fname,
                difference=True
            )
            print(f"    Saved: {fname}")


def main(dataset_id):
    """Main function"""

    print("=" * 60)
    print(f"DROUGHT FREQUENCY ANALYSIS: {dataset_id}")
    print("=" * 60)

    # Handle special 'comparison' dataset ID for 4-panel figure
    if dataset_id.lower() == 'comparison':
        print("\nGenerating 4-panel comparison figure...")
        plot_4panel_comparison(ssi_window=12)
        print("=" * 60)
        print("4-panel comparison figure completed successfully!")
        return

    # Analyze drought frequency for this dataset
    plot_drought_frequency_analysis(dataset_id, ssi_window=12)

    # If not stationary, also compare with stationary
    if dataset_id != 'stationary_ensemble':
        compare_drought_frequencies(['stationary_ensemble', dataset_id], ssi_window=12,
                                    vmin=-1, vmax=1)

    print("=" * 60)
    print("Drought frequency analysis completed successfully!")


if __name__ == "__main__":

    # Get the dataset_id from command line arguments
    if len(sys.argv) != 2:
        print("Usage: python 09_plot_drought_frequency.py <dataset_id>")
        print(f"Available datasets: {list(DATASET_CONFIGS.keys())}")
        print("Special option: 'comparison' - generates 4-panel comparison figure")
        sys.exit(1)

    dataset_id = sys.argv[1]

    # Skip verification for special 'comparison' option
    if dataset_id.lower() != 'comparison':
        verify_dataset_id(dataset_id)

    main(dataset_id)