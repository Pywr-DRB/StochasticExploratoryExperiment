"""
Comprehensive diagnostic testing for drought copula distributions.

This script systematically evaluates:
1. Marginal distribution fits (severity and magnitude)
2. Uniformity of transformed data
3. Copula choice (Gaussian vs. Student-t)
4. Tail dependence assessment

Tests are performed for all SSI windows [3, 6, 12] and datasets.

The goal is to determine the optimal statistical model for joint
severity-magnitude drought probability modeling.

Usage:
------
python 07_test_copula_distributions.py [dataset_id]

Arguments:
    dataset_id : str, optional (default='stationary_ensemble')
        Dataset identifier to analyze

Example:
    python 07_test_copula_distributions.py stationary_ensemble

Output:
-------
figures/copula_diagnostics/distribution_tests/<dataset_id>/
    - marginal_fit_comparison_ssi{window}.png
    - uniformity_tests_ssi{window}.png
    - copula_comparison_ssi{window}.png
    - tail_dependence_ssi{window}.png
    - distribution_test_summary.txt
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import (
    kstest, anderson, shapiro,
    multivariate_normal, multivariate_t,
    norm, t, expon, gamma, lognorm, weibull_min, genexpon
)
import warnings
warnings.filterwarnings('ignore')

from config import *
from methods.copula import load_drought_events


# =============================================================================
# CANDIDATE DISTRIBUTIONS
# =============================================================================

# Candidate distributions for severity (log-transformed, positive support)
SEVERITY_CANDIDATES = {
    'genexpon': stats.genexpon,
    'gamma': stats.gamma,
    'weibull': stats.weibull_min,
    'lognorm': stats.lognorm,
    'expon': stats.expon,
}

# Candidate distributions for magnitude (log-transformed, positive support)
MAGNITUDE_CANDIDATES = {
    'norm': stats.norm,
    'truncnorm_0': lambda: stats.truncnorm(0, np.inf, loc=0, scale=1),  # Truncated at 0
    'gamma': stats.gamma,
    'lognorm': stats.lognorm,
}


# =============================================================================
# GOODNESS-OF-FIT TESTS
# =============================================================================

def fit_and_test_distribution(data, dist, dist_name):
    """
    Fit distribution and calculate goodness-of-fit statistics.

    Parameters
    ----------
    data : np.ndarray
        Data to fit (positive values)
    dist : scipy.stats distribution
        Distribution to fit
    dist_name : str
        Distribution name

    Returns
    -------
    dict
        Dictionary with fit results and test statistics
    """
    data = data[np.isfinite(data) & (data > 0)]

    # Handle truncnorm specially
    if dist_name == 'truncnorm_0':
        # Fit normal to data, then create truncated version
        mu, sigma = stats.norm.fit(data)
        a = (0 - mu) / sigma  # Lower bound in standardized form
        b = np.inf
        params = (a, b, mu, sigma)
        fitted_dist = stats.truncnorm(a, b, loc=mu, scale=sigma)
    else:
        # Standard fitting
        try:
            params = dist.fit(data)
            fitted_dist = dist
        except Exception as e:
            return {
                'dist_name': dist_name,
                'params': None,
                'aic': np.inf,
                'bic': np.inf,
                'ks_stat': np.nan,
                'ks_pvalue': np.nan,
                'ad_stat': np.nan,
                'error': str(e)
            }

    # Log-likelihood
    try:
        if dist_name == 'truncnorm_0':
            log_lik = np.sum(fitted_dist.logpdf(data))
        else:
            log_lik = np.sum(fitted_dist.logpdf(data, *params))
    except:
        log_lik = -np.inf

    # AIC and BIC
    k = len(params)
    n = len(data)
    aic = 2 * k - 2 * log_lik
    bic = k * np.log(n) - 2 * log_lik

    # Kolmogorov-Smirnov test
    if dist_name == 'truncnorm_0':
        ks_stat, ks_pvalue = stats.kstest(data, lambda x: fitted_dist.cdf(x))
    else:
        ks_stat, ks_pvalue = stats.kstest(data, dist.cdf, args=params)

    # Anderson-Darling test (for norm, expon, gamma if available)
    try:
        if dist_name in ['norm', 'expon', 'gamma'] and dist_name != 'truncnorm_0':
            ad_result = stats.anderson(data, dist=dist_name)
            ad_stat = ad_result.statistic
        else:
            ad_stat = np.nan
    except:
        ad_stat = np.nan

    return {
        'dist_name': dist_name,
        'params': params,
        'log_lik': log_lik,
        'aic': aic,
        'bic': bic,
        'ks_stat': ks_stat,
        'ks_pvalue': ks_pvalue,
        'ad_stat': ad_stat,
        'n_params': k,
        'fitted_dist': fitted_dist
    }


def test_uniformity(u_data):
    """
    Test if transformed data is uniform on [0,1].

    Parameters
    ----------
    u_data : np.ndarray
        Transformed data (should be uniform)

    Returns
    -------
    dict
        Dictionary with uniformity test results
    """
    # Kolmogorov-Smirnov test for uniformity
    ks_stat, ks_pvalue = stats.kstest(u_data, 'uniform')

    # Anderson-Darling test for uniformity
    # Transform to normal quantiles first
    z = stats.norm.ppf(np.clip(u_data, 1e-10, 1-1e-10))
    ad_result = stats.anderson(z, dist='norm')

    # Chi-square test (bin into 10 bins)
    observed, _ = np.histogram(u_data, bins=10, range=(0, 1))
    expected = len(u_data) / 10
    chi2_stat = np.sum((observed - expected)**2 / expected)
    chi2_pvalue = 1 - stats.chi2.cdf(chi2_stat, df=9)

    return {
        'ks_stat': ks_stat,
        'ks_pvalue': ks_pvalue,
        'ad_stat': ad_result.statistic,
        'chi2_stat': chi2_stat,
        'chi2_pvalue': chi2_pvalue,
    }


# =============================================================================
# COPULA COMPARISON
# =============================================================================

def fit_and_compare_copulas(U, df_original):
    """
    Fit Gaussian and Student-t copulas and compare.

    Parameters
    ----------
    U : np.ndarray
        (n, 2) array of uniform marginals
    df_original : pd.DataFrame
        Original data for Kendall's tau

    Returns
    -------
    dict
        Dictionary with copula comparison results
    """
    eps = 1e-12
    U_clip = np.clip(U, eps, 1 - eps)

    # Gaussian copula
    z1 = norm.ppf(U_clip[:, 0])
    z2 = norm.ppf(U_clip[:, 1])
    rho_gauss = np.corrcoef(z1, z2)[0, 1]
    rho_gauss = np.clip(rho_gauss, -0.999, 0.999)

    cov_gauss = np.array([[1.0, rho_gauss], [rho_gauss, 1.0]])
    loglik_gauss = multivariate_normal.logpdf(
        np.column_stack([z1, z2]),
        cov=cov_gauss
    ).sum()

    # Student-t copula (MLE for rho and nu)
    def nll_t(params):
        a, b = params
        rho_t = np.tanh(a)
        nu_t = np.exp(b) + 2.0
        z = stats.t.ppf(U_clip, df=nu_t)
        ll2 = multivariate_t.logpdf(
            z,
            loc=np.zeros(2),
            shape=np.array([[1.0, rho_t], [rho_t, 1.0]]),
            df=nu_t
        )
        ll1 = stats.t.logpdf(z[:, 0], df=nu_t) + stats.t.logpdf(z[:, 1], df=nu_t)
        return -(np.sum(ll2 - ll1))

    # Initial guess
    a0 = np.arctanh(np.clip(rho_gauss, -0.99, 0.99))
    b0 = np.log(10.0 - 2.0)

    from scipy.optimize import minimize
    opt = minimize(nll_t, x0=np.array([a0, b0]), method='L-BFGS-B')
    rho_t = np.tanh(opt.x[0])
    nu_t = np.exp(opt.x[1]) + 2.0
    loglik_t = -opt.fun

    # AIC and BIC
    n = len(U)
    aic_gauss = 2 * 1 - 2 * loglik_gauss  # 1 parameter (rho)
    bic_gauss = 1 * np.log(n) - 2 * loglik_gauss
    aic_t = 2 * 2 - 2 * loglik_t  # 2 parameters (rho, nu)
    bic_t = 2 * np.log(n) - 2 * loglik_t

    # Likelihood ratio test
    lr_stat = 2 * (loglik_t - loglik_gauss)
    lr_pvalue = 1 - stats.chi2.cdf(lr_stat, df=1)  # 1 extra parameter

    # Tail dependence
    tau, _ = stats.kendalltau(df_original['severity'], df_original['magnitude'])

    # Empirical tail dependence
    qL, qU = 0.05, 0.95
    lambda_L = np.mean((U[:, 0] <= qL) & (U[:, 1] <= qL)) / qL
    lambda_U = np.mean((U[:, 0] >= qU) & (U[:, 1] >= qU)) / (1.0 - qU)

    # Theoretical tail dependence for t-copula
    lambda_t_theory = 2 * t.cdf(-np.sqrt(((nu_t + 1) * (1 - rho_t)) / (1 + rho_t)), df=nu_t + 1)

    return {
        'gaussian': {
            'rho': rho_gauss,
            'loglik': loglik_gauss,
            'aic': aic_gauss,
            'bic': bic_gauss,
        },
        't_copula': {
            'rho': rho_t,
            'nu': nu_t,
            'loglik': loglik_t,
            'aic': aic_t,
            'bic': bic_t,
            'lambda_theory': lambda_t_theory,
        },
        'comparison': {
            'lr_stat': lr_stat,
            'lr_pvalue': lr_pvalue,
            'delta_aic': aic_t - aic_gauss,
            'delta_bic': bic_t - bic_gauss,
        },
        'tail_dependence': {
            'kendall_tau': tau,
            'lambda_L_empirical': lambda_L,
            'lambda_U_empirical': lambda_U,
        }
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_marginal_comparison(data, fits, metric_name, ssi_window, dataset_id, output_dir):
    """
    Plot comparison of all candidate marginal distributions.

    Parameters
    ----------
    data : np.ndarray
        Original data
    fits : list of dict
        Fit results for each candidate distribution
    metric_name : str
        'severity' or 'magnitude'
    ssi_window : int
        SSI window size
    dataset_id : str
        Dataset identifier
    output_dir : str
        Output directory

    Returns
    -------
    str
        Path to saved figure
    """
    # Sort by AIC
    fits_sorted = sorted([f for f in fits if f['params'] is not None], key=lambda x: x['aic'])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # Plot top 4 distributions
    for idx, fit in enumerate(fits_sorted[:4]):
        ax = axes[idx]

        # Histogram
        ax.hist(data, bins=100, density=True, alpha=0.6, color='skyblue',
                edgecolor='black', label='Empirical')

        # Fitted distribution
        x = np.linspace(data.min(), data.max(), 500)
        if fit['dist_name'] == 'truncnorm_0':
            pdf = fit['fitted_dist'].pdf(x)
        else:
            pdf = fit['fitted_dist'].pdf(x, *fit['params'])

        ax.plot(x, pdf, 'r-', lw=2.5, label=f"{fit['dist_name'].title()}")

        # Labels and stats
        ax.set_xlabel(f"{metric_name.title()} (log)", fontsize=11, fontweight='bold')
        ax.set_ylabel('Density', fontsize=11, fontweight='bold')
        ax.set_title(
            f"Rank {idx+1}: {fit['dist_name'].upper()}\n"
            f"AIC={fit['aic']:.1f}, BIC={fit['bic']:.1f}, KS p={fit['ks_pvalue']:.3f}",
            fontsize=12, fontweight='bold'
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"{metric_name.title()} Marginal Distribution Comparison\n"
        f"{dataset_id} - SSI-{ssi_window}",
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()

    fname = f"{output_dir}/marginal_fit_comparison_{metric_name}_ssi{ssi_window}.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close()

    return fname


def plot_uniformity_diagnostics(U, uniformity_tests, best_fits, ssi_window, dataset_id, output_dir):
    """
    Plot uniformity diagnostics for transformed data.

    Parameters
    ----------
    U : np.ndarray
        (n, 2) transformed uniform marginals
    uniformity_tests : dict
        Uniformity test results
    best_fits : dict
        Best fit distributions
    ssi_window : int
        SSI window size
    dataset_id : str
        Dataset identifier
    output_dir : str
        Output directory

    Returns
    -------
    str
        Path to saved figure
    """
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Severity uniformity
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(U[:, 0], bins=50, density=True, alpha=0.7, color='skyblue',
            edgecolor='black', label='Transformed Severity')
    ax1.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Uniform(0,1)')
    ax1.set_xlabel('U (Severity)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Density', fontsize=11, fontweight='bold')
    ax1.set_title(
        f"Severity Uniformity\n"
        f"KS p={uniformity_tests['severity']['ks_pvalue']:.3f}",
        fontsize=12, fontweight='bold'
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Magnitude uniformity
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(U[:, 1], bins=50, density=True, alpha=0.7, color='lightcoral',
            edgecolor='black', label='Transformed Magnitude')
    ax2.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Uniform(0,1)')
    ax2.set_xlabel('U (Magnitude)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Density', fontsize=11, fontweight='bold')
    ax2.set_title(
        f"Magnitude Uniformity\n"
        f"KS p={uniformity_tests['magnitude']['ks_pvalue']:.3f}",
        fontsize=12, fontweight='bold'
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Q-Q plot: Severity
    ax3 = fig.add_subplot(gs[1, 0])
    theoretical_q = np.linspace(0, 1, len(U[:, 0]))
    empirical_q = np.sort(U[:, 0])
    ax3.scatter(theoretical_q, empirical_q, alpha=0.5, s=10, color='skyblue')
    ax3.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Uniform')
    ax3.set_xlabel('Theoretical Quantiles', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Empirical Quantiles', fontsize=11, fontweight='bold')
    ax3.set_title('Severity Q-Q Plot (Uniform)', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Q-Q plot: Magnitude
    ax4 = fig.add_subplot(gs[1, 1])
    empirical_q_mag = np.sort(U[:, 1])
    ax4.scatter(theoretical_q, empirical_q_mag, alpha=0.5, s=10, color='lightcoral')
    ax4.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Uniform')
    ax4.set_xlabel('Theoretical Quantiles', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Empirical Quantiles', fontsize=11, fontweight='bold')
    ax4.set_title('Magnitude Q-Q Plot (Uniform)', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Joint copula
    ax5 = fig.add_subplot(gs[2, :])
    ax5.scatter(U[:, 0], U[:, 1], alpha=0.2, s=10, color='purple')
    ax5.axhline(0.5, color='red', linestyle='--', alpha=0.5)
    ax5.axvline(0.5, color='red', linestyle='--', alpha=0.5)
    ax5.set_xlabel('U1 (Severity)', fontsize=11, fontweight='bold')
    ax5.set_ylabel('U2 (Magnitude)', fontsize=11, fontweight='bold')
    ax5.set_title('Joint Empirical Copula', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)

    fig.suptitle(
        f"Uniformity Diagnostics: {dataset_id} (SSI-{ssi_window})\n"
        f"Severity: {best_fits['severity']['dist_name']}, "
        f"Magnitude: {best_fits['magnitude']['dist_name']}",
        fontsize=14, fontweight='bold'
    )

    fname = f"{output_dir}/uniformity_tests_ssi{ssi_window}.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close()

    return fname


def plot_copula_comparison(copula_results, ssi_window, dataset_id, output_dir):
    """
    Plot comparison of Gaussian vs. Student-t copula.

    Parameters
    ----------
    copula_results : dict
        Copula comparison results
    ssi_window : int
        SSI window size
    dataset_id : str
        Dataset identifier
    output_dir : str
        Output directory

    Returns
    -------
    str
        Path to saved figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Extract results
    gauss = copula_results['gaussian']
    t_cop = copula_results['t_copula']
    comp = copula_results['comparison']
    tail = copula_results['tail_dependence']

    # Panel 1: Log-likelihood comparison
    ax1 = axes[0, 0]
    copulas = ['Gaussian', 'Student-t']
    logliks = [gauss['loglik'], t_cop['loglik']]
    colors = ['#1f77b4', '#ff7f0e']
    ax1.bar(copulas, logliks, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Log-Likelihood', fontsize=11, fontweight='bold')
    ax1.set_title('Log-Likelihood Comparison', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(logliks):
        ax1.text(i, v, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')

    # Panel 2: AIC/BIC comparison
    ax2 = axes[0, 1]
    x = np.arange(2)
    width = 0.35
    ax2.bar(x - width/2, [gauss['aic'], gauss['bic']], width,
           label='Gaussian', color='#1f77b4', alpha=0.7, edgecolor='black')
    ax2.bar(x + width/2, [t_cop['aic'], t_cop['bic']], width,
           label='Student-t', color='#ff7f0e', alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Information Criterion', fontsize=11, fontweight='bold')
    ax2.set_title('AIC/BIC Comparison (Lower is Better)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['AIC', 'BIC'])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Panel 3: Tail dependence
    ax3 = axes[1, 0]
    tail_metrics = ['Lower\nTail', 'Upper\nTail', 't-Copula\nTheory']
    tail_values = [tail['lambda_L_empirical'], tail['lambda_U_empirical'], t_cop['lambda_theory']]
    colors_tail = ['#2ca02c', '#d62728', '#9467bd']
    bars = ax3.bar(tail_metrics, tail_values, color=colors_tail, alpha=0.7, edgecolor='black')
    ax3.axhline(0, color='gray', linestyle='--', linewidth=1, label='Gaussian (λ=0)')
    ax3.set_ylabel('Tail Dependence (λ)', fontsize=11, fontweight='bold')
    ax3.set_title('Tail Dependence Analysis', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, tail_values):
        ax3.text(bar.get_x() + bar.get_width()/2, val,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

    # Panel 4: Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary_text = (
        f"COPULA COMPARISON SUMMARY\n"
        f"{'='*40}\n\n"
        f"Gaussian Copula:\n"
        f"  ρ = {gauss['rho']:.4f}\n"
        f"  Log-likelihood = {gauss['loglik']:.2f}\n"
        f"  AIC = {gauss['aic']:.2f}\n"
        f"  BIC = {gauss['bic']:.2f}\n\n"
        f"Student-t Copula:\n"
        f"  ρ = {t_cop['rho']:.4f}\n"
        f"  ν = {t_cop['nu']:.2f}\n"
        f"  Log-likelihood = {t_cop['loglik']:.2f}\n"
        f"  AIC = {t_cop['aic']:.2f}\n"
        f"  BIC = {t_cop['bic']:.2f}\n\n"
        f"Likelihood Ratio Test:\n"
        f"  LR statistic = {comp['lr_stat']:.2f}\n"
        f"  p-value = {comp['lr_pvalue']:.4f}\n\n"
        f"Recommendation:\n"
        f"  ΔAIC = {comp['delta_aic']:.2f}\n"
        f"  ΔBIC = {comp['delta_bic']:.2f}\n"
    )

    # Recommendation
    if comp['delta_aic'] < -2:
        recommendation = "  → Use Student-t (ΔAIC < -2)"
    elif comp['delta_aic'] > 2:
        recommendation = "  → Use Gaussian (ΔAIC > 2)"
    else:
        recommendation = "  → Inconclusive (|ΔAIC| < 2)"

    summary_text += recommendation + "\n"
    summary_text += f"\nKendall's τ = {tail['kendall_tau']:.4f}"

    ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle(
        f"Copula Comparison: {dataset_id} (SSI-{ssi_window})",
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()

    fname = f"{output_dir}/copula_comparison_ssi{ssi_window}.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close()

    return fname


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def analyze_distributions_for_ssi_window(dataset_id, ssi_window, output_dir):
    """
    Complete distribution analysis for one SSI window.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    ssi_window : int
        SSI window size
    output_dir : str
        Output directory

    Returns
    -------
    dict
        Complete analysis results
    """
    print(f"\n{'='*60}")
    print(f"Analyzing SSI-{ssi_window}")
    print(f"{'='*60}")

    # Load data
    print("Loading drought events...")
    df = load_drought_events(dataset_id, ssi_window)
    print(f"  Loaded {len(df):,} events")

    # Test all candidate distributions for severity
    print("\nTesting severity marginal distributions...")
    severity_fits = []
    for name, dist in SEVERITY_CANDIDATES.items():
        print(f"  Testing {name}...")
        fit = fit_and_test_distribution(df['severity'].values, dist, name)
        severity_fits.append(fit)

    # Test all candidate distributions for magnitude
    print("\nTesting magnitude marginal distributions...")
    magnitude_fits = []
    for name, dist_or_func in MAGNITUDE_CANDIDATES.items():
        print(f"  Testing {name}...")
        if callable(dist_or_func) and name == 'truncnorm_0':
            # Handle truncnorm specially
            fit = fit_and_test_distribution(df['magnitude'].values,
                                           stats.truncnorm, name)
        elif callable(dist_or_func):
            dist = dist_or_func()
            fit = fit_and_test_distribution(df['magnitude'].values, dist, name)
        else:
            fit = fit_and_test_distribution(df['magnitude'].values, dist_or_func, name)
        magnitude_fits.append(fit)

    # Select best distributions (lowest AIC)
    best_severity = min([f for f in severity_fits if f['params'] is not None],
                       key=lambda x: x['aic'])
    best_magnitude = min([f for f in magnitude_fits if f['params'] is not None],
                         key=lambda x: x['aic'])

    print(f"\nBest severity distribution: {best_severity['dist_name']} "
          f"(AIC={best_severity['aic']:.2f})")
    print(f"Best magnitude distribution: {best_magnitude['dist_name']} "
          f"(AIC={best_magnitude['aic']:.2f})")

    # Transform to uniform
    print("\nTransforming to uniform marginals...")
    eps = 1e-12

    if best_severity['dist_name'] == 'truncnorm_0':
        u_severity = best_severity['fitted_dist'].cdf(df['severity'].values)
    else:
        u_severity = best_severity['fitted_dist'].cdf(
            df['severity'].values, *best_severity['params']
        )

    if best_magnitude['dist_name'] == 'truncnorm_0':
        u_magnitude = best_magnitude['fitted_dist'].cdf(df['magnitude'].values)
    else:
        u_magnitude = best_magnitude['fitted_dist'].cdf(
            df['magnitude'].values, *best_magnitude['params']
        )

    u_severity = np.clip(u_severity, eps, 1 - eps)
    u_magnitude = np.clip(u_magnitude, eps, 1 - eps)
    U = np.column_stack([u_severity, u_magnitude])

    # Test uniformity
    print("\nTesting uniformity of transformed data...")
    uniformity_severity = test_uniformity(u_severity)
    uniformity_magnitude = test_uniformity(u_magnitude)

    print(f"  Severity: KS p={uniformity_severity['ks_pvalue']:.4f}")
    print(f"  Magnitude: KS p={uniformity_magnitude['ks_pvalue']:.4f}")

    # Compare copulas
    print("\nComparing Gaussian vs. Student-t copula...")
    copula_results = fit_and_compare_copulas(U, df)

    print(f"  Gaussian AIC: {copula_results['gaussian']['aic']:.2f}")
    print(f"  Student-t AIC: {copula_results['t_copula']['aic']:.2f}")
    print(f"  ΔAIC: {copula_results['comparison']['delta_aic']:.2f}")

    # Generate plots
    print("\nGenerating diagnostic plots...")

    plot_marginal_comparison(df['severity'].values, severity_fits,
                            'severity', ssi_window, dataset_id, output_dir)

    plot_marginal_comparison(df['magnitude'].values, magnitude_fits,
                            'magnitude', ssi_window, dataset_id, output_dir)

    plot_uniformity_diagnostics(
        U,
        {'severity': uniformity_severity, 'magnitude': uniformity_magnitude},
        {'severity': best_severity, 'magnitude': best_magnitude},
        ssi_window, dataset_id, output_dir
    )

    plot_copula_comparison(copula_results, ssi_window, dataset_id, output_dir)

    return {
        'severity_fits': severity_fits,
        'magnitude_fits': magnitude_fits,
        'best_severity': best_severity,
        'best_magnitude': best_magnitude,
        'uniformity': {
            'severity': uniformity_severity,
            'magnitude': uniformity_magnitude,
        },
        'copula': copula_results,
    }


def generate_summary_report(all_results, dataset_id, output_dir):
    """
    Generate text summary report of all analyses.

    Parameters
    ----------
    all_results : dict
        Results for all SSI windows
    dataset_id : str
        Dataset identifier
    output_dir : str
        Output directory
    """
    lines = []
    lines.append("=" * 80)
    lines.append(f"DISTRIBUTION TEST SUMMARY: {dataset_id}")
    lines.append("=" * 80)
    lines.append("")

    for ssi_window, results in sorted(all_results.items()):
        lines.append(f"\nSSI-{ssi_window} RESULTS")
        lines.append("-" * 80)

        # Best distributions
        lines.append("\nBEST MARGINAL DISTRIBUTIONS:")
        best_sev = results['best_severity']
        best_mag = results['best_magnitude']

        lines.append(f"\n  Severity: {best_sev['dist_name'].upper()}")
        lines.append(f"    AIC: {best_sev['aic']:.2f}")
        lines.append(f"    BIC: {best_sev['bic']:.2f}")
        lines.append(f"    KS statistic: {best_sev['ks_stat']:.4f}")
        lines.append(f"    KS p-value: {best_sev['ks_pvalue']:.4f}")

        lines.append(f"\n  Magnitude: {best_mag['dist_name'].upper()}")
        lines.append(f"    AIC: {best_mag['aic']:.2f}")
        lines.append(f"    BIC: {best_mag['bic']:.2f}")
        lines.append(f"    KS statistic: {best_mag['ks_stat']:.4f}")
        lines.append(f"    KS p-value: {best_mag['ks_pvalue']:.4f}")

        # Uniformity tests
        lines.append("\n\nUNIFORMITY TESTS (After Transformation):")
        unif_sev = results['uniformity']['severity']
        unif_mag = results['uniformity']['magnitude']

        lines.append(f"\n  Severity:")
        lines.append(f"    KS p-value: {unif_sev['ks_pvalue']:.4f}")
        lines.append(f"    Chi-square p-value: {unif_sev['chi2_pvalue']:.4f}")

        lines.append(f"\n  Magnitude:")
        lines.append(f"    KS p-value: {unif_mag['ks_pvalue']:.4f}")
        lines.append(f"    Chi-square p-value: {unif_mag['chi2_pvalue']:.4f}")

        # Copula comparison
        lines.append("\n\nCOPULA COMPARISON:")
        cop = results['copula']

        lines.append(f"\n  Gaussian Copula:")
        lines.append(f"    ρ: {cop['gaussian']['rho']:.4f}")
        lines.append(f"    Log-likelihood: {cop['gaussian']['loglik']:.2f}")
        lines.append(f"    AIC: {cop['gaussian']['aic']:.2f}")
        lines.append(f"    BIC: {cop['gaussian']['bic']:.2f}")

        lines.append(f"\n  Student-t Copula:")
        lines.append(f"    ρ: {cop['t_copula']['rho']:.4f}")
        lines.append(f"    ν: {cop['t_copula']['nu']:.2f}")
        lines.append(f"    Log-likelihood: {cop['t_copula']['loglik']:.2f}")
        lines.append(f"    AIC: {cop['t_copula']['aic']:.2f}")
        lines.append(f"    BIC: {cop['t_copula']['bic']:.2f}")

        lines.append(f"\n  Comparison:")
        lines.append(f"    ΔAIC (t - Gaussian): {cop['comparison']['delta_aic']:.2f}")
        lines.append(f"    ΔBIC (t - Gaussian): {cop['comparison']['delta_bic']:.2f}")
        lines.append(f"    LR test p-value: {cop['comparison']['lr_pvalue']:.4f}")

        # Recommendation
        delta_aic = cop['comparison']['delta_aic']
        if delta_aic < -2:
            rec = "RECOMMEND: Student-t copula (ΔAIC < -2)"
        elif delta_aic > 2:
            rec = "RECOMMEND: Gaussian copula (ΔAIC > 2)"
        else:
            rec = "INCONCLUSIVE: Models similar (|ΔAIC| < 2)"

        lines.append(f"\n  → {rec}")

        # Tail dependence
        lines.append(f"\n\nTAIL DEPENDENCE:")
        tail = cop['tail_dependence']
        lines.append(f"    Kendall's τ: {tail['kendall_tau']:.4f}")
        lines.append(f"    Empirical λ_L: {tail['lambda_L_empirical']:.4f}")
        lines.append(f"    Empirical λ_U: {tail['lambda_U_empirical']:.4f}")
        lines.append(f"    t-Copula λ (theory): {cop['t_copula']['lambda_theory']:.4f}")

        lines.append("\n" + "=" * 80)

    # Overall recommendations
    lines.append("\n\nOVERALL RECOMMENDATIONS")
    lines.append("=" * 80)

    for ssi_window, results in sorted(all_results.items()):
        best_sev = results['best_severity']['dist_name']
        best_mag = results['best_magnitude']['dist_name']
        delta_aic = results['copula']['comparison']['delta_aic']

        if delta_aic < -2:
            copula_rec = "Student-t"
        elif delta_aic > 2:
            copula_rec = "Gaussian"
        else:
            copula_rec = "Either (similar)"

        lines.append(f"\nSSI-{ssi_window}:")
        lines.append(f"  Severity: {best_sev}")
        lines.append(f"  Magnitude: {best_mag}")
        lines.append(f"  Copula: {copula_rec}")

    lines.append("\n" + "=" * 80)

    # Write to file
    summary_text = "\n".join(lines)
    fname = f"{output_dir}/distribution_test_summary.txt"
    with open(fname, 'w') as f:
        f.write(summary_text)

    print(f"\n\nSaved summary: {fname}")
    print(summary_text)


# =============================================================================
# MAIN
# =============================================================================

def main(dataset_id='stationary_ensemble'):
    """
    Main function to run complete distribution analysis.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    """
    print("=" * 80)
    print("COPULA DISTRIBUTION TESTING")
    print("=" * 80)
    print(f"Dataset: {dataset_id}")
    print(f"SSI windows: {SSI_WINDOWS}")
    print("=" * 80)

    # Verify dataset
    verify_dataset_id(dataset_id)

    # Create output directory
    output_dir = f"{FIG_DIR}/copula_diagnostics/distribution_tests/{dataset_id}"
    os.makedirs(output_dir, exist_ok=True)

    # Run analysis for each SSI window
    all_results = {}

    for ssi_window in SSI_WINDOWS:
        try:
            results = analyze_distributions_for_ssi_window(
                dataset_id, ssi_window, output_dir
            )
            all_results[ssi_window] = results
        except FileNotFoundError as e:
            print(f"\nWARNING: {e}")
            print(f"Skipping SSI-{ssi_window}...")
            continue
        except Exception as e:
            print(f"\nERROR in SSI-{ssi_window}: {e}")
            import traceback
            traceback.print_exc()
            print(f"Skipping SSI-{ssi_window}...")
            continue

    # Generate summary report
    if all_results:
        generate_summary_report(all_results, dataset_id, output_dir)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        print("Usage: python 07_test_copula_distributions.py [dataset_id]")
        print(f"Available datasets: {list(DATASET_CONFIGS.keys())}")
        print("Default: stationary_ensemble")
        sys.exit(1)

    dataset_id = sys.argv[1] if len(sys.argv) == 2 else 'stationary_ensemble'
    verify_dataset_id(dataset_id)

    main(dataset_id)
