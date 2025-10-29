"""
Plotting functions for copula diagnostics.

This module provides visualization functions for validating copula fits
and analyzing tail dependence in drought severity-magnitude relationships.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import t


def plot_joint_scatter(df, ssi_window, dataset_id, output_dir):
    """
    Plot joint scatter of severity and magnitude with marginal distributions.

    Parameters
    ----------
    df : pd.DataFrame
        Drought events with 'severity' and 'magnitude' columns
    ssi_window : int
        SSI window size
    dataset_id : str
        Dataset identifier
    output_dir : str
        Output directory for figures

    Returns
    -------
    str
        Path to saved figure
    """
    g = sns.jointplot(
        data=df,
        x='severity',
        y='magnitude',
        kind="scatter",
        height=8,
        marginal_kws=dict(bins=50, fill=True),
        alpha=0.3,
        s=20
    )

    g.set_axis_labels('Severity (log)', 'Magnitude (log)')
    g.figure.suptitle(
        f'Joint Distribution: {dataset_id} (SSI-{ssi_window})',
        y=1.02, fontsize=14, fontweight='bold'
    )

    fname = f"{output_dir}/01_joint_scatter_{dataset_id}_ssi{ssi_window}.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close()

    return fname


def plot_marginal_fits(df, marginals, ssi_window, dataset_id, output_dir):
    """
    Plot fitted marginal distributions vs empirical histograms.

    Parameters
    ----------
    df : pd.DataFrame
        Drought events with 'severity' and 'magnitude' columns
    marginals : dict
        Fitted marginal distributions
    ssi_window : int
        SSI window size
    dataset_id : str
        Dataset identifier
    output_dir : str
        Output directory for figures

    Returns
    -------
    str
        Path to saved figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Severity marginal
    x_sev = np.linspace(0, df['severity'].max(), 1000)
    ax1.hist(df['severity'], bins=200, density=True, alpha=0.7,
            color='skyblue', label='Empirical')
    # Get distribution name (handle truncnorm_0 special case)
    sev_dist_name = marginals['severity_dist'].name if hasattr(marginals['severity_dist'], 'name') else 'truncnorm'
    ax1.plot(x_sev,
            marginals['severity_dist'].pdf(x_sev),
            'r-', lw=2, label=f"{sev_dist_name.title()}")
    ax1.set_xlabel('Severity (log)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax1.set_title('Severity Marginal Distribution', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)

    # Magnitude marginal
    x_mag = np.linspace(0, df['magnitude'].max(), 1000)
    ax2.hist(df['magnitude'], bins=100, density=True, alpha=0.7,
            color='lightcoral', label='Empirical')
    # Get distribution name (handle truncnorm_0 special case)
    mag_dist_name = marginals['magnitude_dist'].name if hasattr(marginals['magnitude_dist'], 'name') else 'truncnorm'
    ax2.plot(x_mag,
            marginals['magnitude_dist'].pdf(x_mag),
            'r-', lw=2, label=f"{mag_dist_name.title()}")
    ax2.set_xlabel('Magnitude (log)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax2.set_title('Magnitude Marginal Distribution', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f'{dataset_id} - SSI-{ssi_window} Marginal Fits',
                fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()

    fname = f"{output_dir}/02_marginal_fits_{dataset_id}_ssi{ssi_window}.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close()

    return fname


def plot_empirical_copula(U, ssi_window, dataset_id, output_dir):
    """
    Plot empirical copula (uniform marginals) with histograms.

    Parameters
    ----------
    U : np.ndarray
        (n, 2) array of uniform marginals
    ssi_window : int
        SSI window size
    dataset_id : str
        Dataset identifier
    output_dir : str
        Output directory for figures

    Returns
    -------
    str
        Path to saved figure
    """
    g = sns.jointplot(
        x=U[:, 0],
        y=U[:, 1],
        kind='scatter',
        alpha=0.1,
        s=20,
        height=8,
        marginal_kws={'bins': 50, 'alpha': 0.7}
    )

    g.set_axis_labels('U1 (Severity uniform)', 'U2 (Magnitude uniform)')
    g.figure.suptitle(
        f'Empirical Copula: {dataset_id} (SSI-{ssi_window})',
        y=1.02, fontsize=14, fontweight='bold'
    )

    # Add reference lines for independence
    g.ax_joint.axhline(y=0.5, color='red', linestyle='--', alpha=0.5,
                      label='Independence')
    g.ax_joint.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)

    fname = f"{output_dir}/03_empirical_copula_{dataset_id}_ssi{ssi_window}.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close()

    return fname


def plot_qq_plots(df, marginals, ssi_window, dataset_id, output_dir):
    """
    Plot Q-Q plots for marginal distributions.

    Parameters
    ----------
    df : pd.DataFrame
        Drought events with 'severity' and 'magnitude' columns
    marginals : dict
        Fitted marginal distributions
    ssi_window : int
        SSI window size
    dataset_id : str
        Dataset identifier
    output_dir : str
        Output directory for figures

    Returns
    -------
    str
        Path to saved figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Severity Q-Q plot
    stats.probplot(
        df['severity'],
        dist=marginals['severity_dist'],
        sparams=marginals['severity_params'],
        plot=ax1
    )
    ax1.set_title('Severity Q-Q Plot', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Magnitude Q-Q plot
    stats.probplot(
        df['magnitude'],
        dist=marginals['magnitude_dist'],
        sparams=marginals['magnitude_params'],
        plot=ax2
    )
    ax2.set_title('Magnitude Q-Q Plot', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f'{dataset_id} - SSI-{ssi_window} Q-Q Plots',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    fname = f"{output_dir}/04a_qq_plots_{dataset_id}_ssi{ssi_window}.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close()

    return fname


def plot_copula_comparison(U, copula, ssi_window, dataset_id, output_dir):
    """
    Plot empirical vs simulated copula data.

    Parameters
    ----------
    U : np.ndarray
        (n, 2) array of empirical uniform marginals
    copula : dict
        Fitted copula parameters
    ssi_window : int
        SSI window size
    dataset_id : str
        Dataset identifier
    output_dir : str
        Output directory for figures

    Returns
    -------
    str
        Path to saved figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Empirical copula data
    ax1.scatter(U[:, 0], U[:, 1], alpha=0.2, s=5, label='Empirical', color='blue')
    ax1.set_xlabel('U1 (Severity)', fontsize=11)
    ax1.set_ylabel('U2 (Magnitude)', fontsize=11)
    ax1.set_title('Empirical Copula Data', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Simulated copula data (using Gaussian copula)
    from methods.copula import simulate_copula
    n_sim = U.shape[0]
    rho = copula['rho']
    U_sim = simulate_copula(n_sim, copula_type='gaussian', rho=rho)

    ax2.scatter(U_sim[:, 0], U_sim[:, 1], alpha=0.2, s=5, label='Simulated', color='red')
    ax2.set_xlabel('U1 (Severity)', fontsize=11)
    ax2.set_ylabel('U2 (Magnitude)', fontsize=11)
    ax2.set_title(f'Simulated Copula Data (Gaussian, ρ={rho:.3f})',
                 fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f'{dataset_id} - SSI-{ssi_window} Copula Comparison',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    fname = f"{output_dir}/04b_copula_comparison_{dataset_id}_ssi{ssi_window}.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close()

    return fname


def plot_copula_diagnostics(df, marginals, copula, tail, ssi_window, dataset_id, output_dir):
    """
    Plot comprehensive copula diagnostics (Q-Q plots, simulated data comparison).

    Parameters
    ----------
    df : pd.DataFrame
        Drought events
    marginals : dict
        Fitted marginal distributions
    copula : dict
        Fitted copula parameters
    tail : dict
        Tail dependence diagnostics
    ssi_window : int
        SSI window size
    dataset_id : str
        Dataset identifier
    output_dir : str
        Output directory for figures

    Returns
    -------
    str
        Path to saved figure
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Severity Q-Q plot
    stats.probplot(
        df['severity'],
        dist=marginals['severity_dist'],
        sparams=marginals['severity_params'],
        plot=ax1
    )
    ax1.set_title('Severity Q-Q Plot', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Magnitude Q-Q plot
    stats.probplot(
        df['magnitude'],
        dist=marginals['magnitude_dist'],
        sparams=marginals['magnitude_params'],
        plot=ax2
    )
    ax2.set_title('Magnitude Q-Q Plot', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Empirical copula data
    U = copula['U']
    ax3.scatter(U[:, 0], U[:, 1], alpha=0.2, s=5, label='Empirical', color='blue')
    ax3.set_xlabel('U1 (Severity)', fontsize=11)
    ax3.set_ylabel('U2 (Magnitude)', fontsize=11)
    ax3.set_title('Empirical Copula Data', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Simulated copula data (using Gaussian copula)
    from methods.copula import simulate_copula
    n_sim = U.shape[0]
    rho = copula['rho']
    U_sim = simulate_copula(n_sim, copula_type='gaussian', rho=rho)

    ax4.scatter(U_sim[:, 0], U_sim[:, 1], alpha=0.2, s=5, label='Simulated', color='red')
    ax4.set_xlabel('U1 (Severity)', fontsize=11)
    ax4.set_ylabel('U2 (Magnitude)', fontsize=11)
    ax4.set_title(f'Simulated Copula Data (Gaussian, ρ={rho:.3f})',
                 fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    fig.suptitle(f'{dataset_id} - SSI-{ssi_window} Copula Diagnostics',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    fname = f"{output_dir}/04_copula_diagnostics_{dataset_id}_ssi{ssi_window}.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close()

    return fname


def plot_tail_dependence(U, tail, tail_curves, ssi_window, dataset_id, output_dir):
    """
    Plot tail dependence curves.

    Parameters
    ----------
    U : np.ndarray
        (n, 2) array of uniform marginals
    tail : dict
        Tail dependence diagnostics
    tail_curves : dict
        Tail dependence curves
    ssi_window : int
        SSI window size
    dataset_id : str
        Dataset identifier
    output_dir : str
        Output directory for figures

    Returns
    -------
    str
        Path to saved figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    q = tail_curves['q']
    lambda_U = tail_curves['lambda_U']
    lambda_L = tail_curves['lambda_L']

    ax.plot(q, lambda_U, label=r'$\hat{\lambda}_U(q)$ (Upper tail)',
           linewidth=2, color='#d62728')
    ax.plot(q, lambda_L, label=r'$\hat{\lambda}_L(1-q)$ (Lower tail)',
           linestyle='--', linewidth=2, color='#1f77b4')

    # Add theoretical tail dependence if t-copula was fitted
    if tail['has_tail_dependence'] and 't_copula_nu' in tail:
        rho = tail['t_copula_rho']
        nu = tail['t_copula_nu']
        lam_asym = 2 * t.cdf(-np.sqrt(((nu + 1) * (1 - rho)) / (1 + rho)), df=nu + 1)
        ax.axhline(lam_asym, color='gray', alpha=0.6, linewidth=2,
                  label=fr'$t$-copula asymptotic $\lambda={lam_asym:.3f}$')
    else:
        ax.axhline(0.0, color='gray', alpha=0.6, linewidth=2,
                  label='Gaussian copula (λ=0)')

    ax.set_xlabel('Quantile (q)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Empirical Tail Dependence', fontsize=12, fontweight='bold')
    ax.set_title(f'Tail Dependence: {dataset_id} (SSI-{ssi_window})',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    fname = f"{output_dir}/05_tail_dependence_{dataset_id}_ssi{ssi_window}.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close()

    return fname


def generate_diagnostics_summary(df, marginals, copula, tail, interarrival,
                                ssi_window, dataset_id, output_dir):
    """
    Generate text summary of copula diagnostics.

    Parameters
    ----------
    df : pd.DataFrame
        Drought events
    marginals : dict
        Fitted marginal distributions
    copula : dict
        Fitted copula parameters
    tail : dict
        Tail dependence diagnostics
    interarrival : float
        Expected interarrival time in years
    ssi_window : int
        SSI window size
    dataset_id : str
        Dataset identifier
    output_dir : str
        Output directory for summary

    Returns
    -------
    str
        Path to saved summary file
    """
    summary = []
    summary.append("=" * 80)
    summary.append(f"COPULA DIAGNOSTICS SUMMARY: {dataset_id} (SSI-{ssi_window})")
    summary.append("=" * 80)
    summary.append("")

    # Data summary
    summary.append("DATA SUMMARY:")
    summary.append(f"  Number of drought events: {len(df):,}")
    summary.append(f"  Expected interarrival time: {interarrival:.2f} years")
    summary.append("")

    # Marginal distributions
    summary.append("MARGINAL DISTRIBUTIONS:")
    summary.append(f"  Severity: {marginals['severity_dist'].name}")
    summary.append(f"    Parameters: {marginals['severity_params']}")
    summary.append(f"  Magnitude: {marginals['magnitude_dist'].name}")
    summary.append(f"    Parameters: {marginals['magnitude_params']}")
    summary.append("")

    # Copula
    summary.append("COPULA:")
    summary.append(f"  Type: Gaussian")
    summary.append(f"  Correlation (ρ): {copula['rho']:.4f}")
    summary.append(f"  Log-likelihood: {copula['loglik']:.2f}")
    summary.append("")

    # Tail dependence
    summary.append("TAIL DEPENDENCE:")
    summary.append(f"  Kendall's tau: {tail['tau']:.4f}")
    summary.append(f"  ρ from tau: {tail['rho_from_tau']:.4f}")
    summary.append(f"  ρ from Gaussian copula: {copula['rho']:.4f}")
    summary.append(f"  Difference: {copula['rho'] - tail['rho_from_tau']:.4f}")
    summary.append(f"  Empirical λ_L (q=0.05): {tail['lambda_L']:.3f}")
    summary.append(f"  Empirical λ_U (q=0.95): {tail['lambda_U']:.3f}")
    summary.append(f"  Tail dependence detected: {tail['has_tail_dependence']}")

    if tail['has_tail_dependence'] and 't_copula_nu' in tail:
        summary.append("")
        summary.append("  t-COPULA FIT (alternative):")
        summary.append(f"    Correlation (ρ): {tail['t_copula_rho']:.4f}")
        summary.append(f"    Degrees of freedom (ν): {tail['t_copula_nu']:.2f}")

    summary.append("")
    summary.append("=" * 80)

    # Print to console
    summary_text = "\n".join(summary)
    print(summary_text)

    # Save to file
    fname = f"{output_dir}/00_diagnostics_summary_{dataset_id}_ssi{ssi_window}.txt"
    with open(fname, 'w') as f:
        f.write(summary_text)
    print(f"  Saved summary: {fname}")

    return fname
