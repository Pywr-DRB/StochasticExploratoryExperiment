"""
Copula-based methods for drought severity-magnitude analysis.

This module provides functions for fitting and analyzing copulas for
joint probability modeling of drought severity and magnitude.

The methodology is used in:
- 06_ssi_copula_diagnostics.py
- 09_plot_drought_frequency.py
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import multivariate_normal, multivariate_t, norm, t
from scipy.optimize import minimize


def load_drought_events(dataset_id, ssi_window, drought_metrics_dir='./pywrdrb/drought_metrics'):
    """
    Load drought events for a dataset and SSI window.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    ssi_window : int
        SSI window size (months)
    drought_metrics_dir : str, optional
        Directory containing drought metrics files

    Returns
    -------
    pd.DataFrame
        Drought events with log-transformed severity and magnitude
    """
    import os
    fname = f"{drought_metrics_dir}/{dataset_id}_ssi{ssi_window}_drought_events.csv"

    if not os.path.exists(fname):
        raise FileNotFoundError(
            f"Drought events file not found: {fname}\n"
            f"Run 05_calculate_ssi_drought_metrics.py first!"
        )

    df = pd.read_csv(fname)

    # Remove infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['magnitude', 'severity'])

    # Log-transform severity and magnitude (these are positive values)
    df['severity'] = np.log(np.abs(df['severity']))
    df['magnitude'] = np.log(np.abs(df['magnitude']))

    return df


def get_marginal_distribution(metric_name, distribution_config=None):
    """
    Get the scipy distribution object for a drought metric.

    Parameters
    ----------
    metric_name : str
        Metric name ('severity' or 'magnitude')
    distribution_config : dict, optional
        Dictionary mapping metric names to distribution names
        If None, uses default from config

    Returns
    -------
    scipy.stats distribution
        Distribution object
    """
    if distribution_config is None:
        from methods.config import DROUGHT_MARGINAL_DISTRIBUTIONS
        distribution_config = DROUGHT_MARGINAL_DISTRIBUTIONS

    dist_name = distribution_config.get(metric_name)
    if dist_name is None:
        raise ValueError(f"No distribution specified for metric: {metric_name}")

    # Get distribution from scipy.stats
    if dist_name == 'genexpon':
        return stats.genexpon
    elif dist_name == 'truncnorm_0':
        # Return marker for special handling
        return 'truncnorm_0'
    elif dist_name == 'norm':
        return stats.norm
    elif dist_name == 'lognorm':
        return stats.lognorm
    elif dist_name == 'gamma':
        return stats.gamma
    elif dist_name == 'expon':
        return stats.expon
    elif dist_name == 'weibull_min':
        return stats.weibull_min
    else:
        # Try to get from scipy.stats by name
        try:
            return getattr(stats, dist_name)
        except AttributeError:
            raise ValueError(f"Unknown distribution: {dist_name}")


def fit_marginal_distributions(df, distribution_config=None):
    """
    Fit marginal distributions to severity and magnitude.

    Parameters
    ----------
    df : pd.DataFrame
        Drought events with 'severity' and 'magnitude' columns
    distribution_config : dict, optional
        Dictionary mapping metric names to distribution names
        If None, uses default from config

    Returns
    -------
    dict
        Dictionary with fitted distributions and parameters:
        {
            'severity_dist': distribution object,
            'severity_params': fitted parameters,
            'magnitude_dist': distribution object,
            'magnitude_params': fitted parameters
        }
    """
    # Fit severity
    severity_data = df['severity'].values
    severity_data = severity_data[np.isfinite(severity_data) & (severity_data > 0)]
    severity_dist_obj = get_marginal_distribution('severity', distribution_config)

    if severity_dist_obj == 'truncnorm_0':
        # Fit normal, then create truncated version
        mu, sigma = stats.norm.fit(severity_data)
        a = (0 - mu) / sigma
        b = np.inf
        severity_params = (a, b, mu, sigma)
        severity_dist = stats.truncnorm(a, b, loc=mu, scale=sigma)
    else:
        # Fit parameters and create frozen distribution
        severity_params = severity_dist_obj.fit(severity_data)
        severity_dist = severity_dist_obj(*severity_params)

    # Fit magnitude
    magnitude_data = df['magnitude'].values
    magnitude_data = magnitude_data[np.isfinite(magnitude_data) & (magnitude_data > 0)]
    magnitude_dist_obj = get_marginal_distribution('magnitude', distribution_config)

    if magnitude_dist_obj == 'truncnorm_0':
        # Fit normal, then create truncated version
        mu, sigma = stats.norm.fit(magnitude_data)
        a = (0 - mu) / sigma
        b = np.inf
        magnitude_params = (a, b, mu, sigma)
        magnitude_dist = stats.truncnorm(a, b, loc=mu, scale=sigma)
    else:
        # Fit parameters and create frozen distribution
        magnitude_params = magnitude_dist_obj.fit(magnitude_data)
        magnitude_dist = magnitude_dist_obj(*magnitude_params)

    return {
        'severity_dist': severity_dist,
        'severity_params': severity_params,
        'magnitude_dist': magnitude_dist,
        'magnitude_params': magnitude_params,
    }


def transform_to_uniform(df, marginals):
    """
    Transform data to uniform marginals using fitted distributions.

    Parameters
    ----------
    df : pd.DataFrame
        Drought events with 'severity' and 'magnitude' columns
    marginals : dict
        Dictionary with fitted marginal distributions

    Returns
    -------
    np.ndarray
        (n, 2) array of uniform marginals [U1, U2]
    """
    eps = 1e-12

    # Transform to uniform using fitted marginals (frozen distributions)
    u1 = marginals['severity_dist'].cdf(df['severity'].to_numpy(float))
    u2 = marginals['magnitude_dist'].cdf(df['magnitude'].to_numpy(float))

    # Clip to avoid numerical issues
    u1 = np.clip(u1, eps, 1 - eps)
    u2 = np.clip(u2, eps, 1 - eps)

    return np.column_stack([u1, u2])


def fit_gaussian_copula(df, marginals):
    """
    Fit Gaussian copula using normal-scores correlation.

    This matches the exact methodology in 09_plot_drought_frequency.py.

    Parameters
    ----------
    df : pd.DataFrame
        Drought events with 'severity' and 'magnitude' columns
    marginals : dict
        Dictionary with fitted marginal distributions

    Returns
    -------
    dict
        Dictionary with copula parameters and transformed data:
        {
            'rho': correlation parameter,
            'loglik': log-likelihood,
            'u1': uniform marginal for severity,
            'u2': uniform marginal for magnitude,
            'U': (n, 2) array of uniform marginals
        }
    """
    # Transform to uniform
    U = transform_to_uniform(df, marginals)
    u1, u2 = U[:, 0], U[:, 1]

    # Calculate correlation in normal scores (Gaussian copula parameter)
    z1 = stats.norm.ppf(u1)
    z2 = stats.norm.ppf(u2)
    rho = float(np.corrcoef(z1, z2)[0, 1])
    rho = float(np.clip(rho, -0.999, 0.999))

    # Calculate log-likelihood
    cov = np.array([[1.0, rho], [rho, 1.0]])
    loglik = multivariate_normal.logpdf(
        np.column_stack([z1, z2]),
        cov=cov
    ).sum()

    return {
        'rho': rho,
        'loglik': loglik,
        'u1': u1,
        'u2': u2,
        'U': U,
    }


def fit_t_copula(U, rho_init=None):
    """
    Fit t-copula using maximum likelihood estimation.

    Parameters
    ----------
    U : np.ndarray
        (n, 2) array of uniform marginals
    rho_init : float, optional
        Initial guess for correlation parameter
        If None, uses Gaussian copula estimate

    Returns
    -------
    dict
        Dictionary with t-copula parameters:
        {
            'rho': correlation parameter,
            'nu': degrees of freedom,
            'loglik': log-likelihood
        }
    """
    U_eps = np.clip(U, 1e-12, 1 - 1e-12)

    def nll_t(params):
        """Negative log-likelihood for t-copula."""
        a, b = params
        rho_t = np.tanh(a)
        nu_t = np.exp(b) + 2.0

        # Clip rho to ensure positive definite covariance
        rho_t = np.clip(rho_t, -0.999, 0.999)

        z = stats.t.ppf(U_eps, df=nu_t)

        # Create covariance matrix with regularization if needed
        cov_t = np.array([[1.0, rho_t], [rho_t, 1.0]], dtype=float)
        if np.linalg.det(cov_t) < 1e-10:
            cov_t += np.eye(2) * 1e-8

        # Create frozen distribution and compute log-likelihood
        mvt = stats.multivariate_t(loc=np.zeros(2), shape=cov_t, df=nu_t)
        ll2 = mvt.logpdf(z)
        ll1 = stats.t.logpdf(z[:, 0], df=nu_t) + stats.t.logpdf(z[:, 1], df=nu_t)
        return -(np.sum(ll2 - ll1))

    # Initial guess
    if rho_init is None:
        # Use Gaussian copula estimate
        z1 = norm.ppf(U[:, 0])
        z2 = norm.ppf(U[:, 1])
        rho_init = np.corrcoef(z1, z2)[0, 1]

    a0 = np.arctanh(np.clip(rho_init, -0.99, 0.99))
    b0 = np.log(10.0 - 2.0)

    # Optimize
    opt = minimize(nll_t, x0=np.array([a0, b0]), method='L-BFGS-B')

    rho_t = np.tanh(opt.x[0])
    nu_t = np.exp(opt.x[1]) + 2.0
    loglik = -opt.fun

    return {
        'rho': rho_t,
        'nu': nu_t,
        'loglik': loglik,
    }


def calculate_kendalls_tau(df):
    """
    Calculate Kendall's tau for severity and magnitude.

    Parameters
    ----------
    df : pd.DataFrame
        Drought events with 'severity' and 'magnitude' columns

    Returns
    -------
    dict
        Dictionary with:
        {
            'tau': Kendall's tau,
            'rho_from_tau': Gaussian copula correlation from tau
        }
    """
    tau, _ = stats.kendalltau(df['severity'], df['magnitude'])
    rho_from_tau = np.sin(0.5 * np.pi * tau)

    return {
        'tau': tau,
        'rho_from_tau': rho_from_tau,
    }


def calculate_empirical_tail_dependence(U, q_lower=0.05, q_upper=0.95):
    """
    Calculate empirical tail dependence coefficients.

    Parameters
    ----------
    U : np.ndarray
        (n, 2) array of uniform marginals
    q_lower : float, optional
        Lower quantile for tail dependence (default: 0.05)
    q_upper : float, optional
        Upper quantile for tail dependence (default: 0.95)

    Returns
    -------
    dict
        Dictionary with tail dependence coefficients:
        {
            'lambda_L': lower tail dependence,
            'lambda_U': upper tail dependence
        }
    """
    lambda_L = np.mean((U[:, 0] <= q_lower) & (U[:, 1] <= q_lower)) / q_lower
    lambda_U = np.mean((U[:, 0] >= q_upper) & (U[:, 1] >= q_upper)) / (1.0 - q_upper)

    return {
        'lambda_L': lambda_L,
        'lambda_U': lambda_U,
    }


def calculate_tail_dependence_curves(U, q_range=None):
    """
    Calculate empirical tail dependence as a function of quantile.

    Parameters
    ----------
    U : np.ndarray
        (n, 2) array of uniform marginals
    q_range : np.ndarray, optional
        Quantile range to evaluate (default: linspace(0.90, 0.999, 50))

    Returns
    -------
    dict
        Dictionary with tail dependence curves:
        {
            'q': quantile values,
            'lambda_U': upper tail dependence curve,
            'lambda_L': lower tail dependence curve
        }
    """
    eps = 1e-12
    Ue = np.clip(U, eps, 1 - eps)

    if q_range is None:
        q_range = np.linspace(0.90, 0.999, 50)

    p = 1.0 - q_range

    # Upper tail: P(U1>q, U2>q)/(1-q)
    joint_upper = ((Ue[:, 0, None] > q_range) & (Ue[:, 1, None] > q_range)).mean(axis=0)
    lambda_U = joint_upper / p

    # Lower tail: P(U1<=p, U2<=p)/p with p = 1-q
    joint_lower = ((Ue[:, 0, None] <= p) & (Ue[:, 1, None] <= p)).mean(axis=0)
    lambda_L = joint_lower / p

    return {
        'q': q_range,
        'lambda_U': lambda_U,
        'lambda_L': lambda_L,
    }


def check_tail_dependence(U, df, threshold_L=0.02, threshold_U=0.02, fit_t_if_detected=True):
    """
    Check for tail dependence and optionally fit t-copula if detected.

    Parameters
    ----------
    U : np.ndarray
        (n, 2) array of uniform marginals
    df : pd.DataFrame
        Original drought data for Kendall's tau calculation
    threshold_L : float, optional
        Lower tail dependence threshold (default: 0.02)
    threshold_U : float, optional
        Upper tail dependence threshold (default: 0.02)
    fit_t_if_detected : bool, optional
        If True, fit t-copula when tail dependence is detected (default: True)

    Returns
    -------
    dict
        Dictionary with tail dependence diagnostics
    """
    # Calculate Kendall's tau
    tau_result = calculate_kendalls_tau(df)

    # Empirical tail dependence at fixed quantiles
    tail_dep = calculate_empirical_tail_dependence(U, q_lower=0.05, q_upper=0.95)

    # Check if tail dependence suggests t-copula
    has_tail_dependence = (
        (tail_dep['lambda_L'] > threshold_L) or
        (tail_dep['lambda_U'] > threshold_U)
    )

    result = {
        'tau': tau_result['tau'],
        'rho_from_tau': tau_result['rho_from_tau'],
        'lambda_L': tail_dep['lambda_L'],
        'lambda_U': tail_dep['lambda_U'],
        'has_tail_dependence': has_tail_dependence,
    }

    # If tail dependence detected, fit t-copula
    if has_tail_dependence and fit_t_if_detected:
        t_result = fit_t_copula(U, rho_init=tau_result['rho_from_tau'])
        result['t_copula_rho'] = t_result['rho']
        result['t_copula_nu'] = t_result['nu']
        result['t_copula_loglik'] = t_result['loglik']

    return result


def calculate_interarrival_time(df, n_years=70):
    """
    Calculate expected interarrival time between drought events.

    Parameters
    ----------
    df : pd.DataFrame
        Drought events with 'start', 'end', and 'realization_id'
    n_years : int, optional
        Number of years in simulation (default: 70)

    Returns
    -------
    float
        Expected interarrival time in years
    """
    df = df.copy()
    df['start'] = pd.to_datetime(df['start'])
    df['end'] = pd.to_datetime(df['end'])

    df_sorted = df.sort_values(['realization_id', 'start'])
    starts = df_sorted.groupby('realization_id')['start'].apply(
        lambda s: s.sort_values().diff().dt.days.dropna()
    ).to_numpy()

    if starts.size == 0:
        # Fallback: average years per event using counts
        counts = df_sorted.groupby('realization_id').size().to_numpy()
        counts = counts[counts > 0]
        if counts.size == 0:
            return np.nan
        E_L_years = float(np.mean(n_years / counts))
    else:
        E_L_years = float(np.mean(starts) / 365.25)

    return E_L_years


def simulate_copula(n_samples, copula_type='gaussian', rho=0.5, nu=None):
    """
    Simulate samples from a fitted copula.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    copula_type : str, optional
        Copula type: 'gaussian' or 't' (default: 'gaussian')
    rho : float, optional
        Correlation parameter (default: 0.5)
    nu : float, optional
        Degrees of freedom for t-copula (required if copula_type='t')

    Returns
    -------
    np.ndarray
        (n_samples, 2) array of uniform marginals
    """
    if copula_type == 'gaussian':
        z1, z2 = np.random.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n_samples).T
        u1, u2 = stats.norm.cdf(z1), stats.norm.cdf(z2)
    elif copula_type == 't':
        if nu is None:
            raise ValueError("Degrees of freedom 'nu' required for t-copula")
        # Generate from multivariate t
        mean = np.zeros(2)
        cov = np.array([[1.0, rho], [rho, 1.0]])
        samples = multivariate_t.rvs(loc=mean, shape=cov, df=nu, size=n_samples)
        u1, u2 = stats.t.cdf(samples[:, 0], df=nu), stats.t.cdf(samples[:, 1], df=nu)
    else:
        raise ValueError(f"Unknown copula type: {copula_type}")

    return np.column_stack([u1, u2])
