"""
Statistical analysis for episode-level vulnerability assessment.

This module provides functions for comparing episode populations,
fitting predictive models, and computing confidence intervals.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import scipy.stats as stats


def compare_episode_populations(
    episodes: pd.DataFrame,
    group_col: str,
    feature_cols: List[str],
    group_a: str = 'cascade',
    group_b: str = 'contained'
) -> pd.DataFrame:
    """
    Compare feature distributions across episode groups.

    For each feature, computes descriptive statistics, performs
    Mann-Whitney U test, and calculates effect size (Cohen's d).

    Parameters
    ----------
    episodes : pd.DataFrame
        Episodes with features and classification
    group_col : str
        Column containing group labels (e.g., 'cascade_classification')
    feature_cols : List[str]
        Features to compare
    group_a : str
        First group label (typically 'cascade')
    group_b : str
        Second group label (typically 'contained')

    Returns
    -------
    results : pd.DataFrame
        Comparison results with statistics for each feature
    """
    results = []

    group_a_data = episodes[episodes[group_col] == group_a]
    group_b_data = episodes[episodes[group_col] == group_b]

    p_values = []

    for feat in feature_cols:
        if feat not in episodes.columns:
            continue

        a = group_a_data[feat].dropna()
        b = group_b_data[feat].dropna()

        if len(a) < 2 or len(b) < 2:
            continue

        # Descriptive stats
        a_mean, a_std = a.mean(), a.std()
        b_mean, b_std = b.mean(), b.std()

        # Mann-Whitney U test (non-parametric)
        try:
            stat, p = stats.mannwhitneyu(a, b, alternative='two-sided')
        except ValueError:
            stat, p = np.nan, np.nan

        p_values.append(p)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(a) - 1) * a_std**2 + (len(b) - 1) * b_std**2) /
            (len(a) + len(b) - 2)
        )
        if pooled_std > 0:
            cohens_d = (a_mean - b_mean) / pooled_std
        else:
            cohens_d = np.nan

        results.append({
            'feature': feat,
            f'{group_a}_n': len(a),
            f'{group_a}_mean': a_mean,
            f'{group_a}_std': a_std,
            f'{group_b}_n': len(b),
            f'{group_b}_mean': b_mean,
            f'{group_b}_std': b_std,
            'test_statistic': stat,
            'p_value': p,
            'effect_size_d': cohens_d
        })

    results_df = pd.DataFrame(results)

    # Multiple testing correction (Benjamini-Hochberg)
    if len(results_df) > 0 and 'p_value' in results_df.columns:
        valid_p = results_df['p_value'].dropna()
        if len(valid_p) > 0:
            try:
                from statsmodels.stats.multitest import multipletests
                _, p_corrected, _, _ = multipletests(
                    results_df['p_value'].fillna(1.0),
                    method='fdr_bh'
                )
                results_df['p_value_corrected'] = p_corrected
            except ImportError:
                # If statsmodels not available, skip correction
                results_df['p_value_corrected'] = results_df['p_value']

    return results_df


def fit_cascade_model(
    episodes: pd.DataFrame,
    feature_cols: List[str],
    outcome_col: str = 'cascade_classification'
) -> Dict:
    """
    Fit logistic regression model for cascade probability.

    Model: P(cascade | stress episode) ~ features

    Parameters
    ----------
    episodes : pd.DataFrame
        Episodes with features
    feature_cols : List[str]
        Predictor features
    outcome_col : str
        Column containing outcome classification

    Returns
    -------
    model_results : dict
        Dictionary containing:
        - model: fitted statsmodels result object
        - coefficients: DataFrame with estimates, SEs, p-values, odds ratios
        - model_fit_stats: dict with AIC, BIC, pseudo-R2
        - scaler: StandardScaler used for feature scaling
    """
    try:
        from sklearn.preprocessing import StandardScaler
        import statsmodels.api as sm
    except ImportError:
        raise ImportError(
            "statsmodels and sklearn required for cascade model fitting. "
            "Install with: pip install statsmodels scikit-learn"
        )

    # Filter to stress episodes only
    stress_types = ['E1', 'E1d', 'E1c']
    stress_eps = episodes[episodes['episode_type'].isin(stress_types)].copy()

    if len(stress_eps) == 0:
        return {'error': 'No stress episodes found'}

    # Binary outcome: cascade vs. not cascade
    stress_eps['is_cascade'] = (stress_eps[outcome_col] == 'cascade').astype(int)

    # Check if there are both classes
    if stress_eps['is_cascade'].nunique() < 2:
        return {'error': 'Need both cascade and non-cascade episodes to fit model'}

    # Prepare features - only use columns that exist and have data
    available_features = [f for f in feature_cols if f in stress_eps.columns]
    if len(available_features) == 0:
        return {'error': 'No valid features found'}

    # Drop rows with missing values in features or outcome
    model_data = stress_eps[available_features + ['is_cascade']].dropna()

    if len(model_data) < 10:
        return {'error': f'Insufficient data after dropping NA ({len(model_data)} rows)'}

    X = model_data[available_features]
    y = model_data['is_cascade']

    # Standardize features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=available_features,
        index=X.index
    )
    X_scaled = sm.add_constant(X_scaled)

    # Fit logistic regression
    try:
        model = sm.Logit(y, X_scaled)
        result = model.fit(disp=False, maxiter=100)
    except Exception as e:
        return {'error': f'Model fitting failed: {str(e)}'}

    # Extract coefficients
    coef_df = pd.DataFrame({
        'feature': result.params.index,
        'coefficient': result.params.values,
        'std_error': result.bse.values,
        'z_stat': result.tvalues.values,
        'p_value': result.pvalues.values,
        'odds_ratio': np.exp(result.params.values)
    })

    return {
        'model': result,
        'coefficients': coef_df,
        'scaler': scaler,
        'features_used': available_features,
        'model_fit_stats': {
            'aic': result.aic,
            'bic': result.bic,
            'pseudo_r2': result.prsquared,
            'n_observations': len(y),
            'n_cascade': y.sum(),
            'n_non_cascade': len(y) - y.sum()
        }
    }


def compute_proportion_ci(
    n_success: int,
    n_total: int,
    confidence: float = 0.95,
    method: str = 'wilson'
) -> tuple:
    """
    Compute confidence interval for a proportion.

    Parameters
    ----------
    n_success : int
        Number of successes
    n_total : int
        Total number of observations
    confidence : float
        Confidence level (default 0.95)
    method : str
        Method for CI calculation ('wilson' or 'normal')

    Returns
    -------
    ci_low, ci_high : tuple of float
        Lower and upper confidence interval bounds
    """
    if n_total == 0:
        return (np.nan, np.nan)

    p = n_success / n_total
    alpha = 1 - confidence

    if method == 'wilson':
        # Wilson score interval
        z = stats.norm.ppf(1 - alpha / 2)
        denominator = 1 + z**2 / n_total
        center = (p + z**2 / (2 * n_total)) / denominator
        margin = z * np.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2)) / denominator
        ci_low = max(0, center - margin)
        ci_high = min(1, center + margin)
    else:
        # Normal approximation
        se = np.sqrt(p * (1 - p) / n_total)
        z = stats.norm.ppf(1 - alpha / 2)
        ci_low = max(0, p - z * se)
        ci_high = min(1, p + z * se)

    return (ci_low, ci_high)


def compute_cascade_rates_by_realization(
    episodes: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute cascade rates for each realization.

    Parameters
    ----------
    episodes : pd.DataFrame
        Episodes with cascade_classification

    Returns
    -------
    rates : pd.DataFrame
        Cascade rates by realization
    """
    stress_types = ['E1', 'E1d', 'E1c']
    stress_eps = episodes[episodes['episode_type'].isin(stress_types)]

    results = []
    for r in stress_eps['realization_id'].unique():
        r_eps = stress_eps[stress_eps['realization_id'] == r]
        n_total = len(r_eps)
        n_cascade = (r_eps['cascade_classification'] == 'cascade').sum()
        n_partial = r_eps['cascade_classification'].isin(['partial_demand', 'partial_flow']).sum()
        n_contained = (r_eps['cascade_classification'] == 'contained').sum()

        results.append({
            'realization_id': r,
            'n_stress_episodes': n_total,
            'n_cascade': n_cascade,
            'n_partial': n_partial,
            'n_contained': n_contained,
            'cascade_rate': n_cascade / n_total if n_total > 0 else np.nan,
            'partial_rate': n_partial / n_total if n_total > 0 else np.nan,
            'contained_rate': n_contained / n_total if n_total > 0 else np.nan,
        })

    return pd.DataFrame(results)


def get_discriminating_features(
    comparison_results: pd.DataFrame,
    p_threshold: float = 0.05,
    effect_threshold: float = 0.2
) -> List[str]:
    """
    Get features that significantly discriminate between groups.

    Parameters
    ----------
    comparison_results : pd.DataFrame
        Results from compare_episode_populations()
    p_threshold : float
        Maximum p-value (corrected) for significance
    effect_threshold : float
        Minimum absolute effect size

    Returns
    -------
    features : List[str]
        List of discriminating feature names
    """
    if 'p_value_corrected' not in comparison_results.columns:
        p_col = 'p_value'
    else:
        p_col = 'p_value_corrected'

    mask = (
        (comparison_results[p_col] < p_threshold) &
        (comparison_results['effect_size_d'].abs() >= effect_threshold)
    )

    return comparison_results.loc[mask, 'feature'].tolist()
