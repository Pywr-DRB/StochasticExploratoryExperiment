"""
CART-based bin edge selection for Sankey-Parallel Coordinate figure.

Uses Classification and Regression Trees (CART) to identify data-driven
bin edges that best separate satisficing from non-satisficing drought events.

For each metric, a shallow decision tree (depth 1-3) is fit to predict
joint satisficing status. The split thresholds from the tree become bin
edges in the Sankey figure, yielding bins that are directly interpretable
as "this threshold separates satisficing from non-satisficing behavior."

Handles class imbalance via class_weight='balanced' and uses bootstrap
stability analysis to flag unreliable thresholds.

References
----------
Breiman et al. (1984) Classification and Regression Trees.
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple


@dataclass
class CARTBinResult:
    """Result of CART bin edge computation for a single metric."""
    metric: str
    thresholds: List[float]          # CART-derived split points (sorted)
    balanced_accuracy: float         # Training balanced accuracy
    bootstrap_cv: float              # Coefficient of variation from bootstrap
    stability: str                   # 'stable', 'moderate', 'unstable'
    n_bins: int                      # Number of bins (len(thresholds) + 1)
    bin_edges: List[float]           # Full bin edges including data min/max
    bin_labels: Optional[List[str]]  # Auto-generated labels


def fit_univariate_cart(X_col, y, max_depth=2, min_samples_leaf_frac=0.05,
                         random_state=42):
    """
    Fit a shallow CART on a single feature to find split thresholds.

    Parameters
    ----------
    X_col : np.ndarray, shape (n,)
        Single feature values.
    y : np.ndarray, shape (n,)
        Binary target (1=satisficing, 0=failure).
    max_depth : int
        Maximum tree depth (1 = single stump, 2 = up to 3 bins, etc.)
    min_samples_leaf_frac : float
        Minimum leaf size as fraction of data.
    random_state : int

    Returns
    -------
    thresholds : list of float
        Sorted split thresholds.
    balanced_accuracy : float
        Balanced accuracy on training data.
    tree : DecisionTreeClassifier
        The fitted tree.
    """
    from sklearn.metrics import balanced_accuracy_score

    X = X_col.reshape(-1, 1) if X_col.ndim == 1 else X_col
    min_leaf = max(3, int(min_samples_leaf_frac * len(y)))

    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_leaf,
        class_weight='balanced',
        random_state=random_state,
    )
    tree.fit(X, y)

    y_pred = tree.predict(X)
    ba = balanced_accuracy_score(y, y_pred)

    # Extract thresholds
    thresholds = []
    tree_ = tree.tree_
    for i in range(tree_.node_count):
        if tree_.feature[i] != -2:  # Not a leaf
            thresholds.append(tree_.threshold[i])

    return sorted(set(thresholds)), ba, tree


def bootstrap_threshold_stability(X_col, y, n_bootstrap=200,
                                    random_state=42):
    """
    Assess stability of the depth-1 split threshold via stratified bootstrap.

    Parameters
    ----------
    X_col : np.ndarray, shape (n,)
        Single feature values.
    y : np.ndarray, shape (n,)
        Binary target.
    n_bootstrap : int
        Number of bootstrap resamples.
    random_state : int

    Returns
    -------
    mean_threshold : float
    std_threshold : float
    cv : float
        Coefficient of variation (std/|mean|). Lower is more stable.
    """
    rng = np.random.RandomState(random_state)
    thresholds = []

    classes = np.unique(y)
    for b in range(n_bootstrap):
        # Stratified bootstrap
        indices = []
        for cls in classes:
            cls_idx = np.where(y == cls)[0]
            boot_idx = rng.choice(cls_idx, size=len(cls_idx), replace=True)
            indices.extend(boot_idx)
        indices = np.array(indices)

        X_b = X_col[indices].reshape(-1, 1)
        y_b = y[indices]

        tree = DecisionTreeClassifier(
            max_depth=1,
            min_samples_leaf=max(2, int(0.03 * len(y_b))),
            class_weight='balanced',
            random_state=b,
        )
        tree.fit(X_b, y_b)

        # Extract threshold
        tree_ = tree.tree_
        for i in range(tree_.node_count):
            if tree_.feature[i] != -2:
                thresholds.append(tree_.threshold[i])
                break

    if not thresholds:
        return np.nan, np.nan, np.inf

    arr = np.array(thresholds)
    mean_t = arr.mean()
    std_t = arr.std()
    cv = std_t / abs(mean_t) if abs(mean_t) > 1e-10 else np.inf

    return mean_t, std_t, cv


def compute_cart_bin_edges(metrics_df, metrics_to_bin,
                            target_col='classification',
                            max_depth=2,
                            n_bootstrap=200,
                            extend_pct=0.01):
    """
    Compute CART-derived bin edges for a set of metrics.

    For each metric, fits a shallow CART to predict joint satisficing and
    uses the split thresholds as bin edges. Falls back to quantile bins
    if CART finds no splits or if stability is poor.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Event metrics DataFrame with classification column.
    metrics_to_bin : list of str
        Column names to compute bin edges for.
    target_col : str
        Classification column name ('classification').
    max_depth : int
        Maximum CART depth per metric (1=2 bins, 2=3 bins max).
    n_bootstrap : int
        Bootstrap resamples for stability assessment.
    extend_pct : float
        Fraction to extend bin edges beyond data range.

    Returns
    -------
    dict[str, CARTBinResult]
        Mapping from metric name to CARTBinResult.
    """
    # Binary target: 1 = satisficing (all_pass), 0 = any failure
    y = (metrics_df[target_col] == 'all_pass').astype(int).values

    results = {}

    for metric in metrics_to_bin:
        X = metrics_df[metric].values
        valid = ~np.isnan(X)
        X_valid = X[valid]
        y_valid = y[valid]

        if len(X_valid) < 10:
            # Too few samples, fall back to quantile
            results[metric] = _fallback_quantile(X_valid, metric)
            continue

        # Fit CART
        thresholds, ba, tree = fit_univariate_cart(
            X_valid, y_valid, max_depth=max_depth
        )

        # Bootstrap stability
        _, _, cv = bootstrap_threshold_stability(
            X_valid, y_valid, n_bootstrap=n_bootstrap
        )

        # Classify stability
        if np.isnan(cv) or np.isinf(cv):
            stability = 'unstable'
        elif cv < 0.10:
            stability = 'stable'
        elif cv < 0.30:
            stability = 'moderate'
        else:
            stability = 'unstable'

        if not thresholds:
            # No splits found, fall back
            results[metric] = _fallback_quantile(X_valid, metric)
            continue

        # Build full bin edges extending beyond data range
        data_min = float(X_valid.min())
        data_max = float(X_valid.max())
        data_range = data_max - data_min if data_max > data_min else 1.0
        ext = extend_pct * data_range

        bin_edges = [data_min - ext] + list(thresholds) + [data_max + ext]
        n_bins = len(bin_edges) - 1

        # Auto-generate bin labels
        bin_labels = _make_bin_labels(metric, bin_edges, thresholds)

        results[metric] = CARTBinResult(
            metric=metric,
            thresholds=thresholds,
            balanced_accuracy=ba,
            bootstrap_cv=cv,
            stability=stability,
            n_bins=n_bins,
            bin_edges=bin_edges,
            bin_labels=bin_labels,
        )

    return results


def _fallback_quantile(X_valid, metric):
    """Create a quantile-based fallback CARTBinResult."""
    if len(X_valid) < 3:
        edges = [float(X_valid.min()) - 0.01, float(X_valid.max()) + 0.01]
        return CARTBinResult(
            metric=metric, thresholds=[], balanced_accuracy=0.5,
            bootstrap_cv=np.inf, stability='unstable',
            n_bins=1, bin_edges=edges, bin_labels=['All'],
        )

    q = np.quantile(X_valid, [0, 1/3, 2/3, 1.0])
    edges = list(np.unique(q))
    if len(edges) < 3:
        edges = [float(X_valid.min()), float(np.median(X_valid)),
                 float(X_valid.max())]
    edges[0] -= 0.01
    edges[-1] += 0.01
    thresholds = edges[1:-1]

    return CARTBinResult(
        metric=metric, thresholds=thresholds,
        balanced_accuracy=0.5,
        bootstrap_cv=np.inf, stability='unstable',
        n_bins=len(edges) - 1, bin_edges=edges,
        bin_labels=_make_bin_labels(metric, edges, thresholds),
    )


def _make_bin_labels(metric, bin_edges, thresholds):
    """
    Generate human-readable bin labels from edges.

    Uses metric-specific formatting where possible.
    """
    n_bins = len(bin_edges) - 1
    labels = []

    # Format thresholds based on metric scale
    if metric in ('min_storage_pct', 'storage_at_start_pct'):
        fmt = lambda v: f"{v:.0f}%"
    elif metric in ('nyc_diversion_sat_ratio', 'contribution_ratio'):
        fmt = lambda v: f"{v:.2f}"
    elif metric in ('severity', 'avg_severity', 'magnitude'):
        fmt = lambda v: f"{v:.1f}"
    elif metric in ('duration_days',):
        def fmt(v):
            if v < 90:
                return f"{v:.0f}d"
            elif v < 365:
                return f"{v/30:.0f}mo"
            else:
                return f"{v/365:.1f}yr"
    elif metric in ('start_month',):
        fmt = lambda v: f"{v:.0f}"
    elif metric in ('total_nyc_contribution_mg', 'total_inflow_mg',
                     'total_montague_shortage_mg'):
        def fmt(v):
            if abs(v) >= 1e6:
                return f"{v/1e6:.0f}M MG"
            elif abs(v) >= 1000:
                return f"{v/1000:.0f}K MG"
            else:
                return f"{v:.0f} MG"
    elif metric in ('max_consec_montague_days',):
        fmt = lambda v: f"{v:.0f}d"
    else:
        fmt = lambda v: f"{v:.2f}"

    for i in range(n_bins):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]
        if i == 0:
            labels.append(f"< {fmt(hi)}")
        elif i == n_bins - 1:
            labels.append(f"> {fmt(lo)}")
        else:
            labels.append(f"{fmt(lo)}-{fmt(hi)}")

    return labels


def cart_results_to_axis_configs(cart_results, axis_order, fixed_axes=None):
    """
    Convert CARTBinResults into AxisConfig objects for the Sankey figure.

    Parameters
    ----------
    cart_results : dict[str, CARTBinResult]
        From compute_cart_bin_edges().
    axis_order : list of str
        Metrics in desired top-to-bottom order.
    fixed_axes : dict[str, dict], optional
        Override specific axes with fixed AxisConfig kwargs.
        E.g., {'start_month': {'bin_edges': [0.5,3.5,7.5,12.5],
                                'bin_labels': ['Jan-Mar','Apr-Jul','Aug-Dec']}}

    Returns
    -------
    list of AxisConfig
        Ready-to-use axis configs for SankeyFigureConfig.
    """
    from methods.plotting.sankey_parallel import AxisConfig

    if fixed_axes is None:
        fixed_axes = {}

    axis_configs = []
    for metric in axis_order:
        if metric in fixed_axes:
            # Use provided fixed config
            kwargs = fixed_axes[metric].copy()
            kwargs['metric'] = metric
            axis_configs.append(AxisConfig(**kwargs))
        elif metric in cart_results:
            r = cart_results[metric]
            # Annotate label with stability
            stability_marker = ''
            if r.stability == 'unstable':
                stability_marker = ' *'
            elif r.stability == 'moderate':
                stability_marker = ' ~'

            label = _default_label(metric) + stability_marker

            axis_configs.append(AxisConfig(
                metric=metric,
                label=label,
                bin_edges=r.bin_edges,
                bin_labels=r.bin_labels,
            ))
        else:
            # Metric not in CART results, use quantile fallback
            axis_configs.append(AxisConfig(
                metric=metric,
                label=_default_label(metric),
                bin_edges='quantile',
            ))

    return axis_configs


def _default_label(metric):
    """Map metric column name to a display label."""
    labels = {
        'start_month': 'Drought Start',
        'severity': 'Drought Severity',
        'duration_days': 'Drought Duration',
        'magnitude': 'Drought Magnitude',
        'storage_at_start_pct': 'Storage at\nDrought Onset',
        'total_nyc_contribution_mg': 'NYC Contribution\nto Montague',
        'contribution_ratio': 'Contribution\nRatio',
        'max_consec_montague_days': 'Montague\nViolations',
        'min_storage_pct': 'Min Storage\nDuring Drought',
        'total_montague_shortage_mg': 'Montague\nShortage',
        'nyc_diversion_sat_ratio': 'NYC Diversion\nSatisfaction',
    }
    return labels.get(metric, metric.replace('_', ' ').title())


def print_cart_summary(cart_results, stream=None):
    """
    Print a formatted summary of CART bin edge results.

    Parameters
    ----------
    cart_results : dict[str, CARTBinResult]
    stream : file-like, optional
        Defaults to stdout.
    """
    import sys
    out = stream or sys.stdout

    out.write("\nCART Bin Edge Summary\n")
    out.write(f"{'Metric':<35} {'Bins':>4} {'Bal.Acc':>8} {'CV':>8} "
              f"{'Stability':<10} Thresholds\n")
    out.write(f"{'-'*35} {'-'*4} {'-'*8} {'-'*8} {'-'*10} {'-'*30}\n")

    for metric, r in cart_results.items():
        thresh_str = ', '.join(f'{t:.2f}' for t in r.thresholds)
        out.write(
            f"{metric:<35} {r.n_bins:>4} {r.balanced_accuracy:>8.3f} "
            f"{r.bootstrap_cv:>8.3f} {r.stability:<10} [{thresh_str}]\n"
        )
    out.write("\n")
