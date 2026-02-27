"""
Diagnostic analysis: CART-based bin edge selection for Sankey figure.

This script evaluates whether classification trees can identify meaningful
bin boundaries that separate satisficing from non-satisficing drought events.

Approach:
  1. Univariate CART: Fit a separate shallow tree per metric (1D splits).
     This finds the threshold on each individual metric that best separates
     satisficing from non-satisficing events.
  2. Multivariate CART: Fit one tree using all metrics as features.
     The split hierarchy reveals which metrics are most discriminative and
     the interaction structure.
  3. Depth selection: For each approach, sweep max_depth = 1..5 and evaluate
     using stratified cross-validation with appropriate metrics for imbalanced
     data (balanced accuracy, F1-macro, MCC).
  4. Stability analysis: Bootstrap the tree fitting to assess split-point
     stability (coefficient of variation of thresholds across bootstrap samples).

Key concern: With N=107 events and only 7 failures (6.5% failure rate),
CART is operating in a severely imbalanced regime. We use class_weight='balanced'
and evaluate with metrics robust to imbalance.

Usage:
    python diagnostics_cart_bin_selection.py

Output:
    Prints diagnostic tables to console.
    Saves summary to figures/F14_sankey_parallel/cart_diagnostics.txt
"""

import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, balanced_accuracy_score, f1_score, matthews_corrcoef
import warnings
warnings.filterwarnings("ignore")

from methods.config import ROOT_DIR, FIG_DIR, DATASET_CONFIGS

# =============================================================================
# CONFIGURATION
# =============================================================================

EVENT_METRICS_DIR = os.path.join(ROOT_DIR, 'pywrdrb', 'event_metrics')
OUTPUT_DIR = os.path.join(FIG_DIR, 'F14_sankey_parallel')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Metrics to evaluate as potential axes (features for CART)
AXIS_METRICS = [
    'start_month',
    'severity',
    'duration_days',
    'magnitude',
    'storage_at_start_pct',
    'total_nyc_contribution_mg',
    'contribution_ratio',
    'max_consec_montague_days',
    'min_storage_pct',
    'total_montague_shortage_mg',
    'nyc_diversion_sat_ratio',
]

# Target definitions
TARGETS = {
    'joint_satisficing': lambda df: (df['storage_ok'] & df['montague_ok']).astype(int),
    'montague_ok': lambda df: df['montague_ok'].astype(int),
    'storage_ok': lambda df: df['storage_ok'].astype(int),
}

# Tree depth range to evaluate
MAX_DEPTHS = [1, 2, 3, 4, 5]

# Cross-validation folds (use fewer if minority class is very small)
N_BOOTSTRAP = 200


def load_data(dataset_id='stationary_ensemble', ssi_window=6):
    fname = os.path.join(EVENT_METRICS_DIR,
                         f'{dataset_id}_ssi{ssi_window}_event_metrics.csv')
    df = pd.read_csv(fname)
    print(f"Loaded {len(df)} events from {os.path.basename(fname)}")
    print(f"  Class distribution: {df['classification'].value_counts().to_dict()}")
    return df


# =============================================================================
# 1. UNIVARIATE CART: One tree per metric
# =============================================================================

def univariate_cart_analysis(df, target_name, target_func, output_lines):
    """
    Fit a 1D decision stump (depth=1) per metric to find the single
    best split point separating satisficing from non-satisficing.

    For depth > 1, find 2-3 thresholds per metric.
    """
    y = target_func(df)
    n_pos = y.sum()
    n_neg = len(y) - n_pos

    output_lines.append(f"\n{'='*70}")
    output_lines.append(f"UNIVARIATE CART ANALYSIS - Target: {target_name}")
    output_lines.append(f"  N={len(y)}, Positive(satisficing)={n_pos}, Negative(fail)={n_neg}")
    output_lines.append(f"  Failure rate: {n_neg/len(y)*100:.1f}%")
    output_lines.append(f"{'='*70}")

    if n_neg < 2:
        output_lines.append(f"  WARNING: Only {n_neg} failure(s). Univariate CART unreliable.")
        output_lines.append(f"  Splits will be shown but should not be trusted.")
        output_lines.append("")

    results = []

    for metric in AXIS_METRICS:
        X = df[[metric]].values
        best_depth = None
        best_score = -1
        best_tree = None

        for depth in [1, 2, 3]:
            tree = DecisionTreeClassifier(
                max_depth=depth,
                min_samples_leaf=max(3, int(0.05 * len(y))),  # At least 5% of data
                class_weight='balanced',
                random_state=42
            )
            tree.fit(X, y)

            # Balanced accuracy on training data (not ideal, but with N=107
            # and stratified CV with <7 failures, CV is unstable)
            y_pred = tree.predict(X)
            ba = balanced_accuracy_score(y, y_pred)

            if ba > best_score:
                best_score = ba
                best_depth = depth
                best_tree = tree

        # Extract split thresholds from best tree
        thresholds = _extract_thresholds(best_tree)
        n_leaves = best_tree.get_n_leaves()

        # Gini importance (always 1.0 for univariate, but useful for depth>1)
        gini_imp = best_tree.feature_importances_[0]

        results.append({
            'metric': metric,
            'best_depth': best_depth,
            'n_splits': len(thresholds),
            'n_leaves': n_leaves,
            'balanced_accuracy': best_score,
            'thresholds': thresholds,
        })

    # Sort by balanced accuracy
    results.sort(key=lambda x: x['balanced_accuracy'], reverse=True)

    output_lines.append(f"\n{'Metric':<35} {'Depth':>5} {'Splits':>6} {'Bal.Acc':>8}  Thresholds")
    output_lines.append(f"{'-'*35} {'-'*5} {'-'*6} {'-'*8}  {'-'*30}")

    for r in results:
        thresh_str = ', '.join(f'{t:.2f}' for t in sorted(r['thresholds']))
        output_lines.append(
            f"{r['metric']:<35} {r['best_depth']:>5} {r['n_splits']:>6} "
            f"{r['balanced_accuracy']:>8.3f}  [{thresh_str}]"
        )

    return results


def _extract_thresholds(tree):
    """Extract unique split thresholds from a fitted tree."""
    thresholds = []
    tree_ = tree.tree_
    for i in range(tree_.node_count):
        if tree_.feature[i] != -2:  # Not a leaf
            thresholds.append(tree_.threshold[i])
    return sorted(set(thresholds))


# =============================================================================
# 2. MULTIVARIATE CART: One tree using all metrics
# =============================================================================

def multivariate_cart_analysis(df, target_name, target_func, output_lines):
    """
    Fit a single tree using all metrics. Evaluate across depths 1-5.
    Use stratified CV where possible.
    """
    y = target_func(df)
    X = df[AXIS_METRICS].copy()
    n_neg = len(y) - y.sum()

    output_lines.append(f"\n{'='*70}")
    output_lines.append(f"MULTIVARIATE CART ANALYSIS - Target: {target_name}")
    output_lines.append(f"{'='*70}")

    # Determine CV strategy based on minority class size
    if n_neg >= 5:
        n_folds = min(5, n_neg)
        output_lines.append(f"  Using {n_folds}-fold stratified CV (minority class n={n_neg})")
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        use_cv = True
    else:
        output_lines.append(f"  WARNING: Minority class n={n_neg} too small for reliable CV.")
        output_lines.append(f"  Using training-set evaluation only (overfit expected).")
        use_cv = False

    scorers = {
        'balanced_accuracy': make_scorer(balanced_accuracy_score),
        'f1_macro': make_scorer(f1_score, average='macro', zero_division=0),
        'mcc': make_scorer(matthews_corrcoef),
    }

    output_lines.append(f"\n{'Depth':>5} {'Bal.Acc':>8} {'F1-Mac':>8} {'MCC':>8}  Top Features (splits)")
    output_lines.append(f"{'-'*5} {'-'*8} {'-'*8} {'-'*8}  {'-'*40}")

    best_depth = 1
    best_ba = 0

    for depth in MAX_DEPTHS:
        tree = DecisionTreeClassifier(
            max_depth=depth,
            min_samples_leaf=max(3, int(0.05 * len(y))),
            class_weight='balanced',
            random_state=42,
        )

        if use_cv:
            scores = cross_validate(tree, X, y, cv=cv, scoring=scorers,
                                     return_train_score=False, error_score='raise')
            ba = scores['test_balanced_accuracy'].mean()
            f1 = scores['test_f1_macro'].mean()
            mcc = scores['test_mcc'].mean()
            ba_std = scores['test_balanced_accuracy'].std()
        else:
            tree.fit(X, y)
            y_pred = tree.predict(X)
            ba = balanced_accuracy_score(y, y_pred)
            f1 = f1_score(y, y_pred, average='macro', zero_division=0)
            mcc = matthews_corrcoef(y, y_pred)
            ba_std = 0.0

        # Refit on full data to get feature importances and tree structure
        tree.fit(X, y)
        importances = tree.feature_importances_
        top_features = sorted(zip(AXIS_METRICS, importances),
                              key=lambda x: x[1], reverse=True)
        top_str = ', '.join(f'{f}({i:.2f})' for f, i in top_features[:3] if i > 0)

        std_str = f" +/- {ba_std:.3f}" if use_cv else " (train)"
        output_lines.append(
            f"{depth:>5} {ba:>8.3f}{std_str:>12} {f1:>8.3f} {mcc:>8.3f}  {top_str}"
        )

        if ba > best_ba:
            best_ba = ba
            best_depth = depth

    # Print best tree structure
    output_lines.append(f"\n  Best depth: {best_depth} (balanced accuracy = {best_ba:.3f})")

    tree = DecisionTreeClassifier(
        max_depth=best_depth,
        min_samples_leaf=max(3, int(0.05 * len(y))),
        class_weight='balanced',
        random_state=42,
    )
    tree.fit(X, y)

    output_lines.append(f"\n  Tree structure (depth={best_depth}):")
    tree_text = export_text(tree, feature_names=AXIS_METRICS, max_depth=5)
    for line in tree_text.split('\n'):
        output_lines.append(f"    {line}")

    # Extract all split points by feature
    output_lines.append(f"\n  Split points by feature:")
    splits_by_feature = _extract_splits_by_feature(tree, AXIS_METRICS)
    for feat, thresholds in sorted(splits_by_feature.items()):
        thresh_str = ', '.join(f'{t:.3f}' for t in thresholds)
        output_lines.append(f"    {feat}: [{thresh_str}]")

    return tree, splits_by_feature


def _extract_splits_by_feature(tree, feature_names):
    """Extract split thresholds grouped by feature name."""
    splits = {}
    tree_ = tree.tree_
    for i in range(tree_.node_count):
        if tree_.feature[i] != -2:  # Not a leaf
            feat_name = feature_names[tree_.feature[i]]
            threshold = tree_.threshold[i]
            if feat_name not in splits:
                splits[feat_name] = []
            splits[feat_name].append(threshold)
    # Deduplicate and sort
    for feat in splits:
        splits[feat] = sorted(set(splits[feat]))
    return splits


# =============================================================================
# 3. BOOTSTRAP STABILITY ANALYSIS
# =============================================================================

def bootstrap_stability_analysis(df, target_name, target_func, output_lines):
    """
    Bootstrap resample and refit univariate stumps to assess
    split-point stability. Reports mean, std, and CV of thresholds.
    """
    y = target_func(df)
    n_neg = len(y) - y.sum()

    output_lines.append(f"\n{'='*70}")
    output_lines.append(f"BOOTSTRAP STABILITY ANALYSIS - Target: {target_name}")
    output_lines.append(f"  {N_BOOTSTRAP} bootstrap resamples, depth=1 stumps")
    output_lines.append(f"{'='*70}")

    if n_neg < 3:
        output_lines.append(f"  WARNING: Only {n_neg} failure(s). Bootstrap will be unstable.")
        output_lines.append(f"  Results shown for completeness but should be interpreted cautiously.\n")

    rng = np.random.RandomState(42)

    results = {}
    for metric in AXIS_METRICS:
        thresholds = []
        for b in range(N_BOOTSTRAP):
            # Stratified bootstrap: resample with replacement, preserving class ratio
            idx = _stratified_bootstrap(y, rng)
            X_b = df[[metric]].values[idx]
            y_b = y.values[idx]

            tree = DecisionTreeClassifier(
                max_depth=1,
                min_samples_leaf=max(2, int(0.03 * len(y_b))),
                class_weight='balanced',
                random_state=b,
            )
            tree.fit(X_b, y_b)

            # Extract threshold (depth=1 means at most 1 split)
            t = _extract_thresholds(tree)
            if t:
                thresholds.append(t[0])

        if thresholds:
            arr = np.array(thresholds)
            results[metric] = {
                'mean': arr.mean(),
                'std': arr.std(),
                'cv': arr.std() / abs(arr.mean()) if abs(arr.mean()) > 1e-10 else np.inf,
                'q25': np.percentile(arr, 25),
                'q75': np.percentile(arr, 75),
                'n_valid': len(thresholds),
            }
        else:
            results[metric] = {
                'mean': np.nan, 'std': np.nan, 'cv': np.nan,
                'q25': np.nan, 'q75': np.nan, 'n_valid': 0,
            }

    output_lines.append(f"\n{'Metric':<35} {'Mean':>10} {'Std':>10} {'CV':>8} {'IQR':>20} {'Valid':>6}")
    output_lines.append(f"{'-'*35} {'-'*10} {'-'*10} {'-'*8} {'-'*20} {'-'*6}")

    for metric in AXIS_METRICS:
        r = results[metric]
        iqr_str = f"[{r['q25']:.2f}, {r['q75']:.2f}]"
        output_lines.append(
            f"{metric:<35} {r['mean']:>10.3f} {r['std']:>10.3f} {r['cv']:>8.3f} {iqr_str:>20} {r['n_valid']:>6}"
        )

    # Identify stable vs unstable metrics
    output_lines.append(f"\n  Stability assessment (CV < 0.10 = stable, 0.10-0.30 = moderate, > 0.30 = unstable):")
    for metric in AXIS_METRICS:
        cv = results[metric]['cv']
        if np.isnan(cv) or np.isinf(cv):
            stability = "UNDEFINED"
        elif cv < 0.10:
            stability = "STABLE"
        elif cv < 0.30:
            stability = "MODERATE"
        else:
            stability = "UNSTABLE"
        output_lines.append(f"    {metric:<35} CV={cv:.3f}  [{stability}]")

    return results


def _stratified_bootstrap(y, rng):
    """Stratified bootstrap: resample with replacement within each class."""
    indices = []
    for cls in y.unique():
        cls_idx = np.where(y.values == cls)[0]
        boot_idx = rng.choice(cls_idx, size=len(cls_idx), replace=True)
        indices.extend(boot_idx)
    return np.array(indices)


# =============================================================================
# 4. RECOMMENDATIONS
# =============================================================================

def generate_recommendations(univariate_results, multivariate_splits,
                              bootstrap_results, output_lines):
    """
    Synthesize CART results into bin edge recommendations.
    """
    output_lines.append(f"\n{'='*70}")
    output_lines.append(f"BIN EDGE RECOMMENDATIONS")
    output_lines.append(f"{'='*70}")

    output_lines.append(f"\nApproach: For each metric, combine:")
    output_lines.append(f"  1. Univariate CART split point (best separating threshold)")
    output_lines.append(f"  2. Bootstrap stability (is the split reliable?)")
    output_lines.append(f"  3. Multivariate CART splits (does this metric matter in context?)")
    output_lines.append(f"  4. Physical meaning (does the threshold make domain sense?)")

    output_lines.append(f"\nCAVEAT: With N=107 events and ~7 failures (5 local realizations),")
    output_lines.append(f"  all thresholds are provisional. HPC-scale (2000 realizations,")
    output_lines.append(f"  ~40,000+ events) will yield stable, publishable thresholds.")
    output_lines.append(f"  The methodology is sound; the sample size is the limitation.\n")

    for metric in AXIS_METRICS:
        uni = next((r for r in univariate_results if r['metric'] == metric), None)
        boot = bootstrap_results.get(metric, {})
        multi_thresholds = multivariate_splits.get(metric, [])

        output_lines.append(f"  {metric}:")
        if uni and uni['thresholds']:
            output_lines.append(f"    Univariate split(s): {[f'{t:.3f}' for t in uni['thresholds']]}")
            output_lines.append(f"    Univariate balanced accuracy: {uni['balanced_accuracy']:.3f}")
        if boot.get('mean') is not None and not np.isnan(boot['mean']):
            output_lines.append(f"    Bootstrap mean: {boot['mean']:.3f} (CV={boot['cv']:.3f})")
        if multi_thresholds:
            output_lines.append(f"    Multivariate split(s): {[f'{t:.3f}' for t in multi_thresholds]}")
        if not multi_thresholds and (not uni or not uni['thresholds']):
            output_lines.append(f"    No discriminative splits found.")
        output_lines.append("")


# =============================================================================
# MAIN
# =============================================================================

def run_diagnostics_for_dataset(dataset_id, ssi_window):
    """Run full CART diagnostics for one dataset/SSI combination."""
    try:
        df = load_data(dataset_id, ssi_window)
    except FileNotFoundError as e:
        print(f"  Skipping {dataset_id} SSI-{ssi_window}: {e}")
        return None

    output_lines = []
    output_lines.append("CART-BASED BIN SELECTION DIAGNOSTICS")
    output_lines.append(f"Dataset: {dataset_id}, SSI-{ssi_window}")
    output_lines.append(f"N events: {len(df)}")
    output_lines.append(f"Classification: {df['classification'].value_counts().to_dict()}")

    all_univariate = {}
    all_multi_splits = {}
    all_bootstrap = {}

    for target_name, target_func in TARGETS.items():
        y = target_func(df)
        n_neg = len(y) - y.sum()

        # Univariate
        uni_results = univariate_cart_analysis(df, target_name, target_func, output_lines)
        all_univariate[target_name] = uni_results

        # Multivariate
        tree, splits = multivariate_cart_analysis(df, target_name, target_func, output_lines)
        all_multi_splits[target_name] = splits

        # Bootstrap stability
        boot = bootstrap_stability_analysis(df, target_name, target_func, output_lines)
        all_bootstrap[target_name] = boot

    # Primary target for bin recommendations
    generate_recommendations(
        all_univariate['joint_satisficing'],
        all_multi_splits['joint_satisficing'],
        all_bootstrap['joint_satisficing'],
        output_lines,
    )

    # Print everything
    report = '\n'.join(output_lines)
    print(report)

    # Save to file
    report_path = os.path.join(
        OUTPUT_DIR,
        f'cart_diagnostics_{dataset_id}_ssi{ssi_window}.txt'
    )
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nSaved diagnostic report to: {report_path}")
    return report_path


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='CART-based bin selection diagnostics'
    )
    parser.add_argument('--datasets', nargs='+', type=str,
                        default=list(DATASET_CONFIGS.keys()),
                        help='Dataset IDs to process')
    parser.add_argument('--ssi_window', type=int, default=6,
                        help='SSI window (default: 6)')
    args = parser.parse_args()

    for dataset_id in args.datasets:
        run_diagnostics_for_dataset(dataset_id, args.ssi_window)


if __name__ == '__main__':
    main()
