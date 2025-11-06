"""
SI3: Comprehensive Evaluation of Clustering Potential for Drought Characteristics

This script evaluates whether drought characteristics cluster meaningfully by:
1. Loading drought characteristic data for a given SSI window
2. Testing different numbers of clusters with feature-tuned data
3. Repeating analysis with PCA-transformed data
4. Comparing clustering quality metrics and providing statistical validation
5. Generating comprehensive diagnostic plots

The analysis answers:
- Does the data cluster well (with separation)?
- Does PCA transformation improve clustering?
- What is the optimal number of clusters, if any?
- Is clustering statistically sound with this data?

Usage:
    python SI3_evaluate_cluster_potential.py <dataset_id> <ssi_window>

Example:
    python SI3_evaluate_cluster_potential.py stationary_ensemble 6
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings("ignore")

# Clustering algorithms
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples
)

# Statistical tests
from scipy.stats import f_oneway, kruskal
from sklearn.metrics import pairwise_distances

from config import *

# Output directories
FIG_DIR_CLUSTER = f"{FIG_DIR}/clustering_analysis"
os.makedirs(FIG_DIR_CLUSTER, exist_ok=True)

DATA_DIR_CLUSTER = f"{ROOT_DIR}/pywrdrb/drought_metrics/clustering_analysis"
os.makedirs(DATA_DIR_CLUSTER, exist_ok=True)


def load_drought_characteristics(dataset_id, ssi_window):
    """
    Load drought characteristic data for a given dataset and SSI window.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    ssi_window : int
        SSI window (3, 6, or 12 months)

    Returns
    -------
    pd.DataFrame
        Drought characteristics with columns:
        start, end, duration, magnitude, severity, max_severity_date, realization_id
    """
    verify_dataset_id(dataset_id)

    fname = f"./pywrdrb/drought_metrics/{dataset_id}_ssi{ssi_window}_drought_events.csv"

    if not os.path.exists(fname):
        raise FileNotFoundError(f"Drought metrics file not found: {fname}")

    print(f"Loading drought characteristics from: {fname}")
    df = pd.read_csv(fname)

    # Convert date columns to datetime
    date_cols = ['start', 'end', 'max_severity_date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    print(f"  Loaded {len(df)} drought events")
    print(f"  Columns: {list(df.columns)}")

    return df


def prepare_features(df, feature_cols=None):
    """
    Prepare features for clustering with basic tuning.

    Parameters
    ----------
    df : pd.DataFrame
        Drought characteristics
    feature_cols : list, optional
        List of columns to use as features. If None, uses all numeric columns
        except realization_id

    Returns
    -------
    X : np.ndarray
        Standardized feature matrix (N_samples, N_features)
    feature_names : list
        Names of features used
    scaler : StandardScaler
        Fitted scaler object
    """
    # Select features
    if feature_cols is None:
        # Use all numeric columns except identifiers
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['realization_id', 'Unnamed: 0']
        feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    print(f"\nPreparing features for clustering:")
    print(f"  Selected features: {feature_cols}")

    # Extract features
    X = df[feature_cols].values

    # Remove rows with any NaN values
    valid_mask = ~np.isnan(X).any(axis=1)
    X = X[valid_mask]
    print(f"  Valid samples (no NaN): {X.shape[0]}/{len(df)}")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"  Feature matrix shape: {X_scaled.shape}")
    print(f"  Feature means (after scaling): {X_scaled.mean(axis=0)}")
    print(f"  Feature stds (after scaling): {X_scaled.std(axis=0)}")

    return X_scaled, feature_cols, scaler


def apply_pca(X, n_components=None, variance_threshold=0.95):
    """
    Apply PCA transformation to features.

    Parameters
    ----------
    X : np.ndarray
        Standardized feature matrix
    n_components : int, optional
        Number of components. If None, uses variance_threshold
    variance_threshold : float
        Cumulative variance threshold for automatic component selection

    Returns
    -------
    X_pca : np.ndarray
        PCA-transformed features
    pca : PCA
        Fitted PCA object
    """
    print(f"\nApplying PCA transformation:")

    if n_components is None:
        # Use variance threshold
        pca = PCA(n_components=variance_threshold)
    else:
        pca = PCA(n_components=n_components)

    X_pca = pca.fit_transform(X)

    print(f"  Original dimensions: {X.shape[1]}")
    print(f"  PCA components: {X_pca.shape[1]}")
    print(f"  Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"  Cumulative variance: {pca.explained_variance_ratio_.cumsum()}")

    return X_pca, pca


def evaluate_clustering(X, k_range=(2, 10), algorithm='kmeans'):
    """
    Evaluate clustering for different numbers of clusters.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    k_range : tuple
        Range of k values to test (min, max)
    algorithm : str
        Clustering algorithm: 'kmeans', 'hierarchical', 'gmm'

    Returns
    -------
    results : pd.DataFrame
        Clustering metrics for each k value
    models : dict
        Fitted clustering models for each k
    """
    print(f"\nEvaluating {algorithm} clustering for k={k_range[0]} to {k_range[1]}:")

    results = []
    models = {}

    for k in range(k_range[0], k_range[1] + 1):
        print(f"  Testing k={k}...")

        # Fit clustering model
        if algorithm == 'kmeans':
            model = KMeans(n_clusters=k, n_init=20, random_state=42)
        elif algorithm == 'hierarchical':
            model = AgglomerativeClustering(n_clusters=k)
        elif algorithm == 'gmm':
            model = GaussianMixture(n_components=k, n_init=10, random_state=42)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # Get cluster labels
        if algorithm == 'gmm':
            labels = model.fit_predict(X)
        else:
            labels = model.fit_predict(X)

        # Calculate metrics
        metrics = {}
        metrics['k'] = k
        metrics['n_samples'] = len(X)

        # Cluster sizes
        unique_labels, counts = np.unique(labels, return_counts=True)
        metrics['n_clusters_found'] = len(unique_labels)
        metrics['min_cluster_size'] = counts.min()
        metrics['max_cluster_size'] = counts.max()
        metrics['cluster_size_std'] = counts.std()

        # Silhouette score (higher is better, range: -1 to 1)
        if len(unique_labels) > 1:
            metrics['silhouette'] = silhouette_score(X, labels)
        else:
            metrics['silhouette'] = np.nan

        # Calinski-Harabasz score (higher is better)
        if len(unique_labels) > 1:
            metrics['calinski_harabasz'] = calinski_harabasz_score(X, labels)
        else:
            metrics['calinski_harabasz'] = np.nan

        # Davies-Bouldin score (lower is better)
        if len(unique_labels) > 1:
            metrics['davies_bouldin'] = davies_bouldin_score(X, labels)
        else:
            metrics['davies_bouldin'] = np.nan

        # Inertia (for k-means only)
        if algorithm == 'kmeans':
            metrics['inertia'] = model.inertia_
        else:
            metrics['inertia'] = np.nan

        # BIC (for GMM only)
        if algorithm == 'gmm':
            metrics['bic'] = model.bic(X)
            metrics['aic'] = model.aic(X)
        else:
            metrics['bic'] = np.nan
            metrics['aic'] = np.nan

        results.append(metrics)
        models[k] = (model, labels)

    results_df = pd.DataFrame(results)
    print(f"\n  Clustering evaluation complete!")

    return results_df, models


def gap_statistic(X, k_range=(2, 10), n_refs=10, algorithm='kmeans'):
    """
    Calculate Gap Statistic for determining optimal k.

    The Gap Statistic compares within-cluster dispersion to that expected
    under a null reference distribution (uniform random data).

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    k_range : tuple
        Range of k values to test
    n_refs : int
        Number of reference datasets
    algorithm : str
        Clustering algorithm

    Returns
    -------
    gaps : np.ndarray
        Gap statistic for each k
    s_k : np.ndarray
        Standard error for each k
    """
    print(f"\nCalculating Gap Statistic (n_refs={n_refs}):")

    k_values = np.arange(k_range[0], k_range[1] + 1)
    gaps = np.zeros(len(k_values))
    s_k = np.zeros(len(k_values))

    # Compute dispersions on actual data
    W_k = np.zeros(len(k_values))
    for i, k in enumerate(k_values):
        if algorithm == 'kmeans':
            model = KMeans(n_clusters=k, n_init=10, random_state=42)
        elif algorithm == 'gmm':
            model = GaussianMixture(n_components=k, n_init=5, random_state=42)
        else:
            raise ValueError(f"Gap statistic not implemented for {algorithm}")

        if algorithm == 'gmm':
            labels = model.fit_predict(X)
        else:
            model.fit(X)
            labels = model.labels_

        # Within-cluster dispersion
        W_k[i] = _calculate_dispersion(X, labels)

    # Compute dispersions on reference datasets
    W_k_refs = np.zeros((len(k_values), n_refs))
    for ref in range(n_refs):
        # Generate uniform reference data
        X_ref = np.random.uniform(X.min(axis=0), X.max(axis=0), size=X.shape)

        for i, k in enumerate(k_values):
            if algorithm == 'kmeans':
                model = KMeans(n_clusters=k, n_init=5, random_state=ref)
            elif algorithm == 'gmm':
                model = GaussianMixture(n_components=k, n_init=3, random_state=ref)

            if algorithm == 'gmm':
                labels = model.fit_predict(X_ref)
            else:
                model.fit(X_ref)
                labels = model.labels_

            W_k_refs[i, ref] = _calculate_dispersion(X_ref, labels)

    # Calculate gap statistic
    log_W_k = np.log(W_k)
    log_W_k_refs = np.log(W_k_refs)
    gaps = log_W_k_refs.mean(axis=1) - log_W_k

    # Calculate standard error
    sdk = log_W_k_refs.std(axis=1)
    s_k = sdk * np.sqrt(1 + 1.0 / n_refs)

    print(f"  Gap statistics calculated for k={k_range[0]} to {k_range[1]}")

    return gaps, s_k, k_values


def _calculate_dispersion(X, labels):
    """Calculate within-cluster dispersion."""
    dispersion = 0.0
    for label in np.unique(labels):
        cluster_points = X[labels == label]
        if len(cluster_points) > 0:
            center = cluster_points.mean(axis=0)
            dispersion += np.sum((cluster_points - center) ** 2)
    return dispersion


def statistical_validation(X, labels, feature_names):
    """
    Perform statistical tests to validate cluster separation.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    labels : np.ndarray
        Cluster labels
    feature_names : list
        Names of features

    Returns
    -------
    validation_results : dict
        Statistical test results
    """
    print(f"\nPerforming statistical validation of clusters:")

    results = {}

    # Get unique clusters
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    print(f"  Number of clusters: {n_clusters}")

    # ANOVA for each feature
    anova_results = []
    for i, feature_name in enumerate(feature_names):
        feature_values = X[:, i]
        groups = [feature_values[labels == label] for label in unique_labels]

        # Perform one-way ANOVA
        f_stat, p_value = f_oneway(*groups)

        # Also perform Kruskal-Wallis (non-parametric)
        h_stat, p_value_kw = kruskal(*groups)

        anova_results.append({
            'feature': feature_name,
            'f_statistic': f_stat,
            'p_value_anova': p_value,
            'h_statistic': h_stat,
            'p_value_kruskal': p_value_kw,
            'significant_anova': p_value < 0.05,
            'significant_kruskal': p_value_kw < 0.05
        })

    anova_df = pd.DataFrame(anova_results)
    results['anova'] = anova_df

    # Count how many features show significant separation
    n_sig_anova = (anova_df['p_value_anova'] < 0.05).sum()
    n_sig_kruskal = (anova_df['p_value_kruskal'] < 0.05).sum()

    print(f"  Features with significant cluster separation (α=0.05):")
    print(f"    ANOVA: {n_sig_anova}/{len(feature_names)}")
    print(f"    Kruskal-Wallis: {n_sig_kruskal}/{len(feature_names)}")

    # Calculate pairwise cluster separation
    cluster_centers = np.array([X[labels == label].mean(axis=0) for label in unique_labels])
    pairwise_dist = pairwise_distances(cluster_centers, metric='euclidean')

    results['pairwise_distances'] = pairwise_dist
    results['mean_pairwise_distance'] = pairwise_dist[np.triu_indices_from(pairwise_dist, k=1)].mean()

    print(f"  Mean pairwise cluster center distance: {results['mean_pairwise_distance']:.3f}")

    # Within-cluster variance
    within_variance = []
    for label in unique_labels:
        cluster_points = X[labels == label]
        if len(cluster_points) > 1:
            variance = np.var(cluster_points, axis=0).mean()
            within_variance.append(variance)

    results['mean_within_variance'] = np.mean(within_variance)

    print(f"  Mean within-cluster variance: {results['mean_within_variance']:.3f}")

    return results


def create_diagnostic_plots(X, X_pca, results_original, results_pca, models_original,
                            models_pca, feature_names, dataset_id, ssi_window,
                            gaps_original=None, gaps_pca=None):
    """
    Create comprehensive diagnostic plots.

    Parameters
    ----------
    X : np.ndarray
        Original feature matrix
    X_pca : np.ndarray
        PCA-transformed feature matrix
    results_original : pd.DataFrame
        Clustering results for original features
    results_pca : pd.DataFrame
        Clustering results for PCA features
    models_original : dict
        Fitted models for original features
    models_pca : dict
        Fitted models for PCA features
    feature_names : list
        Feature names
    dataset_id : str
        Dataset identifier
    ssi_window : int
        SSI window
    gaps_original : tuple, optional
        Gap statistic results for original features (gaps, s_k, k_values)
    gaps_pca : tuple, optional
        Gap statistic results for PCA features
    """
    print(f"\nCreating diagnostic plots...")

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 12)

    # Figure 1: Clustering metrics comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Clustering Quality Metrics: {dataset_id}, SSI-{ssi_window}',
                 fontsize=16, fontweight='bold')

    # Silhouette score
    ax = axes[0, 0]
    ax.plot(results_original['k'], results_original['silhouette'],
            marker='o', label='Original Features', linewidth=2)
    ax.plot(results_pca['k'], results_pca['silhouette'],
            marker='s', label='PCA Features', linewidth=2)
    ax.set_xlabel('Number of Clusters (k)', fontsize=11)
    ax.set_ylabel('Silhouette Score', fontsize=11)
    ax.set_title('Silhouette Score (higher is better)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Calinski-Harabasz score
    ax = axes[0, 1]
    ax.plot(results_original['k'], results_original['calinski_harabasz'],
            marker='o', label='Original Features', linewidth=2)
    ax.plot(results_pca['k'], results_pca['calinski_harabasz'],
            marker='s', label='PCA Features', linewidth=2)
    ax.set_xlabel('Number of Clusters (k)', fontsize=11)
    ax.set_ylabel('Calinski-Harabasz Score', fontsize=11)
    ax.set_title('Calinski-Harabasz Score (higher is better)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Davies-Bouldin score
    ax = axes[0, 2]
    ax.plot(results_original['k'], results_original['davies_bouldin'],
            marker='o', label='Original Features', linewidth=2)
    ax.plot(results_pca['k'], results_pca['davies_bouldin'],
            marker='s', label='PCA Features', linewidth=2)
    ax.set_xlabel('Number of Clusters (k)', fontsize=11)
    ax.set_ylabel('Davies-Bouldin Score', fontsize=11)
    ax.set_title('Davies-Bouldin Score (lower is better)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Inertia (elbow plot)
    ax = axes[1, 0]
    ax.plot(results_original['k'], results_original['inertia'],
            marker='o', label='Original Features', linewidth=2)
    ax.plot(results_pca['k'], results_pca['inertia'],
            marker='s', label='PCA Features', linewidth=2)
    ax.set_xlabel('Number of Clusters (k)', fontsize=11)
    ax.set_ylabel('Inertia', fontsize=11)
    ax.set_title('Within-Cluster Sum of Squares (Elbow Method)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Gap statistic (if provided)
    ax = axes[1, 1]
    if gaps_original is not None:
        gaps_o, s_k_o, k_vals_o = gaps_original
        ax.errorbar(k_vals_o, gaps_o, yerr=s_k_o, marker='o',
                   label='Original Features', linewidth=2, capsize=5)
    if gaps_pca is not None:
        gaps_p, s_k_p, k_vals_p = gaps_pca
        ax.errorbar(k_vals_p, gaps_p, yerr=s_k_p, marker='s',
                   label='PCA Features', linewidth=2, capsize=5)
    ax.set_xlabel('Number of Clusters (k)', fontsize=11)
    ax.set_ylabel('Gap Statistic', fontsize=11)
    ax.set_title('Gap Statistic (higher is better)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Cluster size distribution
    ax = axes[1, 2]
    k_test = 4  # Example k for cluster size comparison
    if k_test in models_original and k_test in models_pca:
        _, labels_orig = models_original[k_test]
        _, labels_pca = models_pca[k_test]

        counts_orig = np.bincount(labels_orig)
        counts_pca = np.bincount(labels_pca)

        x = np.arange(k_test)
        width = 0.35
        ax.bar(x - width/2, counts_orig, width, label='Original Features', alpha=0.8)
        ax.bar(x + width/2, counts_pca, width, label='PCA Features', alpha=0.8)
        ax.set_xlabel('Cluster ID', fontsize=11)
        ax.set_ylabel('Number of Samples', fontsize=11)
        ax.set_title(f'Cluster Size Distribution (k={k_test})', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fname = f"{FIG_DIR_CLUSTER}/{dataset_id}_ssi{ssi_window}_metrics_comparison.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close()

    # Figure 2: Silhouette analysis for optimal k
    optimal_k = results_original.loc[results_original['silhouette'].idxmax(), 'k']
    optimal_k = int(optimal_k)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Silhouette Analysis: k={optimal_k}', fontsize=16, fontweight='bold')

    # Original features
    if optimal_k in models_original:
        _, labels_orig = models_original[optimal_k]
        ax = axes[0]
        _plot_silhouette_analysis(X, labels_orig, ax, 'Original Features')

    # PCA features
    if optimal_k in models_pca:
        _, labels_pca = models_pca[optimal_k]
        ax = axes[1]
        _plot_silhouette_analysis(X_pca, labels_pca, ax, 'PCA Features')

    plt.tight_layout()
    fname = f"{FIG_DIR_CLUSTER}/{dataset_id}_ssi{ssi_window}_silhouette_analysis.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close()

    # Figure 3: Feature distribution by cluster
    if optimal_k in models_original and X.shape[1] <= 6:
        _, labels = models_original[optimal_k]

        n_features = X.shape[1]
        n_cols = 3
        n_rows = int(np.ceil(n_features / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
        fig.suptitle(f'Feature Distributions by Cluster: k={optimal_k}',
                     fontsize=16, fontweight='bold')

        axes = axes.flatten() if n_features > 1 else [axes]

        for i, feature_name in enumerate(feature_names):
            ax = axes[i]

            for cluster_id in range(optimal_k):
                cluster_data = X[labels == cluster_id, i]
                ax.hist(cluster_data, alpha=0.5, label=f'Cluster {cluster_id}', bins=30)

            ax.set_xlabel(feature_name, fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        fname = f"{FIG_DIR_CLUSTER}/{dataset_id}_ssi{ssi_window}_feature_distributions.png"
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        print(f"  Saved: {fname}")
        plt.close()

    # Figure 4: PCA biplot with clusters
    if optimal_k in models_pca and X_pca.shape[1] >= 2:
        _, labels_pca = models_pca[optimal_k]

        fig, ax = plt.subplots(figsize=(12, 10))

        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_pca,
                           cmap='tab10', alpha=0.6, s=30)

        # Add cluster centers
        for cluster_id in range(optimal_k):
            cluster_center = X_pca[labels_pca == cluster_id].mean(axis=0)
            ax.scatter(cluster_center[0], cluster_center[1],
                      c='red', marker='X', s=200,
                      edgecolors='black', linewidths=2)

        ax.set_xlabel('First Principal Component', fontsize=12)
        ax.set_ylabel('Second Principal Component', fontsize=12)
        ax.set_title(f'PCA Biplot with Clusters: k={optimal_k}', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, label='Cluster ID')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fname = f"{FIG_DIR_CLUSTER}/{dataset_id}_ssi{ssi_window}_pca_biplot.png"
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        print(f"  Saved: {fname}")
        plt.close()

    # Figure 5: Pairwise feature scatter (for original features)
    if optimal_k in models_original and X.shape[1] >= 2 and X.shape[1] <= 6:
        _, labels = models_original[optimal_k]

        # Create pairplot
        df_plot = pd.DataFrame(X, columns=feature_names)
        df_plot['Cluster'] = labels

        g = sns.pairplot(df_plot, hue='Cluster', palette='tab10',
                        diag_kind='kde', plot_kws={'alpha': 0.6})
        g.fig.suptitle(f'Pairwise Feature Relationships: k={optimal_k}',
                      fontsize=16, fontweight='bold', y=1.01)

        fname = f"{FIG_DIR_CLUSTER}/{dataset_id}_ssi{ssi_window}_pairplot.png"
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        print(f"  Saved: {fname}")
        plt.close()

    print(f"  All diagnostic plots created!")


def _plot_silhouette_analysis(X, labels, ax, title):
    """Helper function to plot silhouette analysis."""
    silhouette_vals = silhouette_samples(X, labels)
    silhouette_avg = silhouette_score(X, labels)

    y_lower = 10
    unique_labels = np.unique(labels)

    for i, label in enumerate(unique_labels):
        cluster_silhouette_vals = silhouette_vals[labels == label]
        cluster_silhouette_vals.sort()

        size_cluster_i = cluster_silhouette_vals.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.tab10(i / len(unique_labels))
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals,
                        facecolor=color, edgecolor=color, alpha=0.7)

        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(label))
        y_lower = y_upper + 10

    ax.set_xlabel('Silhouette Coefficient', fontsize=11)
    ax.set_ylabel('Cluster', fontsize=11)
    ax.set_title(title, fontsize=12)

    ax.axvline(x=silhouette_avg, color="red", linestyle="--", linewidth=2,
              label=f'Average: {silhouette_avg:.3f}')
    ax.legend()
    ax.set_xlim([-0.2, 1])


def generate_summary_report(results_original, results_pca, validation_original,
                           validation_pca, dataset_id, ssi_window):
    """
    Generate a comprehensive text summary report.

    Parameters
    ----------
    results_original : pd.DataFrame
        Clustering results for original features
    results_pca : pd.DataFrame
        Clustering results for PCA features
    validation_original : dict
        Statistical validation for original features
    validation_pca : dict
        Statistical validation for PCA features
    dataset_id : str
        Dataset identifier
    ssi_window : int
        SSI window
    """
    print(f"\nGenerating summary report...")

    report = []
    report.append("=" * 80)
    report.append(f"CLUSTERING ANALYSIS SUMMARY REPORT")
    report.append(f"Dataset: {dataset_id}, SSI Window: {ssi_window} months")
    report.append("=" * 80)
    report.append("")

    # Question 1: Does the data cluster well?
    report.append("QUESTION 1: Does the data cluster well (with separation)?")
    report.append("-" * 80)

    # Best silhouette scores
    best_k_orig = results_original.loc[results_original['silhouette'].idxmax(), 'k']
    best_sil_orig = results_original.loc[results_original['silhouette'].idxmax(), 'silhouette']

    best_k_pca = results_pca.loc[results_pca['silhouette'].idxmax(), 'k']
    best_sil_pca = results_pca.loc[results_pca['silhouette'].idxmax(), 'silhouette']

    report.append(f"Original Features:")
    report.append(f"  Best Silhouette Score: {best_sil_orig:.4f} (k={int(best_k_orig)})")
    report.append(f"  Interpretation: ", end="")
    if best_sil_orig > 0.5:
        report.append("Strong clustering structure")
    elif best_sil_orig > 0.25:
        report.append("Moderate clustering structure")
    else:
        report.append("Weak clustering structure")

    report.append("")
    report.append(f"PCA Features:")
    report.append(f"  Best Silhouette Score: {best_sil_pca:.4f} (k={int(best_k_pca)})")
    report.append(f"  Interpretation: ", end="")
    if best_sil_pca > 0.5:
        report.append("Strong clustering structure")
    elif best_sil_pca > 0.25:
        report.append("Moderate clustering structure")
    else:
        report.append("Weak clustering structure")

    report.append("")

    # Statistical significance
    anova_orig = validation_original['anova']
    n_sig_orig = (anova_orig['p_value_anova'] < 0.05).sum()
    pct_sig_orig = 100 * n_sig_orig / len(anova_orig)

    anova_pca = validation_pca['anova']
    n_sig_pca = (anova_pca['p_value_anova'] < 0.05).sum()
    pct_sig_pca = 100 * n_sig_pca / len(anova_pca)

    report.append(f"Statistical Significance (ANOVA, α=0.05):")
    report.append(f"  Original Features: {n_sig_orig}/{len(anova_orig)} features ({pct_sig_orig:.1f}%) show significant separation")
    report.append(f"  PCA Features: {n_sig_pca}/{len(anova_pca)} components ({pct_sig_pca:.1f}%) show significant separation")
    report.append("")

    # Mean cluster separation
    sep_orig = validation_original['mean_pairwise_distance']
    sep_pca = validation_pca['mean_pairwise_distance']

    report.append(f"Cluster Separation (mean pairwise distance):")
    report.append(f"  Original Features: {sep_orig:.4f}")
    report.append(f"  PCA Features: {sep_pca:.4f}")
    report.append("")

    # Within-cluster variance
    var_orig = validation_original['mean_within_variance']
    var_pca = validation_pca['mean_within_variance']

    report.append(f"Within-Cluster Variance:")
    report.append(f"  Original Features: {var_orig:.4f}")
    report.append(f"  PCA Features: {var_pca:.4f}")
    report.append("")

    # Question 2: Does PCA improve clustering?
    report.append("=" * 80)
    report.append("QUESTION 2: Does PCA transformation improve clustering?")
    report.append("-" * 80)

    improvement = best_sil_pca - best_sil_orig
    pct_improvement = 100 * improvement / abs(best_sil_orig) if best_sil_orig != 0 else 0

    report.append(f"Silhouette Score Comparison:")
    report.append(f"  Original: {best_sil_orig:.4f}")
    report.append(f"  PCA: {best_sil_pca:.4f}")
    report.append(f"  Change: {improvement:+.4f} ({pct_improvement:+.1f}%)")
    report.append("")

    if improvement > 0.05:
        report.append("CONCLUSION: PCA transformation IMPROVES clustering quality")
    elif improvement < -0.05:
        report.append("CONCLUSION: PCA transformation DEGRADES clustering quality")
    else:
        report.append("CONCLUSION: PCA transformation has MINIMAL impact on clustering quality")
    report.append("")

    # Question 3: What is the optimal number of clusters?
    report.append("=" * 80)
    report.append("QUESTION 3: What is the optimal number of clusters?")
    report.append("-" * 80)

    # Consider multiple metrics
    report.append("Optimal k by different metrics:")
    report.append("")

    report.append("Original Features:")
    k_sil = int(results_original.loc[results_original['silhouette'].idxmax(), 'k'])
    k_ch = int(results_original.loc[results_original['calinski_harabasz'].idxmax(), 'k'])
    k_db = int(results_original.loc[results_original['davies_bouldin'].idxmin(), 'k'])

    report.append(f"  Silhouette Score: k = {k_sil}")
    report.append(f"  Calinski-Harabasz: k = {k_ch}")
    report.append(f"  Davies-Bouldin: k = {k_db}")

    # Find consensus
    from collections import Counter
    k_votes = [k_sil, k_ch, k_db]
    k_consensus = Counter(k_votes).most_common(1)[0][0]
    report.append(f"  Consensus: k = {k_consensus}")
    report.append("")

    report.append("PCA Features:")
    k_sil_pca = int(results_pca.loc[results_pca['silhouette'].idxmax(), 'k'])
    k_ch_pca = int(results_pca.loc[results_pca['calinski_harabasz'].idxmax(), 'k'])
    k_db_pca = int(results_pca.loc[results_pca['davies_bouldin'].idxmin(), 'k'])

    report.append(f"  Silhouette Score: k = {k_sil_pca}")
    report.append(f"  Calinski-Harabasz: k = {k_ch_pca}")
    report.append(f"  Davies-Bouldin: k = {k_db_pca}")

    k_votes_pca = [k_sil_pca, k_ch_pca, k_db_pca]
    k_consensus_pca = Counter(k_votes_pca).most_common(1)[0][0]
    report.append(f"  Consensus: k = {k_consensus_pca}")
    report.append("")

    # Question 4: Is clustering statistically sound?
    report.append("=" * 80)
    report.append("QUESTION 4: Is clustering statistically sound with this data?")
    report.append("-" * 80)

    # Criteria for statistical soundness
    criteria_met = 0
    total_criteria = 5

    report.append("Statistical Soundness Criteria:")
    report.append("")

    # Criterion 1: Silhouette score > 0.25
    crit1 = best_sil_orig > 0.25 or best_sil_pca > 0.25
    criteria_met += int(crit1)
    report.append(f"1. Silhouette score > 0.25: {'✓ PASS' if crit1 else '✗ FAIL'}")
    report.append(f"   (Original: {best_sil_orig:.3f}, PCA: {best_sil_pca:.3f})")

    # Criterion 2: >50% features show significant separation
    crit2 = pct_sig_orig > 50 or pct_sig_pca > 50
    criteria_met += int(crit2)
    report.append(f"2. >50% features with significant separation (ANOVA): {'✓ PASS' if crit2 else '✗ FAIL'}")
    report.append(f"   (Original: {pct_sig_orig:.1f}%, PCA: {pct_sig_pca:.1f}%)")

    # Criterion 3: Davies-Bouldin < 2.0 (rule of thumb)
    db_orig = results_original.loc[results_original['davies_bouldin'].idxmin(), 'davies_bouldin']
    db_pca = results_pca.loc[results_pca['davies_bouldin'].idxmin(), 'davies_bouldin']
    crit3 = db_orig < 2.0 or db_pca < 2.0
    criteria_met += int(crit3)
    report.append(f"3. Davies-Bouldin score < 2.0: {'✓ PASS' if crit3 else '✗ FAIL'}")
    report.append(f"   (Original: {db_orig:.3f}, PCA: {db_pca:.3f})")

    # Criterion 4: Consistent optimal k across metrics (±1 cluster)
    k_range_orig = max(k_votes) - min(k_votes)
    k_range_pca = max(k_votes_pca) - min(k_votes_pca)
    crit4 = k_range_orig <= 2 or k_range_pca <= 2
    criteria_met += int(crit4)
    report.append(f"4. Consistent optimal k across metrics (range ≤ 2): {'✓ PASS' if crit4 else '✗ FAIL'}")
    report.append(f"   (Original range: {k_range_orig}, PCA range: {k_range_pca})")

    # Criterion 5: Reasonable cluster sizes (no cluster < 5% of data)
    best_k_to_check = k_consensus
    min_size_orig = results_original.loc[results_original['k'] == best_k_to_check, 'min_cluster_size'].values[0]
    min_size_pca = results_pca.loc[results_pca['k'] == best_k_to_check, 'min_cluster_size'].values[0]
    n_samples = results_original.loc[0, 'n_samples']
    min_pct_orig = 100 * min_size_orig / n_samples
    min_pct_pca = 100 * min_size_pca / n_samples
    crit5 = min_pct_orig >= 5 or min_pct_pca >= 5
    criteria_met += int(crit5)
    report.append(f"5. No cluster < 5% of data: {'✓ PASS' if crit5 else '✗ FAIL'}")
    report.append(f"   (Original min: {min_pct_orig:.1f}%, PCA min: {min_pct_pca:.1f}%)")
    report.append("")

    # Overall conclusion
    report.append(f"Criteria Met: {criteria_met}/{total_criteria}")
    report.append("")

    if criteria_met >= 4:
        report.append("OVERALL CONCLUSION: Clustering is STATISTICALLY SOUND")
        report.append("The data exhibits meaningful cluster structure that can be reliably identified.")
    elif criteria_met >= 2:
        report.append("OVERALL CONCLUSION: Clustering is MODERATELY SOUND")
        report.append("Some cluster structure exists, but results should be interpreted cautiously.")
    else:
        report.append("OVERALL CONCLUSION: Clustering is NOT STATISTICALLY SOUND")
        report.append("The data does not exhibit strong cluster structure. Alternative approaches recommended.")

    report.append("")
    report.append("=" * 80)
    report.append("RECOMMENDATIONS")
    report.append("=" * 80)
    report.append("")

    if criteria_met >= 4:
        report.append(f"✓ Proceed with clustering using k = {k_consensus} clusters")
        if best_sil_pca > best_sil_orig:
            report.append("✓ Use PCA-transformed features for improved clustering quality")
        else:
            report.append("✓ Original features provide adequate clustering quality")
    elif criteria_met >= 2:
        report.append("⚠ Clustering may be useful but consider:")
        report.append("  - Validating results with domain knowledge")
        report.append("  - Using ensemble clustering methods")
        report.append("  - Collecting additional features")
    else:
        report.append("✗ Clustering not recommended. Consider:")
        report.append("  - Alternative dimensionality reduction techniques")
        report.append("  - Density-based clustering (DBSCAN, HDBSCAN)")
        report.append("  - Analyzing data without clustering")
        report.append("  - Collecting more discriminative features")

    report.append("")
    report.append("=" * 80)

    # Join report lines
    report_text = "\n".join(str(line) for line in report)

    # Save to file
    fname = f"{DATA_DIR_CLUSTER}/{dataset_id}_ssi{ssi_window}_clustering_summary.txt"
    with open(fname, 'w') as f:
        f.write(report_text)

    print(f"  Saved: {fname}")

    # Also print to console
    print("\n" + report_text)

    return report_text


def save_results(results_original, results_pca, dataset_id, ssi_window):
    """Save clustering results to CSV."""
    print(f"\nSaving results to CSV...")

    # Save original features results
    fname_orig = f"{DATA_DIR_CLUSTER}/{dataset_id}_ssi{ssi_window}_clustering_original.csv"
    results_original.to_csv(fname_orig, index=False)
    print(f"  Saved: {fname_orig}")

    # Save PCA features results
    fname_pca = f"{DATA_DIR_CLUSTER}/{dataset_id}_ssi{ssi_window}_clustering_pca.csv"
    results_pca.to_csv(fname_pca, index=False)
    print(f"  Saved: {fname_pca}")


def main(dataset_id, ssi_window):
    """
    Main analysis function.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    ssi_window : int
        SSI window (3, 6, or 12 months)
    """
    print("=" * 80)
    print(f"CLUSTERING ANALYSIS: {dataset_id}, SSI-{ssi_window}")
    print("=" * 80)

    # Step 1: Load drought characteristics
    df = load_drought_characteristics(dataset_id, ssi_window)

    # Step 2: Prepare features (original)
    print("\n" + "=" * 80)
    print("ORIGINAL FEATURES ANALYSIS")
    print("=" * 80)
    X, feature_names, scaler = prepare_features(df)

    # Step 3: Evaluate clustering on original features
    results_original, models_original = evaluate_clustering(X, k_range=(2, 10), algorithm='kmeans')

    # Step 4: Calculate gap statistic for original features
    print("\nCalculating gap statistic (this may take a few minutes)...")
    gaps_original = gap_statistic(X, k_range=(2, 10), n_refs=10, algorithm='kmeans')

    # Step 5: Statistical validation for optimal k (original)
    optimal_k = int(results_original.loc[results_original['silhouette'].idxmax(), 'k'])
    _, labels_optimal = models_original[optimal_k]
    validation_original = statistical_validation(X, labels_optimal, feature_names)

    # Step 6: PCA transformation
    print("\n" + "=" * 80)
    print("PCA-TRANSFORMED FEATURES ANALYSIS")
    print("=" * 80)
    X_pca, pca = apply_pca(X, variance_threshold=0.95)

    # Step 7: Evaluate clustering on PCA features
    results_pca, models_pca = evaluate_clustering(X_pca, k_range=(2, 10), algorithm='kmeans')

    # Step 8: Calculate gap statistic for PCA features
    print("\nCalculating gap statistic for PCA features...")
    gaps_pca = gap_statistic(X_pca, k_range=(2, 10), n_refs=10, algorithm='kmeans')

    # Step 9: Statistical validation for optimal k (PCA)
    optimal_k_pca = int(results_pca.loc[results_pca['silhouette'].idxmax(), 'k'])
    _, labels_optimal_pca = models_pca[optimal_k_pca]

    # For PCA, we need to recreate feature names
    pca_feature_names = [f'PC{i+1}' for i in range(X_pca.shape[1])]
    validation_pca = statistical_validation(X_pca, labels_optimal_pca, pca_feature_names)

    # Step 10: Create diagnostic plots
    print("\n" + "=" * 80)
    print("GENERATING DIAGNOSTIC PLOTS")
    print("=" * 80)
    create_diagnostic_plots(X, X_pca, results_original, results_pca,
                           models_original, models_pca, feature_names,
                           dataset_id, ssi_window,
                           gaps_original, gaps_pca)

    # Step 11: Generate summary report
    print("\n" + "=" * 80)
    print("GENERATING SUMMARY REPORT")
    print("=" * 80)
    report = generate_summary_report(results_original, results_pca,
                                    validation_original, validation_pca,
                                    dataset_id, ssi_window)

    # Step 12: Save results
    save_results(results_original, results_pca, dataset_id, ssi_window)

    print("\n" + "=" * 80)
    print("CLUSTERING ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {DATA_DIR_CLUSTER}/")
    print(f"Figures saved to: {FIG_DIR_CLUSTER}/")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        print(f"\nAvailable datasets: {list(DATASET_CONFIGS.keys())}")
        print(f"Available SSI windows: {SSI_WINDOWS}")
        sys.exit(1)

    dataset_id = sys.argv[1]
    ssi_window = int(sys.argv[2])

    # Validate inputs
    verify_dataset_id(dataset_id)
    if ssi_window not in SSI_WINDOWS:
        print(f"ERROR: Invalid SSI window. Must be one of {SSI_WINDOWS}")
        sys.exit(1)

    main(dataset_id, ssi_window)
