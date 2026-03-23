"""
PCA Diagnostic: Drought Event Feature Reduction for Outcome Discrimination

Applies PCA to antecedent + hazard features (excluding action/response and
storage outcomes to prevent leakage) and evaluates whether principal components
discriminate FFMP drought zone at minimum storage.

Features used (no outcome leakage):
  Antecedent: storage_at_start_pct, start_month (sin/cos encoded)
  Hazard:     severity, magnitude, duration_days, severity_rate,
              peak_severity_month (sin/cos encoded), total_inflow_mg

Outcome (target): ffmp_zone_at_min (FFMP zone classification)

Figures produced:
  1. PCA Overview — scree plot, loadings heatmap, biplots
  2. Zone Separation — box plots of PC scores by FFMP zone with KW stats
  3. Cross-Dataset — PC1 vs PC2 for each climate scenario
  4. Pairwise PCs — scatter matrix for top-discriminating PCs
  5. Raw vs PCA — side-by-side comparison of discrimination quality

Usage:
    python diagnostics_pca_discrimination.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

from methods.config import ROOT_DIR, FIG_DIR
from methods.plotting.styles import (
    FFMP_ZONE_COLORS, DATASET_LABELS,
    FONTSIZE_LABEL, DPI_HIGH, apply_publication_style,
)

FIG_OUTPUT_DIR = f"{FIG_DIR}/diagnostics_pca_discrimination"
os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

SSI_WINDOW = 3
MIN_DURATION = 30
DATASETS = ['stationary_ensemble', 'climate_adjusted_low', 'climate_adjusted_high']
FFMP_ZONE_ORDER = ['Normal', 'Watch', 'Warning', 'Emergency']

# PCA feature names after cyclical month encoding (10 features)
PCA_FEATURES = [
    'storage_at_start_pct',
    'start_month_sin', 'start_month_cos',
    'severity', 'magnitude', 'duration_days', 'severity_rate',
    'peak_month_sin', 'peak_month_cos',
    'total_inflow_mg',
]

FEATURE_SHORT = {
    'storage_at_start_pct': 'Start Storage',
    'start_month_sin': 'Start Month (sin)',
    'start_month_cos': 'Start Month (cos)',
    'severity': 'Severity',
    'magnitude': 'Magnitude',
    'duration_days': 'Duration',
    'severity_rate': 'Severity Rate',
    'peak_month_sin': 'Peak Month (sin)',
    'peak_month_cos': 'Peak Month (cos)',
    'total_inflow_mg': 'Total Inflow',
}


# ── data helpers ──────────────────────────────────────────────────────

def load_events(dataset_id):
    """Load event metrics, filter, and add cyclical month features."""
    df = pd.read_csv(
        f'{ROOT_DIR}/pywrdrb/event_metrics/'
        f'{dataset_id}_ssi{SSI_WINDOW}_event_metrics.csv'
    )
    df = df[df['duration_days'] >= MIN_DURATION].copy()
    df['severity'] = df['severity'].abs()
    df['magnitude'] = df['magnitude'].abs()

    # Cyclical encoding so month 12 ≈ month 1
    df['start_month_sin'] = np.sin(2 * np.pi * df['start_month'] / 12)
    df['start_month_cos'] = np.cos(2 * np.pi * df['start_month'] / 12)
    df['peak_month_sin'] = np.sin(2 * np.pi * df['peak_severity_month'] / 12)
    df['peak_month_cos'] = np.cos(2 * np.pi * df['peak_severity_month'] / 12)
    return df


def fit_pca(df, n_components=None):
    """Standardise features and fit PCA; return (X_scaled, X_pca, pca, scaler)."""
    X = df[PCA_FEATURES].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    if n_components is None:
        n_components = len(PCA_FEATURES)
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    return X_scaled, X_pca, pca, scaler


# ── statistics ────────────────────────────────────────────────────────

def kruskal_wallis_by_zone(df, scores, n_cols=None):
    """Kruskal-Wallis H-test for each column of *scores* across FFMP zones."""
    if n_cols is None:
        n_cols = scores.shape[1]
    zones = df['ffmp_zone_at_min'].values
    unique_zones = [z for z in FFMP_ZONE_ORDER if z in np.unique(zones)]
    rows = []
    for i in range(n_cols):
        groups = [scores[zones == z, i] for z in unique_zones if (zones == z).sum() > 0]
        if len(groups) < 2:
            rows.append({'col': i, 'H': np.nan, 'p': np.nan, 'eta_sq': np.nan})
            continue
        H, p = stats.kruskal(*groups)
        n, k = len(zones), len(groups)
        eta_sq = max(0, (H - k + 1) / (n - k))
        rows.append({'col': i, 'H': H, 'p': p, 'eta_sq': eta_sq})
    return pd.DataFrame(rows)


def raw_feature_eta(df, feat):
    """η² for a single raw feature across FFMP zones."""
    vals = df[feat].fillna(0).values
    zones = df['ffmp_zone_at_min'].values
    groups = [vals[zones == z] for z in FFMP_ZONE_ORDER if (zones == z).sum() > 0]
    if len(groups) < 2:
        return 0.0
    H, _ = stats.kruskal(*groups)
    n, k = len(vals), len(groups)
    return max(0, (H - k + 1) / (n - k))


# ── plotting helpers ──────────────────────────────────────────────────

def _zone_scatter(ax, df, x, y, xlabel, ylabel, is_pc=False, legend=True):
    """Scatter coloured by FFMP zone."""
    for zone in FFMP_ZONE_ORDER:
        m = df['ffmp_zone_at_min'] == zone
        if m.sum() == 0:
            continue
        xv = x[m.values] if is_pc else df.loc[m, x]
        yv = y[m.values] if is_pc else df.loc[m, y]
        ax.scatter(xv, yv, c=FFMP_ZONE_COLORS[zone], alpha=0.5, s=18,
                   edgecolors='black', linewidths=0.3, zorder=3, label=zone)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend(fontsize=8, framealpha=0.9, edgecolor='#ccc', loc='best')
    ax.axhline(0, color='grey', lw=0.5, alpha=0.5) if is_pc else None
    ax.axvline(0, color='grey', lw=0.5, alpha=0.5) if is_pc else None
    ax.grid(alpha=0.12, linestyle='--')
    ax.set_axisbelow(True)


def _biplot(ax, df, X_pca, pca, pcx, pcy, title):
    """Scatter + loading arrows for two PCs."""
    vr = pca.explained_variance_ratio_
    _zone_scatter(ax, df,
                  X_pca[:, pcx], X_pca[:, pcy],
                  f'PC{pcx+1} ({vr[pcx]*100:.1f}%)',
                  f'PC{pcy+1} ({vr[pcy]*100:.1f}%)',
                  is_pc=True, legend=True)

    # loading arrows
    loadings = pca.components_
    scale = np.abs(X_pca[:, [pcx, pcy]]).max() * 0.85
    for i, feat in enumerate(PCA_FEATURES):
        lx = loadings[pcx, i] * scale
        ly = loadings[pcy, i] * scale
        if np.sqrt(lx**2 + ly**2) < scale * 0.15:
            continue
        ax.annotate('', xy=(lx, ly), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        ox = 0.15 * scale * np.sign(lx)
        oy = 0.15 * scale * np.sign(ly)
        ax.text(lx + ox, ly + oy, FEATURE_SHORT[feat], fontsize=7,
                ha='center', va='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', fc='white',
                          alpha=0.8, ec='none'))
    ax.set_title(title, fontsize=12, fontweight='bold')


# ── figure functions ──────────────────────────────────────────────────

def plot_fig1_overview(df, X_pca, pca, fname):
    """Scree, loadings heatmap, biplots (2×2)."""
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.30,
                           left=0.08, right=0.95, top=0.94, bottom=0.06)
    vr = pca.explained_variance_ratio_
    n_pcs = len(vr)

    # (a) scree
    ax = fig.add_subplot(gs[0, 0])
    pcs = np.arange(1, n_pcs + 1)
    cumvar = np.cumsum(vr)
    ax.bar(pcs, vr * 100, color='#42A5F5', edgecolor='black', lw=0.5,
           alpha=0.8, label='Individual')
    ax2 = ax.twinx()
    ax2.plot(pcs, cumvar * 100, 'o-', color='#D32F2F', lw=2, ms=6,
             label='Cumulative')
    ax2.axhline(80, color='#D32F2F', ls='--', alpha=0.4)
    ax2.axhline(90, color='#D32F2F', ls=':', alpha=0.4)
    ax2.set_ylabel('Cumulative Variance (%)')
    ax2.set_ylim(0, 105)
    for i in range(min(5, n_pcs)):
        ax.text(i + 1, vr[i] * 100 + 0.8, f'{vr[i]*100:.1f}%',
                ha='center', va='bottom', fontsize=8)
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance (%)')
    ax.set_xticks(pcs)
    ax.set_title('(a) Scree Plot', fontsize=12, fontweight='bold')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, loc='center right')
    ax.grid(axis='y', alpha=0.15, ls='--')

    # (b) loadings heatmap
    ax = fig.add_subplot(gs[0, 1])
    n_show = min(6, n_pcs)
    L = pca.components_.T[:, :n_show]
    im = ax.imshow(L, cmap='RdBu_r', vmin=-0.6, vmax=0.6, aspect='auto')
    ax.set_xticks(range(n_show))
    ax.set_xticklabels([f'PC{i+1}\n({vr[i]*100:.1f}%)' for i in range(n_show)],
                       fontsize=9)
    ax.set_yticks(range(len(PCA_FEATURES)))
    ax.set_yticklabels([FEATURE_SHORT[f] for f in PCA_FEATURES], fontsize=9)
    for r in range(len(PCA_FEATURES)):
        for c in range(n_show):
            v = L[r, c]
            ax.text(c, r, f'{v:.2f}', ha='center', va='center', fontsize=8,
                    color='white' if abs(v) > 0.35 else 'black')
    fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02, label='Loading')
    ax.set_title('(b) PCA Loadings', fontsize=12, fontweight='bold')

    # (c) biplot PC1 vs PC2
    ax = fig.add_subplot(gs[1, 0])
    _biplot(ax, df, X_pca, pca, 0, 1, '(c) PC1 vs PC2')

    # (d) biplot PC1 vs PC3
    ax = fig.add_subplot(gs[1, 1])
    _biplot(ax, df, X_pca, pca, 0, 2, '(d) PC1 vs PC3')

    fig.suptitle(
        f'PCA Diagnostic: Hazard + Antecedent Features '
        f'(SSI-{SSI_WINDOW}, n={len(df)})',
        fontsize=14, fontweight='bold', y=0.97)
    fig.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close()


def plot_fig2_zone_boxes(df, X_pca, pca, kw_results, fname):
    """Box + jitter plots of PC scores by FFMP zone."""
    vr = pca.explained_variance_ratio_
    n_show = min(6, X_pca.shape[1])

    fig, axes = plt.subplots(2, 3, figsize=(16, 10),
                              gridspec_kw={'hspace': 0.45, 'wspace': 0.30})

    for idx in range(n_show):
        ax = axes.flat[idx]
        pc_data, labels, colors = [], [], []
        for zone in FFMP_ZONE_ORDER:
            m = df['ffmp_zone_at_min'] == zone
            vals = X_pca[m.values, idx]
            if len(vals) == 0:
                continue
            pc_data.append(vals)
            labels.append(f'{zone}\n(n={len(vals)})')
            colors.append(FFMP_ZONE_COLORS[zone])

        bp = ax.boxplot(pc_data, labels=labels, patch_artist=True,
                        widths=0.6, showfliers=True,
                        flierprops=dict(marker='.', ms=3, alpha=0.3))
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        for med in bp['medians']:
            med.set(color='black', lw=2)

        rng = np.random.default_rng(42)
        for j, (vals, c) in enumerate(zip(pc_data, colors)):
            jitter = rng.normal(0, 0.08, size=len(vals))
            ax.scatter(j + 1 + jitter, vals, c=c, alpha=0.15, s=8,
                       edgecolors='none', zorder=2)

        row = kw_results[kw_results['col'] == idx].iloc[0]
        p_val, eta = row['p'], row['eta_sq']
        sig = ('***' if p_val < 0.001 else '**' if p_val < 0.01
               else '*' if p_val < 0.05 else 'ns')
        ax.set_title(
            f'PC{idx+1} ({vr[idx]*100:.1f}% var)\n'
            f'KW p={p_val:.2e} {sig},  η²={eta:.3f}',
            fontsize=10, fontweight='bold')
        ax.set_ylabel(f'PC{idx+1} score')
        ax.grid(axis='y', alpha=0.15, ls='--')
        ax.set_axisbelow(True)

    fig.suptitle(
        f'PC Score Distributions by FFMP Zone '
        f'(SSI-{SSI_WINDOW}, n={len(df)})',
        fontsize=14, fontweight='bold', y=0.99)
    fig.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close()


def plot_fig3_cross_dataset(all_data, fname):
    """PC1 vs PC2 for each dataset (1×3)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True,
                              gridspec_kw={'wspace': 0.12})

    for i, (did, (d, Xp, p, _)) in enumerate(all_data.items()):
        ax = axes[i]
        vr = p.explained_variance_ratio_
        for zone in FFMP_ZONE_ORDER:
            m = d['ffmp_zone_at_min'] == zone
            if m.sum() == 0:
                continue
            ax.scatter(Xp[m.values, 0], Xp[m.values, 1],
                       c=FFMP_ZONE_COLORS[zone], alpha=0.5, s=18,
                       edgecolors='black', linewidths=0.3, zorder=3,
                       label=zone if i == 0 else '_')
        n_warn = d['ffmp_zone_at_min'].isin(['Warning', 'Emergency']).sum()
        label = DATASET_LABELS.get(did, did)
        ax.set_title(f'{label}\n(n={len(d)}, Warn+Emerg={n_warn})',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel(f'PC1 ({vr[0]*100:.1f}%)')
        if i == 0:
            ax.set_ylabel(f'PC2 ({vr[1]*100:.1f}%)')
        ax.axhline(0, color='grey', lw=0.5, alpha=0.5)
        ax.axvline(0, color='grey', lw=0.5, alpha=0.5)
        ax.grid(alpha=0.12, ls='--')
        ax.set_axisbelow(True)

    handles = [
        Line2D([0], [0], marker='o', color='none',
               markerfacecolor=FFMP_ZONE_COLORS[z],
               markeredgecolor='black', markeredgewidth=0.4,
               markersize=8, label=z)
        for z in FFMP_ZONE_ORDER
    ]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=10,
               frameon=True, framealpha=0.9, edgecolor='#ccc',
               bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(f'Cross-Dataset PCA Comparison (SSI-{SSI_WINDOW})',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close()


def plot_fig4_pairwise(df, X_pca, pca, kw_results, fname):
    """Scatter matrix of top-4 discriminating PCs (lower triangle)."""
    vr = pca.explained_variance_ratio_
    top_pcs = (kw_results.sort_values('eta_sq', ascending=False)
               .head(4)['col'].values)
    n = len(top_pcs)

    fig, axes = plt.subplots(n - 1, n - 1, figsize=(14, 14),
                              gridspec_kw={'hspace': 0.35, 'wspace': 0.35})

    for row in range(n - 1):
        for col in range(n - 1):
            ax = axes[row, col]
            if col > row:
                ax.set_visible(False)
                continue
            pcy = top_pcs[row + 1]
            pcx = top_pcs[col]
            for zone in FFMP_ZONE_ORDER:
                m = df['ffmp_zone_at_min'] == zone
                if m.sum() == 0:
                    continue
                ax.scatter(X_pca[m.values, pcx], X_pca[m.values, pcy],
                           c=FFMP_ZONE_COLORS[zone], alpha=0.4, s=12,
                           edgecolors='black', linewidths=0.2, zorder=3)
            ax.set_xlabel(f'PC{pcx+1} ({vr[pcx]*100:.1f}%)', fontsize=9)
            ax.set_ylabel(f'PC{pcy+1} ({vr[pcy]*100:.1f}%)', fontsize=9)
            ax.axhline(0, color='grey', lw=0.5, alpha=0.5)
            ax.axvline(0, color='grey', lw=0.5, alpha=0.5)
            ax.grid(alpha=0.12, ls='--')
            ax.set_axisbelow(True)

    handles = [
        Line2D([0], [0], marker='o', color='none',
               markerfacecolor=FFMP_ZONE_COLORS[z],
               markeredgecolor='black', markeredgewidth=0.4,
               markersize=8, label=z)
        for z in FFMP_ZONE_ORDER
    ]
    fig.legend(handles=handles, loc='upper right', ncol=1, fontsize=10,
               frameon=True, framealpha=0.9, edgecolor='#ccc',
               bbox_to_anchor=(0.95, 0.95))
    fig.suptitle(
        f'Pairwise PC Scatter — Top Discriminating PCs '
        f'(SSI-{SSI_WINDOW}, n={len(df)})',
        fontsize=13, fontweight='bold', y=0.98)
    fig.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close()


def plot_fig5_raw_vs_pca(df, X_pca, pca, fname):
    """Top row = best raw feature pairs; bottom row = PC pairs."""
    vr = pca.explained_variance_ratio_
    fig, axes = plt.subplots(2, 3, figsize=(18, 11),
                              gridspec_kw={'hspace': 0.35, 'wspace': 0.28})

    raw_pairs = [
        ('storage_at_start_pct', 'magnitude',
         'Start Storage (%)', 'Magnitude'),
        ('duration_days', 'magnitude',
         'Duration (days)', 'Magnitude'),
        ('storage_at_start_pct', 'severity_rate',
         'Start Storage (%)', 'Severity Rate'),
    ]
    for col, (xf, yf, xl, yl) in enumerate(raw_pairs):
        ax = axes[0, col]
        _zone_scatter(ax, df, xf, yf, xl, yl, legend=(col == 0))
        ax.set_title(f'Raw: {xl} vs {yl}', fontsize=10, fontweight='bold')

    pc_pairs = [(0, 1), (0, 2), (1, 2)]
    for col, (px, py) in enumerate(pc_pairs):
        ax = axes[1, col]
        _zone_scatter(ax, df,
                      X_pca[:, px], X_pca[:, py],
                      f'PC{px+1} ({vr[px]*100:.1f}%)',
                      f'PC{py+1} ({vr[py]*100:.1f}%)',
                      is_pc=True, legend=(col == 0))
        ax.set_title(f'PCA: PC{px+1} vs PC{py+1}',
                     fontsize=10, fontweight='bold')

    fig.suptitle(
        f'Raw Features vs PCA: Outcome Discrimination Comparison\n'
        f'(SSI-{SSI_WINDOW}, n={len(df)})',
        fontsize=14, fontweight='bold', y=0.99)
    fig.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close()


# ── main ──────────────────────────────────────────────────────────────

def main():
    apply_publication_style()
    plt.rcParams.update({'font.size': 10.5, 'axes.labelsize': 11})

    print("=" * 70)
    print("PCA DIAGNOSTIC: Drought Event Feature Reduction")
    print(f"SSI window: {SSI_WINDOW} months  |  Min duration: {MIN_DURATION} days")
    print(f"Features ({len(PCA_FEATURES)} after cyclical month encoding):")
    for f in PCA_FEATURES:
        print(f"    {FEATURE_SHORT[f]}")
    print("=" * 70)

    # ── baseline dataset ──────────────────────────────────────────────
    print("\n--- Loading baseline (stationary_ensemble) ---")
    df = load_events('stationary_ensemble')
    print(f"  {len(df)} events (duration >= {MIN_DURATION} days)")
    zone_counts = df['ffmp_zone_at_min'].value_counts()
    for z in FFMP_ZONE_ORDER:
        print(f"    {z}: {zone_counts.get(z, 0)}")

    X_scaled, X_pca, pca, scaler = fit_pca(df)

    # ── variance summary ──────────────────────────────────────────────
    vr = pca.explained_variance_ratio_
    print(f"\n  Explained variance:")
    for i, v in enumerate(vr):
        cum = sum(vr[:i + 1])
        print(f"    PC{i+1}: {v*100:5.2f}%   (cumulative {cum*100:5.1f}%)")

    # ── loadings summary ──────────────────────────────────────────────
    print(f"\n  Dominant loadings per PC:")
    for i in range(min(5, len(PCA_FEATURES))):
        li = pd.Series(pca.components_[i], index=PCA_FEATURES)
        top = li.abs().sort_values(ascending=False).head(3)
        desc = ', '.join(f'{FEATURE_SHORT[f]}={li[f]:+.2f}' for f in top.index)
        print(f"    PC{i+1}: {desc}")

    # ── Kruskal-Wallis on PCs ─────────────────────────────────────────
    kw_pc = kruskal_wallis_by_zone(df, X_pca)
    print(f"\n  Kruskal-Wallis  (PC score ~ FFMP zone):")
    print(f"    {'PC':<6} {'H':>8} {'p-value':>12} {'η²':>8} {'Sig':>6}")
    print(f"    {'─' * 42}")
    for _, r in kw_pc.iterrows():
        sig = ('***' if r['p'] < 0.001 else '**' if r['p'] < 0.01
               else '*' if r['p'] < 0.05 else 'ns')
        print(f"    PC{int(r['col'])+1:<4} {r['H']:>8.1f} "
              f"{r['p']:>12.2e} {r['eta_sq']:>8.4f} {sig:>6}")

    # ── Kruskal-Wallis on raw features ────────────────────────────────
    raw_vals = df[PCA_FEATURES].fillna(0).values
    kw_raw = kruskal_wallis_by_zone(df, raw_vals)
    print(f"\n  Kruskal-Wallis  (raw feature ~ FFMP zone):")
    print(f"    {'Feature':<20} {'H':>8} {'p-value':>12} {'η²':>8} {'Sig':>6}")
    print(f"    {'─' * 56}")
    for idx, (_, r) in enumerate(kw_raw.iterrows()):
        feat = PCA_FEATURES[idx]
        sig = ('***' if r['p'] < 0.001 else '**' if r['p'] < 0.01
               else '*' if r['p'] < 0.05 else 'ns')
        print(f"    {FEATURE_SHORT[feat]:<20} {r['H']:>8.1f} "
              f"{r['p']:>12.2e} {r['eta_sq']:>8.4f} {sig:>6}")

    # ── figures ───────────────────────────────────────────────────────
    print("\n--- Generating figures ---")

    plot_fig1_overview(df, X_pca, pca,
                       f"{FIG_OUTPUT_DIR}/pca_overview.png")

    plot_fig2_zone_boxes(df, X_pca, pca, kw_pc,
                          f"{FIG_OUTPUT_DIR}/pca_zone_separation.png")

    # cross-dataset
    print("\n--- Cross-dataset PCA ---")
    all_data = {}
    for did in DATASETS:
        d = load_events(did)
        Xs, Xp, p, s = fit_pca(d)
        all_data[did] = (d, Xp, p, s)
        kw = kruskal_wallis_by_zone(d, Xp, n_cols=3)
        label = DATASET_LABELS.get(did, did)
        print(f"\n  {label} (n={len(d)}):")
        for _, r in kw.iterrows():
            sig = ('***' if r['p'] < 0.001 else '**' if r['p'] < 0.01
                   else '*' if r['p'] < 0.05 else 'ns')
            print(f"    PC{int(r['col'])+1}: H={r['H']:.1f}, "
                  f"p={r['p']:.2e}, η²={r['eta_sq']:.4f} {sig}")

    plot_fig3_cross_dataset(all_data,
                             f"{FIG_OUTPUT_DIR}/pca_cross_dataset.png")

    plot_fig4_pairwise(df, X_pca, pca, kw_pc,
                        f"{FIG_OUTPUT_DIR}/pca_pairwise_scatter.png")

    plot_fig5_raw_vs_pca(df, X_pca, pca,
                          f"{FIG_OUTPUT_DIR}/pca_vs_raw_comparison.png")

    # ── summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    best_raw_feat = max(PCA_FEATURES, key=lambda f: raw_feature_eta(df, f))
    best_raw_eta = raw_feature_eta(df, best_raw_feat)
    best_pc = kw_pc.sort_values('eta_sq', ascending=False).iloc[0]

    print(f"\n  Best raw feature:  {FEATURE_SHORT[best_raw_feat]}"
          f"  (η² = {best_raw_eta:.4f})")
    print(f"  Best PC:           PC{int(best_pc['col'])+1}"
          f"  (η² = {best_pc['eta_sq']:.4f})")

    if best_pc['eta_sq'] > best_raw_eta:
        pct = (best_pc['eta_sq'] - best_raw_eta) / best_raw_eta * 100
        print(f"  → PCA improves best single-feature discrimination "
              f"by {pct:.1f}%")
    else:
        print(f"  → Raw features already provide best "
              f"single-feature discrimination")

    cumvar = np.cumsum(vr)
    n80 = int(np.searchsorted(cumvar, 0.80) + 1)
    n90 = int(np.searchsorted(cumvar, 0.90) + 1)
    print(f"\n  PCs for 80% variance: {n80}/{len(PCA_FEATURES)}")
    print(f"  PCs for 90% variance: {n90}/{len(PCA_FEATURES)}")
    n_sig = (kw_pc['p'] < 0.05).sum()
    print(f"  Significant PCs (p < 0.05): {n_sig}/{len(kw_pc)}")

    print("\n" + "=" * 70)
    print("DONE — figures saved to:")
    print(f"  {FIG_OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == '__main__':
    main()
