"""
SI: Drought Event Discrimination Analysis

Systematic analysis of which features discriminate drought events by
storage outcome severity. Uses expanded event metrics including antecedent
conditions, hazard characteristics, system actions, and outcomes.

Outputs:
  - Correlation heatmap (antecedent/hazard/action features vs outcomes)
  - Decision tree feature importance (CART)
  - K-Means cluster profiles
  - Key interaction scatter plots
  - All saved as SI figures

Usage:
    python SI_drought_event_discrimination.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

from methods.config import ROOT_DIR, FIG_DIR
from methods.plotting.styles import (
    DATASET_COLORS, DATASET_LABELS,
    FFMP_ZONE_COLORS,
    FONTSIZE_LABEL, FONTSIZE_MEDIUM,
    DPI_HIGH, apply_publication_style,
)

FIG_OUTPUT_DIR = f"{FIG_DIR}/SI_drought_discrimination"
os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

SSI_WINDOW = 3
MIN_DURATION = 30
DATASETS = ['stationary_ensemble', 'climate_adjusted_low', 'climate_adjusted_high']

# Feature groupings (causal stages)
ANTECEDENT = ['storage_at_start_pct', 'start_month']
HAZARD = ['severity', 'magnitude', 'duration_days', 'severity_rate',
          'peak_severity_month', 'total_inflow_mg']
ACTION = ['contribution_ratio', 'nyc_diversion_inflow_ratio']
OUTCOME = ['event_min_storage_pct', 'storage_drawdown_pct',
           'nyc_diversion_sat_ratio',
           'total_nyc_shortage_mg', 'nyc_shortage_pct',
           'max_consec_montague_days', 'total_montague_shortage_mg',
           'max_consec_trenton_days', 'total_trenton_shortage_mg']

VALID_PREDICTORS = ANTECEDENT + HAZARD + ACTION

FFMP_ZONE_ORDER = ['Normal', 'Watch', 'Warning', 'Emergency']

# Short labels for plots
FEATURE_SHORT = {
    'storage_at_start_pct': 'Start Storage',
    'start_month': 'Start Month',
    'severity': 'Severity',
    'magnitude': 'Magnitude',
    'duration_days': 'Duration',
    'severity_rate': 'Severity Rate',
    'peak_severity_month': 'Peak Month',
    'total_inflow_mg': 'Total Inflow',
    'contribution_ratio': 'Contrib Ratio',
    'nyc_diversion_inflow_ratio': 'Div/Inflow',
    'nyc_diversion_sat_ratio': 'Demand Sat',
}


def load_events(dataset_id):
    """Load and clean event metrics."""
    df = pd.read_csv(
        f'{ROOT_DIR}/pywrdrb/event_metrics/{dataset_id}_ssi{SSI_WINDOW}_event_metrics.csv'
    )
    df = df[df['duration_days'] >= MIN_DURATION].copy()
    df['severity'] = df['severity'].abs()
    df['magnitude'] = df['magnitude'].abs()
    return df


def classify_outcome(df):
    """Classify events using dynamic FFMP zone at min storage date."""
    df = df.copy()
    # FFMP zone-based classification (dynamic seasonal thresholds)
    df['is_severe'] = (df['ffmp_zone_at_min'] == 'Emergency').astype(int)
    df['is_stressed'] = df['ffmp_zone_at_min'].isin(['Emergency', 'Warning']).astype(int)
    df['outcome'] = df['ffmp_zone_at_min']
    return df


def plot_correlation_heatmap(df, ax, title=''):
    """Plot correlation of predictors with outcome metrics."""
    predictors = VALID_PREDICTORS
    targets = ['event_min_storage_pct', 'storage_drawdown_pct',
               'nyc_shortage_pct', 'total_montague_shortage_mg']
    target_short = ['Min Storage', 'Drawdown', 'NYC Shortage %', 'Montague Shortage']

    corr = pd.DataFrame(index=predictors, columns=targets, dtype=float)
    for p in predictors:
        for t in targets:
            corr.loc[p, t] = df[p].corr(df[t])

    im = ax.imshow(corr.values.astype(float), cmap='RdBu_r', vmin=-0.7, vmax=0.7, aspect='auto')

    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(target_short, fontsize=8, rotation=30, ha='right')
    ax.set_yticks(range(len(predictors)))
    ax.set_yticklabels([FEATURE_SHORT.get(p, p) for p in predictors], fontsize=8)

    # Annotate cells
    for i in range(len(predictors)):
        for j in range(len(targets)):
            val = corr.iloc[i, j]
            color = 'white' if abs(val) > 0.4 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=7, color=color)

    # Stage separators
    ax.axhline(len(ANTECEDENT) - 0.5, color='black', linewidth=1.5)
    ax.axhline(len(ANTECEDENT) + len(HAZARD) - 0.5, color='black', linewidth=1.5)

    ax.set_title(title, fontsize=11, fontweight='bold')
    return im


def run_decision_trees(df):
    """Run CART analysis and return results as text."""
    X = df[VALID_PREDICTORS].fillna(0).values
    feature_names = [FEATURE_SHORT.get(f, f) for f in VALID_PREDICTORS]
    results = {}

    for label, target in [('FFMP Emergency', 'is_severe'),
                           ('FFMP Warning+Emergency', 'is_stressed')]:
        y = df[target].values
        n_pos = y.sum()
        if n_pos < 3:
            results[label] = {'text': f'Too few positive cases ({n_pos})',
                              'importance': {}, 'accuracy': 0}
            continue

        dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
        dt.fit(X, y)
        acc = dt.score(X, y)
        tree_text = export_text(dt, feature_names=feature_names, decimals=1)

        imp = pd.Series(dt.feature_importances_, index=VALID_PREDICTORS)
        imp = imp.sort_values(ascending=False)

        # Random Forest for more stable importance
        rf = RandomForestClassifier(n_estimators=200, max_depth=4,
                                     min_samples_leaf=5, random_state=42)
        rf.fit(X, y)
        rf_imp = pd.Series(rf.feature_importances_, index=VALID_PREDICTORS)
        rf_imp = rf_imp.sort_values(ascending=False)

        results[label] = {
            'text': tree_text,
            'accuracy': acc,
            'n_pos': n_pos,
            'cart_importance': imp,
            'rf_importance': rf_imp,
            'tree': dt,
            'feature_names': feature_names,
        }
    return results


def plot_feature_importance(dt_results, ax, title=''):
    """Plot feature importance comparison (CART vs RF)."""
    for i, (label, res) in enumerate(dt_results.items()):
        if 'rf_importance' not in res:
            continue
        rf_imp = res['rf_importance']
        top = rf_imp.head(8)
        y_pos = np.arange(len(top))
        color = '#D32F2F' if 'Severe' in label else '#FF9800'
        offset = 0.2 if i == 0 else -0.2

        bars = ax.barh(y_pos + offset, top.values, height=0.35, color=color,
                       alpha=0.7, edgecolor='black', linewidth=0.3,
                       label=f'{label} (n={res["n_pos"]}, acc={res["accuracy"]:.0%})')

        ax.set_yticks(y_pos)
        ax.set_yticklabels([FEATURE_SHORT.get(f, f) for f in top.index], fontsize=8)

    ax.set_xlabel('Random Forest Feature Importance', fontsize=10)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(axis='x', alpha=0.15, linestyle='--')


def plot_interaction_scatter(df, ax):
    """Key interaction: start month × start storage → min storage outcome."""
    sizes = 12 + 8 * df['magnitude']
    for zone in FFMP_ZONE_ORDER:
        m = df['ffmp_zone_at_min'] == zone
        if m.sum() == 0:
            continue
        ax.scatter(df.loc[m, 'start_month'], df.loc[m, 'storage_at_start_pct'],
                   s=sizes[m], c=FFMP_ZONE_COLORS[zone], alpha=0.7,
                   edgecolors='black', linewidths=0.4, zorder=3, label=zone)

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'], fontsize=8)
    ax.set_xlabel('Drought Start Month')
    ax.set_ylabel('Storage at Start (%)')
    ax.set_title('(c) Start Month x Start Storage', fontsize=11, fontweight='bold')
    ax.legend(fontsize=7.5, framealpha=0.9, edgecolor='#ccc', loc='lower left')
    ax.grid(alpha=0.12, linestyle='--')
    ax.set_axisbelow(True)


def plot_peak_vs_outcome(df, ax):
    """Peak severity month vs min storage, sized by magnitude, colored by FFMP zone."""
    sizes = 12 + 8 * df['magnitude']
    for zone in FFMP_ZONE_ORDER:
        m = df['ffmp_zone_at_min'] == zone
        if m.sum() == 0:
            continue
        ax.scatter(df.loc[m, 'peak_severity_month'], df.loc[m, 'event_min_storage_pct'],
                   s=sizes[m], c=FFMP_ZONE_COLORS[zone], alpha=0.7,
                   edgecolors='black', linewidths=0.4, zorder=3, label=zone)

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'], fontsize=8)
    ax.set_xlabel('Peak Severity Month')
    ax.set_ylabel('Min Storage (%)')
    ax.set_title('(d) Peak Timing \u2192 Outcome', fontsize=11, fontweight='bold')
    ax.legend(fontsize=7.5, framealpha=0.9, edgecolor='#ccc', loc='lower left')
    ax.grid(alpha=0.12, linestyle='--')
    ax.set_axisbelow(True)


def main():
    apply_publication_style()
    plt.rcParams.update({'font.size': 10.5, 'axes.labelsize': 11})

    # Load baseline data
    print(f"Loading SSI-{SSI_WINDOW} events (duration >= {MIN_DURATION} days)...")
    df = load_events('stationary_ensemble')
    df = classify_outcome(df)
    zone_counts = df['ffmp_zone_at_min'].value_counts()
    print(f"  {len(df)} events — FFMP zone at min storage:")
    for z in ['Normal', 'Watch', 'Warning', 'Emergency']:
        print(f"    {z}: {zone_counts.get(z, 0)}")

    # ================================================================
    # FIGURE 1: Correlation + Feature Importance (2-panel)
    # ================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                     gridspec_kw={'width_ratios': [1.2, 1], 'wspace': 0.35})

    im = plot_correlation_heatmap(df, ax1, title='(a) Feature-Outcome Correlations')
    fig.colorbar(im, ax=ax1, shrink=0.8, pad=0.02, label='Pearson r')

    dt_results = run_decision_trees(df)
    plot_feature_importance(dt_results, ax2, title='(b) Feature Importance (Random Forest)')

    fig.suptitle(f'Baseline Ensemble: SSI-{SSI_WINDOW} Drought Event Discrimination (n={len(df)})',
                 fontsize=13, fontweight='bold', y=1.01)
    fname = f"{FIG_OUTPUT_DIR}/discrimination_correlation_importance.png"
    fig.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close()

    # Print decision tree rules
    for label, res in dt_results.items():
        print(f"\n  === CART: {label} (accuracy={res.get('accuracy',0):.1%}) ===")
        print(res.get('text', 'N/A'))

    # ================================================================
    # FIGURE 1b: Decision Tree Visualizations
    # ================================================================
    tree_entries = [(label, res) for label, res in dt_results.items() if 'tree' in res]
    if tree_entries:
        fig, axes = plt.subplots(len(tree_entries), 1,
                                  figsize=(16, 5 * len(tree_entries)),
                                  gridspec_kw={'hspace': 0.4})
        if len(tree_entries) == 1:
            axes = [axes]

        for ax, (label, res) in zip(axes, tree_entries):
            dt = res['tree']
            feat_names = res['feature_names']
            n_pos = res['n_pos']
            n_total = len(df)
            acc = res['accuracy']

            plot_tree(dt, feature_names=feat_names,
                      class_names=['Pass', 'Fail'],
                      filled=True, rounded=True, fontsize=9,
                      impurity=False, proportion=True, ax=ax)
            ax.set_title(f'{label}  —  accuracy={acc:.1%}, '
                         f'n_fail={n_pos}/{n_total}',
                         fontsize=12, fontweight='bold')

        fname = f"{FIG_OUTPUT_DIR}/discrimination_decision_trees.png"
        fig.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
        print(f"  Saved: {fname}")
        plt.close()

    # ================================================================
    # FIGURE 2: Key interactions (2x2)
    # ================================================================
    fig = plt.figure(figsize=(13, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.32, wspace=0.30,
                            left=0.08, right=0.97, top=0.94, bottom=0.06)

    # (a) Start storage vs magnitude, color=FFMP zone, size=duration, shape=peak season
    ax = fig.add_subplot(gs[0, 0])
    peak_markers = {
        'Winter peak': ('D', [12,1,2]),
        'Spring peak': ('^', [3,4,5]),
        'Summer peak': ('o', [6,7,8]),
        'Fall peak': ('s', [9,10,11]),
    }
    sizes_a = 12 + 0.12 * df['duration_days']
    for label, (marker, months) in peak_markers.items():
        for zone in FFMP_ZONE_ORDER:
            m = df['peak_severity_month'].isin(months) & (df['ffmp_zone_at_min'] == zone)
            if m.sum() == 0:
                continue
            ax.scatter(df.loc[m, 'storage_at_start_pct'], df.loc[m, 'magnitude'],
                       s=sizes_a[m], c=FFMP_ZONE_COLORS[zone], alpha=0.7, marker=marker,
                       edgecolors='black', linewidths=0.4, zorder=3)
    # Season shape legend
    for label, (marker, _) in peak_markers.items():
        ax.scatter([], [], marker=marker, c='grey', s=30, edgecolors='black',
                   linewidths=0.3, label=label)
    for dur, lab in [(90, '90d'), (365, '1yr'), (1000, '3yr')]:
        s = 12 + 0.12 * dur
        ax.scatter([], [], s=s, marker='o', c='grey', alpha=0.4, edgecolors='black',
                   linewidths=0.3, label=lab)
    ax.legend(fontsize=6.5, framealpha=0.9, edgecolor='#ccc', ncol=2, loc='upper right',
              title='Peak Season / Duration', title_fontsize=7)
    ax.set_xlabel('Storage at Drought Start (%)')
    ax.set_ylabel(f'Drought Magnitude (SSI-{SSI_WINDOW})')
    ax.grid(alpha=0.12, linestyle='--')
    ax.set_axisbelow(True)
    ax.text(0.02, 0.97, '(a) Antecedent \u00d7 Hazard', transform=ax.transAxes,
            fontsize=10, va='top', fontweight='bold')

    # (b) Duration vs magnitude, color=FFMP zone, size=contribution
    ax = fig.add_subplot(gs[0, 1])
    contrib_pct = (df['contribution_ratio'] * 100).clip(0, 100)
    sizes_b = np.where(contrib_pct < 5, 10, 10 + 3.5 * (contrib_pct - 5))
    for zone in FFMP_ZONE_ORDER:
        m = df['ffmp_zone_at_min'] == zone
        if m.sum() == 0:
            continue
        ax.scatter(df.loc[m, 'duration_days'], df.loc[m, 'magnitude'],
                   s=sizes_b[m], c=FFMP_ZONE_COLORS[zone], alpha=0.7,
                   edgecolors='black', linewidths=0.4, zorder=3, label=zone)
    for pct, lab in [(5, '<5%'), (20, '20%'), (50, '50%')]:
        s = 10 if pct < 5 else 10 + 3.5 * (pct - 5)
        ax.scatter([], [], s=s, c='grey', alpha=0.4, edgecolors='black',
                   linewidths=0.3, label=f'{lab} contrib')
    ax.legend(fontsize=7, framealpha=0.9, edgecolor='#ccc', loc='upper left', ncol=2)
    ax.set_xlabel('Duration (days)')
    ax.set_ylabel(f'Drought Magnitude (SSI-{SSI_WINDOW})')
    ax.grid(alpha=0.12, linestyle='--')
    ax.set_axisbelow(True)
    ax.text(0.35, 0.04, '(b) Duration \u00d7 Magnitude', transform=ax.transAxes,
            fontsize=10, va='bottom', fontweight='bold')

    # (c) Start month × start storage → outcome
    ax = fig.add_subplot(gs[1, 0])
    plot_interaction_scatter(df, ax)

    # (d) Peak timing → outcome
    ax = fig.add_subplot(gs[1, 1])
    plot_peak_vs_outcome(df, ax)
    for mag, lab in [(3, '3'), (10, '10'), (25, '25')]:
        s = 12 + 8 * mag
        ax.scatter([], [], s=s, c='grey', alpha=0.4, edgecolors='black',
                   linewidths=0.3, label=f'Mag={lab}')
    ax.legend(fontsize=7, framealpha=0.9, edgecolor='#ccc', loc='lower left')

    fig.suptitle(f'Baseline Ensemble: Drought Event Discrimination (SSI-{SSI_WINDOW}, n={len(df)})',
                 fontsize=13, fontweight='bold', y=0.97)
    fname = f"{FIG_OUTPUT_DIR}/discrimination_interactions.png"
    fig.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close()

    # ================================================================
    # FIGURE 3: Cross-scenario comparison
    # ================================================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True,
                              gridspec_kw={'wspace': 0.08})

    for i, did in enumerate(DATASETS):
        ax = axes[i]
        d = load_events(did)
        d = classify_outcome(d)
        label = DATASET_LABELS[did]

        sizes = 12 + 0.12 * d['duration_days']
        for zone in FFMP_ZONE_ORDER:
            m = d['ffmp_zone_at_min'] == zone
            if m.sum() == 0:
                continue
            ax.scatter(d.loc[m, 'storage_at_start_pct'], d.loc[m, 'magnitude'],
                       s=sizes[m], c=FFMP_ZONE_COLORS[zone], alpha=0.7,
                       edgecolors='black', linewidths=0.4, zorder=3,
                       label=zone if i == 0 else '_')

        n_warn = d['ffmp_zone_at_min'].isin(['Warning', 'Emergency']).sum()
        n_watch = d['ffmp_zone_at_min'].isin(['Watch', 'Warning', 'Emergency']).sum()
        ax.set_title(f'{label}\n(n={len(d)}, Watch+={n_watch}, Warn+={n_warn})',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Storage at Start (%)')
        if i == 0:
            ax.set_ylabel(f'Drought Magnitude (SSI-{SSI_WINDOW})')
        ax.grid(alpha=0.12, linestyle='--')
        ax.set_axisbelow(True)

    # Shared FFMP zone legend
    zone_handles = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor=FFMP_ZONE_COLORS[z],
               markeredgecolor='black', markeredgewidth=0.4, markersize=8, label=z)
        for z in FFMP_ZONE_ORDER
    ]
    fig.legend(handles=zone_handles, loc='lower center', ncol=4,
               fontsize=9, frameon=True, framealpha=0.9, edgecolor='#ccc',
               bbox_to_anchor=(0.5, -0.05))
    fname = f"{FIG_OUTPUT_DIR}/discrimination_cross_scenario.png"
    fig.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close()

    # ================================================================
    # Print summary statistics
    # ================================================================
    print("\n" + "=" * 70)
    print("CROSS-SCENARIO SUMMARY")
    print("=" * 70)
    for did in DATASETS:
        d = load_events(did)
        d = classify_outcome(d)
        label = DATASET_LABELS[did]
        n = len(d)
        n_sev = d['is_severe'].sum()
        n_str = d['is_stressed'].sum()
        print(f"\n  {label} (n={n}):")
        print(f"    Severe: {n_sev} ({100*n_sev/n:.1f}%)")
        print(f"    Stressed: {n_str} ({100*n_str/n:.1f}%)")
        print(f"    Avg min storage: {d['event_min_storage_pct'].mean():.1f}%")
        print(f"    Avg contribution ratio: {d['contribution_ratio'].mean():.3f}")
        print(f"    Avg NYC shortage: {d['nyc_shortage_pct'].mean():.2f}%")
        print(f"    Events with Montague shortage: {(d['max_consec_montague_days']>0).sum()}")
        print(f"    Events with Trenton shortage: {(d['max_consec_trenton_days']>0).sum()}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
