"""
Preliminary drought-storyline figure (k=4, 'full' hazard feature set, all
droughts pooled across the three ensembles).

Three panels:
  (a) Storyline fingerprints — standardized mean hazard profile per cluster.
  (b) H1: impact per storyline — % reaching FFMP Emergency + mean min storage.
  (c) H2: within-ensemble storyline shares (how the drought-type mix shifts
      across climate scenarios).

Reuses build_dataset() from cluster_diagnostics so the clustering is identical
to the diagnostics. Exploratory; see FINDINGS.md for interpretation/caveats.
Run via S7_run_SI_scripts.sh.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from methods.config import FIG_DIR
from methods.plotting.styles import (
    apply_publication_style, save_fig, DPI_HIGH, DATASET_LABELS_SHORT,
    DATASET_COLORS,
)
from cluster_diagnostics import build_dataset, FEATURE_SETS, DATASETS, RANDOM_STATE

K = 4
FEATS = FEATURE_SETS['full']
FIG_OUT = os.path.join(FIG_DIR, 'SI_drought_storylines')

# readable feature labels for the fingerprint panel
FEAT_LABELS = {
    'log_duration': 'duration', 'log_magnitude': 'magnitude',
    'severity_abs': 'peak sev.', 'avg_severity_abs': 'mean intens.',
    'onset_rate': 'onset rate', 'recovery_rate': 'recovery rate',
    'onset_sin': 'season (sin)', 'onset_cos': 'season (cos)',
    'prior_3m_surplus': 'antecedent wet', 'peakedness': 'peakedness',
}


def _auto_name(row):
    """Short descriptor from a cluster's hazard profile (original units)."""
    bits = []
    bits.append('long' if row['duration'] > 11 else
                ('short' if row['duration'] < 6 else 'mod'))
    bits.append('severe' if row['severity_abs'] > 1.9 else 'mild')
    if row['onset_rate'] > 1.4:
        bits.append('fast-onset')
    elif row['onset_rate'] < 0.8:
        bits.append('slow-onset')
    season = {12: 'win', 1: 'win', 2: 'win', 3: 'spr', 4: 'spr', 5: 'spr',
              6: 'sum', 7: 'sum', 8: 'sum', 9: 'aut', 10: 'aut', 11: 'aut'}
    bits.append(season.get(int(round(row['start_month'])), ''))
    return ' '.join(b for b in bits if b)


def main():
    apply_publication_style()
    os.makedirs(FIG_OUT, exist_ok=True)
    print("Preliminary storyline figure: k=%d, features=%s" % (K, FEATS))

    df, _ = build_dataset()
    sub = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATS).reset_index(drop=True)
    Xs = StandardScaler().fit_transform(sub[FEATS].values)
    km = KMeans(n_clusters=K, n_init=10, random_state=RANDOM_STATE).fit(Xs)
    sub['cluster'] = km.labels_
    Xs_df = pd.DataFrame(Xs, columns=FEATS)
    Xs_df['cluster'] = km.labels_

    # order clusters by % Emergency (worst first) for stable, meaningful labels
    emrg = sub.groupby('cluster').apply(
        lambda g: 100.0 * (g['ffmp_zone_at_min'] == 'Emergency').mean())
    order = list(emrg.sort_values(ascending=False).index)
    remap = {old: new for new, old in enumerate(order)}
    sub['cl'] = sub['cluster'].map(remap)
    Xs_df['cl'] = Xs_df['cluster'].map(remap)
    K_ids = list(range(K))

    # names
    prof = sub.groupby('cl')[['duration', 'severity_abs', 'onset_rate',
                              'start_month']].mean()
    names = {c: f"S{c+1}: {_auto_name(prof.loc[c])}" for c in K_ids}
    for c in K_ids:
        print(f"  {names[c]}  (n={int((sub['cl']==c).sum())}, "
              f"%Emerg={emrg[order[c]]:.1f})")

    # ---- figure ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2),
                             gridspec_kw={'width_ratios': [1.25, 1.0, 1.0],
                                          'wspace': 0.32})

    # (a) fingerprint heatmap: standardized centroid per feature
    ax = axes[0]
    cent = Xs_df.groupby('cl')[FEATS].mean().loc[K_ids]
    im = ax.imshow(cent.values, aspect='auto', cmap='RdBu_r',
                   vmin=-1.5, vmax=1.5)
    ax.set_xticks(range(len(FEATS)))
    ax.set_xticklabels([FEAT_LABELS.get(f, f) for f in FEATS],
                       rotation=45, ha='right', fontsize=9)
    ax.set_yticks(K_ids)
    ax.set_yticklabels([names[c] for c in K_ids], fontsize=9)
    ax.set_title('(a) Storyline hazard fingerprints\n(standardized means)',
                 fontsize=11, fontweight='bold')
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('z-score', fontsize=9)

    # (b) H1 impact: % Emergency bars + min-storage annotation
    ax = axes[1]
    em = [emrg[order[c]] for c in K_ids]
    minst = sub.groupby('cl')['event_min_storage_pct'].mean().loc[K_ids]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, K))
    ax.bar(K_ids, em, color=colors, edgecolor='k', linewidth=0.6)
    for c in K_ids:
        ax.text(c, em[c] + 0.15, f"min stor.\n{minst[c]:.0f}%", ha='center',
                va='bottom', fontsize=8)
    ax.set_xticks(K_ids)
    ax.set_xticklabels([f"S{c+1}" for c in K_ids])
    ax.set_ylabel('% events reaching FFMP Emergency')
    ax.set_title('(b) H1: impact differs by storyline', fontsize=11,
                 fontweight='bold')
    ax.set_ylim(0, max(em) * 1.25 + 0.5)

    # (c) H2 within-ensemble storyline shares
    ax = axes[2]
    ct = pd.crosstab(sub['cl'], sub['dataset'])[
        [d for d in DATASETS if d in sub['dataset'].unique()]]
    share = ct.div(ct.sum(axis=0), axis=1) * 100.0
    x = np.arange(len(DATASETS))
    bottom = np.zeros(len(DATASETS))
    for c in K_ids:
        vals = [share.loc[c, d] for d in DATASETS]
        ax.bar(x, vals, bottom=bottom, color=colors[c], edgecolor='k',
               linewidth=0.5, label=names[c])
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS_SHORT.get(d, d) for d in DATASETS],
                       rotation=12, ha='right', fontsize=9)
    ax.set_ylabel('% of ensemble drought events')
    ax.set_title('(c) H2: storyline mix shifts with climate', fontsize=11,
                 fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.12),
              ncol=2, frameon=False)

    fig.suptitle('Preliminary drought storylines (SSI-3, k=4, hazard features; '
                 'continuum partitioned — see FINDINGS.md)',
                 fontsize=13, fontweight='bold', y=1.02)
    save_fig(fig, os.path.join(FIG_OUT, 'SI_drought_storylines_preliminary'),
             dpi=DPI_HIGH)
    plt.close(fig)


if __name__ == '__main__':
    main()
