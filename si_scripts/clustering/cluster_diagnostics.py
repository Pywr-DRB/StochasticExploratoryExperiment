"""
Drought-event clustering diagnostics (exploratory; NO figures).

Decides whether SSI-3 drought events, pooled across the three climate
ensembles, form distinct / stable / interpretable clusters ("storylines")
that (H1) differ in water-supply impact and (H2) differ in frequency across
climate scenarios.

Method discipline (see PLAN.md, LITERATURE_NOTES.md, and the saved feedback
memory 'clustering-metric-discipline'):
  * Cluster ONLY on HAZARD characteristics of the drought.
  * Keep metric CATEGORIES strictly separate:
       HAZARD (clustering inputs) | ANTECEDENT | OPERATIONAL | OUTCOME
    Antecedent/operational/outcome are EXTERNAL — used to characterize clusters
    after fitting, never to define them (avoids circularity).
  * Cluster on standardized ABSOLUTE metrics for interpretability. PCA is
    reported only as a redundancy DIAGNOSTIC; it is NOT used as a clustering
    input. (We do NOT use PRIM or any input-space rule induction.)

Hazard DYNAMICS features (onset_rate, recovery_rate, onset season) are derived
from the existing drought_events CSVs (max_severity_date, recovery_period) and
validated in assess_dynamics_metrics.py — no pipeline rerun required. Impact /
operational metrics are merged in from the event_metrics CSVs.

Run via S7_run_SI_scripts.sh (sbatch). Prints a full report to stdout (line-
buffered) and writes a summary JSON to outputs/<config>/data/clustering/.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score,
)
from scipy.stats import chi2_contingency

from methods.config import (
    CONFIG_DIR, DROUGHT_METRICS_DIR, GRID_N_BINS, N_YEARS,
    MIN_COUNT_PER_BIN as MIN_COUNT,
    FOCAL_FRAC_THRESH, FOCAL_RP_THRESH_YEARS, FOCAL_WORST_STORAGE_THRESH,
    DATASET_CONFIGS,
)
from methods.load import load_event_metrics
from methods.return_period import (
    compute_return_period_grid_exceedance as compute_return_period_grid,
)
from methods.plotting.heatmap import (
    make_shared_edges_logmag, assign_grid_bins,
    compute_emergency_grid, compute_min_storage_grid,
    identify_focal_region,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
SSI_WINDOW = 3
DATASETS = list(DATASET_CONFIGS.keys())
RANDOM_STATE = 0
K_RANGE = list(range(2, 11))
SWEEP_SAMPLE = 40000
SIL_SAMPLE = 5000
WARD_SAMPLE = 6000
N_BOOT = 15
CANDIDATE_KS = [3, 4, 5, 6]

OUT_DIR = os.path.join(CONFIG_DIR, 'data', 'clustering')

# ---- metric CATEGORIES (kept strictly separate) ----
# HAZARD clustering-input groups (built up incrementally to show the value of
# the new dynamics/seasonality axes):
HAZARD_SIZE = ['log_duration', 'log_magnitude', 'severity_abs', 'avg_severity_abs']
HAZARD_RATES = ['onset_rate', 'recovery_rate']         # NEW, size-independent
HAZARD_SEASON = ['onset_sin', 'onset_cos']             # NEW, orthogonal timing
HAZARD_SHAPE = ['peakedness']
HAZARD_ANTECEDENT = ['prior_3m_surplus']               # exposure (antecedent
                                                       # wetness) — per user, a
                                                       # hazard/exposure feature
FEATURE_SETS = {
    'size_only':         HAZARD_SIZE + HAZARD_SHAPE,
    'size+rates':        HAZARD_SIZE + HAZARD_RATES + HAZARD_SHAPE,
    'full':              (HAZARD_SIZE + HAZARD_RATES + HAZARD_SEASON
                          + HAZARD_ANTECEDENT + HAZARD_SHAPE),
}

# EXTERNAL categories (characterization only; never clustered on)
OPERATIONAL = ['total_nyc_diversion_mg', 'total_nyc_contribution_mg',
               'contribution_ratio']
OUTCOME = ['event_min_storage_pct', 'storage_drawdown_pct', 'nyc_shortage_pct',
           'total_montague_shortage_mg', 'max_consec_montague_days']

# original-unit hazard columns for readable cluster profiles
HAZARD_DESCRIBE = ['duration', 'severity_abs', 'magnitude_abs',
                   'avg_severity_abs', 'onset_rate', 'recovery_rate',
                   'time_to_peak_m', 'recovery_period', 'peakedness',
                   'prior_3m_surplus', 'start_month']


# --------------------------------------------------------------------------- #
# Data assembly
# --------------------------------------------------------------------------- #
def _derive_hazard_dynamics(de):
    """Derive validated hazard-dynamics features from a drought_events frame."""
    d = de.copy()
    d['severity_abs'] = d['severity'].abs()
    d['avg_severity_abs'] = d['avg_severity'].abs()
    d['magnitude_abs'] = d['magnitude'].abs()
    d['log_duration'] = np.log1p(d['duration'])
    d['log_magnitude'] = np.log1p(d['magnitude_abs'])
    d['time_to_peak_m'] = d['duration'] - d['recovery_period']
    d['onset_rate'] = d['severity_abs'] / d['time_to_peak_m'].clip(lower=0.5)
    d['recovery_rate'] = d['severity_abs'] / d['recovery_period'].clip(lower=0.5)
    d['peakedness'] = d['severity_abs'] / d['avg_severity_abs'].replace(0, np.nan)
    d['start_month'] = d['start'].dt.month
    d['onset_sin'] = np.sin(2 * np.pi * d['start_month'] / 12.0)
    d['onset_cos'] = np.cos(2 * np.pi * d['start_month'] / 12.0)
    return d


def build_dataset():
    """Merge hazard (+dynamics, antecedent) from drought_events with
    impact/operational metrics from event_metrics. Returns pooled DataFrame
    plus the per-ensemble event_metrics frames (for the focal-region grid)."""
    pooled = []
    em_frames = {}
    print("\nAssembling dataset (drought_events hazard + event_metrics impact):")
    for ds in DATASETS:
        # hazard + antecedent (+dynamics)
        f = os.path.join(DROUGHT_METRICS_DIR,
                         f"{ds}_ssi{SSI_WINDOW}_drought_events.csv")
        de = pd.read_csv(f, parse_dates=['start', 'end', 'max_severity_date'])
        de['realization_id'] = de['realization_id'].astype(int)
        de = _derive_hazard_dynamics(de)

        # impact + operational
        em = load_event_metrics(ds, SSI_WINDOW)
        em = em.copy()
        em['realization_id'] = em['realization_id'].astype(int)
        em['start'] = pd.to_datetime(em['start'])
        em_frames[ds] = em

        keep_em = (['realization_id', 'start'] + OPERATIONAL + OUTCOME
                   + ['ffmp_zone_at_min', 'severity', 'magnitude'])
        keep_em = [c for c in keep_em if c in em.columns]
        merged = de.merge(em[keep_em], on=['realization_id', 'start'],
                          how='inner', suffixes=('', '_em'))
        rate = 100.0 * len(merged) / len(em)
        print(f"  {ds}: drought_events={len(de):,}  event_metrics={len(em):,}  "
              f"merged={len(merged):,}  ({rate:.1f}% of event_metrics matched)")
        merged['dataset'] = ds
        pooled.append(merged)
    return pd.concat(pooled, ignore_index=True), em_frames


def focal_mask_for(df, em_frames):
    """Boolean mask (aligned to df) flagging focal-region events, using the
    Fig9/Fig10/SI21 identification on the event_metrics frames."""
    sev_edges, mag_edges, _, _ = make_shared_edges_logmag(
        em_frames, DATASETS, n_bins=GRID_N_BINS)
    T_W, frac, mn = {}, {}, {}
    for ds in DATASETS:
        _, _, T_W[ds], _ = compute_return_period_grid(
            em_frames[ds], sev_edges, mag_edges, N_YEARS, min_count=MIN_COUNT)
        frac[ds], _ = compute_emergency_grid(
            em_frames[ds], sev_edges, mag_edges, min_count=MIN_COUNT)
        mn[ds], _ = compute_min_storage_grid(
            em_frames[ds], sev_edges, mag_edges, min_count=MIN_COUNT)
    focal_cells = identify_focal_region(
        T_W, frac, mn, DATASETS, rp_thresh_years=FOCAL_RP_THRESH_YEARS,
        frac_thresh=FOCAL_FRAC_THRESH, storage_thresh=FOCAL_WORST_STORAGE_THRESH)
    # bin the merged df by its (abs) severity/magnitude
    tmp = df[['severity_abs', 'magnitude_abs']].rename(
        columns={'severity_abs': 'severity', 'magnitude_abs': 'magnitude'})
    binned = assign_grid_bins(tmp, sev_edges, mag_edges)
    mask = pd.Series(False, index=df.index)
    for i, j in focal_cells:
        mask |= (binned['sev_bin'] == i) & (binned['mag_bin'] == j)
    return mask.values, len(focal_cells)


# --------------------------------------------------------------------------- #
# Diagnostics (unchanged core; PCA is diagnostic-only)
# --------------------------------------------------------------------------- #
def report_correlations(X, feat_names):
    print("\n  Feature correlation matrix (Pearson):")
    corr = pd.DataFrame(X, columns=feat_names).corr()
    with pd.option_context('display.float_format', '{:+.2f}'.format,
                           'display.width', 200, 'display.max_columns', 50):
        print(corr.to_string())
    hi = [(feat_names[a], feat_names[b], corr.iloc[a, b])
          for a in range(len(feat_names)) for b in range(a + 1, len(feat_names))
          if abs(corr.iloc[a, b]) >= 0.85]
    if hi:
        print("  High collinearity (|r| >= 0.85):")
        for a, b, r in hi:
            print(f"    {a:<18s} ~ {b:<18s} r = {r:+.2f}")


def report_pca_diagnostic(Xs, feat_names):
    """PCA is reported ONLY to reveal redundancy/effective dimensionality.
    Clustering is performed on the standardized absolute features, NOT PCs."""
    pca = PCA().fit(Xs)
    cum = np.cumsum(pca.explained_variance_ratio_)
    print("\n  [PCA diagnostic only — clustering uses absolute metrics] "
          "explained-variance ratio:")
    for i, (ev, c) in enumerate(zip(pca.explained_variance_ratio_, cum)):
        print(f"    PC{i+1}: {ev:6.3f}   cumulative {c:6.3f}")
    print(f"  effective dims (95% var): {int(np.searchsorted(cum, 0.95) + 1)} "
          f"of {len(feat_names)}")


def sweep_k(Xs, label):
    print(f"\n  KMeans sweep ({label}): k | inertia | silhouette | "
          f"Calinski-Harabasz | Davies-Bouldin")
    rng = np.random.RandomState(RANDOM_STATE)
    sidx = (rng.choice(Xs.shape[0], SIL_SAMPLE, replace=False)
            if Xs.shape[0] > SIL_SAMPLE else np.arange(Xs.shape[0]))
    rows = {}
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(Xs)
        lab = km.labels_
        sil = silhouette_score(Xs[sidx], lab[sidx])
        rows[k] = dict(inertia=km.inertia_, silhouette=sil,
                       ch=calinski_harabasz_score(Xs, lab),
                       db=davies_bouldin_score(Xs, lab))
        print(f"    {k:2d} | {km.inertia_:12.1f} | {sil:+.3f} | "
              f"{rows[k]['ch']:10.1f} | {rows[k]['db']:.3f}")
    best = max(rows, key=lambda k: rows[k]['silhouette'])
    print(f"  -> best silhouette at k={best} ({rows[best]['silhouette']:+.3f})")
    return rows


def sweep_ward(Xs, label):
    rng = np.random.RandomState(RANDOM_STATE)
    idx = (rng.choice(Xs.shape[0], WARD_SAMPLE, replace=False)
           if Xs.shape[0] > WARD_SAMPLE else np.arange(Xs.shape[0]))
    Xsub = Xs[idx]
    print(f"\n  Ward sweep ({label}, n={len(idx)}): k | silhouette")
    for k in K_RANGE:
        lab = AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(Xsub)
        print(f"    {k:2d} | {silhouette_score(Xsub, lab):+.3f}")


def sweep_gmm(Xs, label):
    print(f"\n  GMM sweep ({label}, full cov): k | BIC | AIC")
    bics = {}
    for k in K_RANGE:
        g = GaussianMixture(n_components=k, covariance_type='full',
                            random_state=RANDOM_STATE, max_iter=200).fit(Xs)
        bics[k] = g.bic(Xs)
        print(f"    {k:2d} | {bics[k]:12.1f} | {g.aic(Xs):12.1f}")
    print(f"  -> min BIC at k={min(bics, key=lambda k: bics[k])}")


def stability_ari(Xs, k, n_boot=N_BOOT):
    ref = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(Xs)
    n = Xs.shape[0]
    aris = []
    for b in range(n_boot):
        rng = np.random.RandomState(1000 + b)
        samp = rng.choice(n, n, replace=True)
        km = KMeans(n_clusters=k, n_init=5, random_state=b).fit(Xs[samp])
        aris.append(adjusted_rand_score(ref, km.predict(Xs)))
    return float(np.mean(aris)), float(np.std(aris))


def _group_means(d, cols, title):
    cols = [c for c in cols if c in d.columns]
    if not cols:
        return
    print(f"\n  {title}:")
    g = d.groupby('cluster')[cols].mean()
    with pd.option_context('display.float_format', '{:.2f}'.format,
                           'display.width', 220, 'display.max_columns', 60):
        print(g.to_string())


def characterize(d, labels, k, fs_name):
    d = d.copy()
    d['cluster'] = labels
    print(f"\n  === Cluster characterization (k={k}, features={fs_name}) ===")
    sizes = d['cluster'].value_counts().sort_index()
    print("  Cluster sizes:")
    for c, nn in sizes.items():
        print(f"    cluster {c}: {nn:7d}  ({100*nn/len(d):5.1f}%)")

    _group_means(d, HAZARD_DESCRIBE, "Mean HAZARD profile (original units; "
                                     "clustering basis, incl. antecedent)")
    _group_means(d, OPERATIONAL, "Mean OPERATIONAL profile (external)")
    _group_means(d, OUTCOME, "Mean OUTCOME profile (external)")

    if 'ffmp_zone_at_min' in d.columns:
        emrg = d.groupby('cluster').apply(
            lambda g: 100.0 * (g['ffmp_zone_at_min'] == 'Emergency').mean())
        print("\n  % reaching FFMP Emergency (OUTCOME) by cluster:")
        for c, v in emrg.items():
            print(f"    cluster {c}: {v:5.1f}%")

    ct = pd.crosstab(d['cluster'], d['dataset'])[
        [c for c in DATASETS if c in d['dataset'].unique()]]
    print("\n  Cluster x ensemble contingency (counts):")
    print(ct.to_string())
    chi2, p, dof, _ = chi2_contingency(ct.values)
    v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
    print(f"  chi2={chi2:.1f}  dof={dof}  p={p:.3e}  Cramers_V={v:.3f}")
    frac = ct.div(ct.sum(axis=0), axis=1) * 100.0
    print("\n  Within-ensemble cluster share (% of each ensemble's events):")
    with pd.option_context('display.float_format', '{:.1f}'.format):
        print(frac.to_string())
    return dict(sizes=sizes.to_dict(), chi2=float(chi2), p=float(p),
                cramers_v=float(v), within_ensemble_pct=frac.round(2).to_dict())


# --------------------------------------------------------------------------- #
def run_scope(scope, df):
    print("\n" + "=" * 78)
    print(f"SCOPE: {scope}  (n = {len(df):,} events)")
    print("=" * 78)
    summary = {'scope': scope, 'n_events': int(len(df)), 'feature_sets': {}}

    for fs_name, feats in FEATURE_SETS.items():
        sub = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feats)
        print("\n" + "-" * 70)
        print(f"FEATURE SET '{fs_name}': {feats}")
        print(f"  rows used: {len(sub):,} (dropped {len(df)-len(sub)})")
        Xs_full = StandardScaler().fit_transform(sub[feats].values)

        rng = np.random.RandomState(RANDOM_STATE)
        sw = (rng.choice(Xs_full.shape[0], SWEEP_SAMPLE, replace=False)
              if Xs_full.shape[0] > SWEEP_SAMPLE else np.arange(Xs_full.shape[0]))
        Xs = Xs_full[sw]
        tag = f'standardized, n={len(sw):,}'

        report_correlations(Xs_full, feats)
        report_pca_diagnostic(Xs_full, feats)
        km_rows = sweep_k(Xs, tag)
        sweep_ward(Xs, tag)
        sweep_gmm(Xs, tag)

        print(f"\n  Bootstrap stability (mean +/- std ARI, n={len(sw):,}):")
        for k in CANDIDATE_KS:
            m, s = stability_ari(Xs, k)
            print(f"    k={k}: ARI = {m:+.3f} +/- {s:.3f}")

        fs_sum = {}
        for k in CANDIDATE_KS:
            lab = KMeans(n_clusters=k, n_init=10,
                         random_state=RANDOM_STATE).fit_predict(Xs_full)
            fs_sum[k] = characterize(sub, lab, k, fs_name)
        summary['feature_sets'][fs_name] = {
            'features': feats,
            'kmeans_internal': {int(k): km_rows[k] for k in km_rows},
            'characterization': fs_sum,
        }
    return summary


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("#" * 78)
    print("# DROUGHT-EVENT CLUSTERING DIAGNOSTICS (SSI-3) — enriched hazard set")
    print(f"# datasets: {DATASETS}")
    print("#" * 78)

    df, em_frames = build_dataset()
    print(f"\nPooled merged events: {len(df):,}")

    fmask, n_cells = focal_mask_for(df, em_frames)
    print(f"Focal region: {n_cells} cells; {int(fmask.sum()):,} focal events")

    summaries = {}
    summaries['all'] = run_scope('all_droughts', df)
    summaries['focal'] = run_scope('focal_region',
                                   df.loc[fmask].reset_index(drop=True))

    out = os.path.join(OUT_DIR, 'cluster_diagnostics_summary.json')
    with open(out, 'w') as f:
        json.dump(summaries, f, indent=2, default=str)
    print(f"\nSaved summary: {out}\nDONE.")


if __name__ == '__main__':
    main()
