# Drought-Event Clustering ("Storylines") — Analysis Plan

## Motivation & manuscript hypotheses

Extend the focal-region drought analysis (Fig9/Fig10, SI21) from a single
threshold-brushed "high-consequence" set toward a **typology of drought
storylines**. Two manuscript-relevant hypotheses:

- **H1 — Distinct, differently-impactful types.** SSI-3 droughts fall into a
  small number of characteristic clusters that impose *qualitatively different*
  stresses on the NYC supply system (e.g. fast vs. slow drawdown, demand-driven
  shortage vs. Montague-target driven release, deep-storage vs. shallow events).
- **H2 — Climate-scenario frequency shift.** The *prevalence* of each cluster
  differs across the three climate ensembles (stationary, wetter-winter/drier-
  summer, wetter-winter/similar-summer), i.e. climate change reshuffles which
  storyline dominates.

This mirrors the **storyline** framing of Shepherd et al. (2018, *Climatic
Change*) and Sillmann et al. (2021, *Earth's Future*): each cluster is a
physically self-consistent drought type whose *plausibility/prevalence* shifts
with climate. It complements the process-based hydrological-drought typology of
Van Loon & Van Lanen (2012, *HESS*) — ours is data-driven rather than
mechanism-defined.

## Core design decision: cluster on HAZARD, characterize by IMPACT

To avoid circularity ("severe droughts have severe impacts"), we **cluster only
on drought hazard/antecedent characteristics** and treat **impact metrics** and
**ensemble label** as *external* variables used to describe clusters after
fitting. This lets H1 (impact differentiation) and H2 (scenario frequency)
emerge as findings rather than being built into the clustering.

- **Hazard / antecedent features (clustering inputs):** duration, severity
  (peak |SSI|), magnitude (cumulative deficit), avg_severity (mean intensity),
  severity_rate, peakedness (= severity/avg_severity), onset month encoded
  cyclically (sin/cos), antecedent storage at drought start.
- **Impact features (external, post-hoc):** min NYC storage, storage drawdown,
  NYC demand-shortage %, Montague shortage (volume + max consecutive days),
  contribution ratio, fraction reaching FFMP Emergency.
- **Group label (external):** climate ensemble.

Pool events across all three ensembles before clustering (per user request) so
a single, scenario-agnostic typology is learned; the ensemble label is then
used only to test H2.

## Data

Per-event metrics already exist (no rerun needed for the first pass):
`outputs/<config>/data/event_metrics/<dataset>_ssi3_event_metrics.csv`
(~70k / 83k / 65k events for the three ensembles; ~218k pooled). Loaded via
`methods.load.load_event_metrics` (min_duration=30 d). Peakedness and cyclic
month are derived from existing columns.

Two scopes are evaluated: **all droughts** (full typology) and the
**focal-region subset** (severe space from Fig9/Fig10, reusing
`methods.plotting.heatmap` + `methods.return_period`).

## Methods (diagnostics-first, no figures yet)

Implemented in `si_scripts/clustering/cluster_diagnostics.py`. Steps:

1. **Feature engineering & screening.** log-transform skewed positives
   (duration, magnitude, inflow, severity_rate); derive peakedness, month
   sin/cos. Report the feature correlation matrix to expose collinearity
   (duration/magnitude/avg_severity are near-redundant by construction).
2. **Standardize** (z-score) all features.
3. **PCA** — report explained-variance ratio and #components for 90/95%; cluster
   both on standardized features and on PCA-whitened leading components.
4. **Choose k** — sweep k=2..10 with three algorithms:
   - KMeans: inertia (elbow), **silhouette**, Calinski–Harabasz, Davies–Bouldin.
   - Ward agglomerative (on a subsample): silhouette per k.
   - GMM (full covariance): **BIC/AIC** per k.
   Heuristic thresholds (Kaufman & Rousseeuw): silhouette >0.5 reasonable, 0.25–
   0.5 weak, ≤0.25 negligible.
5. **Stability** — bootstrap (B≈20): refit KMeans on resamples, predict the full
   set, **Adjusted Rand Index** vs. the reference labeling; report mean±std per
   candidate k (ARI >0.6 = patterns, >0.85 = strong).
6. **Cluster characterization (H1)** — for candidate k: cluster sizes; mean
   hazard profile (original units); mean **impact** profile; % reaching
   Emergency. Look for clusters that are similarly severe but diverge in impact.
7. **Scenario composition (H2)** — cluster × ensemble contingency table; χ²
   test of independence + Cramér's V; and within-ensemble cluster fractions
   (the quantity H2 is about).

All numeric results are printed to `logs/SI.out` and key tables written to
`outputs/<config>/data/clustering/`.

## Go / no-go criteria

Proceed to a manuscript figure only if **most** of these hold:
- A defensible k exists with silhouette ≥ ~0.25 (ideally ≥0.4) and a clear
  elbow/BIC knee, agreeing across KMeans/Ward/GMM (cross-ARI high).
- Clusters are **stable** (bootstrap ARI ≥ ~0.6).
- Clusters are **interpretable** — nameable storylines, not arbitrary slices.
- **H1**: impact profiles differ across clusters *beyond* what raw severity
  alone explains (similar-severity clusters with different impact).
- **H2**: cluster×ensemble χ² significant **with non-trivial effect size**
  (Cramér's V, and visibly different within-ensemble fractions).

If clusters are weak/unstable/uninterpretable, or impact/scenario differences
are negligible or fully explained by severity, **recommend dropping** the
clustering and retaining the simpler threshold-brushed storyline (SI21).

## Candidate additional metrics (if separability is poor)

These require rerunning the event-metrics driver (`06_calculate_performance_
metrics.py` → `methods/metrics/event_metrics.py`), which reads the within-event
SSI/timeseries:
- **Onset / intensification rate** (flash vs. creeping): ΔSSI from start to peak
  ÷ time-to-peak.
- **Recovery / termination rate**: ΔSSI from peak to end ÷ time-from-peak.
- **Time-to-peak fraction** (peak position within event).
- **Season-span / multi-season flag** (Van Loon wet→dry analogue).
- **Demand-weighted severity** (severity co-located with high-demand season).
These would be added to `_calculate_single_event` and re-exported to the CSVs.

## Deliverables (for review)

1. `PLAN.md` (this file).
2. `cluster_diagnostics.py` + its printed report (`logs/SI.out`) and saved
   summary tables.
3. `FINDINGS.md` — diagnostic summary, methods recap, go/no-go recommendation.
4. A **preliminary figure** *iff* diagnostics support it (candidate: parallel-
   axis colored by cluster + a cluster×ensemble frequency panel, or PCA scatter
   + centroid profiles).
