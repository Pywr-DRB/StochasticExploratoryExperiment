# Drought-Event Clustering — Diagnostic Findings & Recommendation

_SSI-3 drought events pooled across the three climate ensembles
(stationary, climate_adjusted_low, climate_adjusted_high). n = 213,938 pooled;
focal-region subset n = 24,378. Clustering on standardized **absolute** hazard
metrics (no PCA); operational/outcome/ensemble treated as external. No PRIM._

## Methods (as implemented)

- **Features (hazard/exposure only):** built incrementally —
  `size_only` (log_duration, log_magnitude, severity, avg_severity, peakedness);
  `size+rates` (+ onset_rate, recovery_rate); `full` (+ onset season sin/cos,
  prior_3m_surplus). Onset/recovery rates and antecedent wetness are derived
  from the existing `drought_events` SSI definitions (no pipeline rerun;
  validated in `assess_dynamics_metrics.py`, 100% merge with event_metrics).
- **Algorithms:** KMeans, Ward, GMM swept over k=2–10; standardized absolute
  features (PCA reported only as a redundancy diagnostic).
- **Validation:** silhouette / Calinski-Harabasz / Davies-Bouldin / GMM-BIC for
  k; **bootstrap Adjusted Rand Index** for stability.
- **Characterization (external):** per-cluster operational + outcome profiles,
  %FFMP-Emergency, and cluster×ensemble χ² / Cramér's V / within-ensemble
  shares.

## Results

### 1. The events are a CONTINUUM, not gap-separated clusters
Best silhouette is always at k=2 and **declines as features are added**
(all-droughts: size_only 0.337 → size+rates 0.263 → full 0.186; focal similar).
GMM-BIC never reaches a minimum within k≤10 (favours k=max). By every internal
index there is **no natural number of clusters** — structure is graded.

### 2. ...but the partitions are STABLE and INTERPRETABLE
Bootstrap ARI ≈ 0.87–0.97 at k=3–4: the KMeans partition is highly reproducible.
So we can reliably *partition* the continuum even though it has no gaps. The
`full` set at **k=4** yields four physically interpretable archetypes:

| # | size | character | onset | season | antecedent | %Emergency |
|---|------|-----------|-------|--------|------------|-----------|
| 0 | 20% | mod-severity, short | **fast** (1.85) | spring | moderate | 2.3% |
| 1 | 35% | mild, short | slow | spring | moderate | 0.1% |
| 2 | 27% | **severe, ~16 mo** | slow | summer | low | **10.3%** |
| 3 | 19% | mild, moderate | — | **autumn** | **driest** | 0.3% |

### 3. H1 (different impacts) — supported, including the non-trivial version
Cluster 2 is the high-consequence type (min storage 46%, drawdown 45%, Montague
shortage 39, 10.3% Emergency) vs <0.5% Emergency elsewhere. Importantly,
clusters 0/1/3 have *comparable severity* (~1.4–2.0) yet differ by onset speed,
season, and antecedent wetness — i.e. impact/character differ at fixed severity,
the genuinely novel result the size-only features could not surface.

### 4. H2 (scenario frequency shift) — supported, and STRENGTHENED by the new metrics
Cluster×ensemble Cramér's V rises from 0.116 (size_only) to **0.196 (full, k=4)**
— the expanded metrics roughly doubled scenario discrimination even as silhouette
fell. Within-ensemble shares (stationary → climate_high):
- prolonged-severe (cl 2): **34.5% → 19.1%** (↓↓)
- autumn/dry-antecedent (cl 3): 29.1% → 17.8% (↓)
- mild-short (cl 1): 24.2% → 42.7% (↑↑)
- flash-spring (cl 0): 12.2% → 20.4% (↑)
Climate change shifts the basin away from prolonged-severe and autumn droughts
toward milder, shorter, faster-onset spring droughts — physically sensible.

## Recommendation: QUALIFIED PROCEED — frame as a descriptive typology, not "natural clusters"

The clustering is **worth including**, but must be framed honestly:

- **Do NOT claim** we discovered well-separated natural clusters. We did not —
  silhouette is weak and falls with more features; the population is a continuum.
- **DO claim** we partition that continuum into a small set of **stable,
  reproducible, physically interpretable drought storylines** (sensu Shepherd
  2018; cf. Van Loon typology), and demonstrate (H1) they impose distinct
  impacts and (H2) their prevalence shifts across climate scenarios. The
  evidence for inclusion is **stability + interpretability + the scenario
  χ²/shares**, NOT silhouette.
- **Lead with the full k=4 partition** — highest scenario discrimination
  (V=0.196), four nameable archetypes, ARI≈0.94.
- **Report the continuum honestly** as a limitation/feature (graded structure;
  silhouette modest) — this is consistent with the literature and pre-empts the
  obvious reviewer critique. The Steinmann et al. (2020) caveat about untested
  behaviour on stochastic ensembles is answered by our bootstrap-ARI stability.

### Alternative if a cleaner story is preferred
Because structure is graded, a **mechanism-defined archetype scheme** (e.g.
season × duration × onset-speed bins) would be even more defensible and need no
"why k=4" justification, while recovering the same physical types. The
data-driven k=4 happens to align with such archetypes, so either framing works;
the data-driven version is more objective, the mechanism version more
interpretable.

### If rejected
If reviewers/PI prefer to avoid the continuum complication entirely, the
threshold-brushed storyline in SI21 (now with onset/recovery/antecedent axes)
already conveys the high-consequence narrative without committing to clusters.

## Artifacts
- `cluster_diagnostics.py` (+ JSON summary in `data/clustering/`), full report
  in the run log.
- `assess_dynamics_metrics.py` — metric validation.
- `PLAN.md`, `LITERATURE_NOTES.md`.
- Preliminary figure: `cluster_preliminary_figure.py` (storyline profiles +
  H1 impact + H2 scenario shares).
