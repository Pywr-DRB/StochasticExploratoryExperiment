# Literature notes & positioning — drought-event clustering

## Primary positioning reference

**Steinmann, P., Auping, W. L., & Kwakkel, J. H. (2020). Behavior-based
scenario discovery using time series clustering. *Technological Forecasting &
Social Change*, 156, 120052. https://doi.org/10.1016/j.techfore.2020.120052**

What they do: replace the conventional binary-threshold classification in
scenario discovery with **time-series clustering of full model-output
trajectories** (Complexity-Invariant Distance, a DTW variant; hierarchical hard
clustering; k chosen via internal validity indices, Arbelaitz et al. 2013),
then run PRIM rule induction on each behaviorally distinct cluster to tie
scenarios to *dynamics over time* rather than a single endpoint. They cluster
one univariate output (oil price) over a 2,000-run ensemble.

**How we relate (similar but different):** same motivation — partition an
ensemble into behaviorally distinct groups *before* relating them to drivers —
but we differ in (i) **unit of analysis**: discrete drought *events* extracted
from runs, not whole trajectories; and (ii) **similarity definition**: we use
the **feature-based** branch of time-series clustering (Liao 2005) — interpretable,
physically grounded scalar features (duration, peak severity, cumulative
deficit, mean intensity, peakedness, onset timing, antecedent storage) — rather
than their data/shape-based DTW distance. Liao (2005)'s taxonomy (feature- /
data- / model-based) is cited *by Steinmann et al. themselves*, so our choice is
a recognized sibling method, not a departure from their framework.

### Warnings from Steinmann et al. we must heed (and how we respond)

1. **Metric/feature-choice sensitivity** — "central claims do not rest on this
   particular choice"; other metrics may suit other models. → We test
   robustness across **two feature sets** and report internal validity indices
   for k (citing Arbelaitz et al. 2013, as they do).
2. **Hard vs. soft clustering** — they concede a *soft* clustering "might have
   been more appropriate" (clusters overlapped; only 1 of 6 cleanly separable),
   but note soft assignment is hard to embed in PRIM. → **We do not use PRIM**,
   so this constraint does not bind us; we additionally fit a **GMM (soft,
   BIC-selected)** and can report boundary/overlap behavior directly.
3. **Untested on noisy / stochastic ensembles** — they explicitly flag that
   sensitivity to "noisy time series outputs" / stochastic models "remains to be
   seen." **This is exactly our setting (a stochastic ensemble).** → We treat
   this as both a caveat *and* a contribution: we quantify cluster robustness
   with **bootstrap Adjusted Rand Index stability**, a formal robustness test
   they did *not* perform.
4. **Cluster separability in driver/scenario space matters** — significant
   false positives arose from **PRIM's orthogonal boxes**; overlap undermines
   attributing outcomes to a scenario. → **We deliberately do NOT use PRIM (or
   any rule-induction over the input space).** For H2 (climate-scenario
   frequency shift) we relate clusters to scenario directly via the
   cluster×ensemble contingency, χ², Cramér's V, and within-ensemble cluster
   shares — avoiding the orthogonal-box false-positive problem entirely. This is
   the principal methodological divergence from Steinmann et al.

### Honest caveats about the paper (do not over-cite precision)
They do not state whether series were normalized ("typical settings"), nor the
linkage type or the specific validity index/scores behind k = 6; and they did
no bootstrap/robustness testing. Cite for *concept and motivation*, not for a
precise validation protocol.

### Draft citation sentence
> Steinmann et al. (2020) introduced behavior-based scenario discovery, using
> time-series clustering of full model-output trajectories so that scenario
> rule-induction is applied per behaviorally distinct subset rather than against
> a single endpoint threshold. We adopt the same cluster-then-relate logic but
> operate on discrete drought *events* described by interpretable scalar
> features (the feature-based branch of the time-series-clustering taxonomy,
> Liao 2005), and — addressing a robustness gap they note for stochastic
> ensembles — we assess cluster stability via bootstrap Adjusted Rand Index.

## Supporting references (from literature scan)

- **Shepherd et al. (2018)**, *Climatic Change* 151:555–571 — storyline concept
  (physically self-consistent events; plausibility over probability).
- **Sillmann et al. (2021)**, *Earth's Future* 9, e2020EF001783 — event-based
  storylines bridging physical climate to impacts/decisions.
- **Van Loon & Van Lanen (2012)**, *HESS* 16:1915–1946 — process-based
  hydrological-drought typology (6 types); severe droughts skew to specific
  types (conceptual anchor for H1).
- **Liao (2005)**, *Pattern Recognition* 38:1857–1874 — time-series clustering
  taxonomy (feature / data / model-based).
- **Rousseeuw (1987)** silhouette; **Tibshirani, Walther & Hastie (2001)** gap
  statistic; **Arbelaitz et al. (2013)** validity-index comparison;
  **Hubert & Arabie (1985)** Adjusted Rand Index; **Hennig (2007)** cluster-wise
  stability — validation toolkit.
- **Tijdeman et al. (2020)** *WRR*; **Otkin et al. (2024)** *WRR* (flash-drought
  intensification) — event extraction (run theory) and onset/recovery-rate
  feature definitions.
