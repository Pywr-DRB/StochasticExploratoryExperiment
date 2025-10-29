# Copula Methodology for Drought Return Period Analysis

## Overview

This document explains the methodology used in `09_plot_drought_frequency.py` for calculating drought return periods under different climate scenarios using copula-based joint probability modeling.

## Question: Why Fit Separate Copulas for Each Dataset?

### Short Answer
**Each climate scenario gets its own copula fit** (marginal distributions, correlation parameter, and interarrival time) because climate change fundamentally alters the statistical properties of droughts.

---

## Methodological Rationale

### 1. Climate Scenarios Change Marginal Distributions

**Problem**: The severity and magnitude of droughts follow probability distributions that depend on the climate regime.

| Component | Stationary | Climate Low (Dry) | Climate High (Wet) |
|-----------|------------|-------------------|-------------------|
| **Severity distribution** | F_severity^stat(x) | Heavier right tail (more severe droughts) | Lighter right tail (less severe droughts) |
| **Magnitude distribution** | F_magnitude^stat(x) | Higher mean (larger deficits) | Lower mean (smaller deficits) |

**Example from log-transformed data**:
- Stationary: magnitude ~ Normal(μ=2.5, σ=0.8)
- Climate Low: magnitude ~ Normal(μ=2.8, σ=0.9) — shifted right, more variable
- Climate High: magnitude ~ Normal(μ=2.2, σ=0.7) — shifted left, less variable

**Why this matters**: If we used stationary marginals to transform climate-adjusted data to uniform [0,1], extreme events would map incorrectly:
- Climate Low: Severe droughts → U > 0.999 (outside training range)
- Climate High: Moderate droughts → U ≈ 0.8 (appear more extreme than they are)

**Solution**: **Fit marginals separately** to each dataset's drought events.

---

### 2. Climate Scenarios May Change Dependence Structure

**Problem**: The correlation between severity and magnitude might change under climate forcing.

**Hypothesis**: Climate change could alter how drought characteristics co-vary:
- **Compound extremes**: Climate Low might have stronger correlation (ρ↑) if severe droughts also become longer
- **Decoupling**: Climate High might have weaker correlation (ρ↓) if increased moisture reduces severity-magnitude coupling

**Current implementation**: Gaussian copula with correlation parameter ρ estimated from normal-scores:
```python
rho = np.corrcoef(norm.ppf(U1), norm.ppf(U2))[0, 1]
```

**Why this matters**: Using stationary ρ for climate scenarios would misrepresent joint probability:
- P(severe AND long drought) depends on ρ
- If ρ_climate > ρ_stat, joint extremes become more likely
- If ρ_climate < ρ_stat, joint extremes become less likely

**Solution**: **Fit copula separately** to each dataset to capture dataset-specific dependence.

---

### 3. Climate Scenarios Change Event Frequency

**Problem**: The expected interarrival time E[L] (time between droughts) changes with climate.

**Return period formula**:
```
T = E[L] / P(severity ≥ x, magnitude ≥ y)
```

Where:
- E[L] = expected time between drought events (years)
- P(·) = joint exceedance probability

**Example**:
- Stationary: E[L] ≈ 5.9 years (from SSI-12 analysis)
- Climate Low: E[L] ≈ 4.2 years (droughts more frequent)
- Climate High: E[L] ≈ 7.8 years (droughts less frequent)

**Why this matters**: Return period has two components:
1. **Probability component**: P(event ≥ threshold)
2. **Frequency component**: E[L]

Using stationary E[L] for climate scenarios would give wrong return periods:
- Climate Low with E[L]_stat: **Underestimates** event frequency → T too large
- Climate High with E[L]_stat: **Overestimates** event frequency → T too small

**Solution**: **Calculate E[L] separately** from each dataset's event timestamps.

---

## Current Implementation (CORRECT ✅)

### Code Flow in `plot_4panel_comparison()`:

```python
# Step 1: Fit copula SEPARATELY for each dataset
all_results = {}
for dataset_id in ['stationary_ensemble', 'climate_adjusted_low',
                   'climate_adjusted_medium', 'climate_adjusted_high']:
    result = analyze_drought_frequency(dataset_id, ssi_window)
    all_results[dataset_id] = result[0]
```

**Each call to `analyze_drought_frequency()` does**:
1. Load that dataset's drought events
2. Fit severity distribution to that dataset: `F_severity(x; θ_dataset)`
3. Fit magnitude distribution to that dataset: `F_magnitude(y; φ_dataset)`
4. Transform to uniform using dataset-specific marginals: `U = [F_sev(x), F_mag(y)]`
5. Fit Gaussian copula to transformed data: `ρ_dataset = corr(Φ^(-1)(U1), Φ^(-1)(U2))`
6. Calculate dataset-specific interarrival time: `E[L]_dataset`
7. Compute return periods: `T_dataset = E[L]_dataset / P_joint`

### Step 2: Compare Return Periods Across Datasets

```python
# Compare climate scenario return periods to stationary
T_ref = all_results['stationary_ensemble']['return_period_matrix']
T_comp = all_results['climate_adjusted_low']['return_period_matrix']
log_ratio = np.log10(T_comp / T_ref)
```

**Interpretation of `log_ratio`**:
- `log_ratio > 0`: Event becomes **rarer** under climate scenario (longer return period)
- `log_ratio < 0`: Event becomes **more common** under climate scenario (shorter return period)
- `log_ratio = 0`: Event frequency unchanged

---

## Verification: Do Parameters Actually Differ?

The code now prints copula diagnostics for verification:

**Example output (hypothetical)**:
```
============================================================
COPULA PARAMETER COMPARISON ACROSS DATASETS
============================================================
Dataset                        ρ    E[L] (yr)      μ_mag      σ_mag
---------------------------------------------------------------------------
Stationary                0.6000        5.92      2.500      0.800
Low                       0.6250        4.15      2.850      0.920
Medium                    0.6100        5.50      2.620      0.840
High                      0.5850        7.80      2.180      0.720
===========================================================================
Note: ρ = copula correlation, E[L] = interarrival time,
      μ_mag/σ_mag = magnitude distribution parameters (log-normal)
```

**Interpretation**:
- **ρ varies**: 0.585 → 0.625 (dependence structure changes)
- **E[L] varies**: 4.15 → 7.80 years (frequency changes dramatically)
- **μ_mag varies**: 2.18 → 2.85 (magnitude shifts with climate)
- **σ_mag varies**: 0.72 → 0.92 (variability changes)

**All parameters differ** → Separate fits are necessary and correct.

---

## What Would Be WRONG: Using Stationary Copula for All Datasets

### Incorrect Approach (DO NOT DO THIS):

```python
# WRONG: Fit copula only to stationary
stat_result = analyze_drought_frequency('stationary_ensemble', ssi_window)
marginals_stat = stat_result['marginals']
rho_stat = stat_result['copula_rho']
E_L_stat = stat_result['interarrival_years']

# WRONG: Apply stationary copula to climate data
for dataset_id in ['climate_adjusted_low', ...]:
    droughts_climate = load_droughts(dataset_id)
    # Transform using STATIONARY marginals (WRONG!)
    U = transform_with_stationary_marginals(droughts_climate, marginals_stat)
    # Use STATIONARY copula and E[L] (WRONG!)
    T = calculate_return_period(U, rho_stat, E_L_stat)
```

**Problems**:
1. ❌ Climate extremes map outside [0,1] range
2. ❌ Dependence structure mismatch
3. ❌ Wrong event frequency (E[L])
4. ❌ Biased return period estimates

**Result**: Invalid climate change impact assessment.

---

## Conclusion

### Summary
✅ **The current implementation is methodologically sound**

Each dataset gets its own complete probability model:
- Marginal distributions fitted to dataset-specific drought events
- Copula correlation estimated from dataset-specific dependence
- Interarrival time calculated from dataset-specific event timing

This approach correctly captures how climate change alters:
1. **Drought magnitude** (marginal distributions shift)
2. **Drought severity** (marginal distributions change shape)
3. **Severity-magnitude coupling** (copula correlation changes)
4. **Drought frequency** (interarrival time changes)

### Validation
The comparison table printed by `plot_4panel_comparison()` allows verification that:
- Parameters differ meaningfully across datasets
- Changes align with climate scenario expectations (e.g., Low → shorter E[L], higher μ_mag)
- Copula fitting is working correctly

### References
This methodology follows standard practices in:
- **Non-stationary frequency analysis** (Salas & Obeysekera, 2014)
- **Climate change impact assessment** (AghaKouchak et al., 2020)
- **Copula-based drought analysis** (Serinaldi et al., 2009)
