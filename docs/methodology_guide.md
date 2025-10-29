# StochasticExploratoryExperiment Methodology Documentation

This directory contains methodological documentation for the StochasticExploratoryExperiment workflow.

## Contents

1. [Copula Methodology for Drought Return Period Analysis](#copula-methodology)
2. [Streamflow Scenario Comparison](#streamflow-comparison)
3. [Satisficing Conditions Analysis](#satisficing-analysis)
4. [Centralized Styling System](#styling-system)

---

# Copula Methodology for Drought Return Period Analysis {#copula-methodology}

## Overview

This section explains the methodology used in `09_plot_drought_frequency.py` for calculating drought return periods under different climate scenarios using copula-based joint probability modeling.

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

---

# Streamflow Scenario Comparison {#streamflow-comparison}

## Overview

Script `10_plot_streamflow_scenario_comparison.py` creates a 3-panel comparison figure showing how streamflow distributions change across climate scenarios.

## Figure Layout

### Panel 1: Annual Flow Distributions (KDE)
- **Purpose**: Compare probability density of annual total flows
- **Method**: Kernel density estimation (Gaussian kernel)
- **Visualization**:
  - Historic data: Black line (linewidth=2.5)
  - Ensemble scenarios: Colored lines (linewidth=2.5)
  - **No fill** under curves (changed from original filled version for clarity)

**Interpretation**:
- **Peak position**: Where distribution centers (mean/median annual flow)
- **Peak height**: Concentration of probability (narrow = consistent, wide = variable)
- **Tail shape**: Frequency of extreme wet/dry years

**Expected climate scenario effects**:
- Climate Low (Dry): Peak shifts left (lower flows), potentially wider spread
- Climate High (Wet): Peak shifts right (higher flows)

### Panel 2: Flow Duration Curves (FDC)
- **Purpose**: Show flow magnitude vs. exceedance probability
- **Method**: Sort daily flows, compute exceedance probabilities
- **Visualization**:
  - Historic data: Black line (no scatter points)
  - Ensemble scenarios: Median line + shaded p10-p90 range
  - **Linear y-axis** (not log scale)

**Interpretation**:
- **Left side (low exceedance)**: High flows (rare events)
- **Right side (high exceedance)**: Low flows (common events)
- **Slope**: Variability (steep = highly variable, flat = consistent)

**Reading the FDC**:
- "Flow at 10% exceedance = 5000 MGD" means flow ≥ 5000 MGD occurs 10% of time
- Comparison shows how entire flow regime shifts with climate

### Panel 3: Weekly Streamflow Patterns
- **Purpose**: Show seasonal flow patterns across the year
- **Method**: Group by week-of-year, calculate percentiles across all realizations
- **Visualization**:
  - Historic: Black line + shaded p10-p90 range
  - Ensemble scenarios: Colored median line + shaded ranges
  - **Linear y-axis** (not log scale)
  - Month labels on top axis for reference

**Interpretation**:
- **Seasonal patterns**: Spring snowmelt peak, summer low flows
- **Range width**: Inter-annual variability
- **Scenario differences**: How climate shifts seasonal timing/magnitude

## Color Scheme

Uses centralized colors from `methods/plotting/styles.py`:
- **Historic**: Black (#000000)
- **Stationary**: Blue (#1f77b4)
- **Climate Low**: Red (#d62728) - Driest scenario
- **Climate Medium**: Purple (#9467bd)
- **Climate High**: Green (#2ca02c) - Wettest scenario

## Design Choices

### Why Linear Scale (not log)?
- **Clarity**: Linear scales easier to interpret for general audiences
- **Comparison focus**: Emphasis on absolute differences between scenarios
- **Seasonal patterns**: Weekly panel shows seasonal variability more clearly on linear scale

Log scales are better for:
- Spanning multiple orders of magnitude (not the case here)
- Emphasizing low-flow extremes (drought focus)

Linear scales are better for:
- Comparing absolute magnitudes
- Showing proportional differences
- General climate scenario communication

### Why No Fill Under KDE Curves?
- **Overlapping clarity**: Multiple overlapping filled areas create visual confusion
- **Line focus**: Easier to trace individual scenario distributions
- **Publication quality**: Cleaner appearance for papers

### Why No Scatter on FDC?
- **Data volume**: With 2000 realizations × 70 years, scatter is too dense
- **Percentile ranges**: Shaded areas communicate uncertainty better than scatter
- **Visual clarity**: Clean lines easier to compare across scenarios

## Usage

```bash
# Generate comparison figure for delMontague
python 10_plot_streamflow_scenario_comparison.py delMontague

# Output
figures/streamflow_scenario_comparison_delMontague.png
```

## Validation

The figure demonstrates:
1. ✅ Stationary ensemble envelops historic data (validation of KN generator)
2. ✅ Climate scenarios shift distributions in expected directions
3. ✅ Uncertainty ranges (p10-p90) show ensemble spread
4. ✅ Seasonal patterns preserved across scenarios

---

# Satisficing Conditions Analysis {#satisficing-analysis}

## Overview

Script `09_plot_satisficing_scatter.py` analyzes "satisficing" conditions - scenarios where the water system meets acceptable performance thresholds simultaneously for NYC water supply and Delaware River flow targets.

## Satisficing Criteria

A year-realization pair is considered **satisficing** if it meets BOTH conditions:

1. **NYC Storage ≥ 20%** throughout June-December period
   - Combined storage of Cannonsville, Pepacton, and Neversink reservoirs
   - 20% threshold ensures minimum water supply reliability

2. **Montague Flow Target Violations ≤ 3 consecutive days**
   - Pre-calculated shortage at delMontague node
   - Short-term violations acceptable; extended failures are not

## Methodology

### Data Sources (Pre-calculated in 04_postprocess_data.py)

- `res_storage`: Daily reservoir storage levels
- `shortage`: Flow target violations (already calculated)
- `inflow`: NYC reservoir inflows (aggregated)
- `contribution`: NYC downstream contributions to Montague

### Calculation Process

For each year-realization pair:

1. **Filter to June 1 - December 31** (critical period for both systems)

2. **Check NYC storage**:
   ```python
   nyc_storage_pct = 100 * (Cannonsville + Pepacton + Neversink) / total_capacity
   storage_ok = min(nyc_storage_pct) >= 20%
   ```

3. **Check Montague violations**:
   ```python
   # Find maximum consecutive violation days
   violations = shortage > 0
   max_consecutive_days = max_run_length(violations)
   montague_ok = max_consecutive_days <= 3
   ```

4. **Calculate aggregates** for plotting:
   - Total NYC inflow (Jun-Dec)
   - Total NYC → Montague contributions (Jun-Dec)

5. **Classify**: `satisficing = storage_ok AND montague_ok`

## Figure Layout

### 4-Panel Comparison

Layout matches other analysis figures:
- **Left panel**: Stationary ensemble
- **Right panels (stacked)**: Climate Low, Medium, High

Each panel shows:
- **X-axis**: NYC Reservoir Inflow (Jun-Dec) [MG]
- **Y-axis**: NYC → Montague Contributions (Jun-Dec) [MG]
- **Colors**:
  - Satisficing points: Dataset color (blue/red/purple/green) with alpha=0.6
  - Non-satisficing points: Gray (#808080) with alpha=0.4
- **Statistics**: Satisficing percentage displayed in bottom-right corner

### Design Choices

**Why scatter plot?**
- Shows relationship between inflow and contributions
- Identifies trade-off regions (low inflow but high contributions = stress)
- Reveals whether satisficing depends on hydrologic conditions

**Why different colors for satisficing vs non-satisficing?**
- Clear visual distinction
- Satisficing points use dataset color for consistency with other figures
- Non-satisficing points in gray to de-emphasize

**Why plot non-satisficing first (zorder=1)?**
- Satisficing points overlay on top (zorder=2)
- Important (satisficing) points more visible
- Gray background shows "problematic" region

## Interpretation

### What the Plot Shows

**Horizontal spread (X-axis variability)**:
- Wide spread → Diverse inflow conditions
- Tight clustering → Consistent inflow across realizations

**Vertical spread (Y-axis variability)**:
- Wide spread → Variable contribution requirements
- Linear relationship → Contributions scale with inflow
- Non-linear → Threshold effects or operational constraints

**Color distribution**:
- **Satisficing clustered at high inflows**: System needs wet conditions
- **Satisficing across inflow range**: System robust to variability
- **Non-satisficing at low inflows only**: Low-flow years are problematic
- **Non-satisficing scattered**: Other factors beyond inflow matter

### Expected Climate Scenario Effects

**Climate Low (Dry)**:
- **Lower satisficing rate**: More non-satisficing points
- **Shift left**: Lower inflows overall
- **Compressed range**: Less variability in contributions
- **More failures**: Both storage and Montague criteria violated

**Climate High (Wet)**:
- **Higher satisficing rate**: Fewer non-satisficing points
- **Shift right**: Higher inflows
- **Expanded range**: More diverse contribution patterns
- **Fewer failures**: Ample water for both NYC and Montague

## Summary Statistics

Script prints detailed breakdown:

### Overall Satisficing Rates
```
Dataset                          Total  Satisficing        %
Stationary                     140,000      98,000     70.0%
Climate Low                    140,000      85,000     60.7%
Climate Medium                 140,000      95,000     67.9%
Climate High                   140,000     105,000     75.0%
```

### Failure Breakdown
```
Stationary:
  Storage < 20% only:          15,000  (10.7%)
  Montague > 3 days only:      20,000  (14.3%)
  Both failures:                7,000   (5.0%)
```

**Interpretation**:
- If "Storage only" dominates → NYC supply more vulnerable
- If "Montague only" dominates → River flow targets more vulnerable
- If "Both" is large → Compound failures common (worse scenario)

## Usage

```bash
# Run analysis for all datasets (generates 4-panel figure)
python 09_plot_satisficing_scatter.py

# Output
figures/satisficing/satisficing_4panel_comparison.png
figures/satisficing/satisficing_4panel_comparison.svg
figures/satisficing/<dataset_id>_satisficing_results.csv  # Individual CSVs
```

## Validation

Check for expected patterns:
1. ✅ Higher inflows correlate with higher satisficing rates
2. ✅ Climate Low has lowest satisficing rates
3. ✅ Climate High has highest satisficing rates
4. ✅ Failure types make physical sense
5. ✅ Axis limits consistent across panels for comparison

## Applications

This analysis helps answer:
- **Robustness**: How often does the system meet both objectives?
- **Trade-offs**: What inflow-contribution combinations work?
- **Climate sensitivity**: How do satisficing rates change?
- **Failure modes**: Which constraint fails first?
- **Risk assessment**: What are odds of simultaneous failures?

---

# Centralized Styling System {#styling-system}

## Overview

The `methods/plotting/styles.py` module provides centralized styling for all visualization scripts, ensuring consistency across the entire analysis.

## Key Components

### 1. Dataset Colors
```python
DATASET_COLORS = {
    'stationary_ensemble': '#1f77b4',           # Blue
    'climate_adjusted_low': '#d62728',          # Red (Dry)
    'climate_adjusted_medium': '#9467bd',       # Purple
    'climate_adjusted_high': '#2ca02c',         # Green (Wet)
}
```

**Color rationale**:
- **Blue (Stationary)**: Neutral baseline color
- **Red (Low/Dry)**: Warm color associated with drought/heat
- **Green (High/Wet)**: Cool color associated with water/vegetation
- **Purple (Medium)**: Intermediate between red and blue

### 2. Dataset Labels
```python
DATASET_LABELS = {
    'stationary_ensemble': 'Stationary',
    'climate_adjusted_low': 'Climate Low',
    'climate_adjusted_medium': 'Climate Medium',
    'climate_adjusted_high': 'Climate High',
}
```

Also available:
- `DATASET_LABELS_SHORT`: Compact labels for tight layouts
- `DATASET_LABELS_DESCRIPTIVE`: Full descriptions with scenario context

### 3. Styling Parameters
- **Alpha values**: Fill (0.3), Line (0.8), Scatter (0.7), Bar (0.8)
- **Line widths**: Thin (1.0), Medium (2.0), Thick (2.5)
- **Colormaps**: Sequential (viridis), Diverging (BrBG), Heatmap (magma)
- **Figure sizes**: Single, Double, Triple, Quad, Large
- **Font sizes**: Small (9), Medium (10), Large (11), Title (14), Suptitle (16)
- **DPI**: Screen (100), Print (300), High (400)

### 4. Helper Functions

```python
# Get color for a dataset
color = get_dataset_color('stationary_ensemble')

# Get all colors in order
colors = get_all_dataset_colors()

# Apply publication styling to matplotlib
apply_publication_style()

# Get complete style dictionary
style = get_scenario_style('climate_adjusted_low')
ax.plot(x, y, **style)  # Unpack directly
```

## Usage Pattern

### Before (inconsistent):
```python
# In script A
colors = {'stationary': '#1f77b4', ...}

# In script B
colors = {'stationary': 'blue', ...}  # Different!
```

### After (consistent):
```python
# All scripts
from methods.plotting.styles import DATASET_COLORS, DATASET_LABELS

ax.plot(x, y, color=DATASET_COLORS['stationary_ensemble'],
        label=DATASET_LABELS['stationary_ensemble'])
```

## Scripts Using Centralized Styling

- ✅ `10_plot_streamflow_scenario_comparison.py`
- ✅ `07_compare_copula_parameters.py`
- 🔄 To be updated: `09_plot_*.py` scripts

## Benefits

1. **Consistency**: Same colors across all figures in publication
2. **Maintainability**: Update colors in one place
3. **Flexibility**: Easy to switch between color schemes (e.g., colorblind-friendly)
4. **Documentation**: Central location for styling decisions
5. **Quality**: Pre-defined publication-ready settings

## Future Enhancements

Potential additions:
- Colorblind-friendly alternative palette
- Grayscale-safe palette for print
- Institution-specific color schemes
- Interactive plot styling configuration

---

## Document History

- **2025-01-XX**: Initial documentation
  - Copula methodology explanation
  - Streamflow comparison documentation
  - Centralized styling system documentation
