# Data-Driven Vulnerability Event Analysis

## Implementation Plan for StochasticExploratoryExperiment

**Purpose:** Identify what conditions lead to water supply failures through data-driven event identification and feature selection, without imposing arbitrary drought thresholds.

**Philosophy:** Let the data reveal which factors matter, rather than pre-specifying drought definitions.

---

## Overview

```
Weekly Time Series (3.6M observations)
            ↓
Step 1: Event Definition (storage drawdowns)
            ↓
Events (~10K-50K units)
            ↓
Step 2: Feature Extraction (30-40 features per event)
            ↓
Step 3: Feature Selection (MI, SHAP)
            ↓
Step 4: Consensus Features (robust subset)
            ↓
Step 5: Scenario Discovery (CART, factor maps)
            ↓
Interpretable Rules + Visualizations
```

---

## Directory Structure

```
methods/
├── vulnerability/                    # NEW MODULE
│   ├── __init__.py
│   ├── config.py                     # Configuration dataclass
│   ├── events.py                     # Step 1: Event identification
│   ├── features.py                   # Step 2: Feature extraction
│   ├── selection.py                  # Steps 3-4: Feature selection
│   ├── discovery.py                  # Step 5: Scenario discovery
│   └── io.py                         # Save/load utilities
│
05_vulnerability_analysis.py          # Main script
```

---

## Step 1: Event Definition

### Concept

Events are **storage drawdown periods** — times when the NYC reservoir system is losing water. No inflow thresholds, no drought indices. Just observable storage dynamics.

### Definition

An event begins when:
- Storage reaches a local peak (higher than neighboring weeks)

An event ends when:
- Storage reaches a local trough AND
- System returns to FFMP Level ≤ 2 (Normal or better) for ≥ 2 consecutive weeks

### Filters

- **Minimum duration:** 4 weeks (removes transient fluctuations)
- **Minimum drawdown:** 10 percentage points (removes trivial variation)

### Outcome Classification

Each event is classified by outcome:
- **Shortage:** Demand shortage OR flow shortage occurred during event
- **Recovered:** Storage declined but no shortage occurred

### Implementation: `methods/vulnerability/events.py`

```python
def identify_drawdown_events(
    weekly_ts: pd.DataFrame,
    min_duration_weeks: int = 4,
    min_drawdown_pct: float = 10.0,
    recovery_zone_threshold: int = 2,  # FFMP level
    recovery_persistence_weeks: int = 2
) -> pd.DataFrame:
    """
    Identify storage drawdown events from weekly time series.
    
    Parameters
    ----------
    weekly_ts : pd.DataFrame
        Weekly time series with columns:
        - realization_id
        - week (absolute week index)
        - storage_pct (NYC aggregate storage as % of capacity)
        - ffmp_zone (1-7, where 1-2 = normal)
        - any_shortage (binary)
    min_duration_weeks : int
        Minimum event duration to include
    min_drawdown_pct : float  
        Minimum peak-to-trough storage decline
    recovery_zone_threshold : int
        FFMP zone that indicates recovery (≤ this value)
    recovery_persistence_weeks : int
        Consecutive weeks at recovery zone to end event
        
    Returns
    -------
    events : pd.DataFrame
        One row per event with columns:
        - event_id
        - realization_id
        - start_week, end_week
        - peak_week, trough_week
        - duration_weeks
        - storage_at_peak, storage_at_trough
        - drawdown_pct
        - outcome ('shortage' or 'recovered')
        - shortage_week (if applicable)
    """
```

**Algorithm outline:**

1. For each realization:
   - Find local maxima in storage (peaks)
   - From each peak, walk forward tracking:
     - Running minimum storage (trough)
     - Whether shortage occurred
     - FFMP zone history
   - End event when recovery condition met (zone ≤ 2 for 2+ weeks) or end of series
   - Apply duration and drawdown filters
   - Record event with outcome

### Key Design Decisions

**Why storage-based rather than inflow-based:**
- Storage integrates all forcing (inflow, demand, operations)
- No arbitrary threshold on "what is a drought"
- Events are defined by system response, not forcing

**Why FFMP zone for recovery:**
- Operationally meaningful (system returns to normal operations)
- More robust than storage threshold (accounts for seasonality)
- Persistence requirement avoids premature event termination

**Why binary outcome:**
- Avoids distributional pathologies of continuous satisfaction ratios
- Clear classification task for scenario discovery
- Either the system failed during the event or it didn't

---

## Step 2: Feature Extraction

### Concept

For each event, extract features that characterize:
- What was the state when the event started?
- What was the history before the event?
- What happened during the event?
- What was the operational response?

### Feature Categories

#### A. Onset Conditions (State at Event Start)

| Feature | Description | Source |
|---------|-------------|--------|
| `storage_pct_onset` | Storage at peak (event start) | storage_pct at peak_week |
| `week_of_year_onset` | Seasonality | week_of_year at peak_week |
| `sin_week_onset`, `cos_week_onset` | Harmonic encoding of season | Computed |
| `ffmp_zone_onset` | FFMP zone at start | ffmp_zone at peak_week |
| `storage_trend_4wk_onset` | Storage trajectory entering event | Slope of storage over prior 4 weeks |
| `days_since_last_recharge` | Time since last significant storage increase | Computed from storage history |

#### B. Antecedent Conditions (History Before Event)

| Feature | Description | Source |
|---------|-------------|--------|
| `inflow_mean_4wk_pre` | Mean inflow 4 weeks before onset | inflow_agg, lagged |
| `inflow_mean_12wk_pre` | Mean inflow 12 weeks before onset | inflow_agg, lagged |
| `inflow_std_12wk_pre` | Standardized inflow deficit 12 weeks pre | Computed from climatology |
| `demand_mean_4wk_pre` | Mean demand 4 weeks before | demand, lagged |
| `cum_deficit_12wk_pre` | Cumulative (demand - inflow) 12 weeks pre | Computed |
| `n_stress_weeks_12wk_pre` | Count of weeks with inflow < demand in prior 12 weeks | Computed |

#### C. Forcing During Event

| Feature | Description | Source |
|---------|-------------|--------|
| `inflow_mean_during` | Mean inflow during event | Mean of inflow_agg during event |
| `inflow_min_during` | Minimum weekly inflow | Min of inflow_agg during event |
| `inflow_std_mean_during` | Mean standardized inflow | Mean of inflow_std during event |
| `inflow_std_min_during` | Worst inflow anomaly | Min of inflow_std during event |
| `demand_mean_during` | Mean demand during event | Mean of demand during event |
| `demand_max_during` | Peak demand during event | Max of demand during event |
| `demand_std_max_during` | Worst demand anomaly | Max of demand_std during event |
| `net_deficit_during` | Cumulative (demand - inflow) | Sum during event |
| `duration_weeks` | Event duration | end_week - start_week |
| `drawdown_rate` | Rate of storage decline | drawdown_pct / duration_weeks |

#### D. Derived/Interaction Features

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `storage_x_inflow_deficit` | storage_pct_onset × inflow_std_mean_during | Low storage + bad inflow interaction |
| `onset_storage_buffer` | storage_pct_onset - 30 | Buffer above critical threshold |
| `compound_stress` | demand_std_max_during - inflow_std_min_during | Combined forcing intensity |
| `antecedent_x_during` | cum_deficit_12wk_pre × net_deficit_during | Pre-stressed + continued stress |
| `seasonal_risk` | 1 if onset in Jul-Oct, else 0 | High-risk season indicator |
| `late_season_low_storage` | (onset in Jul-Oct) AND (storage < 60) | Risky seasonal timing |

### Implementation: `methods/vulnerability/features.py`

```python
def extract_event_features(
    events: pd.DataFrame,
    weekly_ts: pd.DataFrame,
    climatology: pd.DataFrame
) -> pd.DataFrame:
    """
    Extract comprehensive feature set for each event.
    
    Parameters
    ----------
    events : pd.DataFrame
        Events from identify_drawdown_events()
    weekly_ts : pd.DataFrame
        Full weekly time series
    climatology : pd.DataFrame
        Weekly climatology (mean, std by week of year)
        
    Returns
    -------
    event_features : pd.DataFrame
        Events with all extracted features
        Columns include all features from categories A-D plus outcome
    """
```

**Implementation approach:**

1. For each event, extract the relevant slice of weekly_ts
2. Compute each feature category
3. Handle edge cases (events near start of realization, missing data)
4. Return merged DataFrame with events + features

### Expected Feature Count

- Onset: ~8 features
- Antecedent: ~8 features  
- During: ~10 features
- Derived: ~8 features
- **Total: ~34 features**

---

## Step 3: Feature Selection

### Concept

Apply multiple feature selection methods to identify which features best discriminate shortage from recovered events. Methods with different assumptions provide robustness.

### Method A: Mutual Information

**Approach:** Measure statistical dependence between each feature and the outcome, without assuming linear relationships.

```python
from sklearn.feature_selection import mutual_info_classif

def compute_mutual_information(
    X: pd.DataFrame,
    y: pd.Series,
    n_neighbors: int = 5
) -> pd.Series:
    """
    Compute mutual information between each feature and outcome.
    
    Returns
    -------
    mi_scores : pd.Series
        MI score for each feature, sorted descending
    """
    mi = mutual_info_classif(X, y, n_neighbors=n_neighbors, random_state=42)
    return pd.Series(mi, index=X.columns).sort_values(ascending=False)
```

**Advantages:**
- Detects nonlinear relationships
- No model assumptions
- Fast

**Limitations:**
- Univariate (ignores feature interactions)
- Sensitive to n_neighbors parameter

### Method B: SHAP Values from Gradient Boosting

**Approach:** Fit XGBoost model, compute SHAP values to understand feature contributions.

```python
import xgboost as xgb
import shap

def compute_shap_importance(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict = None
) -> Tuple[pd.Series, shap.Explanation]:
    """
    Fit XGBoost and compute SHAP-based feature importance.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Binary outcome
    params : dict
        XGBoost parameters (defaults provided)
        
    Returns
    -------
    shap_importance : pd.Series
        Mean |SHAP| for each feature, sorted descending
    shap_values : shap.Explanation
        Full SHAP values for further analysis
    """
    if params is None:
        params = {
            'objective': 'binary:logistic',
            'max_depth': 4,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
    
    model = xgb.XGBClassifier(**params)
    model.fit(X, y)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    
    # Mean absolute SHAP value per feature
    importance = pd.Series(
        np.abs(shap_values.values).mean(axis=0),
        index=X.columns
    ).sort_values(ascending=False)
    
    return importance, shap_values
```

**Advantages:**
- Captures interactions (via the model)
- Theoretically grounded (Shapley values)
- Provides instance-level explanations

**Limitations:**
- Dependent on model fit quality
- Computationally more expensive

### Implementation: `methods/vulnerability/selection.py`

```python
def run_feature_selection(
    event_features: pd.DataFrame,
    outcome_col: str = 'outcome',
    methods: List[str] = ['mutual_information', 'shap']
) -> Dict[str, pd.Series]:
    """
    Run multiple feature selection methods.
    
    Returns
    -------
    rankings : Dict[str, pd.Series]
        Feature importance rankings from each method
    """
```

---

## Step 4: Consensus Features

### Concept

Features that rank highly across multiple methods are more likely to be genuinely important rather than method-specific artifacts.

### Approach

1. Normalize rankings to [0, 1] scale (1 = most important)
2. Compute mean rank across methods
3. Identify consensus top features
4. Flag features with high disagreement between methods

```python
def identify_consensus_features(
    rankings: Dict[str, pd.Series],
    top_k: int = 15,
    min_agreement_threshold: float = 0.7
) -> Tuple[List[str], pd.DataFrame]:
    """
    Identify features with consensus importance across methods.
    
    Parameters
    ----------
    rankings : Dict[str, pd.Series]
        Rankings from each method (from run_feature_selection)
    top_k : int
        Number of top features to consider from each method
    min_agreement_threshold : float
        Minimum normalized mean rank to be considered consensus
        
    Returns
    -------
    consensus_features : List[str]
        Features with consistent high importance
    ranking_comparison : pd.DataFrame
        Full comparison of rankings across methods
    """
```

### Output

**Example consensus features output:**

| Feature | MI Rank | SHAP Rank | Mean Rank | Consensus |
|---------|---------|-----------|-----------|-----------|
| storage_pct_onset | 1 | 2 | 1.5 | Yes |
| inflow_std_min_during | 2 | 1 | 1.5 | Yes |
| cum_deficit_12wk_pre | 3 | 4 | 3.5 | Yes |
| net_deficit_during | 5 | 3 | 4.0 | Yes |
| demand_std_max_during | 4 | 8 | 6.0 | Yes |
| ... | | | | |
| sin_week_onset | 15 | 28 | 21.5 | No |

---

## Step 5: Scenario Discovery

### Concept

Use consensus features to build interpretable models that reveal decision boundaries between shortage and recovered events.

### Method A: CART (Classification Tree)

**Purpose:** Generate explicit, interpretable rules.

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree

def fit_scenario_tree(
    X: pd.DataFrame,
    y: pd.Series,
    max_depth: int = 4,
    min_samples_leaf: int = 50
) -> Tuple[DecisionTreeClassifier, Dict]:
    """
    Fit classification tree for scenario discovery.
    
    Returns
    -------
    tree : DecisionTreeClassifier
        Fitted tree
    rules : Dict
        Extracted decision rules with support and shortage rates
    """
```

**Output:** Rules like:
```
IF storage_pct_onset < 52.3
AND inflow_std_min_during < -1.4
AND cum_deficit_12wk_pre < -8.2
THEN shortage_prob = 0.67 (n=342)
```

### Method B: Factor Maps

**Purpose:** Visualize shortage/recovery regions in 2D feature space.

```python
def create_factor_map(
    event_features: pd.DataFrame,
    x_feature: str,
    y_feature: str,
    outcome_col: str = 'outcome',
    grid_resolution: int = 50,
    model = None  # Optional: fitted model for decision boundary
) -> matplotlib.figure.Figure:
    """
    Create 2D factor map showing shortage regions.
    
    Visualizes:
    - Scatter of events colored by outcome
    - Contours of shortage probability (if model provided)
    - Marginal distributions
    """
```

**Key factor maps to generate:**

1. `storage_pct_onset` vs `inflow_std_min_during` — State × Forcing
2. `cum_deficit_12wk_pre` vs `net_deficit_during` — Antecedent × During
3. `storage_pct_onset` vs `week_of_year_onset` — State × Seasonality
4. Top 2 consensus features (whatever they are)

### Implementation: `methods/vulnerability/discovery.py`

```python
def run_scenario_discovery(
    event_features: pd.DataFrame,
    consensus_features: List[str],
    output_dir: Path
) -> Dict:
    """
    Run full scenario discovery analysis.
    
    Produces:
    - Fitted CART model and extracted rules
    - Factor maps for key feature pairs
    - Summary statistics
    
    Returns
    -------
    results : Dict
        - 'tree': fitted DecisionTreeClassifier
        - 'rules': extracted decision rules
        - 'feature_importance': tree-based importance
        - 'figures': paths to generated figures
    """
```

---

## Main Script: `05_vulnerability_analysis.py`

```python
#!/usr/bin/env python3
"""
Data-driven vulnerability event analysis.

Identifies storage drawdown events, extracts features, selects important
features via multiple methods, and performs scenario discovery.

Usage:
    python 05_vulnerability_analysis.py <dataset_id>
    
Example:
    python 05_vulnerability_analysis.py stationary_ensemble
"""

import argparse
from pathlib import Path

from methods.config import verify_dataset_id
from methods.verification import verify_postprocessing_output
from methods.postprocess import load_combined_data
from methods.vulnerability.config import VulnerabilityConfig
from methods.vulnerability.events import identify_drawdown_events
from methods.vulnerability.features import (
    extract_event_features,
    compute_climatology_for_features
)
from methods.vulnerability.selection import (
    run_feature_selection,
    identify_consensus_features
)
from methods.vulnerability.discovery import run_scenario_discovery
from methods.vulnerability.io import save_vulnerability_results


def main(dataset_id: str):
    """Main vulnerability analysis workflow."""
    
    print("=" * 70)
    print(f"VULNERABILITY EVENT ANALYSIS: {dataset_id}")
    print("=" * 70)
    
    # Verify prerequisites
    verify_dataset_id(dataset_id)
    verify_postprocessing_output(dataset_id)
    
    # Load configuration
    config = VulnerabilityConfig(dataset_id=dataset_id)
    
    # Step 0: Load data and prepare weekly time series
    print("\n[Step 0] Loading data...")
    weekly_ts = load_and_prepare_weekly_data(dataset_id, config)
    print(f"  Loaded {len(weekly_ts):,} weekly observations")
    print(f"  Realizations: {weekly_ts['realization_id'].nunique()}")
    
    # Step 1: Identify events
    print("\n[Step 1] Identifying drawdown events...")
    events = identify_drawdown_events(
        weekly_ts,
        min_duration_weeks=config.min_duration_weeks,
        min_drawdown_pct=config.min_drawdown_pct,
        recovery_zone_threshold=config.recovery_zone_threshold,
        recovery_persistence_weeks=config.recovery_persistence_weeks
    )
    n_shortage = (events['outcome'] == 'shortage').sum()
    n_recovered = (events['outcome'] == 'recovered').sum()
    print(f"  Identified {len(events):,} events")
    print(f"    Shortage: {n_shortage:,} ({100*n_shortage/len(events):.1f}%)")
    print(f"    Recovered: {n_recovered:,} ({100*n_recovered/len(events):.1f}%)")
    
    # Step 2: Extract features
    print("\n[Step 2] Extracting event features...")
    climatology = compute_climatology_for_features(weekly_ts)
    event_features = extract_event_features(events, weekly_ts, climatology)
    print(f"  Extracted {len(event_features.columns) - len(events.columns)} features")
    
    # Step 3: Feature selection
    print("\n[Step 3] Running feature selection...")
    feature_cols = [c for c in event_features.columns 
                    if c not in ['event_id', 'realization_id', 'outcome', 
                                 'start_week', 'end_week', 'peak_week', 'trough_week',
                                 'shortage_week']]
    
    X = event_features[feature_cols].copy()
    y = (event_features['outcome'] == 'shortage').astype(int)
    
    # Handle missing values
    X = X.fillna(X.median())
    
    rankings = run_feature_selection(X, y, methods=['mutual_information', 'shap'])
    
    for method, ranking in rankings.items():
        print(f"\n  {method} top 10:")
        for i, (feat, score) in enumerate(ranking.head(10).items()):
            print(f"    {i+1}. {feat}: {score:.4f}")
    
    # Step 4: Consensus features
    print("\n[Step 4] Identifying consensus features...")
    consensus_features, ranking_comparison = identify_consensus_features(
        rankings, 
        top_k=config.consensus_top_k
    )
    print(f"  Consensus features ({len(consensus_features)}):")
    for feat in consensus_features:
        print(f"    - {feat}")
    
    # Step 5: Scenario discovery
    print("\n[Step 5] Running scenario discovery...")
    discovery_results = run_scenario_discovery(
        event_features,
        consensus_features,
        output_dir=config.output_dir
    )
    
    print("\n  Decision tree rules:")
    for rule in discovery_results['rules'][:5]:  # Top 5 rules
        print(f"    {rule}")
    
    # Save results
    print("\n[Step 6] Saving results...")
    save_vulnerability_results(
        events=events,
        event_features=event_features,
        rankings=rankings,
        consensus_features=consensus_features,
        discovery_results=discovery_results,
        config=config
    )
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {config.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Data-driven vulnerability event analysis"
    )
    parser.add_argument("dataset_id", type=str, help="Dataset identifier")
    args = parser.parse_args()
    
    main(args.dataset_id)
```

---

## Configuration: `methods/vulnerability/config.py`

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from methods.config import ROOT_DIR


@dataclass
class VulnerabilityConfig:
    """Configuration for vulnerability event analysis."""
    
    dataset_id: str = 'stationary_ensemble'
    
    # Event definition parameters
    min_duration_weeks: int = 4
    min_drawdown_pct: float = 10.0
    recovery_zone_threshold: int = 2  # FFMP zone ≤ this = recovered
    recovery_persistence_weeks: int = 2
    
    # Feature extraction
    antecedent_windows: List[int] = field(default_factory=lambda: [4, 12])
    
    # Feature selection
    consensus_top_k: int = 15
    shap_xgb_params: dict = field(default_factory=lambda: {
        'objective': 'binary:logistic',
        'max_depth': 4,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'random_state': 42
    })
    
    # Scenario discovery
    tree_max_depth: int = 4
    tree_min_samples_leaf: int = 50
    
    # Output
    output_dir: Path = field(
        default_factory=lambda: Path(ROOT_DIR) / "pywrdrb" / "vulnerability_analysis"
    )
    figure_format: str = 'png'
    figure_dpi: int = 300
```

---

## Output Structure

```
pywrdrb/vulnerability_analysis/
├── stationary_ensemble/
│   ├── events.parquet                    # All identified events
│   ├── event_features.parquet            # Events with all features
│   ├── feature_rankings.csv              # Rankings from all methods
│   ├── consensus_features.json           # Selected feature list
│   ├── scenario_tree.pkl                 # Fitted CART model
│   ├── decision_rules.txt                # Extracted rules (human readable)
│   ├── figures/
│   │   ├── factor_map_storage_vs_inflow.png
│   │   ├── factor_map_antecedent_vs_during.png
│   │   ├── factor_map_storage_vs_season.png
│   │   ├── decision_tree.png
│   │   ├── shap_summary.png
│   │   └── feature_importance_comparison.png
│   └── config.yaml                       # Configuration used
```

---

## Validation Approach

### Train/Test Split

Split realizations (not events) to ensure independence:

```python
def split_events_by_realization(
    event_features: pd.DataFrame,
    test_fraction: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split events by realization for validation."""
    
    realizations = event_features['realization_id'].unique()
    n_test = int(len(realizations) * test_fraction)
    
    rng = np.random.RandomState(random_state)
    test_realizations = rng.choice(realizations, n_test, replace=False)
    
    test_mask = event_features['realization_id'].isin(test_realizations)
    
    return event_features[~test_mask], event_features[test_mask]
```

### Metrics to Report

- **Event statistics:** Count, class balance, duration distribution
- **Feature selection agreement:** Rank correlation between methods
- **Tree performance:** Accuracy, precision, recall on test set
- **Rule quality:** Coverage and precision of top rules

---

## Dependencies

```
# Add to requirements.txt or environment
xgboost
shap
scikit-learn  # Already likely present
```

---

## Implementation Priority

1. **`events.py`** — Event identification (foundation for everything)
2. **`features.py`** — Feature extraction (enables analysis)
3. **`selection.py`** — Feature selection methods
4. **`discovery.py`** — CART and factor maps
5. **`05_vulnerability_analysis.py`** — Main script
6. **`io.py`** — Save/load utilities

---

## Notes for Implementation

### Data Loading

Reuse existing patterns from `methods/postprocess.py`:
- `load_combined_data()` or `load_episode_analysis_data()`
- `preprocess_to_weekly()` for aggregation
- `compute_weekly_climatology()` for standardization

### FFMP Zone Mapping

Pywr-DRB uses numeric zones. Verify mapping:
- Zone 1-2: Normal operations (1a, 1b, 1c treated as 1-2)
- Zone 3+: Drought operations

Check `data.res_level` for zone values and confirm threshold.

### Edge Cases

- Events at start of realization (limited antecedent data)
- Events at end of realization (may not reach recovery)
- Realizations with no events (possible if no significant drawdowns)
- Missing data in features (use median imputation)

### Efficiency

- Event identification: O(realizations × weeks) — fast
- Feature extraction: O(events × features) — moderate
- SHAP computation: O(events × trees × features) — slowest step
- Consider subsetting for initial testing (100 realizations)

---

## Expected Outcomes

### Primary Outputs

1. **Consensus feature list:** Which factors matter for shortage prediction
2. **Decision rules:** Interpretable conditions leading to shortage
3. **Factor maps:** Visual representation of vulnerability regions

### Scientific Questions Answered

- What storage level at event onset predicts shortage?
- How much does antecedent deficit matter vs. forcing during event?
- Are there seasonal patterns in vulnerability?
- What feature interactions are important?

### Comparison Across Climate Scenarios

Run separately on each dataset, compare:
- Event frequency (more/fewer drawdown events?)
- Class balance (higher/lower shortage rate?)
- Feature importance (same factors matter?)
- Decision boundaries (thresholds shift?)

---

*End of Implementation Plan*