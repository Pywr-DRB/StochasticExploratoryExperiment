"""
F1: Ensemble validation figure.

Four-panel figure showing:
- (A) Autocorrelation comparison for synthetic vs historic
- (B) Annual flow duration curve ranges for synthetic vs historic
- (C) Weekly streamflow percentile bands for synthetic vs historic
- (D) Levene & Wilcoxon p-values by month

Usage:
    python F1_plot_ensemble_diagnostics.py <dataset_id>
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from methods.plotting.ensemble_summary import plot_ensemble_summary_figure
from methods.load import load_baseline_historical_flow, load_and_combine_ensemble_sets
from methods.config import (
    FIG_DIR, ENSEMBLE_SETS, DATASET_CONFIGS, BASELINE_DATASET,
    verify_dataset_id,
)

# Output directory
FIG_OUTPUT_DIR = f"{FIG_DIR}/F1_ensemble_diagnostics"


def plot_manuscript_ensemble_figure(dataset_id: str):
    """Generate ensemble validation figure."""
    verify_dataset_id(dataset_id)
    ensemble_set_specs = ENSEMBLE_SETS[dataset_id]

    # Check if ensemble data exists
    missing_sets = [spec.set_id + 1 for spec in ensemble_set_specs
                    if not os.path.exists(spec.files['gage_flow'])]
    if missing_sets:
        print(f"ERROR: Missing ensemble sets: {missing_sets}")
        return False

    # Load data
    print("Loading data...")
    Q_historic = load_baseline_historical_flow(period='baseline', gage_flow=False, flowtype=BASELINE_DATASET)
    Q_historic.replace(0, np.nan, inplace=True)
    Q_historic.drop(columns=['delTrenton'], inplace=True, errors='ignore')

    syn_ensemble = load_and_combine_ensemble_sets(ensemble_set_specs, by_site=False)
    print(f"Loaded {len(syn_ensemble)} realizations")

    # Create output directory and generate figure
    os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)
    fname = f"{FIG_OUTPUT_DIR}/F1_{dataset_id}_ensemble_diagnostics.png"

    plot_ensemble_summary_figure(
        Q_historic=Q_historic,
        Q_synthetic=syn_ensemble,
        dataset_id=dataset_id,
        fname=fname,
        percentiles=(0.5, 99.5),
        figsize=(9, 9),
    )

    print(f"Saved: {fname}")
    return True


def main(dataset_id: str):
    """Main function."""
    print(f"F1: Ensemble validation - {dataset_id}")
    success = plot_manuscript_ensemble_figure(dataset_id)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python F1_plot_ensemble_diagnostics.py <dataset_id>")
        print(f"Available datasets: {list(DATASET_CONFIGS.keys())}")
        sys.exit(1)

    dataset_id = sys.argv[1]
    verify_dataset_id(dataset_id)
    main(dataset_id)