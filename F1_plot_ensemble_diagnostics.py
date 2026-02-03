"""
Generate main manuscript figure for ensemble validation (Figure 1).

This script creates a 2x1 panel figure showing:
- (A) Weekly streamflow percentiles (5th-95th + median) for synthetic vs historic
- (B) Annual flow duration curve ranges for synthetic vs historic

For comprehensive diagnostic plots, see SI0_full_ensemble_diagnostics.py

Usage:
    python F1_plot_ensemble_diagnostics.py <dataset_id>
"""

import sys
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from methods.plotting.ensemble_summary import plot_ensemble_summary_figure
from methods.plotting.styles import apply_publication_style
from methods.load import load_baseline_historical_flow, load_and_combine_ensemble_sets
from methods.config import *


def plot_manuscript_ensemble_figure(dataset_id: str):
    """
    Generate main manuscript figure for ensemble validation.

    Creates a 2x1 figure with:
    - Panel A: Weekly streamflow climatology (5th-95th percentile + median)
    - Panel B: Annual FDC range comparison

    Parameters
    ----------
    dataset_id : str
        Dataset identifier to analyze

    Returns
    -------
    bool
        True if successful
    """
    # Verify dataset
    verify_dataset_id(dataset_id)
    dataset_config = DATASET_CONFIGS[dataset_id]

    print(f"Generating manuscript figure for: {dataset_id}")
    print(f"Dataset type: {dataset_config['type']}")

    # Get ensemble set specs for this dataset
    ensemble_set_specs = ENSEMBLE_SETS[dataset_id]

    # Check if ensemble data exists
    missing_sets = []
    for spec in ensemble_set_specs:
        if not os.path.exists(spec.files['gage_flow']):
            missing_sets.append(spec.set_id + 1)

    if missing_sets:
        print(f"ERROR: Missing ensemble sets: {missing_sets}")
        print("Run ensemble generation (01_generate_ensemble_sets.py) first!")
        return False

    # Load historical data
    print("Loading historical data...")
    Q_historic = load_baseline_historical_flow(period='baseline', gage_flow=False, flowtype=BASELINE_DATASET)
    Q_historic.replace(0, np.nan, inplace=True)
    Q_historic.drop(columns=['delTrenton'], inplace=True, errors='ignore')

    print(f"Loaded historic data: {Q_historic.shape[0]} days, {Q_historic.shape[1]} sites")

    # Load synthetic ensemble (by_site=False for full realizations)
    print("Loading synthetic ensemble...")
    syn_ensemble = load_and_combine_ensemble_sets(ensemble_set_specs, by_site=False)

    n_realizations = len(syn_ensemble)
    print(f"Loaded synthetic ensemble: {n_realizations} realizations")

    # Create output directory
    os.makedirs(f"{FIG_DIR}/MAIN", exist_ok=True)

    # Generate the 2x1 summary figure (uses aggregate NYC reservoir flows)
    print("Generating manuscript figure (aggregate NYC reservoir flows)...")

    fname = f"{FIG_DIR}/MAIN/F1_{dataset_id}_ensemble_diagnostics.png"

    plot_ensemble_summary_figure(
        Q_historic=Q_historic,
        Q_synthetic=syn_ensemble,
        dataset_id=dataset_id,
        fname=fname,
        percentiles=(5, 95),
        figsize=(9, 9),
    )

    print(f"\nManuscript figure saved: {fname}")
    return True


def main(dataset_id: str):
    """Main function."""
    print("=" * 60)
    print(f"ENSEMBLE VALIDATION FIGURE (F1): {dataset_id}")
    print("=" * 60)

    success = plot_manuscript_ensemble_figure(dataset_id)

    if success:
        print("=" * 60)
        print("Manuscript figure generated successfully!")
        print("\nFor comprehensive diagnostics, run:")
        print(f"  python SI0_full_ensemble_diagnostics.py {dataset_id}")
    else:
        print("=" * 60)
        print("ERROR: Figure generation failed!")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python F1_plot_ensemble_diagnostics.py <dataset_id>")
        print(f"Available datasets: {list(DATASET_CONFIGS.keys())}")
        sys.exit(1)

    dataset_id = sys.argv[1]
    verify_dataset_id(dataset_id)

    main(dataset_id)
