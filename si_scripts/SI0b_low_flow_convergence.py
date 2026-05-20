"""
Standalone ensemble convergence diagnostics (low-flow + drought metrics).

Produces convergence plots that can be (re)generated without running the
full SI0 suite:

LOW-FLOW EXTREMES (ensemble MIN across realizations × water years):
  1. Minimum 7-day mean flow (acute drought extreme).
  2. Minimum annual Q95 (chronic-low-flow-year extreme).
  Sites: NYC aggregate inflow and delMontague.

DROUGHT-EVENT METRICS (pooled mean across drought events; one figure per
SSI window from SSI_WINDOWS):
  Default panels: mean duration, mean magnitude, mean severity.
  Loaded from outputs/.../data/drought_metrics/{dataset_id}_ssi{N}_drought_events.csv.

Bootstrap median should decrease monotonically with N for ensemble-min
metrics and flatten for pooled-mean metrics. No averaging is applied to the
low-flow panels; the drought panels use pooled-mean by design.

Usage:
    python SI0b_low_flow_convergence.py <dataset_id>
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from methods.plotting.ensemble_summary import (
    plot_low_flow_convergence,
    plot_drought_metric_convergence,
)
from methods.load import (
    load_baseline_historical_flow,
    load_and_combine_ensemble_sets,
    load_drought_events,
)
from methods.config import *
from methods.ensemble_utils import ENSEMBLE_SETS

FIG_DIR = f"{FIG_DIR}/SI0_ensemble_diagnostics/convergence"


def run_low_flow_convergence(dataset_id: str) -> bool:
    verify_dataset_id(dataset_id)
    dataset_config = DATASET_CONFIGS[dataset_id]
    print(f"Low-flow convergence for: {dataset_id} ({dataset_config['type']})")

    ensemble_set_specs = ENSEMBLE_SETS[dataset_id]
    missing = [s.set_id + 1 for s in ensemble_set_specs
               if not os.path.exists(s.files['gage_flow'])]
    if missing:
        print(f"ERROR: Missing ensemble sets: {missing}")
        print("Run ensemble generation (01_generate_ensemble_sets.py) first!")
        return False

    os.makedirs(FIG_DIR, exist_ok=True)

    print("Loading historical data...")
    Q = load_baseline_historical_flow(
        period='baseline', gage_flow=True, flowtype=BASELINE_DATASET
    )
    Q.replace(0, np.nan, inplace=True)
    Q.drop(columns=['delTrenton'], inplace=True, errors='ignore')

    print("Loading ensemble (by_site=True)...")
    Q_syn = load_and_combine_ensemble_sets(ensemble_set_specs, by_site=True)
    print("Loading ensemble (by_site=False) for NYC aggregation...")
    syn_ensemble = load_and_combine_ensemble_sets(ensemble_set_specs, by_site=False)

    realization_ids = list(syn_ensemble.keys())
    print(f"  N realizations: {len(realization_ids)}")

    print("Building NYC aggregate (cannonsville + pepacton + neversink)...")
    nyc_syn = pd.concat(
        {rid: syn_ensemble[rid][NYC_RESERVOIRS].sum(axis=1)
         for rid in realization_ids},
        axis=1,
    )
    nyc_obs = Q[NYC_RESERVOIRS].sum(axis=1)

    targets = [
        ('nyc_aggregate', 'NYC aggregate', nyc_syn, nyc_obs),
        ('delMontague', 'delMontague', Q_syn.get('delMontague'), Q.get('delMontague')),
    ]

    for site_id, site_label, syn_df, obs_series in targets:
        if syn_df is None:
            print(f"  Skipping {site_id} (not in ensemble)")
            continue
        print(f"  Plotting low-flow convergence for {site_label}...")
        fname = f"{FIG_DIR}/{dataset_id}_{site_id}_low_flow_convergence.png"
        plot_low_flow_convergence(
            Q_syn_site=syn_df,
            realization_ids=realization_ids,
            site_label=site_label,
            Q_obs=obs_series,
            fname=fname,
        )

    # Drought-event metric convergence (mean, one figure per SSI window)
    print("\nDrought metric convergence (pooled-mean):")
    for ssi_window in SSI_WINDOWS:
        print(f"  SSI-{ssi_window}: loading drought events...")
        try:
            droughts = load_drought_events(
                dataset_id, ssi_window=ssi_window, observed=False,
            )
        except FileNotFoundError as e:
            print(f"    Skipping SSI-{ssi_window} (events file missing): {e}")
            continue

        try:
            obs_droughts = load_drought_events(
                dataset_id, ssi_window=ssi_window, observed=True,
            )
        except FileNotFoundError:
            obs_droughts = None

        n_events = len(droughts)
        n_obs_events = 0 if obs_droughts is None else len(obs_droughts)
        print(f"    {n_events} ensemble events; {n_obs_events} observed events")

        fname = f"{FIG_DIR}/{dataset_id}_drought_metric_mean_convergence_ssi{ssi_window}.png"
        plot_drought_metric_convergence(
            droughts=droughts,
            realization_ids=realization_ids,
            ssi_window=ssi_window,
            obs_droughts=obs_droughts,
            fname=fname,
        )

    print(f"Done. Figures in: {FIG_DIR}")
    return True


def main(dataset_id: str):
    print("=" * 60)
    print(f"LOW-FLOW ENSEMBLE CONVERGENCE (SI0b): {dataset_id}")
    print("=" * 60)
    ok = run_low_flow_convergence(dataset_id)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python SI0b_low_flow_convergence.py <dataset_id>")
        print(f"Available datasets: {list(DATASET_CONFIGS.keys())}")
        sys.exit(1)
    dataset_id = sys.argv[1]
    verify_dataset_id(dataset_id)
    main(dataset_id)
