"""
Core functions for drought analysis using SSI-based metrics.

This module contains functions for:
- Calculating SSI-based drought metrics for historical and ensemble data
- Processing drought events across multiple SSI windows
- Calculating satisficing conditions during drought/non-drought years
"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import pywrdrb
from sglib.droughts.ssi import SSIDroughtMetrics, SSI

from .config import BASELINE_DATASET
from .load import load_baseline_historical_flow, load_wrf1960s_historical_flow
from .metrics.satisficing import (
    calculate_satisficing_conditions,
    calculate_satisficing_during_droughts,
    calculate_satisficing_non_drought_periods
)


def calculate_historic_observed_droughts(ssi_windows, output_dir):
    """
    Calculate SSI-based drought metrics for historic observed data.

    Parameters
    ----------
    ssi_windows : list of int
        SSI window sizes in months (e.g., [3, 6, 12])
    output_dir : str
        Directory to save drought metrics

    Returns
    -------
    bool
        True if successful
    """
    print("=" * 60)
    print("CALCULATING HISTORIC OBSERVED DROUGHTS")
    print("=" * 60)

    # Load historic reconstruction data
    Q = load_baseline_historical_flow(gage_flow=True, period='full', flowtype=BASELINE_DATASET)
    Q.replace(0, np.nan, inplace=True)
    Q.drop(columns=['delTrenton'], inplace=True)

    if BASELINE_DATASET == 'wrfaorc_withObsScaled':
        Q_1960s = load_wrf1960s_historical_flow(gage_flow=True)
        Q_1960s.replace(0, np.nan, inplace=True)
        Q_1960s.drop(columns=['delTrenton'], inplace=True)
        # Combine the two datasets
        Q_full = pd.concat([Q_1960s, Q], axis=0).sort_index()
        Q_full.replace(0, np.nan, inplace=True)
        Q_full.dropna(axis=0, how='any', inplace=True)
    else:
        Q_full = Q.copy()

    # Calculate nyc_aggregate for historical data
    nyc_gages = ["01425000", "01417000", "01436000"]
    Q_full['nyc_aggregate'] = Q_full[nyc_gages].sum(axis=1)
    Q_monthly = Q_full.resample('MS').sum()
    Q_monthly.replace(0, np.nan, inplace=True)
    Q_monthly.dropna(axis=0, how='any', inplace=True)

    print(f"Loaded historic data with {Q_full.shape[0] // 365} years of daily data for {Q_full.shape[1]} sites.")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    node = 'nyc_aggregate'

    # Process each SSI window
    for ssi_window in ssi_windows:
        print(f"\nProcessing SSI window: {ssi_window} months")

        # Initialize calculators
        drought_calculator = SSIDroughtMetrics()
        ssi_calculator = SSI(normal_scores_transform=False, timescale=ssi_window)

        # Fit SSI on historical data
        ssi_calculator.fit(Q_monthly.loc[:, node])

        # Calculate SSI for historical data
        ssi_obs = ssi_calculator.transform(Q_monthly.loc[:, node])
        obs_droughts = drought_calculator.calculate_drought_metrics(ssi_obs)

        # Save observed drought metrics
        obs_droughts.reset_index(inplace=True, drop=True)

        # If baseline dataset is wrfaorc_withObsScaled, drop droughts spanning 1960s-1970s gap
        if BASELINE_DATASET == 'wrfaorc_withObsScaled':
            obs_droughts = obs_droughts[~((obs_droughts['start'] < pd.to_datetime('1970-01-01')) &
                                           (obs_droughts['end'] > pd.to_datetime('1979-01-01')))]

        obs_fname = f"{output_dir}/observed_ssi{ssi_window}_drought_events.csv"
        obs_droughts.to_csv(obs_fname, index=False)
        print(f"  Saved observed drought metrics: {obs_fname}")

    print("\n" + "=" * 60)
    print("Historic observed droughts calculation completed!")
    print("=" * 60)
    return True


def calculate_ensemble_droughts(dataset_id, ssi_windows, output_dir):
    """
    Calculate SSI-based drought metrics for ensemble realizations.

    Note: This is a SERIAL version (no MPI). For large ensembles, use the MPI version.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    ssi_windows : list of int
        SSI window sizes in months
    output_dir : str
        Directory to save drought metrics

    Returns
    -------
    bool
        True if successful
    """
    print("=" * 80)
    print(f"CALCULATING ENSEMBLE DROUGHTS: {dataset_id}")
    print("=" * 80)

    # Load postprocessed data
    fname = f'./pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5'
    if not os.path.exists(fname):
        raise FileNotFoundError(
            f"Postprocessed data not found: {fname}\n"
            "Run step 4 (postprocessing) first!"
        )

    print(f"  Loading inflow data from {fname}...")
    data = pywrdrb.Data()
    data.load_from_export(fname, results_sets=['inflow'])

    # Get realizations
    realizations = sorted(data.inflow[dataset_id].keys())
    print(f"  Found {len(realizations)} realizations")

    # Load historic data for SSI fitting
    Q = load_baseline_historical_flow(gage_flow=True, period='full', flowtype=BASELINE_DATASET)
    Q.replace(0, np.nan, inplace=True)
    Q.drop(columns=['delTrenton'], inplace=True)

    if BASELINE_DATASET == 'wrfaorc_withObsScaled':
        Q_1960s = load_wrf1960s_historical_flow(gage_flow=True)
        Q_1960s.replace(0, np.nan, inplace=True)
        Q_1960s.drop(columns=['delTrenton'], inplace=True)
        Q_full = pd.concat([Q_1960s, Q], axis=0).sort_index()
        Q_full.replace(0, np.nan, inplace=True)
        Q_full.dropna(axis=0, how='any', inplace=True)
    else:
        Q_full = Q.copy()

    nyc_gages = ["01425000", "01417000", "01436000"]
    Q_full['nyc_aggregate'] = Q_full[nyc_gages].sum(axis=1)
    Q_monthly = Q_full.resample('MS').sum()
    Q_monthly.replace(0, np.nan, inplace=True)
    Q_monthly.dropna(axis=0, how='any', inplace=True)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    node = 'nyc_aggregate'

    # Process each SSI window
    for ssi_window in ssi_windows:
        print(f"\nProcessing SSI-{ssi_window}...")

        # Initialize calculators
        drought_calculator = SSIDroughtMetrics()
        ssi_calculator = SSI(normal_scores_transform=False, timescale=ssi_window)

        # Fit SSI on historical data
        ssi_calculator.fit(Q_monthly.loc[:, node])

        # Process each realization
        all_droughts = []

        for r in realizations:
            if r % 10 == 0 and r > 0:
                print(f"    Processed {r}/{len(realizations)} realizations...")

            # Get NYC inflow for this realization
            nyc_inflow = data.inflow[dataset_id][r]['nyc']

            # Resample to monthly
            nyc_inflow_monthly = nyc_inflow.resample('MS').sum()
            nyc_inflow_monthly.replace(0, np.nan, inplace=True)
            nyc_inflow_monthly.dropna(inplace=True)

            # Calculate SSI
            ssi_r = ssi_calculator.transform(nyc_inflow_monthly)

            # Calculate drought metrics
            droughts_r = drought_calculator.calculate_drought_metrics(ssi_r)
            droughts_r['realization_id'] = r

            all_droughts.append(droughts_r)

        # Combine all realizations
        ensemble_droughts = pd.concat(all_droughts, ignore_index=True)

        # Save
        fname_out = f"{output_dir}/{dataset_id}_ssi{ssi_window}_drought_events.csv"
        ensemble_droughts.to_csv(fname_out, index=False)
        print(f"  Saved: {fname_out}")
        print(f"  Total drought events: {len(ensemble_droughts)}")

    print("\n" + "=" * 80)
    print(f"ENSEMBLE DROUGHT CALCULATION COMPLETE: {dataset_id}")
    print("=" * 80)

    return True


def calculate_satisficing_by_drought(dataset_id, ssi_window, output_dir):
    """
    Calculate satisficing conditions for drought/non-drought years.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    ssi_window : int
        SSI window size
    output_dir : str
        Directory to save satisficing results

    Returns
    -------
    bool
        True if successful
    """
    from .load import load_drought_events
    from .verification import verify_postprocessing_output

    print("=" * 80)
    print(f"SATISFICING ANALYSIS: {dataset_id}, SSI-{ssi_window}")
    print("=" * 80)

    # Verify postprocessed data exists
    verify_postprocessing_output(dataset_id)

    # Load postprocessed data
    fname = f'./pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5'
    print(f"  Loading data from {fname}...")

    data = pywrdrb.Data()
    data.load_from_export(
        fname,
        results_sets=['res_storage', 'shortage', 'mrf_target', 'inflow', 'contribution']
    )

    # Load drought events
    print(f"  Loading drought events (SSI-{ssi_window})...")
    drought_events_df = load_drought_events(dataset_id, ssi_window)

    if drought_events_df is None or len(drought_events_df) == 0:
        print(f"  ERROR: No drought events found for {dataset_id}, SSI-{ssi_window}")
        print(f"  Run step 5 (drought metrics calculation) first!")
        return False

    print(f"  Loaded {len(drought_events_df)} drought events")

    # Calculate satisficing for all years
    print("\nCalculating: All Years...")
    all_years_results = calculate_satisficing_conditions(
        data, dataset_id,
        period_type='year',
        evaluate_all_years=True,
        storage_threshold=20.0,
        violation_days=3
    )
    print(f"  Evaluated {len(all_years_results)} year-realization pairs")

    # Calculate satisficing for years with droughts
    print("\nCalculating: Years with Droughts...")
    drought_results = calculate_satisficing_during_droughts(
        data, dataset_id, drought_events_df,
        storage_threshold=20.0,
        violation_days=3
    )
    print(f"  Evaluated {len(drought_results)} year-realization pairs with droughts")

    # Calculate satisficing for years without droughts
    print("\nCalculating: Years without Droughts...")
    non_drought_results = calculate_satisficing_non_drought_periods(
        data, dataset_id, drought_events_df,
        storage_threshold=20.0,
        violation_days=3
    )
    print(f"  Evaluated {len(non_drought_results)} year-realization pairs without droughts")

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    fname1 = f"{output_dir}/{dataset_id}_ssi{ssi_window}_all_years.csv"
    all_years_results.to_csv(fname1, index=False)
    print(f"\nSaved: {fname1}")

    fname2 = f"{output_dir}/{dataset_id}_ssi{ssi_window}_years_with_droughts.csv"
    drought_results.to_csv(fname2, index=False)
    print(f"Saved: {fname2}")

    fname3 = f"{output_dir}/{dataset_id}_ssi{ssi_window}_years_without_droughts.csv"
    non_drought_results.to_csv(fname3, index=False)
    print(f"Saved: {fname3}")

    print("\n" + "=" * 80)
    print(f"SATISFICING ANALYSIS COMPLETE: {dataset_id}, SSI-{ssi_window}")
    print("=" * 80)

    return True
