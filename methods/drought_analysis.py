"""
Core functions for drought analysis using SSI-based metrics.

This module contains functions for:
- Fitting the SSI calculator on the historical record
- Calculating SSI drought events for the historical record
- Statistical comparison of drought and non-drought years
"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from synhydro.droughts.ssi import get_drought_metrics, SSI

from .config import BASELINE_DATASET, SSI_NODE, SSI_NODE_CONFIGS
from .load import load_baseline_historical_flow, load_wrf1960s_historical_flow


def fit_ssi_calculator(ssi_window, node=None):
    """
    Fit an SSI calculator on the baseline period (1980-2019) historical flow.

    All SSI calculations (historical record and ensembles) use this helper so
    the SSI parameterization is consistent.

    Parameters
    ----------
    ssi_window : int
        SSI timescale in months (e.g., 3, 6, 12).
    node : str or None
        Column name to fit SSI on. If None, uses SSI_NODE from config.

    Returns
    -------
    ssi_calculator : SSI
        Fitted SSI calculator ready for .transform() calls.
    """
    if node is None:
        node = SSI_NODE
    node_config = SSI_NODE_CONFIGS[node]

    Q_baseline = load_baseline_historical_flow(
        gage_flow=node_config['historical_gage_flow'], 
        period='full', flowtype=BASELINE_DATASET
    )
    Q_baseline.replace(0, np.nan, inplace=True)
    for col in node_config['drop_columns']:
        if col in Q_baseline.columns:
            Q_baseline.drop(columns=[col], inplace=True)

    # If flow is nyc_aggregate then sum the individual reservoirs to get the aggregate flow for SSI fitting
    if node_config['derived']:
        Q_baseline[node] = Q_baseline[node_config['derive_from']].sum(axis=1)

    # Extract only the target node series before resampling/cleanup
    # to avoid dropna removing months due to NaN in unrelated columns
    node_daily = Q_baseline[node].copy()
    node_monthly = node_daily.resample('MS').sum()
    node_monthly.replace(0, np.nan, inplace=True)
    node_monthly.dropna(inplace=True)

    ssi_calculator = SSI(normal_scores_transform=False, timescale=ssi_window)
    ssi_calculator.fit(node_monthly)
    return ssi_calculator


def calculate_historic_observed_droughts(ssi_windows, output_dir, node=None):
    """
    Calculate SSI-based drought metrics for historic observed data.

    Parameters
    ----------
    ssi_windows : list of int
        SSI window sizes in months (e.g., [3, 6, 12])
    output_dir : str
        Directory to save drought metrics
    node : str or None
        Flow node for SSI calculation. If None, uses SSI_NODE from config.

    Returns
    -------
    bool
        True if successful
    """
    if node is None:
        node = SSI_NODE
    node_config = SSI_NODE_CONFIGS[node]

    print(f"Calculating historic observed droughts (node={node})")

    # Load FULL historic record for transform (to find all droughts)
    Q = load_baseline_historical_flow(
        gage_flow=node_config['historical_gage_flow'], period='full', flowtype=BASELINE_DATASET
    )
    Q.replace(0, np.nan, inplace=True)
    for col in node_config['drop_columns']:
        if col in Q.columns:
            Q.drop(columns=[col], inplace=True)

    if BASELINE_DATASET == 'wrfaorc_withObsScaled':
        Q_1960s = load_wrf1960s_historical_flow(
            gage_flow=node_config['historical_gage_flow']
        )
        Q_1960s.replace(0, np.nan, inplace=True)
        for col in node_config['drop_columns']:
            if col in Q_1960s.columns:
                Q_1960s.drop(columns=[col], inplace=True)
        Q_full = pd.concat([Q_1960s, Q], axis=0).sort_index()
        Q_full.replace(0, np.nan, inplace=True)
        Q_full.dropna(axis=0, how='any', inplace=True)
    else:
        Q_full = Q.copy()

    if node_config['derived']:
        Q_full[node] = Q_full[node_config['derive_from']].sum(axis=1)

    # Extract only the target node series before resampling/cleanup
    node_daily = Q_full[node].copy()
    node_monthly = node_daily.resample('MS').sum()
    node_monthly.replace(0, np.nan, inplace=True)
    node_monthly.dropna(inplace=True)

    print(f"Loaded historic data with {Q_full.shape[0] // 365} years of daily data for {Q_full.shape[1]} sites.")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process each SSI window
    for ssi_window in ssi_windows:
        print(f"\nProcessing SSI window: {ssi_window} months")

        # Fit SSI on baseline period (1980-2019)
        ssi_calculator = fit_ssi_calculator(ssi_window, node=node)
        # Transform the FULL historic record
        ssi_obs = ssi_calculator.transform(node_monthly)
        obs_droughts = get_drought_metrics(ssi_obs)

        # Save observed drought metrics
        obs_droughts.reset_index(inplace=True, drop=True)

        # For the WRF-AORC baseline, drop droughts spanning the 1960s-1970s data gap
        if BASELINE_DATASET == 'wrfaorc_withObsScaled':
            obs_droughts = obs_droughts[~((obs_droughts['start'] < pd.to_datetime('1970-01-01')) &
                                           (obs_droughts['end'] > pd.to_datetime('1979-01-01')))]

        obs_fname = f"{output_dir}/observed_ssi{ssi_window}_drought_events.csv"
        obs_droughts.to_csv(obs_fname, index=False)
        print(f"  Saved observed drought metrics: {obs_fname}")

    print("Historic observed droughts complete.")
    return True


