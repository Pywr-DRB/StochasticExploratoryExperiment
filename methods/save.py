"""
Save utilities for writing analysis results to files.

This module provides functions for saving various analysis outputs
including zone probabilities and satisficing results.

All save_*() functions should be imported from this module.

Note: Performance metrics saving is in methods.postprocess since it's
tightly coupled with the calculation logic there.
"""

import os
import pandas as pd

from methods.config import ROOT_DIR


# Output directories
ZONE_PROB_DIR = f"{ROOT_DIR}/pywrdrb/zone_probabilities"
SATISFICING_ANALYSIS_DIR = f"{ROOT_DIR}/pywrdrb/satisficing_analysis"


def save_zone_probabilities(df, dataset_id, period='weekly', output_dir=None):
    """
    Save zone probabilities to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Zone probabilities DataFrame
    dataset_id : str
        Dataset identifier
    period : str
        Time period ('daily' or 'weekly')
    output_dir : str, optional
        Output directory path. Defaults to ZONE_PROB_DIR.

    Returns
    -------
    str
        Path to the saved file
    """
    if output_dir is None:
        output_dir = ZONE_PROB_DIR

    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/{dataset_id}_zone_probs_{period}.csv"
    df.to_csv(output_file)
    print(f"  Saved: {output_file}")
    return output_file


def save_satisficing_results(all_years_results, drought_results, non_drought_results,
                             dataset_id, ssi_window, output_dir=None):
    """
    Save satisficing analysis results to CSV files.

    Parameters
    ----------
    all_years_results : pd.DataFrame
        All years results
    drought_results : pd.DataFrame
        Years with drought events results
    non_drought_results : pd.DataFrame
        Years without drought events results
    dataset_id : str
        Dataset identifier
    ssi_window : int
        SSI window
    output_dir : str, optional
        Output directory path. Defaults to SATISFICING_ANALYSIS_DIR.

    Returns
    -------
    list
        List of saved file paths
    """
    if output_dir is None:
        output_dir = SATISFICING_ANALYSIS_DIR

    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"SAVING RESULTS TO CSV (SSI-{ssi_window})")
    print("=" * 80)

    fnames = []

    # Save all years
    fname = f"{output_dir}/{dataset_id}_ssi{ssi_window}_all_years.csv"
    all_years_results.to_csv(fname, index=False)
    fnames.append(fname)
    print(f"Saved: {fname}")

    # Save drought years
    fname = f"{output_dir}/{dataset_id}_ssi{ssi_window}_years_with_droughts.csv"
    drought_results.to_csv(fname, index=False)
    fnames.append(fname)
    print(f"Saved: {fname}")

    # Save non-drought years
    fname = f"{output_dir}/{dataset_id}_ssi{ssi_window}_years_without_droughts.csv"
    non_drought_results.to_csv(fname, index=False)
    fnames.append(fname)
    print(f"Saved: {fname}")

    # Create combined dataset with condition label
    all_years_labeled = all_years_results.copy()
    all_years_labeled['condition'] = 'all_years'

    drought_labeled = drought_results.copy()
    drought_labeled['condition'] = 'drought'

    non_drought_labeled = non_drought_results.copy()
    non_drought_labeled['condition'] = 'non_drought'

    combined = pd.concat([all_years_labeled, drought_labeled, non_drought_labeled],
                         ignore_index=True)

    fname = f"{output_dir}/{dataset_id}_ssi{ssi_window}_combined.csv"
    combined.to_csv(fname, index=False)
    fnames.append(fname)
    print(f"Saved combined: {fname}")

    print("=" * 80)
    return fnames
