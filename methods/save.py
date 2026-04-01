"""
Save utilities for writing analysis results to files.

All save_*() functions should be imported from this module.

Note: Performance metrics saving is in methods.postprocess since it's
tightly coupled with the calculation logic there.
"""

import os
import pandas as pd

from methods.config import ZONE_PROB_DIR, SATISFICING_DIR


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


def save_annual_satisficing(df, dataset_id, ssi_window, output_dir=None):
    """
    Save annual satisficing results to a single CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Annual satisficing DataFrame with columns: year, realization,
        satisficing, nyc_min_storage_pct, montague_max_consec_shortage_days,
        nyc_inflow, montague_contrib, n_droughts_in_year.
    dataset_id : str
        Dataset identifier.
    ssi_window : int
        SSI window (3, 6, or 12).
    output_dir : str, optional
        Output directory. Defaults to SATISFICING_DIR.

    Returns
    -------
    str
        Path to the saved file.
    """
    if output_dir is None:
        output_dir = SATISFICING_DIR

    os.makedirs(output_dir, exist_ok=True)
    fname = f"{output_dir}/{dataset_id}_ssi{ssi_window}_annual_satisficing.csv"
    df.to_csv(fname, index=False)

    n_drought = (df['n_droughts_in_year'] > 0).sum()
    n_non_drought = (df['n_droughts_in_year'] == 0).sum()
    print(f"  Saved: {fname}")
    print(f"    {len(df)} total year-realization pairs "
          f"({n_drought} drought, {n_non_drought} non-drought)")

    return fname
