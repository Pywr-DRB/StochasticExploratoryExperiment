"""
Save utilities for writing analysis results to files.

Performance-metric CSVs are written by methods.postprocess.save_metrics_csv.
"""

import os

from methods.config import ZONE_PROB_DIR


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


