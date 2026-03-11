"""
Verification utilities for checking data existence and validity.
"""

import os


def verify_dataset_id(dataset_id):
    """
    Verify that the dataset_id is valid.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier to verify

    Returns
    -------
    bool
        True if valid

    Raises
    ------
    ValueError
        If dataset_id is not in DATASET_CONFIGS
    """
    # Import here to avoid circular imports
    from methods.config import DATASET_CONFIGS

    if dataset_id not in DATASET_CONFIGS:
        raise ValueError(
            f"Invalid dataset_id: {dataset_id}. "
            f"Must be one of {list(DATASET_CONFIGS.keys())}"
        )
    return True


def verify_postprocessing_output(dataset_id):
    """
    Verify that postprocessing outputs exist for a dataset.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier

    Raises
    ------
    FileNotFoundError
        If postprocessing outputs are not found
    """
    from methods.config import ROOT_DIR

    verify_dataset_id(dataset_id)

    output_file = f"{ROOT_DIR}/pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5"

    if not os.path.exists(output_file):
        raise FileNotFoundError(
            f"Postprocessed data not found: {output_file}\n"
            f"Run 04_postprocess_data.py for {dataset_id} first!"
        )

    print(f"[OK] Postprocessing outputs verified for {dataset_id}")
