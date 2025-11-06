"""
Verification utilities for checking data existence and validity.

This module provides functions to verify that required files and datasets exist
before running analysis scripts. These verification functions are used throughout
the workflow to ensure proper sequencing of operations.
"""

import os
import sys
from config import ROOT_DIR


def verify_file_exists(filepath, error_message=None):
    """
    Verify that a file exists, raise FileNotFoundError if not.

    Parameters
    ----------
    filepath : str
        Path to file to verify
    error_message : str, optional
        Custom error message to display. If None, a default message is used.

    Raises
    ------
    FileNotFoundError
        If the file does not exist
    """
    if not os.path.exists(filepath):
        if error_message is None:
            error_message = f"Required file not found: {filepath}"
        raise FileNotFoundError(error_message)


def verify_prep_outputs(dataset_id):
    """
    Verify that preprocessing outputs exist for a dataset.

    This function checks that the prep_pywrdrb_inputs step has been completed
    by verifying the existence of required input files.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier

    Raises
    ------
    FileNotFoundError
        If preprocessing outputs are not found
    """
    # Check for the main prep directory
    prep_dir = f"{ROOT_DIR}/pywrdrb/prep/{dataset_id}"

    if not os.path.exists(prep_dir):
        raise FileNotFoundError(
            f"Preprocessing directory not found: {prep_dir}\n"
            f"Run 02_prep_pywrdrb_inputs.py for {dataset_id} first!"
        )

    # Check for specific required files
    required_files = [
        f"{prep_dir}/datetime.csv",
        f"{prep_dir}/inflows.hdf5"
    ]

    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        raise FileNotFoundError(
            f"Missing preprocessing files for {dataset_id}:\n" +
            "\n".join(f"  - {f}" for f in missing_files) +
            f"\n\nRun 02_prep_pywrdrb_inputs.py for {dataset_id} first!"
        )

    print(f"✓ Preprocessing outputs verified for {dataset_id}")


def verify_simulation_outputs(dataset_id):
    """
    Verify that simulation outputs exist for a dataset.

    This function checks that the run_pywrdrb_simulations step has been completed
    by verifying the existence of simulation output files.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier

    Raises
    ------
    FileNotFoundError
        If simulation outputs are not found
    """
    output_file = f"{ROOT_DIR}/pywrdrb/outputs/{dataset_id}.hdf5"

    if not os.path.exists(output_file):
        raise FileNotFoundError(
            f"Simulation output not found: {output_file}\n"
            f"Run 03_run_pywrdrb_simulations.py for {dataset_id} first!"
        )

    print(f"✓ Simulation outputs verified for {dataset_id}")


def verify_postprocessing_output(dataset_id):
    """
    Verify that postprocessing outputs exist for a dataset.

    This function checks that the postprocess_data step has been completed
    by verifying the existence of the postprocessed HDF5 file.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier

    Raises
    ------
    FileNotFoundError
        If postprocessing outputs are not found
    """
    output_file = f"{ROOT_DIR}/pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5"

    if not os.path.exists(output_file):
        raise FileNotFoundError(
            f"Postprocessed data not found: {output_file}\n"
            f"Run 04_postprocess_data.py for {dataset_id} first!"
        )

    print(f"✓ Postprocessing outputs verified for {dataset_id}")
