"""
Verification utilities for checking data existence and validity.

This module provides functions to verify that required files and datasets exist
before running analysis scripts. These verification functions are used throughout
the workflow to ensure proper sequencing of operations.

All verify_*() functions should be imported from this module.
"""

import os
import sys


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
    from methods.config import ROOT_DIR

    verify_dataset_id(dataset_id)

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

    print(f"[OK] Preprocessing outputs verified for {dataset_id}")


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
    from methods.config import ROOT_DIR

    verify_dataset_id(dataset_id)

    output_file = f"{ROOT_DIR}/pywrdrb/outputs/{dataset_id}.hdf5"

    if not os.path.exists(output_file):
        raise FileNotFoundError(
            f"Simulation output not found: {output_file}\n"
            f"Run 03_run_pywrdrb_simulations.py for {dataset_id} first!"
        )

    print(f"[OK] Simulation outputs verified for {dataset_id}")


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
    from methods.config import ROOT_DIR

    verify_dataset_id(dataset_id)

    output_file = f"{ROOT_DIR}/pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5"

    if not os.path.exists(output_file):
        raise FileNotFoundError(
            f"Postprocessed data not found: {output_file}\n"
            f"Run 04_postprocess_data.py for {dataset_id} first!"
        )

    print(f"[OK] Postprocessing outputs verified for {dataset_id}")


def verify_realization_id_consistency(dataset_id):
    """
    Verify that realization IDs are consistent across generation and simulation for a dataset.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier

    Returns
    -------
    bool
        True if all sets are consistent or files don't exist yet
    """
    from methods.config import (
        N_ENSEMBLE_SETS,
        get_ensemble_set_spec,
        get_hdf5_realization_numbers
    )

    verify_dataset_id(dataset_id)
    print(f"Verifying {dataset_id} realization ID consistency...")

    all_ok = True
    for set_id in range(N_ENSEMBLE_SETS):
        set_spec = get_ensemble_set_spec(set_id, dataset_id)

        # Check expected vs actual realization IDs
        expected_ids = set_spec.realizations

        if os.path.exists(set_spec.files['gage_flow']):
            actual_ids = get_hdf5_realization_numbers(set_spec.files['gage_flow'])
            actual_ids = [int(x) for x in actual_ids]

            if set(expected_ids) != set(actual_ids):
                print(f"  MISMATCH in Set {set_id + 1}:")
                print(f"    Expected: {expected_ids}")
                print(f"    Actual:   {actual_ids}")
                all_ok = False
            else:
                print(f"  Set {set_id + 1}: OK")
        else:
            print(f"  Set {set_id + 1}: File not found (skipped)")

    return all_ok


def verify_ensemble_outputs(dataset_id):
    """
    Verify that ensemble generation outputs exist for a dataset.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier

    Raises
    ------
    FileNotFoundError
        If ensemble outputs are not found
    """
    from methods.config import ENSEMBLE_SETS

    verify_dataset_id(dataset_id)

    missing_sets = []
    for spec in ENSEMBLE_SETS[dataset_id]:
        if not os.path.exists(spec.files['gage_flow']):
            missing_sets.append(spec.set_id + 1)

    if missing_sets:
        raise FileNotFoundError(
            f"Missing ensemble sets for {dataset_id}: {missing_sets}\n"
            f"Run 01_generate_ensemble_sets.py for {dataset_id} first!"
        )

    print(f"[OK] Ensemble outputs verified for {dataset_id}")
