"""
Print summary utilities for experiment configuration and results.

This module provides functions for printing formatted summaries of
experiment configurations, ensemble sets, and analysis results.
"""


def print_prep_status(dataset_id):
    """
    Print detailed status of Pywr-DRB input preparation for all ensemble sets.

    Reports per-set status after the prep step.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier to report on

    Returns
    -------
    bool
        True if all sets are properly prepared
    """
    import os
    from methods.config import N_ENSEMBLE_SETS
    from methods.ensemble_utils import get_ensemble_set_spec
    from methods.verification import verify_dataset_id

    verify_dataset_id(dataset_id)
    print(f"\nVerifying Pywr-DRB input preparation for {dataset_id}...")

    all_prepared = True
    successful_sets = []
    failed_sets = []
    missing_files = {}

    # Required files to check
    required_files = [
        'predicted_inflow',
        'diversion_nyc',
        'diversion_nj',
        'predicted_diversions'
    ]

    for set_id in range(N_ENSEMBLE_SETS):
        set_spec = get_ensemble_set_spec(set_id, dataset_id)
        set_complete = True
        missing_in_set = []

        for file_key in required_files:
            fname = set_spec.files[file_key]
            if not os.path.exists(fname):
                set_complete = False
                missing_in_set.append(file_key)
                all_prepared = False

        if set_complete:
            successful_sets.append(set_id + 1)
        else:
            failed_sets.append(set_id + 1)
            missing_files[set_id + 1] = missing_in_set

    if all_prepared:
        print(f"All {N_ENSEMBLE_SETS} ensemble sets prepared.")
        print(f"  All required files present:")
        for file_key in required_files:
            print(f"    - {file_key}")
    else:
        print(f"WARNING: {len(failed_sets)} sets not properly prepared: {failed_sets}")
        print(f"Successfully prepared: {len(successful_sets)} sets")
        print(f"\nMissing files by set:")
        for set_id, missing in missing_files.items():
            print(f"  Set {set_id}: {missing}")

    return all_prepared


def print_simulation_status(dataset_id):
    """
    Print detailed status of Pywr-DRB simulations for all ensemble sets.

    Reports per-set status after the simulation step, including a file-size
    check and a load check on the first and last sets.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier to report on

    Returns
    -------
    bool
        True if all sets have valid simulation outputs
    """
    import os
    import pywrdrb
    from methods.config import N_ENSEMBLE_SETS, N_REALIZATIONS_PER_ENSEMBLE_SET
    from methods.ensemble_utils import get_ensemble_set_spec
    from methods.verification import verify_dataset_id

    verify_dataset_id(dataset_id)
    print(f"\nVerifying Pywr-DRB simulation outputs for {dataset_id}...")

    all_completed = True
    successful_sets = []
    failed_sets = []

    for set_id in range(N_ENSEMBLE_SETS):
        set_spec = get_ensemble_set_spec(set_id, dataset_id)

        if not os.path.exists(set_spec.output_file):
            print(f"FAIL:  Set {set_id + 1}: Output file not found")
            all_completed = False
            failed_sets.append(set_id + 1)
            continue

        # Check file size (basic validation)
        file_size = os.path.getsize(set_spec.output_file)
        if file_size < 1024:  # Less than 1KB is suspicious
            print(f"FAIL:  Set {set_id + 1}: Output file too small ({file_size} bytes)")
            all_completed = False
            failed_sets.append(set_id + 1)
            continue

        # Try to load with Pywr-DRB to verify format (optimization: only check first and last)
        if set_id == 0 or set_id == N_ENSEMBLE_SETS - 1:
            try:
                test_data = pywrdrb.Data(results_sets=["major_flow"])
                test_data.load_output(output_filenames=[set_spec.output_file])
                n_realizations = len(list(test_data.major_flow.values())[0])

                if n_realizations != N_REALIZATIONS_PER_ENSEMBLE_SET:
                    print(f"WARNING: Set {set_id + 1}: Expected {N_REALIZATIONS_PER_ENSEMBLE_SET} realizations, found {n_realizations}")
                    failed_sets.append(set_id + 1)
                else:
                    print(f"Set {set_id + 1}: valid output ({n_realizations} realizations, {file_size//1024//1024} MB)")
                    successful_sets.append(set_id + 1)

            except Exception as e:
                print(f"FAIL:  Set {set_id + 1}: Error loading output file - {str(e)}")
                all_completed = False
                failed_sets.append(set_id + 1)
        else:
            # For middle sets, just check file size
            print(f"Set {set_id + 1}: output exists ({file_size//1024//1024} MB)")
            successful_sets.append(set_id + 1)

    if all_completed:
        print(f"All {N_ENSEMBLE_SETS} ensemble sets have valid simulation outputs.")
    else:
        print(f"WARNING: {len(failed_sets)} sets have invalid outputs: {failed_sets}")

    return all_completed
