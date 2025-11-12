#!/usr/bin/env python3
"""
Serial workflow for running the complete stochastic exploratory experiment.

This script runs the entire workflow in serial mode, which is useful for:
- Debugging on local machines
- Running smaller experiments
- Testing changes before deploying to HPC

The workflow consists of three main steps:
1. Generate ensemble sets (synthetic flows)
2. Prepare Pywr-DRB inputs (inflows and diversions)
3. Run Pywr-DRB simulations

You can optionally run a subset of ensemble sets for faster testing.
"""

import sys
import os
import argparse
from datetime import datetime

# Import the core functions from methods modules
from methods.generate import generate_ensemble_set
from methods.prepare import prep_ensemble_set
from methods.simulate import run_ensemble_set_simulations

# Import config
from methods.config import (
    DATASET_CONFIGS,
    N_ENSEMBLE_SETS,
    N_REALIZATIONS_PER_ENSEMBLE_SET,
    verify_dataset_id,
    get_ensemble_set_spec,
    ensure_ensemble_set_dirs
)


def run_serial_workflow(dataset_id, start_step=1, end_step=3,
                        ensemble_sets=None, skip_existing=True):
    """
    Run the complete workflow in serial mode

    Parameters:
    -----------
    dataset_id : str
        Dataset identifier (e.g., 'stationary_ensemble', 'climate_adjusted_low')
    start_step : int
        Step to start from (1=generate, 2=prep, 3=simulate)
    end_step : int
        Step to end at (1=generate, 2=prep, 3=simulate)
    ensemble_sets : list of int, optional
        List of ensemble set IDs to process (0-indexed).
        If None, processes all sets.
    skip_existing : bool
        If True, skip ensemble sets that already have outputs
    """

    # Verify dataset
    verify_dataset_id(dataset_id)
    dataset_config = DATASET_CONFIGS[dataset_id]

    # Determine which ensemble sets to process
    if ensemble_sets is None:
        ensemble_sets = list(range(N_ENSEMBLE_SETS))
    else:
        # Validate ensemble set IDs
        for set_id in ensemble_sets:
            if set_id < 0 or set_id >= N_ENSEMBLE_SETS:
                raise ValueError(f"Invalid ensemble set ID: {set_id}. Must be 0-{N_ENSEMBLE_SETS-1}")

    n_sets = len(ensemble_sets)

    print("=" * 80)
    print("SERIAL WORKFLOW FOR STOCHASTIC EXPLORATORY EXPERIMENT")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset: {dataset_id}")
    print(f"Dataset type: {dataset_config['type']}")
    print(f"Description: {dataset_config['description']}")
    print(f"Processing {n_sets} ensemble set(s): {ensemble_sets}")
    print(f"Realizations per set: {N_REALIZATIONS_PER_ENSEMBLE_SET}")
    print(f"Steps to run: {start_step} through {end_step}")
    print(f"Skip existing outputs: {skip_existing}")
    print("=" * 80)
    print()

    # Ensure directories exist
    ensure_ensemble_set_dirs(dataset_id)

    # =========================================================================
    # STEP 1: Generate Ensemble Sets
    # =========================================================================
    if start_step <= 1 <= end_step:
        print("\n" + "=" * 80)
        print("STEP 1: GENERATING ENSEMBLE SETS")
        print("=" * 80)

        for i, set_id in enumerate(ensemble_sets):
            set_spec = get_ensemble_set_spec(set_id, dataset_id)

            # Check if this set already exists
            if skip_existing and os.path.exists(set_spec.files['gage_flow']) and \
               os.path.exists(set_spec.files['catchment_inflow']):
                print(f"\nSet {set_id + 1}/{N_ENSEMBLE_SETS} ({i+1}/{n_sets}): Skipping (already exists)")
                continue

            print(f"\nSet {set_id + 1}/{N_ENSEMBLE_SETS} ({i+1}/{n_sets}): Generating ensemble...")
            start_time = datetime.now()

            try:
                success = generate_ensemble_set(
                    set_id=set_id,
                    dataset_id=dataset_id,
                    use_mpi=False
                )

                elapsed = (datetime.now() - start_time).total_seconds()

                if success:
                    print(f"Set {set_id + 1}: SUCCESS (elapsed: {elapsed:.1f}s)")
                else:
                    print(f"Set {set_id + 1}: FAILED (elapsed: {elapsed:.1f}s)")
                    raise RuntimeError(f"Ensemble generation failed for set {set_id + 1}")

            except Exception as e:
                print(f"Set {set_id + 1}: ERROR - {str(e)}")
                raise

        print("\nStep 1 completed successfully!")

    # =========================================================================
    # STEP 2: Prepare Pywr-DRB Inputs
    # =========================================================================
    if start_step <= 2 <= end_step:
        print("\n" + "=" * 80)
        print("STEP 2: PREPARING PYWR-DRB INPUTS")
        print("=" * 80)

        for i, set_id in enumerate(ensemble_sets):
            set_spec = get_ensemble_set_spec(set_id, dataset_id)

            # Check if inputs are already prepared
            if skip_existing and os.path.exists(set_spec.files['predicted_inflow']) and \
               os.path.exists(set_spec.files['diversion_nj']) and \
               os.path.exists(set_spec.files['diversion_nyc']) and \
               os.path.exists(set_spec.files['predicted_diversions']):
                print(f"\nSet {set_id + 1}/{N_ENSEMBLE_SETS} ({i+1}/{n_sets}): Skipping (already prepared)")
                continue

            # Check if ensemble set exists
            if not os.path.exists(set_spec.files['catchment_inflow']):
                print(f"\nSet {set_id + 1}/{N_ENSEMBLE_SETS} ({i+1}/{n_sets}): ERROR - Ensemble not generated yet")
                raise RuntimeError(f"Ensemble set {set_id + 1} not found. Run step 1 first.")

            print(f"\nSet {set_id + 1}/{N_ENSEMBLE_SETS} ({i+1}/{n_sets}): Preparing inputs...")
            start_time = datetime.now()

            try:
                success = prep_ensemble_set(
                    set_id=set_id,
                    dataset_id=dataset_id,
                    use_mpi=False
                )

                elapsed = (datetime.now() - start_time).total_seconds()

                if success:
                    print(f"Set {set_id + 1}: SUCCESS (elapsed: {elapsed:.1f}s)")
                else:
                    print(f"Set {set_id + 1}: FAILED (elapsed: {elapsed:.1f}s)")
                    raise RuntimeError(f"Input preparation failed for set {set_id + 1}")

            except Exception as e:
                print(f"Set {set_id + 1}: ERROR - {str(e)}")
                raise

        print("\nStep 2 completed successfully!")

    # =========================================================================
    # STEP 3: Run Pywr-DRB Simulations
    # =========================================================================
    if start_step <= 3 <= end_step:
        print("\n" + "=" * 80)
        print("STEP 3: RUNNING PYWR-DRB SIMULATIONS")
        print("=" * 80)

        for i, set_id in enumerate(ensemble_sets):
            set_spec = get_ensemble_set_spec(set_id, dataset_id)

            # Check if simulations are already complete
            if skip_existing and os.path.exists(set_spec.output_file):
                file_size = os.path.getsize(set_spec.output_file)
                if file_size > 1024:  # More than 1KB
                    print(f"\nSet {set_id + 1}/{N_ENSEMBLE_SETS} ({i+1}/{n_sets}): Skipping (already simulated)")
                    continue

            # Check if inputs are prepared
            if not os.path.exists(set_spec.files['predicted_inflow']):
                print(f"\nSet {set_id + 1}/{N_ENSEMBLE_SETS} ({i+1}/{n_sets}): ERROR - Inputs not prepared yet")
                raise RuntimeError(f"Inputs for set {set_id + 1} not found. Run step 2 first.")

            print(f"\nSet {set_id + 1}/{N_ENSEMBLE_SETS} ({i+1}/{n_sets}): Running simulations...")
            start_time = datetime.now()

            try:
                success = run_ensemble_set_simulations(
                    set_id=set_id,
                    dataset_id=dataset_id,
                    use_mpi=False
                )

                elapsed = (datetime.now() - start_time).total_seconds()

                if success:
                    print(f"Set {set_id + 1}: SUCCESS (elapsed: {elapsed:.1f}s)")
                else:
                    print(f"Set {set_id + 1}: FAILED (elapsed: {elapsed:.1f}s)")
                    raise RuntimeError(f"Simulation failed for set {set_id + 1}")

            except Exception as e:
                print(f"Set {set_id + 1}: ERROR - {str(e)}")
                raise

        print("\nStep 3 completed successfully!")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("WORKFLOW COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset: {dataset_id}")
    print(f"Processed {n_sets} ensemble set(s)")
    print("=" * 80)


def main():
    """Main function with argument parsing"""

    parser = argparse.ArgumentParser(
        description='Run the stochastic exploratory experiment workflow in serial mode',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full workflow for all ensemble sets
  python serial_workflow.py stationary_ensemble

  # Run only the first ensemble set for debugging
  python serial_workflow.py stationary_ensemble --sets 0

  # Run sets 0-2 (first 3 sets)
  python serial_workflow.py stationary_ensemble --sets 0 1 2

  # Run only step 2 (input preparation) for set 0
  python serial_workflow.py stationary_ensemble --sets 0 --start-step 2 --end-step 2

  # Run steps 1-2 without skipping existing outputs
  python serial_workflow.py climate_adjusted_low --end-step 2 --no-skip
        """
    )

    parser.add_argument(
        'dataset_id',
        type=str,
        choices=list(DATASET_CONFIGS.keys()),
        help='Dataset identifier to process'
    )

    parser.add_argument(
        '--sets',
        type=int,
        nargs='+',
        default=None,
        metavar='SET_ID',
        help=f'Ensemble set IDs to process (0-{N_ENSEMBLE_SETS-1}). Default: all sets'
    )

    parser.add_argument(
        '--start-step',
        type=int,
        choices=[1, 2, 3],
        default=1,
        help='Step to start from (1=generate, 2=prep, 3=simulate). Default: 1'
    )

    parser.add_argument(
        '--end-step',
        type=int,
        choices=[1, 2, 3],
        default=3,
        help='Step to end at (1=generate, 2=prep, 3=simulate). Default: 3'
    )

    parser.add_argument(
        '--no-skip',
        action='store_true',
        help='Do not skip existing outputs (reprocess everything)'
    )

    args = parser.parse_args()

    # Validate step order
    if args.start_step > args.end_step:
        parser.error("start-step must be <= end-step")

    # Run workflow
    try:
        run_serial_workflow(
            dataset_id=args.dataset_id,
            start_step=args.start_step,
            end_step=args.end_step,
            ensemble_sets=args.sets,
            skip_existing=not args.no_skip
        )

        sys.exit(0)

    except Exception as e:
        print(f"\n{'='*80}")
        print("WORKFLOW FAILED")
        print("="*80)
        print(f"Error: {str(e)}")
        print("="*80)

        import traceback
        traceback.print_exc()

        sys.exit(1)


if __name__ == "__main__":
    main()
