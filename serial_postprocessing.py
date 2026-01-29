#!/usr/bin/env python3
"""
Serial postprocessing workflow for the stochastic exploratory experiment.

This script runs the complete postprocessing workflow in serial mode, which is useful for:
- Processing on local machines
- Running smaller experiments
- Testing changes before deploying to HPC

The workflow consists of four main steps:
4. Postprocess data (combine ensemble sets, calculate metrics)
5. Calculate SSI drought metrics
6. Calculate satisficing by drought conditions
7. Calculate storage zone probabilities

You can optionally run a subset of steps for faster testing.

Usage:
    python serial_postprocessing.py stationary_ensemble
    python serial_postprocessing.py stationary_ensemble --start-step 5 --end-step 6
    python serial_postprocessing.py stationary_ensemble --ssi-windows 6 12
"""

import sys
import os
import argparse
from datetime import datetime

# Import the core functions from methods modules
from methods.postprocess import postprocess_dataset
from methods.drought_analysis import (
    calculate_historic_observed_droughts,
    calculate_ensemble_droughts,
    calculate_satisficing_by_drought
)
from methods.storage_zones import calculate_zone_probabilities, calculate_storage_percentiles

# Import config
from methods.config import (
    DATASET_CONFIGS,
    SSI_WINDOWS,
    verify_dataset_id
)


def run_serial_postprocessing(dataset_id, start_step=4, end_step=7,
                               ssi_windows=None, recombine=True,
                               skip_existing=True):
    """
    Run the complete postprocessing workflow in serial mode.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier (e.g., 'stationary_ensemble', 'climate_adjusted_low')
    start_step : int
        Step to start from (4=postprocess, 5=droughts, 6=satisficing, 7=zones)
    end_step : int
        Step to end at (4=postprocess, 5=droughts, 6=satisficing, 7=zones)
    ssi_windows : list of int, optional
        SSI window sizes to process (e.g., [3, 6, 12]).
        If None, uses all windows from config.
    recombine : bool
        If True, recombine ensemble sets from scratch in step 4
    skip_existing : bool
        If True, skip steps that already have outputs
    """

    # Verify dataset
    verify_dataset_id(dataset_id)
    dataset_config = DATASET_CONFIGS[dataset_id]

    # Set SSI windows
    if ssi_windows is None:
        ssi_windows = SSI_WINDOWS
    else:
        # Validate SSI windows
        for window in ssi_windows:
            if window not in SSI_WINDOWS:
                raise ValueError(f"Invalid SSI window: {window}. Must be one of {SSI_WINDOWS}")

    print("=" * 80)
    print("SERIAL POSTPROCESSING WORKFLOW")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset: {dataset_id}")
    print(f"Dataset type: {dataset_config['type']}")
    print(f"Description: {dataset_config['description']}")
    print(f"Steps to run: {start_step} through {end_step}")
    print(f"SSI windows: {ssi_windows}")
    print(f"Recombine ensemble sets: {recombine}")
    print(f"Skip existing outputs: {skip_existing}")
    print("=" * 80)
    print()

    # =========================================================================
    # STEP 4: Postprocess Data
    # NOTE: The parallel version (04_postprocess_data_mpi.py) reimplements
    # postprocessing logic inline for MPI distribution. Any changes to
    # methods/postprocess.py must also be applied to 04_postprocess_data_mpi.py.
    # =========================================================================
    if start_step <= 4 <= end_step:
        print("\n" + "=" * 80)
        print("STEP 4: POSTPROCESSING DATA")
        print("=" * 80)

        # Check if already complete
        output_file = f'./pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5'
        metrics_file = f'./pywrdrb/performance_metrics/{dataset_id}_performance_metrics.csv'

        if skip_existing and not recombine and os.path.exists(output_file) and os.path.exists(metrics_file):
            print(f"\nSkipping (already complete)")
            print(f"  Found: {output_file}")
            print(f"  Found: {metrics_file}")
        else:
            start_time = datetime.now()

            try:
                data = postprocess_dataset(dataset_id, recombine=recombine)
                elapsed = (datetime.now() - start_time).total_seconds()
                print(f"\nStep 4 completed successfully (elapsed: {elapsed:.1f}s)")

            except Exception as e:
                print(f"\nStep 4 FAILED: {str(e)}")
                raise

    # =========================================================================
    # STEP 5: Calculate SSI Drought Metrics
    # =========================================================================
    if start_step <= 5 <= end_step:
        print("\n" + "=" * 80)
        print("STEP 5: CALCULATING SSI DROUGHT METRICS")
        print("=" * 80)

        drought_metrics_dir = "./pywrdrb/drought_metrics"
        os.makedirs(drought_metrics_dir, exist_ok=True)

        # Step 5a: Calculate historic observed droughts (once per baseline)
        print("\n" + "-" * 80)
        print("STEP 5a: Historic Observed Droughts")
        print("-" * 80)

        obs_files = [f"{drought_metrics_dir}/observed_ssi{w}_drought_events.csv" for w in ssi_windows]
        if skip_existing and all(os.path.exists(f) for f in obs_files):
            print("Skipping historic observed droughts (already complete)")
        else:
            start_time = datetime.now()

            try:
                calculate_historic_observed_droughts(ssi_windows, drought_metrics_dir)
                elapsed = (datetime.now() - start_time).total_seconds()
                print(f"\nStep 5a completed successfully (elapsed: {elapsed:.1f}s)")

            except Exception as e:
                print(f"\nStep 5a FAILED: {str(e)}")
                raise

        # Step 5b: Calculate ensemble droughts
        print("\n" + "-" * 80)
        print("STEP 5b: Ensemble Droughts")
        print("-" * 80)

        ens_files = [f"{drought_metrics_dir}/{dataset_id}_ssi{w}_drought_events.csv" for w in ssi_windows]
        if skip_existing and all(os.path.exists(f) for f in ens_files):
            print(f"Skipping ensemble droughts for {dataset_id} (already complete)")
        else:
            start_time = datetime.now()

            try:
                calculate_ensemble_droughts(dataset_id, ssi_windows, drought_metrics_dir)
                elapsed = (datetime.now() - start_time).total_seconds()
                print(f"\nStep 5b completed successfully (elapsed: {elapsed:.1f}s)")

            except Exception as e:
                print(f"\nStep 5b FAILED: {str(e)}")
                raise

        print("\nStep 5 completed successfully!")

    # =========================================================================
    # STEP 6: Calculate Satisficing by Drought
    # =========================================================================
    if start_step <= 6 <= end_step:
        print("\n" + "=" * 80)
        print("STEP 6: CALCULATING SATISFICING BY DROUGHT")
        print("=" * 80)

        satisficing_dir = "./pywrdrb/satisficing_analysis"
        os.makedirs(satisficing_dir, exist_ok=True)

        for ssi_window in ssi_windows:
            print(f"\n" + "-" * 80)
            print(f"SSI Window: {ssi_window} months")
            print("-" * 80)

            # Check if already complete
            req_files = [
                f"{satisficing_dir}/{dataset_id}_ssi{ssi_window}_all_years.csv",
                f"{satisficing_dir}/{dataset_id}_ssi{ssi_window}_years_with_droughts.csv",
                f"{satisficing_dir}/{dataset_id}_ssi{ssi_window}_years_without_droughts.csv"
            ]

            if skip_existing and all(os.path.exists(f) for f in req_files):
                print(f"Skipping SSI-{ssi_window} (already complete)")
                continue

            start_time = datetime.now()

            try:
                calculate_satisficing_by_drought(dataset_id, ssi_window, satisficing_dir)
                elapsed = (datetime.now() - start_time).total_seconds()
                print(f"\nSSI-{ssi_window} completed (elapsed: {elapsed:.1f}s)")

            except Exception as e:
                print(f"\nSSI-{ssi_window} FAILED: {str(e)}")
                raise

        print("\nStep 6 completed successfully!")

    # =========================================================================
    # STEP 7: Calculate Storage Zone Probabilities
    # =========================================================================
    if start_step <= 7 <= end_step:
        print("\n" + "=" * 80)
        print("STEP 7: CALCULATING STORAGE ZONE PROBABILITIES")
        print("=" * 80)

        zone_prob_dir = "./pywrdrb/zone_probabilities"
        os.makedirs(zone_prob_dir, exist_ok=True)

        period = 'weekly'  # Can be 'daily', 'weekly', or 'monthly'

        # Check if already complete
        output_file = f"{zone_prob_dir}/{dataset_id}_zone_probs_{period}.csv"

        if skip_existing and os.path.exists(output_file):
            print(f"Skipping (already complete)")
            print(f"  Found: {output_file}")
        else:
            start_time = datetime.now()

            try:
                calculate_zone_probabilities(
                    dataset_id,
                    period=period,
                    output_dir=zone_prob_dir
                )

                # Calculate storage percentiles (matches 07_calculate_storage_zone_probabilities.py)
                calculate_storage_percentiles(dataset_id, period)

                elapsed = (datetime.now() - start_time).total_seconds()
                print(f"\nStep 7 completed successfully (elapsed: {elapsed:.1f}s)")

            except Exception as e:
                print(f"\nStep 7 FAILED: {str(e)}")
                raise

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("POSTPROCESSING WORKFLOW COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset: {dataset_id}")
    print(f"Steps completed: {start_step} through {end_step}")
    print("=" * 80)


def main():
    """Main function with argument parsing."""

    parser = argparse.ArgumentParser(
        description='Run the postprocessing workflow in serial mode',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full postprocessing workflow
  python serial_postprocessing.py stationary_ensemble

  # Run only SSI drought calculations
  python serial_postprocessing.py stationary_ensemble --start-step 5 --end-step 5

  # Run with specific SSI windows
  python serial_postprocessing.py climate_adjusted_low --ssi-windows 6 12

  # Run without skipping existing outputs
  python serial_postprocessing.py stationary_ensemble --no-skip

  # Force recombination of ensemble sets in step 4
  python serial_postprocessing.py stationary_ensemble --recombine
        """
    )

    parser.add_argument(
        'dataset_id',
        type=str,
        choices=list(DATASET_CONFIGS.keys()),
        help='Dataset identifier to process'
    )

    parser.add_argument(
        '--start-step',
        type=int,
        choices=[4, 5, 6, 7],
        default=4,
        help='Step to start from (4=postprocess, 5=droughts, 6=satisficing, 7=zones). Default: 4'
    )

    parser.add_argument(
        '--end-step',
        type=int,
        choices=[4, 5, 6, 7],
        default=7,
        help='Step to end at (4=postprocess, 5=droughts, 6=satisficing, 7=zones). Default: 7'
    )

    parser.add_argument(
        '--ssi-windows',
        type=int,
        nargs='+',
        default=None,
        metavar='WINDOW',
        help=f'SSI window sizes to process (months). Default: {SSI_WINDOWS}'
    )

    parser.add_argument(
        '--recombine',
        action='store_true',
        help='Force recombination of ensemble sets in step 4 (default: False)'
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
        run_serial_postprocessing(
            dataset_id=args.dataset_id,
            start_step=args.start_step,
            end_step=args.end_step,
            ssi_windows=args.ssi_windows,
            recombine=args.recombine,
            skip_existing=not args.no_skip
        )

        sys.exit(0)

    except Exception as e:
        print(f"\n{'='*80}")
        print("POSTPROCESSING WORKFLOW FAILED")
        print("="*80)
        print(f"Error: {str(e)}")
        print("="*80)

        import traceback
        traceback.print_exc()

        sys.exit(1)


if __name__ == "__main__":
    main()
