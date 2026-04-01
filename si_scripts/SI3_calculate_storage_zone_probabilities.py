"""
Calculate reservoir storage zone probabilities and percentiles for all datasets.

Efficiently computes zone probabilities and storage percentiles by:
- Loading only res_storage data per dataset
- Processing realizations iteratively (memory efficient)
- Caching FFMP boundaries
- Saving results to CSV for fast reloading

Outputs:
- Zone probabilities: probability of storage being in each FFMP zone
- Storage percentiles: 1st, 5th, 10th, 25th, 50th, 75th, 90th, 95th, 99th percentile
  storage levels for each week of the year

Usage:
  python 07_calculate_storage_zone_probabilities.py [dataset_id]
  python 07_calculate_storage_zone_probabilities.py --all
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings("ignore")

import pywrdrb
from methods.config import *
from methods.save import save_zone_probabilities
from methods.storage_zones import (
    calculate_zone_probabilities,
    calculate_storage_percentiles
)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        print(f"\nAvailable datasets: {list(DATASET_CONFIGS.keys())}")
        sys.exit(1)
    
    arg = sys.argv[1]
    period = 'weekly' 
    
    if arg == '--all':
        print("=" * 60)
        print("CALCULATING ZONE PROBABILITIES & PERCENTILES FOR ALL DATASETS")
        print("=" * 60)

        for dataset_id in DATASET_CONFIGS.keys():
            # Calculate zone probabilities
            df = calculate_zone_probabilities(dataset_id, period)
            if df is not None:
                save_zone_probabilities(df, dataset_id, period)

            # Calculate storage percentiles
            calculate_storage_percentiles(dataset_id, period)
            print()

        print("=" * 60)
        print("All zone probabilities and storage percentiles calculated!")

    else:
        dataset_id = arg
        verify_dataset_id(dataset_id)

        print("=" * 60)
        print(f"CALCULATING ZONE PROBABILITIES & PERCENTILES: {dataset_id}")
        print("=" * 60)

        # Calculate zone probabilities
        df = calculate_zone_probabilities(dataset_id, period)
        if df is not None:
            save_zone_probabilities(df, dataset_id, period)

        # Calculate storage percentiles
        calculate_storage_percentiles(dataset_id, period)

        print("=" * 60)
        print("Done!")


if __name__ == "__main__":
    main()