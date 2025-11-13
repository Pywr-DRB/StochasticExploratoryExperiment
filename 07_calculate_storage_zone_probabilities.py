"""
Calculate reservoir storage zone probabilities for all datasets.

Efficiently computes zone probabilities by:
- Loading only res_storage data per dataset
- Processing realizations iteratively (memory efficient)
- Caching FFMP boundaries
- Saving results to CSV for fast reloading

Usage:
  python 09a_calculate_storage_zone_probabilities.py [dataset_id]
  python 09a_calculate_storage_zone_probabilities.py --all
"""

import sys
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import pywrdrb
from methods.config import *
from methods.storage_zones import (
    get_ordered_threshold_columns,
    calculate_zone_probabilities
)


# Output directory for zone probability CSVs
ZONE_PROB_DIR = f"{ROOT_DIR}/pywrdrb/zone_probabilities"
os.makedirs(ZONE_PROB_DIR, exist_ok=True)

# Note: get_ordered_threshold_columns and calculate_zone_probabilities are now imported from methods.storage_zones


def save_zone_probabilities(df, dataset_id, period='weekly'):
    """Save zone probabilities to CSV."""
    output_file = f"{ZONE_PROB_DIR}/{dataset_id}_zone_probs_{period}.csv"
    df.to_csv(output_file)
    print(f"  Saved: {output_file}")
    return output_file


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
        print("CALCULATING ZONE PROBABILITIES FOR ALL DATASETS")
        print("=" * 60)
        
        for dataset_id in DATASET_CONFIGS.keys():
            df = calculate_zone_probabilities(dataset_id, period)
            if df is not None:
                save_zone_probabilities(df, dataset_id, period)
            print()
        
        print("=" * 60)
        print("All zone probabilities calculated!")
        
    else:
        dataset_id = arg
        verify_dataset_id(dataset_id)
        
        print("=" * 60)
        print(f"CALCULATING ZONE PROBABILITIES: {dataset_id}")
        print("=" * 60)
        
        df = calculate_zone_probabilities(dataset_id, period)
        if df is not None:
            save_zone_probabilities(df, dataset_id, period)
        
        print("=" * 60)
        print("Done!")


if __name__ == "__main__":
    main()