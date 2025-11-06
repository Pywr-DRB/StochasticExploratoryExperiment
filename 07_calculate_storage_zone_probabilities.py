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
from methods.load import load_ffmp_boundaries
from methods.utils import calculate_water_year_period_index
from methods.verification import verify_postprocessing_output


# Output directory for zone probability CSVs
ZONE_PROB_DIR = f"{ROOT_DIR}/pywrdrb/zone_probabilities"
os.makedirs(ZONE_PROB_DIR, exist_ok=True)


def get_ordered_threshold_columns(ffmp_boundaries):
    """Get threshold columns ordered by median value."""
    num_cols = ffmp_boundaries.select_dtypes(include=[np.number]).columns.tolist()
    med = ffmp_boundaries[num_cols].median(axis=0).sort_values()
    return list(med.index)


def calculate_zone_probabilities(dataset_id, period='weekly', pct_extents=(0.0, 100.0)):
    """
    Calculate storage zone probabilities for a dataset.
    
    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    period : str
        Time aggregation: 'daily', 'weekly', or 'monthly'
    pct_extents : tuple
        Min/max storage percentage bounds
        
    Returns
    -------
    pd.DataFrame
        Columns: period, zone_0, zone_1, ..., zone_N (probabilities as %)
        Index: unique period values (1..P)
    """
    verify_dataset_id(dataset_id)
    dataset_config = DATASET_CONFIGS[dataset_id]
    
    print(f"\nCalculating zone probabilities for: {dataset_id}")
    print(f"  Dataset type: {dataset_config['type']}")
    print(f"  Period: {period}")
    
    # Load FFMP boundaries
    ffmp_boundaries = load_ffmp_boundaries()
    thr_cols = get_ordered_threshold_columns(ffmp_boundaries)
    Z = len(thr_cols) + 1  # Number of zones

    # Verify postprocessed data exists
    verify_postprocessing_output(dataset_id)

    # Load ensemble storage data (only res_storage for this dataset)
    fname = f'./pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5'
    print(f"  Loading res_storage for {dataset_id}...")
    data = pywrdrb.Data()
    data.load_from_export(fname, results_sets=['res_storage'])
    
    realization_ids = list(data.res_storage[dataset_id].keys())
    n_realizations = len(realization_ids)
    print(f"  Found {n_realizations} realizations")
    
    # Get NYC reservoir names
    nyc_reservoirs = ['cannonsville', 'pepacton', 'neversink']
    
    # Get datetime index from first realization
    first_real = data.res_storage[dataset_id][realization_ids[0]]
    datetime_index = first_real.index
    
    # Resample to target period if needed
    if period == 'daily':
        agg_period = 'D'
    elif period == 'weekly':
        agg_period = 'W'
    else:  # monthly
        agg_period = 'MS'
    
    # Align dates between storage and FFMP boundaries
    common_idx = datetime_index.intersection(ffmp_boundaries.index)
    if common_idx.empty:
        print("ERROR: No overlapping dates!")
        return None
    
    print(f"  Common date range: {common_idx[0]} to {common_idx[-1]} ({len(common_idx)} days)")
    
    # Get aligned FFMP boundaries
    B = ffmp_boundaries.loc[common_idx]
    B_vals = B[thr_cols].to_numpy()  # (N_dates, K)
    
    # Calculate period indices
    p_idx = calculate_water_year_period_index(common_idx, period=period, origin='june1')
    periods_sorted = np.sort(np.unique(p_idx))
    P = periods_sorted.size
    period_to_pos = {p: i for i, p in enumerate(periods_sorted)}
    
    print(f"  Unique periods: {P}")
    print(f"  Zones: {Z}")
    
    # Initialize counts array: (P, Z)
    counts = np.zeros((P, Z), dtype=np.int64)
    
    # Zone edges
    lo, hi = pct_extents
    hi_inc = np.nextafter(hi, np.inf)
    
    # Process each realization iteratively (memory efficient)
    print(f"  Processing realizations...")
    for i, real_id in enumerate(realization_ids):
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{n_realizations}...")
        
        # Get NYC aggregate storage for this realization
        res_data = data.res_storage[dataset_id][real_id]
        nyc_total = res_data.loc[common_idx, nyc_reservoirs].sum(axis=1)
        
        # Resample to period (min within period)
        if period != 'daily':
            nyc_total = nyc_total.resample(agg_period).min()
            # Recompute period indices for resampled dates
            p_idx_real = calculate_water_year_period_index(nyc_total.index, period=period, origin='june1')
            # Also need to resample FFMP boundaries
            B_real = B.resample(agg_period).median()
            B_vals_real = B_real[thr_cols].to_numpy()
        else:
            p_idx_real = p_idx
            B_vals_real = B_vals
        
        # Convert to percentage of max capacity
        max_cap = nyc_total.max()
        if max_cap > 0:
            nyc_pct = (nyc_total / max_cap * 100).to_numpy()
        else:
            continue  # Skip if no data
        
        # Histogram each timestep into zones
        for t, p in enumerate(p_idx_real):
            # Dynamic edges for this timestep
            edges = np.concatenate(([lo], np.sort(B_vals_real[t]), [hi_inc]))
            
            # Which zone is this storage value in?
            zone_idx = np.searchsorted(edges, nyc_pct[t], side='right') - 1
            zone_idx = np.clip(zone_idx, 0, Z - 1)
            
            counts[period_to_pos[p], zone_idx] += 1
    
    # Convert counts to probabilities (%)
    totals = counts.sum(axis=1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        probs = np.where(totals > 0, 100.0 * counts / totals, 0.0)
    
    # Create DataFrame
    zone_cols = [f'zone_{i}' for i in range(Z)]
    df = pd.DataFrame(probs, index=periods_sorted, columns=zone_cols)
    df.index.name = 'period'
    
    print(f"  Zone probabilities calculated!")
    print(f"    Shape: {df.shape}")
    print(f"    Total probability per period (should be ~100): {df.sum(axis=1).mean():.2f}%")
    
    return df


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