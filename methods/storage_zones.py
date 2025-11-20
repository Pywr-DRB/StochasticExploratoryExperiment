"""
Core functions for calculating reservoir storage zone probabilities.

This module contains functions for:
- Calculating storage zone probabilities based on FFMP boundaries
- Processing ensemble realizations efficiently
- Saving results to CSV for analysis
"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import pywrdrb
from .config import NYC_TOTAL_CAPACITY, PERIOD_ORIGIN
from .load import load_ffmp_boundaries
from .utils import calculate_water_year_period_index
from .verification import verify_postprocessing_output


def get_ordered_threshold_columns(ffmp_boundaries):
    """Get FFMP threshold columns ordered by median value."""
    num_cols = ffmp_boundaries.select_dtypes(include=[np.number]).columns.tolist()
    med = ffmp_boundaries[num_cols].median(axis=0).sort_values()
    return list(med.index)


def calculate_zone_probabilities(dataset_id, period='weekly', pct_extents=(0.0, 100.0), output_dir=None):
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
    output_dir : str, optional
        Directory to save results. If None, uses default location.

    Returns
    -------
    pd.DataFrame
        Columns: period, zone_0, zone_1, ..., zone_N (probabilities as %)
        Index: unique period values (1..P)
    """
    from .config import verify_dataset_id, DATASET_CONFIGS

    verify_dataset_id(dataset_id)
    dataset_config = DATASET_CONFIGS[dataset_id]

    print(f"\n{'='*80}")
    print(f"CALCULATING ZONE PROBABILITIES: {dataset_id}")
    print(f"{'='*80}")
    print(f"  Dataset type: {dataset_config['type']}")
    print(f"  Period: {period}")

    # Load FFMP boundaries
    ffmp_boundaries = load_ffmp_boundaries()
    thr_cols = get_ordered_threshold_columns(ffmp_boundaries)
    Z = len(thr_cols) + 1  # Number of zones

    # Verify postprocessed data exists
    verify_postprocessing_output(dataset_id)

    # Load ensemble storage data
    fname = f'./pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5'
    print(f"  Loading res_storage for {dataset_id}...")
    data = pywrdrb.Data()
    data.load_from_export(fname, results_sets=['res_storage'])

    realizations = sorted(data.res_storage[dataset_id].keys())
    print(f"  Found {len(realizations)} realizations")

    # Get period indices (using configured origin)
    sample_storage = data.res_storage[dataset_id][realizations[0]]
    p_idx = calculate_water_year_period_index(sample_storage.index, period=period, origin=PERIOD_ORIGIN)
    periods_sorted = np.sort(np.unique(p_idx))
    P = len(periods_sorted)

    print(f"  Processing {len(realizations)} realizations...")
    print(f"  Found {P} unique {period} periods")

    # Initialize counter array: shape (Z, P)
    # Each cell counts how many (realization, day) pairs fall in that (zone, period)
    zone_counts = np.zeros((Z, P), dtype=np.int64)
    total_counts = np.zeros(P, dtype=np.int64)

    # NYC reservoir names
    nyc_reservoirs = ['cannonsville', 'pepacton', 'neversink']

    # Prepare FFMP boundaries for day-of-year matching
    # Extract day-of-year from FFMP boundaries (seasonal pattern)
    ffmp_doy = ffmp_boundaries.copy()
    ffmp_doy['dayofyear'] = ffmp_boundaries.index.dayofyear

    # Get available days in FFMP (could be 365 or 366 depending on if reconstruction includes leap year)
    available_doys = set(ffmp_doy['dayofyear'].values)
    has_feb29 = 60 in available_doys  # Day 60 is Feb 29 in leap years

    # Create lookup table - group by dayofyear and take mean if there are duplicates
    # (there could be duplicates if FFMP spans multiple years)
    ffmp_doy_lookup = ffmp_doy.groupby('dayofyear')[thr_cols].mean()

    # Process each realization
    for r_idx, r in enumerate(realizations):
        if (r_idx + 1) % 10 == 0:
            print(f"    Processed {r_idx + 1}/{len(realizations)} realizations...")

        # Get NYC storage
        nyc_storage = data.res_storage[dataset_id][r][nyc_reservoirs].sum(axis=1)
        nyc_storage_pct = 100.0 * nyc_storage / NYC_TOTAL_CAPACITY

        # Get period index for this realization (using configured origin)
        p_idx_r = calculate_water_year_period_index(nyc_storage.index, period=period, origin=PERIOD_ORIGIN)

        # Get FFMP boundaries aligned with storage dates using day-of-year matching
        # This works because FFMP boundaries follow a seasonal pattern that repeats annually
        storage_doy = nyc_storage.index.dayofyear.values

        # Handle leap year mismatches using vectorized operations
        # - If FFMP has Feb 29 but storage date doesn't: shift days after Feb 29 up by 1
        # - If storage has Feb 29 but FFMP doesn't: map Feb 29 to Feb 28, shift others down by 1
        storage_doy_adjusted = storage_doy.copy()

        # Vectorized leap year check
        years = nyc_storage.index.year.values
        is_leap = ((years % 4 == 0) & ((years % 100 != 0) | (years % 400 == 0)))

        if has_feb29:
            # FFMP includes Feb 29 (day 60)
            # For non-leap years, days >= 60 need to shift up by 1
            mask = (~is_leap) & (storage_doy >= 60)
            storage_doy_adjusted[mask] = storage_doy[mask] + 1
        else:
            # FFMP does not include Feb 29
            # For leap years, map Feb 29 to Feb 28 and shift days > 60 down by 1
            mask_feb29 = is_leap & (storage_doy == 60)
            mask_after = is_leap & (storage_doy > 60)
            storage_doy_adjusted[mask_feb29] = 59
            storage_doy_adjusted[mask_after] = storage_doy[mask_after] - 1

        # Create aligned FFMP boundaries using vectorized lookup
        # This is much faster than list comprehension
        ffmp_aligned = ffmp_doy_lookup.loc[storage_doy_adjusted].values
        ffmp_aligned = pd.DataFrame(
            ffmp_aligned,
            index=nyc_storage.index,
            columns=thr_cols
        )

        # For each day, determine which zone the storage falls into
        for period_val in periods_sorted:
            # Mask for this period
            period_mask = (p_idx_r == period_val)

            # Get storage values for this period
            storage_pct_period = nyc_storage_pct[period_mask]
            ffmp_period = ffmp_aligned[period_mask]

            # Count total days in this period
            n_days = len(storage_pct_period)
            total_counts[period_val - 1] += n_days

            # For each threshold column, count days below threshold
            for j, thr_col in enumerate(thr_cols):
                threshold_values = ffmp_period[thr_col]
                below_threshold = (storage_pct_period < threshold_values).sum()

                if j == 0:
                    # Zone 0: below first threshold
                    zone_counts[0, period_val - 1] += below_threshold
                else:
                    # Zone j: between threshold j-1 and j
                    prev_threshold_values = ffmp_period[thr_cols[j-1]]
                    between = ((storage_pct_period >= prev_threshold_values) &
                              (storage_pct_period < threshold_values)).sum()
                    zone_counts[j, period_val - 1] += between

            # Last zone: above highest threshold
            last_threshold_values = ffmp_period[thr_cols[-1]]
            above_last = (storage_pct_period >= last_threshold_values).sum()
            zone_counts[-1, period_val - 1] += above_last

    # Convert counts to probabilities (percent)
    zone_probs = np.zeros((Z, P))
    for i in range(P):
        if total_counts[i] > 0:
            zone_probs[:, i] = 100.0 * zone_counts[:, i] / total_counts[i]

    # Create DataFrame
    df = pd.DataFrame(zone_probs.T, columns=[f'zone_{i}' for i in range(Z)])
    df['period'] = periods_sorted
    df = df.set_index('period')

    print(f"  Calculated probabilities for {Z} zones across {P} periods")

    # Save to CSV
    if output_dir is None:
        output_dir = "./pywrdrb/zone_probabilities"

    os.makedirs(output_dir, exist_ok=True)
    csv_file = f"{output_dir}/{dataset_id}_zone_probs_{period}.csv"
    df.to_csv(csv_file)
    print(f"  Saved: {csv_file}")

    print(f"\n{'='*80}")
    print(f"ZONE PROBABILITIES COMPLETE: {dataset_id}")
    print(f"{'='*80}")

    return df
