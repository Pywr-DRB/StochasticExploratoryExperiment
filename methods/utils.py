import pywrdrb
import pandas as pd
import numpy as np
import sys


def distribute_realizations_across_ranks(realization_ids, rank, num_ranks):
    """
    Distribute realizations across MPI ranks using deterministic slicing.

    This function partitions a list of realization IDs across MPI ranks,
    ensuring that each rank gets approximately equal work with the
    remainder distributed among the first ranks.

    Parameters
    ----------
    realization_ids : list
        List of realization IDs to distribute
    rank : int
        Current MPI rank (0-indexed)
    num_ranks : int
        Total number of MPI ranks

    Returns
    -------
    list
        Subset of realization IDs assigned to this rank

    Examples
    --------
    >>> ids = list(range(100))
    >>> distribute_realizations_across_ranks(ids, 0, 4)  # Rank 0 gets IDs 0-24
    >>> distribute_realizations_across_ranks(ids, 1, 4)  # Rank 1 gets IDs 25-49
    """
    n = len(realization_ids)
    base = n // num_ranks
    extra = n % num_ranks

    if rank < extra:
        start = rank * (base + 1)
        end = start + base + 1
    else:
        start = rank * base + extra
        end = start + base

    return realization_ids[start:end]


def calculate_water_year_period_index(dates, period='daily', origin='june1'):
    """
    Map dates to water year or calendar year period index (1-based).

    This function converts dates to period indices based on either water year
    starting from June 1 or calendar year starting from January 1.
    Supports daily (1-366), weekly (1-53), or monthly (1-12) periods.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Dates to convert to period indices
    period : str, default='daily'
        Period type: 'daily', 'weekly', or 'monthly'
    origin : str, default='june1'
        Year origin: 'june1' for water year or 'jan1' for calendar year

    Returns
    -------
    np.ndarray
        Period indices (1-based) for each date

    Examples
    --------
    >>> dates = pd.date_range('2020-06-01', '2020-06-07')
    >>> calculate_water_year_period_index(dates, 'daily', 'june1')
    array([1, 2, 3, 4, 5, 6, 7])
    >>> calculate_water_year_period_index(dates, 'weekly', 'june1')
    array([1, 1, 1, 1, 1, 1, 1])
    >>> dates2 = pd.date_range('2020-01-01', '2020-01-07')
    >>> calculate_water_year_period_index(dates2, 'daily', 'jan1')
    array([1, 2, 3, 4, 5, 6, 7])
    """
    dates = pd.DatetimeIndex(dates)

    if period not in ('daily', 'weekly', 'monthly'):
        raise ValueError("period must be one of {'daily','weekly','monthly'}")

    if origin not in ('june1', 'jan1'):
        raise ValueError("origin must be one of {'june1', 'jan1'}")

    if origin == 'june1':
        # Water year starting June 1
        june1_this_year = pd.to_datetime(dates.year.astype(str) + '-06-01')
        is_after_june1 = dates >= june1_this_year
        june1_prev_year = pd.to_datetime((dates.year - 1).astype(str) + '-06-01')

        # Day of water year (1-based)
        day_of_year = np.where(
            is_after_june1,
            (dates - june1_this_year).days + 1,
            (dates - june1_prev_year).days + 1
        )

        if period == 'monthly':
            year_month = ((dates.month - 6) % 12) + 1
            return year_month

    else:  # origin == 'jan1'
        # Calendar year starting January 1
        jan1_this_year = pd.to_datetime(dates.year.astype(str) + '-01-01')
        day_of_year = (dates - jan1_this_year).days + 1

        if period == 'monthly':
            return dates.month

    # Common processing for daily and weekly periods
    if period == 'daily':
        return day_of_year
    elif period == 'weekly':
        return ((day_of_year - 1) // 7) + 1


def get_parameter_subset_to_export(all_parameter_names, results_set_subset):
    output_loader = pywrdrb.load.Output(output_filenames=[]) # empty dataloader to use methods
    keep_keys = []
    for results_set in results_set_subset:
        if results_set == "all":
            continue
        
        keys_subset, _ = output_loader.get_keys_and_column_names_for_results_set(all_parameter_names, 
                                                                                 results_set)
        
        keep_keys.extend(keys_subset)
    return keep_keys

