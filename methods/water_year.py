"""
Water-year calendar utilities.

Water year N runs from June 1 of year N to May 31 of year N+1.
Day 1 = June 1, Day 366 = May 31 (leap year).
"""

import numpy as np
import pandas as pd

# Water-year month boundaries (day-of-water-year for 1st of each month)
MONTH_STARTS_WY = [1, 31, 62, 93, 123, 154, 184, 215, 246, 274, 305, 335]
MONTH_LABELS_WY = ['Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov',
                   'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May']

MIN_DAYS_FOR_COMPLETE_WATER_YEAR = 360


def get_water_year(date):
    """Return water year for a date.  WY N: June 1 N → May 31 N+1."""
    if date.month >= 6:
        return date.year
    return date.year - 1


def get_water_year_doy(date):
    """Return day-of-water-year (1-366) for a single date."""
    wy = get_water_year(date)
    june1 = pd.Timestamp(year=wy, month=6, day=1)
    return (date - june1).days + 1


def vectorized_water_year(dates):
    """Return array of water years for a DatetimeIndex."""
    months = dates.month.values
    years = dates.year.values
    return np.where(months >= 6, years, years - 1)


def count_water_years(start_date, end_date, min_days=300):
    """Count full water years in a simulation date range.

    Only water years with at least `min_days` days are counted,
    excluding partial years at the start/end of the simulation.
    """
    idx = pd.date_range(start_date, end_date, freq='D')
    wy = np.where(idx.month >= 6, idx.year, idx.year - 1)
    unique_wys, counts = np.unique(wy, return_counts=True)
    return int(np.sum(counts >= min_days))


def vectorized_water_year_doy(dates):
    """Return array of day-of-water-year (1-366) for a DatetimeIndex."""
    water_years = vectorized_water_year(dates)
    june1_dates = np.array(
        [np.datetime64(f'{y}-06-01') for y in water_years],
        dtype='datetime64[D]',
    )
    dates_np = dates.values.astype('datetime64[D]')
    return (dates_np - june1_dates).astype(int) + 1
