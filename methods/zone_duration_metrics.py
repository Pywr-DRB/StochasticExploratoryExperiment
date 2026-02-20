"""
Calculate drought zone event duration metrics.

This module calculates event-based duration percentiles for NYC storage zones.
Events are defined as continuous periods where storage remains in a specific zone,
with a minimum ending criterion (must be above zone for 7+ days to end the event).
"""

import numpy as np
import pandas as pd
import pywrdrb
from .config import NYC_RESERVOIRS, NYC_TOTAL_CAPACITY


def calculate_zone_events(zone_series, zone_number, min_end_days=7):
    """
    Identify continuous events where zone equals zone_number.

    An event ends when the zone has been different for min_end_days consecutive days.

    Parameters
    ----------
    zone_series : pd.Series
        Timeseries of zone values with datetime index
    zone_number : int
        Zone to identify events for (4=Warning, 5=Watch, 6=Emergency)
    min_end_days : int
        Minimum number of consecutive days outside zone to end event

    Returns
    -------
    list of dict
        Each dict has keys: 'start_date', 'end_date', 'duration_days'
    """
    events = []
    in_event = False
    event_start = None
    days_outside_zone = 0

    for date, zone in zone_series.items():
        if zone == zone_number:
            if not in_event:
                # Start new event
                in_event = True
                event_start = date
                days_outside_zone = 0
            else:
                # Continue event (reset outside counter)
                days_outside_zone = 0
        else:
            if in_event:
                days_outside_zone += 1
                if days_outside_zone >= min_end_days:
                    # End event
                    event_end = date - pd.Timedelta(days=min_end_days)
                    duration = (event_end - event_start).days + 1
                    if duration > 0:
                        events.append({
                            'start_date': event_start,
                            'end_date': event_end,
                            'duration_days': duration
                        })
                    in_event = False
                    event_start = None
                    days_outside_zone = 0

    # Handle case where event extends to end of series
    if in_event:
        event_end = zone_series.index[-1]
        duration = (event_end - event_start).days + 1
        if duration > 0:
            events.append({
                'start_date': event_start,
                'end_date': event_end,
                'duration_days': duration
            })

    return events


def calculate_zone_duration_percentiles(dataset_id, zones=[4, 5, 6],
                                       percentiles=[5, 50, 95],
                                       min_end_days=7):
    """
    Calculate duration percentiles for drought zone events across all realizations.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    zones : list of int
        Zone numbers to analyze (default: [4, 5, 6] = Warning, Watch, Emergency)
    percentiles : list of int
        Percentiles to calculate (default: [5, 50, 95])
    min_end_days : int
        Minimum days outside zone to end event

    Returns
    -------
    pd.DataFrame
        Columns: zone, p5, p50, p95 (duration in days)
        Index: zone numbers
    """
    fname = f'./pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5'
    data = pywrdrb.Data()
    data.load_from_export(fname, results_sets=['res_level'])

    results = []

    for zone_num in zones:
        all_durations = []

        realizations = sorted(data.res_level[dataset_id].keys())
        for r in realizations:
            zone_series = data.res_level[dataset_id][r]['nyc']
            events = calculate_zone_events(zone_series, zone_num, min_end_days)
            durations = [e['duration_days'] for e in events]
            all_durations.extend(durations)

        # Calculate percentiles
        if len(all_durations) > 0:
            pct_values = np.percentile(all_durations, percentiles)
            result = {'zone': zone_num}
            for i, p in enumerate(percentiles):
                result[f'p{p}'] = pct_values[i]
            results.append(result)
        else:
            # No events found
            result = {'zone': zone_num}
            for p in percentiles:
                result[f'p{p}'] = 0
            results.append(result)

    df = pd.DataFrame(results)
    df = df.set_index('zone')
    return df


def calculate_drought_zone_events(zone_series, min_end_days=7):
    """
    Identify contiguous drought episodes and attribute each to its most severe zone.

    A drought episode is a contiguous period where the NYC combined zone level is >= 4
    (Warning, Watch, or Emergency). The episode ends once the zone has been below 4 for
    at least min_end_days consecutive days. The episode is attributed to the maximum zone
    level reached at any point during it; lower zones passed through on the way up or down
    are not recorded separately.

    Parameters
    ----------
    zone_series : pd.Series
        Daily timeseries of NYC combined zone levels with a datetime index.
        Zone definitions: 6=Emergency, 5=Watch, 4=Warning, 3=Normal, 1-2=Flood/Above.
    min_end_days : int
        Minimum consecutive days below zone 4 required to end an episode (default: 7).

    Returns
    -------
    list of dict
        One dict per episode with keys:
        - 'start_date'    : pd.Timestamp, first day of the episode (zone >= 4)
        - 'end_date'      : pd.Timestamp, last day of the episode (zone >= 4)
        - 'duration_days' : int, total episode length in days (inclusive)
        - 'max_zone'      : int, maximum zone level reached during the episode
    """
    events = []
    in_episode = False
    episode_start = None
    last_drought_date = None
    days_outside = 0
    max_zone_in_episode = 0

    for date, zone in zone_series.items():
        if zone >= 4:
            if not in_episode:
                in_episode = True
                episode_start = date
                days_outside = 0
            else:
                days_outside = 0
            last_drought_date = date
            max_zone_in_episode = max(max_zone_in_episode, int(zone))
        else:
            if in_episode:
                days_outside += 1
                if days_outside >= min_end_days:
                    duration = (last_drought_date - episode_start).days + 1
                    if duration > 0:
                        events.append({
                            'start_date': episode_start,
                            'end_date': last_drought_date,
                            'duration_days': duration,
                            'max_zone': max_zone_in_episode,
                        })
                    in_episode = False
                    episode_start = None
                    last_drought_date = None
                    days_outside = 0
                    max_zone_in_episode = 0

    # Handle episode that extends to the end of the series
    if in_episode and last_drought_date is not None:
        duration = (last_drought_date - episode_start).days + 1
        if duration > 0:
            events.append({
                'start_date': episode_start,
                'end_date': last_drought_date,
                'duration_days': duration,
                'max_zone': max_zone_in_episode,
            })

    return events


def calculate_zone_frequency(dataset_id, zones=[4, 5, 6]):
    """
    Calculate frequency of years where minimum storage falls into each zone.

    Uses the "exactly" definition: the worst zone reached in that year.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    zones : list of int
        Zone numbers to analyze (default: [4, 5, 6])

    Returns
    -------
    dict
        {zone_num: fraction_of_years}
    """
    from .load import load_performance_metrics

    metrics = load_performance_metrics(dataset_id)

    zone_mapping = {
        4: 'years_exactly_warning',
        5: 'years_exactly_watch',
        6: 'years_exactly_emergency',
    }

    n_years_per_realization = 70  # Typical simulation length

    result = {}
    for zone_num in zones:
        metric_key = zone_mapping.get(zone_num)
        if metric_key and metric_key in metrics.columns:
            # Get fraction across realizations
            fractions = metrics[metric_key] / n_years_per_realization
            result[zone_num] = {
                'mean': fractions.mean(),
                'p5': fractions.quantile(0.05),
                'p50': fractions.quantile(0.50),
                'p95': fractions.quantile(0.95),
            }
        else:
            result[zone_num] = {'mean': 0, 'p5': 0, 'p50': 0, 'p95': 0}

    return result
