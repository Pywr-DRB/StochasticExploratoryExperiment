"""
Calculate drought zone event duration metrics.

This module calculates event-based duration percentiles for NYC storage zones.
Events are defined as continuous periods where storage remains in a specific zone,
with a minimum ending criterion (must be above zone for 7+ days to end the event).
"""


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
        Zone definitions: 6=Emergency, 5=Warning, 4=Watch, 3=Normal, 1-2=Flood/Above.
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


