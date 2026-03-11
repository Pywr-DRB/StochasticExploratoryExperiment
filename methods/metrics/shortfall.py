import numpy as np
import pandas as pd

import pywrdrb
from pywrdrb.pywr_drb_node_data import downstream_node_lags, immediate_downstream_nodes_dict
from pywrdrb.utils.timeseries import subset_timeseries
from pywrdrb.utils.constants import cfs_to_mgd

from methods.config import DEFAULT_SHORTAGE_TOLERANCE_MGD


def calculate_shortage_series(target, flow, tolerance=None,
                              min_duration=3, warmup_days=3):
    """
    Calculate shortage timeseries from target/demand and flow/delivery series.

    This is the single authoritative function for computing shortage timeseries
    anywhere in the codebase.  All other code that needs a shortage series
    should call this rather than reimplementing the logic.

    Parameters
    ----------
    target : pd.Series
        Target or demand timeseries (MGD).
    flow : pd.Series
        Actual flow or delivery timeseries (MGD).
    tolerance : float or None
        Minimum shortage magnitude (MGD) to retain.  Values below this are
        set to 0.  If None, uses ``DEFAULT_SHORTAGE_TOLERANCE_MGD`` from
        config (currently 1.0 MGD).
    min_duration : int
        Minimum consecutive days for a shortage event.  Events shorter than
        this are zeroed out.  Set to 0 to disable.  Default: 3.
    warmup_days : int
        Number of days at the start of the series to zero out (model
        warm-up artefacts).  Set to 0 to disable.  Default: 3.

    Returns
    -------
    pd.Series
        Shortage timeseries with tolerance and duration filters applied.
    """
    if tolerance is None:
        tolerance = DEFAULT_SHORTAGE_TOLERANCE_MGD

    if not isinstance(target, pd.Series) or not isinstance(flow, pd.Series):
        raise TypeError(
            "Both target and flow must be pd.Series with datetime index. "
            f"Got target={type(target).__name__}, flow={type(flow).__name__}."
        )
    if len(target) != len(flow):
        raise ValueError(
            f"target and flow must have the same length. "
            f"Got {len(target)} and {len(flow)}."
        )
    if len(target) == 0:
        raise ValueError("target and flow must not be empty.")

    shortage = (target - flow).clip(lower=0)

    # Zero out model warm-up period
    if warmup_days > 0 and len(shortage) > warmup_days:
        shortage.iloc[:warmup_days] = 0.0

    # Apply tolerance — treat sub-tolerance values as zero
    shortage[shortage < tolerance] = 0.0

    # Filter short-duration events
    if min_duration > 0:
        positive = (shortage > 0).astype(int)
        durations = positive.groupby(
            positive.diff().ne(0).cumsum()
        ).cumsum()
        shortage[durations < min_duration] = 0.0

    return shortage


def calculate_shortage_by_day_of_year(data, dataset_id, location):
    """
    Calculate shortage occurrence by day of year for a specific location.

    Parameters
    ----------
    data : pywrdrb.Data
        Data object with shortage or demand/delivery data
    dataset_id : str
        Dataset identifier
    location : str
        Location identifier: 'delMontague', 'delTrenton', or 'nyc'

    Returns
    -------
    np.ndarray
        Array of length 366 with count of shortage days for each day of year
    """
    realizations = list(data.shortage[dataset_id].keys())
    n_realizations = len(realizations)

    print(f"  Processing {location}...")
    print(f"    Realizations: {n_realizations}")

    # Initialize array for all days of year (366 to account for leap years)
    shortage_counts = np.zeros(366, dtype=int)

    for r in realizations:
        if location in ['delMontague', 'delTrenton']:
            # Use pre-calculated shortage
            shortage = data.shortage[dataset_id][r][location]

            # Shortage > 0 means violation
            violation_days = shortage > 0

        elif location == 'nyc':
            # Calculate NYC diversion shortage using central function
            delivery = data.ibt_diversions[dataset_id][r]['delivery_nyc']
            demand = data.ibt_demands[dataset_id][r]['demand_nyc']

            shortage = calculate_shortage_series(
                demand, delivery, min_duration=0, warmup_days=0
            )

            # Any shortage > 0 is a violation
            violation_days = shortage > 0

        else:
            raise ValueError(f"Unknown location: {location}")

        # Get day of year for each violation
        dates = violation_days.index
        day_of_year = dates.dayofyear

        # Count violations for each day of year
        for doy, is_violation in zip(day_of_year, violation_days):
            if is_violation:
                shortage_counts[doy - 1] += 1  # Convert 1-indexed to 0-indexed

    print(f"    Total shortage days: {shortage_counts.sum():,}")
    print(f"    Max shortage days for a single DOY: {shortage_counts.max()}")

    return shortage_counts

def calculate_hashimoto_metrics(flows,
                                thresholds,
                                eps=1e-9,
                                shortfall_break_length=7,
                                tolerance=None):
    
    ### Check inputs
    # Make sure both have datetime index
    if not isinstance(flows, pd.Series) or not isinstance(thresholds, pd.Series):
        raise ValueError("Both flows and thresholds must be pandas Series with datetime index.")

    if len(flows) != len(thresholds):
        raise ValueError("Flows and thresholds must have the same length.")
    
    # Get the dates for later
    dates = flows.index
    
    # now convert to numpy arrays
    flows = flows.values
    thresholds = thresholds.values

    # Resolve tolerance
    if tolerance is None:
        tolerance = DEFAULT_SHORTAGE_TOLERANCE_MGD

    # Deficit array with tolerance applied (deficits below tolerance are zero)
    deficits = np.maximum(thresholds - flows, 0)
    is_deficit = deficits >= tolerance

    ### reliability is the fraction of time steps without a deficit (above tolerance)
    reliability_frac = (~is_deficit).mean()

    ### resiliency is the probability of recovering if currently in deficit
    if reliability_frac < 1 - eps:
        resiliency = np.logical_and(is_deficit[:-1],
                                    ~is_deficit[1:]).mean() / (1 - reliability_frac)
    else:
        resiliency = np.nan

    ### define individual events & get event-specific metrics
    durations = []          # length of each event
    intensities = []        # intensity of each event = avg deficit within event
    severities = []         # severity = duration * intensity
    vulnerabilities = []    # vulnerability = max daily deficit within event
    event_starts = []       # define event to start with nonzero shortfall and end with the next shortfall date that preceeds shortfall_break_length non-shortfall dates.
    event_ends = []

    if reliability_frac > eps and reliability_frac < 1 - eps:
        duration = 0
        severity = 0
        vulnerability = 0
        in_event = False
        for i in range(len(flows)):
            d = dates[i]
            if in_event or is_deficit[i]:
                ### is this the start of a new event?
                if not in_event:
                    event_starts.append(d)

                ### if this is part of event, we add to metrics whether today is deficit or not
                duration += 1
                s = deficits[i]
                severity += s
                vulnerability = max(vulnerability, s)
                ### now check if next shortfall_break_length days include any deficits. if not, end event.
                in_event = np.any(is_deficit[i+1: i+1+shortfall_break_length])
                if not in_event:
                    event_ends.append(dates[min(i+1, len(dates)-1)])
                    durations.append(duration)
                    severities.append(severity)
                    intensities.append(severity / duration)
                    vulnerabilities.append(vulnerability)
                    in_event = False
                    duration = 0
                    severity = 0
                    vulnerability = 0

    # Combine into a pd.DataFrame
    events_df = pd.DataFrame({
        'start': event_starts,
        'end': event_ends,
        'duration': durations,
        'severity': severities,
        'intensity': intensities,
        'vulnerability': vulnerabilities
    })
    
    ### Results dict will contain:
    # 'reliability': float,
    # 'resiliency': float,
    # 'events': pd.DataFrame with columns:
    #   'start', 'end', 'duration', 'severity', 'intensity', 'vulnerability'
    resultsdict = {}
    resultsdict['reliability'] = reliability_frac * 100
    resultsdict['resiliency'] = resiliency * 100
    resultsdict['events'] = events_df

    return resultsdict


def add_trenton_equiv_flow(data):

    blueMarsh_conservation_release_mgd = 50 * cfs_to_mgd
    
    ### Check data requirements
    # make sure data is a pywrdrb.Data object
    assert isinstance(data, pywrdrb.Data), \
        "data must be a pywrdrb.Data object."
    
    # data must have major_flow and res_release attributes
    necessary_results_sets = [
        "major_flow",
        "res_release"
    ]
    for result_set in necessary_results_sets:
        if not hasattr(data, result_set):
            raise ValueError(
                f"pywrdrb.Data object must contain {result_set} as an attribute."
                )
            
    ### Get models and realizations
    models = list(data.major_flow.keys())
    
    ### Loop through models and realizations
    for m in models:
        
        # get model-specific realizations
        realizations = list(data.major_flow[m].keys())
        
        for r in realizations:
            
            # get major flow for this model and realization
            flows = data.major_flow[m][r]['delTrenton']
            flows = flows.copy()
            
            # add blueMarsh releases beyond the conservation release
            if m in data.res_release:
                blueMarsh_total_release = data.res_release[m][r]['blueMarsh']
                blueMarsh_excess_release = blueMarsh_total_release - blueMarsh_conservation_release_mgd
                blueMarsh_excess_release[blueMarsh_excess_release < 0] = 0
                
                # account for lag at blue marsh
                lag = downstream_node_lags['blueMarsh']
                downstream_node = immediate_downstream_nodes_dict['blueMarsh']
                while downstream_node != 'output_del':
                    lag += downstream_node_lags[downstream_node]
                    downstream_node = immediate_downstream_nodes_dict[downstream_node]
                
                if lag > 0:
                    blueMarsh_excess_release.iloc[lag:] = blueMarsh_excess_release.iloc[:-lag]
                
                flows += blueMarsh_excess_release

                # Store in the data.major_flow attribute
                data.major_flow[m][r]['delTrenton_equiv'] = flows
                
            else:
                # Raise warning that this model is skipped
                print(
                    f"Model {m} does not have res_release, skipping trenton equiv flow calc for this data."
                )
                
    return data


def get_flow_and_target_values(data, 
                               node, 
                               model, 
                               realization,
                               start_date,
                               end_date):
    
    m = model
    r = realization
    
    # Shortage at flow target locations
    if node in ['delTrenton', 'delMontague']:        
        if node == 'delTrenton':
            # For delTrenton, we use the delTrenton_equiv flow
            flows = data.major_flow[m][r]['delTrenton_equiv']
        elif node == 'delMontague':
            # For Montague, we use the major flow directly
            flows = data.major_flow[m][r][node]
            
        # for observational data, we don't have the mrf_target
        # in this case, we want to use targets from the reconstruction period
        # this should be the best match for the change in flow targets requirements
        if m == 'obs':
            use_model = 'reconstruction'
            thresholds = data.mrf_target[use_model][r][node]
            
        # otherwise, use the simulated mrf_target
        else:
            thresholds = data.mrf_target[m][r][node]
        
        # subset the timeseries to the specified date range
        flows = subset_timeseries(flows, start_date, end_date)
        thresholds = subset_timeseries(thresholds, start_date, end_date)        
        
    # Shortage of diversions for NYC and NJ diversions
    elif node in ['nyc', 'nj']:
        ibt_diversions = data.ibt_diversions[m][r]
        ibt_demands = data.ibt_demands[m][r]
        flows = subset_timeseries(ibt_diversions[f'delivery_{node}'], start_date, end_date)
        thresholds = subset_timeseries(ibt_demands[f'demand_{node}'], start_date, end_date)

    # Not currently supported for any other nodes
    else:
        raise ValueError(f"Not setup to handle node {node} in get_flow_and_target_values().")
    
    return flows, thresholds
    



