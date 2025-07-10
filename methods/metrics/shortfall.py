import numpy as np
import pandas as pd

import pywrdrb
from pywrdrb.utils.lists import majorflow_list
from pywrdrb.pywr_drb_node_data import downstream_node_lags, immediate_downstream_nodes_dict
from pywrdrb.utils.timeseries import subset_timeseries


def calculate_hashimoto_metrics(flows, 
                                thresholds,
                                eps=1e-9,
                                shortfall_break_length=7):
    
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

    ### reliability is the fraction of time steps above threshold
    reliability = (flows > thresholds).mean()
    
    ### resiliency is the probability of recovering to above threshold if currently under threshold
    if reliability < 1 - eps:
        resiliency = np.logical_and(flows[:-1] < thresholds[:-1], \
                                    (flows[1:] >= thresholds[1:])).mean() / (1 - reliability)
    else:
        resiliency = np.nan

    ### define individual events & get event-specific metrics
    durations = []          # length of each event
    intensities = []        # intensity of each event = avg deficit within event
    severities = []         # severity = duration * intensity
    vulnerabilities = []    # vulnerability = max daily deficit within event
    event_starts = []       # define event to start with nonzero shortfall and end with the next shortfall date that preceeds shortfall_break_length non-shortfall dates.
    event_ends = []
    
    if reliability > eps and reliability < 1 - eps:
        duration = 0
        severity = 0
        vulnerability = 0
        in_event = False
        for i in range(len(flows)):
            v = flows[i]
            t = thresholds[i]
            d = dates[i]
            if in_event or v < t:
                ### is this the start of a new event?
                if not in_event:
                    event_starts.append(d)
                
                ### if this is part of event, we add to metrics whether today is deficit or not
                duration += 1
                s = max(t - v, 0)
                severity += s
                vulnerability = max(vulnerability, s)
                ### now check if next shortfall_break_length days include any deficits. if not, end event.
                in_event = np.any(flows[i+1: i+1+shortfall_break_length] < \
                                    thresholds[i+1: i+1+shortfall_break_length])
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
    resultsdict['reliability'] = reliability * 100
    resultsdict['resiliency'] = resiliency * 100
    resultsdict['events'] = events_df

    return resultsdict


def add_trenton_equiv_flow(data):
    
    ### Check data requirements
    # make sure data is a pywrdrb.Data object
    assert isinstance(data, pywrdrb.Data), \
        "data must be a pywrdrb.Data object."
    
    # data must have major_flow and lower_basin_mrf_contributions attributes
    necessary_results_sets = [
        "major_flow",
        "lower_basin_mrf_contributions"
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
            
            # add blueMarsh contributions, accounting for lag
            if m in data.lower_basin_mrf_contributions:
                lower_basin_mrf_contributions = data.lower_basin_mrf_contributions[m][r]
                lower_basin_mrf = lower_basin_mrf_contributions['mrf_trenton_blueMarsh']
                
                # account for lag at blue marsh
                lag = downstream_node_lags['blueMarsh']
                downstream_node = immediate_downstream_nodes_dict['blueMarsh']
                while downstream_node != 'output_del':
                    lag += downstream_node_lags[downstream_node]
                    downstream_node = immediate_downstream_nodes_dict[downstream_node]
                
                if lag > 0:
                    lower_basin_mrf.iloc[lag:] = lower_basin_mrf.iloc[:-lag]
                
                flows += lower_basin_mrf

                # Store in the data.major_flow attribute
                data.major_flow[m][r]['delTrenton_equiv'] = flows
                
            else:
                # Raise warning that this model is skipped
                print(
                    f"Model {m} does not have lower_basin_mrf_contributions, skipping trenton equiv flow calc for this data."
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
    



def get_shortfall_metrics(data, 
                          nodes,
                          models = None, 
                          shortfall_threshold=0.95, 
                          shortfall_break_length=7, 
                          units='MG',
                          start_date=None, 
                          end_date=None):
    """

    """

    ### Check inputs
    # List of results_sets which are used in the calculation.
    # each of these should be an attribute of the data object. 
    necessary_results_sets = []
    if 'delMontague' or 'delTrenton' in nodes:
        necessary_results_sets.append("major_flow")
        necessary_results_sets.append("mrf_target")
        if 'delTrenton' in nodes:
            necessary_results_sets.append("lower_basin_mrf_contributions")            
    if 'nyc' in nodes or 'nj' in nodes:
        necessary_results_sets.append("ibt_diversions")
        necessary_results_sets.append("ibt_demands")

    for result_set in necessary_results_sets:
        if not hasattr(data, result_set):
            raise ValueError(
                f"pywrdrb.Data object must contain {result_set} as an attribute."
                )

    # If start_date and end_date are not provided, 
    # use the full range, but this will be different 
    # based on different datasets/model runs
    start_date_input = start_date
    end_date_input = end_date
    units_daily = 'BGD' if units == 'MG' else 'MCM/D' 
    eps = 1e-9
    
    ### Get a list of the models available in the data object
    if models is None:
        models = list(data.major_flow.keys())

    ### Add the Trenton equivalent flow to the data object
    # The Trenton flow target is based on the 'Trenton Equivalent Flow'
    # which is the sum of the Trenton flow and additional Blue Marsh releases toward trenton.
    # This only needs to be done if:
    # 1. 'delTrenton' is one of the nodes.
    # 2. The `delTrenton_equiv` is not already in the major_flow results.
    if 'delTrenton' in nodes:
        data = add_trenton_equiv_flow(data)

    ### Storage
    # final shortage_events will be organized as:
    # shortage_events[model][scenario/realization] = pd.DataFrame with columns:
    #   'start', 'end', 'duration', 'severity', 'intensity', 'vulnerability', 'node'
    shortage_event_dict = {}
    reliability_dict = {}
    resiliency_dict = {}

    for m in models:
        shortage_event_dict[m] = {}
        reliability_dict[m] = {}
        resiliency_dict[m] = {}
        
        realizations = list(data.major_flow[m].keys())
        realizations = [int(r) for r in realizations]  # Ensure realizations are integers

        for r in realizations:
            
            ### Setup dataframes 
            # store results for this specific realization    
            realization_df_setup = False
            
            ### Different nodes indicate different shortage types
            # if node is in majorflow_list, then shortage = flow target violation
            # if node is 'nyc' or 'nj', then shortage = IBT diversion > demand
            for node in nodes:
                
                # Get stat_date from data if not provided
                if start_date_input is None:                    
                    start_date = data.major_flow[m][r].index[0]
                    end_date = data.major_flow[m][r].index[-1]
                else:
                    start_date = pd.to_datetime(start_date_input)
                    end_date = pd.to_datetime(end_date_input)
                    # If start_date or end_date are not in the index,
                    # subset the timeseries to the closest available dates
                    if start_date not in data.major_flow[m][r].index:
                        start_date = data.major_flow[m][r].index[data.major_flow[m][r].index.get_loc(start_date, method='nearest')]
                    if end_date not in data.major_flow[m][r].index:
                        end_date = data.major_flow[m][r].index[data.major_flow[m][r].index.get_loc(end_date, method='nearest')]
                
                if m == 'obs':
                    start_date = '1960-01-01'
                    end_date = '2023-12-31'
                

                # Get the flow and target values for this node, model, and realization
                flows, thresholds = get_flow_and_target_values(data, node,
                                                                m, r,
                                                                start_date, end_date)
                
                # Apply the shortfall threshold
                # This is only applied to the thresholds, not the flows.
                thresholds = thresholds * shortfall_threshold - eps

                ### Calculate Hashimoto metrics for specific flow and threshold
                # This function expects both flows and thresholds to be pandas 
                # Series with datetime index.
                node_shortage_resultsdict = calculate_hashimoto_metrics(flows, 
                                                                        thresholds, 
                                                                        eps=eps, 
                                                                        shortfall_break_length=shortfall_break_length)

                reliability = node_shortage_resultsdict['reliability']
                resiliency = node_shortage_resultsdict['resiliency']
                events = node_shortage_resultsdict['events']
                
                # Add the node name to the events dataframe
                events['node'] = node

                # Add reliability and resiliency to the results dataframes
                if not realization_df_setup:
                    reliability_df = pd.DataFrame({'node': [node], 'value': [reliability]})
                    resiliency_df = pd.DataFrame({'node': [node], 'value': [resiliency]})
                    events_df = events.copy()
                    realization_df_setup = True
                else:
                    reliability_df = pd.concat([reliability_df, 
                                                pd.DataFrame({'node': [node], 'value': [reliability]})], 
                                                ignore_index=True)
                    resiliency_df = pd.concat([resiliency_df,
                                                pd.DataFrame({'node': [node], 'value': [resiliency]})], 
                                                ignore_index=True)
                    
                    if not events.empty:
                        # Add the events to the events dataframe
                        events_df = pd.concat([events_df, 
                                            events], 
                                            ignore_index=True)

            # Reset index                
            reliability_df.reset_index(drop=True, inplace=True)
            resiliency_df.reset_index(drop=True, inplace=True)
            events_df.reset_index(drop=True, inplace=True)

            # Save data in the dictionaries
            reliability_dict[m][r] = reliability_df.copy()
            resiliency_dict[m][r] = resiliency_df.copy()
            shortage_event_dict[m][r] = events_df.copy()

    return shortage_event_dict, reliability_dict, resiliency_dict



    
    
