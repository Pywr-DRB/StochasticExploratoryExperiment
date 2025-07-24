import sys
import numpy as np
import pandas as pd

import pywrdrb

from methods.utils import combine_multiple_ensemble_sets_in_data
from methods.metrics.shortfall import get_flow_and_target_values, add_trenton_equiv_flow
from config import RECONSTRUCTION_OUTPUT_FNAME
from config import STATIONARY_ENSEMBLE_SETS, CLIMATE_ADJUSTED_ENSEMBLE_SETS
from config import verify_ensemble_type

if __name__ == "__main__":
    
    # Get ensemble type from command line arguments
    ensemble_type = sys.argv[1]
    inflow_type = f'{ensemble_type}_ensemble'
    verify_ensemble_type(ensemble_type)

    if ensemble_type == 'stationary':
        ensemble_set_spec_list = STATIONARY_ENSEMBLE_SETS
    elif ensemble_type == 'climate_adjusted':
        ensemble_set_spec_list = CLIMATE_ADJUSTED_ENSEMBLE_SETS

    # Load pywrdrb output
    output_filenames = [ensemble_set_spec_list[i].output_file for i in range(len(ensemble_set_spec_list))]
    output_filenames.append(RECONSTRUCTION_OUTPUT_FNAME)

    results_sets = [
        "major_flow", "inflow", "res_storage",
        "lower_basin_mrf_contributions", "mrf_target", 
        "ibt_diversions", "ibt_demands",
    ]

    ### Load simulation output data
    data = pywrdrb.Data(results_sets=results_sets, print_status=True)
    data.load_output(output_filenames=output_filenames)
    data.load_observations()

    data = add_trenton_equiv_flow(data)

    data = combine_multiple_ensemble_sets_in_data(data, results_sets, ensemble_type=ensemble_type)

    # Load ensemble drought event data
    fname = f"./pywrdrb/drought_metrics/{ensemble_type}_ensemble_drought_events.csv"
    synthetic_drought_events = pd.read_csv(fname)

    fname = f"./pywrdrb/drought_metrics/observed_drought_events.csv"
    observed_drought_events = pd.read_csv(fname)

    # For each drought event, we calculate the following performance metrics:
    # Max shortage 
    # total shortage
    # max duration of shortage
    # date of max shortage
    for col in ['max_shortage', 'total_shortage', 'max_duration', 'date_of_max_shortage']:
        synthetic_drought_events[f'{col}_delMontague'] = np.nan
        synthetic_drought_events[f'{col}_delTrenton'] = np.nan

    ### Calculate shortage percentiles

    print('Calculating shortage percentiles...')

    model = inflow_type

    for node in ['delMontague', 'delTrenton']:
        
        # Loop through drought events and calculate metrics during the drought
        realizations = synthetic_drought_events['realization_id'].unique()

        for i, r in enumerate(realizations):
            if i % 50 == 0:
                print(f"Calculating {node} shortfall metrics for realization {i+1} of {len(realizations)}...")
                
            # Get drought events with this realization
            drought_events = synthetic_drought_events[synthetic_drought_events['realization_id'] == r]
            
            # If empty, skip
            if drought_events.empty:
                print(f"No drought events found for realization {r}.")
                continue
            
            # Loop through drought events
            for _, drought_event in drought_events.iterrows():
                # Get the start and end dates for the drought event
                start_date = drought_event['start']
                end_date = drought_event['end']
                # Get the flow and target values for this node, model, and realization
                flow_series, target_series = get_flow_and_target_values(data, node,
                                                                        model, r,
                                                                        start_date=start_date, 
                                                                        end_date=end_date)
                # Calculate shortages
                shortage_series = target_series - flow_series
                shortage_series[shortage_series < 0] = 0  # Set negative shortages                
                
                synthetic_drought_events.loc[drought_event.name, f'max_shortage_{node}'] = shortage_series.max()
                synthetic_drought_events.loc[drought_event.name, f'total_shortage_{node}'] = shortage_series.sum()
                
                # max duration of continuous shortage
                shortage_durations = (shortage_series > 0).astype(int).groupby((shortage_series > 0).astype(int).diff().ne(0).cumsum()).cumsum()
                max_duration = shortage_durations.max()
                synthetic_drought_events.loc[drought_event.name, f'max_duration_{node}'] = max_duration
                
                # date of max shortage
                if max_duration > 0:
                    max_shortage_date = shortage_series.idxmax()
                    synthetic_drought_events.loc[drought_event.name, f'date_of_max_shortage_{node}'] = max_shortage_date

    # Save the results
    fname = f"./pywrdrb/drought_metrics/{ensemble_type}_ensemble_drought_events_with_shortage_metrics.csv"

    synthetic_drought_events.to_csv(fname, index=False)
    print(f"Saved synthetic drought events with metrics to {fname}")
