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

    # Load ensemble data from processed HDF5 - should have everything we need inside
    fname = f'./pywrdrb/outputs/{ensemble_type}_ensemble_with_postprocessing.hdf5'
    data = pywrdrb.Data()
    data.load_from_export(fname)
    
    
    
    # Loop through different SSI windows
    for SSI_WINDOW in [12]: # [3,6,12]
        # Load ensemble drought event data
        fname = f"./pywrdrb/drought_metrics/{ensemble_type}_ensemble_ssi{SSI_WINDOW}_drought_events.csv"
        synthetic_drought_events = pd.read_csv(fname)

        fname = f"./pywrdrb/drought_metrics/observed_ssi{SSI_WINDOW}_drought_events.csv"
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
                    
                    # Shortage timeseries was calculated in 05_postprocess_data.py
                    # and is available in the data object
                    shortage_series = data.shortage[inflow_type][r].loc[start_date:end_date, node]

                    # get shortage metrics during this drought
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
        fname = f"./pywrdrb/drought_metrics/{ensemble_type}_ensemble_ssi{SSI_WINDOW}_drought_events_with_shortage_metrics.csv"

        synthetic_drought_events.to_csv(fname, index=False)
        print(f"Saved synthetic drought events with metrics to {fname}")
    print(f"Done with all SSI-shortage calculations for {ensemble_type}")
