"""
This script does post-processing of the pywrdrb output to calculate:
- Shortages for different nodes
- Total downstream flow contributions

Each data is organized as:
{'key': pd.DataFrame}

Where the pd.DataFrame has realizations as columns and time as index.

The results are saved in an HDF5 file for later.
"""
import sys
import numpy as np
import pandas as pd

import pywrdrb
from sglib import HDF5Manager

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
        "major_flow", 
        "inflow", 
        "res_storage",
        "lower_basin_mrf_contributions", 
        "mrf_target", 
        "ibt_diversions", 
        "ibt_demands",
        "nyc_release_components"
    ]
    
    ### Load simulation output data
    data = pywrdrb.Data(results_sets=results_sets, print_status=True)
    data.load_output(output_filenames=output_filenames)
    data = add_trenton_equiv_flow(data)
    data = combine_multiple_ensemble_sets_in_data(data, results_sets, ensemble_type=ensemble_type)

    ### Calculate shortages
    print('Calculating shortages for different nodes...')
    shortage_dict = {}
    realizations = list(data.major_flow[inflow_type].keys())
    model = inflow_type
    for node in ['delMontague', 'delTrenton', 'nyc', 'nj']:
        if node not in shortage_dict:
            shortage_dict[node] = {}
        print(f"Calculating shortfalls for node {node}...")
        
        for i, r in enumerate(realizations):            
            flow_series, target_series = get_flow_and_target_values(data, node,
                                                                    model, r,
                                                                    start_date=None, 
                                                                    end_date=None)
            # Calculate shortages
            shortage_series = target_series - flow_series
            shortage_series[shortage_series < 0] = 0  # Set negative shortages (surplus) to zero
            
            # Set first 2 days to 0.0 due to model warmup
            shortage_series.iloc[:2] = 0.0
            
            shortage_dict[node][r] = shortage_series
        
        # convert to DataFrame
        shortage_dict[node] = pd.DataFrame(shortage_dict[node])    
        shortage_dict[node].columns = shortage_dict[node].columns.astype(str)

    ### Caclulate total downstream contribution requirements
    print('Calculating total downstream contributions...')
    contribution_dict = {}
    contribution_dict['mrf_montagueTrenton_nyc'] = {}
    nyc_reservoirs = ['cannonsville', 'pepacton', 'neversink']
    contribution_columns = [f'mrf_montagueTrenton_{res}' for res in nyc_reservoirs]
    for i, r in enumerate(realizations):
        total_nyc_contribution = data.nyc_release_components[inflow_type][r].loc[:, contribution_columns].sum(axis=1)
        contribution_dict['mrf_montagueTrenton_nyc'][r] = total_nyc_contribution
        
    # convert to DataFrame
    contribution_dict['mrf_montagueTrenton_nyc'] = pd.DataFrame(contribution_dict['mrf_montagueTrenton_nyc'])    
    contribution_dict['mrf_montagueTrenton_nyc'].columns = contribution_dict['mrf_montagueTrenton_nyc'].columns.astype(str)
    
    ### Export to hdf5
    shortage_fname = f"./pywrdrb/shortages/{ensemble_type}_ensemble_shortages.hdf5"
    print(f"Saving shortages to hdf5: {shortage_fname}")
    hdf_manager = HDF5Manager()
    hdf_manager.export_ensemble_to_hdf5(
        dict=shortage_dict,
        output_file=shortage_fname
    )
    
    contribution_fname = f"./pywrdrb/contributions/{ensemble_type}_ensemble_contributions.hdf5"
    print(f"Saving contributions to hdf5: {contribution_fname}")
    hdf_manager.export_ensemble_to_hdf5(
        dict=contribution_dict,
        output_file=contribution_fname
    )