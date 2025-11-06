"""
This script does the following:

- Load gauge flows from all ensemble set gage_flow_mgd files
- Load simulation results from all ensemble set outputs
- Combine the flows and simulation results from all sets into a single ensemble key



Every dictionary in the final data object has format:

dict = {
    inflow_type: {
        realization_id : pd.DataFrame
    }
}

The pd.DataFrame has datetime index and node names as columns. 
"""



import sys
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import pywrdrb
from sglib import SSIDroughtMetrics, SSI
from sglib import Ensemble, HDF5Manager

from methods.utils import combine_multiple_ensemble_sets_in_data
from methods.metrics.shortfall import get_flow_and_target_values, add_trenton_equiv_flow

from methods.utils import combine_multiple_ensemble_sets_in_data
from methods.load import load_drb_reconstruction, load_and_combine_ensemble_sets
from methods.config import RECONSTRUCTION_OUTPUT_FNAME
from methods.config import STATIONARY_ENSEMBLE_SETS, CLIMATE_ADJUSTED_ENSEMBLE_SETS
from methods.config import verify_ensemble_type


if __name__ == "__main__":
    
    ### Settings
    # Get ensemble type from command line arguments
    ensemble_type = sys.argv[1]
    inflow_type = f'{ensemble_type}_ensemble'
    verify_ensemble_type(ensemble_type)
    
    # Use appropriate list of ensemble set specs
    if ensemble_type == 'stationary':
        ensemble_set_specs = STATIONARY_ENSEMBLE_SETS
    elif ensemble_type == 'climate_adjusted':
        ensemble_set_specs = CLIMATE_ADJUSTED_ENSEMBLE_SETS
    else:
        raise ValueError(f"Invalid ensemble type: {ensemble_type}. Must be 'stationary' or 'climate_adjusted'.")
    
    ### Load data through pywrdrb API #######################################
    ## Setup pathnavigator
    pn_config = pywrdrb.get_pn_config()
    for set in ensemble_set_specs:
        dataset_dir = set.directory
        dataset = set.directory.split('/')[-1]
        pn_config[f"flows/{dataset}"] = os.path.abspath(dataset_dir)
    pywrdrb.load_pn_config(pn_config)
    
    ## Synthetic ensemble 
    # Load synthetic ensembles and combine within data object
    # This will load the full natural flow (gage_flow_mgd.hdf5) but NOT the simulation outputs
    # Later the simulation outputs will be loaded and combined into a single export
    ensemble_set_names = [set.directory.split('/')[-1] for set in ensemble_set_specs]
    results_sets = ['major_flow']
    data = pywrdrb.Data(results_sets=results_sets,)
    data.load_hydrologic_model_flow(ensemble_set_names)
    data = combine_multiple_ensemble_sets_in_data(data, results_sets, ensemble_type=ensemble_type)

    # Rename the major_flow to gage_flow
    gage_flow_dict = {}
    gage_flow_dict[inflow_type] = data.major_flow[inflow_type]

    ## Start a new data object for output data
    output_filenames = [ensemble_set_specs[i].output_file for i in range(len(ensemble_set_specs))]
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
    data = pywrdrb.Data(results_sets=results_sets, 
                        print_status=True)
    data.load_output(output_filenames=output_filenames)
    data.load_observations()
    data = add_trenton_equiv_flow(data)
    data = combine_multiple_ensemble_sets_in_data(data, results_sets, ensemble_type=ensemble_type)


    ### Post-process data ##############################################
    # Need to calculate some additional results_sets and add them to the data object later
    # each new results_set will be a dictionary with realization_id as keys
    # and pd.DataFrame as values, with datetime index and node names as columns
    print('Calculating shortages for different nodes...')
    all_shortage_dict = {}
    
    for model in ['reconstruction', inflow_type]:
        shortage_dict = {}
        realizations = list(data.major_flow[model].keys())
        for i, r in enumerate(realizations):            
            if r not in shortage_dict:
                shortage_dict[r] = {}
            for node in ['delMontague', 'delTrenton', 'nyc', 'nj']:
                
                flow_series, target_series = get_flow_and_target_values(data, node,
                                                                        model, r,
                                                                        start_date=None, 
                                                                        end_date=None)
                # Calculate shortages
                shortage_series = target_series - flow_series
                shortage_series[shortage_series < 0] = 0  # Set negative shortages (surplus) to zero
                shortage_series.iloc[:3] = 0.0 # set first 3 days to 0.0 due to model warmup
                
                # ignore shortages when the duration of consecutive shortage>0 days is <3
                shortage_durations = (shortage_series > 0).astype(int).groupby((shortage_series > 0).astype(int).diff().ne(0).cumsum()).cumsum()
                shortage_series[shortage_durations < 3] = 0.0

                shortage_dict[r][node] = shortage_series
            
            # convert to DataFrame
            shortage_dict[r] = pd.DataFrame(shortage_dict[r])

        # Format as: {inflow_type: {realization_id: pd.DataFrame}}
        all_shortage_dict[model] = shortage_dict

    ## Calculate downstream contributions from NYC reservoirs
    # This is the sum of downstream contributions from all 3 reservoirs 
    # used to meet the flow targets at Montague and Trenton
    #TODO: This should be disaggregated for Montague and Trenton
    print('Calculating total downstream contributions...')
    nyc_reservoirs = ['cannonsville', 'pepacton', 'neversink']
    contribution_columns = [f'mrf_montagueTrenton_{res}' for res in nyc_reservoirs]
    all_contribution_dict = {}
    for model in ['reconstruction', inflow_type]:
        contribution_dict = {}
        realizations = list(data.major_flow[model].keys())
        for i, r in enumerate(realizations):
            total_nyc_contribution = data.nyc_release_components[model][r].loc[:, contribution_columns].sum(axis=1)
            contribution_dict[r] = total_nyc_contribution.to_frame(name='mrf_montagueTrenton_nyc')
    
        # Format as: {inflow_type: {realization_id: pd.DataFrame}}
        all_contribution_dict = {model: contribution_dict}
    
    ### Calculate aggregate NYC inflow ###########################################
    # Add this as a new column to data.inflow[inflow_type] for each realization
    
    for model in ['reconstruction', inflow_type]:
        realizations = list(data.inflow[model].keys())
        for i, r in enumerate(realizations):
            data.inflow[model][r].loc[:, 'nyc'] = data.inflow[model][r].loc[:, nyc_reservoirs].sum(axis=1)

    ### Organize data to be kept for later #######################################
    # Includes results_sets:
    # - gage_flow
    # - inflow
    # - res_storage
    # Also, we include the following 'model' options as keys to each dict:
    # - obs
    # - reconstruction
    # - inflow_type
    keep_data = pywrdrb.Data()
    keep_data.gage_flow = gage_flow_dict
    keep_data.shortage = all_shortage_dict
    keep_data.contribution = all_contribution_dict
    
    # Make copies of output results_sets just for the combined ensemble inflow_type
    inflow_dict = {}
    major_flow_dict = {}
    res_storage_dict = {}
    ibt_diversions_dict = {}
    ibt_demands_dict = {}
    for model in ['reconstruction', inflow_type]:
        inflow_dict[model] = data.inflow[model]
        major_flow_dict[model] = data.major_flow[model]
        res_storage_dict[model] = data.res_storage[model]
        ibt_diversions_dict[model] = data.ibt_diversions[model]
        ibt_demands_dict[model] = data.ibt_demands[model]

    keep_data.inflow = inflow_dict    
    keep_data.major_flow = major_flow_dict
    keep_data.res_storage = res_storage_dict
    keep_data.ibt_diversions = ibt_diversions_dict
    keep_data.ibt_demands = ibt_demands_dict
        
    ### Export the new data object to HDF5
    fname = f'./pywrdrb/outputs/{ensemble_type}_ensemble_with_postprocessing.hdf5'
    keep_data.export(fname)
    print(f"Done processing data, saved to file: {fname}")