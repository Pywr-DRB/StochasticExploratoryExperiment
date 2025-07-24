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
from methods.load import load_drb_reconstruction, load_and_combine_ensemble_sets
from config import STATIONARY_ENSEMBLE_SETS, CLIMATE_ADJUSTED_ENSEMBLE_SETS
from config import verify_ensemble_type

if __name__ == "__main__":

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

    ### Load data through pywrdrb API    
    # Setup pathnavigator
    pn_config = pywrdrb.get_pn_config()
    for set in ensemble_set_specs:
        dataset_dir = set.directory
        dataset = set.directory.split('/')[-1]
        pn_config[f"flows/{dataset}"] = os.path.abspath(dataset_dir)
    pywrdrb.load_pn_config(pn_config)

    ### Loading data
    ## Historic reconstruction data
    # Total flow
    Q = load_drb_reconstruction()
    Q.replace(0, np.nan, inplace=True)
    Q.drop(columns=['delTrenton'], inplace=True)  # Remove Trenton gage as it is not used in the ensemble
    Q_monthly = Q.resample('MS').sum()

    print(f"Loaded reconstruction data with {Q.shape[0]// 365} years of daily data for {Q.shape[1]} sites.")

    ## Synthetic ensemble
    # Load synthetic ensembles and combine within data object
    ensemble_set_names = [set.directory.split('/')[-1] for set in ensemble_set_specs]
    results_sets = ['major_flow']
    data = pywrdrb.Data(results_sets=results_sets,)
    data.load_hydrologic_model_flow(ensemble_set_names)
    data = combine_multiple_ensemble_sets_in_data(data, results_sets, ensemble_type=ensemble_type)

    # Keep just the combined ensemble dict
    syn_ensemble = data.major_flow[inflow_type]
    realization_ids = list(syn_ensemble.keys())
    n_realizations = len(realization_ids)

    print(f"Loaded synthetic {ensemble_type} ensemble with {n_realizations} realizations.")


    ### SSI Drought Metrics
    ssi_window = 24
    node = 'delMontague' 
    drought_calculator = SSIDroughtMetrics()

    ssi_calculator = SSI(normal_scores_transform=False,
                         timescale=ssi_window)
    
    ssi_calculator.fit(Q_monthly.loc[:, node])

    ssi_obs = ssi_calculator.get_training_ssi()
    obs_droughts = drought_calculator.calculate_drought_metrics(ssi_obs)

    # Calculate SSI for each realization in the synthetic ensemble
    syn_ssi = pd.DataFrame(index=syn_ensemble[realization_ids[0]].resample('MS').sum().index,
                        columns=np.arange(0, n_realizations))
    syn_droughts = None
    
    for i in realization_ids:
        if i % 50 == 0:
            print(f"Calculating SSI for realization {i} of {n_realizations}...")
        
        Qsi = syn_ensemble[i].loc[:, node]
        Qsi_monthly = Qsi.resample('MS').sum()
        
        syn_ssi.loc[:,int(i)] = ssi_calculator.transform(Qsi_monthly)
        
        if syn_droughts is None:
            drought_chars = drought_calculator.calculate_drought_metrics(syn_ssi.loc[:,int(i)])
            print(f"First realization {i} drought characteristics: {drought_chars.columns.tolist()}")            
            drought_chars['realization_id'] = i
            syn_droughts = drought_chars.copy()
            
        else:
            drought_chars = drought_calculator.calculate_drought_metrics(syn_ssi.loc[:,int(i)])
            drought_chars['realization_id'] = i
            syn_droughts = pd.concat([syn_droughts, 
                                    drought_chars], axis=0)

        if drought_chars.empty:
                print(f"No drought events found for realization {i}.")
                continue
            
    # Put in a dict with node name as key 
    syn_ssi_dict = {node: syn_ssi}
    # For all pd.DataFrames, make sure columns labels are str
    for key, df in syn_ssi_dict.items():
        syn_ssi_dict[key].columns = df.columns.astype(str)
    
    ### Save SSI values to hdf5
    ssi_fname = f"./pywrdrb/drought_metrics/{ensemble_type}_ensemble_ssi{ssi_window}.hdf5"    
    print(f"Saving SSI values to hdf5: {ssi_fname}")
    hdf_manager = HDF5Manager()
    hdf_manager.export_ensemble_to_hdf5(dict = syn_ssi_dict,
                                         output_file = ssi_fname)
    
    # save drought metrics
    obs_droughts.reset_index(inplace=True, drop=True)
    obs_droughts.to_csv(f"./pywrdrb/drought_metrics/observed_drought_events.csv")

    syn_droughts.reset_index(inplace=True, drop=True)
    syn_droughts.to_csv(f"./pywrdrb/drought_metrics/{ensemble_type}_ensemble_drought_events.csv")

    print("Drought metrics saved successfully.")