import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sglib.droughts.ssi import SSIDroughtMetrics, SSI

from methods.load import load_drb_reconstruction, load_and_combine_ensemble_sets
from config import STATIONARY_ENSEMBLE_SETS, CLIMATE_ADJUSTED_ENSEMBLE_SETS
from config import verify_ensemble_type

if __name__ == "__main__":

    # Get ensemble type from command line arguments
    ensemble_type = sys.argv[1]
    verify_ensemble_type(ensemble_type)

    # Use appropriate list of ensemble set specs
    if ensemble_type == 'stationary':
        ensemble_set_specs = STATIONARY_ENSEMBLE_SETS
    elif ensemble_type == 'climate_adjusted':
        ensemble_set_specs = CLIMATE_ADJUSTED_ENSEMBLE_SETS
    else:
        raise ValueError(f"Invalid ensemble type: {ensemble_type}. Must be 'stationary' or 'climate_adjusted'.")
        
    ### Loading data
    ## Historic reconstruction data
    # Total flow
    Q = load_drb_reconstruction()
    Q.replace(0, np.nan, inplace=True)
    Q.drop(columns=['delTrenton'], inplace=True)  # Remove Trenton gage as it is not used in the ensemble
    Q_monthly = Q.resample('MS').sum()

    # Catchment inflows
    Q_inflows = load_drb_reconstruction(gage_flow=False)
    Q_inflows.replace(0, np.nan, inplace=True)
    Q_inflows.drop(columns=['delTrenton'], inplace=True)  # Remove Trenton gage as it is not used in the ensemble

    print(f"Loaded reconstruction data with {Q.shape[0]// 365} years of daily data for {Q.shape[1]} sites.")
    

    ## Synthetic ensemble
    syn_ensemble = load_and_combine_ensemble_sets(ensemble_set_specs, by_site=False)

    realization_ids = list(syn_ensemble.keys())
    n_realizations = len(realization_ids)

    print(f"Loaded synthetic {ensemble_type} ensemble with {n_realizations} realizations.")


    ### SSI Drought Metrics
    drought_calculator = SSIDroughtMetrics()

    ssi_calculator = SSI(normal_scores_transform=False)
    ssi_calculator.fit(Q_monthly.loc[:,'delMontague'])

    ssi_obs = ssi_calculator.get_training_ssi()
    obs_droughts = drought_calculator.calculate_drought_metrics(ssi_obs)

    # Calculate SSI for each realization in the synthetic ensemble
    syn_ssi = pd.DataFrame(index=syn_ensemble[realization_ids[0]].resample('MS').sum().index,
                        columns=np.arange(0, n_realizations))
    syn_droughts = None

    for i in realization_ids:
        if i % 50 == 0:
            print(f"Calculating SSI for realization {i} of {n_realizations}...")
        
        Qsi = syn_ensemble[i].loc[:, 'delMontague']
        Qsi_monthly = Qsi.resample('MS').sum()
        
        syn_ssi.loc[:,int(i)] = ssi_calculator.transform(Qsi_monthly)
        
        if syn_droughts is None:
            drought_chars = drought_calculator.calculate_drought_metrics(syn_ssi.loc[:,int(i)])
            print(f"First realization {i} drought characteristics: {drought_chars.columns.tolist()}")
                    
            syn_droughts = drought_chars.copy()
        else:
            drought_chars = drought_calculator.calculate_drought_metrics(syn_ssi.loc[:,int(i)])
            
            syn_droughts = pd.concat([syn_droughts, 
                                    drought_chars], axis=0)

        if drought_chars.empty:
                print(f"No drought events found for realization {i}.")
                continue
            

    # save drought metrics
    obs_droughts.reset_index(inplace=True, drop=True)
    obs_droughts.to_csv(f"./pywrdrb/drought_metrics/observed_drought_events.csv")

    syn_droughts.reset_index(inplace=True, drop=True)
    syn_droughts.to_csv(f"./pywrdrb/drought_metrics/{ensemble_type}_ensemble_drought_events.csv")

    print("Drought metrics saved successfully.")