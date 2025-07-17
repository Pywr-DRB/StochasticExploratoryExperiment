import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sglib.utils.load import HDF5Manager

file_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = f"{file_dir}/../data"

def load_drb_reconstruction(gage_flow=True):
    """
    Load the DRB reconstruction data.

    Returns:
        pd.DataFrame: DataFrame containing the DRB reconstruction data.
    """
    if gage_flow:
        fname = 'gage_flow_obs_pub_nhmv10_BC_ObsScaled_median.csv'
    else:
        fname = 'catchment_inflow_obs_pub_nhmv10_BC_ObsScaled_median.csv'
    
    Q = pd.read_csv(f'{data_dir}/{fname}')
    Q.drop(columns=['datetime'], inplace=True)  # Drop the first column if it's an index
    
    datetime = pd.date_range(start='1945-01-01', 
                             periods=Q.shape[0], 
                             freq='D')
    
    Q.index = datetime
    # Q = Q.replace(0, np.nan)  # Replace zeros with NaN
    # Q = Q.dropna(axis=1, how='any')
    
    return Q


def load_and_combine_ensemble_sets(ensemble_sets, 
                                   by_site = True):
    """
    Load and combine all ensemble set data into a single dictionary.
    
    WARNING: 
    This should only be used when the realizations do NOT matter. 
    In this function, all realizations are combined and renumbered 
    without regard to their original set IDs.
    
    Parameters:
    - ensemble_sets: List of ensemble set specifications.
    
    Returns:
    - Combined dict.
    """
    all_data = {}
    realization_id = 0
    for i, set_spec in enumerate(ensemble_sets):
        gageflow_set_file = set_spec.files['gage_flow']
        
        hdf_manager = HDF5Manager()
        ensemble_set_data = hdf_manager.load_ensemble(gageflow_set_file)
        
        if by_site:
            # extract just the data by site
            Qs_gageflow = ensemble_set_data.data_by_site
            
            # add to all_data
            for site in Qs_gageflow:
                if site not in all_data:
                    all_data[site] = Qs_gageflow[site].copy()
                else:
                    # If site already exists, append the new data
                    all_data[site] = pd.concat([all_data[site], Qs_gageflow[site]], axis=1)
                    
                # reset columns to be realization integers 0, ... N
                all_data[site].columns = np.arange(0, all_data[site].shape[1])
        
        else:
            # extract just the data by realization
            Qs_gageflow = ensemble_set_data.data_by_realization
            
            # add to all_data
            for real in Qs_gageflow:
                all_data[realization_id] = Qs_gageflow[real]
                realization_id += 1
    
    return all_data
