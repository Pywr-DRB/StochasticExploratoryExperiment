import sys
import os
import numpy as np
import pandas as pd
from mpi4py import MPI
import warnings
warnings.filterwarnings("ignore")

import pywrdrb
from sglib import SSIDroughtMetrics, SSI
from sglib import Ensemble, HDF5Manager


from methods.load import load_drb_reconstruction
from config import *

EXPORT_SSI_HDF5 = False

def calculate_ssi_drought_metrics(dataset_id, ssi_windows=[3, 6, 12]):
    """
    Calculate SSI-based drought metrics for a dataset using MPI parallelization
    
    Parameters:
    -----------
    dataset_id : str
        Dataset identifier to process
    ssi_windows : list
        List of SSI window sizes (months) to calculate
    """
    
    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Verify dataset
    verify_dataset_id(dataset_id)
    dataset_config = DATASET_CONFIGS[dataset_id]
    
    if rank == 0:
        print(f"Calculating SSI drought metrics for: {dataset_id}")
        print(f"Dataset type: {dataset_config['type']}")
        print(f"SSI windows: {ssi_windows}")
        print(f"Using {size} MPI ranks")
    
    ### Loading data (all ranks load data - could optimize with broadcast if memory is tight)
    ## Historic reconstruction data
    Q = load_drb_reconstruction()
    Q.replace(0, np.nan, inplace=True)
    Q.drop(columns=['delTrenton'], inplace=True)
    Q_monthly = Q.resample('MS').sum()

    if rank == 0:
        print(f"Loaded reconstruction data with {Q.shape[0]// 365} years of daily data for {Q.shape[1]} sites.")

    ## Load synthetic ensemble
    fname = f'./pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5'
    if not os.path.exists(fname):
        if rank == 0:
            print(f"ERROR: Postprocessed data not found: {fname}")
            print("Run postprocessing (04_postprocess_data.py) first!")
        return False
    
    data = pywrdrb.Data()
    data.load_from_export(fname)
    
    # Keep just the combined ensemble dict
    syn_ensemble = data.gage_flow[dataset_id]
    realization_ids = list(syn_ensemble.keys())
    n_realizations = len(realization_ids)

    if rank == 0:
        print(f"Loaded synthetic {dataset_id} ensemble with {n_realizations} realizations.")

    ### SSI Drought Metrics
    for ssi_window in ssi_windows:
        
        if rank == 0:
            print(f"\nProcessing SSI window: {ssi_window} months")
        
        node = 'delMontague'
        
        # Initialize SSI calculator (all ranks need this)
        drought_calculator = SSIDroughtMetrics()
        ssi_calculator = SSI(normal_scores_transform=False, timescale=ssi_window)
        
        # Fit SSI model on historical data
        ssi_calculator.fit(Q_monthly.loc[:, node])
        
        # Calculate SSI for historical data (only rank 0)
        if rank == 0:
            ssi_obs = ssi_calculator.get_training_ssi()
            obs_droughts = drought_calculator.calculate_drought_metrics(ssi_obs)
        
        # Distribute realizations across ranks
        realizations_per_rank = len(realization_ids) // size
        extra_realizations = len(realization_ids) % size
        
        if rank < extra_realizations:
            start_idx = rank * (realizations_per_rank + 1)
            end_idx = start_idx + realizations_per_rank + 1
        else:
            start_idx = rank * realizations_per_rank + extra_realizations
            end_idx = start_idx + realizations_per_rank
        
        my_realizations = realization_ids[start_idx:end_idx]
        
        if rank == 0:
            print(f"  Rank 0 processing {len(my_realizations)} realizations")
        
        # Process assigned realizations
        local_ssi_data = {}
        local_drought_data = []
        
        for i, real_id in enumerate(my_realizations):
            if rank == 0 and i % 10 == 0:
                print(f"    Rank 0: {i}/{len(my_realizations)} realizations processed...")
            
            Qsi = syn_ensemble[real_id].loc[:, node]
            Qsi_monthly = Qsi.resample('MS').sum()
            
            # Calculate SSI
            ssi_values = ssi_calculator.transform(Qsi_monthly)
            local_ssi_data[str(real_id)] = ssi_values
            
            # Calculate drought metrics
            drought_chars = drought_calculator.calculate_drought_metrics(ssi_values)
            if not drought_chars.empty:
                drought_chars['realization_id'] = real_id
                local_drought_data.append(drought_chars)
        
        # Gather results from all ranks
        all_ssi_data = comm.gather(local_ssi_data, root=0)
        all_drought_data = comm.gather(local_drought_data, root=0)
        
        # Combine and save on rank 0
        if rank == 0:
            # Combine SSI data
            combined_ssi = {}
            for rank_data in all_ssi_data:
                combined_ssi.update(rank_data)
            
            # Create SSI DataFrame
            syn_ssi = pd.DataFrame(index=syn_ensemble[realization_ids[0]].resample('MS').sum().index,
                                  columns=np.arange(0, n_realizations))
            
            for real_id, ssi_values in combined_ssi.items():
                syn_ssi.loc[:, int(real_id)] = ssi_values
            
            # Put in dict with node name as key
            syn_ssi_dict = {node: syn_ssi}
            for key, df in syn_ssi_dict.items():
                syn_ssi_dict[key].columns = df.columns.astype(str)
            
            # Save SSI values to hdf5 (optional)
            if EXPORT_SSI_HDF5:
                ssi_fname = f"./pywrdrb/drought_metrics/{dataset_id}_ssi{ssi_window}.hdf5"
                print(f"  Saving SSI values to hdf5: {ssi_fname}")
                hdf_manager = HDF5Manager()
                hdf_manager.export_ensemble_to_hdf5(dict=syn_ssi_dict,
                                                    output_file=ssi_fname)
            
            # Combine drought data
            syn_droughts = pd.DataFrame()
            for rank_data in all_drought_data:
                for drought_df in rank_data:
                    syn_droughts = pd.concat([syn_droughts, drought_df], axis=0)
            
            # Save drought metrics
            obs_droughts.reset_index(inplace=True, drop=True)
            obs_fname = f"./pywrdrb/drought_metrics/observed_ssi{ssi_window}_drought_events.csv"
            obs_droughts.to_csv(obs_fname, index=False)
            print(f"  Saved observed drought metrics: {obs_fname}")
            
            syn_droughts.reset_index(inplace=True, drop=True)
            syn_fname = f"./pywrdrb/drought_metrics/{dataset_id}_ssi{ssi_window}_drought_events.csv"
            syn_droughts.to_csv(syn_fname, index=False)
            print(f"  Saved synthetic drought metrics: {syn_fname}")
    
    if rank == 0:
        print(f"\nAll drought metrics saved successfully for {dataset_id}!")
    
    return True


def main(dataset_id):
    """Main function"""
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("=" * 60)
        print(f"SSI DROUGHT METRICS CALCULATION: {dataset_id}")
        print("=" * 60)
        
        # Create output directory if it doesn't exist
        os.makedirs("./pywrdrb/drought_metrics", exist_ok=True)
    
    # Calculate drought metrics (using default SSI windows)
    success = calculate_ssi_drought_metrics(dataset_id, ssi_windows=[3, 6, 12])
    
    if rank == 0:
        if success:
            print("=" * 60)
            print("SSI drought metrics calculation completed successfully!")
        else:
            print("=" * 60)
            print("ERROR: SSI drought metrics calculation failed!")
            sys.exit(1)


if __name__ == "__main__":
    
    # Get the dataset_id from command line arguments
    if len(sys.argv) != 2:
        print("Usage: mpirun -np N python 05_calculate_ssi_drought_metrics.py <dataset_id>")
        print(f"Available datasets: {list(DATASET_CONFIGS.keys())}")
        sys.exit(1)
    
    dataset_id = sys.argv[1]
    verify_dataset_id(dataset_id)
    
    main(dataset_id)