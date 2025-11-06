import sys
import os
import numpy as np
import pandas as pd
from mpi4py import MPI
import warnings
warnings.filterwarnings("ignore")

import pywrdrb

from methods.config import *


def calculate_hashimoto_metrics_during_droughts(dataset_id, ssi_windows=[3, 6, 12]):
    """
    Calculate Hashimoto reliability metrics during drought events using MPI parallelization
    
    Parameters:
    -----------
    dataset_id : str
        Dataset identifier to process
    ssi_windows : list
        List of SSI window sizes to process
    """
    
    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Verify dataset
    verify_dataset_id(dataset_id)
    dataset_config = DATASET_CONFIGS[dataset_id]
    
    if rank == 0:
        print(f"Calculating Hashimoto metrics during droughts for: {dataset_id}")
        print(f"Dataset type: {dataset_config['type']}")
        print(f"SSI windows: {ssi_windows}")
        print(f"Using {size} MPI ranks")
    
    # Load ensemble data from processed HDF5 (all ranks need this)
    fname = f'./pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5'
    if not os.path.exists(fname):
        if rank == 0:
            print(f"ERROR: Postprocessed data not found: {fname}")
            print("Run postprocessing (04_postprocess_data.py) first!")
        return False
    
    data = pywrdrb.Data()
    data.load_from_export(fname)
    
    # Loop through different SSI windows
    for SSI_WINDOW in ssi_windows:
        
        if rank == 0:
            print(f"\nProcessing SSI window: {SSI_WINDOW} months")
        
        # Load ensemble drought event data
        syn_drought_fname = f"./pywrdrb/drought_metrics/{dataset_id}_ssi{SSI_WINDOW}_drought_events.csv"
        obs_drought_fname = f"./pywrdrb/drought_metrics/observed_ssi{SSI_WINDOW}_drought_events.csv"
        
        if not os.path.exists(syn_drought_fname):
            if rank == 0:
                print(f"  WARNING: Drought events file not found: {syn_drought_fname}")
                print(f"  Run 05_calculate_ssi_drought_metrics.py first for {dataset_id}!")
            continue
        
        synthetic_drought_events = pd.read_csv(syn_drought_fname)
        observed_drought_events = pd.read_csv(obs_drought_fname)
        
        # Initialize columns for shortage metrics (only on rank 0 initially)
        for col in ['max_shortage', 'total_shortage', 'max_duration', 'date_of_max_shortage']:
            synthetic_drought_events[f'{col}_delMontague'] = np.nan
            synthetic_drought_events[f'{col}_delTrenton'] = np.nan

        ### Calculate shortage metrics
        if rank == 0:
            print('  Calculating shortage metrics...')

        for node in ['delMontague', 'delTrenton']:
            
            if rank == 0:
                print(f"    Processing {node}...")
            
            # Get unique realizations
            realizations = synthetic_drought_events['realization_id'].unique()
            
            # Distribute realizations across ranks
            realizations_per_rank = len(realizations) // size
            extra_realizations = len(realizations) % size
            
            if rank < extra_realizations:
                start_idx = rank * (realizations_per_rank + 1)
                end_idx = start_idx + realizations_per_rank + 1
            else:
                start_idx = rank * realizations_per_rank + extra_realizations
                end_idx = start_idx + realizations_per_rank
            
            my_realizations = realizations[start_idx:end_idx]
            
            # Process assigned realizations
            local_results = []
            
            for i, r in enumerate(my_realizations):
                if rank == 0 and i % 20 == 0:
                    print(f"      Rank 0: {i}/{len(my_realizations)} realizations processed...")
                
                # Get drought events for this realization
                drought_events = synthetic_drought_events[synthetic_drought_events['realization_id'] == r]
                
                # Skip if no drought events
                if drought_events.empty:
                    continue
                
                # Loop through drought events
                for idx, drought_event in drought_events.iterrows():
                    # Get the start and end dates for the drought event
                    start_date = drought_event['start']
                    end_date = drought_event['end']
                    
                    # Get shortage timeseries from data object
                    shortage_series = data.shortage[dataset_id][r].loc[start_date:end_date, node]
                    
                    # Calculate shortage metrics
                    max_shortage = shortage_series.max()
                    total_shortage = shortage_series.sum()
                    
                    # Max duration of continuous shortage
                    shortage_bool = (shortage_series > 0).astype(int)
                    shortage_groups = shortage_bool.diff().ne(0).cumsum()
                    shortage_durations = shortage_bool.groupby(shortage_groups).cumsum()
                    max_duration = shortage_durations.max()
                    
                    # Date of max shortage
                    date_of_max = shortage_series.idxmax() if max_duration > 0 else None
                    
                    # Store results with index for later assignment
                    local_results.append({
                        'idx': idx,
                        f'max_shortage_{node}': max_shortage,
                        f'total_shortage_{node}': total_shortage,
                        f'max_duration_{node}': max_duration,
                        f'date_of_max_shortage_{node}': date_of_max
                    })
            
            # Gather results from all ranks
            all_results = comm.gather(local_results, root=0)
            
            # Combine results on rank 0
            if rank == 0:
                # Flatten results from all ranks
                combined_results = []
                for rank_results in all_results:
                    combined_results.extend(rank_results)
                
                # Update the dataframe with calculated metrics
                for result in combined_results:
                    idx = result['idx']
                    for key in [f'max_shortage_{node}', f'total_shortage_{node}', 
                               f'max_duration_{node}', f'date_of_max_shortage_{node}']:
                        synthetic_drought_events.loc[idx, key] = result[key]
        
        # Save the results (only rank 0)
        if rank == 0:
            output_fname = f"./pywrdrb/drought_metrics/{dataset_id}_ssi{SSI_WINDOW}_drought_events_with_shortage_metrics.csv"
            synthetic_drought_events.to_csv(output_fname, index=False)
            print(f"  Saved drought events with shortage metrics: {output_fname}")
    
    if rank == 0:
        print(f"\nAll SSI-shortage calculations completed for {dataset_id}!")
    
    return True


def main(dataset_id):
    """Main function"""
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("=" * 60)
        print(f"HASHIMOTO METRICS DURING DROUGHTS: {dataset_id}")
        print("=" * 60)
        
        # Verify dataset
        verify_dataset_id(dataset_id)
        dataset_config = DATASET_CONFIGS[dataset_id]
        
        print(f"Dataset type: {dataset_config['type']}")
        print(f"Description: {dataset_config['description']}")
        print("=" * 60)
    
    # Calculate Hashimoto metrics for default SSI windows
    success = calculate_hashimoto_metrics_during_droughts(dataset_id, ssi_windows=[12])
    
    if rank == 0:
        if success:
            print("=" * 60)
            print("Hashimoto metrics calculation completed successfully!")
        else:
            print("=" * 60)
            print("ERROR: Hashimoto metrics calculation failed!")
            sys.exit(1)


if __name__ == "__main__":
    
    # Get the dataset_id from command line arguments
    if len(sys.argv) != 2:
        print("Usage: mpirun -np N python 06_calculate_hashimoto_metrics_during_droughts.py <dataset_id>")
        print(f"Available datasets: {list(DATASET_CONFIGS.keys())}")
        sys.exit(1)
    
    dataset_id = sys.argv[1]
    verify_dataset_id(dataset_id)
    
    main(dataset_id)