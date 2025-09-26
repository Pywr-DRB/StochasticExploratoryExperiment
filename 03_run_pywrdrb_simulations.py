#!/usr/bin/env python3
"""
Run Pywr-DRB simulations for all ensemble sets in parallel using MPI rank distribution.
Automatically distributes ensemble sets across available MPI ranks.
"""

import os
import sys
import re
import glob
import math
import numpy as np
import pandas as pd
from mpi4py import MPI

import warnings
warnings.filterwarnings("ignore")

import pywrdrb
from pywrdrb.utils.hdf5 import get_hdf5_realization_numbers, combine_batched_hdf5_outputs

from methods.utils import get_parameter_subset_to_export
from config import *

def run_ensemble_set_simulations(set_id, dataset_id):
    """
    Run Pywr-DRB simulations for a single ensemble set
    
    Parameters:
    -----------
    set_id : int
        Ensemble set identifier (0-indexed)
    dataset_id : str
        Dataset identifier (e.g., 'stationary_ensemble', 'climate_adjusted_ssp245_min')
    """
    
    # Get MPI info for this function call
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Get ensemble set specification
    set_spec = get_ensemble_set_spec(set_id, dataset_id)
    catchment_inflow_file = set_spec.files['catchment_inflow']
    ensemble_dir = set_spec.directory
    output_file = set_spec.output_file
    
    print(f"Rank {rank}: Running Pywr-DRB simulations for {dataset_id} ensemble set {set_id + 1}")
    print(f"  Input file: {catchment_inflow_file}")
    print(f"  Output file: {output_file}")
    
    # Check if input file exists
    if not os.path.exists(catchment_inflow_file):
        print(f"Error: Input file not found: {catchment_inflow_file}")
        return False
    
    # Setup pathnavigator for this specific ensemble set
    pn_config = pywrdrb.get_pn_config()
    pn_config[f"flows/{dataset_id}_set{set_id + 1}"] = os.path.abspath(ensemble_dir)
    pywrdrb.load_pn_config(pn_config)
    
    try:
        # Clear old batched output files if they exist
        if rank == 0:
            batch_pattern = f"{os.path.dirname(output_file)}/{dataset_id}_set{set_id + 1}_rank*_batch*.hdf5"
            model_pattern = f"{os.path.dirname(output_file)}/../models/{dataset_id}_set{set_id + 1}_rank*_batch*.json"
            
            for pattern in [batch_pattern, model_pattern]:
                old_files = glob.glob(pattern)
                for file in old_files:
                    if os.path.exists(file):
                        os.remove(file)
        
        comm.Barrier()  # Wait for cleanup
        
        # Get realization IDs for this ensemble set
        if rank == 0:
            realization_ids = get_hdf5_realization_numbers(catchment_inflow_file)
            print(f"Set {set_id + 1}: Found {len(realization_ids)} realizations")
        else:
            realization_ids = None

        # VALIDATION: Check if realization_ids match expected global IDs
        expected_realization_ids = set_spec.realizations
        if rank == 0:
            if type(realization_ids[0]) is str:
                expected_realization_ids = [str(r) for r in expected_realization_ids]

            if set(realization_ids) != set(expected_realization_ids):
                print(f"WARNING: Set {set_id + 1} - HDF5 realization IDs don't match expected global IDs")
                print(f"  Expected: {expected_realization_ids}")
                print(f"  Found in HDF5: {realization_ids}")
        
        # Broadcast realization IDs
        realization_ids = comm.bcast(realization_ids, root=0)
        
        # Optimized distribution: balance load across ranks
        # Use numpy array_split for better load balancing
        rank_realization_ids = np.array_split(realization_ids, size)[rank]
        rank_realization_ids = list(rank_realization_ids)
        n_rank_realizations = len(rank_realization_ids)
        
        print(f"Set {set_id + 1}, Rank {rank}: Processing {n_rank_realizations} realizations")
        
        if n_rank_realizations == 0:
            print(f"Set {set_id + 1}, Rank {rank}: No realizations assigned")
            return True
        
        # Split rank realizations into batches for memory management
        n_batches = math.ceil(n_rank_realizations / N_REALIZATIONS_PER_PYWRDRB_BATCH)
        batched_indices = {}
        
        for i in range(n_batches):
            batch_start = i * N_REALIZATIONS_PER_PYWRDRB_BATCH
            batch_end = min((i + 1) * N_REALIZATIONS_PER_PYWRDRB_BATCH, n_rank_realizations)
            batched_indices[i] = rank_realization_ids[batch_start:batch_end]
        
        print(f"Set {set_id + 1}, Rank {rank}: Running {n_batches} batches")
        
        # Run individual batches
        batch_filenames = []
        for batch, indices in batched_indices.items():
            print(f"Set {set_id + 1}, Rank {rank}: Running sim batch {batch + 1}/{n_batches} with {len(indices)} realizations")
            
            # Model options for this batch
            model_options = {
                "inflow_ensemble_indices": indices,
            }
            
            # Add salinity LSTM options if enabled
            if SALINITY_LSTM_PREDICTIONS:
                model_options.update(SALINITY_LSTM_OPTIONS)
            
            # Build model
            mb = pywrdrb.ModelBuilder(
                inflow_type=f'{dataset_id}_set{set_id + 1}',
                start_date=START_DATE,
                end_date=END_DATE,
                options=model_options,
            )
            
            # Save model
            model_fname = f"{os.path.dirname(output_file)}/../models/{dataset_id}_set{set_id + 1}_rank{rank}_batch{batch}.json"
            mb.make_model()
            mb.write_model(model_fname)
            
            # Load model
            model = pywrdrb.Model.load(model_fname)
            
            # Get list of parameters for specific results sets
            all_parameter_names = [p.name for p in model.parameters if p.name]
            subset_parameter_names = get_parameter_subset_to_export(
                all_parameter_names, 
                results_set_subset=SAVE_RESULTS_SETS
            )
            export_parameters = [p for p in model.parameters if p.name in subset_parameter_names]
            
            # Setup output recorder
            batch_output_filename = f"{os.path.dirname(output_file)}/{dataset_id}_set{set_id + 1}_rank{rank}_batch{batch}.hdf5"
            recorder = pywrdrb.OutputRecorder(
                model=model,
                output_filename=batch_output_filename,
                parameters=export_parameters
            )
            
            # Run simulation
            model.run()
            
            batch_filenames.append(batch_output_filename)
            
            # Clean up model object to free memory (optimization)
            del model
        
        # Wait for all ranks to complete their batches
        comm.Barrier()
        
        # Combine all batched outputs for this ensemble set (rank 0 only)
        if rank == 0:
            print(f'Set {set_id + 1}: Combining batched outputs...')
            
            # Find all batch files for this set
            batch_pattern = f"{os.path.dirname(output_file)}/{dataset_id}_set{set_id + 1}_rank*_batch*.hdf5"
            all_batch_files = glob.glob(batch_pattern)
            
            if not all_batch_files:
                print(f"Set {set_id + 1}: No batch files found!")
                return False
            
            # IMPORTANT! Sort batch files by (rank, batch) to preserve realization order
            def sort_batch_files(filename):
                match = re.search(r'rank(\d+)_batch(\d+)', filename)
                if match:
                    rank_num = int(match.group(1))
                    batch_num = int(match.group(2))
                    return (rank_num, batch_num)
                return (0, 0)
            
            all_batch_files.sort(key=sort_batch_files)
            
            # If output file already exists, remove it
            if os.path.exists(output_file):
                print(f"Set {set_id + 1}: Removing existing output file: {output_file}")
                os.remove(output_file)
            
            print(f"Set {set_id + 1}: Found {len(all_batch_files)} batch files to combine")
            
            # Combine batch files
            combine_batched_hdf5_outputs(all_batch_files, output_file)
            
            # Cleanup batch files if requested
            if WorkflowFlags.CLEANUP_PYWRDRB_BATCH_FILES:
                for file in all_batch_files:
                    if os.path.exists(file):
                        os.remove(file)
                
                # Also cleanup model files
                model_pattern = f"{os.path.dirname(output_file)}/../models/{dataset_id}_set{set_id + 1}_rank*_batch*.json"
                model_files = glob.glob(model_pattern)
                for file in model_files:
                    if os.path.exists(file):
                        os.remove(file)
                
                print(f"Set {set_id + 1}: Cleaned up {len(all_batch_files)} batch files")
            
            print(f"Set {set_id + 1}: Simulations completed successfully!")
            print(f"  Output file: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"Error processing set {set_id + 1}: {str(e)}")
        return False


def parallel_run_all_sets(dataset_id):
    """
    Distribute Pywr-DRB simulations across available MPI ranks
    
    Parameters:
    -----------
    dataset_id : str
        Dataset identifier to simulate
    """
    
    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Verify dataset
    verify_dataset_id(dataset_id)
    dataset_config = DATASET_CONFIGS[dataset_id]
    
    if rank == 0:
        print("=" * 60)
        print(f"PARALLEL PYWRDRB SIMULATIONS: {dataset_id}")
        print("=" * 60)
        print(f"Dataset type: {dataset_config['type']}")
        print(f"Description: {dataset_config['description']}")
        print(f"Total ensemble sets: {N_ENSEMBLE_SETS}")
        print(f"Realizations per set: {N_REALIZATIONS_PER_ENSEMBLE_SET}")
        print(f"Pywr-DRB batch size: {N_REALIZATIONS_PER_PYWRDRB_BATCH}")
        print(f"Available MPI ranks: {size}")
        print(f"Simulation period: {START_DATE} to {END_DATE}")
        
        # Check which sets are ready for processing
        ready_sets = []
        for set_id in range(N_ENSEMBLE_SETS):
            set_spec = get_ensemble_set_spec(set_id, dataset_id)
            if os.path.exists(set_spec.files['catchment_inflow']):
                ready_sets.append(set_id)
        
        print(f"Ready ensemble sets: {len(ready_sets)}")
        
        if len(ready_sets) < N_ENSEMBLE_SETS:
            missing_sets = set(range(N_ENSEMBLE_SETS)) - set(ready_sets)
            print(f"Warning: Missing ensemble sets: {sorted(missing_sets)}")
            print("Run ensemble generation and prep first!")
        
        # Calculate optimal rank distribution
        if size >= N_ENSEMBLE_SETS:
            ranks_per_set = size // N_ENSEMBLE_SETS
            print(f"Ranks per ensemble set: {ranks_per_set}")
        else:
            print(f"Sets per rank: {N_ENSEMBLE_SETS // size + 1}")
        print("=" * 60)
    
    comm.Barrier()  # Wait for status messages
    
    # Track success/failure
    success_count = 0
    total_processed = 0
    
    if size >= N_ENSEMBLE_SETS:
        # More ranks than sets - distribute ranks across sets
        ranks_per_set = size // N_ENSEMBLE_SETS
        set_id = rank // ranks_per_set
        
        # Only process if we're within valid set range
        if set_id < N_ENSEMBLE_SETS:
            # Create sub-communicator for this ensemble set
            color = set_id
            local_comm = comm.Split(color, rank)
            
            # Store original communicator
            original_comm = MPI.COMM_WORLD
            
            # Temporarily replace global communicator for the simulation function
            MPI.COMM_WORLD = local_comm
            
            try:
                success = run_ensemble_set_simulations(set_id, dataset_id)
                total_processed = 1
                success_count = 1 if success else 0
            finally:
                # Restore original communicator
                MPI.COMM_WORLD = original_comm
                local_comm.Free()
    else:
        # More sets than ranks - each rank processes multiple sets sequentially
        sets_per_rank = N_ENSEMBLE_SETS // size
        extra_sets = N_ENSEMBLE_SETS % size
        
        if rank < extra_sets:
            my_sets = list(range(rank * (sets_per_rank + 1), (rank + 1) * (sets_per_rank + 1)))
        else:
            start = extra_sets * (sets_per_rank + 1) + (rank - extra_sets) * sets_per_rank
            my_sets = list(range(start, start + sets_per_rank))
        
        # Process each set sequentially on this rank
        for set_id in my_sets:
            # Create single-rank communicator
            local_comm = MPI.COMM_SELF
            original_comm = MPI.COMM_WORLD
            MPI.COMM_WORLD = local_comm
            
            try:
                success = run_ensemble_set_simulations(set_id, dataset_id)
                total_processed += 1
                success_count += 1 if success else 0
            finally:
                MPI.COMM_WORLD = original_comm
    
    # Collect results from all ranks
    comm.Barrier()
    
    # Sum up success counts across all ranks
    total_success = comm.reduce(success_count, op=MPI.SUM, root=0)
    total_attempts = comm.reduce(total_processed, op=MPI.SUM, root=0)
    
    if rank == 0:
        print("\n" + "=" * 60)
        print(f"PYWRDRB SIMULATIONS COMPLETED: {dataset_id}")
        print("=" * 60)
        print(f"Successfully processed: {total_success}/{total_attempts} sets")
        
        if total_success == N_ENSEMBLE_SETS:
            print("SUCCESS: All ensemble sets simulated successfully!")
        else:
            failed_count = N_ENSEMBLE_SETS - total_success
            print(f"WARNING: {failed_count} sets failed or were skipped")
            
            # Try to identify which sets might have failed
            failed_sets = []
            for set_id in range(N_ENSEMBLE_SETS):
                set_spec = get_ensemble_set_spec(set_id, dataset_id)
                if not os.path.exists(set_spec.output_file):
                    failed_sets.append(set_id + 1)
            
            if failed_sets:
                print(f"  Failed sets: {failed_sets}")
        
        print("=" * 60)
        print(f"Done with Pywr-DRB simulations for {dataset_id}!")


def verify_simulation_outputs(dataset_id):
    """
    Verify that all ensemble sets have been properly simulated
    
    Parameters:
    -----------
    dataset_id : str
        Dataset identifier to verify
    """
    
    verify_dataset_id(dataset_id)
    print(f"\nVerifying Pywr-DRB simulation outputs for {dataset_id}...")
    
    all_completed = True
    successful_sets = []
    failed_sets = []
    
    for set_id in range(N_ENSEMBLE_SETS):
        set_spec = get_ensemble_set_spec(set_id, dataset_id)
        
        if not os.path.exists(set_spec.output_file):
            print(f"FAIL:  Set {set_id + 1}: Output file not found")
            all_completed = False
            failed_sets.append(set_id + 1)
            continue
        
        # Check file size (basic validation)
        file_size = os.path.getsize(set_spec.output_file)
        if file_size < 1024:  # Less than 1KB is suspicious
            print(f"FAIL:  Set {set_id + 1}: Output file too small ({file_size} bytes)")
            all_completed = False
            failed_sets.append(set_id + 1)
            continue
        
        # Try to load with Pywr-DRB to verify format (optimization: only check first and last)
        if set_id == 0 or set_id == N_ENSEMBLE_SETS - 1:
            try:
                test_data = pywrdrb.Data(results_sets=["major_flow"])
                test_data.load_output(output_filenames=[set_spec.output_file])
                n_realizations = len(list(test_data.major_flow.values())[0])
                
                if n_realizations != N_REALIZATIONS_PER_ENSEMBLE_SET:
                    print(f"WARNING: Set {set_id + 1}: Expected {N_REALIZATIONS_PER_ENSEMBLE_SET} realizations, found {n_realizations}")
                    failed_sets.append(set_id + 1)
                else:
                    print(f"SUCCESS: Set {set_id + 1}: Valid output ({n_realizations} realizations, {file_size//1024//1024} MB)")
                    successful_sets.append(set_id + 1)
                    
            except Exception as e:
                print(f"FAIL:  Set {set_id + 1}: Error loading output file - {str(e)}")
                all_completed = False
                failed_sets.append(set_id + 1)
        else:
            # For middle sets, just check file size
            print(f"SUCCESS: Set {set_id + 1}: Output exists ({file_size//1024//1024} MB)")
            successful_sets.append(set_id + 1)
    
    if all_completed:
        print(f"SUCCESS: All {N_ENSEMBLE_SETS} ensemble sets have valid simulation outputs!")
    else:
        print(f"WARNING: {len(failed_sets)} sets have invalid outputs: {failed_sets}")
    
    return all_completed


def main(dataset_id):
    """Main function"""
    
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print(f"Starting Pywr-DRB simulations for {dataset_id}...")
        print_experiment_summary(dataset_id)
    
    # Run all ensemble set simulations in parallel
    parallel_run_all_sets(dataset_id)
    
    # Verify outputs (only on rank 0)
    if rank == 0:
        verify_simulation_outputs(dataset_id)
        print(f"\nPywr-DRB simulation workflow completed for {dataset_id}!")


if __name__ == "__main__":
    
    # Get the dataset_id from command line arguments
    if len(sys.argv) != 2:
        print("Usage: python 03_run_pywrdrb_simulations.py <dataset_id>")
        print(f"Available datasets: {list(DATASET_CONFIGS.keys())}")
        sys.exit(1)
    
    dataset_id = sys.argv[1]
    verify_dataset_id(dataset_id)
    
    main(dataset_id)