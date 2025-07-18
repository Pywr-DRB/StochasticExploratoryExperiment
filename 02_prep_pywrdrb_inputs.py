"""
Prepare Pywr-DRB inputs for all ensemble sets in parallel using MPI rank distribution.
Automatically distributes ensemble sets across available MPI ranks.
"""

import os
import sys
from mpi4py import MPI
import warnings
warnings.filterwarnings("ignore")

import pywrdrb
from pywrdrb.utils.hdf5 import get_hdf5_realization_numbers
from pywrdrb.pre import PredictedInflowEnsemblePreprocessor

from config import *

def prep_ensemble_set(set_id, ensemble_type):
    """
    Prepare Pywr-DRB inputs for a single ensemble set
    
    Parameters:
    -----------
    set_id : int
        Ensemble set identifier (0-indexed)
    """
    
    # Get MPI info for this function call
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Get ensemble set specification
    set_spec = get_ensemble_set_spec(set_id, ensemble_type)
    catchment_inflow_file = set_spec.files['catchment_inflow']
    ensemble_dir = set_spec.directory
    
    if rank == 0:
        print(f"Set {set_id+1} Rank {rank}: Preparing inputs for ensemble set {set_id + 1}")
        print(f"  Input file: {catchment_inflow_file}")
        print(f"  Ensemble directory: {ensemble_dir}")
    
    # Check if input file exists
    if not os.path.exists(catchment_inflow_file):
        print(f"Error: Input file not found: {catchment_inflow_file}")
        return False
    
    # Setup pathnavigator for this specific ensemble set
    pn_config = pywrdrb.get_pn_config()
    pn_config[f"flows/{ensemble_type}_ensemble_set{set_id + 1}"] = os.path.abspath(ensemble_dir)
    pywrdrb.load_pn_config(pn_config)
    
    try:
        if rank == 0:
            # Get realization numbers from the HDF5 file
            realization_ids = get_hdf5_realization_numbers(catchment_inflow_file)
            print(f"Set {set_id + 1}: Found {len(realization_ids)} realizations")
    
        else:
            realization_ids = None
        realization_ids = comm.bcast(realization_ids, root=0)
        
        
        # Initialize the preprocessor
        preprocessor = PredictedInflowEnsemblePreprocessor(
            flow_type=f"{ensemble_type}_ensemble_set{set_id + 1}",
            ensemble_hdf5_file=catchment_inflow_file,
            realization_ids=realization_ids,  
            start_date=None,  # Use full range
            end_date=None,
            modes=('regression_disagg',),
            use_log=True,
            remove_zeros=False,
            use_const=False,
            use_mpi=True  # Enable MPI within the preprocessor
        )
        
        # Process the data
        if rank == 0:
            print(f"Set {set_id + 1}: Loading and predicting future inflows...")
        
        preprocessor.load()
        preprocessor.process()        
        preprocessor.save()
        
        if rank == 0:
            print(f"Set {set_id + 1}: Pywr-DRB inputs prepared successfully.")
        
        return True
        
    except Exception as e:
        print(f"Error processing set {set_id + 1}: {str(e)}")
        return False


def parallel_prep_all_sets(ensemble_type):
    """
    Distribute Pywr-DRB input preparation across available MPI ranks
    """
    
    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    
    assert(size >= N_ENSEMBLE_SETS), \
        f"Error: More ensemble sets ({N_ENSEMBLE_SETS}) than available MPI ranks ({size}). " \
        "Increase the number of ranks or reduce the number of ensemble sets."
    
    if rank == 0:
        print("=" * 60)
        print("PARALLEL PYWRDRB INPUT PREPARATION")
        print("=" * 60)
        print(f"Total ensemble sets: {N_ENSEMBLE_SETS}")
        print(f"Available MPI ranks: {size}")
        
        # Check which sets exist
        existing_sets = get_existing_ensemble_sets(ensemble_type)
        print(f"Found {len(existing_sets)} existing ensemble sets")
        
        if len(existing_sets) < N_ENSEMBLE_SETS:
            missing_sets = set(range(N_ENSEMBLE_SETS)) - set(existing_sets)
            print(f"Warning: Missing ensemble sets: {sorted(missing_sets)}")
            print("Run ensemble generation first!")
        
        # Calculate optimal rank distribution
        if size >= N_ENSEMBLE_SETS:
            ranks_per_set = size // N_ENSEMBLE_SETS
            print(f"Ranks per ensemble set: {ranks_per_set}")
        else:
            print(f"More sets than ranks - will process sets sequentially")
        print("=" * 60)
    
    comm.Barrier()  # Wait for status messages
    
    # Track success/failure
    success_count = 0
    total_processed = 0
    
    # Strategy 1: More ranks than sets (preferred for large nodes)
    ranks_per_set = size // N_ENSEMBLE_SETS
    set_id = rank // ranks_per_set
    
    # Only process if we're within valid set range
    if set_id < N_ENSEMBLE_SETS:
        # Create sub-communicator for this ensemble set
        color = set_id
        local_comm = comm.Split(color, rank)
        
        # Store original communicator
        original_comm = MPI.COMM_WORLD
        
        # Temporarily replace global communicator for the prep function
        MPI.COMM_WORLD = local_comm
        
        try:
            success = prep_ensemble_set(set_id, ensemble_type)
            total_processed = 1
            success_count = 1 if success else 0
        finally:
            # Restore original communicator
            MPI.COMM_WORLD = original_comm
            local_comm.Free()
    
    
    # Collect results from all ranks
    comm.Barrier()
    
    # Sum up success counts across all ranks
    total_success = comm.reduce(success_count, op=MPI.SUM, root=0)
    total_attempts = comm.reduce(total_processed, op=MPI.SUM, root=0)
    
    if rank == 0:
        print("\n" + "=" * 60)
        print("PYWRDRB INPUT PREPARATION COMPLETED")
        print("=" * 60)
        print(f"Successfully processed: {total_success}/{total_attempts} sets")
        
        if total_success == total_attempts:
            print("SUCCESS: All ensemble sets prepared successfully!")
        else:
            failed_count = total_attempts - total_success
            print(f"WARNING: Warning: {failed_count} ranks failed or were skipped")
            
            # Try to identify which sets might have failed
            # by checking for expected output files
            failed_sets = []
            for set_id in range(N_ENSEMBLE_SETS):
                set_spec = get_ensemble_set_spec(set_id, ensemble_type)
                # Check if preprocessed files exist (this is a rough check)
                f = set_spec.files['predicted_inflow']
                if not os.path.exists(f):
                    failed_sets.append(set_id + 1)
                
            if failed_sets:
                print(f"  Potentially failed sets: {failed_sets}")
        
        print("=" * 60)


def verify_prep_outputs(ensemble_type):
    """
    Verify that all ensemble sets have been properly prepared
    """
    
    print("\nVerifying Pywr-DRB input preparation...")
    
    all_prepared = True
    
    for set_id in range(N_ENSEMBLE_SETS):
        set_spec = get_ensemble_set_spec(set_id, ensemble_type)
        
        fname = set_spec.files['predicted_inflow']
        if not os.path.exists(fname):
            print(f"FAIL:  Set {set_id + 1}: Predicted inflow file {fname} not found")
            all_prepared = False
            continue
        
    if all_prepared:
        print("SUCCESS: All ensemble sets properly prepared!")
    else:
        print("WARNING: Some ensemble sets may not be properly prepared")
    
    return all_prepared


def main(ensemble_type):
    """Main function"""
    
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("Starting Pywr-DRB input preparation for all ensemble sets...")
        print_experiment_summary(ensemble_type)
    
    # Prepare all ensemble sets in parallel
    parallel_prep_all_sets(ensemble_type)
    
    # Verify outputs (only on rank 0)
    if rank == 0:
        verify_prep_outputs(ensemble_type)
        print("\nPywr-DRB input preparation workflow completed!")


if __name__ == "__main__":

    # Get the ensemble_type from command line arguments
    ensemble_type = sys.argv[1]
    verify_ensemble_type(ensemble_type)
    
    main(ensemble_type)