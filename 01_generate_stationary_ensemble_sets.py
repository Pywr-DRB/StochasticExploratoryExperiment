"""
Generate all ensemble sets in parallel using MPI rank distribution.
Automatically distributes ensemble sets across available MPI ranks.
"""

import sys
import os
import numpy as np
import pandas as pd
import scipy.stats as stats
from mpi4py import MPI
import warnings
warnings.filterwarnings("ignore")

from pywrdrb.pre.flows import _subtract_upstream_catchment_inflows
from pywrdrb.pywr_drb_node_data import immediate_downstream_nodes_dict
from pywrdrb.pywr_drb_node_data import downstream_node_lags

from sglib.methods.nonparametric.kirsch_nowak import KirschNowakGenerator
from sglib.utils.load import HDF5Manager

from methods.load import load_drb_reconstruction
from config import *

def generate_ensemble_set(set_id, ensemble_type):
    """
    Generate a single ensemble set with proper MPI distribution
    
    Parameters:
    -----------
    set_id : int
        Ensemble set identifier (0-indexed)
    """
    
    assert ensemble_type in ensemble_type_opts, \
        f"Invalid ensemble_type: {ensemble_type}. Must be one of {ensemble_type_opts}"
        
    # Get MPI info for this function call (now using sub-communicator)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Get ensemble set specification
    set_spec = get_ensemble_set_spec(set_id, ensemble_type=ensemble_type)
    set_realization_ids = set_spec.realizations
    n_realizations = set_spec.n_realizations
    output_dir = set_spec.directory
    
    if rank == 0:
        print(f"Set {set_id + 1}: Generating ensemble with {size} ranks")
        print(f"  Total realizations: {n_realizations}")
        print(f"  Output directory: {output_dir}")
    
    # Ensure output directory exists
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
    comm.Barrier()
    
    # Load and broadcast data
    if rank == 0:
        Q = load_drb_reconstruction(gage_flow=True)
        Q_inflow = load_drb_reconstruction(gage_flow=False)
        Q_all = Q.copy()
        Q = Q.loc[:, pywrdrb_nodes_to_generate]
        print(f"Set {set_id + 1}: Loaded data for {Q.shape[1]} nodes, {Q.shape[0]} days")
    else:
        Q = None
        Q_inflow = None
        Q_all = None
    
    Q = comm.bcast(Q, root=0)
    Q_inflow = comm.bcast(Q_inflow, root=0)
    Q_all = comm.bcast(Q_all, root=0)
    
    # Fit KN generator and broadcast
    if rank == 0:
        print(f"Set {set_id + 1}: Fitting Kirsch-Nowak model...")

    kn_gen = KirschNowakGenerator(Q, debug=False)
    kn_gen.preprocessing()
    kn_gen.fit()
    
    # If ensemble_type is 'climate_adjusted', we need to adjust the monthly mean 
    if ensemble_type == 'climate_adjusted':
    
        if rank == 0:
            print(f'Set {set_id + 1}: Adjusting monthly means for climate change...')
        
        prior_mean_month = kn_gen.mean_month
        new_mean_month = prior_mean_month.copy() * pd.NA

        for i, site in enumerate(new_mean_month):
            new_mean_month.loc[:, site] = np.exp(prior_mean_month.loc[:, site]) * (1 + np.array(monthly_mean_flow_prc_change) / 100.0)

        # Convert back to log scale
        new_mean_month = np.log(new_mean_month.astype(float))
        
        # Pass back to generator, overwriting the prior means
        kn_gen.mean_month = new_mean_month
    
    # DISTRIBUTE REALIZATION GENERATION ACROSS RANKS
    # Each rank generates a subset of realizations
    realizations_per_rank = n_realizations // size
    extra_realizations = n_realizations % size
    
    # Calculate how many realizations this rank should generate
    if rank < extra_realizations:
        local_n_realizations = realizations_per_rank + 1
        local_start_idx = rank * (realizations_per_rank + 1)
    else:
        local_n_realizations = realizations_per_rank
        local_start_idx = rank * realizations_per_rank + extra_realizations
    
    if rank == 0:
        print(f"Set {set_id + 1}: Distributing realizations across {size} ranks")
        print(f"  Base realizations per rank: {realizations_per_rank}")
        print(f"  Extra realizations: {extra_realizations}")
    

    # Generate local ensemble subset
    if local_n_realizations > 0:
        local_syn_ensemble = kn_gen.generate(n_realizations=local_n_realizations,
                                           n_years=N_YEARS,
                                           as_array=False)
    else:
        local_syn_ensemble = {}
    
    # Fit and broadcast KDEs for non-major nodes
    if rank == 0:
        print(f"Set {set_id + 1}: Fitting KDEs for non-major nodes...")
        kdes = {}
        for upstream in pywrdrb_nodes_to_generate:
            downstream = immediate_downstream_nodes_dict[upstream]
            if downstream not in pywrdrb_nodes_to_regress:
                continue
            
            xs = Q_inflow.loc[:, upstream]
            ys = Q_inflow.loc[:, downstream]
            frac = ys / xs
            frac = frac[~np.isnan(frac)]
            
            kde_name = f"{upstream}_to_{downstream}"
            kdes[kde_name] = stats.gaussian_kde(frac)
    else:
        kdes = None
    
    kdes = comm.bcast(kdes, root=0)
    
    # Generate flows at non-major nodes for local ensemble
    if local_n_realizations > 0:
        if rank == 0:
            print(f"Set {set_id + 1}: Generating non-major node flows...")
        
        n_local_samples = local_syn_ensemble[0].shape[0] * local_n_realizations
        local_realization_ids = list(local_syn_ensemble.keys())
        
        for upstream in pywrdrb_nodes_to_generate:
            downstream = immediate_downstream_nodes_dict[upstream]
            if downstream not in pywrdrb_nodes_to_regress:
                continue
            
            kde_name = f"{upstream}_to_{downstream}"
            kde = kdes[kde_name]
            
            # Generate samples for local realizations
            samples = kde.resample(n_local_samples)
            samples = samples.reshape((local_syn_ensemble[0].shape[0], local_n_realizations))
            samples = np.clip(samples, 0, 1)
            
            for i, realization in enumerate(local_realization_ids):
                upstream_flow = local_syn_ensemble[realization][upstream].values
                downstream_inflow = upstream_flow * samples[:, i]
                
                # Account for lag
                lag = downstream_node_lags[downstream]
                if lag > 0:
                    downstream_gage_flow = downstream_inflow.copy()
                    downstream_gage_flow[lag:] += upstream_flow[:-lag]
                    downstream_gage_flow[:lag] += upstream_flow[:lag]
                else:
                    downstream_gage_flow = downstream_inflow + upstream_flow
                
                local_syn_ensemble[realization][downstream] = downstream_gage_flow
    
    # Create marginal catchment inflows for local ensemble
    if local_n_realizations > 0:
        if rank == 0:
            print(f"Set {set_id + 1}: Creating marginal catchment inflows...")
        
        local_inflow_ensemble = {}
        for real in local_syn_ensemble:
            local_syn_ensemble[real]['delTrenton'] = 0.0
            flows_i = local_syn_ensemble[real].copy()
            flows_i.index = pd.date_range(start=START_DATE, 
                                          periods=len(flows_i), 
                                          freq='D')
            
            local_inflow_ensemble[real] = _subtract_upstream_catchment_inflows(flows_i)
    else:
        local_inflow_ensemble = {}
    
    # GATHER ALL LOCAL ENSEMBLES TO RANK 0
    if rank == 0:
        print(f"Set {set_id + 1}: Gathering ensemble data from all ranks...")
    
    all_syn_ensembles = comm.gather(local_syn_ensemble, root=0)
    all_inflow_ensembles = comm.gather(local_inflow_ensemble, root=0)
    
    # COMBINE AND SAVE ON RANK 0 ONLY
    if rank == 0:
        print(f"Set {set_id + 1}: Combining and saving ensemble data...")
        
        # Combine all local ensembles
        combined_syn_ensemble = {}
        combined_inflow_ensemble = {}
        
        realization_counter = 0
        for rank_syn, rank_inflow in zip(all_syn_ensembles, all_inflow_ensembles):
            for local_real_id in rank_syn:
                # Assign global realization ID
                global_real_id = realization_counter
                combined_syn_ensemble[global_real_id] = rank_syn[local_real_id]
                combined_inflow_ensemble[global_real_id] = rank_inflow[local_real_id]
                realization_counter += 1
        
        # Verify we have the correct number of realizations
        if len(combined_syn_ensemble) != n_realizations:
            print(f"Error: Expected {n_realizations} realizations, got {len(combined_syn_ensemble)}")
            return False
        
        # Get datetime index from first realization
        syn_datetime = combined_inflow_ensemble[0].index
        
        # Reorganize data structure
        Q_syn = {}
        Qs_inflows = {}
        
        # Get sites from first realization
        sites = combined_inflow_ensemble[0].columns
        
        for site in sites:
            Q_syn[site] = np.zeros((len(syn_datetime), n_realizations), dtype=float)
            Qs_inflows[site] = np.zeros((len(syn_datetime), n_realizations), dtype=float)
            
            for i in range(n_realizations):
                Q_syn[site][:, i] = combined_syn_ensemble[i][site].values 
                Qs_inflows[site][:, i] = combined_inflow_ensemble[i][site].values
            
            
            # Convert to DataFrame with realization IDs
            # IMPORTANT: Use set-specific realization IDs
            real_cols = [str(i) for i in set_realization_ids]
            
            Q_syn[site] = pd.DataFrame(Q_syn[site], 
                                       index=syn_datetime, 
                                       columns=real_cols)
            Qs_inflows[site] = pd.DataFrame(Qs_inflows[site],
                                            index=syn_datetime, 
                                            columns=real_cols)
        
        # Save results
        print(f"Set {set_id + 1}: Saving results...")
        hdf_manager = HDF5Manager()
        
        gage_flow_fname = set_spec.files['gage_flow']
        catchment_inflow_fname = set_spec.files['catchment_inflow']
        
        hdf_manager.export_ensemble_to_hdf5(Q_syn, gage_flow_fname)
        hdf_manager.export_ensemble_to_hdf5(Qs_inflows, catchment_inflow_fname)
        
        print(f"Set {set_id + 1} completed successfully!")
        print(f"  Gage flow file: {gage_flow_fname}")
        print(f"  Catchment inflow file: {catchment_inflow_fname}")
        
        return True
    
    return True

def parallel_generate_all_sets(ensemble_type):
    """
    Distribute ensemble set generation across available MPI ranks
    """
        
    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Make sure more ranks than sets (preferred for large nodes)
    assert size > N_ENSEMBLE_SETS, \
        f"Requires more MPI ranks than ensemble sets. Got {size} ranks, {N_ENSEMBLE_SETS} sets."
    
    if rank == 0:
        print("=" * 60)
        print("PARALLEL ENSEMBLE SET GENERATION")
        print("=" * 60)
        print(f"Total ensemble sets: {N_ENSEMBLE_SETS}")
        print(f"Realizations per set: {N_REALIZATIONS_PER_ENSEMBLE_SET}")
        print(f"Available MPI ranks: {size}")
        print(f"Years per realization: {N_YEARS}")
        
        # Calculate optimal rank distribution
        if size >= N_ENSEMBLE_SETS:
            ranks_per_set = size // N_ENSEMBLE_SETS
            print(f"Ranks per ensemble set: {ranks_per_set}")
        else:
            print(f"More sets than ranks - will process sets sequentially")
        print("=" * 60)
    
    # Ensure all directories exist
    if rank == 0:
        ensure_ensemble_set_dirs()
    
    comm.Barrier()  # Wait for directories to be created
    
    ranks_per_set = size // N_ENSEMBLE_SETS
    set_id = rank // ranks_per_set
    
    # Only generate if we're within valid set range
    if set_id < N_ENSEMBLE_SETS:
        # Create sub-communicator for this ensemble set
        color = set_id
        local_comm = comm.Split(color, rank)
        
        # Store original communicator
        original_comm = MPI.COMM_WORLD
        
        # Temporarily replace global communicator for the generation function
        # (This is a bit hacky but allows reuse of existing generation code)
        MPI.COMM_WORLD = local_comm
        
        try:
            true_if_success = generate_ensemble_set(set_id, ensemble_type=ensemble_type)
            assert true_if_success, f"Set {set_id + 1} generation failed on rank {rank}"

        finally:
            # Restore original communicator
            MPI.COMM_WORLD = original_comm
            local_comm.Free()


    # Synchronize all ranks
    comm.Barrier()
    
    if rank == 0:
        print("\n" + "=" * 60)
        print("ALL ENSEMBLE SETS GENERATED SUCCESSFULLY!")
        print("=" * 60)
        
        # Verify all sets were created
        existing_sets = get_existing_ensemble_sets(ensemble_type=ensemble_type)
        if len(existing_sets) == N_ENSEMBLE_SETS:
            print(f"SUCCESS: All {N_ENSEMBLE_SETS} ensemble sets verified")
        else:
            print(f"WARNING: Only {len(existing_sets)}/{N_ENSEMBLE_SETS} sets found")
            missing = set(range(N_ENSEMBLE_SETS)) - set(existing_sets)
            print(f"  Missing sets: {sorted(missing)}")


def main(ensemble_type):
    """Main function"""
    
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        # Print configuration summary
        print_experiment_summary(ensemble_type=ensemble_type)
    
    # Generate all ensemble sets in parallel
    parallel_generate_all_sets(ensemble_type=ensemble_type)

    if rank == 0:
        print("\nEnsemble generation workflow completed!")


if __name__ == "__main__":
    
    # Get the ensemble_type from command line arguments
    ensemble_type = sys.argv[1]
    verify_ensemble_type(ensemble_type)
    
    
    main(ensemble_type=ensemble_type)