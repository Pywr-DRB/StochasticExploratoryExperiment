import numpy as np
import pandas as pd

from sglib.utils.load import HDF5Manager
from pywrdrb.pre.flows import _subtract_upstream_catchment_inflows

from methods.moea_generator import MOEAKirschNowakGenerator
from methods.load import load_drb_reconstruction
from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if __name__ == "__main__":
        
    # Borg output file name
    output_fname = f"./syn_nfe{50000}_seed{711}.csv"
    
    # Load source flow data
    if rank == 0:
        print("Rank 0: Loading DRB reconstruction data...")
        Q = load_drb_reconstruction()
        Q.replace(0, np.nan, inplace=True)
        Q.dropna(axis=1, how='any', inplace=True)
    else:
        Q = None
    # Synchronize the data across all processes
    Q = comm.bcast(Q, root=0)
    
    # Initialize the MOEA generator
    if rank == 0:
        print("Rank 0: Initializing MOEA generator...")
        generator = MOEAKirschNowakGenerator(Q=Q,
                                            debug=True)
        print("Rank 0: Preprocessing data generator...") 
        generator.preprocessing()
        
        print("Rank 0: Fitting generator...")
        generator.fit()
        
        print("Rank 0: Generator fitted. Loading Borg output...")
        borg_output = generator.load_borg_output(output_fname, nobjs=4, nconstr=1)
        generator.borg_output = borg_output
        
        print("Rank 0: Converting Borg output to M matrix...")
        M_matrix = generator.convert_borg_output_to_M_array()
        n_realizations = M_matrix.shape[0]
        n_years = M_matrix.shape[1] - 1
    else:
        generator = None
        M_matrix = None
        n_realizations = None
        n_years = None
        
        
    if rank == 0:
        print('Distributing generator and M_matrix to all processes...')
    # Synchronize the generator across all processes
    generator = comm.bcast(generator, root=0)
    M_matrix = comm.bcast(M_matrix, root=0)
    n_realizations = comm.bcast(n_realizations, root=0)
    n_years = comm.bcast(n_years, root=0)

    if rank == 0:
        print("Borg output loaded and generator initialized.")

    # split realizations across processes
    n_realizations_per_process = n_realizations // size
    start_index = rank * n_realizations_per_process
    end_index = start_index + n_realizations_per_process
    if rank == size - 1:  # Last process takes the remainder
        end_index = n_realizations
        
    M_matrix_rank = M_matrix[start_index:end_index, :, :]
    if rank == 0:
        print(f"Generating synthetic ensemble with {n_realizations} realizations, "
              f"each with {n_years} years of data ({n_years * 12} months)")
        
            
    # Generate synthetic ensemble
    Qse_monthly_local = {}
    Qse_daily_local = {}
    for i in range(M_matrix_rank.shape[0]):
        print(f"Rank {rank}: Generating realization {i} of {M_matrix_rank.shape[0]}...")
        
        Qse_monthly_local[i] = generator.generate_single_series(n_years=n_years,
                                                                M=M_matrix_rank[i],
                                                                as_array=False)
        
        # disaggregate to daily
        Qse_daily_local[i] = generator.nowak_disaggregator.disaggregate_monthly_flows(
            Qs_monthly = Qse_monthly_local[i]
            )
        
    # Synchronize the generated series across all processes
    Qse_daily_list = comm.gather(Qse_daily_local, root=0)
    Qse_daily = {}
    idx = 1
    for local_dict in Qse_daily_list:
        for _, value in local_dict.items():
            Qse_daily[idx] = value
            idx += 1

    if rank == 0:
        print(f"Gathered synthetic monthly flows from all processes.")
        print(f"\nNumber of realizations: {len(Qse_daily)}")
        print(f"\nQse_daily type: {type(Qse_daily)}")    

    syn_ensemble = Qse_daily    
    
    n_realizations = len(syn_ensemble)

    if rank == 0:
        print(f"Completed generation of synthetic ensemble with:\n")
        print(f"  - {n_realizations} realizations")
        print(f"  - {syn_ensemble[0].shape[0]} months of data per realization")
        print(f"  - {syn_ensemble[0].shape[1]} sites per realization")
    
    
    ### Postprocessing
    ### Create marginal catchment inflows by subtracting upstream inflows
    inflow_ensemble = {}
    for real in syn_ensemble:
        syn_ensemble[real]['delTrenton'] = 0.0
        
        flows_i = syn_ensemble[real].copy()
        
        # change the datetime index to be 2000-01-01 to 2010-01-01
        flows_i.index = pd.date_range(start='1970-01-01', 
                                      periods=len(flows_i), freq='D')
        
        inflow_ensemble[real] = _subtract_upstream_catchment_inflows(flows_i)
    
    # rearrange so that the node name is the dict key and realizations 
    # are the columns of the pd.DataFrame    
    Q_syn = {}
    Qs_inflows = {}
    syn_datetime = inflow_ensemble[0].index
    for site in inflow_ensemble[0].columns:
        Q_syn[site] = np.zeros((len(syn_datetime), n_realizations),
                    dtype=float)
        Qs_inflows[site] = np.zeros((len(syn_datetime), n_realizations),
                        dtype=float)
        
        
        for i in range(n_realizations):
            Q_syn[site][:, i] = syn_ensemble[i][site].values 
            Qs_inflows[site][:, i] = inflow_ensemble[i][site].values
                
        # Convert to DataFrame
        Q_syn[site] = pd.DataFrame(Q_syn[site], 
                                index=syn_datetime, 
                                columns=[str(i) for i in range(n_realizations)])
        Qs_inflows[site] = pd.DataFrame(Qs_inflows[site],
                                        index=syn_datetime, 
                                        columns=[str(i) for i in range(n_realizations)])

    ### Save
    hdf_manager = HDF5Manager()

    fname = "./pywrdrb/inputs/borg_stationary_ensemble/gage_flow_mgd.hdf5"
    hdf_manager.export_ensemble_to_hdf5(Q_syn, fname)
    
    fname = "./pywrdrb/inputs/borg_stationary_ensemble/catchment_inflow_mgd.hdf5"
    hdf_manager.export_ensemble_to_hdf5(Qs_inflows, fname)