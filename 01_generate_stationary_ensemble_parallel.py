#%%
import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
from mpi4py import MPI

from pywrdrb.pre.flows import _subtract_upstream_catchment_inflows
from pywrdrb.pywr_drb_node_data import immediate_downstream_nodes_dict
from pywrdrb.pywr_drb_node_data import downstream_node_lags

from sglib.methods.nonparametric.kirsch_nowak import KirschNowakGenerator
from sglib.utils.load import HDF5Manager

from methods.load import load_drb_reconstruction
from config import pywrdrb_nodes_to_generate, pywrdrb_nodes_to_regress
from config import N_REALIZATIONS, N_YEARS
from config import START_DATE

# Two arguments are expected:
# 1. The number of realizations to generate (default is N_REALIZATIONS)
# 2. The output filepath with name
assert len(sys.argv) == 3, "Usage: python generate_stationary_ensemble_parallel.py <N_REALIZATIONS> <output_filepath>" 
N_REALIZATIONS = int(sys.argv[1])
output_filepath = sys.argv[2]


# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Calculate realizations per rank
realizations_per_rank = N_REALIZATIONS // size
remaining_realizations = N_REALIZATIONS % size

# Distribute realizations across ranks
if rank < remaining_realizations:
    n_realizations_local = realizations_per_rank + 1
    start_idx = rank * n_realizations_local
else:
    n_realizations_local = realizations_per_rank
    start_idx = rank * realizations_per_rank + remaining_realizations

end_idx = start_idx + n_realizations_local

if rank == 0:
    print(f"Running on {size} ranks")
    print(f"Total realizations: {N_REALIZATIONS}")
    print(f"Realizations per rank: {[realizations_per_rank + (1 if i < remaining_realizations else 0) for i in range(size)]}")

#%% Loading data ##################################
Q = load_drb_reconstruction(gage_flow=True)
Q_inflow = load_drb_reconstruction(gage_flow=False)
Q_all = Q.copy()

Q = Q.loc[:, pywrdrb_nodes_to_generate]
if rank == 0:
    print(f"Loaded streamflow data.")

#%% Generation  ##################################
# Initialize the generator (all ranks need this for the same model)
kn_gen = KirschNowakGenerator(Q, debug=False)

# Preprocess the data
kn_gen.preprocessing()

# Fit the model
if rank == 0:
    print("Fitting the model...")
kn_gen.fit()

#%%
# Generate local realizations
if rank == 0:
    print("Generating synthetic ensemble...")
    
n_years = N_YEARS
syn_ensemble_local = kn_gen.generate(n_realizations=n_realizations_local,
                                    n_years=n_years, 
                                    as_array=False)

# Renumber local realizations to global indices
syn_ensemble_local_renumbered = {}
for i, local_key in enumerate(syn_ensemble_local.keys()):
    global_key = start_idx + i
    syn_ensemble_local_renumbered[global_key] = syn_ensemble_local[local_key]

syn_ensemble_local = syn_ensemble_local_renumbered

### Handle non-major nodes
## Fit KDEs to the fraction of downstream flow (all ranks do this)
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

## Use KDE samples to generate synthetics at non-major nodes
if rank == 0:
    print("Generating synthetic flows at non-major nodes...")

n_samples_local = syn_ensemble_local[start_idx].shape[0] * n_realizations_local
realization_ids_local = list(syn_ensemble_local.keys())

for upstream in pywrdrb_nodes_to_generate:
    downstream = immediate_downstream_nodes_dict[upstream]
    if downstream not in pywrdrb_nodes_to_regress:
        continue
    
    kde_name = f"{upstream}_to_{downstream}"
    kde = kdes[kde_name]
    
    # Generate samples for local realizations
    samples_local = kde.resample(n_samples_local)
    samples_local = samples_local.reshape((syn_ensemble_local[start_idx].shape[0], n_realizations_local))
    
    # Clip samples to be between 0 and 1
    samples_local = np.clip(samples_local, 0, 1)
    
    for i, realization in enumerate(realization_ids_local):
        # Get the upstream flow for this realization
        upstream_flow = syn_ensemble_local[realization][upstream].values
        
        # Calculate the marginal downstream inflow
        downstream_inflow = upstream_flow * samples_local[:, i]
        
        # Calculate total flow at downstream node, accounting for lag
        lag = downstream_node_lags[downstream]
        if lag > 0:
            downstream_gage_flow = downstream_inflow
            downstream_gage_flow[lag:] += upstream_flow[:-lag]
            downstream_gage_flow[:lag] += upstream_flow[:lag]
        else:
            downstream_gage_flow = downstream_inflow + upstream_flow
            
        # Store in ensemble
        syn_ensemble_local[realization][downstream] = downstream_gage_flow

### Postprocessing
## Create marginal catchment inflows by subtracting upstream inflows
inflow_ensemble_local = {}
for real in syn_ensemble_local:
    syn_ensemble_local[real]['delTrenton'] = 0.0
    
    flows_i = syn_ensemble_local[real].copy()
    
    # Change the datetime index
    flows_i.index = pd.date_range(start=START_DATE, 
                                    periods=len(flows_i), 
                                    freq='D')
    flows_i.index.name = 'datetime'
    
    inflow_ensemble_local[real] = _subtract_upstream_catchment_inflows(flows_i)

### Gather results from all ranks
if rank == 0:
    print("Gathering results from all ranks...")

# Gather all local ensembles to rank 0
all_syn_ensembles = comm.gather(syn_ensemble_local, root=0)
all_inflow_ensembles = comm.gather(inflow_ensemble_local, root=0)

# Only rank 0 processes the combined results
if rank == 0:
    # Combine all ensembles
    syn_ensemble = {}
    inflow_ensemble = {}
    
    for rank_ensembles in all_syn_ensembles:
        syn_ensemble.update(rank_ensembles)
    
    for rank_ensembles in all_inflow_ensembles:
        inflow_ensemble.update(rank_ensembles)
    
    print(f"Combined {len(syn_ensemble)} realizations from all ranks")
    
    # Rearrange so that the node name is the dict key and realizations 
    # are the columns of the pd.DataFrame    
    Q_syn = {}
    Qs_inflows = {}
    syn_datetime = inflow_ensemble[0].index
    
    for site in inflow_ensemble[0].columns:
        Q_syn[site] = np.zeros((len(syn_datetime), N_REALIZATIONS), dtype=float)
        Qs_inflows[site] = np.zeros((len(syn_datetime), N_REALIZATIONS), dtype=float)
        
        for i in range(N_REALIZATIONS):
            Q_syn[site][:, i] = syn_ensemble[i][site].values 
            Qs_inflows[site][:, i] = inflow_ensemble[i][site].values
                
        # Convert to DataFrame
        Q_syn[site] = pd.DataFrame(Q_syn[site], 
                                index=syn_datetime, 
                                columns=[str(i) for i in range(N_REALIZATIONS)])
        Q_syn[site].index.name = 'datetime'
        
        Qs_inflows[site] = pd.DataFrame(Qs_inflows[site],
                                        index=syn_datetime, 
                                        columns=[str(i) for i in range(N_REALIZATIONS)])
        Qs_inflows[site].index.name = 'datetime'


    ### Save
    hdf_manager = HDF5Manager()

    fname = "./pywrdrb/inputs/stationary_ensemble/gage_flow_mgd.hdf5"
    hdf_manager.export_ensemble_to_hdf5(Q_syn, fname)

    fname = "./pywrdrb/inputs/stationary_ensemble/catchment_inflow_mgd.hdf5"
    hdf_manager.export_ensemble_to_hdf5(Qs_inflows, fname)

    print("Done with stochastic generation of stationary ensemble.")

# Ensure all ranks wait for rank 0 to finish
comm.Barrier()

if rank == 0:
    print("All ranks completed successfully.")