#%%
import sys
import numpy as np
import pandas as pd
import scipy.stats as stats

from pywrdrb.pre.flows import _subtract_upstream_catchment_inflows
from pywrdrb.pywr_drb_node_data import immediate_downstream_nodes_dict
from pywrdrb.pywr_drb_node_data import downstream_node_lags

from sglib.methods.nonparametric.kirsch_nowak import KirschNowakGenerator
from sglib.utils.load import HDF5Manager

from methods.load import load_drb_reconstruction
from config import pywrdrb_nodes_to_generate, pywrdrb_nodes_to_regress


#%% Loading data ##################################
Q = load_drb_reconstruction(gage_flow=True)
Q_inflow = load_drb_reconstruction(gage_flow=False)
Q_all =  Q.copy()

Q = Q.loc[:, pywrdrb_nodes_to_generate]
Q.drop(columns=['delTrenton'], inplace=True) # delTrenton doesn't get inflow (it goes to delDRCanal)
print(f"Loaded streamflow data.")

#%% Generation  ##################################
# Initialize the generator
kn_gen = KirschNowakGenerator(Q, debug=False)

# Preprocess the data
kn_gen.preprocessing()

# Fit the model
print("Fitting the model...")
kn_gen.fit()

#%%
# M=None
# n_years = 50
# n_realizations = 10

# n_years_buffered = n_years + 1
# if not kn_gen.fitted:
#     raise RuntimeError("Call preprocessing() and fit() before generate().")

# if M is None:
#     M = kn_gen._get_bootstrap_indices(n_years_buffered, max_idx=kn_gen.Y.shape[0])
# else:
#     M = np.asarray(M)
#     if M.shape != (n_years_buffered, kn_gen.n_months):
#         raise ValueError(f"M must have shape ({n_years_buffered}, {kn_gen.n_months})")

# M_prime = M[:kn_gen.Y_prime.shape[0], :]

# X = kn_gen._create_bootstrap_tensor(M, use_Y_prime=False)
# X_prime = kn_gen._create_bootstrap_tensor(M_prime, use_Y_prime=True)

# Z = np.zeros_like(X)
# Z_prime = np.zeros_like(X_prime)

# for s in range(kn_gen.n_sites):
#     Z[:, :, s] = X[:, :, s] @ kn_gen.U_site[s]
#     Z_prime[:, :, s] = X_prime[:, :, s] @ kn_gen.U_prime_site[s]


# ZC = kn_gen._combine_Z_and_Z_prime(Z, Z_prime)
# Q_syn = kn_gen._destandardize_flows(ZC)


# Q_syn = np.exp(Q_syn)

# Q_flat = kn_gen._reshape_output(Q_syn)

# synthetic_index = kn_gen._get_synthetic_index(n_years)
# Qs = pd.DataFrame(Q_flat, columns=kn_gen.site_names, index=synthetic_index)


# Qs_monthly = Qs.groupby([Qs.index.year, Qs.index.month]).sum()
# Qs_monthly.index = pd.MultiIndex.from_tuples(Qs_monthly.index, names=['year', 'month'])
# Qsm = Qs_monthly
# Qsm = np.log(Qsm.clip(lower=1e-6))  # Avoid log(0) issues


# s_mean_month = Qsm.groupby(level='month').mean()
#%%
# Generate 10 years
print("Generating synthetic ensemble...")
n_years = 50
n_realizations = 10
syn_ensemble = kn_gen.generate(n_realizations=n_realizations,
                                n_years=n_years, 
                                as_array=False)


### Handle non-major nodes
## Fit KDEs to the fraction of downstream flow
# For each pair (upstream, downstream) fit a KDE
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
print("Generating synthetic flows at non-major nodes...")

# samples = number of days x number of realizations
n_samples = syn_ensemble[0].shape[0] * n_realizations
realization_ids = list(syn_ensemble.keys())
for upstream in pywrdrb_nodes_to_generate:
    downstream = immediate_downstream_nodes_dict[upstream]
    if downstream not in pywrdrb_nodes_to_regress:
        continue
    
    kde_name = f"{upstream}_to_{downstream}"
    kde = kdes[kde_name]
    
    # Samples are fraction of upstream flow
    # downstream_flow = upstream_flow * fraction
    samples = kde.resample(n_samples)
    samples = samples.reshape((syn_ensemble[0].shape[0], n_realizations))
    
    print(f"\n### Flow faction summary for {kde_name}:")
    print(f"Max flow fraction: {np.max(samples)}")
    print(f"Min flow fraction: {np.min(samples)}")
    print(f"Median flow fraction: {np.median(samples)}")
    
    # clip samples to be between 0 and 1
    samples = np.clip(samples, 0, 1)
    
    for realization in realization_ids:
        # Get the upstream flow for this realization
        upstream_flow = syn_ensemble[realization][upstream].values
        
        # Calculate the marginal downstream inflow
        downstream_inflow = upstream_flow * samples[:, realization_ids.index(realization)]
        
        # calculate total flow at donwstream node, so we need to add upstream flow
        # accounting for lag
        lag = downstream_node_lags[downstream]
        if lag > 0:
            downstream_gage_flow = downstream_inflow
            downstream_gage_flow[lag:] += upstream_flow[:-lag]
            downstream_gage_flow[:lag] += upstream_flow[:lag]
        else:
            downstream_gage_flow = downstream_inflow + upstream_flow
            
        # store in ensemble
        syn_ensemble[realization][downstream] = downstream_gage_flow
    

### Postprocessing
## Create marginal catchment inflows by subtracting upstream inflows
inflow_ensemble = {}
for real in syn_ensemble:
    syn_ensemble[real]['delTrenton'] = 0.0
    
    flows_i = syn_ensemble[real].copy()
    
    # change the datetime index to be 2000-01-01 to 2010-01-01
    flows_i.index = pd.date_range(start='1970-01-01', 
                                    periods=len(flows_i), 
                                    freq='D')
    
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

fname = "./pywrdrb/inputs/stationary_ensemble/gage_flow_mgd.hdf5"
hdf_manager.export_ensemble_to_hdf5(Q_syn, fname)

fname = "./pywrdrb/inputs/stationary_ensemble/catchment_inflow_mgd.hdf5"
hdf_manager.export_ensemble_to_hdf5(Qs_inflows, fname)

print("Done with stochastic generation of stationary ensemble.")