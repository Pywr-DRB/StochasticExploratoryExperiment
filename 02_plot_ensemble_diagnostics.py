import numpy as np
import pandas as pd

from pywrdrb.utils.hdf5 import extract_realization_from_hdf5
from pywrdrb.utils.hdf5 import get_hdf5_realization_numbers

from sglib.plotting.plot import plot_autocorrelation, plot_fdc_ranges, plot_flow_ranges
from sglib.plotting.plot import plot_correlation
from sglib.plotting.drought import drought_metric_scatter_plot
from sglib.droughts.ssi import SSIDroughtMetrics, SSI
from sglib.utils.load import HDF5Manager

from methods.load import load_drb_reconstruction
from config import gage_flow_ensemble_fname, catchment_inflow_ensemble_fname
from config import FIG_DIR


gage_flow_ensemble_fname = "./pywrdrb/inputs/borg_stationary_ensemble/gage_flow_mgd.hdf5"
catchment_inflow_ensemble_fname = "./pywrdrb/inputs/borg_stationary_ensemble/catchment_inflow_mgd.hdf5"


### Loading data
## Historic reconstruction data
# Total flow
Q = load_drb_reconstruction()
Q.replace(0, np.nan, inplace=True)
Q.drop(columns=['delTrenton'], inplace=True)  # Remove Trenton gage as it is not used in the ensemble

# Catchment inflows
Q_inflows = load_drb_reconstruction(gage_flow=False)
Q_inflows.replace(0, np.nan, inplace=True)
Q_inflows.drop(columns=['delTrenton'], inplace=True)  # Remove Trenton gage as it is not used in the ensemble
Q_monthly = Q.resample('MS').sum()

print(f"Loaded reconstruction data with {Q.shape[0]// 365} years of daily data for {Q.shape[1]} sites.")



## Synthetic ensemble
hdf_manager = HDF5Manager()
Qs_gageflow = hdf_manager.load_ensemble(gage_flow_ensemble_fname)
Q_syn = Qs_gageflow.data_by_site
syn_ensemble = Qs_gageflow.data_by_realization

realization_ids = Qs_gageflow.realization_ids
n_realizations = len(realization_ids)


# Qs_inflows = hdf_manager.load_ensemble(catchment_inflow_ensemble_fname)
# Qs_inflows = Qs_inflows.data_by_site
# inflow_ensemble = Qs_inflows.data_by_realization


print(f"Loaded synthetic ensemble with {n_realizations} realizations for {len(Q_syn)} sites.")

### SSI Drought Metrics
drought_calculator = SSIDroughtMetrics()

ssi_calculator = SSI()
ssi_calculator.fit(Q_monthly.loc[:,'delMontague'])

ssi_obs = ssi_calculator.get_training_ssi()
obs_droughts = drought_calculator.calculate_drought_metrics(ssi_obs)

# Calculate SSI for each realization in the synthetic ensemble
syn_ssi = pd.DataFrame(index=syn_ensemble[realization_ids[0]].resample('MS').sum().index,
                       columns=np.arange(0, n_realizations))
syn_droughts = None

for i in realization_ids:
    if i % 10 == 0:
        print(f"Calculating SSI for realization {i} of {n_realizations}...")
    
    Qsi = syn_ensemble[i].loc[:, 'delMontague']
    Qsi_monthly = Qsi.resample('MS').sum()
    
    syn_ssi.loc[:,int(i)] = ssi_calculator.transform(Qsi_monthly)
    
    if syn_droughts is None:
        drought_chars = drought_calculator.calculate_drought_metrics(syn_ssi.loc[:,int(i)])
        print(f"First realization {i} drought characteristics: {drought_chars.columns.tolist()}")
        
        if drought_chars.empty:
            print(f"No drought events found for realization {i}.")
            continue
        
        syn_droughts = drought_chars.copy()
    else:
        drought_chars = drought_calculator.calculate_drought_metrics(syn_ssi.loc[:,int(i)])
        
        if drought_chars.empty:
            print(f"No drought events found for realization {i}.")
            continue
        
        syn_droughts = pd.concat([syn_droughts, 
                                  drought_chars], axis=0)

print(f"Calculated SSI drought metrics for {n_realizations} realizations.")
print(f"Observed drought metrics: {obs_droughts.shape[0]} events")
print(f"Synthetic drought metrics: {syn_droughts.shape[0]} events")
print(f"syn_droughts columns: {syn_droughts.columns.tolist()}")

drought_metric_scatter_plot(obs_droughts, 
                            syn_drought_metrics=syn_droughts, 
                            x_char='severity', y_char='magnitude', color_char='duration',
                            fname=f"{FIG_DIR}/borg_stationary_drought_metrics_scatter.png")



# print(f"Historic flow columns: {Q.columns.tolist()}")
# print(f"Ensemble flow columns: {syn_ensemble[realization_ids[0]].columns.tolist()}")


# ### Plotting
# plot_correlation(Q, syn_ensemble[realization_ids[0]],
#                  savefig=True,
#                  fname=f"{FIG_DIR}gage_correlation_syn.png")

# plot_correlation(Q_inflows, inflow_ensemble[realization_ids[0]].loc[:, Q_inflows.columns],
#                     savefig=True,
#                     fname=f"{FIG_DIR}inflow_correlation_syn.png")

# for plot_site in ['delLordville', 'delMontague', 'delDRCanal']:

#     plot_autocorrelation(Q.loc[:, plot_site], 
#                         Q_syn[plot_site], 
#                         lag_range=np.arange(1,60, 5), timestep='daily',
#                         savefig=True,
#                         fname=f"{FIG_DIR}gage_autocorr_{plot_site}_syn.png",)

#     plot_flow_ranges(Q.loc[:,plot_site], 
#                     Q_syn[plot_site], 
#                     timestep='daily',
#                     savefig=True,
#                     fname=f"{FIG_DIR}gage_flow_ranges_{plot_site}_syn.png",)

#     plot_fdc_ranges(Q_inflows.loc[:,plot_site], 
#                     Qs_inflows[plot_site],
#                     savefig=True,
#                     fname=f"{FIG_DIR}inflow_fdc_{plot_site}.png",)

#     plot_fdc_ranges(Q.loc[:,plot_site],
#                     Q_syn[plot_site],
#                     savefig=True,
#                     fname=f"{FIG_DIR}gage_flow_fdc_{plot_site}_syn.png",)