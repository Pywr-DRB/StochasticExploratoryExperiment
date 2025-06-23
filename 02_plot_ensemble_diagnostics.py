#%%
import numpy as np
import pandas as pd


from sglib.plotting.monthly_flow_statistics import plot_validation
from sglib.plotting.plot import plot_autocorrelation, plot_fdc_ranges, plot_flow_ranges
from sglib.plotting.plot import plot_correlation
from sglib.plotting.drought import drought_metric_scatter_plot
from sglib.droughts.ssi import SSIDroughtMetrics, SSI
from sglib.utils.load import HDF5Manager

from methods.load import load_drb_reconstruction
from config import gage_flow_ensemble_fname, catchment_inflow_ensemble_fname
from config import FIG_DIR, pywrdrb_nodes, pywrdrb_nodes_to_generate, pywrdrb_nodes_to_regress


#%%
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



#%% Plot statistical validation
for site in pywrdrb_nodes:

    if site == 'delTrenton':
        continue
    
    logscale = False
    
    fname = f"{site}_log.png" if logscale else f"{site}.png"
    fname = f"{FIG_DIR}/statistical_validation/{fname}"
    
    plot_validation(H_df=Q.loc[:, [site]], 
                    S_df=Q_syn[site].loc[:'2019-12-31', :],
                    scale='monthly',
                    logspace=logscale,
                    fname=fname,
                    sitename=site)
                            


#%%

print(f"Loaded synthetic ensemble with {n_realizations} realizations for {len(Q_syn)} sites.")

### SSI Drought Metrics
drought_calculator = SSIDroughtMetrics()

ssi_calculator = SSI(normal_scores_transform=False)
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

# save drought metrics
obs_droughts.reset_index(inplace=True, drop=True)
obs_droughts.to_csv(f"./pywrdrb/drought_metrics/observed_drought_events.csv")

syn_droughts.reset_index(inplace=True, drop=True)
syn_droughts.to_csv(f"./pywrdrb/drought_metrics/synthetic_drought_events.csv")


## Plot scatter of drought metrics
fname = f"delMontague_stationary_drought_metrics_scatter.png"
fname = f"{FIG_DIR}/drought_metrics/{fname}"

drought_metric_scatter_plot(obs_droughts, 
                            syn_drought_metrics=syn_droughts, 
                            x_char='severity', 
                            y_char='magnitude', 
                            color_char='duration',
                            fname=fname)


#%% Plotting

## Spatial correlation of flows

Qs_df = syn_ensemble[realization_ids[0]].drop(columns=['delTrenton', 'datetime'])
Qs_df = Qs_df.loc[:, Q.columns]

fname = f"gage_flow_ensemble_syn.png"
fname = f"{FIG_DIR}/spatial_correlation/{fname}"
plot_correlation(Q, Qs_df,
                 savefig=True,
                 fname=fname)



#%%
for plot_site in ['cannonsville', '01425000', 'delLordville', 'delMontague', 'delDRCanal']:

    fname = f"{site}_gage_flow.png"
    fname = f"{FIG_DIR}/autocorrelation/{fname}"
    plot_autocorrelation(Q.loc[:, plot_site], 
                        Q_syn[plot_site], 
                        lag_range=np.arange(1,60, 5), timestep='daily',
                        savefig=True,
                        fname=fname)


    plot_flow_ranges(Q.loc[:,plot_site], 
                    Q_syn[plot_site], 
                    timestep='monthly',
                    y_scale='linear',
                    savefig=True,
                    fname=f"{FIG_DIR}/gage_flow_ranges_{plot_site}_syn.png",)
    
    plot_flow_ranges(np.log(Q.loc[:,plot_site]), 
                    np.log(Q_syn[plot_site]), 
                    timestep='monthly',
                    y_scale='linear',
                    savefig=True,
                    fname=f"{FIG_DIR}/gage_log_flow_ranges_{plot_site}_syn.png",)
