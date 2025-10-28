#%% 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pywrdrb

# from sglib.plotting.drought import drought_metric_scatter_plot
from methods.plotting.uh_drought import drought_metric_scatter_plot, drought_metric_joint_plot


from methods.metrics.shortfall import get_shortfall_metrics, calculate_hashimoto_metrics
from config import RECONSTRUCTION_OUTPUT_FNAME, STATIONARY_ENSEMBLE_OUTPUT_FNAME

#%% Load data from export

fname = './pywrdrb/outputs/stationary_ensemble_with_postprocessing.hdf5'

data = pywrdrb.Data()
data.load_from_export(file=fname)

#%% Load drought events

obs_droughts = pd.read_csv("./pywrdrb/drought_metrics/observed_drought_events.csv")
syn_droughts = pd.read_csv("./pywrdrb/drought_metrics/synthetic_drought_events.csv")

# Drop syn_droughts with magnitude or duration -inf 
syn_droughts = syn_droughts[(syn_droughts['magnitude'] != -np.inf) & (syn_droughts['duration'] != -np.inf)].copy()

# For both drought data, use absolute values for magnitude and severity
for char in ['magnitude', 'severity']:
    obs_droughts[char] = obs_droughts[char].abs()
    syn_droughts[char] = syn_droughts[char].abs()

# Convert start dates to datetime
obs_droughts['start'] = pd.to_datetime(obs_droughts['start'])
syn_droughts['start'] = pd.to_datetime(syn_droughts['start'])

# Get start month
obs_droughts['start_month'] = obs_droughts['start'].dt.month
syn_droughts['start_month'] = syn_droughts['start'].dt.month


# # Keep only droughts after 1960
# obs_droughts = obs_droughts.loc[obs_droughts.start >= '1960-01-01'].copy()


obs_shortage = data.shortage['obs'][0]

# Remove obs_shortage before 1954
obs_shortage = obs_shortage.loc[obs_shortage.start >= '1955-01-01'].copy()

realizations = list(data.shortage['stationary_ensemble'].keys())
syn_shortage = pd.concat([data.shortage['stationary_ensemble'][i] for i in realizations], axis=0)
syn_shortage.reset_index(inplace=True, drop=True)

# Drop shortages with duration == 1
obs_shortage = obs_shortage[obs_shortage.duration > 1].copy()
syn_shortage = syn_shortage[syn_shortage.duration > 1].copy()


#%% Find shortages for each drought
# For each drought event, check if shortage at Montague occurred

syn_drought_shortages = syn_droughts.copy()
syn_drought_shortages['shortage_count'] = 0
syn_drought_shortages['shortage_vulnerability'] = 0.0
syn_drought_shortages['shortage_severity'] = 0.0
syn_drought_shortages['shortage_duration'] = 0

syn_drought_shortages['time_from_drought_start_till_shortage_start'] = np.inf


for row, drought_metrics in syn_droughts.iterrows():
    start = pd.to_datetime(drought_metrics['start'])
    end = pd.to_datetime(drought_metrics['end'])
    
    # Get shortages during this drought
    shortages = syn_shortage[(syn_shortage.start >= start) & (syn_shortage.end <= end)]
    
    syn_drought_shortages.loc[row, 'shortage_count'] = shortages.shape[0]

    # for severity, take the sum of the severity of all shortages
    syn_drought_shortages.loc[row, 'shortage_severity'] = shortages['severity'].sum() if not shortages.empty else 0.0
    syn_drought_shortages.loc[row, 'shortage_vulnerability'] = shortages['vulnerability'].max() if not shortages.empty else 0.0
    syn_drought_shortages.loc[row, 'shortage_duration'] = shortages['duration'].sum() if not shortages.empty else 0.0
    
    # get time between drought start and first shortage start in avg months
    if not shortages.empty:
        # Find start_ date of shortage with max duration within this shortages df
        max_shortage_idx = shortages['duration'].idxmax()
        syn_drought_shortages.loc[row, 'shortage_start'] = shortages.loc[max_shortage_idx, 'start']
        
        syn_drought_shortages.loc[row, 'time_from_drought_start_till_shortage_start'] = (syn_drought_shortages.loc[row, 'shortage_start'] - start).days / 30.0



#%% Make a plot of the frequency of drought start dates
"""
Creates a two panel plot with many elements:

Left panel: 
    KDE plot of the start month of droughts in the synthetic data.
    The KDE is horizontal, so the density is along the x axis and the start month is on the y. 
Right panel:
    Box plots of drought durations for all droughts within a specific month. 
    There are 12 box plots, one for each month of the year.
    Duration is measured in months also. 
    
The two panels share a common y-axis, which is the start month of the droughts.

"""

nonzero_syn_drought_shortages = syn_drought_shortages[syn_drought_shortages['shortage_duration'] > 7].copy()


fig, axs = plt.subplots(ncols=2, 
                       figsize=(12, 5),
                       sharey=True,
                       gridspec_kw={'width_ratios': [1, 1.5]})

# KDE plot of drought start month
ax = axs[0]

sns.kdeplot(data = syn_droughts,
            y='start_month',
            label='Synthetic',
            color='orange',
            clip=(1, 12),
            bw_adjust=0.9,
            fill=True,
            ax=ax)
ax.set_xlim(0.2, 0)


# Box plots of drought duration by start month
ax = axs[1]

sns.boxplot(syn_droughts,
            x='duration',
            y='start_month',
            orient='h', 
            fliersize=0,
            ax=ax)

sns.stripplot(nonzero_syn_drought_shortages,
             x='time_from_drought_start_till_shortage_start',
             y='start_month',
             orient='h',
             ax=ax)

ax.set_xlim(0, 60)

plt.show()
