#%% 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pywrdrb

from sglib.plotting.drought import drought_metric_scatter_plot

from methods.metrics.shortfall import get_shortfall_metrics, calculate_hashimoto_metrics
from config import RECONSTRUCTION_OUTPUT_FNAME, STATIONARY_ENSEMBLE_OUTPUT_FNAME

#%% Load data from export

fname = './pywrdrb/outputs/stationary_ensemble_with_postprocessing.hdf5'

data = pywrdrb.Data()
data.load_from_export(file=fname)

#%% Load drought events

obs_droughts = pd.read_csv("./pywrdrb/drought_metrics/observed_drought_events.csv")
syn_droughts = pd.read_csv("./pywrdrb/drought_metrics/synthetic_drought_events.csv")

# Keep only droughts after 1960
obs_droughts = obs_droughts.loc[obs_droughts.start >= '1960-01-01'].copy()


#%% Plot shortage events

obs_shortage = data.shortage['obs'][0]

# Remove obs_shortage before 1954
obs_shortage = obs_shortage.loc[obs_shortage.start >= '1955-01-01'].copy()

realizations = list(data.shortage['stationary_ensemble'].keys())
syn_shortage = pd.concat([data.shortage['stationary_ensemble'][i] for i in realizations], axis=0)
syn_shortage.reset_index(inplace=True, drop=True)

# Drop shortages with duration == 1
obs_shortage = obs_shortage[obs_shortage.duration > 1].copy()
syn_shortage = syn_shortage[syn_shortage.duration > 1].copy()


drought_metric_scatter_plot(obs_shortage, 
                            syn_drought_metrics=syn_shortage, 
                            x_char='vulnerability', y_char='intensity', color_char='duration')

#%% Find shortages for each drought
# For each drought event, check if shortage at Montague occurred

syn_drought_shortages = syn_droughts.copy()
syn_drought_shortages['shortage_count'] = 0
syn_drought_shortages['shortage_vulnerability'] = 0.0
syn_drought_shortages['shortage_severity'] = 0.0
syn_drought_shortages['shortage_duration'] = 0.0


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
    
#%% Make a scatter plot of droughts, and color by shortage (binary)

fig, ax = plt.subplots(figsize=(8, 8))

# First plot droughts where shortage occurred
droughts_with_shortage = syn_drought_shortages[syn_drought_shortages['shortage_count'] > 0]
ax.scatter(droughts_with_shortage['severity'], 
           droughts_with_shortage['magnitude'], 
           c='r', 
           label='Droughts with Shortage',
           alpha=0.7)
# Now plot droughts where no shortage occurred
droughts_without_shortage = syn_drought_shortages[syn_drought_shortages['shortage_count'] == 0]
ax.scatter(droughts_without_shortage['severity'], 
           droughts_without_shortage['magnitude'], 
           c='gray', 
           label='Droughts without Shortage',
           alpha=0.5)

plt.legend()
plt.show()


#%% Plot drought severity vs shortage vulnerability
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(droughts_with_shortage['severity'], 
           droughts_with_shortage['shortage_vulnerability'], 
           c='r', 
           label='Droughts with Shortage',
           alpha=0.7)
ax.scatter(droughts_without_shortage['severity'],
              droughts_without_shortage['shortage_vulnerability'], 
              c='gray', 
              label='Droughts without Shortage',
              alpha=0.5)
plt.xlabel('Drought Severity')
plt.ylabel('Shortage Vulnerability')
plt.legend()
plt.show()