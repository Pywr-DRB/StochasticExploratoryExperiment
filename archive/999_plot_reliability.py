import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sglib import HDF5Manager


### Load shortage series
# Each ensemble (data_by_site) is a dict with:
# {site_name (str) : pd.DataFrame} 
# and the pd.DataFrame has columns for each realization
# and daily datetime index

# Stationary ensemble
hdf_manager = HDF5Manager()
stationary_shortage_fname = f"./pywrdrb/shortages/stationary_ensemble_shortages.hdf5"
stationary_shortage_ensemble = hdf_manager.load_ensemble(stationary_shortage_fname)
stationary_shortage = stationary_shortage_ensemble.data_by_site

stationary_contributions_fname = f"./pywrdrb/contributions/stationary_ensemble_contributions.hdf5"
stationary_contributions_ensemble = hdf_manager.load_ensemble(stationary_contributions_fname)
stationary_contributions = stationary_contributions_ensemble.data_by_site

# Climate adjusted ensemble
climate_adjusted_shortage_fname = f"./pywrdrb/shortages/climate_adjusted_ensemble_shortages.hdf5"
climate_adjusted_shortage_ensemble = hdf_manager.load_ensemble(climate_adjusted_shortage_fname)
climate_adjusted_shortage = climate_adjusted_shortage_ensemble.data_by_site

### Calculate summative metrics

# number of years with shortages
def calculate_yearly_shortage_count(shortage_data, threshold=0):
   """
   Calculate number of years with shortages for each column.
   
   Parameters:
   shortage_data: DataFrame with datetime index and 1000 columns
   threshold: minimum value to count as shortage
   
   Returns:
   Series with 1000 values (one per column) representing 
   number of years where that column had values > threshold
   """
   # Group by year and check if any value > threshold per year per column
   yearly_has_shortage = shortage_data.resample('YS').apply(lambda x: (x > threshold).any())
   
   # Count years with shortage for each column
   yearly_counts = yearly_has_shortage.sum()
   
   return yearly_counts


threshold = 100.0  # Define a threshold for shortage

stationary_shortage_yearly_counts = {}
for site, data in stationary_shortage.items():
    print(f"Calculating yearly shortage counts for {site} in stationary ensemble...")
    yearly_counts = calculate_yearly_shortage_count(data, threshold=threshold)
    stationary_shortage_yearly_counts[site] = yearly_counts

climate_adjusted_shortage_yearly_counts = {}
for site, data in climate_adjusted_shortage.items():
    print(f"Calculating yearly shortage counts for {site} in climate adjusted ensemble...")
    yearly_counts = calculate_yearly_shortage_count(data, threshold=threshold)
    climate_adjusted_shortage_yearly_counts[site] = yearly_counts



# Box plots of yearly shortage counts
# figure should have 4 subplots, one for each site
# each subplot should have two box plots:
# one for stationary ensemble and one for climate adjusted ensemble

fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True)
axes = axes.flatten()
sites = list(stationary_shortage_yearly_counts.keys())
for i, site in enumerate(sites):
    stationary_counts = stationary_shortage_yearly_counts[site]
    climate_adjusted_counts = climate_adjusted_shortage_yearly_counts[site]
    
    # Create box plots
    axes[i].boxplot([stationary_counts, climate_adjusted_counts], 
                    labels=['Stationary', 'Climate Adjusted'])
    
    axes[i].set_title(f'Yearly Shortage Counts for {site}')
    axes[i].set_ylabel('Number of Years with Shortages')
    axes[i].set_xticks([1, 2])
    axes[i].set_xticklabels(['Stationary', 'Climate Adjusted'])

plt.tight_layout()
plt.savefig('yearly_shortage_counts_boxplots.png', dpi=300)

### Plot shortage vs contributions
realization_ids = list(stationary_contributions['mrf_montagueTrenton_nyc'].keys())
stationary_contribution_matrix = [stationary_contributions['mrf_montagueTrenton_nyc'][i] for i in realization_ids]
stationary_contribution_matrix = pd.concat(stationary_contribution_matrix, axis=1)

shortage_matrix = [stationary_shortage['delMontague'][i] for i in realization_ids]
shortage_matrix = pd.concat(shortage_matrix, axis=1)

# calculate rolling sum of contributions 
window = 60
roll_stationary_contribution_matrix = stationary_contribution_matrix.rolling(window=window, min_periods=window).sum().dropna()
roll_shortage_matrix = shortage_matrix.rolling(window=window, min_periods=window).sum().dropna()
snip_shortage_matrix = shortage_matrix.loc[roll_stationary_contribution_matrix.index, :]

plt.clf()
plt.scatter(roll_stationary_contribution_matrix.values[:,:100].flatten(), roll_shortage_matrix.values[:,:100].flatten(), alpha=0.5, s=1)
plt.savefig('stationary_contribution_vs_shortage.png', dpi=300)