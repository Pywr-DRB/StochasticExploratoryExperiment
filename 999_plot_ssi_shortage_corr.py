import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import pywrdrb
from sglib import HDF5Manager

from methods.utils import combine_multiple_ensemble_sets_in_data
from config import STATIONARY_ENSEMBLE_SETS, CLIMATE_ADJUSTED_ENSEMBLE_SETS


ensemble_type =  'stationary' # 'climate_adjusted'  # or 'stationary'
inflow_type = f'{ensemble_type}_ensemble'

if ensemble_type == 'stationary':
    ensemble_set_spec_list = STATIONARY_ENSEMBLE_SETS
elif ensemble_type == 'climate_adjusted':
    ensemble_set_spec_list = CLIMATE_ADJUSTED_ENSEMBLE_SETS

output_filenames = [ensemble_set_spec_list[i].output_file for i in range(len(ensemble_set_spec_list))]

results_sets = [
    "major_flow",
    "mrf_target"
]

# Load the data
data = pywrdrb.Data(results_sets=results_sets, print_status=True)
data.load_output(output_filenames=output_filenames)
data = combine_multiple_ensemble_sets_in_data(data, results_sets, ensemble_type=ensemble_type)

# Calculate shortage for each realization
shortage_dict = {}
realization_ids = list(data.major_flow[inflow_type].keys())
for i in realization_ids:
    target = data.mrf_target[inflow_type][i]['delMontague']
    flow = data.major_flow[inflow_type][i]['delMontague']
    shortage = target - flow
    shortage[shortage < 0] = 0  # Set negative shortages (surplus) to zero
    shortage[:2] = 0.0  # Set first 2 days to 0.0 due to model warmup
    
    # Store in dictionary
    shortage_dict[i] = shortage

# Convert to DataFrame
shortage_dict[i] = pd.DataFrame(shortage_dict[i])
ys = [shortage_dict[i] for i in realization_ids]
ys = pd.concat(ys, axis=1)



### Load SSI
ssi_window = 24
hdf_manager = HDF5Manager()

# SSI is monthly
ssi_fname = f"./pywrdrb/drought_metrics/{ensemble_type}_ensemble_ssi{ssi_window}.hdf5"
syn_ssi_ensemble = hdf_manager.load_ensemble(ssi_fname)
syn_ssi = syn_ssi_ensemble.data_by_site

# Shortages are daily
shortage_fname = f"./pywrdrb/shortages/{ensemble_type}_ensemble_shortages.hdf5"
syn_shortage_ensemble = hdf_manager.load_ensemble(shortage_fname)
syn_shortage = syn_shortage_ensemble.data_by_site

ys = syn_shortage['delMontague']

# Make scatter plot of SSI vs. shortage for each node


ys.index = pd.to_datetime(ys.index)
ys = ys.resample('MS').sum()
xs = syn_ssi['delMontague'].loc[ys.index, :]

ys = ys.loc[xs.index, :]
xs = xs.loc[ys.index, :]

zs = [data.major_flow[inflow_type][i].loc[:, 'delMontague'] for i in data.major_flow[inflow_type].keys()]
zs = pd.concat(zs, axis=1)


# Keep only data where ys>threshold
threshold = 0.0
mask = ys > threshold
ys = ys.values[mask].flatten()
zs = zs.values[mask].flatten()

plt.clf()
plt.scatter(zs, ys, alpha=0.5)
plt.xlabel('Montague Flow (MGD)')
plt.ylabel('Shortage (MGD)')
# plt.yscale('log')
# plt.xscale('log')
plt.savefig(f'./ssi_shortage_scatter_delMontague_{ensemble_type}.png')


# Plot the distribution of day of year with shortage > 10
ys[ys>10] = 1
ys = ys.fillna(0) 
shortage_day_of_year = ys.groupby(ys.index.dayofyear).sum().sum(axis=1)


import seaborn as sns
plt.clf()
sns.histplot(x=shortage_day_of_year.index, y=shortage_day_of_year.values, bins=100, kde=False)
plt.savefig(f'./ssi_shortage_day_of_year_delMontague_{ensemble_type}.png')

# Plot a single timeseries realization of shortage
plt.clf()
plt.plot(ys.index, ys.iloc[:, 0], label='Shortage')
plt.savefig(f'./ssi_shortage_timeseries_delMontague_{ensemble_type}.png')



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_nonzero_days_histogram(df, threshold=0.0, figsize=(12, 6)):
   """
   Create histogram of days of year where values exceed threshold.
   
   Parameters:
   df: DataFrame with datetime index and numeric columns
   threshold: minimum value to count as "non-zero" (default 0.0)
   figsize: figure size tuple
   """
   # Count non-zero values per day across all columns
   nonzero_counts = (df > threshold).sum(axis=1)
   
   # Group by day of year and sum the counts
   dayofyear_counts = nonzero_counts.groupby(df.index.dayofyear).sum()
   
   # Create histogram
   fig, ax = plt.subplots(figsize=figsize)
   
   # Plot as bar chart for better visualization
   dayofyear_counts.plot(kind='bar', ax=ax, alpha=0.7, width=0.8)
   ax.set_xlabel('Day of Year')
   ax.set_ylabel('Total Non-Zero Values')
   ax.set_title(f'Total Non-Zero Values by Day of Year (threshold > {threshold})')
   
   # Add month labels
   month_starts = [1, 32, 61, 92, 122, 153, 183, 214, 245, 275, 306, 336]
   month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
   
   ax2 = ax.twiny()
   ax2.set_xlim(ax.get_xlim())
   ax2.set_xticks([i-1 for i in month_starts])  # Adjust for 0-based indexing
   ax2.set_xticklabels(month_labels)
   plt.ylim(0, 1000)
   plt.savefig(f'nonzero_days_histogram_{threshold}.png')
   return fig, ax


# Usage
plot_nonzero_days_histogram(ys)
