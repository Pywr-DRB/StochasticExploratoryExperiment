import pywrdrb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime, timedelta

from config import ENSEMBLE_SETS, RECONSTRUCTION_OUTPUT_FNAME



output_filenames = [ENSEMBLE_SETS[i].output_file for i in range(len(ENSEMBLE_SETS))]
output_filenames.append(RECONSTRUCTION_OUTPUT_FNAME)

results_sets = ["major_flow", "res_storage"]


# Load the data
data = pywrdrb.Data(results_sets=results_sets,
                    print_status=True)
data.load_output(output_filenames=output_filenames)
data.load_observations()

# Make a single copy of stationary_ensemble which combines stationary_ensemble_set{i,,,10}
for results_set in results_sets:
    all_set_results_data = {}
    full_results_set_dict = getattr(data, results_set)
    for i in range(1, 11):
        set_name = f'stationary_ensemble_set{i}'
        set_data = full_results_set_dict[set_name]
        set_data_with_reals = {100*(i-1) + k: v for k, v in set_data.items()}
        all_set_results_data.update(set_data_with_reals)
    full_results_set_dict['stationary_ensemble'] = all_set_results_data
    setattr(data, results_set, full_results_set_dict)

inflow_type = 'stationary_ensemble'
realization_ids = list(data.major_flow[inflow_type].keys())

nyc_reservoirs = ['cannonsville', 'pepacton', 'neversink']

# Make a df with nyc_agg storage for all realizations
df_nyc_storage = data.res_storage[inflow_type][realization_ids[0]][nyc_reservoirs].sum(axis=1)
df_nyc_storage = pd.DataFrame(df_nyc_storage, columns=[realization_ids[0]])
for realization_id in realization_ids[1:10]:
    df_storage = data.res_storage[inflow_type][realization_id][nyc_reservoirs].copy()
    df_nyc_storage = pd.concat([df_nyc_storage, df_storage.sum(axis=1)], axis=1)


# Assuming df is your DataFrame with N columns (realizations) and M rows (days)
# Convert index to day of year if not already done

df = df_nyc_storage.copy()

# make as percent of full
df = df / df.max().max() * 100

period = 'weekly'
if period == 'daily':
    # Convert index to day of year
    df['period_of_year'] = pd.to_datetime(df.index).dayofyear
    periods = np.arange(1, 367)
elif period == 'weekly':
    df['period_of_year'] = pd.to_datetime(df.index).isocalendar().week
    # aggregate to week, but do the min storage
    df = df.resample('W').min()
    periods = np.arange(1, 54)  # 53 weeks in a year


n_percentiles = 20  # Adjust based on your data range
percentile_matrix = np.zeros((n_percentiles, len(periods)))
percentiles = np.linspace(0, 1, n_percentiles)
storage_range = np.linspace(0, 100, n_percentiles)  # Assuming storage is in percentage

for i, period in enumerate(periods):
   period_data = df[df['period_of_year'] == period].iloc[:, :-1].values.flatten()
   if len(period_data) > 0:
        # Calculate percentiles for the current period
        percentiles_values = np.quantile(percentiles, np.sort(period_data/100))
        percentile_matrix[:, i] = percentiles_values

# Create meshgrid for plotting
X, Y = np.meshgrid(periods, storage_range)

# Plot
fig, ax = plt.subplots(figsize=(12, 8))
cmap = 'BrBG'
im = ax.pcolormesh(X, Y, percentile_matrix, 
                   cmap=cmap, vmin=0, vmax=1, 
                   shading='auto')

# Customize plot
ax.set_xlabel('Period of Year')
ax.set_ylabel('Storage Volume')
ax.set_title('Storage Distribution by Period of Year')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Percentile Storage Volume')

plt.tight_layout()

# save
plt.savefig('storage_percentile_distribution.png', dpi=300, bbox_inches='tight')
