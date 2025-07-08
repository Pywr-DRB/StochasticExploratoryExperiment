import pywrdrb
import matplotlib.pyplot as plt
import numpy as np

inflow_type = 'stationary_ensemble'
output_fname = f"./pywrdrb/outputs/{inflow_type}.hdf5"
results_sets = ['major_flow', 'res_storage']

data = pywrdrb.Data(results_sets=results_sets, 
                    print_status=True)
data.load_output(output_filenames=[output_fname])

realization_ids = list(data.major_flow[inflow_type].keys())

nyc_reservoirs = ['cannonsville', 'pepacton', 'neversink']

# Plot the aggregate NYC reservoir storage 
# for each realization and for each year
fig, ax = plt.subplots(figsize=(10, 6))
for realization_id in realization_ids:
    df_storage = data.res_storage[inflow_type][realization_id]
    
    # calculate nyc total storge
    df_storage['nyc_agg'] = df_storage[nyc_reservoirs].sum(axis=1)

    years = df_storage.index.year.unique()
    
    # For each year, plot the annual storage series
    # make x axis from 1-365
    for y in years:
        df_year = df_storage[df_storage.index.year == y]
        xs = np.arange(1, len(df_year) + 1)
        ax.plot(xs, df_year['nyc_agg'], 
                color='darkorange', 
                alpha=0.2)

# save
plt.savefig('nyc_reservoir_storage_ensemble.png', dpi=300, bbox_inches='tight')




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime, timedelta

# Assuming df is your DataFrame with N columns (realizations) and M rows (days)
# Convert index to day of year if not already done

df = data.res_storage[inflow_type][0]  # Use the first realization for demonstration
# add all realizations as columns
for realization_id in realization_ids[1:]:
    df_realization = data.res_storage[inflow_type][realization_id]
    df = pd.concat([df, df_realization], axis=1)

df['day_of_year'] = pd.to_datetime(df.index).dayofyear

# Create storage bins for frequency calculation
n_bins = 50  # Adjust based on your data range
storage_min, storage_max = df.iloc[:, :-1].min().min(), df.iloc[:, :-1].max().max()
storage_bins = np.linspace(storage_min, storage_max, n_bins)

# Create 2D histogram for each day of year
days = np.arange(1, 367)  # 366 for leap years
freq_matrix = np.zeros((len(storage_bins)-1, len(days)))

for i, day in enumerate(days):
   day_data = df[df['day_of_year'] == day].iloc[:, :-1].values.flatten()
   if len(day_data) > 0:
       hist, _ = np.histogram(day_data, bins=storage_bins)
       freq_matrix[:, i] = hist

# Create meshgrid for plotting
X, Y = np.meshgrid(days, storage_bins[:-1])

# Plot
fig, ax = plt.subplots(figsize=(12, 8))
im = ax.pcolormesh(X, Y, freq_matrix, cmap='viridis', shading='auto')

# Customize plot
ax.set_xlabel('Day of Year')
ax.set_ylabel('Storage Volume')
ax.set_title('Storage Frequency Distribution by Day of Year')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Frequency')

plt.tight_layout()

# save
plt.savefig('storage_frequency_distribution.png', dpi=300, bbox_inches='tight')
