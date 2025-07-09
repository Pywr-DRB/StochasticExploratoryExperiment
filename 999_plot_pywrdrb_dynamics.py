import pywrdrb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime, timedelta

inflow_type = 'stationary_ensemble'
output_fname = f"./pywrdrb/outputs/{inflow_type}.hdf5"
results_sets = ['major_flow', 'res_storage']

data = pywrdrb.Data(results_sets=results_sets, 
                    print_status=True)
data.load_output(output_filenames=[output_fname])

realization_ids = list(data.major_flow[inflow_type].keys())

nyc_reservoirs = ['cannonsville', 'pepacton', 'neversink']

# Make a df with nyc_agg storage for all realizations
df_nyc_storage = data.res_storage[inflow_type][realization_ids[0]][nyc_reservoirs].sum(axis=1)

df_nyc_storage = pd.DataFrame(df_nyc_storage, columns=[realization_ids[0]])
for realization_id in realization_ids[1:]:
    df_storage = data.res_storage[inflow_type][realization_id][nyc_reservoirs].copy()
    df_nyc_storage[realization_id] = df_storage.sum(axis=1)




# Assuming df is your DataFrame with N columns (realizations) and M rows (days)
# Convert index to day of year if not already done

df = df_nyc_storage.copy()
df['day_of_year'] = pd.to_datetime(df.index).dayofyear

# Create storage bins for frequency calculation
n_bins = 10  # Adjust based on your data range
storage_min, storage_max = df.iloc[:, :-1].min().min(), df.iloc[:, :-1].max().max()
storage_bins = np.linspace(storage_min, storage_max, n_bins)

# Create 2D histogram for each day of year
days = np.arange(1, 367)  # 366 for leap years
freq_matrix = np.zeros((len(storage_bins)-1, len(days)))

for i, day in enumerate(days):
   day_data = df[df['day_of_year'] == day].iloc[:, :-1].values.flatten()
   if len(day_data) > 0:
       hist, _ = np.histogram(day_data, bins=storage_bins, density=True)
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
