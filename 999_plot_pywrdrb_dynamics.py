import pywrdrb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from methods.utils import combine_multiple_ensemble_sets_in_data
from config import STATIONARY_ENSEMBLE_SETS, CLIMATE_ADJUSTED_ENSEMBLE_SETS, RECONSTRUCTION_OUTPUT_FNAME

ensemble_type = 'stationary'  # or 'stationary'
inflow_type = f'{ensemble_type}_ensemble'

if ensemble_type == 'stationary':
    ensemble_set_spec_list = STATIONARY_ENSEMBLE_SETS
elif ensemble_type == 'climate_adjusted':
    ensemble_set_spec_list = CLIMATE_ADJUSTED_ENSEMBLE_SETS

output_filenames = [ensemble_set_spec_list[i].output_file for i in range(len(ensemble_set_spec_list))]
output_filenames.append(RECONSTRUCTION_OUTPUT_FNAME)

results_sets = ["res_storage"]


# Load the data
data = pywrdrb.Data(results_sets=results_sets,
                    print_status=True)
data.load_output(output_filenames=output_filenames)
data.load_observations()

data = combine_multiple_ensemble_sets_in_data(data, results_sets, ensemble_type=ensemble_type)

realization_ids = list(data.res_storage[inflow_type].keys())

nyc_reservoirs = ['cannonsville', 'pepacton', 'neversink']

# Make a df with nyc_agg storage for all realizations
period = 'daily'
if period == 'daily':
    agg_period = 'D'
elif period == 'weekly':
    agg_period = 'W'
elif period == 'annual':
    agg_period = 'YS'

# If you can restructure to get all realizations at once
print('Making DF of all data')
all_data = {rid: data.res_storage[inflow_type][rid][nyc_reservoirs].sum(axis=1).resample(agg_period).min() 
            for rid in realization_ids}
df_nyc_storage = pd.DataFrame(all_data)

df = df_nyc_storage.copy()
df = df / df.max().max() * 100 # Percentage of max storage


# Relabel all columns 1, 2, ..., N
df.columns = [f'{i+1}' for i in range(df.shape[1])]

# Period of year should be relative to June 1st
if period == 'daily':
   # Calculate period_of_year where June 1 = 1, July 31 next year = 365/366
   june_1 = pd.to_datetime(df.index.year.astype(str) + '-06-01')
   
   # For dates >= June 1: days since June 1 + 1
   # For dates < June 1: days since June 1 of previous year + 1
   mask_after_june = df.index >= june_1
   
   df['period_of_year'] = np.where(
       mask_after_june,
       (df.index - june_1).days + 1,
       (df.index - pd.to_datetime((df.index.year - 1).astype(str) + '-06-01')).days + 1
   )
elif period == 'weekly':
    df['period_of_year'] = pd.to_datetime(df.index).isocalendar().week


### Plotting
# Series of zero-storage conditions
print('Plotting...')
fig, axs = plt.subplots(2, 1, figsize=(6, 6), 
                        sharex=True, gridspec_kw={'height_ratios': [3, 0.5]})

zero_storage_period_of_year = []

# First ax: Timeseries of each years storage
threshold = 10.0 # Percent storage of interest
threshold_year_count = 0
all_years = df.index.year.unique()
n_realizations = df.shape[1]-1
for year in all_years:
    # Create date range from June 1 of current year to July 31 of next year
    start_date = pd.Timestamp(year, 6, 1)
    end_date = pd.Timestamp(year + 1, 5, 31)
    # if start or end_date is not in the index, skip this year
    if start_date not in df.index or end_date not in df.index:
        print(f"Skipping year {year} due to missing dates in index.")
        continue
    
    year_data = df[(df.index >= start_date) & (df.index <= end_date)]
    
    # keep only columns with at least one zero-storage condition
    if (year_data <= threshold).any().any():
        has_zero = True
        period_of_year = year_data['period_of_year']
        
        # drop the 'period_of_year' column for plotting
        year_data = year_data.drop(columns=['period_of_year'])
        
        year_data = year_data.loc[:, (year_data <= threshold).any(axis=0)]
        
        # store the period_of_year for zero-storage values
        for col in year_data.columns:
            zero_storage_period_of_year.extend(period_of_year[year_data[col] <= threshold].dropna().values.tolist())
            threshold_year_count += 1

    else:
        has_zero = False
    if has_zero:
        axs[0].plot(period_of_year.values, year_data.values, alpha=0.5, color='orange')


axs[0].set_ylabel('Storage (%)')
axs[0].set_ylim(0, 100)

# Second ax: Box plot of days of the year with zero storage
# X-axis is day of the year, so the box-plot should be horizontal
zero_storage_period_of_year = np.array(zero_storage_period_of_year).flatten()

axs[1].boxplot(zero_storage_period_of_year, vert=False)
axs[1].set_yticks([])  # Hide y-ticks since we only have one box
axs[1].set_xlim(0, 366)  # Set x-limits to cover all days of the year

# Change x-ticks to be 1st of every month starting June 1
axs[1].set_xticks([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366])
axs[1].set_xticklabels(['Jun', 'Jul', 'Aug', 'Sep', 'Oct', 
                        'Nov', 'Dec', 'Jan', 'Feb', 
                        'Mar', 'Apr', 'May', 'Jun'])

axs[0].set_title(f'Simulated Years with NYC storage below {threshold}%\nCount: {threshold_year_count}/{int(len(all_years)*n_realizations)} total years')
plt.tight_layout()
plt.savefig(f'{ensemble_type}_nyc_storage_timeseries_below_{threshold}.png', dpi=300, bbox_inches='tight')









def reshape_to_yearly_columns(df, freq='weekly'):
   """
   Reshape timeseries DataFrame from [N, M] to [365, 70*M] format.
   
   Parameters:
   - df: DataFrame with M simulation columns, N timesteps (70 years)
   - freq: 'weekly' or 'daily' - frequency of original data
   
   Returns:
   - DataFrame with shape [365, 70*M] where each column is one year
   """
   
   if freq == 'weekly':
       # For weekly data: N should be 70*52 = 3640
       weeks_per_year = 52
       expected_rows = 70 * weeks_per_year
       
       # Convert weekly to daily by interpolation/resampling
       daily_data = []
       for col in df.columns:
           # Create daily index for interpolation
           weekly_series = pd.Series(df[col].values, 
                                   index=pd.date_range('2000-01-01', 
                                                     periods=len(df), 
                                                     freq='W'))
           # Interpolate to daily
           daily_series = weekly_series.resample('D').interpolate()
           daily_data.append(daily_series.values)
       
       # Stack daily data
       daily_array = np.column_stack(daily_data)
       
   elif freq == 'daily':
       # For daily data: N should be 70*365 = 25550
       daily_array = df.values
   
   else:
       raise ValueError("freq must be 'weekly' or 'daily'")
   
   # Reshape to [365, 70*M]
   n_years = df.index.year.nunique()
   n_sims = df.shape[1]
   days_per_year = 365
   
   # Trim to exact 70 years if needed
   total_days_needed = n_years * days_per_year
   if daily_array.shape[0] > total_days_needed:
       daily_array = daily_array[:total_days_needed]
   
   # Reshape each simulation column
   reshaped_cols = []
   for sim_idx in range(n_sims):
       sim_data = daily_array[:, sim_idx]
       # Reshape to [365, 70] then flatten column-wise
       yearly_matrix = sim_data.reshape(n_years, days_per_year).T
       reshaped_cols.append(yearly_matrix)
   
   # Concatenate all simulations horizontally
   final_array = np.hstack(reshaped_cols)
   
   # Create column names: sim1_year1, sim1_year2, ..., sim2_year1, etc.
   col_names = []
   for sim_idx in range(n_sims):
       sim_name = df.columns[sim_idx]
       for year in range(n_years):
           col_names.append(f"{sim_name}_year{year+1}")
   
   # Create DataFrame
   result_df = pd.DataFrame(final_array, columns=col_names)
   
   # Add day of year index
   result_df.index = range(1, 366)
   result_df.index.name = 'day_of_year'
   
   return result_df



# Make a plot of storage frequency over the annual year
def plot_storage_percentile_heatmap(df, storage_bins=50, figsize=(15, 8)):
   """
   Create 2D heatmap of storage percentiles by week of year.
   Each week's percentiles sum to 100.
   
   Parameters:
   - df: DataFrame with simulation columns, 'period_of_year' column
   - storage_bins: Number of bins for storage percentage (default 50)
   - figsize: Figure size tuple
   """
   
   # Melt DataFrame to long format
   simulation_cols = [col for col in df.columns if col != 'period_of_year']
   df_long = df.melt(
       id_vars=['period_of_year'], 
       value_vars=simulation_cols,
       var_name='simulation', 
       value_name='storage_pct'
   )
   
   # Create storage bins
   storage_bin_edges = np.linspace(0, 100, storage_bins + 1)
   storage_bin_centers = (storage_bin_edges[:-1] + storage_bin_edges[1:]) / 2
   
   # Digitize storage values into bins
   df_long['storage_bin'] = np.digitize(df_long['storage_pct'], storage_bin_edges) - 1
   df_long['storage_bin'] = np.clip(df_long['storage_bin'], 0, storage_bins - 1)
   
   # Calculate percentiles for each week-storage combination
   percentile_matrix = np.zeros((storage_bins, 52))
   
   for week in range(1, 53):
       week_data = df_long[df_long['period_of_year'] == week]
       bin_counts = np.bincount(week_data['storage_bin'], minlength=storage_bins)
       
       # Convert counts to percentiles (sum to 100)
       total_count = bin_counts.sum()
       if total_count > 0:
           percentiles = (bin_counts / total_count) * 100
       else:
           percentiles = np.zeros(storage_bins)
       
       percentile_matrix[:, week-1] = percentiles
   
   # Create heatmap
   fig, ax = plt.subplots(figsize=figsize)
   
   # Flip storage axis so 0% is at bottom, 100% at top
   percentile_matrix_flipped = np.flipud(percentile_matrix)
   
   im = ax.imshow(
       percentile_matrix_flipped, 
       aspect='auto',
       cmap='viridis',
       extent=[0.5, 52.5, 0, 100]
   )
   
   # Customize plot
   ax.set_xlabel('Week of Year')
   ax.set_ylabel('Storage Percentage (%)')
   ax.set_title('Storage Percentiles by Week of Year')
   
   # Set ticks
   ax.set_xticks(np.arange(1, 53, 4))
   ax.set_yticks(np.arange(0, 101, 10))
   
   # Add colorbar
   cbar = plt.colorbar(im, ax=ax)
   cbar.set_label('Percentile (%)')
   
   plt.tight_layout()
   
   plt.savefig(f'{ensemble_type}_storage_percentile_heatmap.png', dpi=300, bbox_inches='tight')
   
   return fig, ax


# plot_storage_percentile_heatmap(df, storage_bins=8)