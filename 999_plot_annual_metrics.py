"""

Metrics are calculated for June 1 - Dec 31 for each year and each realization.

Metrics include:
- NYC aggregate inflow
- Aggregate full natural flow at Montague
- Sum of NYC ibt diversions
- NYC Montague/Trenton contributions
- Sum of Montague contributions
- Sum of NYC shortages
- Minimum aggregate NYC storage


This is reapeated for both ensemble types: stationary and climate_adjusted.

Final results for both ensembles are stored in dictionaries with, e.g.:

ensemble_nyc_inflow = {
    'stationary': pd.DataFrame,
    'climate_adjusted': pd.DataFrame
}

The plots show KDE distributions of metrics for each year, realization, and variable.

"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import matplotlib.colors as mcolors

import pywrdrb

from methods.utils import calculate_annual_metrics


plot_ensemble_types = ['stationary'] #, 'climate_adjusted']

ensemble_type_colors = {
    'stationary': 'orange',
    'climate_adjusted': 'purple'
}

# Add labels and formatting
metric_names = [
    'NYC Inflow (MGD)',
    'Montague Flow (MGD)', 
    'NYC IBT Diversions (MGD)',
    'NYC Downstream Contributions (MGD)',
    'Montague Violations (MGD)',
    'Min NYC Storage (%)'
]


ensemble_sum_nyc_inflow = {}
ensemble_sum_montague_flow = {}
ensemble_sum_nyc_ibt_diversions = {}
ensemble_sum_nyc_downstream_contributions = {}
ensemble_sum_montague_violations = {}
ensemble_sum_nyc_shortages = {}
ensemble_min_nyc_storage = {}
ensemble_satisficing_outcome = {}

for ensemble_type in plot_ensemble_types:
    inflow_type = f'{ensemble_type}_ensemble'
    fname = f'./pywrdrb/outputs/{ensemble_type}_ensemble_with_postprocessing.hdf5'
    
    nyc_reservoirs = ['cannonsville', 'pepacton', 'neversink']
    print(f"Loading data from {fname} for inflow type: {inflow_type}")
    
    data = pywrdrb.Data()
    data.load_from_export(fname)
    realizations = list(data.inflow[inflow_type].keys())
    years = data.inflow[inflow_type][realizations[0]].index.year.unique()
    
    # First, we need to arrange data so that it has all realizations in a single DataFrame
    # Each pd.DataFrame will contain realizations as columns
    # All data is still daily at this point
    print("Arranging data for realizations...")
    nyc_inflow = {}
    montague_flow = {}
    nyc_ibt_diversions = {}
    nyc_downstream_contributions = {}
    montague_violations = {}
    nyc_shortages = {}
    nyc_storage = {}
    for i, r in enumerate(realizations):
        nyc_inflow[r] = data.inflow[inflow_type][r].loc[:, nyc_reservoirs].sum(axis=1)
        montague_flow[r] = data.gage_flow[inflow_type][r]['delMontague']
        nyc_ibt_diversions[r] = data.ibt_diversions[inflow_type][r].sum(axis=1)
        nyc_downstream_contributions[r] = data.contribution[inflow_type][r]['mrf_montagueTrenton_nyc']
        montague_violations[r] = data.shortage[inflow_type][r]['delMontague']
        nyc_shortages[r] = data.shortage[inflow_type][r]['nyc']
        nyc_storage[r] = data.res_storage[inflow_type][r].loc[:, nyc_reservoirs].sum(axis=1)
    
    # Convert to DataFrames
    nyc_inflow = pd.DataFrame(nyc_inflow)
    montague_flow = pd.DataFrame(montague_flow)
    nyc_ibt_diversions = pd.DataFrame(nyc_ibt_diversions)
    nyc_downstream_contributions = pd.DataFrame(nyc_downstream_contributions)
    montague_violations = pd.DataFrame(montague_violations)
    nyc_shortages = pd.DataFrame(nyc_shortages)
    nyc_storage = pd.DataFrame(nyc_storage)
    
    # Next, calculate aggregate metrics for each year and realization
    # In this case, we calculate metrics for the period June 1 - Dec 31
    # the final pd.DataFrame will have years as index and realizations as columns
    print("Calculating annual metrics...")
    sum_nyc_inflow = calculate_annual_metrics(nyc_inflow, agg_func='sum')
    sum_montague_flow = calculate_annual_metrics(montague_flow, agg_func='sum')
    sum_nyc_ibt_diversions = calculate_annual_metrics(nyc_ibt_diversions, agg_func='sum')
    sum_nyc_downstream_contributions = calculate_annual_metrics(nyc_downstream_contributions, agg_func='sum')
    sum_montague_violations = calculate_annual_metrics(montague_violations, agg_func='sum')
    sum_nyc_shortages = calculate_annual_metrics(nyc_shortages, agg_func='sum')
    min_nyc_storage = calculate_annual_metrics(nyc_storage, agg_func='min')
    
    # The raw flow has 1 day in 2020 which is messing up the dims
    sum_montague_flow = sum_montague_flow.loc[:2019, :]
    
    # For min_nyc_storage, make percentage of total storage
    min_nyc_storage = min_nyc_storage / min_nyc_storage.max().max() * 100 
    
    # For sum_montague_flow, print the number of 0 values
    zero_count = (sum_montague_flow == 0).sum().sum()
    print(f"Number of zero values in sum_montague_flow: {zero_count}")
    sum_montague_flow.replace(0, np.nan, inplace=True) 
    
    # Next, calculate satisficing outcome based on:
    # montague violations < 100 MGD
    # nyc shortages < 100 MGD
    # min nyc storage > 20%
    satisficing_outcome = (sum_montague_violations < 1000) & \
                        (min_nyc_storage > 20)
    
    # Store results in ensemble dictionaries
    ensemble_sum_nyc_inflow[ensemble_type] = sum_nyc_inflow
    ensemble_sum_montague_flow[ensemble_type] = sum_montague_flow
    ensemble_sum_nyc_ibt_diversions[ensemble_type] = sum_nyc_ibt_diversions
    ensemble_sum_nyc_downstream_contributions[ensemble_type] = sum_nyc_downstream_contributions
    ensemble_sum_montague_violations[ensemble_type] = sum_montague_violations
    ensemble_sum_nyc_shortages[ensemble_type] = sum_nyc_shortages
    ensemble_min_nyc_storage[ensemble_type] = min_nyc_storage
    ensemble_satisficing_outcome[ensemble_type] = satisficing_outcome

#########################################################


## List of DataFrames to plot
# Each dataframe will be used to plot 1 kde/row
df_list = [
    ensemble_sum_nyc_inflow,
    ensemble_sum_montague_flow,
    ensemble_sum_nyc_ibt_diversions,
    ensemble_sum_nyc_downstream_contributions,
    ensemble_sum_montague_violations,
    ensemble_min_nyc_storage
]

satisficing_condition_list =[
    'less', 'less', 'greater', 
    'greater', 'greater', 'less'
]

### Plotting 
# Make a multi-row plot with KDE distributions for each metric
# One row per metric.  
# For axs[0]-axs[3] make scatter of all values where satisficing = False
# Scatter should have y = 0 and x = values

print("Plotting KDEs")

# fig, axs = plt.subplots(nrows=len(df_list), 
#                         ncols=1, 
#                         figsize=(8, 10), 
#                         sharex=False)


# # Plot KDEs
# for i, df_dict in enumerate(df_list):
#     for ensemble_type in plot_ensemble_types:
#         df = df_dict[ensemble_type]
#         sns.kdeplot(data=df.values.flatten(), 
#                     ax=axs[i], fill=True, 
#                     color=ensemble_type_colors[ensemble_type],
#                     cut=0, 
#                     label=ensemble_type)

# # Now plot non-satisficing years as scatter points in each KDE
# for i, df_dict in enumerate(df_list):
#     for ensemble_type in plot_ensemble_types:
#         df = df_dict[ensemble_type]
#         satisficing_outcome = ensemble_satisficing_outcome[ensemble_type]

#         xs = df[satisficing_outcome==False].values.flatten()
#         xs = np.float64(xs)  # Ensure float type
#         xs = xs[~np.isnan(xs)]  # Remove NaNs

#         # Get value of axis y-lims 
#         ylims = axs[i].get_ylim()
        
#         # Set y-value at 5% or 10% of y-lims
#         if ensemble_type == 'stationary':
#             y_val = ylims[0] + 0.05 * (ylims[1] - ylims[0])
#         elif ensemble_type == 'climate_adjusted':
#             y_val = ylims[0] + 0.1 * (ylims[1] - ylims[0])
        
#         ys = np.ones(xs.shape[0]) * y_val
        
#         axs[i].scatter(xs, ys, 
#                        color=ensemble_type_colors[ensemble_type], 
#                        s=30, 
#                        alpha=0.5, label=f'{ensemble_type} Unsatisficing')
    


def create_kde_with_satisficing_gradient(x_values, 
                                         satisficing_outcome, 
                                         satisficing_condition, 
                                         ax, color='blue', alpha=0.8, 
                                         label=None,
                                         add_satisficing_line=True,
                                         n_grid=200):
    """
    Create KDE plot with color gradient based on satisficing frequency.
    
    Parameters:
    x_values: array-like, variable observations (flattened)
    satisficing_outcome: array-like, boolean array indicating satisficing outcomes
    ax: matplotlib axis object
    color: base color for high satisficing areas
    alpha: transparency
    label: legend label
    n_grid: number of grid points for KDE evaluation
    bandwidth_factor: factor for interpolation bandwidth (relative to data range)
    """
    
    # Remove NaNs and ensure alignment
    valid_mask = ~np.isnan(x_values)
    x_clean = x_values[valid_mask]
    satisficing_clean = satisficing_outcome[valid_mask]
    
    if len(x_clean) == 0:
        return
    
    # Create KDE
    kde = gaussian_kde(x_clean)
    
    # Define evaluation range
    x_min, x_max = x_clean.min(), x_clean.max()
    x_range = x_max - x_min
    # x_eval = np.linspace(x_min - 0.1*x_range, x_max + 0.1*x_range, n_grid)
    x_eval = np.linspace(x_min, x_max, n_grid)
    
    # Evaluate KDE
    kde_values = kde(x_eval)
    
    # Add KDE outline
    ax.fill_between(x_eval, y1=kde_values, y2=0.0, color=color, alpha=0.2)
    
    if add_satisficing_line:
        # Calculate probability of non-satisficing for X<= x_eval
        if satisficing_condition == 'less':
            satisficing_rates = np.array([
                np.mean(satisficing_clean[x_clean <= x]) for x in x_eval
            ])
        elif satisficing_condition == 'greater':
            satisficing_rates = np.array([
                np.mean(satisficing_clean[x_clean >= x]) for x in x_eval
            ])
        else:
            raise ValueError("satisficing_condition must be 'less' or 'greater'")
        
        ax.plot(x_eval, satisficing_rates * kde_values.max(), color=color, linewidth=2, alpha=alpha, label=label)

        print(f"Min satisficing rate: {satisficing_rates.min()}, Max satisficing rate: {satisficing_rates.max()}")
        return satisficing_rates
    else:
        return None




def create_cdf_with_satisficing_gradient(x_values, 
                                        satisficing_outcome, 
                                        satisficing_condition, 
                                        ax, color='blue', alpha=0.8, 
                                        label=None,
                                        add_satisficing_line=True):
    """
    Create CDF plot with satisficing line showing % satisficing vs exceedance probability.
    
    Parameters:
    x_values: array-like, variable observations (flattened)
    satisficing_outcome: array-like, boolean array indicating satisficing outcomes
    satisficing_condition: 'less' or 'greater' for satisficing direction
    ax: matplotlib axis object
    color: base color for CDF
    alpha: transparency
    label: legend label
    add_satisficing_line: whether to add satisficing percentage line
    n_grid: number of grid points for evaluation
    """
    
    # Remove NaNs and ensure alignment
    valid_mask = ~np.isnan(x_values)
    x_clean = x_values[valid_mask]
    satisficing_clean = satisficing_outcome[valid_mask]
    
    if len(x_clean) == 0:
        return
    
    # Sort values for CDF calculation
    x_sorted = np.sort(x_clean)
    
    # Calculate empirical CDF
    n = len(x_sorted)
    cdf_values = np.arange(1, n + 1) / n
    
    # Convert to exceedance probability (1 - CDF)
    exceedance_prob = 1 - cdf_values
    
    exceedance_prob *= 100
    
    # Plot CDF with exceedance probability on x-axis
    ax.plot(exceedance_prob, x_sorted, color=color, linewidth=1, alpha=alpha)
    
    if add_satisficing_line:
        # Calculate satisficing rates at each x value
        if satisficing_condition == 'less':
            satisficing_rates = np.array([
                np.mean(satisficing_clean[x_clean <= x]) for x in x_sorted
            ])
        elif satisficing_condition == 'greater':
            satisficing_rates = np.array([
                np.mean(satisficing_clean[x_clean >= x]) for x in x_sorted
            ])
        else:
            raise ValueError("satisficing_condition must be 'less' or 'greater'")
        
        # Convert satisficing rates to percentages and plot against exceedance probability
        satisficing_pct = satisficing_rates * 100
        is_decreasing_idx = np.diff(satisficing_pct) < 0
        
        # Make a twin y-axis for the satisficing percentage
        ax2 = ax.twinx()
        ax2.set_ylabel('Satisficing Rate (%)', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        label = f'Satisficing Rate ({satisficing_condition})' if label is None else label
        ax2.set_ylim(0, 100)
        ax2.plot(exceedance_prob, 
                 satisficing_pct, 
                 color=color, linewidth=2,
                 alpha=alpha, linestyle='--', 
                 label=label)
        
        print(f"Min satisficing rate: {satisficing_rates.min():.3f}, Max satisficing rate: {satisficing_rates.max():.3f}")
        return satisficing_rates
    else:
        return None


# Plotting 
print("Plotting dists with satisficing gradients")
satisficing_rates = {}

fig, axs = plt.subplots(nrows=len(df_list), 
                        ncols=1, 
                        figsize=(4, 3* len(df_list)), 
                        sharex=True)

# Plot KDEs with color gradients
for i, df_dict in enumerate(df_list):
    for ensemble_type in plot_ensemble_types:
        df = df_dict[ensemble_type]
        satisficing_outcome_type = ensemble_satisficing_outcome[ensemble_type]
        satisficing_condition = satisficing_condition_list[i]
        
        add_satisficing_line = True if i in [0, 1, 2, 3] else False
        
        # Flatten the data
        x_values = df.values.flatten()
        x_values = np.float64(x_values)  # Ensure float type
        
        satisficing_flat = satisficing_outcome_type.values.flatten()
        
        assert len(x_values) == len(satisficing_flat), \
            f"x_values ({len(x_values)}) and satisficing_flat ({len(satisficing_flat)}) dont have the same length"
        
        # satisficing_rates[i] = create_kde_with_satisficing_gradient(
        #     x_values=x_values,
        #     satisficing_outcome=satisficing_flat,
        #     satisficing_condition=satisficing_condition,
        #     ax=axs[i],
        #     color=ensemble_type_colors[ensemble_type],
        #     alpha=0.7,
        #     n_grid=400,
        #     add_satisficing_line=add_satisficing_line,
        #     label=ensemble_type)
        
        satisficing_rates[i] = create_cdf_with_satisficing_gradient(
            x_values=x_values,
            satisficing_outcome=satisficing_flat,
            satisficing_condition=satisficing_condition,
            ax=axs[i],
            color=ensemble_type_colors[ensemble_type],
            alpha=0.7,
            add_satisficing_line=add_satisficing_line,
            label=ensemble_type)
    
    if i in [0,1,2,3,4]:
        # get current limits
        ylims = axs[i].get_ylim()
        
        if ylims[0] < 1.0:        
            new_ylim = [1, ylims[1]]
        else:
            new_ylim = [ylims[0], ylims[1]]
                
        axs[i].set_ylim(new_ylim)
        axs[i].set_yscale('log')
    
    axs[i].set_xscale('symlog')

        


plt.tight_layout()
if len(plot_ensemble_types) == 1:
    fname = f'annual_metrics_cdf_{plot_ensemble_types[0]}.png'
else:
    fname = 'annual_metrics_cdf'
    for e in plot_ensemble_types:
        fname += f'_{e}'
    fname += '.png'

plt.savefig(fname, dpi=250)


