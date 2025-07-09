#%% 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pywrdrb

from methods.metrics.shortfall import get_shortfall_metrics, calculate_hashimoto_metrics
from methods.metrics.shortfall import get_flow_and_target_values, add_trenton_equiv_flow
from config import RECONSTRUCTION_OUTPUT_FNAME, STATIONARY_ENSEMBLE_OUTPUT_FNAME

#%% Load pywrdrb output

output_filenames = [
    RECONSTRUCTION_OUTPUT_FNAME,
    STATIONARY_ENSEMBLE_OUTPUT_FNAME    
]

results_sets = [
    "major_flow", "inflow", "res_storage",
    "lower_basin_mrf_contributions", "mrf_target", 
    "ibt_diversions", "ibt_demands",
    ]

# Load the data
data = pywrdrb.Data(results_sets=results_sets)
data.load_output(output_filenames=output_filenames)
data.load_observations()

flowtypes = list(data.major_flow.keys())

data = add_trenton_equiv_flow(data)

#%%
def annual_max_positive_streak(series):
   """
   Calculate annual maximum length of consecutive positive values.
   
   Args:
       series: pd.Series with datetime index, daily frequency
   
   Returns:
       pd.Series with annual max streak lengths, indexed by year
   """
   # Create boolean mask for positive values (> 0, not >= 0)
   is_positive = series > 0
   
   # Find streak lengths using groupby on cumulative sum of negations
   # When is_positive changes from True to False, cumsum increments
   streak_groups = (~is_positive).cumsum()
   
   # For each group of consecutive positive values, calculate cumulative count
   # Non-positive values will have streak_length = 0
   streak_lengths = is_positive.groupby(streak_groups).cumsum()
   
   # Get annual maximum streak length for each year
   annual_max = streak_lengths.groupby(series.index.year).max()
   
   return annual_max


#%% Calculate shortage percentiles

# Loop through models and create shortage matrices for each 
shortage_matrix_dict = {}
duration_matrix_dict = {}
for model in ['reconstruction', 'stationary_ensemble']:
    
    shortage_matrix_dict[model] = {}
    duration_matrix_dict[model] = {}
    
    for node in ['delMontague', 'delTrenton']:
        
        if model == 'obs':        
            start_date = '2000-01-01'
            end_date = '2023-12-31'
        else:
            start_date = None
            end_date = None

        # Start by making a matrix of flows and target values across all realizations
        realizations = list(data.major_flow[model].keys())

        shortage = []
        duration = []
        for r in realizations:
            # Get the flow and target values for this node, model, and realization
            flow_series, target_series = get_flow_and_target_values(data, node,
                                                                    model, r,
                                                                    start_date=start_date, 
                                                                    end_date=end_date)

            # Calculate shortages
            shortage_series = target_series - flow_series
            shortage_series[shortage_series < 0] = 0  # Set negative shortages
            
            # Ignore shortage in first 2 days due to model warmup
            shortage_series.iloc[:2] = 0.0
            
            # Get the max duration of shortage
            max_shortage_duration = annual_max_positive_streak(shortage_series)
            
            # Aggregate to annual
            annual_shortage_series = shortage_series.resample('YS').sum()

            shortage.append(annual_shortage_series.values)
            duration.append(max_shortage_duration.values)

        # Convert to numpy arrays
        # Shortage is a matrix with shape (n_realizations, n_periods)
        shortage = np.array(shortage)
        duration = np.array(duration)

        n_realizations = shortage.shape[0]
        max_shortage = np.max(shortage)
        shortage_magnitude_range = np.linspace(0, max_shortage, 100)
        percentile_range = np.linspace(0, 1, 100)

        # Based on simulated flows and targets for a given node and model, 
        # this function calculates shortage across different metrics. 
        # the shortage_matrix should be 2D with shape (n_realizations, n_percentiles)
        # where each cell contains the shortage magnitude for that realization and percentile.
        shortage_matrix = np.zeros((n_realizations, len(percentile_range)))
        duration_matrix = np.zeros((n_realizations, len(percentile_range)))

        for i, percentile in enumerate(percentile_range):
            # Calculate the shortage for each realization at this percentile
            shortage_matrix[:, i] = np.quantile(shortage, percentile, axis=1)
            duration_matrix[:, i] = np.quantile(duration, percentile, axis=1)

        # Store the shortage matrix in the dictionary
        shortage_matrix_dict[model][node] = shortage_matrix
        duration_matrix_dict[model][node] = duration_matrix

#%% Plotting

# plot CDF of shortage magnitudes
for node in ['delMontague', 'delTrenton']:
    for metric in ['magnitude', 'duration']:
        
        if metric == 'magnitude':
            matrix_dict = shortage_matrix_dict
            ylabel = 'Max Annual Shortage Magnitude (MG)'
            
        elif metric == 'duration':
            matrix_dict = duration_matrix_dict
            ylabel = 'Max Annual Shortage Duration (days)'

        
        fig, ax = plt.subplots(figsize=(6, 6))

        xs = percentile_range * 100

        # Loop through each model's shortage matrix and plot the CDF
        for model, matrix_node in matrix_dict.items():

            matrix = matrix_node[node]

            n_realizations = matrix.shape[0]
            
            c = 'black' if model == 'obs' else 'blue' if model == 'reconstruction' else 'orange'
            
            for i in range(n_realizations):
                ys = matrix[i, :]
                ax.plot(xs, ys, linestyle='-', 
                        color=c, alpha=0.2)

        ax.set_xlabel('Shortage Percentile')
        ax.set_ylabel(ylabel)
        plt.title(f'{node} Flow Target')
        plt.xlim(0, 100)
        plt.legend()
        plt.savefig(f'{node}_shortage_{metric}_cdf.png', dpi=300, bbox_inches='tight')
        plt.clf()

#%%
# nodes = ['delMontague']

# shortage_event_dict, reliability_dict, resiliency_dict = get_shortfall_metrics(data=data,
#                                                                                nodes=nodes)
# realizations = list(data.major_flow['stationary_ensemble'].keys())

# # Store dictionaries in the data object
# data.shortage = shortage_event_dict
# data.reliability = reliability_dict
# data.resilience = resiliency_dict
# #%% Export data object for later
# fname = './pywrdrb/outputs/stationary_ensemble_with_postprocessing.hdf5'
# data.export(file=fname)