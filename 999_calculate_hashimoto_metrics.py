#%% 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pywrdrb

from methods.utils import combine_multiple_ensemble_sets_in_data
from methods.metrics.shortfall import get_flow_and_target_values, add_trenton_equiv_flow
from methods.metrics.shortfall import annual_max_positive_streak
from config import RECONSTRUCTION_OUTPUT_FNAME, FIG_DIR
from config import STATIONARY_ENSEMBLE_SETS, CLIMATE_ADJUSTED_ENSEMBLE_SETS

#%% Load pywrdrb output
ensemble_type =  'stationary' # 'climate_adjusted'  # or 'stationary'
inflow_type = f'{ensemble_type}_ensemble'

if ensemble_type == 'stationary':
    ensemble_set_spec_list = STATIONARY_ENSEMBLE_SETS
elif ensemble_type == 'climate_adjusted':
    ensemble_set_spec_list = CLIMATE_ADJUSTED_ENSEMBLE_SETS

output_filenames = [ensemble_set_spec_list[i].output_file for i in range(len(ensemble_set_spec_list))]
output_filenames.append(RECONSTRUCTION_OUTPUT_FNAME)

results_sets = [
    "major_flow", "inflow", "res_storage",
    "lower_basin_mrf_contributions", "mrf_target", 
    "ibt_diversions", "ibt_demands",
]

# Load the data
data = pywrdrb.Data(results_sets=results_sets, print_status=True)
data.load_output(output_filenames=output_filenames)
data.load_observations()

data = add_trenton_equiv_flow(data)

data = combine_multiple_ensemble_sets_in_data(data, results_sets, ensemble_type=ensemble_type)



#%% Calculate shortage percentiles

print('Calculating shortage percentiles...')
# Loop through models and create shortage matrices for each 
annual_shortage_matrix_dict = {}
daily_shortage_matrix_dict = {}
duration_matrix_dict = {}
for model in ['reconstruction', inflow_type]:
    
    annual_shortage_matrix_dict[model] = {}
    daily_shortage_matrix_dict[model] = {}
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

        annual_shortage = []
        daily_shortage = []
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

            annual_shortage.append(annual_shortage_series.values)
            daily_shortage.append(shortage_series.values)
            duration.append(max_shortage_duration.values)

        # Convert to numpy arrays
        # Shortage is a matrix with shape (n_realizations, n_periods)
        annual_shortage = np.array(annual_shortage)
        daily_shortage = np.array(daily_shortage)
        duration = np.array(duration)

        n_realizations = annual_shortage.shape[0]
        percentile_range = np.linspace(0, 1, 100)

        # Based on simulated flows and targets for a given node and model, 
        # this function calculates shortage across different metrics. 
        # the annual_shortage_matrix should be 2D with shape (n_realizations, n_percentiles)
        # where each cell contains the shortage magnitude for that realization and percentile.
        annual_shortage_matrix = np.zeros((n_realizations, len(percentile_range)))
        daily_shortage_matrix = np.zeros((n_realizations, len(percentile_range)))
        duration_matrix = np.zeros((n_realizations, len(percentile_range)))

        for i, percentile in enumerate(percentile_range):
            # Calculate the shortage for each realization at this percentile
            annual_shortage_matrix[:, i] = np.quantile(annual_shortage, percentile, axis=1)
            daily_shortage_matrix[:, i] = np.quantile(daily_shortage, percentile, axis=1)
            duration_matrix[:, i] = np.quantile(duration, percentile, axis=1)

        # Store the shortage matrix in the dictionary
        annual_shortage_matrix_dict[model][node] = annual_shortage_matrix
        daily_shortage_matrix_dict[model][node] = daily_shortage_matrix
        duration_matrix_dict[model][node] = duration_matrix

#%% Plotting

# plot CDF of shortage magnitudes
for timescale in ['annual', 'daily']:
    for node in ['delMontague', 'delTrenton']:
        for metric in ['magnitude', 'duration']:
            
            print(f'Plotting {node} {timescale} {metric} shortage CDF...')
            
            if metric == 'magnitude':
                if timescale == 'daily':
                    matrix_dict = daily_shortage_matrix_dict
                    ylabel = 'Daily Shortage Magnitude (MG)'
                elif timescale == 'annual':
                    matrix_dict = annual_shortage_matrix_dict
                    ylabel = 'Total Annual Shortage Magnitude (MG)'
                
            elif metric == 'duration':
                matrix_dict = duration_matrix_dict
                ylabel = 'Total Annual Shortage Duration (days)'

            
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
            plt.savefig(f'{FIG_DIR}/shortages/{inflow_type}_{node}_{timescale}_shortage_{metric}_cdf.png', dpi=300, bbox_inches='tight')
            plt.clf()