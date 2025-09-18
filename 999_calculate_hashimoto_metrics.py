### 
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pywrdrb



from config import verify_ensemble_type
from methods.metrics.shortfall import annual_max_positive_streak
from config import FIG_DIR

### Load pywrdrb output

ensemble_annual_shortage_matrix_dict = {}
ensemble_daily_shortage_matrix_dict = {}
ensemble_duration_matrix_dict = {}
    
    
# Get ensemble type from command line arguments
for ensemble_type in ['stationary', 'climate_adjusted']:
    inflow_type = f'{ensemble_type}_ensemble'
    verify_ensemble_type(ensemble_type)

    # Load ensemble data from processed HDF5 - should have everything we need inside
    fname = f'./pywrdrb/outputs/{ensemble_type}_ensemble_with_postprocessing.hdf5'
    data = pywrdrb.Data()
    data.load_from_export(fname)


    ### Calculate shortage percentiles

    print('Calculating shortage percentiles...')
    # Loop through models and create shortage matrices for each 
    annual_shortage_matrix_dict = {}
    daily_shortage_matrix_dict = {}
    duration_matrix_dict = {}
    for model in ['reconstruction', inflow_type]:
        
        annual_shortage_matrix_dict[model] = {}
        daily_shortage_matrix_dict[model] = {}
        duration_matrix_dict[model] = {}
        
        for node in ['delMontague', 'delTrenton', 'nyc']:
            
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
                        
                # Shortage timeseries was calculated in 05_postprocess_data.py
                # and is available in the data object
                shortage_series = data.shortage[model][r].loc[:, node]
                
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

    ensemble_annual_shortage_matrix_dict[ensemble_type] = annual_shortage_matrix_dict
    ensemble_daily_shortage_matrix_dict[ensemble_type] = daily_shortage_matrix_dict
    ensemble_duration_matrix_dict[ensemble_type] = duration_matrix_dict

### Plotting
### BAR CHARTS ###


###
### CDFS ###


# Define percentile bands
percentile_bands = [
    (0, 10, 0.15),    # 0-10%, alpha=0.15
    (10, 25, 0.25),   # 10-25%, alpha=0.25
    (25, 50, 0.4),    # 25-50%, alpha=0.4
    (50, 75, 0.25),   # 50-75%, alpha=0.25  
    (75, 90, 0.25),   # 75-90%, alpha=0.25
    (90, 100, 0.15)   # 90-100%, alpha=0.15
]

# plot CDF of shortage magnitudes with percentile bands
for timescale in ['annual', 'daily']:
    for node in ['delMontague', 'delTrenton', 'nyc']:
        for metric in ['magnitude', 'duration']:
            
            print(f'Plotting {node} {timescale} {metric} shortage CDF...')
            if metric == 'magnitude':
                if timescale == 'daily':
                    ensemble_matrix_dict = ensemble_daily_shortage_matrix_dict
                    ylabel = 'Daily Shortage Magnitude (MG)'
                elif timescale == 'annual':
                    ensemble_matrix_dict = ensemble_annual_shortage_matrix_dict
                    ylabel = 'Total Annual Shortage Magnitude (MG)'
                
            elif metric == 'duration':
                ensemble_matrix_dict = ensemble_duration_matrix_dict
                ylabel = 'Total Annual Shortage Duration (days)'
            
            
            fig, ax = plt.subplots(figsize=(6, 6))
            xs = percentile_range * 100
            
            for ensemble_type in ['stationary', 'climate_adjusted']:
                
                matrix_dict = ensemble_matrix_dict[ensemble_type]

                # Plot percentile bands for each model
                for model, matrix_node in matrix_dict.items():
                    matrix = matrix_node[node]
                    
                    
                    # Only plot reconstruction once 
                    if model == 'reconstruction' and ensemble_type == 'stationary':
                        continue

                    if model == 'obs':
                        c = 'black'
                    elif model == 'reconstruction':
                        c = 'blue'
                    elif ensemble_type == 'stationary':
                        c = 'orange'
                    elif ensemble_type == 'climate_adjusted':
                        c = 'purple'
                                        
                    # Calculate percentiles across realizations for each percentile point
                    for lower_pct, upper_pct, alpha in percentile_bands:
                        # For each x-value (shortage percentile), calculate the percentile bounds across realizations
                        lower_bound = np.percentile(matrix, lower_pct, axis=0)
                        upper_bound = np.percentile(matrix, upper_pct, axis=0)
                        
                        ax.fill_between(xs, lower_bound, upper_bound, 
                                    color=c, alpha=alpha)

                    # Add median line for emphasis
                    median = np.percentile(matrix, 50, axis=0)
                    ax.plot(xs, median, color=c, linewidth=2, 
                        label=f'{model} median')
                
            ax.set_xlabel('Shortage Percentile')
            ax.set_ylabel(ylabel)
            plt.title(f'{node} Flow Target')
            plt.xlim(0, 100)
            plt.legend()
            plt.savefig(f'{FIG_DIR}/shortages/{node}_{timescale}_shortage_{metric}_cdf.png', dpi=300, bbox_inches='tight')
            plt.clf()