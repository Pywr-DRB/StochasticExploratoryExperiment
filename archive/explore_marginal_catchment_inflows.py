#%% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

import pywrdrb
from pywrdrb.pywr_drb_node_data import immediate_downstream_nodes_dict

from methods.load import load_drb_reconstruction
from config import RECONSTRUCTION_OUTPUT_FNAME, STATIONARY_ENSEMBLE_OUTPUT_FNAME

#%% Load pywrdrb output

output_filenames = [
    RECONSTRUCTION_OUTPUT_FNAME,
    STATIONARY_ENSEMBLE_OUTPUT_FNAME,
    f"./pywrdrb/outputs/nhmv10.hdf5",
    f"./pywrdrb/outputs/nwmv21.hdf5",
    f"./pywrdrb/outputs/nhmv10_withObsScaled.hdf5",
    f"./pywrdrb/outputs/nwmv21_withObsScaled.hdf5",
    f"./pywrdrb/outputs/wrfaorc_calib_nlcd2016.hdf5",
]

flowtypes = ['wrfaorc_calib_nlcd2016',
            'nhmv10', 
            'nwmv21', 
            'nhmv10_withObsScaled', 
            'nwmv21_withObsScaled',
            'pub_nhmv10_BC_withObsScaled']

results_sets = [
    "major_flow",
    "inflow", 
    "catchment_consumption",
    "catchment_withdrawal"
]

# Load the data
Q = load_drb_reconstruction(gage_flow=True)
data = pywrdrb.Data(results_sets=results_sets)
data.load_output(output_filenames=output_filenames)
data.load_hydrologic_model_flow(flowtypes=flowtypes)
data.load_observations()

flowtypes = list(data.major_flow.keys())

#%% Plot marginal catchment inflows vs reservoir inflows

plot_reservoir = ['cannonsville']
plot_flowtype = 'stationary_ensemble'
plot_flowtype = 'pub_nhmv10_BC_withObsScaled'
plot_flowtype = 'nhmv10'
plot_flowtype = 'reconstruction'
plot_flowtype = 'wrfaorc_calib_nlcd2016'

for reservoir in plot_reservoir:
    downstream_gauge = immediate_downstream_nodes_dict[reservoir]
    
    fig, ax = plt.subplots(figsize=(7,7))

    # Plot the reservoir inflow vs  marginal catchment inflows
    # as scatter
    xs = data.inflow[plot_flowtype][0].loc[:, reservoir]
    ys = data.inflow[plot_flowtype][0].loc[:, downstream_gauge]
    
    # Add a 7 day rolling mean
    xs = xs.rolling(window=7).mean().dropna()
    ys = ys.rolling(window=7).mean().dropna()

    frac = ys/xs
    frac = frac[~np.isinf(frac) & ~np.isnan(frac)]
    frac.plot(kind='hist', 
              bins=100, ax=ax, 
              density=True, alpha=0.5, label='Fraction of inflow')

    # Fit KDE to the fraction
    kde = stats.gaussian_kde(frac)
    x = np.linspace(0, 1, 100)
    ax.plot(x, kde(x), label='KDE', color='red')

    # Take random samples from the KDE
    samples = kde.resample(1000)[0]
    ax.hist(samples, bins=50, density=True, alpha=0.5, label='Random samples from KDE')

    # Make a distribution of the fraction
    plt.xlim(0, 0.5)
    plt.show()

#%% Plot 1-1 of x and y
fig, ax = plt.subplots(figsize=(7,7))
xs = data.inflow[plot_flowtype][0].loc[:, reservoir]
ys = data.inflow[plot_flowtype][0].loc[:, downstream_gauge]
ax.plot(xs, ys, 'o', alpha=0.5)
plt.show()
