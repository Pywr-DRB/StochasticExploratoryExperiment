#%% 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pywrdrb

from config import RECONSTRUCTION_OUTPUT_FNAME, STATIONARY_ENSEMBLE_OUTPUT_FNAME

#%% Load pywrdrb output

output_filenames = [
    RECONSTRUCTION_OUTPUT_FNAME,
    STATIONARY_ENSEMBLE_OUTPUT_FNAME    
]

results_sets = [
    "major_flow",
    "inflow", 
    "catchment_consumption",
    "catchment_withdrawal"
]

# Load the data
data = pywrdrb.Data(results_sets=results_sets)
data.load_output(output_filenames=output_filenames)
data.load_observations()

flowtypes = list(data.major_flow.keys())

#%% 

# Make a plot of catchment withdrawals vs inflows
fig, ax = plt.subplots(figsize=(10, 6))
for flow in flowtypes:
    if flow == 'obs':
        continue
    realizations = list(data.catchment_withdrawal[flow].keys())
    for i in realizations:
        x = data.catchment_withdrawal[flow][i].loc[:, 'delMontague'].values
        y = data.inflow[flow][i].loc[:, 'delMontague'].values
        ax.scatter(x, y, label=f"{flow} - realization {i}", alpha=0.5)
ax.set_xlabel("Withdrawal (mgd)")
ax.set_ylabel("Inflow (mgd)")
ax.set_ylim(0, 100)
ax.set_xlim(0, 100)
plt.show()


# %%
