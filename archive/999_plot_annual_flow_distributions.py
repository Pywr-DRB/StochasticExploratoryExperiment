import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import seaborn as sns
import matplotlib.pyplot as plt

from methods.load import load_and_combine_ensemble_sets
from config import STATIONARY_ENSEMBLE_SETS, CLIMATE_ADJUSTED_ENSEMBLE_SETS
from config import FIG_DIR

### Load data
stationary_ensemble = load_and_combine_ensemble_sets(STATIONARY_ENSEMBLE_SETS, by_site=True)
climate_adjusted_ensemble = load_and_combine_ensemble_sets(CLIMATE_ADJUSTED_ENSEMBLE_SETS, by_site=True)


### Calculate annual flows
site = 'delMontague'  
stationary_annual_flows = stationary_ensemble[site].resample('A').sum()
climate_adjusted_annual_flows = climate_adjusted_ensemble[site].resample('A').sum()

# Drop everything after 2019-12-31
stationary_annual_flows = stationary_annual_flows[stationary_annual_flows.index <= '2019-12-31']
climate_adjusted_annual_flows = climate_adjusted_annual_flows[climate_adjusted_annual_flows.index <= '2019-12-31']


### Plotting 
# KDE of annual flows using seaborn

plt.figure(figsize=(6, 6))
sns.kdeplot(stationary_annual_flows.values.flatten(), label='Stationary Ensemble', color='blue', fill=True, alpha=0.5)
sns.kdeplot(climate_adjusted_annual_flows.values.flatten(), label='Climate Adjusted Ensemble', color='orange', fill=True, alpha=0.5)
plt.title(f'Annual Flow Distributions for {site}')
plt.xlabel('Annual Flow (cfs)')
plt.ylabel('Density')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/annual_flow_distributions_{site}.png')