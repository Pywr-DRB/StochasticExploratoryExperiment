#%%
import numpy as np
import matplotlib.pyplot as plt
from methods.load import load_drb_reconstruction, load_and_combine_ensemble_sets
from config import STATIONARY_ENSEMBLE_SETS


### Loading data
## Historic reconstruction data
# Total flow
Q = load_drb_reconstruction()
Q.replace(0, np.nan, inplace=True)
Q.drop(columns=['delTrenton'], inplace=True)  # Remove Trenton gage as it is not used in the ensemble
Q_monthly = Q.resample('MS').sum()

# Catchment inflows
Q_inflows = load_drb_reconstruction(gage_flow=False)
Q_inflows.replace(0, np.nan, inplace=True)
Q_inflows.drop(columns=['delTrenton'], inplace=True)  # Remove Trenton gage as it is not used in the ensemble

print(f"Loaded reconstruction data with {Q.shape[0]// 365} years of daily data for {Q.shape[1]} sites.")


## Synthetic ensemble
Q_syn = load_and_combine_ensemble_sets(STATIONARY_ENSEMBLE_SETS, by_site=True)
syn_ensemble = load_and_combine_ensemble_sets(STATIONARY_ENSEMBLE_SETS, by_site=False)

realization_ids = list(syn_ensemble.keys())
n_realizations = len(realization_ids)

#%% Convergence metrics with resampling
site = 'delMontague'
n_bootstrap_samples = 50  # Number of bootstrap samples for range estimation

# Initialize arrays to store results
n_realization_subset = range(1, n_realizations + 1, 20) 
mean_annual_ranges = []
var_annual_ranges = []

# Pre-calculate annual sums for all realizations to avoid repeated computation
annual_sums = Q_syn[site].resample('A').sum()

for n_real in n_realization_subset:
    # Bootstrap sampling for current number of realizations
    mean_bootstrap = []
    var_bootstrap = []
    
    for _ in range(n_bootstrap_samples):
        # Randomly sample n_real realizations from all available realizations
        sampled_realizations = np.random.choice(realization_ids, size=n_real, replace=False)
        
        # Calculate metrics for this bootstrap sample
        sample_data = annual_sums.loc[:, sampled_realizations]
        mean_bootstrap.append(sample_data.mean().mean())
        var_bootstrap.append(sample_data.var().mean())
    
    # Store min/max ranges for this number of realizations
    mean_annual_ranges.append([np.min(mean_bootstrap), np.max(mean_bootstrap)])
    var_annual_ranges.append([np.min(var_bootstrap), np.max(var_bootstrap)])

# Convert to arrays for easier plotting
mean_annual_ranges = np.array(mean_annual_ranges)
var_annual_ranges = np.array(var_annual_ranges)

#%% Plot convergence metrics with uncertainty ranges


# Mean annual flow convergence
plt.figure(figsize=(7, 7))
plt.fill_between(n_realization_subset, 
                 mean_annual_ranges[:, 0], 
                 mean_annual_ranges[:, 1], 
                 alpha=0.3, label='Range (min-max)')
plt.plot(n_realization_subset, 
         (mean_annual_ranges[:, 0] + mean_annual_ranges[:, 1]) / 2, 
         linestyle='-', linewidth=2, label='Mean')
plt.title(f'Convergence of Mean Annual Flow for {site}')
plt.xlabel('Number of Realizations')
plt.ylabel('Mean Annual Flow (MG)')
plt.yscale('log')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(f'convergence_mean_annual_flow_{site}_ranges.png', dpi=300)
plt.show()

# Variance of annual flow convergence
plt.figure(figsize=(7, 7))
plt.fill_between(n_realization_subset, 
                 var_annual_ranges[:, 0], 
                 var_annual_ranges[:, 1], 
                 alpha=0.3, label='Range (min-max)')
plt.plot(n_realization_subset, 
         (var_annual_ranges[:, 0] + var_annual_ranges[:, 1]) / 2, 
         linestyle='-', linewidth=2, label='Mean')
plt.title(f'Convergence of Variance of Annual Flow for {site}')
plt.xlabel('Number of Realizations')
plt.ylabel('Variance of Annual Flow (MG^2)')
plt.yscale('log')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(f'convergence_variance_annual_flow_{site}_ranges.png', dpi=300)
plt.show()