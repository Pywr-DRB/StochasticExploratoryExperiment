#%% 
import pywrdrb

from methods.metrics.shortfall import get_shortfall_metrics, calculate_hashimoto_metrics
from config import RECONSTRUCTION_OUTPUT_FNAME, STATIONARY_ENSEMBLE_OUTPUT_FNAME

#%% Load pywrdrb output

output_filenames = [
    RECONSTRUCTION_OUTPUT_FNAME,
    STATIONARY_ENSEMBLE_OUTPUT_FNAME    
]

results_sets = [
    "major_flow", "inflow", "res_storage",
    "lower_basin_mrf_contributions", "mrf_target", "ibt_diversions", "ibt_demands",
    ]

# Load the data
data = pywrdrb.Data(results_sets=results_sets)
data.load_output(output_filenames=output_filenames)
data.load_observations()

flowtypes = list(data.major_flow.keys())


#%%

nodes = ['delMontague']

shortage_event_dict, reliability_dict, resiliency_dict = get_shortfall_metrics(data=data,
                                                                               nodes=nodes)
realizations = list(data.major_flow['stationary_ensemble'].keys())

# Store dictionaries in the data object
data.shortage = shortage_event_dict
data.reliability = reliability_dict
data.resilience = resiliency_dict

#%% Export data object for later

fname = './pywrdrb/outputs/stationary_ensemble_with_postprocessing.hdf5'
data.export(file=fname)
