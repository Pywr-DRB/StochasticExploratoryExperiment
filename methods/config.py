import os
import numpy as np
import pandas as pd
import pywrdrb
from pywrdrb.pywr_drb_node_data import immediate_downstream_nodes_dict

from methods.water_year import count_water_years

# =============================================================================
# CONFIGURATION NAME — determines output directory
# =============================================================================
CONFIG_NAME = "perf_foresight_baseline"

# =============================================================================
# ENSEMBLE CONFIGURATION
# =============================================================================

# Total experiment size
TOTAL_REALIZATIONS = 5
BASELINE_DATASET =  'pub_nhmv10_BC_withObsScaled' # 'wrfaorc_withObsScaled' or 'pub_nhmv10_BC_withObsScaled'

# Ensemble set configuration (for generation and storage)
N_REALIZATIONS_PER_ENSEMBLE_SET = 5  # Memory-manageable chunks
N_ENSEMBLE_SETS = TOTAL_REALIZATIONS // N_REALIZATIONS_PER_ENSEMBLE_SET

# Pywr-DRB simulation batching (within each ensemble set)
N_REALIZATIONS_PER_PYWRDRB_BATCH = 5 # Simulation memory limits
N_PYWRDRB_BATCHES_PER_SET = N_REALIZATIONS_PER_ENSEMBLE_SET // N_REALIZATIONS_PER_PYWRDRB_BATCH

# Temporal configuration
START_DATE = '2030-01-01'
END_DATE = '2100-12-31'

N_YEARS = count_water_years(START_DATE, END_DATE)

# N_YEARS_GENERATE: years of synthetic data to request from Kirsch/Nowak.
# Must cover the full simulation period through END_DATE. count_water_years()
# excludes partial water years at the end (e.g. WY2100 = Jun–Dec 2100 only has
# 214 days, below the 300-day threshold), so N_YEARS alone generates data that
# can fall ~1 year short of END_DATE. Add 2 years of buffer to ensure coverage.
N_YEARS_GENERATE = N_YEARS + 2

# Reconstruction simulation date range and year count
_reconstruction_dates = pywrdrb.utils.dates.model_date_ranges[BASELINE_DATASET]
RECONSTRUCTION_START_DATE = _reconstruction_dates[0]
RECONSTRUCTION_END_DATE = _reconstruction_dates[1]
RECONSTRUCTION_N_YEARS = count_water_years(RECONSTRUCTION_START_DATE, RECONSTRUCTION_END_DATE)

# Period origin for analysis
# 'jan1' = calendar year (Jan 1 - Dec 31), aligns with FFMP boundaries
# 'june1' = water year (Jun 1 - May 31)
PERIOD_ORIGIN = 'june1'

# SSI (Standardized Streamflow Index) window sizes in months
SSI_WINDOWS = (3, 6, 12)

# SSI target node: which flow node is used for drought identification.
# 'nyc_aggregate' = sum of cannonsville + pepacton + neversink catchment inflows
# 'delMontague' = full natural flow at Montague gage
SSI_NODE = 'nyc_aggregate'

SSI_NODE_CONFIGS = {
    'nyc_aggregate': {
        'historical_gage_flow': False,
        'derived': True,
        'derive_from': ['cannonsville', 'pepacton', 'neversink'],
        'results_set': 'inflow',
        'drop_columns': ['delTrenton'],
    },
    'delMontague': {
        'historical_gage_flow': True,
        'derived': False,
        'derive_from': None,
        'results_set': 'major_flow',
        'drop_columns': [],
    },
}

assert SSI_NODE in SSI_NODE_CONFIGS, \
    f"SSI_NODE '{SSI_NODE}' not in SSI_NODE_CONFIGS. Must be one of {list(SSI_NODE_CONFIGS.keys())}"

# Validation checks
assert TOTAL_REALIZATIONS % N_REALIZATIONS_PER_ENSEMBLE_SET == 0, \
    "TOTAL_REALIZATIONS must be divisible by N_REALIZATIONS_PER_ENSEMBLE_SET"
assert N_REALIZATIONS_PER_ENSEMBLE_SET % N_REALIZATIONS_PER_PYWRDRB_BATCH == 0, \
    "N_REALIZATIONS_PER_ENSEMBLE_SET must be divisible by N_REALIZATIONS_PER_PYWRDRB_BATCH"

# =============================================================================
# DATASET CONFIGURATIONS
# =============================================================================


# Load monthly shift data for climate adjustments
fname = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "nyc_inflow_selected_scenarios_PRMS_2020_2059.csv")
monthly_shift_scenarios = pd.read_csv(fname, index_col=0)

DATASET_CONFIGS = {
    'stationary_ensemble': {
        'type': 'stationary',
        'description': 'Stationary ensemble (no climate adjustment)',
        'monthly_prc_change': None
    },
    'climate_adjusted_low': {
        'type': 'climate_adjusted',
        'description': 'Driest climate change',
        'monthly_prc_change': monthly_shift_scenarios.loc[:, 'low'].values
    },
    # 'climate_adjusted_medium': {
    #     'type': 'climate_adjusted', 
    #     'description': 'Mid-range climate change',
    #     'monthly_prc_change': monthly_shift_scenarios.loc[:, 'medium'].values
    # },
    'climate_adjusted_high': {
        'type': 'climate_adjusted',
        'description': 'Wettest climate change',
        'monthly_prc_change': monthly_shift_scenarios.loc[:, 'high'].values
    },
}


# =============================================================================
# Salinity LSTM model settings
# =============================================================================

pywrdrb_ml_plugin_path = os.path.abspath(f"{os.path.dirname(__file__)}/../PywrDRB-ML/")
pywrdrb_salinity_model_path = os.path.abspath(f"{pywrdrb_ml_plugin_path}/models/SalinityLSTM/SalinityLSTM.yml")

SALINITY_LSTM_OPTIONS = {
    "ml_model_type": "lstm",
    "PywrDRB_ML_plugin_path": pywrdrb_ml_plugin_path,
    "model_salinity": pywrdrb_salinity_model_path,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "Q_Trenton_lstm_var_name": "Q_Trenton_bc",
    "Q_Schuylkill_lstm_var_name": "Q_Schuylkill_bc",
    "asycronized_update": False,
    "debug": True
}

SALINITY_LSTM_PREDICTIONS = False


# =============================================================================
# MODELBUILDER OPTIONS
# =============================================================================

FLOW_PREDICTION_MODE = 'perfect_foresight'

# =============================================================================
# FILE STRUCTURE AND PATHS
# =============================================================================

# Root directory (repo root, 1 level above this config file)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Shared inputs (independent of config)
ENSEMBLE_BASE_DIR = os.path.abspath(f"{ROOT_DIR}/pywrdrb/inputs/")

# Config-specific output root
CONFIG_DIR = os.path.abspath(f"{ROOT_DIR}/outputs/{CONFIG_NAME}")

# Data directories (under CONFIG_DIR/data/)
OUTPUT_DIR              = os.path.abspath(f"{CONFIG_DIR}/data/simulations/")
MODEL_DIR               = os.path.abspath(f"{CONFIG_DIR}/data/models/")
DROUGHT_METRICS_DIR     = os.path.abspath(f"{CONFIG_DIR}/data/drought_metrics/")
PERFORMANCE_METRICS_DIR = os.path.abspath(f"{CONFIG_DIR}/data/performance_metrics/")
EVENT_METRICS_DIR       = os.path.abspath(f"{CONFIG_DIR}/data/event_metrics/")
ZONE_PROB_DIR           = os.path.abspath(f"{CONFIG_DIR}/data/zone_probabilities/")
SATISFICING_DIR         = os.path.abspath(f"{CONFIG_DIR}/data/satisficing/")
FOCAL_EVENTS_DIR        = os.path.abspath(f"{CONFIG_DIR}/data/focal_events/")

# Figures directory
FIG_DIR = os.path.abspath(f"{CONFIG_DIR}/figures/")

# =============================================================================
# PYWR-DRB CONFIGURATION
# =============================================================================

# Setup pathnavigator for Pywr-DRB
pn_config = pywrdrb.get_pn_config()
for dataset_id in DATASET_CONFIGS.keys():
    pn_config[f"flows/{dataset_id}"] = os.path.abspath(f"{ENSEMBLE_BASE_DIR}/{dataset_id}/")

# Node information
pywrdrb_nodes = list(immediate_downstream_nodes_dict.keys())

# Nodes to generate using Kirsch-Nowak
pywrdrb_nodes_to_generate = [n for n in pywrdrb_nodes if n[0] != '0']
if 'delTrenton' in pywrdrb_nodes_to_generate:
    pywrdrb_nodes_to_generate.remove('delTrenton')

# Nodes to generate using regression
pywrdrb_nodes_to_regress = [n for n in pywrdrb_nodes if n[0] == '0']

# Results sets to save (memory optimization)
SAVE_RESULTS_SETS = [
    "major_flow", 
    "inflow", 
    "res_storage",
    "lower_basin_mrf_contributions", 
    "mrf_target", 
    "ibt_diversions", 
    "ibt_demands",
    "nyc_release_components",
    "res_level"
]

# Output files
RECONSTRUCTION_OUTPUT_FNAME = f"{OUTPUT_DIR}/reconstruction.hdf5"
WRFAORC_OUTPUT_FNAME = f"{OUTPUT_DIR}/wrfaorc_withObsScaled.hdf5"
WRF1960s_OUTPUT_FNAME = f"{OUTPUT_DIR}/wrf1960s_calib_nlcd2016.hdf5"

# =============================================================================
# NYC RESERVOIR STORAGE CAPACITIES
# =============================================================================

NYC_RESERVOIRS = ['cannonsville', 'pepacton', 'neversink']

# Storage capacities for NYC reservoirs (million gallons)
NYC_STORAGE_CAPACITIES = {
    'cannonsville': 95700,
    'pepacton': 140200,
    'neversink': 34900
}

# Total NYC reservoir storage capacity (million gallons)
NYC_TOTAL_CAPACITY = sum(NYC_STORAGE_CAPACITIES.values())  # 270,800 MG

# Default shortage tolerance (MGD). Shortages below this magnitude are treated
# as zero to filter out numerical noise / trivially small deficits.
DEFAULT_SHORTAGE_TOLERANCE_MGD = 1.0

# =============================================================================
# VALIDATION (runs at import time)
# =============================================================================

def verify_dataset_id(dataset_id):
    """Verify that a dataset_id is valid."""
    if dataset_id not in DATASET_CONFIGS:
        raise ValueError(f"Invalid dataset_id '{dataset_id}'. Must be one of: {list(DATASET_CONFIGS.keys())}")
    return True


def validate_configuration():
    """Validate the configuration parameters"""
    errors = []
    
    if TOTAL_REALIZATIONS <= 0:
        errors.append("TOTAL_REALIZATIONS must be positive")
    
    if N_REALIZATIONS_PER_ENSEMBLE_SET <= 0:
        errors.append("N_REALIZATIONS_PER_ENSEMBLE_SET must be positive")
    
    if N_REALIZATIONS_PER_PYWRDRB_BATCH <= 0:
        errors.append("N_REALIZATIONS_PER_PYWRDRB_BATCH must be positive")
    
    if N_REALIZATIONS_PER_PYWRDRB_BATCH > N_REALIZATIONS_PER_ENSEMBLE_SET:
        errors.append("N_REALIZATIONS_PER_PYWRDRB_BATCH cannot exceed N_REALIZATIONS_PER_ENSEMBLE_SET")
    
    if N_YEARS <= 0:
        errors.append("N_YEARS must be positive")
    
    # Validate dataset configs
    for dataset_id, config in DATASET_CONFIGS.items():
        if 'type' not in config:
            errors.append(f"Dataset {dataset_id} missing 'type' field")
        elif config['type'] not in ['stationary', 'climate_adjusted']:
            errors.append(f"Dataset {dataset_id} has invalid type: {config['type']}")
        
        if config['type'] == 'climate_adjusted' and config.get('monthly_prc_change') is not None:
            prc_change = config['monthly_prc_change']
            if not isinstance(prc_change, (list, np.ndarray)) or len(prc_change) != 12:
                errors.append(f"Dataset {dataset_id} monthly_prc_change must be 12-element array")
    
    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))
    
    return True

# Validate configuration on import
validate_configuration()