import os
import numpy as np
import pandas as pd
import pywrdrb
from pywrdrb.pywr_drb_node_data import immediate_downstream_nodes_dict

from methods.verification import verify_dataset_id

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
N_YEARS = 70
START_DATE = '2030-01-01'
END_DATE = '2099-12-31'

# Period origin for analysis
# 'jan1' = calendar year (Jan 1 - Dec 31), aligns with FFMP boundaries
# 'june1' = water year (Jun 1 - May 31)
PERIOD_ORIGIN = 'june1'

# SSI (Standardized Streamflow Index) window sizes in months
SSI_WINDOWS = (3, 6, 12)

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
# WORKFLOW CONTROL
# =============================================================================

class WorkflowFlags:
    """Control which steps of the workflow to run"""
    RUN_BASELINE = False
    GENERATE_ENSEMBLE_SETS = True
    PREP_PYWRDRB = True
    RUN_PYWRDRB = True
    PLOT_DIAGNOSTICS = False
    PLOT_OUTCOMES = False
    
    # Processing options
    PROCESS_ALL_SETS = True  # Process all sets or specify subset
    TARGET_ENSEMBLE_SETS = None  # None for all, or list of set IDs
    
    # Cleanup options
    CLEANUP_PYWRDRB_BATCH_FILES = True  # Remove batch files after combining within set
    CLEANUP_TEMP_FILES = True

# =============================================================================
# FILE STRUCTURE AND PATHS
# =============================================================================

# root dir is 1 level above this config file
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.abspath(f"{ROOT_DIR}/pywrdrb/outputs/")
from datetime import datetime as _dt
FIG_DIR = os.path.abspath(f"{ROOT_DIR}/figures_{_dt.now().strftime('%m%d%Y')}/")

# Base ensemble directory
ENSEMBLE_BASE_DIR = os.path.abspath(f"{ROOT_DIR}/pywrdrb/inputs/")

# =============================================================================
# ENSEMBLE SET SPECIFICATIONS
# =============================================================================

class EnsembleSetSpec:
    """Specification for a single ensemble set"""
    
    def __init__(self, set_id, dataset_id):
        # Validate dataset_id
        if dataset_id not in DATASET_CONFIGS:
            raise ValueError(f"Invalid dataset_id: {dataset_id}. Must be one of {list(DATASET_CONFIGS.keys())}")
        
        self.dataset_id = dataset_id
        self.dataset_config = DATASET_CONFIGS[dataset_id]
        self.ensemble_type = self.dataset_config['type']  # 'stationary' or 'climate_adjusted'
        self.set_id = set_id
        self.start_realization = set_id * N_REALIZATIONS_PER_ENSEMBLE_SET
        self.end_realization = (set_id + 1) * N_REALIZATIONS_PER_ENSEMBLE_SET
        self.n_realizations = N_REALIZATIONS_PER_ENSEMBLE_SET
        self.realizations = self.get_realization_ids()
        self.realization_ids = self.realizations
        
        # Pywr-DRB batching within this set
        self.pywrdrb_batches = self._create_pywrdrb_batch_specs()
    
    @property
    def directory(self):
        """Get directory path for this ensemble set"""
        return f"{ENSEMBLE_BASE_DIR}/{self.dataset_id}/{self.dataset_id}_set{self.set_id + 1}"
    
    @property
    def files(self):
        """Get file paths for this ensemble set"""
        set_dir = self.directory
        return {
            'gage_flow': f"{set_dir}/gage_flow_mgd.hdf5",
            'catchment_inflow': f"{set_dir}/catchment_inflow_mgd.hdf5",
            'predicted_inflow': f"{set_dir}/predicted_inflows_mgd.hdf5",
            'diversion_nyc': f"{set_dir}/diversion_nyc_extrapolated_mgd.hdf5",
            'diversion_nj': f"{set_dir}/diversion_nj_extrapolated_mgd.hdf5",
            'predicted_diversions': f"{set_dir}/predicted_diversions_mgd.hdf5"
        }
    
    @property
    def output_file(self):
        """Get output filename for this ensemble set"""
        return f"{OUTPUT_DIR}/{self.dataset_id}_set{self.set_id + 1}.hdf5"
    
    def _create_pywrdrb_batch_specs(self):
        """Create Pywr-DRB batch specifications within this ensemble set"""
        batches = []
        for batch_id in range(N_PYWRDRB_BATCHES_PER_SET):
            batch_start = batch_id * N_REALIZATIONS_PER_PYWRDRB_BATCH
            batch_end = (batch_id + 1) * N_REALIZATIONS_PER_PYWRDRB_BATCH
            
            # Global realization IDs
            global_start = self.start_realization + batch_start
            global_end = self.start_realization + batch_end
            
            # Local realization IDs within this set (0-based)
            local_ids = list(range(batch_start, batch_end))
            
            batches.append({
                'batch_id': batch_id,
                'dataset_id': self.dataset_id,
                'set_id': self.set_id,
                'local_start': batch_start,
                'local_end': batch_end,
                'global_start': global_start,
                'global_end': global_end,
                'local_realization_ids': local_ids,
                'n_realizations': N_REALIZATIONS_PER_PYWRDRB_BATCH
            })
        
        return batches
    
    def get_realization_ids(self):
        """Get list of global realization IDs for this set"""
        return list(range(self.start_realization, self.end_realization))
    

# Create ensemble set specifications for all datasets
ENSEMBLE_SETS = {
    dataset_id: [EnsembleSetSpec(i, dataset_id) for i in range(N_ENSEMBLE_SETS)]
    for dataset_id in DATASET_CONFIGS.keys()
}

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
    'cannonsville': 95706,
    'pepacton': 140190,
    'neversink': 34941
}

# Total NYC reservoir storage capacity (million gallons)
NYC_TOTAL_CAPACITY = sum(NYC_STORAGE_CAPACITIES.values())  # 270,837 MG

# Default shortage tolerance (MGD). Shortages below this magnitude are treated
# as zero to filter out numerical noise / trivially small deficits.
DEFAULT_SHORTAGE_TOLERANCE_MGD = 1.0

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================



def get_ensemble_set_spec(set_id, dataset_id):
    """Get ensemble set specification by ID and dataset"""
    if dataset_id not in ENSEMBLE_SETS:
        raise ValueError(f"Invalid dataset_id: {dataset_id}")
    if set_id < 0 or set_id >= N_ENSEMBLE_SETS:
        raise ValueError(f"set_id must be between 0 and {N_ENSEMBLE_SETS-1}")
    return ENSEMBLE_SETS[dataset_id][set_id]

def ensure_ensemble_set_dirs(dataset_id=None):
    """Create all necessary ensemble set directories"""
    os.makedirs(ENSEMBLE_BASE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    
    # If dataset_id specified, only create dirs for that dataset
    if dataset_id:
        dataset_ids = [dataset_id]
    else:
        dataset_ids = DATASET_CONFIGS.keys()
    
    for did in dataset_ids:
        # Create dataset directory
        os.makedirs(f"{ENSEMBLE_BASE_DIR}/{did}", exist_ok=True)
        
        # Create directories for each ensemble set
        for ensemble_set in ENSEMBLE_SETS[did]:
            os.makedirs(ensemble_set.directory, exist_ok=True)

def get_existing_ensemble_sets(dataset_id):
    """Get list of ensemble set specs that have been generated for a dataset"""
    verify_dataset_id(dataset_id)
    existing_sets = []
    for spec in ENSEMBLE_SETS[dataset_id]:
        # Check if both required files exist
        if (os.path.exists(spec.files['gage_flow']) and 
            os.path.exists(spec.files['catchment_inflow'])):
            existing_sets.append(spec)
    return existing_sets

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