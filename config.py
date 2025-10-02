import os
import numpy as np
import pandas as pd
import pywrdrb
from pywrdrb.pywr_drb_node_data import immediate_downstream_nodes_dict
from pywrdrb.utils.hdf5 import get_hdf5_realization_numbers

# =============================================================================
# ENSEMBLE CONFIGURATION
# =============================================================================

# Total experiment size
TOTAL_REALIZATIONS = 1000

# Ensemble set configuration (for generation and storage)
N_REALIZATIONS_PER_ENSEMBLE_SET = 100  # Memory-manageable chunks
N_ENSEMBLE_SETS = TOTAL_REALIZATIONS // N_REALIZATIONS_PER_ENSEMBLE_SET

# Pywr-DRB simulation batching (within each ensemble set)
N_REALIZATIONS_PER_PYWRDRB_BATCH = 10  # Simulation memory limits
N_PYWRDRB_BATCHES_PER_SET = N_REALIZATIONS_PER_ENSEMBLE_SET // N_REALIZATIONS_PER_PYWRDRB_BATCH

# Temporal configuration
N_YEARS = 70
START_DATE = '1950-01-01'
END_DATE = '2019-12-31'

# Validation checks
assert TOTAL_REALIZATIONS % N_REALIZATIONS_PER_ENSEMBLE_SET == 0, \
    "TOTAL_REALIZATIONS must be divisible by N_REALIZATIONS_PER_ENSEMBLE_SET"
assert N_REALIZATIONS_PER_ENSEMBLE_SET % N_REALIZATIONS_PER_PYWRDRB_BATCH == 0, \
    "N_REALIZATIONS_PER_ENSEMBLE_SET must be divisible by N_REALIZATIONS_PER_PYWRDRB_BATCH"

# =============================================================================
# DATASET CONFIGURATIONS
# =============================================================================

# Load monthly shift data for climate adjustments
fname = "./data/summary_nyc_inflow_monthly_mean_prc_change_ssp245_2020_2059.csv"
monthly_shift_range = pd.read_csv(fname, index_col=0)

# Define all dataset configurations


DATASET_CONFIGS = {
    'stationary_ensemble': {
        'type': 'stationary',
        'description': 'Stationary ensemble (no climate adjustment)',
        'monthly_prc_change': None
    },
    'climate_adjusted_ssp245_min': {
        'type': 'climate_adjusted',
        'description': 'SSP2-4.5 2020-2059 minimum change',
        'monthly_prc_change': monthly_shift_range.loc[:, 'min'].values
    },
    'climate_adjusted_ssp245_median': {
        'type': 'climate_adjusted', 
        'description': 'SSP2-4.5 2020-2059 median change',
        'monthly_prc_change': monthly_shift_range.loc[:, 'median'].values
    },
    'climate_adjusted_ssp245_max': {
        'type': 'climate_adjusted',
        'description': 'SSP2-4.5 2020-2059 maximum change',
        'monthly_prc_change': monthly_shift_range.loc[:, 'max'].values
    },
    'climate_adjusted_ssp375_min': {
        'type': 'climate_adjusted',
        'description': 'SSP3-7.0 2020-2059 minimum change',
        'monthly_prc_change': monthly_shift_range.loc[:, 'min'].values
    },
    'climate_adjusted_ssp375_median': {
        'type': 'climate_adjusted',
        'description': 'SSP3-7.0 2020-2059 median change',
        'monthly_prc_change': monthly_shift_range.loc[:, 'median'].values
    },
    'climate_adjusted_ssp375_max': {
        'type': 'climate_adjusted',
        'description': 'SSP3-7.0 2020-2059 maximum change',
        'monthly_prc_change': monthly_shift_range.loc[:, 'max'].values
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

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.abspath(f"{ROOT_DIR}/pywrdrb/outputs/")
FIG_DIR = os.path.abspath(f"{ROOT_DIR}/figures/")

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
            'predicted_inflow': f"{set_dir}/predicted_inflows_mgd.hdf5"
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
    
    def get_local_realization_ids(self):
        """Get list of local realization IDs for this set (0-based)"""
        return list(range(self.n_realizations))

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
    "nyc_release_components"
]

# Output files
RECONSTRUCTION_OUTPUT_FNAME = f"{OUTPUT_DIR}/reconstruction.hdf5"

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def verify_dataset_id(dataset_id):
    """Verify that the dataset_id is valid"""
    if dataset_id not in DATASET_CONFIGS:
        raise ValueError(f"Invalid dataset_id: {dataset_id}. Must be one of {list(DATASET_CONFIGS.keys())}")
    return True

def get_dataset_type(dataset_id):
    """Return 'stationary' or 'climate_adjusted' for a dataset"""
    verify_dataset_id(dataset_id)
    return DATASET_CONFIGS[dataset_id]['type']

def get_all_datasets_of_type(dataset_type):
    """Get all dataset_ids of a given type"""
    if dataset_type not in ['stationary', 'climate_adjusted']:
        raise ValueError(f"Invalid dataset_type: {dataset_type}. Must be 'stationary' or 'climate_adjusted'")
    return [did for did, cfg in DATASET_CONFIGS.items() 
            if cfg['type'] == dataset_type]

def get_ensemble_set_spec(set_id, dataset_id):
    """Get ensemble set specification by ID and dataset"""
    if dataset_id not in ENSEMBLE_SETS:
        raise ValueError(f"Invalid dataset_id: {dataset_id}")
    if set_id < 0 or set_id >= N_ENSEMBLE_SETS:
        raise ValueError(f"set_id must be between 0 and {N_ENSEMBLE_SETS-1}")
    return ENSEMBLE_SETS[dataset_id][set_id]

def get_target_ensemble_sets():
    """Get list of ensemble set IDs to process"""
    if WorkflowFlags.TARGET_ENSEMBLE_SETS is None:
        return list(range(N_ENSEMBLE_SETS))
    else:
        return WorkflowFlags.TARGET_ENSEMBLE_SETS

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

def get_all_ensemble_output_files(dataset_id):
    """Get list of all ensemble set output files for a dataset"""
    verify_dataset_id(dataset_id)
    return [spec.output_file for spec in ENSEMBLE_SETS[dataset_id]]

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

def print_experiment_summary(dataset_id):
    """Print comprehensive experiment configuration summary"""
    verify_dataset_id(dataset_id)
    dataset_config = DATASET_CONFIGS[dataset_id]
    generated_sets = get_existing_ensemble_sets(dataset_id)
    
    print("=" * 80)
    print("ENSEMBLE EXPERIMENT CONFIGURATION")
    print("=" * 80)
    print(f"Dataset ID: {dataset_id}")
    print(f"Dataset Type: {dataset_config['type']}")
    print(f"Description: {dataset_config['description']}")
    if dataset_config['type'] == 'climate_adjusted':
        print(f"Monthly % Changes: {dataset_config['monthly_prc_change']}")
    print()
    print(f"Total Realizations: {TOTAL_REALIZATIONS:,}")
    print(f"Ensemble Sets: {N_ENSEMBLE_SETS}")
    print(f"Realizations per Set: {N_REALIZATIONS_PER_ENSEMBLE_SET}")
    print(f"Years per Realization: {N_YEARS}")
    print(f"Simulation Period: {START_DATE} to {END_DATE}")
    print()
    print("Pywr-DRB Batching:")
    print(f"  Batches per Set: {N_PYWRDRB_BATCHES_PER_SET}")
    print(f"  Realizations per Batch: {N_REALIZATIONS_PER_PYWRDRB_BATCH}")
    print()
    print("Node Configuration:")
    print(f"  Nodes to Generate (KN): {len(pywrdrb_nodes_to_generate)}")
    print(f"  Nodes to Regress: {len(pywrdrb_nodes_to_regress)}")
    print()
    print("File Structure:")
    for i, spec in enumerate(generated_sets):
        print(f"  Set {i+1}: {spec.directory}")
        if i >= 2:  # Limit output for large experiments
            print(f"  ... (and {len(generated_sets)-3} more sets)")
            break
    print("=" * 80)

def print_ensemble_set_summary(set_id, dataset_id):
    """Print summary for a specific ensemble set"""
    spec = get_ensemble_set_spec(set_id, dataset_id)
    print(f"\n{dataset_id} Ensemble Set {set_id + 1} Summary:")
    print(f"  Dataset Type: {spec.ensemble_type}")
    print(f"  Global Realizations: {spec.start_realization}-{spec.end_realization-1}")
    print(f"  Directory: {spec.directory}")
    print(f"  Pywr-DRB Batches: {len(spec.pywrdrb_batches)}")
    print(f"  Output File: {spec.output_file}")

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

def verify_realization_id_consistency(dataset_id):
    """
    Verify that realization IDs are consistent across generation and simulation for a dataset.
    """
    verify_dataset_id(dataset_id)
    print(f"Verifying {dataset_id} realization ID consistency...")
    
    for set_id in range(N_ENSEMBLE_SETS):
        set_spec = get_ensemble_set_spec(set_id, dataset_id)
        
        # Check expected vs actual realization IDs
        expected_ids = set_spec.realizations
        
        if os.path.exists(set_spec.files['gage_flow']):
            actual_ids = get_hdf5_realization_numbers(set_spec.files['gage_flow'])
            actual_ids = [int(x) for x in actual_ids]
            
            if set(expected_ids) != set(actual_ids):
                print(f"MISMATCH in Set {set_id + 1}:")
                print(f"  Expected: {expected_ids}")
                print(f"  Actual:   {actual_ids}")
            else:
                print(f"Set {set_id + 1}: OK")
        else:
            print(f"Set {set_id + 1}: File not found")

# Backward compatibility mappings (can be removed after migration)
def get_dataset_id_from_legacy(ensemble_type, climate_scenario=None):
    """Map old ensemble_type/scenario to new dataset_id for backward compatibility"""
    if ensemble_type == 'stationary':
        return 'stationary_ensemble'
    elif ensemble_type == 'climate_adjusted':
        if climate_scenario:
            # Try to find matching dataset
            for dataset_id in DATASET_CONFIGS:
                if climate_scenario in dataset_id:
                    return dataset_id
        # Default climate adjusted
        climate_datasets = get_all_datasets_of_type('climate_adjusted')
        return climate_datasets[0] if climate_datasets else None
    return None

# Validate configuration on import
validate_configuration()