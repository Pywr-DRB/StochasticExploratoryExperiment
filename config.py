import os
import math
import pywrdrb
from pywrdrb.pywr_drb_node_data import immediate_downstream_nodes_dict

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

ensemble_type_opts = [
    'stationary',
    'climate_adjusted'
]

# =============================================================================
# CLIMATE ADJUSTED ENSEMBLE SETTINGS
# =============================================================================

# Percentage change in monthly mean flow relative to reconstruction
# Starting in January
# Currently, using the same shift for all nodes
# This should be applied during the Kirsch-Nowak generation
monthly_mean_flow_prc_change = [
    15.0,  # January
    35.0,  # February
    -10.0,  # March
    -20.0,  # April
    -10.0,  # May
    0.0,  # June
    30.0,  # July
    -10.0,  # August
    -30.0,  # September
    -20.0,  # October
    -10.0,  # November
    5.0   # December
]



# =============================================================================
# PARALLEL PROCESSING
# =============================================================================

# MPI settings
MAX_MPI_RANKS_PER_ENSEMBLE_SET = 20  # For ensemble generation
MAX_MPI_RANKS_PER_PYWRDRB_BATCH = 1  # Pywr-DRB runs on single core
MAX_PARALLEL_ENSEMBLE_SETS = 2  # Generate sets in parallel
MAX_PARALLEL_PYWRDRB_BATCHES = 10  # Run batches in parallel within set


# =============================================================================
# WORKFLOW CONTROL
# =============================================================================

class WorkflowFlags:
    """Control which steps of the workflow to run"""
    RUN_BASELINE = False
    GENERATE_ENSEMBLE_SETS = True
    PLOT_DIAGNOSTICS = False
    PREP_PYWRDRB = True
    RUN_PYWRDRB = True
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

# Ensemble set directories and files
def get_ensemble_set_dir(set_id, type):
    """Get directory path for a specific ensemble set"""
    return f"{ENSEMBLE_BASE_DIR}/{type}_ensemble/{type}_ensemble_set{set_id + 1}"

def get_ensemble_set_files(set_id, type):
    """Get file paths for a specific ensemble set"""
    set_dir = get_ensemble_set_dir(set_id, type)
    return {
        'gage_flow': f"{set_dir}/gage_flow_mgd.hdf5",
        'catchment_inflow': f"{set_dir}/catchment_inflow_mgd.hdf5", 
        'predicted_inflow': f"{set_dir}/predicted_inflow_mgd.hdf5"
    }

# Output files
RECONSTRUCTION_OUTPUT_FNAME = f"{OUTPUT_DIR}/reconstruction.hdf5"

def get_ensemble_set_output_fname(set_id, type):
    """Get output filename for a specific ensemble set"""
    return f"{OUTPUT_DIR}/{type}_ensemble_set{set_id + 1}.hdf5"

# =============================================================================
# ENSEMBLE SET SPECIFICATIONS
# =============================================================================

class EnsembleSetSpec:
    """Specification for a single ensemble set"""
    
    def __init__(self, set_id, type):
        

        assert type in ensemble_type_opts, \
            f"Invalid ensemble type passed to Ensemble Set Spec: {type}. Must be one of {ensemble_type_opts}"
        
        self.type = type
        self.set_id = set_id
        self.start_realization = set_id * N_REALIZATIONS_PER_ENSEMBLE_SET
        self.end_realization = (set_id + 1) * N_REALIZATIONS_PER_ENSEMBLE_SET
        self.n_realizations = N_REALIZATIONS_PER_ENSEMBLE_SET
        self.realizations = self.get_realization_ids()
        
        
        # File paths
        self.directory = get_ensemble_set_dir(set_id, type)
        self.files = get_ensemble_set_files(set_id, type)
        self.output_file = get_ensemble_set_output_fname(set_id, type)

        # Pywr-DRB batching within this set
        self.pywrdrb_batches = self._create_pywrdrb_batch_specs()
    
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
                'type': self.type,
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

# Create all ensemble set specifications
STATIONARY_ENSEMBLE_SETS = [
    EnsembleSetSpec(i, type='stationary') for i in range(N_ENSEMBLE_SETS)
    ]

CLIMATE_ADJUSTED_ENSEMBLE_SETS = [
    EnsembleSetSpec(i, type='climate_adjusted') for i in range(N_ENSEMBLE_SETS)
    ]

# =============================================================================
# PYWR-DRB CONFIGURATION
# =============================================================================

# Setup pathnavigator for Pywr-DRB
pn_config = pywrdrb.get_pn_config()
pn_config["flows/stationary_ensemble"] = os.path.abspath(f"{ENSEMBLE_BASE_DIR}/stationary_ensemble/")
pn_config["flows/climate_adjusted_ensemble"] = os.path.abspath(f"{ENSEMBLE_BASE_DIR}/climate_adjusted_ensemble/")

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
    "major_flow", "inflow", "res_storage",
    "lower_basin_mrf_contributions", "mrf_target", 
    "ibt_diversions", "ibt_demands",
]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_ensemble_set_spec(set_id, type):
    """Get ensemble set specification by ID"""
    if set_id < 0 or set_id >= N_ENSEMBLE_SETS:
        raise ValueError(f"set_id must be between 0 and {N_ENSEMBLE_SETS-1}")
    
    if type == 'stationary':
        return STATIONARY_ENSEMBLE_SETS[set_id]
    
    elif type == 'climate_adjusted':
        return CLIMATE_ADJUSTED_ENSEMBLE_SETS[set_id]
    
    else:
        raise ValueError(f"Invalid ensemble type: {type}")

def get_target_ensemble_sets():
    """Get list of ensemble set IDs to process"""
    if WorkflowFlags.TARGET_ENSEMBLE_SETS is None:
        return list(range(N_ENSEMBLE_SETS))
    else:
        return WorkflowFlags.TARGET_ENSEMBLE_SETS

def ensure_ensemble_set_dirs():
    """Create all necessary ensemble set directories"""
    os.makedirs(ENSEMBLE_BASE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    
    # Add directories for each ensemble set type
    # These will contain directories for each set also
    os.makedirs(f"{ENSEMBLE_BASE_DIR}/stationary_ensemble", exist_ok=True)
    os.makedirs(f"{ENSEMBLE_BASE_DIR}/climate_adjusted_ensemble", exist_ok=True)
    
    for ensemble_set in STATIONARY_ENSEMBLE_SETS:
        os.makedirs(ensemble_set.directory, exist_ok=True)

    for ensemble_set in CLIMATE_ADJUSTED_ENSEMBLE_SETS:
        os.makedirs(ensemble_set.directory, exist_ok=True)


def get_all_ensemble_output_files(type):
    """Get list of all ensemble set output files"""

    if type == 'stationary':
        return [spec.output_file for spec in STATIONARY_ENSEMBLE_SETS]

    elif type == 'climate_adjusted':
        return [spec.output_file for spec in CLIMATE_ADJUSTED_ENSEMBLE_SETS]

    else:
        raise ValueError(f"Invalid ensemble type: {type}")

def get_existing_ensemble_sets(type):
    """Get list of ensemble set IDs that have been generated"""
    existing_sets = []
    for set_id in range(N_ENSEMBLE_SETS):
        spec = get_ensemble_set_spec(set_id, type=type)
        # Check if both required files exist
        if (os.path.exists(spec.files['gage_flow']) and 
            os.path.exists(spec.files['catchment_inflow'])):
            existing_sets.append(set_id)
    return existing_sets

def print_experiment_summary(type):
    """Print comprehensive experiment configuration summary"""
    
    generated_sets = get_existing_ensemble_sets(type=type)
    
    print("=" * 80)
    print("ENSEMBLE EXPERIMENT CONFIGURATION")
    print("=" * 80)
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
            print(f"  ... (and {N_ENSEMBLE_SETS-3} more sets)")
            break
    print("=" * 80)

def print_ensemble_set_summary(set_id, type):
    """Print summary for a specific ensemble set"""
    spec = get_ensemble_set_spec(set_id, type=type)
    print(f"\n{type} Ensemble Set {set_id + 1} Summary:")
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
    
    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))
    
    return True

# Validate configuration on import
validate_configuration()
