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

# PyWR-DRB simulation batching (within each ensemble set)
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
# PARALLEL PROCESSING
# =============================================================================

# MPI settings
MAX_MPI_RANKS_PER_ENSEMBLE_SET = 20  # For ensemble generation
MAX_MPI_RANKS_PER_PYWRDRB_BATCH = 1  # PyWR-DRB runs on single core
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
ENSEMBLE_BASE_DIR = os.path.abspath(f"{ROOT_DIR}/pywrdrb/inputs/stationary_ensemble/")

# Ensemble set directories and files
def get_ensemble_set_dir(set_id):
    """Get directory path for a specific ensemble set"""
    return f"{ENSEMBLE_BASE_DIR}/stationary_ensemble_set{set_id + 1}"

def get_ensemble_set_files(set_id):
    """Get file paths for a specific ensemble set"""
    set_dir = get_ensemble_set_dir(set_id)
    return {
        'gage_flow': f"{set_dir}/gage_flow_mgd.hdf5",
        'catchment_inflow': f"{set_dir}/catchment_inflow_mgd.hdf5", 
        'predicted_inflow': f"{set_dir}/predicted_inflow_mgd.hdf5"
    }

# Output files
RECONSTRUCTION_OUTPUT_FNAME = f"{OUTPUT_DIR}/reconstruction.hdf5"

def get_ensemble_set_output_fname(set_id):
    """Get output filename for a specific ensemble set"""
    return f"{OUTPUT_DIR}/stationary_ensemble_set{set_id + 1}.hdf5"

# =============================================================================
# ENSEMBLE SET SPECIFICATIONS
# =============================================================================

class EnsembleSetSpec:
    """Specification for a single ensemble set"""
    
    def __init__(self, set_id):
        self.set_id = set_id
        self.start_realization = set_id * N_REALIZATIONS_PER_ENSEMBLE_SET
        self.end_realization = (set_id + 1) * N_REALIZATIONS_PER_ENSEMBLE_SET
        self.n_realizations = N_REALIZATIONS_PER_ENSEMBLE_SET
        
        # File paths
        self.directory = get_ensemble_set_dir(set_id)
        self.files = get_ensemble_set_files(set_id)
        self.output_file = get_ensemble_set_output_fname(set_id)
        
        # PyWR-DRB batching within this set
        self.pywrdrb_batches = self._create_pywrdrb_batch_specs()
    
    def _create_pywrdrb_batch_specs(self):
        """Create PyWR-DRB batch specifications within this ensemble set"""
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
ENSEMBLE_SETS = [EnsembleSetSpec(i) for i in range(N_ENSEMBLE_SETS)]


# =============================================================================
# PYWR-DRB CONFIGURATION
# =============================================================================

# Setup pathnavigator for PyWR-DRB
pn_config = pywrdrb.get_pn_config()
pn_config["flows/stationary_ensemble"] = os.path.abspath(ENSEMBLE_BASE_DIR)

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

def get_ensemble_set_spec(set_id):
    """Get ensemble set specification by ID"""
    if set_id < 0 or set_id >= N_ENSEMBLE_SETS:
        raise ValueError(f"set_id must be between 0 and {N_ENSEMBLE_SETS-1}")
    return ENSEMBLE_SETS[set_id]

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
    
    for ensemble_set in ENSEMBLE_SETS:
        os.makedirs(ensemble_set.directory, exist_ok=True)

def get_all_ensemble_output_files():
    """Get list of all ensemble set output files"""
    return [spec.output_file for spec in ENSEMBLE_SETS]

def get_existing_ensemble_sets():
    """Get list of ensemble set IDs that have been generated"""
    existing_sets = []
    for set_id in range(N_ENSEMBLE_SETS):
        spec = get_ensemble_set_spec(set_id)
        # Check if both required files exist
        if (os.path.exists(spec.files['gage_flow']) and 
            os.path.exists(spec.files['catchment_inflow'])):
            existing_sets.append(set_id)
    return existing_sets

def print_experiment_summary():
    """Print comprehensive experiment configuration summary"""
    print("=" * 80)
    print("HIERARCHICAL ENSEMBLE EXPERIMENT CONFIGURATION")
    print("=" * 80)
    print(f"Total Realizations: {TOTAL_REALIZATIONS:,}")
    print(f"Ensemble Sets: {N_ENSEMBLE_SETS}")
    print(f"Realizations per Set: {N_REALIZATIONS_PER_ENSEMBLE_SET}")
    print(f"Years per Realization: {N_YEARS}")
    print(f"Simulation Period: {START_DATE} to {END_DATE}")
    print()
    print("PyWR-DRB Batching:")
    print(f"  Batches per Set: {N_PYWRDRB_BATCHES_PER_SET}")
    print(f"  Realizations per Batch: {N_REALIZATIONS_PER_PYWRDRB_BATCH}")
    print()
    print("Node Configuration:")
    print(f"  Nodes to Generate (KN): {len(pywrdrb_nodes_to_generate)}")
    print(f"  Nodes to Regress: {len(pywrdrb_nodes_to_regress)}")
    print()
    print("File Structure:")
    for i, spec in enumerate(ENSEMBLE_SETS):
        print(f"  Set {i+1}: {spec.directory}")
        if i >= 2:  # Limit output for large experiments
            print(f"  ... (and {N_ENSEMBLE_SETS-3} more sets)")
            break
    print("=" * 80)

def print_ensemble_set_summary(set_id):
    """Print summary for a specific ensemble set"""
    spec = get_ensemble_set_spec(set_id)
    print(f"\nEnsemble Set {set_id + 1} Summary:")
    print(f"  Global Realizations: {spec.start_realization}-{spec.end_realization-1}")
    print(f"  Directory: {spec.directory}")
    print(f"  PyWR-DRB Batches: {len(spec.pywrdrb_batches)}")
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

# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================

# For scripts that expect the old interface
def get_legacy_ensemble_files():
    """Get ensemble files in legacy format (for backward compatibility)"""
    # Return first ensemble set files as default
    if N_ENSEMBLE_SETS > 0:
        return ENSEMBLE_SETS[0].files
    else:
        return {}

# Legacy variable names
gage_flow_ensemble_fname = get_ensemble_set_files(0)['gage_flow'] if N_ENSEMBLE_SETS > 0 else ""
catchment_inflow_ensemble_fname = get_ensemble_set_files(0)['catchment_inflow'] if N_ENSEMBLE_SETS > 0 else ""