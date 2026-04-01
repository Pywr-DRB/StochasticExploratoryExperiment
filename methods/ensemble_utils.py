"""
Ensemble set utility functions.

Provides helpers for looking up ensemble set specs, creating directory
structures, saving config snapshots, and checking which sets exist on disk.
"""

import os
import json
from datetime import datetime as _dt

from methods.config import (
    ENSEMBLE_BASE_DIR,
    N_ENSEMBLE_SETS, N_REALIZATIONS_PER_ENSEMBLE_SET,
    N_REALIZATIONS_PER_PYWRDRB_BATCH, N_PYWRDRB_BATCHES_PER_SET,
    CONFIG_NAME, CONFIG_DIR, FLOW_PREDICTION_MODE,
    START_DATE, END_DATE,
    TOTAL_REALIZATIONS, SALINITY_LSTM_PREDICTIONS,
    DATASET_CONFIGS,
    OUTPUT_DIR, MODEL_DIR, DROUGHT_METRICS_DIR, PERFORMANCE_METRICS_DIR,
    EVENT_METRICS_DIR, ZONE_PROB_DIR, SATISFICING_DIR, FOCAL_EVENTS_DIR,
    FIG_DIR,
)
from methods.verification import verify_dataset_id


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
# UTILITY FUNCTIONS
# =============================================================================

def get_ensemble_set_spec(set_id, dataset_id):
    """Get ensemble set specification by ID and dataset"""
    if dataset_id not in ENSEMBLE_SETS:
        raise ValueError(f"Invalid dataset_id: {dataset_id}")
    if set_id < 0 or set_id >= N_ENSEMBLE_SETS:
        raise ValueError(f"set_id must be between 0 and {N_ENSEMBLE_SETS-1}")
    return ENSEMBLE_SETS[dataset_id][set_id]


def save_config_json():
    """Save configuration snapshot to config.json in the config output directory."""
    config_record = {
        "config_name": CONFIG_NAME,
        "flow_prediction_mode": FLOW_PREDICTION_MODE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "total_realizations": TOTAL_REALIZATIONS,
        "n_ensemble_sets": N_ENSEMBLE_SETS,
        "dataset_ids": list(DATASET_CONFIGS.keys()),
        "salinity_lstm": SALINITY_LSTM_PREDICTIONS,
        "created": _dt.now().isoformat(),
    }
    os.makedirs(CONFIG_DIR, exist_ok=True)
    config_path = os.path.join(CONFIG_DIR, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_record, f, indent=2)


def ensure_ensemble_set_dirs(dataset_id=None):
    """Create all necessary ensemble set and output directories"""
    # Shared input directories
    os.makedirs(ENSEMBLE_BASE_DIR, exist_ok=True)

    # Config-specific output directories
    for d in [OUTPUT_DIR, MODEL_DIR, DROUGHT_METRICS_DIR, PERFORMANCE_METRICS_DIR,
              EVENT_METRICS_DIR, ZONE_PROB_DIR, SATISFICING_DIR, FOCAL_EVENTS_DIR, FIG_DIR]:
        os.makedirs(d, exist_ok=True)
    save_config_json()

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
