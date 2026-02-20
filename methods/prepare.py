"""
Core function for preparing Pywr-DRB inputs from synthetic ensembles.
This module contains the input preparation logic that can be used in both
serial and parallel modes.
"""

import os
import pywrdrb
from pywrdrb.utils.hdf5 import get_hdf5_realization_numbers
from pywrdrb.pre import (
    PredictedInflowEnsemblePreprocessor,
    ExtrapolatedDiversionEnsemblePreprocessor,
    PredictedDiversionEnsemblePreprocessor
)

from methods.config import get_ensemble_set_spec

# Conditional MPI import
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False


def prep_ensemble_set(set_id, dataset_id, use_mpi=True):
    """
    Prepare Pywr-DRB inputs for a single ensemble set

    Parameters:
    -----------
    set_id : int
        Ensemble set identifier (0-indexed)
    dataset_id : str
        Dataset identifier (e.g., 'stationary_ensemble', 'climate_adjusted_ssp245_min')
    use_mpi : bool
        If True, use MPI for parallel execution. If False, run serially.

    Returns:
    --------
    bool
        True if successful, False otherwise
    """

    # Get MPI info for this function call
    if use_mpi and MPI_AVAILABLE:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        comm = None
        rank = 0
        size = 1

    # Get ensemble set specification
    set_spec = get_ensemble_set_spec(set_id, dataset_id)
    catchment_inflow_file = set_spec.files['catchment_inflow']
    ensemble_dir = set_spec.directory

    if rank == 0:
        print(f"Set {set_id+1} Rank {rank}: Preparing inputs for {dataset_id} ensemble set {set_id + 1}")
        print(f"  Input file: {catchment_inflow_file}")
        print(f"  Ensemble directory: {ensemble_dir}")

    # Check if input file exists
    if not os.path.exists(catchment_inflow_file):
        print(f"Error: Input file not found: {catchment_inflow_file}")
        return False

    # Setup pathnavigator for this specific ensemble set
    pn_config = pywrdrb.get_pn_config()
    pn_config[f"flows/{dataset_id}_set{set_id + 1}"] = os.path.abspath(ensemble_dir)
    pywrdrb.load_pn_config(pn_config)

    try:
        if rank == 0:
            # Get realization numbers from the HDF5 file
            realization_ids = get_hdf5_realization_numbers(catchment_inflow_file)
            print(f"Set {set_id + 1}: Found {len(realization_ids)} realizations")

        else:
            realization_ids = None

        # Broadcast realization IDs (only needed in MPI mode)
        if use_mpi and comm:
            realization_ids = comm.bcast(realization_ids, root=0)

        # =====================================================================
        # Step 1: Process predicted inflows
        # =====================================================================
        if rank == 0:
            print(f"Set {set_id + 1}: Processing predicted inflows...")

        inflow_preprocessor = PredictedInflowEnsemblePreprocessor(
            flow_type=f"{dataset_id}_set{set_id + 1}",
            ensemble_hdf5_file=catchment_inflow_file,
            realization_ids=realization_ids,
            start_date=None,  # Use full range
            end_date=None,
            modes=('regression_disagg',),
            use_log=True,
            remove_zeros=True,
            use_const=False,
            use_mpi=use_mpi  # Enable MPI within the preprocessor
        )

        inflow_preprocessor.load()
        inflow_preprocessor.process()
        inflow_preprocessor.save()

        # Free up memory
        del inflow_preprocessor

        if rank == 0:
            print(f"Set {set_id + 1}: Predicted inflows complete.")

        # =====================================================================
        # Step 2: Process NJ diversions ensemble
        # =====================================================================
        if rank == 0:
            print(f"Set {set_id + 1}: Processing NJ diversions ensemble...")

        # Use the gage_flow file as input for diversions
        gage_flow_file = set_spec.files['gage_flow']

        nj_extrapolator = ExtrapolatedDiversionEnsemblePreprocessor(
            loc="nj",
            flow_type=f"{dataset_id}_set{set_id + 1}",
            ensemble_hdf5_file=gage_flow_file,
            realization_ids=realization_ids,
            use_mpi=use_mpi
        )
        nj_extrapolator.load()
        nj_extrapolator.process()
        nj_extrapolator.save()

        if rank == 0:
            print(f"Set {set_id + 1}: NJ diversions ensemble complete.")

        # =====================================================================
        # Step 3: Process NYC diversions ensemble
        # =====================================================================
        if rank == 0:
            print(f"Set {set_id + 1}: Processing NYC diversions ensemble...")

        nyc_extrapolator = ExtrapolatedDiversionEnsemblePreprocessor(
            loc="nyc",
            flow_type=f"{dataset_id}_set{set_id + 1}",
            ensemble_hdf5_file=gage_flow_file,
            realization_ids=realization_ids,
            use_mpi=use_mpi
        )

        nyc_extrapolator.load()
        nyc_extrapolator.process()
        nyc_extrapolator.save()

        # Free up memory
        del nyc_extrapolator

        if rank == 0:
            print(f"Set {set_id + 1}: NYC diversions ensemble complete.")


        # =====================================================================
        # Step 4: Process predicted diversions ensemble
        # =====================================================================
        if rank == 0:
            print(f"Set {set_id + 1}: Processing predicted diversions ensemble...")

        # Get path to the NJ extrapolated diversions we just created
        nj_div_hdf5 = str(pywrdrb.get_pn_object().sc.get(
            f"flows/{dataset_id}_set{set_id + 1}"
        ) / "diversion_nj_extrapolated_mgd.hdf5")

        diversion_predictor = PredictedDiversionEnsemblePreprocessor(
            flow_type=f"{dataset_id}_set{set_id + 1}",
            ensemble_hdf5_file=nj_div_hdf5,
            realization_ids=realization_ids,
            start_date=None,
            end_date=None,
            modes=('regression_disagg',),
            use_log=True,
            remove_zeros=True,
            use_const=False,
            use_mpi=use_mpi
        )

        diversion_predictor.load()
        diversion_predictor.process()
        diversion_predictor.save()

        if rank == 0:
            print(f"Set {set_id + 1}: Predicted diversions ensemble complete.")
            print(f"Set {set_id + 1}: All Pywr-DRB inputs prepared successfully.")

        return True

    except Exception as e:
        print(f"Error processing set {set_id + 1}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
