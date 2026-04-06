"""
Calculate SSI-based drought metrics for ensembles using MPI parallelization.

Each rank independently:
  1. Reads realization IDs from HDF5 metadata (no data loaded)
  2. Loads ONLY its assigned realizations (selective HDF5 reading)
  3. Computes SSI and drought metrics
  4. Sends results to rank 0 via point-to-point send/recv

No global MPI collectives (bcast, gather, barrier, reduce) are used.

Modes:
  Historic (no MPI):
    python 05_calculate_ssi_drought_metrics.py historic

  Synthetic ensemble (MPI):
    mpirun -np N python 05_calculate_ssi_drought_metrics.py <dataset_id>

  Serial fallback:
    python 05_calculate_ssi_drought_metrics.py <dataset_id>
"""

import sys
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from synhydro.droughts.ssi import get_drought_metrics

from methods.mpi_utils import get_comm, global_point_to_point_gather
from methods.utils import distribute_realizations_across_ranks
from methods.load import get_realization_ids_from_export, load_rank_subset_from_export
from methods.drought_analysis import calculate_historic_observed_droughts, fit_ssi_calculator
from methods.config import *

EXPORT_SSI_HDF5 = False


def calculate_ssi_drought_metrics(dataset_id, ssi_windows=[3, 6, 12]):
    """
    Calculate SSI-based drought metrics for a dataset using MPI parallelization.
    Each rank loads only its assigned realizations directly from HDF5.
    """
    comm, rank, size = get_comm()

    # Verify dataset
    verify_dataset_id(dataset_id)
    dataset_config = DATASET_CONFIGS[dataset_id]

    if rank == 0:
        print(f"[SSI] {dataset_id} | windows={ssi_windows} | {size} ranks")

    # Each rank checks file existence independently
    fname = f'{OUTPUT_DIR}/{dataset_id}_with_postprocessing.hdf5'
    if not os.path.exists(fname):
        print(f"Rank {rank} ERROR: File not found: {fname}")
        return False

    # Determine SSI node and corresponding results_set from config
    node = SSI_NODE
    node_config = SSI_NODE_CONFIGS[node]
    results_set_key = node_config['results_set']

    # Each rank reads realization IDs from HDF5 metadata (fast, no data loaded)
    realization_ids = get_realization_ids_from_export(fname, dataset_id,
                                                      results_set=results_set_key)
    n_realizations = len(realization_ids)

    if rank == 0:
        print(f"  {n_realizations} realizations, SSI node: {node}")

    # Determine this rank's assigned realizations
    my_realizations = distribute_realizations_across_ranks(
        realization_ids, rank, size
    )

    # Ranks with no realizations skip loading but still participate in gathers
    if len(my_realizations) == 0:
        local_syn_ensemble = {}
    else:
        # Each rank loads ONLY its assigned realizations (staggered I/O)
        local_data = load_rank_subset_from_export(
            fname, my_realizations, [results_set_key], rank, size
        )

        # Extract local ensemble dict from the configured results_set
        local_syn_ensemble = getattr(local_data, results_set_key)[dataset_id]
        del local_data  # free wrapper

        # Derive the target node column if needed
        if node_config['derived']:
            for real_id, df in local_syn_ensemble.items():
                df[node] = df[node_config['derive_from']].sum(axis=1)


    # SSI Drought Metrics
    for ssi_window in ssi_windows:

        if rank == 0:
            print(f"\nProcessing SSI window: {ssi_window} months")

        # Each rank fits SSI calculator independently (uses historical obs data)
        ssi_calculator = fit_ssi_calculator(ssi_window, node=node)

        # Process assigned realizations
        local_ssi_data = {}
        local_drought_data = []

        for i, real_id in enumerate(my_realizations):
            Qsi = local_syn_ensemble[real_id].loc[:, node]
            Qsi_monthly = Qsi.resample('MS').sum()

            # Calculate SSI
            ssi_values = ssi_calculator.transform(Qsi_monthly)
            local_ssi_data[str(real_id)] = ssi_values

            # Calculate drought metrics
            drought_chars = get_drought_metrics(ssi_values)
            if not drought_chars.empty:
                drought_chars['realization_id'] = real_id
                local_drought_data.append(drought_chars)

        # Gather SSI data only if exporting to HDF5 (expensive with many ranks)
        if EXPORT_SSI_HDF5:
            all_ssi_data = global_point_to_point_gather(
                comm, local_ssi_data, rank, size, tag=601
            )

        # Gather drought results to rank 0 via point-to-point (no collective gather)
        all_drought_data = global_point_to_point_gather(
            comm, local_drought_data, rank, size, tag=602
        )

        # Combine and save on rank 0
        if rank == 0:
            # Save SSI values to hdf5 (optional)
            if EXPORT_SSI_HDF5:
                # Combine SSI data
                combined_ssi = {}
                for rank_data in all_ssi_data:
                    combined_ssi.update(rank_data)

                # Create SSI DataFrame using rank 0's first realization for the index
                first_real_id = my_realizations[0]
                syn_ssi = pd.DataFrame(
                    index=local_syn_ensemble[first_real_id].resample('MS').sum().index,
                    columns=np.arange(0, n_realizations)
                )

                for real_id, ssi_values in combined_ssi.items():
                    syn_ssi.loc[:, int(real_id)] = ssi_values

                # Put in dict with node name as key
                syn_ssi_dict = {node: syn_ssi}
                for key, df in syn_ssi_dict.items():
                    syn_ssi_dict[key].columns = df.columns.astype(str)

                from synhydro.core.ensemble import Ensemble
                ssi_fname = f"{DROUGHT_METRICS_DIR}/{dataset_id}_ssi{ssi_window}.hdf5"
                print(f"  Saving SSI values to hdf5: {ssi_fname}")
                ssi_ensemble = Ensemble(syn_ssi_dict)
                ssi_ensemble.to_hdf5(ssi_fname)

            # Combine drought data
            syn_droughts = pd.DataFrame()
            for rank_data in all_drought_data:
                for drought_df in rank_data:
                    syn_droughts = pd.concat([syn_droughts, drought_df], axis=0)

            # Save synthetic drought metrics
            syn_droughts.reset_index(inplace=True, drop=True)
            syn_fname = f"{DROUGHT_METRICS_DIR}/{dataset_id}_ssi{ssi_window}_drought_events.csv"
            syn_droughts.to_csv(syn_fname, index=False)
            print(f"  Saved synthetic drought metrics: {syn_fname}")

    if rank == 0:
        print(f"\nAll drought metrics saved successfully for {dataset_id}!")

    return True



def main(dataset_id):
    """Main function for calculating synthetic drought metrics"""

    comm, rank, size = get_comm()

    if rank == 0:
        os.makedirs(DROUGHT_METRICS_DIR, exist_ok=True)

    # Calculate drought metrics (using default SSI windows)
    success = calculate_ssi_drought_metrics(dataset_id, ssi_windows=[3,6,12])

    # Barrier so all ranks finish before MPI_Finalize
    if comm is not None:
        comm.Barrier()

    if rank == 0:
        if not success:
            print("ERROR: SSI drought metrics calculation failed!")
            sys.exit(1)


if __name__ == "__main__":

    # Check for 'historic' mode
    if len(sys.argv) == 2 and sys.argv[1].lower() == 'historic':
        # Calculate historic observed droughts only (no MPI needed)
        print("Running in HISTORIC mode - calculating observed droughts only")
        output_dir = DROUGHT_METRICS_DIR
        success = calculate_historic_observed_droughts(
            ssi_windows=[3, 6, 12], output_dir=output_dir
        )
        if not success:
            sys.exit(1)
    elif len(sys.argv) == 2:
        # Normal mode - calculate synthetic droughts for a dataset
        dataset_id = sys.argv[1]
        verify_dataset_id(dataset_id)
        main(dataset_id)
    else:
        print("Usage:")
        print("  For historic observed droughts:")
        print("    python 05_calculate_ssi_drought_metrics.py historic")
        print()
        print("  For synthetic ensemble droughts:")
        print("    mpirun -np N python 05_calculate_ssi_drought_metrics.py <dataset_id>")
        print("    python 05_calculate_ssi_drought_metrics.py <dataset_id>  (serial fallback)")
        print(f"    Available datasets: {list(DATASET_CONFIGS.keys())}")
        sys.exit(1)
