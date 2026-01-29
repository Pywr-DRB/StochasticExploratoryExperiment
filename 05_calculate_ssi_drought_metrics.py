"""
Calculate SSI-based drought metrics for ensembles using MPI parallelization.

Modes:
  Historic (no MPI):
    python 05_calculate_ssi_drought_metrics.py historic

  Synthetic ensemble (MPI):
    mpirun -np N python 05_calculate_ssi_drought_metrics.py <dataset_id>

The historic observed drought calculation uses the shared function from
methods.drought_analysis. The MPI-parallelized ensemble calculation is
implemented here because it requires MPI send/recv for distributing
realizations across ranks.
"""

import sys
import os
import numpy as np
import pandas as pd
from mpi4py import MPI
import warnings
warnings.filterwarnings("ignore")

import pywrdrb
from sglib.droughts.ssi import SSIDroughtMetrics

from methods.utils import distribute_realizations_across_ranks
from methods.verification import verify_postprocessing_output
from methods.drought_analysis import calculate_historic_observed_droughts, fit_ssi_calculator
from methods.config import *

EXPORT_SSI_HDF5 = False


def calculate_ssi_drought_metrics(dataset_id, ssi_windows=[3, 6, 12]):
    """
    Calculate SSI-based drought metrics for a dataset using MPI parallelization.
    Note: Historic observed droughts are NOT calculated here - run with 'historic' argument separately.
    """
    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Verify dataset
    verify_dataset_id(dataset_id)
    dataset_config = DATASET_CONFIGS[dataset_id]

    if rank == 0:
        print(f"Calculating SSI drought metrics for: {dataset_id}")
        print(f"Dataset type: {dataset_config['type']}")
        print(f"SSI windows: {ssi_windows}")
        print(f"Using {size} MPI ranks")

    # Verify postprocessed data exists
    if rank == 0:
        try:
            verify_postprocessing_output(dataset_id)
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            return False

    # Wait for rank 0 verification
    comm.barrier()

    # Load synthetic ensemble (on disk)
    fname = f'./pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5'

    # --- Only rank 0 loads the export; others wait for metadata ---
    if rank == 0:
        data = pywrdrb.Data()
        data.load_from_export(fname, results_sets=['gage_flow'])

        # Keep just the combined ensemble dict
        syn_ensemble = data.gage_flow[dataset_id]  # no copy: avoid doubling memory
        del data  # free wrapper memory

        realization_ids = list(syn_ensemble.keys())
        n_realizations = len(realization_ids)

        ## Calculate the nyc_aggregate flow as the sum of
        nyc_gages = ["01425000", "01417000", "01436000"]
        for real_id, df in syn_ensemble.items():
            df['nyc_aggregate'] = df[nyc_gages].sum(axis=1)
            
            # add it back to the dict
            syn_ensemble[real_id] = df
        
    else:
        syn_ensemble = None
        realization_ids = None
        n_realizations = None

    # Broadcast small metadata only
    n_realizations = comm.bcast(n_realizations, root=0)
    realization_ids = comm.bcast(realization_ids, root=0)

    # Determine each rank's slice
    my_realizations = distribute_realizations_across_ranks(realization_ids, rank, size)

    # --- Distribute only the needed realizations per rank (send/recv) ---
    if rank == 0:
        for r in range(size):
            r_ids = distribute_realizations_across_ranks(realization_ids, r, size)
            small = {real_id: syn_ensemble[real_id] for real_id in r_ids}
            if r == 0:
                local_syn_ensemble = small
            else:
                comm.send(small, dest=r, tag=101)
        print(f"Rank 0 prepared and distributed per-rank subsets.")
    else:
        local_syn_ensemble = comm.recv(source=0, tag=101)

    print(f"Rank {rank} received {len(local_syn_ensemble)} realizations.")

    if rank == 0:
        print(f"Loaded synthetic {dataset_id} ensemble with {n_realizations} realizations.")

    # SSI Drought Metrics
    for ssi_window in ssi_windows:

        if rank == 0:
            print(f"\nProcessing SSI window: {ssi_window} months")

        node = 'nyc_aggregate'

        # Fit SSI on baseline period (1980-2019) — same on all ranks
        ssi_calculator = fit_ssi_calculator(ssi_window, node=node)
        drought_calculator = SSIDroughtMetrics()

        if rank == 0:
            print(f"  Rank {rank} processing {len(my_realizations)} realizations")

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
            drought_chars = drought_calculator.calculate_drought_metrics(ssi_values)
            if not drought_chars.empty:
                drought_chars['realization_id'] = real_id
                local_drought_data.append(drought_chars)

        # Gather results from all ranks
        all_ssi_data = comm.gather(local_ssi_data, root=0)
        all_drought_data = comm.gather(local_drought_data, root=0)

        # Combine and save on rank 0
        if rank == 0:
            # Combine SSI data
            combined_ssi = {}
            for rank_data in all_ssi_data:
                combined_ssi.update(rank_data)

            # Create SSI DataFrame (use index from any realization)
            syn_ssi = pd.DataFrame(
                index=syn_ensemble[realization_ids[0]].resample('MS').sum().index,
                columns=np.arange(0, n_realizations)
            )

            for real_id, ssi_values in combined_ssi.items():
                syn_ssi.loc[:, int(real_id)] = ssi_values

            # Put in dict with node name as key
            syn_ssi_dict = {node: syn_ssi}
            for key, df in syn_ssi_dict.items():
                syn_ssi_dict[key].columns = df.columns.astype(str)

            # Save SSI values to hdf5 (optional)
            if EXPORT_SSI_HDF5:
                from sglib.core.ensemble import Ensemble
                ssi_fname = f"./pywrdrb/drought_metrics/{dataset_id}_ssi{ssi_window}.hdf5"
                print(f"  Saving SSI values to hdf5: {ssi_fname}")
                # Convert syn_ssi_dict to Ensemble format
                ssi_ensemble = Ensemble(syn_ssi_dict)
                ssi_ensemble.to_hdf5(ssi_fname)

            # Combine drought data
            syn_droughts = pd.DataFrame()
            for rank_data in all_drought_data:
                for drought_df in rank_data:
                    syn_droughts = pd.concat([syn_droughts, drought_df], axis=0)

            # Save synthetic drought metrics
            syn_droughts.reset_index(inplace=True, drop=True)
            syn_fname = f"./pywrdrb/drought_metrics/{dataset_id}_ssi{ssi_window}_drought_events.csv"
            syn_droughts.to_csv(syn_fname, index=False)
            print(f"  Saved synthetic drought metrics: {syn_fname}")

    if rank == 0:
        print(f"\nAll drought metrics saved successfully for {dataset_id}!")

    return True



def main(dataset_id):
    """Main function for calculating synthetic drought metrics"""

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        print("=" * 60)
        print(f"SSI DROUGHT METRICS CALCULATION: {dataset_id}")
        print("=" * 60)

        # Create output directory if it doesn't exist
        os.makedirs("./pywrdrb/drought_metrics", exist_ok=True)

    # Calculate drought metrics (using default SSI windows)
    success = calculate_ssi_drought_metrics(dataset_id, ssi_windows=[3,6,12])

    if rank == 0:
        if success:
            print("=" * 60)
            print("SSI drought metrics calculation completed successfully!")
        else:
            print("=" * 60)
            print("ERROR: SSI drought metrics calculation failed!")
            sys.exit(1)


if __name__ == "__main__":

    # Check for 'historic' mode
    if len(sys.argv) == 2 and sys.argv[1].lower() == 'historic':
        # Calculate historic observed droughts only (no MPI needed)
        # Uses the shared function from methods.drought_analysis
        print("Running in HISTORIC mode - calculating observed droughts only")
        output_dir = "./pywrdrb/drought_metrics"
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
        print(f"    Available datasets: {list(DATASET_CONFIGS.keys())}")
        sys.exit(1)