"""
This script does the following:

- Load gauge flows from all ensemble set gage_flow_mgd files
- Load simulation results from all ensemble set outputs
- Combine the flows and simulation results from all sets into a single ensemble key
- Calculate additional metrics (shortages, contributions)
- Export combined data for analysis

Every dictionary in the final data object has format:

dict = {
    dataset_id: {
        realization_id : pd.DataFrame
    }
}

The pd.DataFrame has datetime index and node names as columns.
"""

import sys
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import pywrdrb
from sglib import SSIDroughtMetrics, SSI
from sglib import Ensemble, HDF5Manager

from methods.utils import combine_multiple_ensemble_sets_in_data
from methods.metrics.shortfall import get_flow_and_target_values, add_trenton_equiv_flow

from methods.load import load_drb_reconstruction, load_and_combine_ensemble_sets
from config import *

# MPI support (optional)
USE_MPI = True
try:
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
    if SIZE == 1:
        USE_MPI = False  # Running with single process, disable MPI
except ImportError:
    USE_MPI = False
    RANK = 0
    SIZE = 1
    COMM = None

def process_dataset(dataset_id):
    """
    Process and combine all ensemble sets for a given dataset

    Parameters:
    -----------
    dataset_id : str
        Dataset identifier to process
    """

    if RANK == 0:
        print(f"Processing dataset: {dataset_id}")

    dataset_config = DATASET_CONFIGS[dataset_id]
    dataset_type = dataset_config['type']

    # Get ensemble set specs for this dataset
    ensemble_set_specs = ENSEMBLE_SETS[dataset_id]

    # Check if all sets have been simulated (only rank 0)
    if RANK == 0:
        missing_sets = []
        for spec in ensemble_set_specs:
            if not os.path.exists(spec.output_file):
                missing_sets.append(spec.set_id + 1)

        if missing_sets:
            print(f"WARNING: Missing output files for sets: {missing_sets}")
            print("Run simulations first!")
            if USE_MPI:
                COMM.Abort(1)  # Abort all MPI ranks
            return False

    # Synchronize after check
    if USE_MPI:
        COMM.barrier()

    ### Load data through pywrdrb API #######################################
    ### OPTIMIZATION: Only rank 0 loads full data to avoid memory overload
    ### Other ranks will access data via direct HDF5 reads as needed
    if RANK == 0:
        print(f"Loading data for postprocessing...")
        print(f"  MPI mode: Only rank 0 loads full dataset to avoid memory overload with {SIZE} processes")
        print(f"  Other ranks will perform lazy HDF5 reads as needed")

    ## Setup pathnavigator (all ranks need this)
    pn_config = pywrdrb.get_pn_config()
    for spec in ensemble_set_specs:
        dataset_dir = spec.directory
        dataset_name = spec.directory.split('/')[-1]
        pn_config[f"flows/{dataset_name}"] = os.path.abspath(dataset_dir)
    pywrdrb.load_pn_config(pn_config)

    ## Only rank 0 loads full data
    if RANK == 0:
        if RANK == 0:
            print("  Rank 0: Loading hydrologic model flow...")

        ensemble_set_names = [spec.directory.split('/')[-1] for spec in ensemble_set_specs]
        results_sets = ['major_flow']
        data = pywrdrb.Data(results_sets=results_sets, print_status=False)
        data.load_hydrologic_model_flow(ensemble_set_names)

        # Combine all sets into single dataset key
        combined_gage_flow = {}
        for set_name in ensemble_set_names:
            set_data = data.major_flow[set_name]
            # Renumber realizations to be continuous across sets
            set_idx = int(set_name.split('_set')[-1]) - 1

            # Check if local IDs are 0-indexed or 1-indexed
            local_ids = list(set_data.keys())
            min_local_id = min(local_ids)

            for local_id, df in set_data.items():
                # Convert to 0-indexed if needed (local IDs might be 1-100 instead of 0-99)
                local_id_normalized = local_id - min_local_id
                global_id = set_idx * N_REALIZATIONS_PER_ENSEMBLE_SET + local_id_normalized
                combined_gage_flow[global_id] = df

        # Store combined gage flow
        gage_flow_dict = {dataset_id: combined_gage_flow}

        print("  Rank 0: Loading simulation outputs...")

        output_filenames = [spec.output_file for spec in ensemble_set_specs]
        output_filenames.append(RECONSTRUCTION_OUTPUT_FNAME)

        results_sets = [
            "major_flow",
            "inflow",
            "res_storage",
            "res_release",
            "mrf_target",
            "ibt_diversions",
            "ibt_demands",
            "nyc_release_components"
        ]

        data = pywrdrb.Data(results_sets=results_sets, print_status=False)
        data.load_output(output_filenames=output_filenames)
        data.load_observations(results_sets=['res_storage', 'major_flow', 'reservoir_downstream_gage'])
        data.res_release['obs'] = {}
        data.res_release['obs'][0] = data.reservoir_downstream_gage['obs'][0]

        data = add_trenton_equiv_flow(data)

        # Combine all sets into single dataset key for each results_set
        for results_set in results_sets:
            combined_data = {}
            full_results_set_dict = getattr(data, results_set)

            for i, spec in enumerate(ensemble_set_specs):
                set_name = f"{dataset_id}_set{i+1}"
                if set_name not in full_results_set_dict:
                    print(f"WARNING: {set_name} not found in {results_set}")
                    continue

                set_data = full_results_set_dict[set_name]

                # Check if local IDs are 0-indexed or 1-indexed
                local_ids = list(set_data.keys())
                if local_ids:
                    min_local_id = min(local_ids)
                else:
                    min_local_id = 0

                # Renumber realizations to be continuous
                for local_id, df in set_data.items():
                    # Convert to 0-indexed if needed
                    local_id_normalized = local_id - min_local_id
                    global_id = i * N_REALIZATIONS_PER_ENSEMBLE_SET + local_id_normalized
                    combined_data[global_id] = df

            # Store combined data back
            full_results_set_dict[dataset_id] = combined_data
            setattr(data, results_set, full_results_set_dict)

        # Replace gage flow with combined version
        data.major_flow[dataset_id] = gage_flow_dict[dataset_id]

        print("  Rank 0: Data loading complete")
    else:
        # Other ranks: Create empty data object
        # They will NOT load full data to save memory
        data = None

    # Sync all ranks
    if USE_MPI:
        COMM.barrier()
    
    ### Post-process data ##############################################
    if RANK == 0:
        print('Calculating shortages for different nodes...')
        if USE_MPI:
            print(f'  WARNING: Only rank 0 has data loaded. Processing will be serial.')
            print(f'  (With {SIZE} ranks, each would need {8}GB RAM = {SIZE * 8}GB total)')
            print(f'  Recommend running with fewer ranks (20-40) for MPI parallelization')

    # With current memory constraints, only rank 0 does the work
    all_shortage_dict = {}

    if RANK == 0:
        for model in ['reconstruction', dataset_id]:
            realizations = list(data.major_flow[model].keys())

            print(f"  Processing model: {model}")
            print(f"    Total realizations: {len(realizations)}")

            shortage_dict = {}

            # Initialize shortage dict for all realizations
            for r in realizations:
                shortage_dict[r] = {}

            # Process each node
            nodes = ['delMontague', 'delTrenton', 'nyc', 'nj']
            for node in nodes:
                print(f"    Processing {node}...")

                for i, r in enumerate(realizations):
                    # Progress reporting (every 10%)
                    if len(realizations) > 10 and (i + 1) % max(1, len(realizations) // 10) == 0:
                        progress = 100 * (i + 1) / len(realizations)
                        print(f"      Progress: {progress:.0f}% ({i+1}/{len(realizations)} realizations)")

                    flow_series, target_series = get_flow_and_target_values(
                        data, node, model, r,
                        start_date=None, end_date=None
                    )

                    # Calculate shortages
                    shortage_series = target_series - flow_series
                    shortage_series[shortage_series < 0] = 0  # Set negative shortages (surplus) to zero
                    shortage_series.iloc[:3] = 0.0  # Set first 3 days to 0.0 due to model warmup

                    # Ignore shortages when duration of consecutive shortage>0 days is <3
                    shortage_durations = (shortage_series > 0).astype(int).groupby(
                        (shortage_series > 0).astype(int).diff().ne(0).cumsum()
                    ).cumsum()
                    shortage_series[shortage_durations < 3] = 0.0

                    shortage_dict[r][node] = shortage_series

            # Convert to DataFrames
            for r in realizations:
                shortage_dict[r] = pd.DataFrame(shortage_dict[r])

            all_shortage_dict[model] = shortage_dict
            print(f"    Completed shortage calculations for {model}")

    # Sync all ranks before continuing
    if USE_MPI:
        COMM.barrier()
    
    ## Calculate downstream contributions from NYC reservoirs (only rank 0)
    if RANK == 0:
        print('Calculating total downstream contributions...')

        nyc_reservoirs = ['cannonsville', 'pepacton', 'neversink']
        contribution_columns = [f'mrf_montagueTrenton_{res}' for res in nyc_reservoirs]
        all_contribution_dict = {}

        for model in ['reconstruction', dataset_id]:
            contribution_dict = {}
            realizations = list(data.major_flow[model].keys())

            for r in realizations:
                total_nyc_contribution = data.nyc_release_components[model][r].loc[:, contribution_columns].sum(axis=1)
                contribution_dict[r] = total_nyc_contribution.to_frame(name='mrf_montagueTrenton_nyc')

            all_contribution_dict[model] = contribution_dict

    ### Calculate aggregate NYC inflow (only rank 0)
    if RANK == 0:
        print('Calculating aggregate NYC inflow...')

        for model in ['reconstruction', dataset_id]:
            realizations = list(data.inflow[model].keys())
            for r in realizations:
                data.inflow[model][r].loc[:, 'nyc'] = data.inflow[model][r].loc[:, nyc_reservoirs].sum(axis=1)

    ### Organize data to be kept for later (only rank 0)
    if RANK == 0:
        print('Organizing final data structure...')

        keep_data = pywrdrb.Data()
        keep_data.gage_flow = gage_flow_dict
        keep_data.shortage = all_shortage_dict
        keep_data.contribution = all_contribution_dict

        # Make copies of output results_sets just for the combined dataset
        inflow_dict = {}
        major_flow_dict = {}
        res_storage_dict = {}
        ibt_diversions_dict = {}
        ibt_demands_dict = {}
        mrf_target_dict = {}

        for model in ['reconstruction', dataset_id]:
            if model in data.inflow:
                inflow_dict[model] = data.inflow[model]
                major_flow_dict[model] = data.major_flow[model]
                res_storage_dict[model] = data.res_storage[model]
                ibt_diversions_dict[model] = data.ibt_diversions[model]
                ibt_demands_dict[model] = data.ibt_demands[model]
                mrf_target_dict[model] = data.mrf_target[model]

        keep_data.inflow = inflow_dict
        keep_data.major_flow = major_flow_dict
        keep_data.res_storage = res_storage_dict
        keep_data.ibt_diversions = ibt_diversions_dict
        keep_data.ibt_demands = ibt_demands_dict
        keep_data.mrf_target = mrf_target_dict

    ### Export the new data object to HDF5 (only on rank 0)
    if RANK == 0:
        fname = f'./pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5'
        print(f"Exporting combined data to {fname}...")
        keep_data.export(fname)
        print(f"Successfully processed and saved data for {dataset_id}!")

    # Synchronize all ranks before returning
    if USE_MPI:
        COMM.barrier()

    return True


def verify_postprocessing_output(dataset_id):
    """
    Verify that postprocessing output exists and is valid
    
    Parameters:
    -----------
    dataset_id : str
        Dataset identifier to verify
    """
    
    fname = f'./pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5'
    
    if not os.path.exists(fname):
        print(f"FAIL: Output file not found: {fname}")
        return False
    
    # Check file size
    file_size = os.path.getsize(fname)
    if file_size < 1024 * 1024:  # Less than 1MB is suspicious
        print(f"WARNING: Output file seems too small ({file_size} bytes)")
        return False
    
    # Try to load and verify structure
    try:
        test_data = pywrdrb.Data()
        test_data.load_from_export(fname)
        
        # Check that key results exist
        expected_attrs = ['gage_flow', 'shortage', 'major_flow', 'res_storage']
        missing_attrs = []
        for attr in expected_attrs:
            if not hasattr(test_data, attr):
                missing_attrs.append(attr)
        
        if missing_attrs:
            print(f"WARNING: Missing expected attributes: {missing_attrs}")
            return False
        
        # Check realization count
        if dataset_id in test_data.major_flow:
            n_realizations = len(test_data.major_flow[dataset_id])
            if n_realizations != TOTAL_REALIZATIONS:
                print(f"WARNING: Expected {TOTAL_REALIZATIONS} realizations, found {n_realizations}")
                return False
        
        print(f"SUCCESS: Postprocessed data verified ({file_size//1024//1024} MB, {n_realizations} realizations)")
        return True
        
    except Exception as e:
        print(f"FAIL: Error loading postprocessed data - {str(e)}")
        return False


def main(dataset_id):
    """Main function"""

    if RANK == 0:
        print("=" * 60)
        print(f"POSTPROCESSING ENSEMBLE DATA: {dataset_id}")
        print("=" * 60)

    # Verify dataset
    verify_dataset_id(dataset_id)
    dataset_config = DATASET_CONFIGS[dataset_id]

    if RANK == 0:
        print(f"Dataset type: {dataset_config['type']}")
        print(f"Description: {dataset_config['description']}")
        print(f"Total realizations: {TOTAL_REALIZATIONS}")
        print(f"Ensemble sets: {N_ENSEMBLE_SETS}")
        if USE_MPI:
            print(f"MPI: {SIZE} processes")
        print("=" * 60)

    # Process the dataset
    success = process_dataset(dataset_id)

    if RANK == 0:
        if success:
            # Verify output
            verify_postprocessing_output(dataset_id)

        print("=" * 60)
        print(f"Postprocessing {'completed successfully' if success else 'failed'}!")


if __name__ == "__main__":
    
    # Get the dataset_id from command line arguments
    if len(sys.argv) != 2:
        print("Usage: python 04_postprocess_data.py <dataset_id>")
        print(f"Available datasets: {list(DATASET_CONFIGS.keys())}")
        sys.exit(1)
    
    dataset_id = sys.argv[1]
    verify_dataset_id(dataset_id)
    
    main(dataset_id)