"""
Core function for generating synthetic flow ensembles using Kirsch-Nowak methods.
This module contains the ensemble generation logic that can be used in both
serial and parallel modes.
"""

import os
import numpy as np
import pandas as pd
import scipy.stats as stats

from pywrdrb.pre.flows import _subtract_upstream_catchment_inflows
from pywrdrb.pywr_drb_node_data import immediate_downstream_nodes_dict
from pywrdrb.pywr_drb_node_data import downstream_node_lags

from synhydro.methods.generation.nonparametric.kirsch import KirschGenerator
from synhydro.methods.disaggregation.temporal.nowak import NowakDisaggregator
from synhydro.core.ensemble import Ensemble

from methods.load import load_baseline_historical_flow
from methods.config import (
    DATASET_CONFIGS,
    N_YEARS_GENERATE,
    START_DATE,
    END_DATE,
    BASELINE_DATASET,
    pywrdrb_nodes_to_generate,
    pywrdrb_nodes_to_regress
)
from methods.ensemble_utils import get_ensemble_set_spec

from methods.mpi_utils import (
    MPI_AVAILABLE,
    point_to_point_bcast,
    point_to_point_gather,
    point_to_point_barrier,
)

# Conditional MPI import
if MPI_AVAILABLE:
    from mpi4py import MPI


def generate_ensemble_set(set_id, dataset_id, use_mpi=True,
                          comm=None, local_rank=None, local_size=None,
                          set_peer_ranks=None, preloaded_data=None):
    """
    Generate a single ensemble set with optional MPI distribution

    Parameters:
    -----------
    set_id : int
        Ensemble set identifier (0-indexed)
    dataset_id : str
        Dataset identifier (e.g., 'stationary_ensemble', 'climate_adjusted_ssp245_min')
    use_mpi : bool
        If True, use MPI for parallel execution. If False, run serially.
    comm : MPI.Comm or None
        MPI communicator (COMM_WORLD). Required when local_rank/local_size are provided.
    local_rank : int or None
        This rank's position within its ensemble set group.
        If None, falls back to COMM_WORLD rank/size.
    local_size : int or None
        Number of ranks collaborating on this ensemble set.
        If None, falls back to COMM_WORLD rank/size.
    set_peer_ranks : list of int or None
        Global MPI ranks assigned to the same ensemble set.
        Used for point-to-point communication. If None, uses COMM_WORLD collectives.
    preloaded_data : tuple or None
        Pre-loaded (Q, Q_baseline, Q_inflow) DataFrames to skip redundant I/O.

    Returns:
    --------
    bool
        True if successful, False otherwise
    """

    # Determine MPI context
    if use_mpi and MPI_AVAILABLE and local_rank is not None:
        # Called from a script that already determined set assignments
        rank = local_rank
        size = local_size
    elif use_mpi and MPI_AVAILABLE:
        # Legacy / standalone mode: use COMM_WORLD directly
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        set_peer_ranks = None
    else:
        # Serial mode
        comm = None
        rank = 0
        size = 1
        set_peer_ranks = None

    # Get ensemble set specification
    set_spec = get_ensemble_set_spec(set_id, dataset_id)
    dataset_config = DATASET_CONFIGS[dataset_id]
    set_realization_ids = set_spec.realizations
    n_realizations = set_spec.n_realizations
    output_dir = set_spec.directory

    if rank == 0:
        print(f"Set {set_id + 1}: Generating {dataset_id} ({n_realizations} realizations, {size} ranks)")

    # Ensure output directory exists
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
    if use_mpi and comm:
        if set_peer_ranks is not None:
            point_to_point_barrier(comm, rank, size, set_peer_ranks)
        else:
            comm.Barrier()

    # Load data — use preloaded_data if provided (already broadcast by caller)
    if preloaded_data is not None:
        Q, Q_baseline, Q_inflow = preloaded_data
    else:
        # Load and broadcast data (optimized: only rank 0 loads)
        if rank == 0:
            # Full Q is 1945-01-01 through 2023-12-31
            Q = load_baseline_historical_flow(gage_flow=True, period='full',
                                              flowtype=BASELINE_DATASET)
            # Baseline is 1980-01-01 through 2019-12-31
            Q_baseline = load_baseline_historical_flow(gage_flow=True, period='baseline',
                                                       flowtype=BASELINE_DATASET)
            Q_inflow = load_baseline_historical_flow(gage_flow=False,
                                                     period='full',
                                                     flowtype=BASELINE_DATASET)
            Q = Q.loc[:, pywrdrb_nodes_to_generate]
            Q_baseline = Q_baseline.loc[:, pywrdrb_nodes_to_generate]

            # CRITICAL: Replace zeros with NaN before fitting
            # Streamflow should never be exactly zero - zeros in historical data are artifacts
            # that would otherwise propagate to synthetic ensembles via Nowak disaggregation
            n_zeros_before = (Q == 0.0).sum().sum()
            if n_zeros_before > 0:
                print(f"Set {set_id + 1}: Replacing {n_zeros_before} zero values with NaN (physically unrealistic)")
                Q.replace(0, np.nan, inplace=True)
                Q_inflow.replace(0, np.nan, inplace=True)
                Q_baseline.replace(0, np.nan, inplace=True)
        else:
            Q = None
            Q_inflow = None
            Q_baseline = None

        # Broadcast data (only needed in MPI mode)
        if use_mpi and comm:
            if set_peer_ranks is not None:
                Q = point_to_point_bcast(comm, Q, rank, set_peer_ranks)
                Q_inflow = point_to_point_bcast(comm, Q_inflow, rank, set_peer_ranks)
                Q_baseline = point_to_point_bcast(comm, Q_baseline, rank, set_peer_ranks)
            else:
                Q = comm.bcast(Q, root=0)
                Q_inflow = comm.bcast(Q_inflow, root=0)
                Q_baseline = comm.bcast(Q_baseline, root=0)

    # Fit Kirsch generator (monthly) and Nowak disaggregator (monthly->daily)
    # (parallel efficiency: all ranks fit independently to avoid broadcast of large object)
    kirsch_gen_baseline = KirschGenerator(debug=False, generate_using_log_flow=True)
    kirsch_gen_baseline.preprocessing(Q_baseline)
    kirsch_gen_baseline.fit()

    nowak_disagg = NowakDisaggregator(debug=False)
    nowak_disagg.preprocessing(Q)
    nowak_disagg.fit()

    # Apply climate adjustments if needed
    if dataset_config['type'] == 'climate_adjusted':
        if rank == 0:
            print(f'Set {set_id + 1}: Applying climate adjustments for {dataset_id}...')

        monthly_prc_change = dataset_config['monthly_prc_change']
        if monthly_prc_change is None or len(monthly_prc_change) != 12:
            raise ValueError(f"Dataset {dataset_id} missing valid monthly_prc_change (need 12 values)")

        # Apply percentage changes to monthly means
        prior_mean_month = kirsch_gen_baseline.mean_month
        new_mean_month = prior_mean_month.copy() * pd.NA

        for i, site in enumerate(new_mean_month):
            # Convert from log scale, apply percentage change, convert back
            # The kirsch_gen stores means in log scale
            new_mean_month.loc[:, site] = np.exp(prior_mean_month.loc[:, site]) * (1 + np.array(monthly_prc_change) / 100.0)

        # Convert back to log scale
        new_mean_month = np.log(new_mean_month.astype(float))

        # Pass back to generator, overwriting the prior means
        kirsch_gen_baseline.mean_month = new_mean_month

    # DISTRIBUTE REALIZATION GENERATION ACROSS RANKS
    # Each rank generates a subset of realizations
    realizations_per_rank = n_realizations // size
    extra_realizations = n_realizations % size

    # Calculate how many realizations this rank should generate
    if rank < extra_realizations:
        local_n_realizations = realizations_per_rank + 1
        local_start_idx = rank * (realizations_per_rank + 1)
    else:
        local_n_realizations = realizations_per_rank
        local_start_idx = rank * realizations_per_rank + extra_realizations

    # Generate local ensemble subset (monthly, then disaggregate to daily)
    # Seed strategy: deterministic per (set_id, rank) so results are reproducible
    # and independent across ranks. Kirsch sets np.random.seed() internally,
    # which also controls the Nowak disaggregator (uses global numpy state).
    generation_seed = set_id * 10000 + rank
    if local_n_realizations > 0:
        # Step 1: Generate monthly flows using Kirsch
        monthly_ensemble_obj = kirsch_gen_baseline.generate(n_realizations=local_n_realizations,
                                                    n_years=N_YEARS_GENERATE,
                                                    seed=generation_seed)

        # Step 2: Disaggregate monthly flows to daily using Nowak
        # Nowak uses global numpy random state (set by Kirsch seed above)
        daily_ensemble_obj = nowak_disagg.disaggregate(monthly_ensemble_obj)

        # Convert Ensemble object to dictionary of DataFrames (by realization)
        local_syn_ensemble = daily_ensemble_obj.data_by_realization
    else:
        local_syn_ensemble = {}

    # Fit KDEs for non-major nodes (optimized: only fit once on rank 0, broadcast parameters)
    if rank == 0:
        kde_params = {}  # Store parameters instead of full KDE objects
        for upstream in pywrdrb_nodes_to_generate:
            downstream = immediate_downstream_nodes_dict[upstream]
            if downstream not in pywrdrb_nodes_to_regress:
                continue

            xs = Q_inflow.loc[:, upstream]
            ys = Q_inflow.loc[:, downstream]
            frac = ys / xs
            frac = frac[~np.isnan(frac)]
            # filter infs
            frac = frac[np.isfinite(frac)]

            # Fit KDE and extract parameters
            kde = stats.gaussian_kde(frac)
            kde_name = f"{upstream}_to_{downstream}"
            # Store dataset and covariance factor from baseline data
            kde_params[kde_name] = {
                'dataset': kde.dataset,
                'covariance_factor': kde.covariance_factor()
            }
    else:
        kde_params = None

    # Broadcast KDE parameters (only needed in MPI mode)
    if use_mpi and comm:
        if set_peer_ranks is not None:
            kde_params = point_to_point_bcast(comm, kde_params, rank, set_peer_ranks)
        else:
            kde_params = comm.bcast(kde_params, root=0)

    # Reconstruct KDEs from parameters on all ranks
    kdes = {}
    for kde_name, params in kde_params.items():
        kde = stats.gaussian_kde(params['dataset'])
        kde.set_bandwidth(bw_method=params['covariance_factor'])
        kdes[kde_name] = kde

    # Generate flows at non-major nodes for local ensemble
    if local_n_realizations > 0:
        n_local_samples = local_syn_ensemble[0].shape[0] * local_n_realizations
        local_realization_ids = list(local_syn_ensemble.keys())

        for upstream in pywrdrb_nodes_to_generate:
            downstream = immediate_downstream_nodes_dict[upstream]
            if downstream not in pywrdrb_nodes_to_regress:
                continue

            kde_name = f"{upstream}_to_{downstream}"
            kde = kdes[kde_name]

            # Generate samples for local realizations
            # Seed per (set_id, rank, node-pair) for reproducibility
            kde_seed = generation_seed + hash(kde_name) % 10000
            samples = kde.resample(n_local_samples, seed=kde_seed)
            samples = samples.reshape((local_syn_ensemble[0].shape[0], local_n_realizations))
            samples = np.clip(samples, 0, 1)

            for i, realization in enumerate(local_realization_ids):
                upstream_flow = local_syn_ensemble[realization][upstream].values
                downstream_inflow = upstream_flow * samples[:, i]

                # Account for lag
                lag = downstream_node_lags[downstream]
                if lag > 0:
                    downstream_gage_flow = downstream_inflow.copy()
                    downstream_gage_flow[lag:] += upstream_flow[:-lag]
                    downstream_gage_flow[:lag] += upstream_flow[:lag]
                else:
                    downstream_gage_flow = downstream_inflow + upstream_flow

                local_syn_ensemble[realization][downstream] = downstream_gage_flow

    # Create marginal catchment inflows for local ensemble
    if local_n_realizations > 0:
        local_inflow_ensemble = {}
        for real in local_syn_ensemble:
            local_syn_ensemble[real]['delTrenton'] = 0.0
            flows_i = local_syn_ensemble[real].copy()
            date_idx = pd.date_range(start=START_DATE, periods=len(flows_i), freq='D')
            flows_i.index = date_idx
            flows_i = flows_i.loc[:END_DATE]

            # Trim gage flow ensemble to match
            local_syn_ensemble[real].index = date_idx
            local_syn_ensemble[real] = local_syn_ensemble[real].loc[:END_DATE]

            local_inflow_ensemble[real] = _subtract_upstream_catchment_inflows(flows_i)
    else:
        local_inflow_ensemble = {}

    # GATHER ALL LOCAL ENSEMBLES TO RANK 0
    # Gather data (only needed in MPI mode)
    if use_mpi and comm:
        if set_peer_ranks is not None:
            all_syn_ensembles = point_to_point_gather(
                comm, local_syn_ensemble, rank, size, set_peer_ranks)
            all_inflow_ensembles = point_to_point_gather(
                comm, local_inflow_ensemble, rank, size, set_peer_ranks)
        else:
            all_syn_ensembles = comm.gather(local_syn_ensemble, root=0)
            all_inflow_ensembles = comm.gather(local_inflow_ensemble, root=0)
    else:
        # Serial mode: just wrap in list
        all_syn_ensembles = [local_syn_ensemble]
        all_inflow_ensembles = [local_inflow_ensemble]

    # COMBINE AND SAVE ON RANK 0 ONLY
    if rank == 0:

        # Combine all local ensembles
        combined_syn_ensemble = {}
        combined_inflow_ensemble = {}

        global_realization_counter = 0
        for rank_syn, rank_inflow in zip(all_syn_ensembles, all_inflow_ensembles):
            for local_real_id in rank_syn:
                # Use the actual global realization ID from set_realization_ids
                global_real_id = set_realization_ids[global_realization_counter]
                combined_syn_ensemble[global_real_id] = rank_syn[local_real_id]
                combined_inflow_ensemble[global_real_id] = rank_inflow[local_real_id]
                global_realization_counter += 1

        # Verify we have the correct number of realizations
        if len(combined_syn_ensemble) != n_realizations:
            print(f"Error: Expected {n_realizations} realizations, got {len(combined_syn_ensemble)}")
            return False

        # Get datetime index from first realization
        combined_inflow_ensemble_real_ids = list(combined_inflow_ensemble.keys())
        syn_datetime = combined_inflow_ensemble[combined_inflow_ensemble_real_ids[0]].index
        sites = combined_inflow_ensemble[combined_inflow_ensemble_real_ids[0]].columns

        # Reorganize data structure (optimized: pre-allocate arrays)
        Q_syn = {}
        Qs_inflows = {}

        for site in sites:
            Q_syn[site] = np.zeros((len(syn_datetime), n_realizations), dtype=np.float32)  # Use float32 to save memory
            Qs_inflows[site] = np.zeros((len(syn_datetime), n_realizations), dtype=np.float32)

            # Trenton flows are the same as DRCanal location
            if site == 'delTrenton':
                use_site = 'delDRCanal'
            else:
                use_site = site

            for i, global_real_id in enumerate(set_realization_ids):
                Q_syn[site][:, i] = combined_syn_ensemble[global_real_id][use_site].values
                Qs_inflows[site][:, i] = combined_inflow_ensemble[global_real_id][use_site].values

            # Convert to DataFrame with realization IDs
            # IMPORTANT: Use set-specific realization IDs
            real_cols = [str(i) for i in set_realization_ids]

            Q_syn[site] = pd.DataFrame(Q_syn[site],
                                       index=syn_datetime,
                                       columns=real_cols)
            Qs_inflows[site] = pd.DataFrame(Qs_inflows[site],
                                            index=syn_datetime,
                                            columns=real_cols)

        gage_flow_fname = set_spec.files['gage_flow']
        catchment_inflow_fname = set_spec.files['catchment_inflow']

        # Create Ensemble objects and save to HDF5
        # Convert site-based dict to realization-based dict for Ensemble
        gage_flow_ensemble_dict = {}
        inflow_ensemble_dict = {}

        for i, real_id in enumerate(set_realization_ids):
            real_col = str(real_id)
            gage_flow_ensemble_dict[int(real_id)] = pd.DataFrame({
                site: Q_syn[site][real_col] for site in sites
            }, index=syn_datetime)
            inflow_ensemble_dict[int(real_id)] = pd.DataFrame({
                site: Qs_inflows[site][real_col] for site in sites
            }, index=syn_datetime)

        gage_flow_ensemble = Ensemble(gage_flow_ensemble_dict)
        inflow_ensemble = Ensemble(inflow_ensemble_dict)

        gage_flow_ensemble.to_hdf5(gage_flow_fname)
        inflow_ensemble.to_hdf5(catchment_inflow_fname)

        print(f"Set {set_id + 1}: Complete.")

        return True

    return True
