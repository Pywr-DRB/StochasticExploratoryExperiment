import os
import h5py
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import pywrdrb
from pywrdrb.path_manager import get_pn_object
from pywrdrb.utils.constants import cfs_to_mgd
from synhydro.core.ensemble import Ensemble
from methods.config import (
    RECONSTRUCTION_OUTPUT_FNAME, WRFAORC_OUTPUT_FNAME, WRF1960s_OUTPUT_FNAME,
    ROOT_DIR, OUTPUT_DIR,
    DROUGHT_METRICS_DIR, PERFORMANCE_METRICS_DIR, EVENT_METRICS_DIR,
    ZONE_PROB_DIR, SATISFICING_DIR,
    N_ENSEMBLE_SETS, N_REALIZATIONS_PER_ENSEMBLE_SET,
    N_YEARS,
)
from methods.ensemble_utils import get_ensemble_set_spec, ENSEMBLE_SETS
from methods.metrics.shortfall import (
    get_flow_and_target_values, add_trenton_equiv_flow,
    calculate_shortage_series,
)
from methods.metrics.satisficing import add_satisficing_category

file_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = f"{file_dir}/../data"

# DROUGHT_METRICS_DIR is now imported from methods.config


def get_realization_ids_from_export(fname, dataset_id, results_set='shortage'):
    """Read realization IDs from a postprocessed HDF5 export without loading data.

    Opens the HDF5 store, scans keys matching /{results_set}/{dataset_id}/*,
    and returns sorted integer realization IDs. This is a fast metadata-only
    operation — no DataFrames are deserialized.

    Parameters
    ----------
    fname : str
        Path to the postprocessed HDF5 file.
    dataset_id : str
        Dataset identifier (e.g., 'stationary_ensemble').
    results_set : str
        Which results set to scan keys from (default 'shortage').

    Returns
    -------
    list
        Sorted list of realization IDs (integers where possible).
    """
    ids = []
    with pd.HDFStore(fname, mode='r') as store:
        prefix = f'/{results_set}/{dataset_id}/'
        for key in store.keys():
            if key.startswith(prefix):
                scenario_id = key.split('/')[-1]
                try:
                    ids.append(int(scenario_id))
                except ValueError:
                    ids.append(scenario_id)
    return sorted(ids)


def load_rank_subset_from_export(fname, realization_ids, results_sets,
                                  rank=0, size=1, stagger_seconds=0.01):
    """Load specific realizations from a postprocessed HDF5 export.

    Each rank loads only its assigned realizations directly from HDF5,
    with a small I/O stagger to avoid overwhelming the parallel filesystem
    when hundreds of ranks open the same file simultaneously.

    Parameters
    ----------
    fname : str
        Path to the postprocessed HDF5 file.
    realization_ids : list
        Realization IDs this rank should load.
    results_sets : list
        Which results sets to load (e.g., ['inflow', 'res_storage']).
    rank : int
        MPI rank (used for stagger timing and logging).
    size : int
        Total MPI ranks (stagger disabled when size <= 1).
    stagger_seconds : float
        Delay per rank in seconds (default 0.01s = 3.2s total for 320 ranks).

    Returns
    -------
    pywrdrb.Data
        Data object containing only the requested realizations.
    """
    import time

    # Stagger file access to avoid I/O contention on parallel filesystem
    if size > 1 and rank > 0:
        time.sleep(stagger_seconds * rank)

    data = pywrdrb.Data()
    data.load_from_export(fname, results_sets=results_sets,
                          realizations=realization_ids)
    return data

def load_annual_metrics(dataset_id):
    """
    Load annual performance metrics CSV.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier

    Returns
    -------
    pd.DataFrame
        Rows: (realization_id, water_year, period).
        Columns: 20 metrics + 3 annotation columns.
    """
    csv_file = f"{PERFORMANCE_METRICS_DIR}/{dataset_id}_annual_metrics.csv"

    if not os.path.exists(csv_file):
        raise FileNotFoundError(
            f"Annual metrics not found: {csv_file}\n"
            f"Run 06_calculate_performance_metrics.py first!"
        )

    return pd.read_csv(csv_file)


def load_hashimoto_metrics(dataset_id):
    """
    Load Hashimoto simulation-level metrics CSV.

    Returns
    -------
    pd.DataFrame
        One row per realization with reliability/resiliency for Montague and Trenton.
    """
    csv_file = f"{PERFORMANCE_METRICS_DIR}/{dataset_id}_hashimoto_metrics.csv"
    if not os.path.exists(csv_file):
        raise FileNotFoundError(
            f"Hashimoto metrics not found: {csv_file}\n"
            f"Run 06_calculate_performance_metrics.py first!"
        )
    return pd.read_csv(csv_file)


def load_hashimoto_events(dataset_id):
    """
    Load Hashimoto per-shortage-event CSV.

    Returns
    -------
    pd.DataFrame
        One row per shortage event per location per realization.
    """
    csv_file = f"{PERFORMANCE_METRICS_DIR}/{dataset_id}_hashimoto_shortage_events.csv"
    if not os.path.exists(csv_file):
        raise FileNotFoundError(
            f"Hashimoto events not found: {csv_file}\n"
            f"Run 06_calculate_performance_metrics.py first!"
        )
    return pd.read_csv(csv_file)


def load_baseline_historical_flow(gage_flow=True, 
                                  period='full',
                                  flowtype='pub_nhmv10_BC_withObsScaled'):
    """
    Load the baseline historical data.

    Returns:
        pd.DataFrame: DataFrame containing the baseline historical data.
    """
    
    assert period in ['baseline', 'full'], "Period must be 'baseline' or 'full'"
    
    flowtype_options = [
        'pub_nhmv10_BC_withObsScaled',
        'wrfaorc_withObsScaled'
    ]
    
    if flowtype not in flowtype_options:
        raise ValueError(f"Invalid flowtype: {flowtype}. Must be one of {flowtype_options}")
    
    # pywrdrb path manager object
    pn = get_pn_object()
    
    if gage_flow:
        fname = str(pn.sc.get(f"flows/{flowtype}") / "gage_flow_mgd.csv")
    else:
        fname = str(pn.sc.get(f"flows/{flowtype}") / "catchment_inflow_mgd.csv")
    
    Q = pd.read_csv(fname, index_col=0, parse_dates=True)
    Q.index = pd.to_datetime(Q.index)
    
    if period == 'baseline':
        # Baseline period is 1980-01-01 to 2019-12-31
        Q = Q.loc['1980-01-01':'2019-12-31', :]
    elif period == 'full':
        # Full period is the entire available data
        Q = Q.loc[:, :]
    
    return Q


def load_wrf1960s_historical_flow(gage_flow=True):
    """
    Load the WRF 1960s historical data.

    Returns:
        pd.DataFrame: DataFrame containing the WRF 1960s historical data.
    """
    flowtype = 'wrf1960s_calib_nlcd2016'
    
    # pywrdrb path manager object
    pn = get_pn_object()
    
    if gage_flow:
        fname = str(pn.sc.get(f"flows/{flowtype}") / "gage_flow_mgd.csv")
    else:
        fname = str(pn.sc.get(f"flows/{flowtype}") / "catchment_inflow_mgd.csv")
    
    Q = pd.read_csv(fname, index_col=0, parse_dates=True)
    Q.index = pd.to_datetime(Q.index)
    
    return Q



def load_drought_events(dataset_id, ssi_window, observed=False, filter_extreme=False):
    """
    Load drought events for a given dataset and SSI window.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    ssi_window : int
        SSI window (3, 6, or 12 months)
    observed : bool
        If True, load observed droughts instead of ensemble droughts
    filter_extreme : bool
        If True, remove droughts with |severity| > 6.0

    Returns
    -------
    pd.DataFrame
        Drought events with date columns converted to datetime
    """
    if observed:
        fname = f"{DROUGHT_METRICS_DIR}/observed_ssi{ssi_window}_drought_events.csv"
    else:
        fname = f"{DROUGHT_METRICS_DIR}/{dataset_id}_ssi{ssi_window}_drought_events.csv"

    if not os.path.exists(fname):
        raise FileNotFoundError(
            f"Drought events file not found: {fname}\n"
            f"Run 05_calculate_ssi_drought_metrics.py first!"
        )

    print(f"Loading drought events from: {fname}")
    droughts = pd.read_csv(fname)

    # Convert date columns
    date_cols = ['start', 'end', 'max_severity_date']
    for col in date_cols:
        if col in droughts.columns:
            droughts[col] = pd.to_datetime(droughts[col])

    # Take absolute values for severity and magnitude
    for metric in ['severity', 'magnitude']:
        if metric in droughts.columns:
            droughts[metric] = droughts[metric].abs()

    # Remove infinite or NaN values
    droughts = droughts.replace([np.inf, -np.inf], np.nan).dropna(
        subset=['severity', 'magnitude', 'duration']
    )

    # Optionally filter extreme values
    if filter_extreme:
        n_before = len(droughts)
        droughts = droughts[droughts['severity'] <= 6.0]
        n_after = len(droughts)
        if n_before > n_after:
            print(f"  Removed {n_before - n_after} droughts with |severity| > 6.0")

    print(f"  Loaded {len(droughts):,} drought events")
    if 'realization_id' in droughts.columns:
        print(f"  Unique realizations: {droughts['realization_id'].nunique()}")

    return droughts


def load_ffmp_boundaries():
    """
    Load FFMP level boundaries from reconstruction data.

    This function uses a cache to avoid reloading the data multiple times
    within a single script execution. The boundaries are converted from
    fraction to percentage (0-100 scale).

    Returns
    -------
    pd.DataFrame
        FFMP level boundaries as percentages (0-100) for NYC reservoir system

    Examples
    --------
    >>> boundaries = load_ffmp_boundaries()
    >>> boundaries.columns
    Index(['L1a', 'L1b', 'L2', 'L3', 'L4', 'L5', 'drought', 'normal'], dtype='object')
    """
    if not hasattr(load_ffmp_boundaries, '_cache'):
        print("Loading FFMP level boundaries from reconstruction...")
        ffmp_data = pywrdrb.Data(results_sets=["ffmp_level_boundaries"])
        ffmp_data.load_output(output_filenames=[RECONSTRUCTION_OUTPUT_FNAME])
        boundaries = ffmp_data.ffmp_level_boundaries['reconstruction'][0] * 100  # Convert to %
        load_ffmp_boundaries._cache = boundaries
    return load_ffmp_boundaries._cache


def load_and_combine_ensemble_sets(ensemble_sets,
                                   by_site = True):
    """
    Load and combine all ensemble set data into a single dictionary.

    WARNING:
    This should only be used when the realizations do NOT matter.
    In this function, all realizations are combined and renumbered
    without regard to their original set IDs.

    Parameters:
    - ensemble_sets: List of ensemble set specifications.

    Returns:
    - Combined dict.
    """
    all_data = {}
    realization_id = 0
    for i, set_spec in enumerate(ensemble_sets):
        gageflow_set_file = set_spec.files['gage_flow']
        set_realization_ids = set_spec.realization_ids

        ensemble_set_data = Ensemble.from_hdf5(gageflow_set_file)

        if by_site:
            # extract just the data by site
            Qs_gageflow = ensemble_set_data.data_by_site

            # add to all_data
            for site in Qs_gageflow:
                if site not in all_data:
                    all_data[site] = Qs_gageflow[site].copy()
                else:
                    # If site already exists, append the new data
                    all_data[site] = pd.concat([all_data[site], Qs_gageflow[site]], axis=1)

                # reset columns to be realization integers 0, ... N
                all_data[site].columns = np.arange(0, all_data[site].shape[1])

        else:
            # extract just the data by realization
            Qs_gageflow = ensemble_set_data.data_by_realization

            # add to all_data
            for real in Qs_gageflow:
                all_data[realization_id] = Qs_gageflow[real]
                realization_id += 1

    return all_data


def load_ensemble_set_data(dataset_id, set_idx, ensemble_set_specs):
    """
    Load pywrdrb simulation data for a single ensemble set.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    set_idx : int
        Ensemble set index (0-based)
    ensemble_set_specs : list
        List of EnsembleSetSpec objects

    Returns
    -------
    data : pywrdrb.Data
        Data object with this ensemble set loaded
    """
    spec = ensemble_set_specs[set_idx]

    # Setup pathnavigator for this specific set
    pn_config = pywrdrb.get_pn_config()
    dataset_dir = spec.directory
    dataset_name = spec.directory.split('/')[-1]
    pn_config[f"flows/{dataset_name}"] = os.path.abspath(dataset_dir)
    pywrdrb.load_pn_config(pn_config)

    # Load simulation outputs for this ensemble set
    output_filenames = [spec.output_file]

    results_sets = [
        "major_flow",
        "inflow",
        "res_storage",
        "res_release",
        "mrf_target",
        "ibt_diversions",
        "ibt_demands",
        "nyc_release_components",
        "res_level"
    ]

    data = pywrdrb.Data(results_sets=results_sets, print_status=False)
    data.load_output(output_filenames=output_filenames)

    return data


def load_and_process_historical_models(dataset_id):
    """
    Load and process historical/reference models (reconstruction, wrfaorc, wrf1960s).

    Computes shortage and contribution for each historical model using the
    same central ``calculate_shortage_series`` function as ensemble postprocessing.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier

    Returns
    -------
    historical_data : dict
        Dictionary containing processed data for historical models, with keys:
        shortage, contribution, inflow, major_flow, res_storage,
        ibt_diversions, ibt_demands, mrf_target, res_level.
    """
    print("\nLoading and processing historical models...")

    # Load historical model outputs
    output_filenames = [
        RECONSTRUCTION_OUTPUT_FNAME,
        WRFAORC_OUTPUT_FNAME,
        WRF1960s_OUTPUT_FNAME
    ]

    results_sets = [
        "major_flow",
        "inflow",
        "res_storage",
        "res_release",
        "mrf_target",
        "ibt_diversions",
        "ibt_demands",
        "nyc_release_components",
        "res_level"
    ]

    data = pywrdrb.Data(results_sets=results_sets, print_status=False)
    data.load_output(output_filenames=output_filenames)

    # Load observations
    data.load_observations(results_sets=['res_storage', 'major_flow', 'reservoir_downstream_gage'])
    data.res_release['obs'] = {}
    data.res_release['obs'][0] = data.reservoir_downstream_gage['obs'][0]

    # Add Trenton equivalent flow
    data = add_trenton_equiv_flow(data)

    # Process historical models
    historical_models = ['reconstruction', 'wrfaorc_withObsScaled', 'wrf1960s_calib_nlcd2016']
    nodes = ['delMontague', 'delTrenton', 'nyc', 'nj']
    nyc_reservoirs = ['cannonsville', 'pepacton', 'neversink']

    historical_data = {
        'shortage': {},
        'contribution': {},
        'inflow': {},
        'major_flow': {},
        'res_storage': {},
        'ibt_diversions': {},
        'ibt_demands': {},
        'mrf_target': {},
        'res_level': {}
    }

    for model in historical_models:
        if model not in data.major_flow:
            print(f"  WARNING: {model} not found in loaded data")
            continue

        realizations = list(data.major_flow[model].keys())
        print(f"  Processing {model} ({len(realizations)} realizations)...")

        shortage_dict = {}
        contribution_dict = {}

        for r in realizations:
            # Calculate shortages for each node
            node_shortages = {}
            for node in nodes:
                flow_series, target_series = get_flow_and_target_values(
                    data, node, model, r, start_date=None, end_date=None
                )
                node_shortages[node] = calculate_shortage_series(target_series, flow_series)

            shortage_dict[r] = pd.DataFrame(node_shortages)

            # Contribution calculations
            contribution_columns = [f'mrf_montagueTrenton_{res}' for res in nyc_reservoirs]
            total_nyc_contribution = data.nyc_release_components[model][r].loc[:, contribution_columns].sum(axis=1)
            contribution_dict[r] = total_nyc_contribution.to_frame(name='mrf_montagueTrenton_nyc')

            # Add NYC aggregate inflow
            data.inflow[model][r].loc[:, 'nyc'] = data.inflow[model][r].loc[:, nyc_reservoirs].sum(axis=1)

        historical_data['shortage'][model] = shortage_dict
        historical_data['contribution'][model] = contribution_dict
        historical_data['inflow'][model] = data.inflow[model]
        historical_data['major_flow'][model] = data.major_flow[model]
        historical_data['res_storage'][model] = data.res_storage[model]
        historical_data['ibt_diversions'][model] = data.ibt_diversions[model]
        historical_data['ibt_demands'][model] = data.ibt_demands[model]
        historical_data['mrf_target'][model] = data.mrf_target[model]
        historical_data['res_level'][model] = data.res_level[model]

    print("  Historical model processing complete")
    return historical_data


def load_gage_flow_data(dataset_id, ensemble_set_specs=None):
    """
    Load and combine gage flow data from all ensemble sets.

    Loads one ensemble set at a time to reduce peak memory usage.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    ensemble_set_specs : list or None
        List of EnsembleSetSpec objects.  If None, built from config.

    Returns
    -------
    combined_gage_flow : dict
        Combined gage flow data with global realization IDs
    """
    import gc

    if ensemble_set_specs is None:
        ensemble_set_specs = [get_ensemble_set_spec(i, dataset_id) for i in range(N_ENSEMBLE_SETS)]

    print("\nLoading gage flow data...")

    # Setup pathnavigator
    pn_config = pywrdrb.get_pn_config()
    for spec in ensemble_set_specs:
        dataset_dir = spec.directory
        dataset_name = spec.directory.split('/')[-1]
        pn_config[f"flows/{dataset_name}"] = os.path.abspath(dataset_dir)
    pywrdrb.load_pn_config(pn_config)

    # Load one set at a time to reduce peak memory
    combined_gage_flow = {}
    for spec in ensemble_set_specs:
        set_name = spec.directory.split('/')[-1]
        set_idx = int(set_name.split('_set')[-1]) - 1

        data = pywrdrb.Data(results_sets=['major_flow'], print_status=False)
        data.load_hydrologic_model_flow([set_name])

        if set_name not in data.major_flow:
            del data
            continue

        set_data = data.major_flow[set_name]
        local_ids = list(set_data.keys())
        min_local_id = min(local_ids) if local_ids else 0

        for local_id, df in set_data.items():
            local_id_normalized = local_id - min_local_id
            global_id = set_idx * N_REALIZATIONS_PER_ENSEMBLE_SET + local_id_normalized
            combined_gage_flow[global_id] = df

        del data
        gc.collect()

    print(f"  Loaded gage flow for {len(combined_gage_flow)} realizations")
    return combined_gage_flow


def load_shortage_data(dataset_id):
    """
    Load pre-calculated shortage data from postprocessing.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier

    Returns
    -------
    pywrdrb.Data
        Data object with shortage, ibt_diversions, and ibt_demands
    """
    fname = f'{OUTPUT_DIR}/{dataset_id}_with_postprocessing.hdf5'

    if not os.path.exists(fname):
        raise FileNotFoundError(
            f"Postprocessed data not found: {fname}\n"
            "Run 04_postprocess_data.py first!"
        )

    print(f"Loading shortage data from: {fname}")
    data = pywrdrb.Data()
    data.load_from_export(fname, results_sets=['shortage', 'ibt_diversions', 'ibt_demands'])
    print("  Data loaded successfully")

    return data


def load_observed_diversions(loc='nyc'):
    """
    Load observed diversion data for training period.

    Parameters
    ----------
    loc : str
        Location identifier ('nyc' or 'nj')

    Returns
    -------
    pd.Series
        Observed diversions with datetime index
    """
    pn = get_pn_object()

    if loc == 'nyc':
        # Load NYC diversions from Excel file
        fname = pn.observations.get_str("_raw", "Pep_Can_Nev_diversions_daily_2000-2021.xlsx")
        diversion = pd.read_excel(fname, index_col=0)
        diversion = diversion.iloc[:, :3]
        diversion.index = pd.to_datetime(diversion.index)
        diversion['aggregate'] = diversion.sum(axis=1)
        diversion = diversion.loc[np.logical_not(np.isnan(diversion['aggregate']))]
        # Convert CFS to MGD
        diversion *= cfs_to_mgd
        return diversion['aggregate']

    elif loc == 'nj':
        # Load NJ diversions from USGS gage flow (Delaware-Raritan Canal)
        fname = pn.observations.get_str("_raw", "streamflow_daily_usgs_mgd.csv")
        gage_flow = pd.read_csv(fname)
        gage_flow.index = pd.DatetimeIndex(gage_flow['datetime']).date
        gage_flow.index = pd.to_datetime(gage_flow.index)

        # Convert gage ID to D_R_Canal
        gage_flow['D_R_Canal'] = gage_flow['01460440']
        diversion = gage_flow['D_R_Canal']

        # Keep only after 1991
        start_date = pd.Timestamp('1991-01-01')
        diversion = diversion.loc[diversion.index >= start_date]

        # Forward fill NA values
        diversion = diversion.fillna(method='ffill')

        # Set negative values to zero
        diversion = diversion.clip(lower=0.0)

        return diversion


def load_ensemble_diversions(dataset_id, loc='nyc'):
    """
    Load ensemble diversion data from HDF5 file.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    loc : str
        Location identifier ('nyc' or 'nj')

    Returns
    -------
    dict
        Dictionary of Series keyed by realization ID
    """
    # Get ensemble set specs
    ensemble_set_specs = ENSEMBLE_SETS[dataset_id]

    # Determine HDF5 filename and column name
    if loc == 'nyc':
        diversion_key = 'diversion_nyc'
        # NYC diversions use aggregate column (sum of pepacton, cannonsville, neversink)
        diversion_column = 'aggregate'
    else:
        diversion_key = 'diversion_nj'
        # NJ diversions use D_R_Canal column
        diversion_column = 'D_R_Canal'

    # Load all ensemble sets
    ensemble_diversions = {}

    for set_spec in ensemble_set_specs:
        if diversion_key not in set_spec.files:
            raise KeyError(f"Diversion file key '{diversion_key}' not found in ensemble set specification")

        fname = set_spec.files[diversion_key]

        if not os.path.exists(fname):
            raise FileNotFoundError(f"Ensemble diversion file not found: {fname}")

        # Load from HDF5
        with h5py.File(fname, 'r') as hf:
            for realization_id in hf.keys():
                real_group = hf[realization_id]

                # Extract datetime
                datetime_data = real_group['datetime'][:]
                if isinstance(datetime_data[0], bytes):
                    dates = [d.decode('utf-8') for d in datetime_data]
                else:
                    dates = datetime_data.tolist()
                dates = pd.to_datetime(dates)

                # Extract diversion column based on location
                div_data = real_group[diversion_column][:]

                ensemble_diversions[int(realization_id)] = pd.Series(div_data, index=dates)

    return ensemble_diversions


def load_zone_probabilities(dataset_id, period='weekly'):
    """
    Load zone probabilities from CSV.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    period : str
        Time period ('daily' or 'weekly')

    Returns
    -------
    pd.DataFrame or None
        Zone probabilities indexed by period, or None if file not found
    """
    csv_file = f"{ZONE_PROB_DIR}/{dataset_id}_zone_probs_{period}.csv"

    if not os.path.exists(csv_file):
        print(f"ERROR: Zone probabilities not found: {csv_file}")
        print("Run 07_calculate_storage_zone_probabilities.py first!")
        return None

    df = pd.read_csv(csv_file, index_col='period')
    return df


def load_storage_percentiles(dataset_id, period='weekly'):
    """
    Load storage percentiles from CSV.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    period : str
        Time period ('daily' or 'weekly')

    Returns
    -------
    pd.DataFrame or None
        Storage percentiles indexed by period with columns p1, p5, p10, etc.
        Returns None if file not found.
    """
    csv_file = f"{ZONE_PROB_DIR}/{dataset_id}_storage_percentiles_{period}.csv"

    if not os.path.exists(csv_file):
        print(f"ERROR: Storage percentiles not found: {csv_file}")
        print("Run 07_calculate_storage_zone_probabilities.py first!")
        return None

    df = pd.read_csv(csv_file, index_col='period')
    return df


def load_reservoir_storage(dataset_id):
    """
    Load reservoir storage data from postprocessed HDF5 file.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier

    Returns
    -------
    dict
        Dictionary mapping realization_id to storage DataFrame
    """
    fname = f"{OUTPUT_DIR}/{dataset_id}_with_postprocessing.hdf5"

    if not os.path.exists(fname):
        raise FileNotFoundError(
            f"Postprocessed data file not found: {fname}\n"
            f"Run 04_postprocess_data.py first!"
        )

    # Load only reservoir storage data
    data = pywrdrb.Data()
    data.load_from_export(fname, results_sets=['res_storage'])

    if dataset_id not in data.res_storage:
        raise KeyError(f"Dataset {dataset_id} not found in res_storage")

    return data.res_storage[dataset_id]


def load_drought_satisficing(dataset_id, ssi_window,
                             storage_threshold=20.0, violation_days=3):
    """
    Load drought events merged with annual performance metrics and annotated
    with satisficing categories.

    Loads the drought events CSV and the annual metrics CSV, filters the
    metrics to the whole-year period (``period == 'all'``), and merges on
    ``(water_year, realization_id)`` so each drought event carries the
    annual metrics for the year it started.  Satisficing category columns
    are then added via :func:`add_satisficing_category`.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier (e.g. ``'stationary_ensemble'``).
    ssi_window : int
        SSI window (3, 6, or 12 months).
    storage_threshold : float
        Minimum acceptable NYC storage percentage (default: 20%).
    violation_days : int
        Maximum acceptable consecutive Montague violation days (default: 3).

    Returns
    -------
    pd.DataFrame
        Drought events with additional columns:
        ``nyc_min_storage_pct``, ``montague_max_consec_shortage_days``,
        ``storage_pass``, ``montague_pass``, ``satisficing_category``.

    Raises
    ------
    FileNotFoundError
        If the drought events or annual metrics CSV is missing.
        Run ``06_calculate_performance_metrics.py`` to generate the metrics.
    """
    events_df = load_drought_events(dataset_id, ssi_window)
    annual_df = load_annual_metrics(dataset_id)

    annual_df = (
        annual_df[annual_df['period'] == 'all']
        .rename(columns={'water_year': 'year'})
    )

    events_df['start'] = pd.to_datetime(events_df['start'])
    events_df['year'] = events_df['start'].dt.year

    merged = events_df.merge(
        annual_df[['year', 'realization_id',
                   'nyc_min_storage_pct', 'montague_max_consec_shortage_days']],
        on=['year', 'realization_id'],
        how='left',
    )
    merged = merged.dropna(subset=['nyc_min_storage_pct'])
    merged = add_satisficing_category(merged, storage_threshold, violation_days)
    print(f"    Merged {len(merged)} drought events with annual satisficing")
    return merged


def compute_event_exceedances(df, metric='severity', n_years=N_YEARS):
    """
    Compute exceedance rates for each drought event across the entire ensemble.

    For each event, computes how many events across ALL realizations have
    metric >= this event's value, normalized by total ensemble-years.

    Parameters
    ----------
    df : pd.DataFrame
        Drought events with 'realization_id' and metric columns
    metric : str
        Metric to use for exceedance calculation (e.g., 'severity', 'magnitude')
    n_years : int
        Number of years per realization for normalization (default: 70)

    Returns
    -------
    exceedances : np.ndarray
        Exceedance rate for each event across entire ensemble (events per year)

    Examples
    --------
    >>> from methods.load import load_drought_events, compute_event_exceedances
    >>> df = load_drought_events('stationary_ensemble', ssi_window=3)
    >>> exceedances = compute_event_exceedances(df, metric='severity')
    >>> # Select events at 0.1 yr^-1 exceedance rate
    >>> target = 0.1
    >>> best_idx = np.argmin(np.abs(exceedances - target))
    >>> selected_event = df.loc[best_idx]
    """
    # Total years across all realizations
    n_realizations = df['realization_id'].nunique()
    total_years = n_years * n_realizations

    # Get all metric values across entire ensemble
    all_values = df[metric].values

    exceedances = np.zeros(len(df))

    # Use enumerate to get positional index (not label index)
    # This handles cases where df.index has gaps due to filtering
    for i, (idx, row) in enumerate(df.iterrows()):
        val = row[metric]

        # Count how many events across ALL realizations have metric >= this value
        n_exceedances = np.sum(all_values >= val)

        # Normalize by total ensemble-years
        exceedances[i] = n_exceedances / total_years

    return exceedances


# =============================================================================
# CONTRIBUTION METRICS
# =============================================================================


def load_contribution_metrics(dataset_id):
    """
    Load pre-computed contribution metrics CSV.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier (e.g., 'stationary_ensemble', 'climate_adjusted_low')

    Returns
    -------
    pd.DataFrame
        Contribution metrics with columns:
        - realization_id, year, annual_max_zone, annual_max_zone_date, annual_min_storage_pct
        - contribution_total_{W}d, contribution_ratio_{W}d, inflow_total_{W}d,
          demand_satisfaction_{W}d, worst_1mo_demand_sat_{W}d
          for W in [30, 60, 90, 120, 150, 180, 270]

    Raises
    ------
    FileNotFoundError
        If pre-computed metrics file does not exist.
    """
    fname = f'{PERFORMANCE_METRICS_DIR}/{dataset_id}_contribution_metrics.csv'

    if not os.path.exists(fname):
        raise FileNotFoundError(
            f"Pre-computed metrics not found: {fname}\n"
            "Run 06_calculate_performance_metrics.py to generate these files."
        )

    df = pd.read_csv(fname)

    if 'annual_max_zone_date' in df.columns:
        df['annual_max_zone_date'] = pd.to_datetime(df['annual_max_zone_date'])

    return df


def load_event_metrics(dataset_id, ssi_window, min_duration=30):
    """Load per-drought-event metrics CSV for a given dataset and SSI window.

    These CSVs are produced by ``08_calculate_event_metrics.py`` and contain
    one row per drought event with hazard characteristics, system actions,
    and outcome metrics (storage, shortage, FFMP zone at minimum, etc.).

    Parameters
    ----------
    dataset_id : str
        Dataset identifier (e.g. ``'stationary_ensemble'``).
    ssi_window : int
        SSI window (3, 6, or 12 months).
    min_duration : int
        Minimum event duration in days to retain (default: 30).

    Returns
    -------
    pd.DataFrame
        Filtered event-level metrics.
    """
    fname = f"{EVENT_METRICS_DIR}/{dataset_id}_ssi{ssi_window}_event_metrics.csv"

    if not os.path.exists(fname):
        raise FileNotFoundError(
            f"Event metrics not found: {fname}\n"
            "Run 08_calculate_event_metrics.py first!"
        )

    df = pd.read_csv(fname)
    df = df[df['duration_days'] >= min_duration].copy()
    df['severity'] = df['severity'].abs()
    df['magnitude'] = df['magnitude'].abs()
    return df