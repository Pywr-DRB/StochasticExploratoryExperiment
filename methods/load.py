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
from methods.config import RECONSTRUCTION_OUTPUT_FNAME, ENSEMBLE_SETS, ROOT_DIR

file_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = f"{file_dir}/../data"

DROUGHT_METRICS_DIR = f"{ROOT_DIR}/pywrdrb/drought_metrics"


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

def load_performance_metrics(dataset_id):
    """
    Load pre-calculated performance metrics from CSV.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier

    Returns
    -------
    metrics_df : pd.DataFrame
        DataFrame with performance metrics for all realizations
    """
    performance_metrics_dir = f"{ROOT_DIR}/pywrdrb/performance_metrics"
    csv_file = f"{performance_metrics_dir}/{dataset_id}_performance_metrics.csv"

    if not os.path.exists(csv_file):
        raise FileNotFoundError(
            f"Performance metrics not found: {csv_file}\n"
            f"Run 04_postprocess_data.py first to calculate metrics!"
        )

    metrics_df = pd.read_csv(csv_file, index_col='realization_id')
    return metrics_df


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
    fname = f'./pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5'

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
    zone_prob_dir = f"{ROOT_DIR}/pywrdrb/zone_probabilities"
    csv_file = f"{zone_prob_dir}/{dataset_id}_zone_probs_{period}.csv"

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
    zone_prob_dir = f"{ROOT_DIR}/pywrdrb/zone_probabilities"
    csv_file = f"{zone_prob_dir}/{dataset_id}_storage_percentiles_{period}.csv"

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
    fname = f"{ROOT_DIR}/pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5"

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


def load_satisficing_results(dataset_id, ssi_window):
    """
    Load satisficing results from 06_calculate_satisficing_by_drought.py.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    ssi_window : int
        SSI window (3, 6, or 12 months)

    Returns
    -------
    dict
        Dictionary with keys 'all_years', 'drought', 'non_drought' containing
        corresponding DataFrames
    """
    data_dir = f"{ROOT_DIR}/pywrdrb/satisficing_analysis"

    files = {
        'all_years': f"{data_dir}/{dataset_id}_ssi{ssi_window}_all_years.csv",
        'drought': f"{data_dir}/{dataset_id}_ssi{ssi_window}_during_droughts.csv",
        'non_drought': f"{data_dir}/{dataset_id}_ssi{ssi_window}_non_drought.csv"
    }

    # Check for alternative naming convention
    alt_files = {
        'drought': f"{data_dir}/{dataset_id}_ssi{ssi_window}_years_with_droughts.csv",
        'non_drought': f"{data_dir}/{dataset_id}_ssi{ssi_window}_years_without_droughts.csv"
    }

    results = {}

    for condition, fname in files.items():
        # Try primary filename first, then alternative
        if not os.path.exists(fname) and condition in alt_files:
            fname = alt_files[condition]

        if not os.path.exists(fname):
            raise FileNotFoundError(
                f"Results file not found: {fname}\n"
                "Run 06_calculate_satisficing_by_drought.py first!"
            )

        print(f"Loading {condition}: {fname}")
        results[condition] = pd.read_csv(fname)

    return results


def compute_event_exceedances(df, metric='severity', n_years=70):
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