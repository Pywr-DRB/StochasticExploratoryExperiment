import os
import h5py
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import pywrdrb
from pywrdrb.path_manager import get_pn_object
from pywrdrb.utils.constants import cfs_to_mgd
from sglib.core.ensemble import Ensemble
from methods.config import RECONSTRUCTION_OUTPUT_FNAME, ENSEMBLE_SETS, ROOT_DIR

file_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = f"{file_dir}/../data"


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



def load_drought_events(dataset_id, ssi_window):
    """
    Load drought events for a given dataset and SSI window.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    ssi_window : int
        SSI window (3, 6, or 12 months)

    Returns
    -------
    pd.DataFrame
        Drought events with date columns converted to datetime
    """
    # Get the root directory (parent of methods/)
    root_dir = os.path.dirname(file_dir)
    fname = f"{root_dir}/pywrdrb/drought_metrics/{dataset_id}_ssi{ssi_window}_drought_events.csv"

    if not os.path.exists(fname):
        raise FileNotFoundError(f"Drought events file not found: {fname}")

    print(f"Loading drought events from: {fname}")
    df = pd.read_csv(fname)

    # Convert date columns
    date_cols = ['start', 'end', 'max_severity_date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    print(f"  Loaded {len(df)} drought events")
    print(f"  Unique realizations: {df['realization_id'].nunique()}")

    return df


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