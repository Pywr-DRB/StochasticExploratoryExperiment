import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import pywrdrb
from pywrdrb.path_manager import get_pn_object
from sglib.utils.load import HDF5Manager
from methods.config import RECONSTRUCTION_OUTPUT_FNAME

file_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = f"{file_dir}/../data"


def load_baseline_historical_flow(gage_flow=True, 
                                  period='baseline',
                                  flowtype='wrfaorc_withObsScaled'):
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

        hdf_manager = HDF5Manager()
        ensemble_set_data = hdf_manager.load_ensemble(gageflow_set_file)

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