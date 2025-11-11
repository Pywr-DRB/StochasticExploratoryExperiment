import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sglib import Ensemble
from sglib.plotting import plot_flow_duration_curve


def plot_fdc_gridded(Qh, Qs,
                     timestep='daily',
                     fname=None):
    """
    Create gridded flow duration curve plots for multiple sites.

    Parameters
    ----------
    Qh : pd.DataFrame
        Historical/observed flows with sites as columns
    Qs : dict
        Dictionary of synthetic flow DataFrames, keyed by site name
    timestep : str
        'daily' or 'monthly'
    fname : str, optional
        Filename to save figure
    """

    # Settings
    sites = list(Qh.columns)
    ncols = 5
    nrows = int(np.ceil(len(sites) / ncols))
    if timestep == 'monthly':
        units = 'MG'
    elif timestep == 'daily':
        units = 'MGD'
    else:
        raise ValueError(f"Unsupported timestep: {timestep}. Supported timesteps are 'daily' and 'monthly'.")

    # Create the plot
    fig, axs = plt.subplots(figsize=(ncols*3, nrows*3),
                        nrows=nrows, ncols=ncols,
                        sharex=True, sharey=True)

    # Pre-compute realization ID mapping once (shared across all sites)
    first_site = sites[0]
    sample_cols = Qs[first_site].columns
    real_id_map = {}
    for col in sample_cols:
        if isinstance(col, str) and col.isdigit():
            real_id_map[col] = int(col)
        elif isinstance(col, (int, np.integer)):
            real_id_map[col] = int(col)
        else:
            real_id_map[col] = col

    for i, site in enumerate(sites):

        ax = axs[i // ncols, i % ncols]

        # For Qs[site], trim so that it is only full years
        # Since FDCs are annual, we dont want partial years included
        syn_index = Qs[site].index
        start_year = syn_index[0].year
        end_year = syn_index[-1].year
        if pd.to_datetime(f'{end_year}-12-01') not in syn_index:
            end_year -= 1
        if pd.to_datetime(f'{start_year}-01-01') not in syn_index:
            start_year += 1

        Qs_trimmed = Qs[site].loc[f'{start_year}-01-01':f'{end_year}-12-31']

        # Convert synthetic data to Ensemble object (using pre-computed mapping)
        ensemble_dict = {}
        for col in Qs_trimmed.columns:
            real_id = real_id_map.get(col, col)
            ensemble_dict[real_id] = pd.DataFrame({site: Qs_trimmed[col]})

        ensemble = Ensemble(ensemble_dict)

        # Plot using new SGLib API
        plot_flow_duration_curve(
            ensemble,
            observed=Qh.loc[:, site],
            site=site,
            show_annual_range=True,
            ax=ax,
            title=site,
            xlabel=None,
            ylabel=None,
            legend=False,
            units=units.lower(),
            log_scale=True
        )

        if i % ncols == 0:
            ax.set_ylabel(f'Flow ({units})')
        if i // ncols == nrows - 1:
            ax.set_xlabel('Exceedance Probability')

    plt.tight_layout()

    if fname is not None:
        plt.savefig(fname, dpi=200)
        plt.close(fig)  # Free memory after saving

    return

def plot_autocorrelation_gridded(Qh, Qs,
                                 timestep='daily',
                                 lag_range=np.arange(1,60, 5),
                                 fname=None):
    """
    Create gridded autocorrelation plots for multiple sites.

    Parameters
    ----------
    Qh : pd.DataFrame
        Historical/observed flows with sites as columns
    Qs : dict
        Dictionary of synthetic flow DataFrames, keyed by site name
    timestep : str
        'daily' or 'monthly'
    lag_range : np.ndarray
        Array of lag values to compute (not used with new API, which uses max_lag)
    fname : str, optional
        Filename to save figure
    """
    from sglib.plotting import plot_autocorrelation

    # Settings
    sites = list(Qh.columns)
    ncols = 5
    nrows = int(np.ceil(len(sites) / ncols))

    if timestep == 'monthly':
        x_label = 'Lag (months)'
        max_lag = int(lag_range.max()) if len(lag_range) > 0 else 30
    elif timestep == 'daily':
        x_label = 'Lag (days)'
        max_lag = int(lag_range.max()) if len(lag_range) > 0 else 60
    else:
        raise ValueError(f"Unsupported timestep: {timestep}. Supported timesteps are 'daily' and 'monthly'.")

    # Create the plot
    fig, axs = plt.subplots(figsize=(ncols*3, nrows*3),
                        nrows=nrows, ncols=ncols,
                        sharex=True, sharey=True)

    # Pre-compute realization ID mapping once (shared across all sites)
    first_site = sites[0]
    sample_cols = Qs[first_site].columns
    real_id_map = {}
    for col in sample_cols:
        if isinstance(col, str) and col.isdigit():
            real_id_map[col] = int(col)
        elif isinstance(col, (int, np.integer)):
            real_id_map[col] = int(col)
        else:
            real_id_map[col] = col

    for i, site in enumerate(sites):

        ax = axs[i // ncols, i % ncols]

        # Convert synthetic data to Ensemble object (using pre-computed mapping)
        ensemble_dict = {}
        for col in Qs[site].columns:
            real_id = real_id_map.get(col, col)
            ensemble_dict[real_id] = pd.DataFrame({site: Qs[site][col]})

        ensemble = Ensemble(ensemble_dict)

        # Plot using new SGLib API
        plot_autocorrelation(
            ensemble,
            observed=Qh.loc[:, site],
            site=site,
            max_lag=max_lag,
            timestep=timestep,
            show_members=None,
            ax=ax,
            title=site,
            xlabel=None,
            ylabel=None,
            legend=False
        )

        if i % ncols == 0:
            ax.set_ylabel('Autocorrelation')
        if i // ncols == nrows - 1:
            ax.set_xlabel(x_label)

    plt.tight_layout()

    if fname is not None:
        plt.savefig(fname, dpi=200)
        plt.close(fig)  # Free memory after saving

    return