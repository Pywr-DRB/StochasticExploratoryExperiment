import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sglib.plotting.plot import plot_fdc_ranges, plot_autocorrelation


def plot_fdc_gridded(Qh, Qs, 
                     timestep='daily',
                     fname=None):
    
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

    for i, site in enumerate(sites):
        
        # use plot_fdc_ranges to plot the FDC
        ax = axs[i // ncols, i % ncols]
        
        
        # For Qs[site], trim so that it is only full years
        # Since FDcs are annual, we dont want partial years included
        syn_index = Qs[site].index
        start_year = syn_index[0].year
        end_year = syn_index[-1].year
        if pd.to_datetime(f'{end_year}-12-01') not in syn_index:
            end_year -= 1
        if pd.to_datetime(f'{start_year}-01-01') not in syn_index:
            start_year += 1
        
        Qs[site] = Qs[site].loc[f'{start_year}-01-01':f'{end_year}-12-31']
        

        plot_fdc_ranges(Qh.loc[:, site],
                        Qs[site],
                        legend=False,
                        ax=ax,
                        title=site,
                        units='MGD',
                        xylabels=False)

        if i % ncols == 0:
            ax.set_ylabel('Flow (MGD)')
        if i // ncols == nrows - 1:
            ax.set_xlabel('Nonexceedance')    

    plt.tight_layout()
    
    if fname is not None:
        plt.savefig(fname, dpi=200)
    
    return

def plot_autocorrelation_gridded(Qh, Qs, 
                                 timestep='daily',
                                 lag_range=np.arange(1,60, 5),
                                 fname=None):
    
    # Settings
    sites = list(Qh.columns)
    ncols = 5
    nrows = int(np.ceil(len(sites) / ncols))

    if timestep == 'monthly':
        x_label = 'Lag (months)'
    elif timestep == 'daily':
        x_label = 'Lag (days)'
    else:
        raise ValueError(f"Unsupported timestep: {timestep}. Supported timesteps are 'daily' and 'monthly'.")

    # Create the plot
    fig, axs = plt.subplots(figsize=(ncols*3, nrows*3), 
                        nrows=nrows, ncols=ncols, 
                        sharex=True, sharey=True)

    for i, site in enumerate(sites):
        
        # use plot_fdc_ranges to plot the FDC
        ax = axs[i // ncols, i % ncols]
        
        fig, ax = plot_autocorrelation(Qh.loc[:, site], 
                        Qs[site], 
                        lag_range=lag_range, 
                        timestep=timestep,
                        savefig=False,
                        fname=None,
                        ax=ax,
                        xy_labels=False)
        
        # Set site as the title
        ax.set_title(site)
        
        if i % ncols == 0:
            ax.set_ylabel('Autocorrelation')
        if i // ncols == nrows - 1:
            ax.set_xlabel(x_label)    

    plt.tight_layout()
    
    if fname is not None:
        plt.savefig(fname, dpi=200)
    
    return