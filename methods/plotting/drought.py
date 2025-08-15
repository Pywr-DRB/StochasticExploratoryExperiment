import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns



def drought_metric_scatter_plot(obs_drought_metrics=None, 
                                syn_drought_metrics=None, 
                                x_char = 'magnitude',
                                y_char = 'duration',
                                color_char = 'severity',
                                size_char=None,
                                y_scale = 'linear',
                                x_scale = 'linear',
                                x_lims = None,
                                y_lims = None,
                                x_label = None,
                                y_label = None,
                                color_label = None,
                                size_label = None,
                                title=None,
                                add_marginals=False,
                                fname=None):
    """
    Create scatter plot of drought characteristics. 
    
    If add_marginals is True, then it is a (2,2) grid with scatter in the lower left, 
    and KDE distributions added along the top and right axes. Legend in the upper left ax quadrant. 
    
    If add_marginals is False, then it is a single scatter plot with legend in the upper right corner.
    
    Parameters
    ----------
    obs_drought_metrics : pd.DataFrame, optional
        DataFrame containing observed drought metrics.
    syn_drought_metrics : pd.DataFrame, optional
        DataFrame containing synthetic drought metrics.
    
    """
    
    
    # Make sure x and y metrics exist in the dataframes
    if obs_drought_metrics is not None:
        assert x_char in obs_drought_metrics.columns, f"{x_char} not found in obs_drought_metrics"
        assert y_char in obs_drought_metrics.columns, f"{y_char} not found in obs_drought_metrics"
        assert color_char in obs_drought_metrics.columns, f"{color_char} not found in obs_drought_metrics"
    if syn_drought_metrics is not None:
        assert x_char in syn_drought_metrics.columns, f"{x_char} not found in syn_drought_metrics"
        assert y_char in syn_drought_metrics.columns, f"{y_char} not found in syn_drought_metrics"
        assert color_char in syn_drought_metrics.columns, f"{color_char} not found in syn_drought_metrics"


    ### Create figure and axes
    if add_marginals:
        fig, axs = plt.subplots(nrows=2, ncols=2, 
                               figsize=(10,10), 
                               gridspec_kw={'width_ratios': [1, 0.2], 
                                            'height_ratios': [0.2, 1]})
        
        ax = axs[1, 0]  # Main scatter plot in the lower left
        # turn off all other axes
        axs[0, 0].axis('off')
        axs[0, 1].axis('off')
        axs[1, 1].axis('off')
        
    else:
        fig, ax = plt.subplots(figsize = (6,7))


    if obs_drought_metrics is None and syn_drought_metrics is None:
        raise ValueError("At least one of obs_drought_metrics or syn_drought_metrics must be provided.")
    
    max_color_val = 0
    if obs_drought_metrics is not None:
        max_color_val = obs_drought_metrics[color_char].max()
    
    if syn_drought_metrics is not None:
        max_color_val = max(max_color_val, syn_drought_metrics[color_char].max())
    
    ## Create main scatter plot
    if obs_drought_metrics is not None:
        
        s = 100 if size_char is None else obs_drought_metrics[size_char]
        
        # If using size_char, scale so max size is 100
        if (size_char is not None) and (len(s) > 1):
            s = np.array(s)
            s_max = s.max()
            s_min = s.min()
            s = 100 * (s - s.min()) / (s.max() - s.min())
        
        p = ax.scatter(obs_drought_metrics[x_char], 
                       obs_drought_metrics[y_char],
                    c= obs_drought_metrics[color_char], 
                    cmap = 'viridis_r', s=s, 
                    vmin = 0, vmax = max_color_val,
                    edgecolor='k', lw=1.5, label='Observed', 
                    zorder=5, alpha=1)
        
    if syn_drought_metrics is not None:
        s = 100 if size_char is None else syn_drought_metrics[size_char]
        # If using size_char, scale so max size is 100 and min is 5
        if (size_char is not None) and (len(s) > 1):
            s = np.array(s)
            s_max = s.max()
            s_min = s.min()
            print(f"size_char: {size_char} | Max val: {s.max()}, Min val: {s.min()}")
            s = 100 * (s - s.min()) / (s.max() - s.min())

        p = ax.scatter(syn_drought_metrics[x_char], 
                   syn_drought_metrics[y_char],
                   c= syn_drought_metrics[color_char], 
                   cmap = 'viridis_r', s=s,
                   vmin = 0, vmax = max_color_val, 
                   edgecolor='none', 
                   label='Synthetic',
                   zorder=1, alpha=0.5)
    
    
    if size_char is not None:
        handles, labels = p.legend_elements(prop="sizes", num=5)
        
        # Need to re-apply s_max and s_min to display the actual values not size
        new_labels = []
        for l in labels:
            # find the numeric part
            num = re.search(r'\d+', l).group()
            
            # Rescale num
            num = float(num) * (s_max - s_min) / 100 + s_min
            new_labels.append(f"{int(num)}")            
        
        size_title = size_label if size_label is not None else size_char.capitalize()
        plt.legend(handles, new_labels, loc="upper right", title=size_title)


    # Modify axes
    xlab = x_label if x_label is not None else x_char.capitalize()
    ylab = y_label if y_label is not None else y_char.capitalize()
    ax.set_xlabel(xlab, fontsize=14)
    ax.set_ylabel(ylab, fontsize=14)
    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)

    if x_lims is not None:
        ax.set_xlim(x_lims)
    if y_lims is not None:
        ax.set_ylim(y_lims)
    
    # If add_marginals is True, add KDE distributions
    if add_marginals:
        bw_adjust = 0.8
        
        # Add KDE along the top axis
        ax_top = fig.add_subplot(2, 2, 1, sharex=ax)
        sns.kdeplot(syn_drought_metrics[x_char], 
                    ax=ax_top, 
                    bw_adjust=bw_adjust,
                    cut=0.0,
                    color='orange', fill=True, alpha=0.5, 
                    label='Synthetic')
        
        # Add lines for each observed drought metric
        if obs_drought_metrics is not None:
            obs_xs = obs_drought_metrics[x_char].dropna()
            ylim = ax_top.get_ylim()
            
            ax_top.vlines(obs_xs, 
                          ylim[0], ylim[1]*0.5, 
                          color='k')
            
        ax_top.set_ylabel('Density', fontsize=14)
        ax_top.set_xlabel('')
        ax_top.set_xscale(x_scale)
        ax_top.set_xlim(ax.get_xlim())
        ax_top.set_yscale('linear')
        
        # Add KDE along the right axis
        ax_right = fig.add_subplot(2, 2, 4, sharey=ax)
        
        sns.kdeplot(y=syn_drought_metrics[y_char], 
                    ax=ax_right, 
                    bw_adjust=bw_adjust,
                    cut=0.0,
                    color='orange', fill=True, alpha=0.5,
                    label='Synthetic')
                
        # add lines for each observed drought metric
        if obs_drought_metrics is not None:
            obs_ys = obs_drought_metrics[y_char].dropna()
            xlim = ax_right.get_xlim()
            ax_right.hlines(obs_ys, 
                            xlim[0], xlim[1]*0.5, 
                            color='k')    

        ax_right.set_xlabel('Density', fontsize=14)
        ax_right.set_ylabel('')
        ax_right.set_yscale(y_scale)
        ax_right.set_ylim(ax.get_ylim())
        ax_right.set_xscale('linear')
        
        # Remove ALL ticks, tick labels and spines from the marginal axes
        for ax_marg in [ax_top, ax_right]:
            ax_marg.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            ax_marg.axis('off')
            ax_marg.spines['top'].set_visible(False)
            ax_marg.spines['right'].set_visible(False)
            ax_marg.spines['left'].set_visible(False)
            ax_marg.spines['bottom'].set_visible(False)
        
    cbar_label = color_label if color_label is not None else color_char.capitalize()
    
    # Put the colorbar horizontally below the bottom axis 
    plt.colorbar(p, ax=ax, orientation='horizontal', 
                 pad=-0.15, 
                 label=cbar_label, aspect=50, 
                 shrink=0.8)

    if title is not None:
        plt.title(title, fontsize=16)
        
    plt.tight_layout()
    
    if fname is not None:
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {fname}")
    return

