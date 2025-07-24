import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def drought_metric_scatter_plot(obs_drought_metrics=None, 
                                syn_drought_metrics=None, 
                                x_char = 'magnitude',
                                y_char = 'duration',
                                color_char = 'severity',
                                size_char=None,
                                fname=None):
    
    fig, ax = plt.subplots(figsize = (7,6))
    
    if obs_drought_metrics is None and syn_drought_metrics is None:
        raise ValueError("At least one of obs_drought_metrics or syn_drought_metrics must be provided.")
    
    max_color_val = 0
    if obs_drought_metrics is not None:
        max_color_val = obs_drought_metrics[color_char].max()
    
    if syn_drought_metrics is not None:
        max_color_val = max(max_color_val, syn_drought_metrics[color_char].max())
    
    if obs_drought_metrics is not None:
        
        s = 100 if size_char is None else obs_drought_metrics[size_char]
        
        # If using size_char, scale so max size is 100
        if (size_char is not None) and (len(s) > 1):
            s = np.array(s)
            s_max = s.max()
            s_min = s.min()
            s = 100 * (s - s.min()) / (s.max() - s.min())
        
        p = ax.scatter(obs_drought_metrics[x_char], 
                    -obs_drought_metrics[y_char],
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
                   -syn_drought_metrics[y_char],
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
        plt.legend(handles, new_labels, loc="upper right", title="Sizes")
    
    plt.colorbar(p).set_label(label = 'Drought Duration (days)',size=15)
    plt.title(f'Drought Characteristics', fontsize = 16)
    if x_char == 'severity':
        plt.xlim(-3.5, -1.0)
    plt.tight_layout()
    
    if fname is not None:
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {fname}")
    
    plt.show()
    return


import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def drought_metric_joint_plot(obs_drought_metrics, 
                              syn_drought_metrics=None, 
                              x_char='magnitude',
                              y_char='duration',
                              color_char='severity',
                              title='Drought Characteristics',
                              xlabel=None,
                              ylabel=None,
                              colorbar_label=None,
                              figsize=(10, 8),
                              kind='scatter',
                              marginal_kws=None,
                              joint_kws=None,
                              fname=None):
    """
    Create a joint plot showing drought metrics with marginal distributions.
    
    Parameters:
    -----------
    obs_drought_metrics : pd.DataFrame
        Observed drought metrics data
    syn_drought_metrics : pd.DataFrame, optional
        Synthetic drought metrics data
    x_char, y_char, color_char : str
        Column names for x, y, and color mapping
    title : str
        Plot title
    xlabel, ylabel : str, optional
        Axis labels (auto-generated if None)
    colorbar_label : str, optional
        Colorbar label (auto-generated if None)
    figsize : tuple
        Figure size (width, height)
    kind : str
        Joint plot kind ('scatter', 'hex', 'kde', 'reg')
    marginal_kws, joint_kws : dict, optional
        Additional keyword arguments for marginal and joint plots
    fname : str, optional
        Save filename
    """
    
    # Set default parameters
    if marginal_kws is None:
        marginal_kws = {'bins': 30, 'alpha': 0.7}
    if joint_kws is None:
        joint_kws = {'s': 100, 'alpha': 0.8, 'edgecolor': 'black', 'linewidth': 0.5}
    
    # Auto-generate labels if not provided
    if xlabel is None:
        xlabel = x_char.replace('_', ' ').title()
    if ylabel is None:
        ylabel = y_char.replace('_', ' ').title()
    if colorbar_label is None:
        colorbar_label = color_char.replace('_', ' ').title()
    
    # Transform y-axis data (negative for duration as in original)
    obs_data = obs_drought_metrics.copy()
    obs_data[y_char] = -obs_data[y_char]
    
    # Calculate color scale limits
    max_color_val = obs_data[color_char].max()
    min_color_val = obs_data[color_char].min()
    
    if syn_drought_metrics is not None:
        syn_data = syn_drought_metrics.copy()
        syn_data[y_char] = -syn_data[y_char]
        max_color_val = max(max_color_val, syn_data[color_char].max())
        min_color_val = min(min_color_val, syn_data[color_char].min())
    
    # Create joint plot with observed data
    g = sns.jointplot(data=obs_data, 
                      x=x_char, 
                      y=y_char,
                      kind=kind,
                      height=figsize[1],
                      ratio=4,  # Ratio of joint plot to marginal plots
                      marginal_kws=marginal_kws,
                      joint_kws={**joint_kws, 'c': obs_data[color_char], 
                                'cmap': 'viridis_r', 'vmin': min_color_val, 'vmax': max_color_val})
    
    # Add synthetic data if provided
    if syn_drought_metrics is not None:
        g.ax_joint.scatter(syn_data[x_char], 
                          syn_data[y_char],
                          c=syn_data[color_char], 
                          cmap='viridis_r', 
                          s=joint_kws['s'],
                          vmin=min_color_val, 
                          vmax=max_color_val,
                          alpha=0.4,
                          edgecolor='none',
                          label='Synthetic',
                          zorder=1)
        
        # Add marginal distributions for synthetic data
        g.ax_marg_x.hist(syn_data[x_char], alpha=0.5, bins=marginal_kws['bins'], 
                        color='orange', label='Synthetic')
        g.ax_marg_y.hist(syn_data[y_char], alpha=0.5, bins=marginal_kws['bins'], 
                        orientation='horizontal', color='orange')
    
    # Customize the plot
    g.ax_joint.set_xlabel(xlabel, fontsize=14)
    g.ax_joint.set_ylabel(ylabel, fontsize=14)
    g.fig.suptitle(title, fontsize=16, y=0.98)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis_r', 
                              norm=plt.Normalize(vmin=min_color_val, vmax=max_color_val))
    sm.set_array([])
    cbar = g.fig.colorbar(sm, ax=g.ax_joint, shrink=0.8, aspect=20)
    cbar.set_label(colorbar_label, fontsize=12)
    
    # Add legend if synthetic data is present
    if syn_drought_metrics is not None:
        # Create custom legend elements
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', alpha=0.7, label='Observed'),
            Patch(facecolor='orange', alpha=0.5, label='Synthetic')
        ]
        g.ax_joint.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    # Apply x-axis limits if severity is the x-axis (as in original)
    if x_char == 'severity':
        g.ax_joint.set_xlim(-3.5, -1.0)
    
    # Improve aesthetics
    g.ax_joint.grid(True, alpha=0.3)
    g.ax_joint.tick_params(labelsize=12)
    
    # Style marginal plots
    g.ax_marg_x.tick_params(labelsize=10)
    g.ax_marg_y.tick_params(labelsize=10)
    
    plt.tight_layout()
    
    if fname is not None:
        g.savefig(fname, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {fname}")
    
    plt.show()
    return g