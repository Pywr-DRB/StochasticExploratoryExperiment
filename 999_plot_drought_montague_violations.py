import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from methods.plotting.drought import drought_metric_scatter_plot
from methods.config import verify_ensemble_type
from methods.config import FIG_DIR



if __name__ == "__main__":
    SSI_WINDOW = 12
    
    # Get ensemble type from command line arguments
    ensemble_type = sys.argv[1]
    inflow_type = f'{ensemble_type}_ensemble'
    verify_ensemble_type(ensemble_type)
    
    ### Load drought events
    syn_droughts = pd.read_csv(f"./pywrdrb/drought_metrics/{ensemble_type}_ensemble_ssi{SSI_WINDOW}_drought_events.csv")
    obs_droughts = pd.read_csv(f"./pywrdrb/drought_metrics/observed_ssi{SSI_WINDOW}_drought_events.csv")

    # Make 'severity', 'magnitude', and 'duration' all absolute values
    for col in ['severity', 'magnitude', 'duration']:
        if col in obs_droughts.columns:
            obs_droughts[col] = np.abs(obs_droughts[col])
        if col in syn_droughts.columns:
            syn_droughts[col] = np.abs(syn_droughts[col])
    
    ### Plot scatter of drought metrics
    fname = f"{ensemble_type}_delMontague_ssi{SSI_WINDOW}_drought_metrics_scatter.png"
    fname = f"{FIG_DIR}/drought_metrics/{fname}"
    
    drought_metric_scatter_plot(obs_drought_metrics=obs_droughts, 
                                syn_drought_metrics=syn_droughts, 
                                x_char='severity', 
                                y_char='magnitude', 
                                color_char='duration',
                                y_scale='linear',
                                x_scale='linear',
                                x_lims=(1, 5),
                                y_lims=(0, 150),
                                color_label='Duration (Months)',
                                x_label='Severity Abs(Min(SSI))',
                                y_label='Magnitude Abs(Sum(SSI))',
                                add_marginals=True,
                                fname=fname)
    
    ### Plot scatter of droughts w/ shortage metrics
    fname = f"{ensemble_type}_delMontague_ssi{SSI_WINDOW}_drought_metrics_scatter_with_shortage.png"
    fname = f"{FIG_DIR}/drought_metrics/{fname}"
    
    
    drought_metric_scatter_plot(obs_drought_metrics=None, 
                                syn_drought_metrics=syn_droughts, 
                                x_char='severity', 
                                y_char='magnitude', 
                                color_char='max_duration_delMontague',
                                size_char='total_shortage_delMontague',
                                fname=fname)
    
    
    print(f"Scatter plot saved to {fname}")