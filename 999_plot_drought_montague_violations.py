import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from methods.plotting.drought import drought_metric_scatter_plot
from config import verify_ensemble_type
from config import FIG_DIR

if __name__ == "__main__":
    
    # Get ensemble type from command line arguments
    ensemble_type = sys.argv[1]
    inflow_type = f'{ensemble_type}_ensemble'
    verify_ensemble_type(ensemble_type)
    
    ### Load drought events
    syn_droughts = pd.read_csv(f"./pywrdrb/drought_metrics/{ensemble_type}_ensemble_drought_events_with_shortage_metrics.csv")
    obs_droughts = pd.read_csv(f"./pywrdrb/drought_metrics/observed_drought_events.csv")

    ### Plot scatter of droughts w/ shortage metrics
    fname = f"{ensemble_type}_delMontague_drought_metrics_scatter_with_shortage.png"
    fname = f"{FIG_DIR}/drought_metrics/{fname}"

    drought_metric_scatter_plot(obs_drought_metrics=None, 
                                syn_drought_metrics=syn_droughts, 
                                x_char='severity', 
                                y_char='magnitude', 
                                color_char='max_duration_delMontague',
                                size_char='total_shortage_delMontague',
                                fname=fname)
    
    fname = f"{ensemble_type}_delMontague_drought_metrics_scatter.png"
    fname = f"{FIG_DIR}/drought_metrics/{fname}"

    drought_metric_scatter_plot(obs_drought_metrics=obs_droughts, 
                                syn_drought_metrics=syn_droughts, 
                                x_char='severity', 
                                y_char='magnitude', 
                                color_char='duration',
                                fname=fname)
    
    print(f"Scatter plot saved to {fname}")