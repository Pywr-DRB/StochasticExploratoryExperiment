#%% 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from methods.plotting.drought import drought_metric_scatter_plot
from config import FIG_DIR

#%% Load data from export

ensemble_type = 'stationary'


#%% Load drought events

syn_droughts = pd.read_csv(f"./pywrdrb/drought_metrics/{ensemble_type}_ensemble_drought_events_with_shortage_metrics.csv")


#%% Plot shortage events


## Plot scatter of drought metrics
fname = f"{ensemble_type}_delMontague_drought_metrics_scatter_with_shortage.png"
fname = f"{FIG_DIR}/drought_metrics/{fname}"

drought_metric_scatter_plot(obs_drought_metrics=None, 
                            syn_drought_metrics=syn_droughts, 
                            x_char='severity', 
                            y_char='magnitude', 
                            color_char='max_duration_delMontague',
                            size_char='max_shortage_delMontague',
                            fname=fname)
