"""
Defines multiple objective functions used in the MOEA+Kirsch experiment.

Each function should accept historic flows (Qh) and synthetic flows (Qs) as input,
and return a single float value representing the objective.
"""
import numpy as np
import pandas as pd
from sglib.droughts.ssi import SSIDroughtMetrics

def combine_historic_and_synthetic(Qh, Qs):
    """
    Combine historic and synthetic flow data into a single array,
    synthetic are added to the end of the historic data.
    
    Parameters:
    -----------
    Qh : array-like
        Historic flow data.
    Qs : array-like
        Synthetic flow data.
    
    Returns:
    --------
    array-like
        Combined flow data.
    """
    Qcombined = Qh.copy()
    Qcombined = np.concatenate((Qcombined, Qs), axis=0)
    return Qcombined.flatten() 


def get_drought_metrics(Qh, Qs):
    """
    Calculate drought metrics based on historic and synthetic flows.
    Parameters:
    -----------
    Qh : array-like
        Historic flow data.
    Qs : array-like
        Synthetic flow data.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing drought metrics such as start date, end date, severity, and duration.
    """
    Qcombined = combine_historic_and_synthetic(Qh, Qs)
    
    # Convert Qcombined to a pd.Series
    Qcombined_series = pd.Series(Qcombined, 
                                 index=pd.date_range(start='1945-01-01', 
                                                     periods=len(Qcombined), 
                                                     freq='D'))
    
    try:
        # Run SSI and drought calculation
        ssi = SSIDroughtMetrics(timescale='M', window=12)
        ssi_values = ssi.calculate_ssi(data=Qcombined_series)
        droughts = ssi.calculate_drought_metrics(ssi_values)
    
        # Keep only the drought in the synthetic period (>2024-01-01)
        droughts = droughts[droughts['start_date'] >= '2024-01-01']

    except Exception as e:
        print(f"Error calculating drought metrics: {e}\n\n")
        
        print("\n   Qcombined shape:", Qcombined.shape)
        print("\n   Qcombined head:", Qcombined[:10])
        print("\n   Qcombined tail:", Qcombined[-10:])
        print("\n   Qcombined dtype:", Qcombined.dtype)
        print(f"\n   Qcombined mean: {np.mean(Qcombined)}")
        print(f'\n   Qcombined min: {np.min(Qcombined)}')
        print(f'\n   Qcombined max: {np.max(Qcombined)}')
        print(f'\n   Qcombined std: {np.std(Qcombined)}')
        
    
    return droughts

def mean_drought_severity(Qh, Qs):
    """
    Calculate the drought severity based on historic and synthetic flows.
    
    Parameters:
    -----------
    Qh : array-like
        Historic flow data.
    Qs : array-like
        Synthetic flow data.
    
    Returns:
    --------
    float
        Drought severity value.
    """
    droughts = get_drought_metrics(Qh, Qs)
    if droughts.empty:
        return 0.0
    
    # Get the mean severity
    mean_severity = droughts['severity'].mean()
    
    return -1.0 * mean_severity

def mean_drought_duration(Qh, Qs):
    """
    Calculate the drought duration based on historic and synthetic flows.
    
    Parameters:
    -----------
    Qh : array-like
        Historic flow data.
    Qs : array-like
        Synthetic flow data.
    
    Returns:
    --------
    float
        Drought duration value.
    """
    droughts = get_drought_metrics(Qh, Qs)
    if droughts.empty:
        return 0.0
    
    # Get the mean duration
    mean_duration = droughts['duration'].mean()
    
    return mean_duration

def mean_drought_magnitude(Qh, Qs):
    """
    Calculate the drought magnitude based on historic and synthetic flows.
    
    Parameters:
    -----------
    Qh : array-like
        Historic flow data.
    Qs : array-like
        Synthetic flow data.
    
    Returns:
    --------
    float
        Drought magnitude value.
    """
    droughts = get_drought_metrics(Qh, Qs)
    if droughts.empty:
        return 0.0
    
    # Get the mean magnitude
    mean_magnitude = droughts['magnitude'].mean()
    
    return -1.0 * mean_magnitude

