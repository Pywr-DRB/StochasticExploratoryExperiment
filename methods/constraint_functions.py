import numpy as np
from scipy.stats import wilcoxon
from scipy.stats import kstest
from scipy.stats import uniform, kstest

def uniform_ks_test(M, alpha=0.05):
    """KS test against theoretical uniform distribution"""
    flat_data = M.flatten()
    # Scale to [0,1] for scipy uniform distribution
    scaled_data = flat_data / 79
    statistic, p_value = kstest(scaled_data, uniform.cdf)
    
    # True if distributions are similar
    if p_value < alpha:
        return 1.0
    else:      
        return 0.0 

def ks_constraint(Qh, Qs, alpha=0.1):
    """Two-sample KS test for distribution equality"""
    statistic, p_value = kstest(Qs, lambda x: np.searchsorted(np.sort(Qh), x) / len(Qh))
    
    # True if distributions are similar
    if p_value < alpha:
        return 1.0
    else:      
        return 0.0 

def willcoxon_test(Qh, Qs,
                   log=True, 
                   alpha=0.05):
    """
    Perform the Wilcoxon signed-rank test to compare two related samples.

    Parameters:
    Qh (list or array-like): Historic flows.
    Qs (list or array-like): Synthetic flows.
    alpha (float): Significance level for the test.

    Returns:
    bool: True if the null hypothesis is rejected, False otherwise.
    """
    if log:
        # Convert to log scale
        d = np.log(Qh + 1) - np.log(Qs + 1)
    else:
        # Direct difference
        d = Qh - Qs
        
    stat, p_value = wilcoxon(d)
    
    if p_value < alpha:
        return 0.0
    else:
        return 1.0