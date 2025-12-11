# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 15:10:04 2025
 bootstrapping iwth bias correction accelerated 
 
@author: conrad
"""

import numpy as np
from scipy.stats import bootstrap

def bootstrap_data(data, num_boots, sig):
    """
    Bootstraps data for conditioning photometry experiments using SciPy's bootstrap.

    Parameters:
    data (ndarray): Photometry data (trials x timepoints)
    num_boots (int): Number of bootstraps (e.g., 5000)
    sig (float): Alpha level (e.g., 0.01)

    Returns:
    ndarray: Lower and upper confidence intervals (2 x timepoints)
    """
    num_trials, window = data.shape

    # Minimum 2 trials to avoid crossing oscillations
    if num_trials > 3:
        bootsCI = np.full((2, window), np.nan)  # Init output array

        # Loop over timepoints since scipy's bootstrap doesn't support axis=-1 for 2D
        for t in range(window):
            timepoint_data = data[:, t]

            # Run bootstrap on 1D slice across trials
            res = bootstrap(
                (timepoint_data,),
                np.mean,
                confidence_level=1 - sig,
                n_resamples=num_boots,
                method='BCa',  # Use 'BCa' for bias correction
                random_state=None,
                axis=0
            )

            # Store lower and upper bounds
            bootsCI[0, t] = res.confidence_interval.low
            bootsCI[1, t] = res.confidence_interval.high
    else:
        print('Less than 3 trials - bootstrapping skipped')
        bootsCI = np.full((2, window), np.nan)

    return bootsCI
