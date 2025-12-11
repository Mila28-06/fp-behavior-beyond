# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 13:04:44 2025

@author: sconrad
"""

import numpy as np

def bootstrap_data(data, num_boots, sig):
    """
    Bootstraps data for conditioning photometry experiments.

    Parameters:
    data (ndarray): Photometry data (trials x timepoints)
    num_boots (int): Number of bootstraps (e.g., 5000)
    sig (float): Alpha level (e.g., 0.01)

    Returns:
    ndarray: Lower and upper confidence intervals (2 x timepoints)
    
    adapted from Isis Alonso-Lozares, adapted from Philip Jean-Richard-dit-Bressel, adapted from Colin Clifford
    """
    num_trials, window = data.shape

    # Minimum 2 trials to avoid crossing oscillations
    if num_trials > 3:
        # Prep bootstrapping variables
        data_boots = np.zeros((num_boots, window))
        bootsCI = np.zeros((2, window))

        for b in range(num_boots):
            # Bootstrap data by resampling trials with replacement
            trial_array = np.random.randint(0, num_trials, num_trials) # confused on this part
            data_boots[b, :] = np.mean(data[trial_array, :], axis=0)

        # Calculate bootstrap confidence intervals
        data_boots = np.sort(data_boots, axis=0)
        lower_conf_index = int(np.ceil(num_boots * (sig / 2))) + 1 #change to minus 1?
        upper_conf_index = int(np.floor(num_boots * (1 - sig / 2)))

        bootsCI[0, :] = data_boots[lower_conf_index, :]
        bootsCI[1, :] = data_boots[upper_conf_index, :]
    else:
        print('Less than 3 trials - bootstrapping skipped')
        bootsCI = np.full((2, window), np.nan)

    return bootsCI
