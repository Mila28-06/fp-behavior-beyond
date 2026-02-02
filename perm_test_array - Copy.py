# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 12:58:09 2025

@author: sconrad
"""

import numpy as np
from itertools import combinations
from scipy.special import comb



def perm_test_array(sample1, sample2, permutations, sidedness='both', exact=False, report_exact=True, show_progress=0):
    all_observations = np.vstack((sample1, sample2))
    observed_difference = np.nanmean(sample1, axis=0) - np.nanmean(sample2, axis=0)
    s1_n = sample1.shape[0]
    all_n = all_observations.shape[0]
    win_size = sample1.shape[1]

    if not exact and permutations > comb(all_n, s1_n):
        exact = True
        all_combinations = list(combinations(range(all_n), s1_n))
        if report_exact:
            print(f"Number of permutations ({permutations}) is higher than the number of possible combinations ({len(all_combinations)});\n"
                  f"Running an exact test (minimum p = {1 / (len(all_combinations) + 1):.5f})")
        permutations = len(all_combinations)

    random_differences = np.zeros((win_size, permutations))

    for n in range(permutations):
        if show_progress and n % show_progress == 0:
            print(f"Permutation {n + 1} of {permutations}")

        if exact:
            permutation = list(all_combinations[n]) + list(set(range(all_n)) - set(all_combinations[n]))
        else:
            permutation = np.random.permutation(all_n)

        random_sample1 = all_observations[permutation[:s1_n], :]
        random_sample2 = all_observations[permutation[s1_n:], :]

        random_differences[:, n] = np.nanmean(random_sample1, axis=0) - np.nanmean(random_sample2, axis=0)

    p_val = np.zeros(win_size)

    if sidedness == 'both':
        for t in range(win_size):
            p_val[t] = (np.sum(np.abs(random_differences[t, :]) > np.abs(observed_difference[t])) + 1) / (permutations + 1)
    elif sidedness == 'smaller':
        p_val = (np.sum(random_differences < observed_difference[:, None], axis=1) + 1) / (permutations + 1)
    elif sidedness == 'larger':
        p_val = (np.sum(random_differences > observed_difference[:, None], axis=1) + 1) / (permutations + 1)

    return p_val, observed_difference
