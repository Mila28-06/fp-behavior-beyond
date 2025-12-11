# -*- coding: utf-8 -*-
"""
Created on 18 Nov 2025

@author: sconrad

plots cross correlations of fiber photometry signal vs movemnet 

to do:
    get working for other variables such head turning
    adapt for fm set ups
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from boxoff import boxoff
from violinPlots import violinPlots
from scipy import stats

nt = False
trial_type = ['approach', 'IR'] #choose one or more trial type here ('approach', 'avoid', 'NR' )
combine_hemispheres = False

xlabels = list(range(-5, 6, 1))

# Load combined data
if nt == True:
    tankfolder = r'\\vs03.herseninstituut.knaw.nl\VS03-CSF-1\Conrad\Innate_approach\Data_analysis\24.35.01\\'
else: 
    tankfolder = r'\\vs03.herseninstituut.knaw.nl\VS03-CSF-1\Conrad\Innate_approach\Data_analysis\24.35.01\\freelymoving\\'

with open(f'{tankfolder}allDatComb.pkl', 'rb') as f:
    d = pickle.load(f)
    
lagData = d['lag correlations']  # for prey laser onset data, rmember to change title ;)
 

# tmp = np.where(perm_VR<0.05)[0]
# id = tmp[consec_idx(tmp, thres)]
# plt.plot(ts[id], 2 * np.ones((len(ts[id]), 2))-0.5, 's', markersize=7, markerfacecolor= [0.65, 0.65, 0.65], color=[0.65, 0.65, 0.65])

# data_list = [lagData]

# if combine_hemispheres:
#     combined_list = [{}, {}]
#     data_list = [lagData, d['ITI']]
    
#     for x, data_set in enumerate(data_list):
#         for key in data_set.keys():
#             # Only process "-L" entries to avoid duplicates
#             if key.endswith('-L'):
#                 base = key[:-2]  # e.g. "PAG" from "PAG-L"
#                 left = data_set.get(f"{base}-L")
#                 right = data_set.get(f"{base}-R")
        
#                 if left is not None and right is not None:
#                     combined_list[x][f"{base}-both"] = {
#                         k: np.concatenate([left[k], right[k]]) for k in left.keys()
#                     }
#                 else:
#                     # Handle missing side gracefully
#                     combined_list[x][f"{base}-both"] = left or right
                    
#         data_list[x] = combined_list[x]
        
#     d['ITI'] = data_list[1]

# lagData = data_list[0]


for site, data in lagData.items():
           
# if site == site_specific:
    # if 'ITI' not in trial_type:
    #     if len(trial_type) > 1:
    #         plot_data = list(range(len(trial_type)))
    #         for i, signal in enumerate(trial_type):
    #             plot_data[i] = data[trial_type[i]][~np.isnan(data[trial_type[i]]).any(axis=1)]
    #     else:
    #         if len(data[trial_type[0]]) > 0:
    #             plot_data = [data[trial_type[0]][~np.isnan(data[trial_type[0]]).any(axis=1)]]
    #             print(f"Total trials for {site}: {len(data[trial_type[0]])}")
            
    if 'ITI' in trial_type:
        if len(trial_type) > 1:
            if len(data[trial_type[0]]) > 0:
                lag1 = data[trial_type[0]][~np.isnan(data[trial_type[0]]).any(axis=1)]
                print(f"Total trials for {site}: {len(data[trial_type[0]])}")
                lag2 = data[trial_type[1]][~np.isnan(data[trial_type[1]]).any(axis=1)]
                print(f"Total ITI for {site}: {len(data[trial_type[1]])}")

                plot_data = [lag1, lag2]
                
    if 'IR' in trial_type:
        if len(trial_type) > 1:
            if len(data[trial_type[0]]) > 0:
                lag1 = data[trial_type[0]][~np.isnan(data[trial_type[0]]).any(axis=1)]
                print(f"Total prey trials for {site}: {len(data[trial_type[0]])}")
                lag2 = data[trial_type[1]][~np.isnan(data[trial_type[1]]).any(axis=1)]
                print(f"Total IR trials for {site}: {len(data[trial_type[1]])}")

                plot_data = [lag1, lag2]
                
        # else:
        #     plot_data = [d['ITI'][site]['ITI']]
        #     print(f"Total ITI traces for {site}: {len(plot_data[0])}")
    peak_lag_per_approach = []
    peak_lag_per_ITI = []
    for x, data in enumerate(plot_data):
        plt.figure()
    
        L = np.size(plot_data[0],1)
        mid = L // 2
    
        # slice range
        rng = 150        # number of points on each side
        sl = slice(mid - rng, mid + rng + 1)
        
        # full lag vector
        lags = np.arange(-(L//2), L//2 + 1)
        
        # truncated lag vector
        lags_trunc = lags[sl]
        
        for cor in data:
            plt.plot(lags_trunc, cor[sl], color = [0.3, 0.3, 0.3], alpha = 0.2)
            
            if x == 0:
                peak_lag_per_approach.append(lags_trunc[np.argmax(cor[sl])])
            else:
                peak_lag_per_ITI.append(lags_trunc[np.argmax(cor[sl])])

        
        
        if x == 0:
            plt_color = [0.47, 0.67, 0.19] #green
        else:
            plt_color = [0, 0, 0] # black
            
        plt.plot(lags_trunc, np.mean(data[:,sl],axis=0), color = plt_color)
        plt.fill_between(lags_trunc,
                 np.mean(data[:,sl], axis=0) + np.std(data[:,sl], axis=0) / np.sqrt(len(data[:,sl])),
                 np.mean(data[:,sl], axis=0) - np.std(data[:,sl], axis=0) / np.sqrt(len(data[:,sl])),
                 color=plt_color, alpha=0.3)
        
        # max point
        max_point = np.max(np.mean(data[:,sl],axis=0))
        max_where = (np.argmax(np.mean(data[:,sl],axis=0)))
        plt.plot(lags_trunc[max_where], max_point, 'D', color = plt_color)
            
        plt.title(f" {site} {trial_type[x]}")
        plt.xlabel(f'Lag (s) \n Peak lag of average is {round(lags_trunc[max_where]/30*1000)} ms from 0')
        plt.ylabel('Normalized cross correlation')
        plt.xticks(np.arange(-rng, rng+1, 30), xlabels)   # adjust tick spacing as needed
        boxoff()
        
        
  
    # # plot max per trial
    # # Shapiro and variance assumption tests
    # [_, p1] = stats.shapiro(peak_lag_per_approach)
    # [_, p2] = stats.shapiro(peak_lag_per_ITI)
    # if p1 < 0.05 or p2 < 0.05:
    #     print("Normality test failed, using MW")
    # [_, p3] = stats.levene(peak_lag_per_approach, peak_lag_per_ITI)
    # if p3 < 0.05:
    #     print("Variances unequal, using welch's")
        
    # if p1 > 0.05 and p2 > 0.05:
    #     if p3 >0.5:
    #         equal_var = True
    #     else:
    #         equal_var = False
    #     [t, p] = stats.ttest_ind(peak_lag_per_approach, peak_lag_per_ITI, equal_var)
    
    # else:
    #     [U, p] = stats.mannwhitneyu(peak_lag_per_approach, peak_lag_per_ITI)
        
    # print(f"p value for {site}: {p}")
    
    # violinPlots(peak_lag_per_approach, peak_lag_per_ITI, [0.47, 0.67, 0.19], [0, 0, 0], 'p',
    #             my_title = 'Peak lags per trial', my_ylabel = 'Lag (s)', yLimits = [-rng, rng],
    #             paired = False, my_yticks = [np.arange(-rng, rng+1, 30), xlabels])
    
    