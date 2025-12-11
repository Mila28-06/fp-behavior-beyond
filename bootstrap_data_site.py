# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 15:45:00 2025

@author: sconrad
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from perm_test_array import perm_test_array
# from bootstrap_data import bootstrap_data
from bootstrap_data_bias_corrected import bootstrap_data
from consec_idx import consec_idx


nt = False
initiate_aligned = True
perm_testing = True # if u wanna test difference between two signals
# site_specific = 'ZI-L' 
trial_type = ['approach', 'IR'] #choose one or more trial type here ('approach', 'avoid', 'NR' )
combine_hemispheres = False

# Load combined data
if nt == True:
    tankfolder = r'\\vs03.herseninstituut.knaw.nl\VS03-CSF-1\Conrad\Innate_approach\Data_analysis\24.35.01\\'
else: 
    tankfolder = r'\\vs03.herseninstituut.knaw.nl\VS03-CSF-1\Conrad\Innate_approach\Data_analysis\24.35.01\\freelymoving\\'

with open(f'{tankfolder}allDatComb.pkl', 'rb') as f:
    d = pickle.load(f)

if initiate_aligned == True:
    trialData = d['trialData']  # for movement aligned data
else: 
    trialData = d['trialData_trialOnset']  # for prey laser onset data, rmember to change title ;)
 


thres = 5  # Consecutive threshold length
pre = 5
post = 25

# PLOTTING SPEED
# plt.figure()
# plt.plot(np.nanmean(d['ITIspeed'], axis = 0), color=[0.6, 0.6, 0.6])
# plt.fill_between(range(0,d['ITIspeed'].shape[1]),
#                 ( np.nanmean(d['ITIspeed'], axis = 0) + np.nanstd(d['ITIspeed'], axis=0) / np.sqrt(len(d['ITIspeed']))),
#                  (np.nanmean(d['ITIspeed'], axis = 0) - np.nanstd(d['ITIspeed'], axis=0) / np.sqrt(len(d['ITIspeed']))),
#                  color=[0.6, 0.6, 0.6], alpha=0.3)
     
# plt.plot(np.nanmean(d['speedTrialsMov'], axis=0), color=[0.78, 0, 0])
# plt.fill_between(range(0,d['speedTrialsMov'].shape[1]),
#                 ( np.nanmean(d['speedTrialsMov'], axis = 0) + np.nanstd(d['speedTrialsMov'], axis=0) / np.sqrt(len(d['speedTrialsMov']))),
#                  (np.nanmean(d['speedTrialsMov'], axis = 0) - np.nanstd(d['speedTrialsMov'], axis=0) / np.sqrt(len(d['speedTrialsMov']))),
#                  color=[0.78, 0, 0], alpha=0.3)
# ymin, ymax = plt.ylim()
# plt.vlines(150, ymin=ymin, ymax=ymax, linestyle='--', color='black')
# ax = plt.gca()
# ax.set_xlim([50, 600])
# plt.xlabel('Frames')
# plt.ylabel('NT speed, mm/s')




# tmp = np.where(perm_VR<0.05)[0]
# id = tmp[consec_idx(tmp, thres)]
# plt.plot(ts[id], 2 * np.ones((len(ts[id]), 2))-0.5, 's', markersize=7, markerfacecolor= [0.65, 0.65, 0.65], color=[0.65, 0.65, 0.65])

data_list = [trialData]

if combine_hemispheres:
    combined_list = [{}, {}]
    data_list = [trialData, d['ITI']]
    
    for x, data_set in enumerate(data_list):
        for key in data_set.keys():
            # Only process "-L" entries to avoid duplicates
            if key.endswith('-L'):
                base = key[:-2]  # e.g. "PAG" from "PAG-L"
                left = data_set.get(f"{base}-L")
                right = data_set.get(f"{base}-R")
        
                if left is not None and right is not None:
                    combined_list[x][f"{base}-both"] = {
                        k: np.concatenate([left[k], right[k]]) for k in left.keys()
                    }
                else:
                    # Handle missing side gracefully
                    combined_list[x][f"{base}-both"] = left or right
                    
        data_list[x] = combined_list[x]
        
    d['ITI'] = data_list[1]

trialData = data_list[0]


for site, data in trialData.items():
    
    # if site != 'PAG-L' or site != 'PAG-R':
    #     continue
   
    plt.figure()
        
# if site == site_specific:
    if 'ITI' not in trial_type and 'IR' not in trial_type:
        if len(trial_type) > 1:
            plot_data = list(range(len(trial_type)))
            for i, signal in enumerate(trial_type):
                plot_data[i] = data[trial_type[i]][~np.isnan(data[trial_type[i]]).any(axis=1)]
                print(f"Total {trial_type[i]} trials for {site}: {len(data[trial_type[i]])}")
        else:
            if len(data[trial_type[0]]) > 0:
                plot_data = [data[trial_type[0]][~np.isnan(data[trial_type[0]]).any(axis=1)]]
                print(f"Total trials for {site}: {len(data[trial_type[0]])}")
            
    if 'ITI' in trial_type or 'IR' in trial_type:
        if len(trial_type) > 1:
            if len(data[trial_type[0]]) > 0:
                signal = data[trial_type[0]][~np.isnan(data[trial_type[0]]).any(axis=1)]
                print(f"Total trials for {site}: {len(data[trial_type[0]])}")
                
                if 'ITI' in trial_type:
                    signal_ITI = d['ITI'][site]['ITI']
                    print(f"Total ITI traces for {site}: {len(signal_ITI)}")
                    plot_data = [signal, signal_ITI]
                else:
                    signal_IR = data[trial_type[1]][~np.isnan(data[trial_type[1]]).any(axis=1)]
                    print(f"Total IR traces for {site}: {len(signal_IR)}")
                    plot_data = [signal, signal_IR]
                
                
        else:
            plot_data = [d['ITI'][site]['ITI']]
            print(f"Total ITI traces for {site}: {len(plot_data[0])}")

    
    
    # finding values to plot bCI and perm test data later...
    
    if len(plot_data) > 1:
        
        ymax_total = np.max([np.mean(plot_data[0], axis = 0), np.mean(plot_data[1], axis = 0)]) + 1
    else:
        ymax_total = np.max([np.mean(plot_data[0], axis = 0)]) + 1 + 2

    
    for index, signal in enumerate(plot_data):
        
    
        # Bootstrapping
        print('bootstrapping ...')
        btsrp_app = bootstrap_data(signal, 10000, 0.0001)
        
        # Colors for plotting
        if trial_type[index] == 'approach':
            plt_color = [0.47, 0.67, 0.19] #green
        elif trial_type[index] == 'NR':
            plt_color = [0.65, 0.65, 0.65] #grey
        elif trial_type[index] == 'ITI' or trial_type[index] == 'IR':
            plt_color = [0.416, 0.741, 0.741] #blue
        else:
            plt_color = [0.78, 0, 0] # red, avoid
          
        ts = np.linspace(-pre, post, signal.shape[1])
    
        # Plot signal
        plt.plot(ts, np.mean(signal, axis=0), color=plt_color, label=trial_type[index])
        plt.fill_between(ts,
                         np.mean(signal, axis=0) + np.std(signal, axis=0) / np.sqrt(len(signal)),
                         np.mean(signal, axis=0) - np.std(signal, axis=0) / np.sqrt(len(signal)),
                         color=plt_color, alpha=0.3)
    
        ymax = ymax_total - index/2
        
        
        # Bootstrap significance
        tmp = np.where(btsrp_app[1, :] < 0)[0]
        if len(tmp) > 1:
            id = tmp[consec_idx(tmp, thres)]
            plt.plot(ts[id], ymax * np.ones((len(ts[id]), 2)) + 1, 's', 
                     markersize=7, markerfacecolor=plt_color, color=plt_color)
            
        tmp = np.where(btsrp_app[0, :] > 0)[0]
        if len(tmp) > 1:
            id = tmp[consec_idx(tmp, thres)]
            plt.plot(ts[id], ymax * np.ones((len(ts[id]), 2)) + 1, 's', 
                     markersize=7, markerfacecolor=plt_color, color=plt_color)
            
    
        
    if perm_testing:
        print('Permuting ...')
        ymax = ymax +0.5
        # perm_test, _ = perm_test_array(trialData[site][trial_type[0]], d['ITI'][site][trial_type[1]], 1000)
        perm_test, _ = perm_test_array(trialData[site][trial_type[0]], trialData[site][trial_type[1]], 1000)
        perm_hits = np.where(perm_test<0.05)[0] # find significant points in permutation test
        perm_x = perm_hits[consec_idx(perm_hits, thres)] # consecutive thersholding
        plt.plot(ts[perm_x], ymax * np.ones((len(ts[perm_x]), 2)), 's', markersize=7, markerfacecolor=  [0.659, 0.427, 0.91], color= [0.659, 0.427, 0.91])
        
    plt.axvline(x=0, linestyle='--', color='black', linewidth=1.5)
    plt.axhline(y=0, linestyle='--', color='black', linewidth=1.5)
    
    # Hide the top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Set tick parameters to remove right and top ticks
    plt.gca().tick_params(axis='x', which='both', direction='out', bottom=True, top=False)
    plt.gca().tick_params(axis='y', which='both', direction='out', left=True, right=False)
    plt.title(f'Site: {site} - {trial_type} trials')
    plt.ylabel('Z-Score')
    if initiate_aligned == True and 'NR' not in trial_type:
        plt.xlabel('Movement onset (s)')
    elif initiate_aligned == True and 'NR' in trial_type:
        plt.xlabel('Shuffled movement onset (s)')
    else:
        plt.xlabel('Prey Laser onset (s)')
    # plt.legend()
    # plt.savefig(f'{tankfolder}{site}_{trial_type}.png', transparent = True)
    plt.show()

