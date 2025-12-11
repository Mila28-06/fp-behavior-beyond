# -*- coding: utf-8 -*-
"""
Created on 24/11/25

This code plot summary graphs per animal per site, with the purpose of data cleaning. 

perhaps best if run on data without trial time exclsion?

@author: sconrad
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
# from perm_test_array import perm_test_array
# from bootstrap_data import bootstrap_data
# from bootstrap_data_bias_corrected import bootstrap_data
# from consec_idx import consec_idx
import pandas as pd
import os

def build_filename(date, ID, channel):
    return f"{date}{ID} Channel {channel}.pkl"

nt = False
initiate_aligned = True
# perm_testing = False # if u wanna test difference between two signals
trial_type = ['approach', 'IR'] #choose one or more trial type here ('approach', 'avoid', 'NR' )

if nt:
    exp_type = 'nt'
else:
    exp_type = 'fm laser'
    
filePath = 'W:\\Conrad\\Innate_approach\\Data_collection\\24.35.01\\'

r_log = pd.read_csv(f"{filePath}\\recordinglog.csv", sep=None, engine="python", encoding='cp1252')
r_log = r_log[r_log['Exp'] == exp_type]

r_log = r_log[r_log['notes'] != 'no ttl alignment']
r_log = r_log[r_log['added to db?'] != 'neurotar data corrupt']

animal_ids = r_log['ID'].unique()
sites = pd.unique(r_log[['1','2']].values.ravel('K'))


# Load combined data
if nt == True:
    tankfolder = r'\\vs03.herseninstituut.knaw.nl\VS03-CSF-1\Conrad\Innate_approach\Data_analysis\24.35.01\\'
else: 
    tankfolder = r'\\vs03.herseninstituut.knaw.nl\VS03-CSF-1\Conrad\Innate_approach\Data_analysis\24.35.01\\freelymoving\\'

files = [f for f in os.listdir(tankfolder) if f.endswith('.pkl') and f != 'allDatComb.pkl' and f != 'approach_times_since_trial_start.pkl' ]

thres = 5  # Consecutive threshold length
pre = 5
post = 25

for ID in animal_ids:
    for site in sites:
        data__single_animal_site = []
        approach_data = []
        control_data = []

        all_animal_data = r_log[r_log['ID'] == ID]

        animal_by_site_data_ch1 = all_animal_data[all_animal_data['1'] == site]
        animal_by_site_data_ch2 = all_animal_data[all_animal_data['2'] == site]
        
        if len(animal_by_site_data_ch1) == 0 and len(animal_by_site_data_ch2) == 0:
            continue

        dates_ch1 = animal_by_site_data_ch1['Date'].unique()
        dates_ch2 = animal_by_site_data_ch2['Date'].unique()

        for date in dates_ch1:
            filename = build_filename(date, ID, 1)
            data_path = os.path.join(tankfolder, filename)

            if os.path.exists(data_path):
                with open(data_path, 'rb') as f:
                    temp_data = pickle.load(f)
                data__single_animal_site.append(temp_data)
            

        for date in dates_ch2:
            filename = build_filename(date, ID, 2)
            data_path = os.path.join(tankfolder, filename)

            if os.path.exists(data_path):
                with open(data_path, 'rb') as f:
                    temp_data = pickle.load(f)
                data__single_animal_site.append(temp_data)
    
    
        for item in data__single_animal_site:
            approach_data.append(item['ZdFoFApproach']) if not np.any(np.isnan(item['ZdFoFApproach'])) else True
            if nt:
                control_data.append(item['ZdFoFNR_yoked']) if not np.any(np.isnan(item['ZdFoFNR_yoked'])) else True
            else:
                control_data.append(item['IR_ZdFoFApproach']) if not np.any(np.isnan(item['IR_ZdFoFApproach'])) else True

        

        
        if len(approach_data) > 0:
            approach_data = np.vstack(approach_data)  
         
        if len(trial_type) > 1:
            if len(control_data) > 0:
                control_data = np.vstack(control_data)
            plot_data = [approach_data, control_data]
        else:
            plot_data = [approach_data]

        
        
        
        plt.figure()
    

        for index, signal in enumerate(plot_data):
            
            if len(signal) == 0:
                print(f"No {trial_type[index]} trials for {ID} {site}")
                continue
            # # Bootstrapping
            # print('bootstrapping ...')
            # btsrp_app = bootstrap_data(signal, 10000, 0.0001)
            
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
            plt.plot(ts, np.mean(signal, axis=0), color=plt_color, label=trial_type[index])
            plt.fill_between(ts,
                             np.mean(signal, axis=0) + np.std(signal, axis=0) / np.sqrt(len(signal)),
                             np.mean(signal, axis=0) - np.std(signal, axis=0) / np.sqrt(len(signal)),
                             color=plt_color, alpha=0.3)
         

        
            # Plot signal
          
        
            # ymax = ymax_total - index/2
            
            
            # # Bootstrap significance
            # tmp = np.where(btsrp_app[1, :] < 0)[0]
            # if len(tmp) > 1:
            #     id = tmp[consec_idx(tmp, thres)]
            #     plt.plot(ts[id], ymax * np.ones((len(ts[id]), 2)) + 1, 's', 
            #              markersize=7, markerfacecolor=plt_color, color=plt_color)
                
            # tmp = np.where(btsrp_app[0, :] > 0)[0]
            # if len(tmp) > 1:
            #     id = tmp[consec_idx(tmp, thres)]
            #     plt.plot(ts[id], ymax * np.ones((len(ts[id]), 2)) + 1, 's', 
            #              markersize=7, markerfacecolor=plt_color, color=plt_color)
                
        
            
        # if perm_testing:
        #     print('Permuting ...')
        #     ymax = ymax +0.5
        #     # perm_test, _ = perm_test_array(trialData[site][trial_type[0]], d['ITI'][site][trial_type[1]], 1000)
        #     perm_test, _ = perm_test_array(plot_data[0], plot_data[1], 1000)
        #     perm_hits = np.where(perm_test<0.05)[0] # find significant points in permutation test
        #     perm_x = perm_hits[consec_idx(perm_hits, thres)] # consecutive thersholding
        #     plt.plot(ts[perm_x], ymax * np.ones((len(ts[perm_x]), 2)), 's', markersize=7, markerfacecolor=  [0.659, 0.427, 0.91], color= [0.659, 0.427, 0.91])
            
        plt.axvline(x=0, linestyle='--', color='black', linewidth=1.5)
        plt.axhline(y=0, linestyle='--', color='black', linewidth=1.5)
        
        # Hide the top and right spines
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        # Set tick parameters to remove right and top ticks
        plt.gca().tick_params(axis='x', which='both', direction='out', bottom=True, top=False)
        plt.gca().tick_params(axis='y', which='both', direction='out', left=True, right=False)
        plt.title(f'{ID} Site: {site} - {trial_type} trials')
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

