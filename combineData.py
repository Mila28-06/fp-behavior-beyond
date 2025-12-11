# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 12:52:25 2025

@author: sconrad
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


nt = False

# Define where the stuff is
if nt == True:
    tankfolder = r'\\vs03.herseninstituut.knaw.nl\VS03-CSF-1\Conrad\Innate_approach\Data_analysis\24.35.01\\'
else:
    tankfolder = r'\\vs03.herseninstituut.knaw.nl\VS03-CSF-1\Conrad\Innate_approach\Data_analysis\24.35.01\\freelymoving\\'

siteList = ['ZI-L', 'ZI-R', 'SC-L', 'SC-R', 'SC to ZI-L', 'SC to ZI-R', 'PAG-L', 'PAG-R']

allDat = [[] for _ in range(18)]
trialDat = {site: {'approach': [], 'avoid': [], 'NR': [], 'IR': []} for site in siteList}  # Dictionary to separate data by site
trialDat_laser_algn = {site: {'approach': [], 'avoid': [], 'NR': [], 'IR': []} for site in siteList}  # Dictionary to separate data by site
lagDat = {site: {'approach': [], 'ITI': [], 'IR': []} for site in siteList}
ITIDat = {site: {'ITI': []} for site in siteList}

# Iterate through experiment folder
files = [f for f in os.listdir(tankfolder) if f.endswith('.pkl') and f != 'allDatComb.pkl' and f != 'approach_times_since_trial_start.pkl' ]

for file in files:
    with open(os.path.join(tankfolder, file), 'rb') as f:
        data = pickle.load(f)
        
        # Extract site information
        site = data.get('site', 'Error')  # Use 'Error' if site is missing
        
    
        # print(f'{site}') 
        
        # Z-scored data
        allDat[0].append(data['ZdFoF'])
        
        if not np.any(np.isnan(data['ZdFoFITI'])):
            allDat[4].append(data['ZdFoFITI'])
            if site in ITIDat:
                ITIDat[site]['ITI'].append(data['ZdFoFITI'])

        
        # Store trial-specific data, approach
        if not np.any(np.isnan((data['ZdFoFApproach']))):
            allDat[12].append(data['ZdFoFApproach'])
            allDat[13].append(data['ZdFoFApproach_trialOnset'])
            if site in trialDat:
                trialDat[site]['approach'].append(data['ZdFoFApproach'])
                trialDat_laser_algn[site]['approach'].append(data['ZdFoFApproach_trialOnset'])
                
        # Store trial-specific data, IR onset
        if not np.any(np.isnan((data['IR_ZdFoFApproach']))):
            if site in trialDat:
                trialDat[site]['IR'].append(data['IR_ZdFoFApproach'])
                trialDat_laser_algn[site]['IR'].append(data['IR_ZdFoFApproach_trialOnset'])

        # avoid
        if not np.any(np.isnan((data['ZdFoFAvoid']))):
            allDat[14].append(data['ZdFoFAvoid'])
            allDat[15].append(data['ZdFoFAvoid_trialOnset'])
            if site in trialDat:
                trialDat[site]['avoid'].append(data['ZdFoFAvoid'])
                trialDat_laser_algn[site]['avoid'].append(data['ZdFoFAvoid_trialOnset'])
         
        # lag correlations
        if not np.any(np.isnan((data['lag_correlations_approach']))):
            if site in lagDat:
                lagDat[site]['approach'].append(data['lag_correlations_approach'])
                
        if not np.any(np.isnan((data['lag_correlations_ITI']))):
            if site in lagDat:
                lagDat[site]['ITI'].append(data['lag_correlations_ITI'])
                
        if not np.any(np.isnan((data['lag_correlations_IR']))):
            if site in lagDat:
                lagDat[site]['IR'].append(data['lag_correlations_IR'])

                
        
        # print([data['mouse'], data['session']])    

        if not np.isscalar(data['ZdFoFNR']):
            allDat[16].append(data['ZdFoFNR'])
            if 'ZdFoFNR_yoked' in data:
                allDat[17].append(data['ZdFoFNR_yoked'])
                
            if site in trialDat:
                if 'ZdFoFNR_yoked' in data:
                    trialDat[site]['NR'].append(data['ZdFoFNR_yoked'])
                    
                trialDat_laser_algn[site]['NR'].append(data['ZdFoFNR'])

        # Speed data
        if (not np.isscalar(data['speedTrialsMov'])) and (np.sum(data['speedTrialsMov']) != 0):
            allDat[3].append(data['speedTrials']) 
            allDat[7].append(data['speedTrialsMov'][data['speedTrialsMov'][:, 0] != 0])
            
        if (not np.isscalar(data['speedITI'])) and (np.sum(data['speedITI']) != 0):
            allDat[5].append(data['speedITI'])

        # Traces for green and isosbestic channels
        allDat[1].append(data['Gdata'])
        allDat[2].append(data['Idata'])
        # allDat[8].append(data['InitGdata'])
        # allDat[9].append(data['InitIdata'])
        if not np.isscalar(data['ITIGdata']):
            allDat[10].append(data['ITIGdata'])
            allDat[11].append(data['ITIIdata'])

# Combine data
allDatComb = {
    # 'trialSignal': np.vstack(allDat[0]), # useless?
    # 'Gtrace': np.vstack(allDat[1]), # useless?
    # 'Itrace': np.vstack(allDat[2]), # useless?
    'trialSpeed': np.concatenate(allDat[3]) if nt == True else np.array([]),
    # 'ITIsignal': np.vstack(allDat[4])if np.isscalar(allDat[4]) else np.array([]), # useless?
    'ITIspeed': np.concatenate(allDat[5]) if nt == True else np.array([]),
    # 'ZdFoFinit': np.vstack(allDat[6]) if [data['mouse'], data['session']] != ['109436', '2025_03_13_'] else np.nan,
    'speedTrialsMov': np.vstack(allDat[7]) if nt == True else np.array([]),
    # 'InitGdata': np.vstack(allDat[8]),
    # 'InitIdata': np.vstack(allDat[9]),
    # 'ITIGdata': np.vstack(allDat[10])if np.isscalar(allDat[10]) else np.array([]),
    # 'ITIIdata': np.vstack(allDat[11])if np.isscalar(allDat[11]) else np.array([]),
    # 'approachSignal': np.vstack(allDat[12]),
    # 'approachSignal_trialOnset': np.vstack(allDat[13]),
    # 'avoidSignal': np.vstack(allDat[14])if np.isscalar(allDat[14]) else np.array([]),
    # 'avoidSignal_trialOnset': np.vstack(allDat[15])if np.isscalar(allDat[15]) else np.array([]),
    # 'NRsignal': np.vstack(allDat[16])if np.isscalar(allDat[16]) else np.array([]),
    # 'NRsignal_yoked': np.vstack(allDat[17])if np.isscalar(allDat[17]) else np.array([]),


    # SITE SPECIFIC DATA
    # movement aligned
    'trialData': {site: {'approach': np.vstack(values['approach']) if values['approach'] else np.array([]),
                        'avoid': np.vstack(values['avoid']) if values['avoid'] else np.array([]),
                        'NR': np.vstack(values['NR']) if values['NR'] else np.array([]),
                        'IR': np.vstack(values['IR']) if values['IR'] else np.array([])}
                  for site, values in trialDat.items()},  # Store separated site data
    # prey laser onset aligned             
    'trialData_trialOnset': {site: {'approach': np.vstack(values['approach']) if values['approach'] else np.array([]),
                        'avoid': np.vstack(values['avoid']) if values['avoid'] else np.array([]),
                        'NR': np.vstack(values['NR']) if values['NR'] else np.array([]),
                        'IR': np.vstack(values['IR']) if values['IR'] else np.array([])}
                 for site, values in trialDat_laser_algn.items()},  # Store separated site data
    
    'ITI': {site: {'ITI': np.vstack(values['ITI']) if values['ITI'] else np.array([])}
            for site, values in ITIDat.items()},
    
    #lag correlations
    'lag correlations': {site: {'approach': np.vstack(values['approach']) if values['approach'] else np.array([]),
                         'ITI': np.vstack(values['ITI']) if values['ITI'] else np.array([]),
                         'IR': np.vstack(values['IR']) if values['IR'] else np.array([])}
                         for site, values in lagDat.items()}
}

# Save combined data
with open(os.path.join(tankfolder, 'allDatComb.pkl'), 'wb') as f:
    pickle.dump(allDatComb, f)

# Plotting

# traceTiming = np.arange(allDatComb['InitGdata'].shape[1])
# plt.figure()

# # Green signal
# plt.plot(traceTiming, np.mean(allDatComb['InitGdata'], axis=0), color='#64a858')
# plt.fill_between(traceTiming,
#                  np.mean(allDatComb['InitGdata'], axis=0) + np.std(allDatComb['InitGdata'], axis=0),
#                  np.mean(allDatComb['InitGdata'], axis=0) - np.std(allDatComb['InitGdata'], axis=0),
#                  color='#64a858', alpha=0.3)

# # Isosbestic signal
# plt.plot(traceTiming, np.mean(allDatComb['InitIdata'], axis=0), color='#7b8be8')
# plt.fill_between(traceTiming,
#                  np.mean(allDatComb['InitIdata'], axis=0) + np.std(allDatComb['InitIdata'], axis=0),
#                  np.mean(allDatComb['InitIdata'], axis=0) - np.std(allDatComb['InitIdata'], axis=0),
#                  color='#7b8be8', alpha=0.3)

# plt.box(False)
# plt.title('Filtered & Baseline Corrected\nGreen and Isosbestic Signals\nFor Prey Approach')
# plt.ylabel('Z-Score')
# plt.xlabel('Time')
# plt.show()
