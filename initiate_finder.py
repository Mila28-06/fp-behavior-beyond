# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 15:09:59 2025

find approach initiation times 

@author: conrad
"""
def initiate_finder_DLC(ttlFile, eventTS_RWD, r_log, fIdx, l, setUp, 
                    stim_dur, sr, plotknee, regenerate_approach_trial_times):
    
    import pickle
    import numpy as np
    import pandas as pd
    # import scipy.io
    # import glob
    import matplotlib.pyplot as plt
    from kneed import KneeLocator
    from scipy.signal import argrelextrema
    from boxoff import boxoff
    
    
    
    def scotts_knee(x, y, knee, title):
        # x = x[::-1]
        y = y[::-1]
        
        plt.figure()
        plt.plot(x,y)
        plt.axvline(knee, color = 'black', ls =':' )
        plt.title(title)
        boxoff()
        plt.show()
        
    def detect_knees(x, y, trim, curve="concave", direction="increasing"):
        x = x[:-trim] # initiate will never be this fast
        y = y[trim:]
        
        knee_index = 0
        
        data = y.copy()
        
        shift_counter = 0
    
        while len(data) > 2:  # Continue until there's enough data to find a knee
            kl = KneeLocator(x[:len(data)], data, curve=curve, direction=direction)
            if kl.knee is not None:
                
                # Remove the data before the knee
                if kl.knee == 0:
                    knee_index = 1
                    shift_counter += 1
                else:
                    # kl.plot_knee()
                    # knees = kl.knee
                    knee_index = kl.knee
                    
                    break 
                
                data = data[knee_index:]  # Slice the data after the knee
                x = x[:-knee_index]  # Slice the x-values as well
            else:
                break  # No more knees detected
                
        return knee_index
      
    def initiate_finder(trial_starts, trial_ends, snout_distance, initiation_array, plotknee, trim_array, trial_type):
        for i, (start, end) in enumerate(zip(trial_starts, trial_ends), 1):
            if end-start > 600: # in case of IR trials lasting longer than 20 seconds
                end = start + 600
            y = snout_distance[start:end][::-1]
            x = np.arange(len(y))  # trial-relative time
            
            trim = trim_array[i]
                        
            local_min = argrelextrema(y, np.less)
            local_min = local_min[0]
            
            if local_min.size == 0: # case where animal always approaches towards laser
                initiation = 0
                
                if plotknee == True:
                    
                    title = f"{trial_type} approach only"
                    scotts_knee(x, y, initiation, title)
                     
            else:    
                knee = detect_knees(x, y, trim)
                initiation = x[-1] - (knee + trim) # flip it back
                
                if plotknee == True:
                    title = f"{trial_type} Approach initiation" 
                    scotts_knee(x, y, initiation, title)
                
            initiation_array[i-1] = int(initiation)
            
        return initiation_array
    
        
        
    # apr_times = [np.nan]
    dlc_savePath = 'W:\\Conrad\\Innate_approach\\Data_analysis\\24.35.01\\DLC'
    approach_time_path = f"{dlc_savePath}\\approach_times_since_trial_start.pkl"
    # pd.to_pickle(apr_times, approach_time_path)

    # with open('W:\\Conrad\\Innate_approach\\Data_analysis\\24.35.01\\DLC\\prey_snout_distance_combined_DLC.pkl', 'rb') as f:
        # prey_snout_distance = pickle.load(f)
        
    with open(f"{dlc_savePath}\\{r_log['Date'][l]}{r_log['ID'][l]} DLC.pkl", 'rb') as f:
        all_data = pickle.load(f)
        prey_snout_distance = all_data['data'][3]
        IR_snout_distance = all_data['data'][4]
        snout_speed = all_data['data'][5]
    # if regenerate_approach_trial_times:
    #     if l == range(len(r_log))[0]:
    #         apr_times = [np.nan]
    #     else:
    #         with open(approach_time_path, 'rb') as f:
    #             apr_times = pickle.load(f)
    
    
    # Detect where prey trials start and end
    valid = ~np.isnan(prey_snout_distance) # boolean mask
    prey_trial_starts = np.where(np.diff(valid.astype(int)) == 1)[0] + 1
    if l == 14:
        prey_trial_starts = prey_trial_starts[1:] # error where trial occured but not ttl registered by rwd. should save this trial somehow? 
        
    prey_trial_ends   = np.where(np.diff(valid.astype(int)) == -1)[0] + 1
    
    # Detect where IR trials start and end
    valid = ~np.isnan(IR_snout_distance) # boolean mask
    IR_trial_starts = np.where(np.diff(valid.astype(int)) == 1)[0] + 1
    IR_trial_ends   = np.where(np.diff(valid.astype(int)) == -1)[0] + 1
    
    
    
    # convert from pandas to array
    prey_snout_distance = prey_snout_distance.to_numpy()
    IR_snout_distance = IR_snout_distance.to_numpy()
    
    
    ttl = pd.read_csv(ttlFile)
    animal_id = str(r_log['ID'][l])
    date = str(r_log['Date'][l]).replace('_', '')
# %% checks for drift between ttl pulses and camera (none seen :) ) 
    if np.any(ttl['Event'] == 'sync'):
        ttl_syncs = ttl.loc[ttl['Event']=='sync'].reset_index(drop = True)
        ttl_syncs['Time'] = (pd.to_datetime(ttl_syncs['DateTime']) - pd.to_datetime(ttl_syncs['DateTime'][0])).dt.total_seconds() + (ttl_syncs['Milliseconds'] / 1000) - ttl_syncs['Milliseconds'][0] / 1000
        ttl_syncs = np.diff(ttl_syncs['Time'][::2])
        
        
        fp_filePath = 'W:\\Conrad\\Innate_approach\\Data_collection\\24.35.01\\'
       
        cam_ttl_file_path = f"{fp_filePath}{animal_id}\\{animal_id}_{date}_001\\{animal_id}_{date}_001_pioverhead_triggers.csv"
        cam_ttl = pd.read_csv(cam_ttl_file_path)
        if 'input type' in cam_ttl.columns:
            cam_ttl = cam_ttl[cam_ttl['input type'] == 'recieved']            
        cam_ttl_diff = np.diff(cam_ttl[cam_ttl['ttl source']=='sync_ttl']['time elapsed (s)'])
        sync_check = (ttl_syncs - cam_ttl_diff)*1000
        
        print(f"The difference between start and end of habituation timing is {round(sync_check[0])} ms")
        print(f"The difference between end of habituation and end session is timing is {round(sync_check[1])} ms")
        
    
# %%

    if ttl['Event'][0] == 'Received trigger':
        #subtracts neurotar ttl input from first laser ttl output
        
        ####
        ### CHANGE THIS SO IT TAKES LAST POSSIBLE NEUROTAR SIGNAL
        ttl['Time'] = (pd.to_datetime(ttl['DateTime']) - pd.to_datetime(ttl['DateTime'][0])).dt.total_seconds() + (ttl['Milliseconds'] / 1000) - ttl['Milliseconds'][0] / 1000
        ###
        
        ttl = ttl.loc[ttl['Event'] !='Received trigger'].reset_index(drop = True) # added because received trigger sometimes added twice
        ttl = ttl.loc[ttl['Event'] !='sync'].reset_index(drop = True)
        ttl['Time'] = ttl['Time'] - stim_dur/sr
        
    elif ttl['Event'][0] == 'sync':
        ttl['Time'] = (pd.to_datetime(ttl['DateTime']) - pd.to_datetime(ttl['DateTime'][0])).dt.total_seconds() + (ttl['Milliseconds'] / 1000) - ttl['Milliseconds'][0] / 1000
        ttl = ttl.loc[ttl['Event'] !='sync'].reset_index(drop = True)
        ttl['Time'] = ttl['Time'] - stim_dur/sr
    else:
        print('No sync pulse detected at start, something has gone wrong')
        

    if fIdx.size == 1:
        expanded_fIdx = [fIdx*3 +offset for offset in range(3)]
    else: 
        expanded_fIdx = [i + offset for i in fIdx*3 for offset in range(3)]
    
    ttl = ttl.drop(expanded_fIdx, axis = 0).reset_index(drop = True)    
        
    ttl = ttl['Time'][::3]
    
    drift = (ttl.iloc[-1] - ttl.iloc[0])*30 - (eventTS_RWD[-1] - eventTS_RWD[0])
    drift_ratio = drift/(eventTS_RWD[-1] - eventTS_RWD[0])
    
    print(f"The frame difference between last and first ttl pulse sent vs recieved is {drift}")

    # syncing between what i see on camera + ttl file
    ttl_file_diff = np.array(np.diff(ttl)*30, dtype = int)
    ttl_select_diff = np.diff(prey_trial_starts)
    ttl_check_pis = ttl_file_diff - ttl_select_diff
    
    ttl_check_pi_rwd = ttl_file_diff - np.diff(eventTS_RWD)
    print(f"The difference in frames between ttl file and recieved RWD system is {ttl_check_pi_rwd}")
    # ttl_pi_drift.append(ttl_check[-1]-ttl_check[0]) # no drift between sync pi and camera :)
    
    anchor_point = np.where(abs(ttl_check_pis) == np.min(abs(ttl_check_pis)))[0][0]
    
    if np.min(abs(ttl_check_pis)) > 2:
        print('total offset of more than 2 frames detected') 
    
    ttl = ttl.reset_index(drop = True)*sr
    ttl_anchored = ttl - ttl[anchor_point] # now clipped
    
    prey_trial_starts_anchored = prey_trial_starts -prey_trial_starts[anchor_point] 
    unsynced_trial_mask = np.where(prey_trial_starts_anchored - ttl_anchored > 30)[0] # 1 second difference ttl pulse and recorded, frames maybe dropped
    if unsynced_trial_mask.size > 0:
        print(f"Sync error {r_log['Date'][l]} {r_log['ID'][l]} for trials {unsynced_trial_mask}")
        
    synced_trials = np.where(prey_trial_starts_anchored - ttl_anchored < 30)[0]   

    # check that indexing is ok, probably will get cleaner data this way
    offset_cam_pi = prey_trial_starts_anchored - ttl_anchored
    
    
    # knee finder...
    trim = 10 # ignore this part of the trial
    prey_trim_array = np.full(len(ttl), trim)
    IR_trim_array = np.full(len(IR_trial_starts), trim)
    
    if isinstance(r_log['IR Trimmer'][l], str):
        trim_temp = r_log['IR Trimmer'][l].replace(" ", "")
        trim_temp = np.array(list(map(str, trim_temp.split(','))))
        for trim_value in trim_temp:
            IR_trim_trial_idx = int(trim_value[0])
            IR_trim_array[IR_trim_trial_idx] = (IR_trial_ends[IR_trim_trial_idx] - 
                                             IR_trial_starts[IR_trim_trial_idx] - 
                                             int(trim_value[2:]))
        # r_log['IR Trimmer'][l]

        
    approach_initiation = np.zeros(len(prey_trial_starts), dtype = int)
    IR_initiation = np.zeros(len(IR_trial_starts), dtype = int)
    
    approach_initiation = initiate_finder(prey_trial_starts, prey_trial_ends, prey_snout_distance, approach_initiation, plotknee, prey_trim_array, trial_type = 'prey')
    IR_initiation = initiate_finder(IR_trial_starts, IR_trial_ends, IR_snout_distance, IR_initiation, plotknee, IR_trim_array, trial_type = 'IR')

    # if np.any(IR_trial_ends-IR_trial_starts < 9*sr):
    #     print(f"Approach IR trial <9sec detected for {animal_id} on {date}")
        
    # if np.any(IR_trial_ends-IR_trial_starts < 18*sr):
    #     print(f"Approach IR trial <18sec detected for {animal_id} on {date}")
    
    # visually check for drift, useful dont delete
    # plt.figure()
    # plt.plot(offset_cam_pi[synced_trials])
    # boxoff()
    # plt.title("off set between ttl pulse pi and obesrved laser from camera, per trial")
    
    short_prey_trials_idx = np.where(approach_initiation < (stim_dur-2*sr))[0] # select trials < 18 seconds to filter out chase trials and nonapproaches
    IR_app_idx = np.where(IR_initiation < (stim_dur-2*sr))[0] 
    # approach_initiation = approach_initiation[synced_trials]
    

    # Compute the union of the two arrays
    app_idx = np.intersect1d(short_prey_trials_idx, synced_trials)
    
    # Wrap back into a tuple to keep same structure as np.where returns
    initTrace = np.full(len(ttl), np.nan)
    initTrace[app_idx] = round(offset_cam_pi[app_idx]) + approach_initiation[app_idx] # init start (frames) relative to trial start
    approachTrials = eventTS_RWD[app_idx] + initTrace[app_idx] # init start in context of entire 
    
    IR_idx = IR_trial_starts 
    IR_initTrace = np.full(len(IR_idx), np.nan)

    IR_onset_idx = []
    IR_initTrace[IR_app_idx] = IR_initiation[IR_app_idx] # moving IR trials initiation times, relative to IR trial start
    IR_onset_idx = IR_trial_starts[IR_app_idx] # IR laser onset, for approach trials
    IR_approachTrials = IR_onset_idx + IR_initTrace[IR_app_idx] # moving IR trials initiation times, relative to recording start
    
    approach_snout_speed = []
    for index in approachTrials:
        approach_snout_speed.append(snout_speed[int(index)-150:int(index)+750]) # sorry for hard coding. 5 seconds before and 25 seconds after initiation
    
    IR_snout_speed = []
    for index in IR_approachTrials:
        IR_snout_speed.append(snout_speed[int(index)-150:int(index)+750])
    # if regenerate_approach_trial_times:
    #     apr_times.append(initTrace)
    #     pd.to_pickle(apr_times, approach_time_path)
    
    return (approachTrials, IR_approachTrials, app_idx, initTrace, IR_initTrace,
            approach_snout_speed, IR_snout_speed,IR_onset_idx, IR_app_idx, IR_idx,
            drift, drift_ratio)
