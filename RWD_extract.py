# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 13:11:28 2025
# extracts data from RWD system, processes it, and collates traces

# default traces made for TTL inputs (trials), but additional scripts can be
# added here to make traces around other events of interest (e.g. movement
# initiation, movement during iter trial intervals, etc)

#scott conrad 02/12/2024, adapted from Isis Alonso-Lozares

"""

debug = False
fm_exp = True # set to false if analyzing nt experiments

regenerate_approach_trial_times = False # used for shuffling trial times for NR trials
automate_initiate_finder = True # set to false to use manually curated initate times (only for trials, not for ITI)

inspect_traces = False # plots individual traces on same graph

plotsignal = False
plotknee = False # plot 'knee' of movement, used to visually verify calculated initiation point

plot_Zscore = True # set to false if you want to see % dFoF

plot_apr_trace = False # plot traces for approach trials
plot_ITI_trace = False # plot traces for movement during Inter-Trial Intervals
plotheatmap = True
plotidvtrials = False
plot_angular_velocity = False
plot_cros_cor = False #cross correlations between signal and fwd movement

initiate_exclusion = 5 # in seconds



import sys
sys.path.append('/Users/sconrad/Documents/GitHub/fp-behavior-beyond')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nt_ITI_movement import nt_ITI_movement
from initiate_finder_DLC import initiate_finder_DLC
from scipy import signal
import glob
import pickle
import random
from boxoff import boxoff
from scipy.stats import (ttest_ind, mannwhitneyu, shapiro, 
                         levene, ttest_rel, wilcoxon)


# Low-pass filter function
def lpFilter(data, sr, lowpass_cutoff, filt_order, db_atten):
    
    # # Design a low-pass filter using Butterworth filter design (old filtering method)
    # nyquist = 0.5 * sr
    # normal_cutoff = lowpass_cutoff / nyquist
    
    # # Design a Butterworth filter
    # b, a = signal.butter(filt_order, normal_cutoff, btype='low', analog=False)
    
    # # Apply the filter to the data
    # # lp_data = signal.filtfilt(b, a, data) #0 phase, noncausal
    # lp_data = signal.lfilter(b, a, data) # IIR (non-linear phase delay)
    
    
    sos = signal.butter(
        filt_order,
        lowpass_cutoff,
        fs=sr,
        btype='low',
        output='sos'
    )
    
    lp_data = signal.sosfiltfilt(sos, data)
    
    
    # # calculate group delay (if using lfilt)

    # b, a = signal.sos2tf(sos) #sosfilt
    # w, gd = signal.group_delay((b, a), fs=sr) #sosfilt

    # # w, gd = signal.group_delay((b, a), fs=sr) #lfilter
    # plt.plot(w, gd)
    # plt.xlabel("Frequency (Hz)")
    # plt.ylabel("Group delay (samples)")
    # plt.title("Group Delay of Butterworth Low-Pass Filter")
    # plt.grid(True)
    # plt.show()
    
    # passband = w < lowpass_cutoff
    # avg_delay = np.mean(gd[passband])
    
    # print(f"Average passband group delay: {avg_delay:.2f} samples")
    # print(f"≈ {avg_delay / sr:.4f} seconds")

    return lp_data

# IRLS dF/F function
def IRLS_dFF(exp_signal, iso_signal, IRLS_constant):
    
    # adapted from [insert correct citation here] Jean-Philip Richard?
    import statsmodels.api as sm
    from statsmodels.robust.robust_linear_model import RLM
    from statsmodels.robust.norms import HuberT
    
    # Reshape and apply IRLS regression
    # Perform robust regression with bisquare weight function (equivalent to 'bisquare' in MATLAB)
    iso_signal_with_intercept = sm.add_constant(iso_signal)  # Add intercept term to iso_signal
    model = RLM(exp_signal, iso_signal_with_intercept, M=HuberT(IRLS_constant))  # Robust regression model
    results = model.fit()

    # Extract the coefficients (intercept, slope)
    IRLS_coeffs = results.params  # This will give the intercept and the slope
    
    # Compute the fitted isosbestic signal
    ft_iso_signal = np.polyval(IRLS_coeffs[::-1], iso_signal)  # Coefficients are reversed for numpy polyval
    
    # Calculate dFF
    dFF = (exp_signal - ft_iso_signal) / ft_iso_signal
    
    return dFF, ft_iso_signal

def apr_shuffle(apr_times):
    valid_arrays = [a for a in apr_times if isinstance(a, np.ndarray) and a.size > 0]
    combined = np.concatenate(valid_arrays)
    random.Random(710).shuffle(combined) # previously used 1337

    return combined


def parse_latency_string(s,sr):
    if pd.isna(s):
        return np.array([])  # Empty array if missing value
    
    # Force to string, remove Excel's forced-text apostrophes
    s = str(s).strip().lstrip("'")
    
    # Split on comma+space, then convert each to integer
    values = []
    for part in s.split(','):
        part = part.strip()
        if part:  # Skip empty
            try:
                values.append(int(float(part)*sr))
            except ValueError:
                pass  # Or log a warning
    return np.array(values)

def trial_cross_correlation(fp_signal, movement):
    n_trials = fp_signal.shape[0]
    xcorrs = []

    for t in range(n_trials):
        xcorr = signal.correlate(fp_signal[t], movement[t], mode='full')
        xcorrs.append(xcorr)

    return np.array(xcorrs)

def norm_xcorr(a, b):
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()
    return signal.correlate(a, b, mode='full') / len(a)

def fm_drift_corrector(rwd_index_array):
    fm_drift_factor = -0.00041931890844608486
    corrected_array = np.full(len(rwd_index_array), 1)
    for x, value in enumerate(rwd_index_array):
        corrected_array[x] = int(value + value*fm_drift_factor)
        
    return corrected_array
        

filePath = 'W:\\Conrad\\Innate_approach\\Data_collection\\24.35.01\\'
nt_savePath = 'W:\\Conrad\\Innate_approach\\Data_analysis\\24.35.01\\'
fm_savePath = 'W:\\Conrad\\Innate_approach\\Data_analysis\\24.35.01\\freelymoving\\'
ntFilePth = 'W:\\Conrad\\Innate_approach\\Data_collection\\Neurotar\\'
fm_approach_times_path = 'W:\\Conrad\\Innate_approach\\Data_analysis\\24.35.01\\DLC\\approach_times_since_trial_start.pkl'
nt_approach_times_path = f"{nt_savePath}\\approach_times_since_trial_start.pkl"    
kpms_general_path = 'E:/Conrad/DLC projects/DLC 06_01_2025/fm_subset_cropped/2026_01_14-14_00_06/results/'


if regenerate_approach_trial_times == False:
    # nt
    with open(nt_approach_times_path, 'rb') as f:
        nt_apr_times_file = pickle.load(f)
        shuffled_nt_apr_times = apr_shuffle(nt_apr_times_file)
        shuffled_nt_apr_times = shuffled_nt_apr_times[shuffled_nt_apr_times >= initiate_exclusion*30]
    # fm
    with open(fm_approach_times_path, 'rb') as f:
        fm_apr_times_file = pickle.load(f)
        shuffled_fm_apr_times = apr_shuffle(fm_apr_times_file)
else:
     shuffled_nt_apr_times = []

r_log = pd.read_csv(f"{filePath}\\recordinglog.csv", sep=None, engine="python", encoding='utf-8-sig')


if fm_exp:
    
    r_log = r_log[r_log['Exp'] == 'fm laser']
    r_log = r_log[:-1]
    # r_log = r_log[r_log['Movement quality'] == 'missing']
    apr_times = shuffled_nt_apr_times

else: 
    r_log = r_log[r_log['Exp'] == 'nt']
    apr_times = shuffled_nt_apr_times

    # r_log = r_log[r_log['Exp'] == 'nt' | r_log['Exp'] == 'fm laser' ]

    
# %%
r_log = r_log[r_log['notes'] != 'no ttl alignment']
r_log = r_log[r_log['added to db?'] != 'neurotar data corrupt']

r_log = r_log.reset_index()

animalIDs = r_log['ID'].unique()
dates = r_log['Date'].unique()
numChannels = 2

excluded_animals = ['105647']


# Preprocessing initialization
lowpass_cutoff = 3  # Low-pass cut-off in Hz
filt_order = 4 # was 0.95 as filter_steepness
db_atten = 90  # dB attenuation
sr = 30  # Sampling rate (Hz or FPS)
ma = 90 * sr  # Moving average window, default = 90 seconds
setUp = 120 * sr  # start data trim from 2 minutes before first trial
stim_dur = 20*sr # prey laser stimulation (seconds converted to frames)

# Trace window and settings
pre = 5  # 5 seconds before ttl
post = 25  # 25 seconds after
before = pre * sr
after = post * sr
traceTiming = np.arange(-pre, post + 1, sr / (before + after))
labelsX = np.arange(-pre, post + 1, 5)
thresh = 10 # speed threshold setting, to find locomotion start, original was 30

driftTable = np.zeros((len(r_log), 4))

fm_drift = []
fm_drift_ratio = []
all_bouts = []

# %%


for l in range(len(r_log)):
    
    idn = str(r_log['ID'][l])
    d = str(r_log['Date'][l]) 
    exp = r_log['Exp'][l]
    
    if idn in excluded_animals: # exclude from analysis
        continue
        
    # Load data
    if exp == 'nt':
        ntFile = f"{ntFilePth}Track_[{d.replace('_', '-', 2)}*{idn}_session*\\*.mat"
        ntFile = next(iter(glob.glob(ntFile, recursive = True)), None)
        manual_latencies_s = r_log['prey trial latency'][l]

        
    ttlFile = f"{ntFilePth}{idn}_{d.replace('_', '')}_01_ttl"
    rawData = pd.read_csv(f"{filePath}{d}{idn}\\Fluorescence.csv", skiprows=1)      
    timestamps = rawData.iloc[:, 0] / 1000
        
    eventTS = rawData.iloc[:, 1].str.contains('Input1*2*0', regex = False, na = False)
    eventTS = eventTS[eventTS].index
    
    
    fIdx = np.array([])
    # code for removing failed laser presentations
    if isinstance(r_log['Failed'][l], int):
        fIdx = np.array(int(r_log['Failed'][l]))
        eventTS = eventTS.delete(np.array(int(r_log['Failed'][l])))
    elif isinstance(r_log['Failed'][l], float) and  np.isnan(r_log['Failed'][l]) == False:
        fIdx = np.array(list(map(int, str(r_log['Failed'][l]).split('.'))))
        if fIdx[1] == 0:
            fIdx = np.delete(fIdx, 1)
        eventTS = eventTS.delete(fIdx)
    elif isinstance(r_log['Failed'][l], str):
    # Check if it's a single integer string (e.g., '0', '2')
        if r_log['Failed'][l].isdigit():
            fIdx = np.array(int(r_log['Failed'][l]))
        else:
            fIdx = np.array(list(map(int, str(r_log['Failed'][l]).split('.'))))
        
        eventTS = eventTS.delete(fIdx)


        
    if exp == 'nt':        
        trialClass = np.array([int(x) for x in str(r_log['trial class'][l]).split('.')]) # 0 is NR, 1 is avoid, 2 is approach
        NR_idx = np.where(trialClass == 0)[0] # index of trials

    else: 
        trialClass = np.array([])
    
        
    # code for selecting normal prey laser trials vs behind
    behindLaserIndex = []
    if pd.isnull(r_log['Behind trials'][l]) == False:
        behindLaserIndex =  np.array([int(x) for x in str(r_log['Behind trials'][l]).split('.')])
        eventTSBehind = eventTS[behindLaserIndex]
        eventTS = eventTS.delete(behindLaserIndex)
        
        
    else:
        eventTSBehind = np.nan
               
    if (l != 14 and exp != 'nt') or exp == 'nt':    
        clipStart = min(eventTS - setUp)
    else:
        clipStart = 0

    
    
    # clipEnd = max(eventTS) + 40 * sr
    clipEnd = len(rawData.iloc[:, 2]) - 5*sr # trims last 5 seconds
    
        
    last_event_noclip = eventTS[-1]
    
    eventTS = eventTS - clipStart
        
    for ch in range(1, numChannels + 1):
        site = r_log[str(ch)][l]
        
        # insert debug condition(s) below
        if debug and idn != '111610' and d != '2025_05_12':
            continue
        
        # # changes site based on histology, for later
        # if id in [x, y ,z]:
        #     if site == 'SC-L':
        #         site = 'sSC-L'
        #     if site == 'SC-R':
        #         site = 'sSC-R'
        
        data_set = []
        data_tile = []
               
        print(f"________________________________________________\nCurrent run: {d}{idn} {site}\n------------------------------------------------")
        
        chIsos = rawData.iloc[int(clipStart):int(clipEnd), 2 * ch]
        chGreen = rawData.iloc[int(clipStart):int(clipEnd), 2 * ch + 1]
        
        proper_trim = len(rawData.iloc[int(clipStart):int(last_event_noclip+25*sr), 2 * ch])
        
        # sample_signal = sample_signal = np.concatenate([np.full(300, 0), np.full(300, 1), np.full(300, 0)])
        # filtered_sample = lpFilter(sample_signal, sr, lowpass_cutoff, filt_order, db_atten)
        
        # plt.figure()
        # plt.plot(sample_signal)
        # plt.plot(filtered_sample)
        # plt.title('Raw and filtered sample data')

        # Apply low-pass filter
        lp_normDatG = lpFilter(chGreen, sr, lowpass_cutoff, filt_order, db_atten)
        lp_normDatI = lpFilter(chIsos, sr, lowpass_cutoff, filt_order, db_atten)
        dFoF, ft_iso_signal = IRLS_dFF(lp_normDatG, lp_normDatI, 3)

        if plotsignal == True:
            fig = plt.figure()
            # ts = list(range(5500,5800))
            plt.plot(lp_normDatG, color = 'green')
            plt.plot(lp_normDatI-65, color = 'blue')
            
            # Fit isosbestic signal to green channel and compute dF/F
            plt.title(f'{idn} {d} {site} Channel {ch} dF/F')
    
            plt.plot(dFoF+25, color = "grey")
        
        # Prepare to store traces
        traces = np.full((len(eventTS), before + after), np.nan)
        tracesGraw = np.full((len(eventTS), before + after), np.nan)
        tracesIraw = np.full((len(eventTS), before + after), np.nan)
        real_tracesGraw = np.full((len(eventTS), before + after), np.nan)
        real_tracesIraw = np.full((len(eventTS), before + after), np.nan)
             
        # get nt data
        if ch == 1:
            
            if exp == 'nt':
                
                
                IR_snout_speed = []
                IR_approachTrials = []
                IR_onset_idx = []
                IR_idx = []
                IR_app_idx = [] # could be nice to figure out for neurotar? 
                
                # Compute latency (adjusted)
                p_latency_ext = np.nan
                if not automate_initiate_finder:
                    
                    p_latency = parse_latency_string(manual_latencies_s, sr) - 5
                    if 2 in trialClass:
                        valid_mask = np.isin(trialClass, [1, 2])
                        
                        # Create an extended latency array
                        p_latency_ext = np.full_like(trialClass, np.nan, dtype=float)
                        p_latency_ext[valid_mask] = p_latency
                        
                        
                        
                (ITIidx, speedITI, approachTrials, 
                 speedTrials, ttl, 
                 app_idx, avd_idx, speedTrialsMov, 
                 avoidTrials, initTrace, approach_fwdSpeed) = nt_ITI_movement(ntFile, ttlFile, 
                                                           eventTS, sr, eventTSBehind, 
                                                              idn, d, fIdx, behindLaserIndex,
                                                              driftTable, l, trialClass, setUp,
                                                              stim_dur, thresh, automate_initiate_finder,
                                                              p_latency_ext,plot_angular_velocity, labelsX, traceTiming)
                
                                                                              
                if 2 in trialClass:
                    if not automate_initiate_finder:
                            # Mask for approach trials (class 2) where latency exceeds sr
                            approach_mask = (trialClass == 2) & (p_latency_ext > initiate_exclusion)
                            # Compute approach trial timestamps
                            approachTrials = eventTS[approach_mask] + p_latency_ext[approach_mask]
                            
                            app_idx = np.intersect1d(np.where(trialClass == 2)[0], np.where(p_latency_ext > initiate_exclusion)[0])
                   
                    else:
                            approach_mask = (trialClass == 2) & (initTrace > initiate_exclusion*sr) 
                            approachTrials = eventTS[approach_mask] + initTrace[approach_mask]
                            app_idx = np.intersect1d(np.where(trialClass == 2)[0], np.where(initTrace > initiate_exclusion*sr)[0])

                              
                if regenerate_approach_trial_times:
                    if l == range(len(r_log))[0]:
                        nt_approach_times = [np.nan]
                    else:
                        with open(nt_approach_times_path, 'rb') as f:
                            nt_approach_times = pickle.load(f)   
                            
                    nt_approach_times.append(initTrace[app_idx])
                    pd.to_pickle(nt_approach_times, nt_approach_times_path)   

                  
                                                                     
            else: 
                
                eventTS = fm_drift_corrector(eventTS)
                
                (approachTrials, IR_approachTrials, app_idx, initTrace, IR_initTrace,
                 approach_snout_speed, IR_snout_speed, IR_onset_idx, IR_app_idx, IR_idx,
                 drift, drift_ratio) = initiate_finder_DLC(ttlFile, eventTS, r_log, 
                                                           fIdx, l, setUp,
                                                           stim_dur,sr, plotknee, regenerate_approach_trial_times)
                 
                                                       
                IR_idx = fm_drift_corrector(IR_idx)
                IR_onset_idx = fm_drift_corrector(IR_onset_idx)
                IR_approachTrials = fm_drift_corrector(IR_approachTrials)
               

                                  
                fm_drift.append(drift)
                fm_drift_ratio.append(drift_ratio)
                    
                if regenerate_approach_trial_times:
                    if l == range(len(r_log))[0]:
                        fm_approach_times = [np.nan]
                    else:
                        with open(fm_approach_times_path, 'rb') as f:
                            fm_approach_times = pickle.load(f)   
                            
                    fm_approach_times.append(initTrace[app_idx])
                    pd.to_pickle(fm_approach_times, fm_approach_times_path)
                
                for iterator, potential_approaches in enumerate([initTrace, IR_initTrace]):
                    if len(potential_approaches) > 0:
                        approach_mask = potential_approaches > initiate_exclusion*sr 
                        if iterator == 0:
                            app_idx = np.where(potential_approaches > initiate_exclusion*sr)[0]
                            approachTrials = eventTS[approach_mask] + potential_approaches[approach_mask]
                        else: 
                            IR_app_idx = np.where(potential_approaches > initiate_exclusion*sr)[0]
                            IR_approachTrials = list(IR_idx[approach_mask] + potential_approaches[approach_mask])
                            for x,float_value in enumerate(IR_approachTrials):
                                IR_approachTrials[x] = int(float_value)

                
                ITIidx = np.nan
                speedITI = np.nan
                speedTrials = np.nan
                speedTrialsMov = np.nan
                avoidTrials = np.nan
                avd_idx = np.array([])
                ttl = eventTS
                long_trial_idx = np.where((initTrace > (stim_dur/2)))[0]

                # NR_idx = np.array(range(4))
                # NR_idx = NR_idx[~np.isin(NR_idx, app_idx)]    
                NR_idx = long_trial_idx
                # print(NR_idx)
            
            if len(NR_idx) > 0:
                if len(apr_times) < len(NR_idx):
                    apr_times = np.concatenate((apr_times, shuffled_nt_apr_times)) # there are less NR trials than approach so i need to reuse
                shuffled_times = apr_times[0:len(NR_idx)]
                apr_times = apr_times[len(NR_idx):]
        
        
        
        
# %%

        # capture manually curated problem prey trials for inspection and eventual deleting
        noisy_prey_trial_mask = np.full((len(eventTS), 2), False)
        inspect_noisy_prey = r_log['Noisy Prey Trials'][l]
        if isinstance(inspect_noisy_prey, str):
            inspect_noisy_prey = inspect_noisy_prey.replace(" ","")
            if ':' in inspect_noisy_prey:
                noisy_trial_ch_split = np.array(list(inspect_noisy_prey.split(';')))
                for channel in noisy_trial_ch_split:
                    channel_number = int(channel[0])-1
                    inspect_noisy_prey = list(channel[2:].split(','))
                    for trial_number in inspect_noisy_prey:
                        noisy_prey_trial_mask[int(trial_number), channel_number] = True
            
            else:
                inspect_noisy_prey = inspect_noisy_prey.split(',')
                for trial_number in inspect_noisy_prey:
                    noisy_prey_trial_mask[int(trial_number), :] = True
                    
        # capture manually curated problem IR trials for inspection and eventual deleting
        noisy_IR_trial_mask = np.full((len(IR_idx), 2), False)
        inspect_noisy_IR = r_log['Noisy IR Trials'][l]
        if isinstance(inspect_noisy_IR, str):
            inspect_noisy_IR = inspect_noisy_IR.replace(" ","")
            if ':' in inspect_noisy_IR:
                noisy_trial_ch_split = np.array(list(inspect_noisy_IR.split(';')))
                for channel in noisy_trial_ch_split:
                    channel_number = int(channel[0])-1
                    inspect_noisy_IR = list(channel[2:].split(','))
                    for trial_number in inspect_noisy_IR:
                        noisy_IR_trial_mask[int(trial_number), channel_number] = True
            
            else:
                inspect_noisy_IR = inspect_noisy_IR.split(',')
                for trial_number in inspect_noisy_IR:
                    noisy_IR_trial_mask[int(trial_number), :] = True
# %%
                    
        prey_inspect_mask = noisy_prey_trial_mask[:,ch-1]  
        IR_inspect_mask = noisy_IR_trial_mask[:,ch-1]            
          
                     
        # Extract traces for all trials, aligned to prey laser onset
        for m, idx in enumerate(eventTS):
            traces[m] = dFoF[idx - before:idx + after]
            
            # below traces are only needed for in depth inspection of signal
            tracesGraw[m] = lp_normDatG[idx - before:idx + after]
            tracesIraw[m] = lp_normDatI[idx - before:idx + after]
            real_tracesGraw[m] = chGreen[idx - before:idx + after]
            real_tracesIraw[m] = chIsos[idx - before:idx + after]
            
        if inspect_traces and np.any(prey_inspect_mask):
            for trial_number, mask in enumerate(prey_inspect_mask):
                if mask:
                    plt.figure()
                    plt.plot(traces[trial_number], color = [0.773, 0.467, 0.788])
                    plt.plot(tracesGraw[trial_number] - np.mean(tracesGraw[trial_number]), color = [0.459, 0.8, 0.361], alpha=0.7)
                    plt.plot(tracesIraw[trial_number] - np.mean(tracesIraw[trial_number]), color = [0.373, 0.671, 0.922], alpha=0.7)
                    plt.title(f'Traces for prey trial {trial_number} (index 0)')
                    boxoff()
        
        # extract IR traces, at IR onset
        IR_traces = []
        IR_tracesGraw = []
        IR_tracesIraw = []
        if len(IR_idx) > 0:
            for index in IR_idx:
                IR_traces.append(dFoF[index - before: index + after])
                IR_tracesGraw.append(lp_normDatG[index - before:index + after])
                IR_tracesIraw.append(lp_normDatI[index - before:index + after])

                
                
            if inspect_traces and np.any(IR_inspect_mask):
                 for trial_number, mask in enumerate(IR_inspect_mask):
                     if mask:
                         plt.figure()
                         plt.plot(IR_traces[trial_number], color = [0.773, 0.467, 0.788])
                         plt.plot(IR_tracesGraw[trial_number] - np.mean(IR_tracesGraw[trial_number]), color = [0.459, 0.8, 0.361], alpha=0.7)
                         plt.plot(IR_tracesIraw[trial_number] - np.mean(IR_tracesIraw[trial_number]), color = [0.373, 0.671, 0.922], alpha=0.7)
                         plt.title(f'Traces for IR trial {trial_number}, Ch {ch}')
                         boxoff()
        
        
        # # Align raw traces to movement if necessary. i do this to show isosbestic doesnt change with movement 
        # tracesInit = np.full((len(approachTrials), before + after), np.nan)
        tracesInitGraw = np.full((len(approachTrials), before + after), np.nan)
        tracesInitIraw = np.full((len(approachTrials), before + after), np.nan)
        approachTrials = approachTrials.astype(np.int64)
        if len(approachTrials) > 0:
            for m in range(len(approachTrials)):
                if np.size(dFoF[approachTrials[m] - before:approachTrials[m] + after]) == 900:
                    # tracesInit[m] = dFoF[approachTrials[m] - before:approachTrials[m] + after]
                    tracesInitGraw[m] = lp_normDatG[approachTrials[m] - before:approachTrials[m] + after]
                    tracesInitIraw[m] = lp_normDatI[approachTrials[m] - before:approachTrials[m] + after]
                    
                else: 
                    print("Not all approach trials included, something went wrong with clipping")       
        else:
            # tracesInit = np.nan
            tracesInitGraw = np.nan
            tracesInitIraw = np.nan
        
        
        
        # dFoF traces for prey approach, movement aligned
        appTracesInit = np.full((len(approachTrials), before + after), np.nan)
             
        if len(approachTrials) > 0:
            
            for m in range(len(approachTrials)):
                if np.size(dFoF[approachTrials[m] - before:approachTrials[m] + after]) == 900:
                    appTracesInit[m] = dFoF[approachTrials[m] - before:approachTrials[m] + after]
                
        else:
            appTracesInit = np.nan
            
         
        # dFoF traces for IR approach, movement aligned
    
     
        if len(IR_approachTrials) > 0:
            IR_appTracesInit = np.full((len(IR_approachTrials), before + after), np.nan)
            # IR_approachTrials = IR_approachTrials.astype(np.int64)
            for m in range(len(IR_approachTrials)):
                if np.size(dFoF[IR_approachTrials[m] - before:IR_approachTrials[m] + after]) == 900:
                    IR_appTracesInit[m] = dFoF[IR_approachTrials[m] - before:IR_approachTrials[m] + after]
                else:
                    print("Not all IR trials included, something went wrong with clipping")
                
        else:
            IR_appTracesInit = np.nan
          
                     
            
        # traces for avoid
        if not np.isnan(avoidTrials):
            avdTracesInit = np.full((len(avoidTrials), before + after), np.nan)
            
            avoidTrials = avoidTrials.astype(np.int64)
            
            if len(avoidTrials) > 0:
                for m in range(len(avoidTrials)):
                    if np.size(dFoF[avoidTrials[m] - before:avoidTrials[m] + after]) == 900:
                        avdTracesInit[m] = dFoF[avoidTrials[m] - before:avoidTrials[m] + after]
                    else:
                        print("Not all avoid trials included, something went wrong with clipping")
                
        else:
            avdTracesInit = np.nan
            
        # traces for No Response (NR), both prey laser aligned and shuffled initiation times from approach times (yoked)
        NRtraces_yoked = NRtrials = np.nan
        if len(NR_idx)>0:
            NRtrials = ttl[NR_idx] #affected by change in ttl 15/8/25
            NRtraces = np.full((len(NRtrials), before + after), np.nan)
            
            NRtrials = NRtrials.astype(np.int64)
            
            if len(NRtrials) > 0:
                for m in range(len(NRtrials)):
                    if np.size(dFoF[NRtrials[m] - before:NRtrials[m] + after]) == 900:
                        NRtraces[m] = dFoF[NRtrials[m] - before:NRtrials[m] + after]
                    else:
                        print("Not all NR trials included, something went wrong with clipping")
            else:
                NRtraces = np.nan
                
            # no response trials aligned to shuffled/yoked movement times
         
                
            NRtraces_yoked = np.full((len(NR_idx), before + after), np.nan)
            if regenerate_approach_trial_times == False:
                for k, nr in enumerate(NR_idx):
                    yoked_ts = int(eventTS[nr] + shuffled_times[k])
                    if np.size(dFoF[yoked_ts - before:yoked_ts + after]) == 900: #lazy fix, should edit so it tries another integer from shuffled times
                        NRtraces_yoked[k] = dFoF[yoked_ts - before:yoked_ts + after]
       
                

        
        # for ITIs
        if ~np.any(np.isnan(ITIidx)):
            tracesITI = np.full((len(ITIidx), before + after), np.nan)
            tracesITIGraw = np.full((len(ITIidx), before + after), np.nan)
            tracesITIIraw = np.full((len(ITIidx), before + after), np.nan)
    
            for m, idx in enumerate(ITIidx):
                if idx is not None:
                    tracesITI[m] = dFoF[idx - before:idx + after]
                    tracesITIGraw[m] = lp_normDatG[idx - before:idx + after]
                    tracesITIIraw[m] = lp_normDatI[idx - before:idx + after]
                else:
                    tracesITI = np.nan
                    tracesITIGraw = np.nan
                    tracesITIGraw = np.nan
        else:
            tracesITI = np.nan
            tracesITIGraw = np.nan
            tracesITIGraw = np.nan
        
    
        
        #################################
        # Baseline correction (z-scoring)
        #################################
        
        # traceDataSD = np.std(traces[:, :pre * sr], axis=1) # axis 1 is along row
        traceDataSD = np.std(traces[:, :pre * sr]) 
        
        ZdFoF = (traces - np.mean(traces[:, :pre * sr], axis=1).reshape(-1, 1)) / traceDataSD
        
        # Collate data 
        if len(IR_traces) > 0:
            IR_traces = np.vstack(IR_traces)
            IR_traceDataSD = np.std(IR_traces[:, :pre * sr]) 
            IR_ZdFoF = (IR_traces - np.mean(IR_traces[:, :pre * sr], axis=1).reshape(-1, 1)) / IR_traceDataSD
        
        traceDataSDG = np.std(tracesGraw[:, :pre * sr])
        Gdata = (tracesGraw - np.mean(tracesGraw[:, :pre * sr], axis=1).reshape(-1, 1)) / traceDataSDG
        
        traceDataSDI = np.std(tracesIraw[:, :pre * sr])
        Idata = (tracesIraw - np.mean(tracesIraw[:, :pre * sr], axis=1).reshape(-1, 1)) / traceDataSDI
        
        # APPROACH INITIATION and PREYLASER ALIGNED
        
        if approachTrials is not np.nan and len(approachTrials) > 0:
            ZdFoFApproach = (appTracesInit - np.mean(traces[app_idx,:pre*sr],axis=1).reshape(-1, 1)) / traceDataSD
            ZdFoFApproach_trialOnset = (traces[app_idx] - np.mean(traces[app_idx,:pre*sr],axis=1).reshape(-1, 1)) / traceDataSD
            
            if plot_apr_trace:
                if plot_Zscore:
                    data_set = ZdFoFApproach
                    data_title = 'Z-Score'
                else:
                    data_set = traces[app_idx]*100
                    data_title = '% dFoF'
                plt.figure()
                plt.plot(np.mean(data_set, axis = 0), color = [0.47, 0.67, 0.19])
                plt.fill_between(range(0,data_set.shape[1]),
                                 np.mean(data_set, axis = 0) + np.std(data_set, axis=0) / np.sqrt(len(data_set)),
                                 np.mean(data_set, axis = 0) - np.std(data_set, axis=0) / np.sqrt(len(data_set)),
                                 color = [0.47, 0.67, 0.19], alpha=0.3)
                ymin, ymax = plt.ylim()
                plt.vlines(150, ymin=ymin, ymax=ymax, linestyle='--', color='black')
                plt.title(f'{idn} {d} {site} Channel {ch} prey Approach')
                plt.ylabel(f'{data_title}')
                
                if plot_Zscore == True:
                    plt.xlabel('Time (frames), aligned to movement')
                else:
                    plt.xlabel('Time (frames), aligned to prey laser onset')
                plt.show()
                
        else:
            ZdFoFApproach = np.nan
            ZdFoFApproach_trialOnset = np.nan
        
        
        # IR trials
        if len(IR_approachTrials) > 0:
            IR_ZdFoFApproach = (IR_appTracesInit - np.mean(IR_traces[IR_app_idx,:pre*sr],axis=1).reshape(-1, 1)) / IR_traceDataSD
            IR_ZdFoFApproach_trialOnset = (IR_traces[IR_app_idx] - np.mean(IR_traces[IR_app_idx,:pre*sr],axis=1).reshape(-1, 1)) / IR_traceDataSD
            
            if plot_apr_trace:
                if plot_Zscore:
                    data_set = IR_ZdFoFApproach
                    data_title = 'Z-Score'
                else:
                    data_set = IR_traces[IR_app_idx]*100
                    data_title = '% dFoF'
                plt.figure()
                plt.plot(np.mean(data_set, axis = 0), color = [0.47, 0.67, 0.19])
                plt.fill_between(range(0,data_set.shape[1]),
                                 np.mean(data_set, axis = 0) + np.std(data_set, axis=0) / np.sqrt(len(data_set)),
                                 np.mean(data_set, axis = 0) - np.std(data_set, axis=0) / np.sqrt(len(data_set)),
                                 color = [0.47, 0.67, 0.19], alpha=0.3)
                ymin, ymax = plt.ylim()
                plt.vlines(150, ymin=ymin, ymax=ymax, linestyle='--', color='black')
                plt.title(f'{idn} {d} {site} Channel {ch} IR Approach')
                plt.ylabel(f'{data_title}')
                
                if plot_Zscore == True:
                    plt.xlabel('Time (frames), aligned to movement')
                else:
                    plt.xlabel('Time (frames), aligned to IR laser onset')
                plt.show()
                
        else:
            IR_ZdFoFApproach = np.nan
            IR_ZdFoFApproach_trialOnset = np.nan
           
            
        # AVOID INITIATION and PREYLASER ALIGNED
        if avoidTrials is not np.nan and len(avoidTrials) > 0:
            ZdFoFAvoid = (avdTracesInit - np.mean(traces[avd_idx,:pre*sr],axis=1).reshape(-1, 1)) / traceDataSD
            ZdFoFAvoid_trialOnset = (traces[avd_idx] - np.mean(traces[avd_idx,:pre*sr],axis=1).reshape(-1, 1)) / traceDataSD

        else:
            ZdFoFAvoid = np.nan
            ZdFoFAvoid_trialOnset = np.nan
            
        # and for NR (only PREYLASER ALIGNED)
        ZdFoFNR = ZdFoFNR_yoked = np.nan
        if NRtrials is not np.nan and len(NRtrials) > 0:
            ZdFoFNR = (NRtraces - np.mean(traces[NR_idx,:pre*sr],axis=1).reshape(-1, 1)) / traceDataSD
            
            ZdFoFNR_yoked = (NRtraces_yoked - np.mean(traces[NR_idx,:pre*sr],axis=1).reshape(-1, 1)) / traceDataSD

            
            
# %% for ITI

        if tracesITI is not np.nan and len(tracesITI) > 0:
            tracesITI = np.vstack(tracesITI);

            traceDataSDITI = np.std(tracesITI[:,:pre*sr])
            ZdFoFITI = (tracesITI - np.mean(tracesITI[:,:pre*sr], axis=1).reshape(-1,1))/ traceDataSDITI

            tracesITIGraw = np.vstack(tracesITIGraw)
            traceDataSDG = np.std(tracesITIGraw[:,:pre*sr])
            ITIGdata = (tracesITIGraw - np.mean(tracesITIGraw[:,:pre*sr],axis=1).reshape(-1,1)) /traceDataSDG

            tracesITIIraw = np.vstack(tracesITIIraw)
            traceDataSDI = np.std(tracesITIIraw[:,:pre*sr])
            ITIIdata = (tracesITIIraw - np.mean(tracesITIIraw[:,:pre*sr],axis=1).reshape(-1,1)) /traceDataSDI
            
            
            if plot_ITI_trace:
                plt.figure()
                if plot_Zscore:
                    ITI_plot_data = ZdFoFITI
                    ITI_y_label = 'Z score'
                else:
                    ITI_plot_data = tracesITI * 100
                    ITI_y_label = '% dF/F'

                plt.plot(np.mean(ITI_plot_data, axis = 0), color = [0.6, 0.6, 0.6])
                plt.fill_between(range(0,ITI_plot_data.shape[1]),
                                ( np.mean(ITI_plot_data, axis = 0) + np.std(ITI_plot_data, axis=0) / np.sqrt(len(ITI_plot_data))),
                                 (np.mean(ITI_plot_data, axis = 0) - np.std(ITI_plot_data, axis=0) / np.sqrt(len(ITI_plot_data))),
                                 color = [0.6, 0.6, 0.6], alpha=0.3)
                ymin, ymax = plt.ylim()
                plt.vlines(150, ymin=ymin, ymax=ymax, linestyle='--', color='black')
                plt.title(f'{idn} {d} {site} Channel {ch} ITI')
                plt.ylabel(f'{ITI_y_label}')
                plt.xlabel('Time (frames) aligned to movement')
    
                plt.show()

        else:
            ZdFoFITI = np.nan
            ITIGdata = np.nan
            ITIIdata = np.nan

# %%
        
        # Plotting heatmaps, prey
        if plot_Zscore:
            data_set = ZdFoF
            data_title = 'ZdFoF'
        else:
            data_set = traces * 100
            data_title = '% dF/F'
            
        if plotheatmap:
            plt.figure()
            plt.imshow(data_set, aspect='auto', cmap='viridis', interpolation='none')     
            plt.colorbar()
            plt.axvline(150, linestyle='--', color='white', linewidth=1.5)
            for wee in range(len(initTrace)):
                if pd.isnull(initTrace[wee]) == False:
                    plt.vlines(x = initTrace[wee] + 150, linestyle='--', color='red', linewidth=1.5,
                               ymin = wee - 0.5, ymax = wee + 0.5 )
            plt.xlabel('Time (s), prey laser aligned')
            plt.ylabel('Trial')
            plt.title(f'{idn} {d} {site} Channel {ch} {data_title}')
            
            # Set x-ticks
            plt.xticks(np.arange(0, len(traceTiming), 150), labelsX)
            
            # Get y-tick positions and labels
            yticks = np.arange(data_set.shape[0])  # Tick positions
            ytick_labels = [str(y) for y in yticks]  # Default labels
            
            # Apply y-ticks to the plot
            plt.yticks(yticks, ytick_labels)  
            
            # Get the current axis and apply custom tick colors
            ax = plt.gca()  # Get current axis
            tick_colors = ['green' if y in app_idx else 'red' if y in avd_idx else 'black' for y in yticks]
            for y, color in zip(yticks, tick_colors):
                ax.get_yticklabels()[y].set_color(color)  # Apply color change
            
            # Show plot
            plt.show()
            
        if plotidvtrials and len(approachTrials) > 0:
            aprchColors = ["#2E8B57", "#228B22", "#6B8E23", "#8FBC8F", "#20B2AA", "#556B2F", "#70a7ff", "#fca103"]
            
            plt.figure()
            for clr, trace in enumerate(data_set[app_idx]):
                plt.plot(trace, color = aprchColors[clr])
            plt.axvline(150, linestyle='--', color='black', linewidth=1.5)
            plt.xticks(np.arange(0, len(traceTiming), 150), labelsX)
            plt.ylabel(f"{data_title}")
            plt.xlabel("Time(s): Aligned to prey laser onset")
            plt.title(f'Approach traces for {idn} {d} {site}')
        
            
        # IR
        if len(IR_traces) > 0:
            if plot_Zscore:
                data_set = IR_ZdFoF
                data_title = 'ZdFoF'
            else:
                data_set = IR_traces * 100
                data_title = '% dF/F'
                
            if plotheatmap:
                plt.figure()
                plt.imshow(data_set, aspect='auto', cmap='viridis', interpolation='none')     
                plt.colorbar()
                plt.axvline(150, linestyle='--', color='white', linewidth=1.5)
                for wee in range(len(IR_initTrace)):
                    if pd.isnull(IR_initTrace[wee]) == False:
                        plt.vlines(x = IR_initTrace[wee] + 150, linestyle='--', color='red', linewidth=1.5,
                                   ymin = wee - 0.5, ymax = wee + 0.5 )
                plt.xlabel('Time (s), IR laser aligned')
                plt.ylabel('Trial')
                plt.title(f'{idn} {d} {site} Channel {ch} {data_title}')
                
                # Set x-ticks
                plt.xticks(np.arange(0, len(traceTiming), 150), labelsX)
                
                # Get y-tick positions and labels
                yticks = np.arange(data_set.shape[0])  # Tick positions
                ytick_labels = [str(y) for y in yticks]  # Default labels
                
                # Apply y-ticks to the plot
                plt.yticks(yticks, ytick_labels)  
                
                # Get the current axis and apply custom tick colors
                ax = plt.gca()  # Get current axis
                tick_colors = ['green' if y in IR_app_idx else 'black' for y in yticks]
                for y, color in zip(yticks, tick_colors):
                    ax.get_yticklabels()[y].set_color(color)  # Apply color change
                
                # Show plot
                plt.show()
        
            
        
        if plotidvtrials and len(approachTrials) > 0:
            aprchColors = ["#2E8B57", "#228B22", "#6B8E23", "#8FBC8F", "#20B2AA", "#556B2F", "#70a7ff", "#fca103"]
            
            plt.figure()
            for clr, trace in enumerate(data_set[app_idx]):
                plt.plot(trace, color = aprchColors[clr])
            plt.axvline(150, linestyle='--', color='black', linewidth=1.5)
            plt.xticks(np.arange(0, len(traceTiming), 150), labelsX)
            plt.ylabel(f"{data_title}")
            plt.xlabel("Time(s): Aligned to prey laser onset")
            plt.title(f'Approach traces for {idn} {d} {site}')
            
# %%
        # find peak lag correlations
        lag_correlations_approach = np.nan
        lag_correlations_ITI = np.nan
        lag_correlations_IR = np.nan
        
        if np.size(appTracesInit) > 1:
            if exp == 'nt':
                lag_correlations_approach = np.array([norm_xcorr(appTracesInit[t], approach_fwdSpeed[t]) 
                          for t in range(appTracesInit.shape[0])])
            else:
                lag_correlations_approach = np.array([norm_xcorr(appTracesInit[t], approach_snout_speed[t]) 
                          for t in range(appTracesInit.shape[0])])
            if plot_cros_cor:
                plt.figure()
            
                L = len(lag_correlations_approach[0])
                mid = L // 2
        
                # slice range
                rng = 250        # number of points on each side
                sl = slice(mid - rng, mid + rng + 1)
                
                # full lag vector
                lags = np.arange(-(L//2), L//2 + 1)
                
                # truncated lag vector
                lags_trunc = lags[sl]
                
                for lag_data in lag_correlations_approach:
                    plt.plot(lags_trunc, lag_data[sl])
                
                plt.xlabel('Lag (samples)')
                plt.ylabel('Normalized cross correlation')
                plt.xticks(np.arange(-rng, rng+1, 50))   # adjust tick spacing as needed
                plt.title('Approach trials')
                boxoff()
        

        if exp == 'nt' and np.size(tracesITI) > 1:
            lag_correlations_ITI = np.array([norm_xcorr(tracesITI[t], speedITI[t]) 
                          for t in range(tracesITI.shape[0])])
            
            if plot_cros_cor:
                plt.figure()
            
                L = len(lag_correlations_ITI[0])
                mid = L // 2
        
                # slice range
                rng = 250        # number of points on each side
                sl = slice(mid - rng, mid + rng + 1)
                
                # full lag vector
                lags = np.arange(-(L//2), L//2 + 1)
                
                # truncated lag vector
                lags_trunc = lags[sl]
                
                for lag_data in lag_correlations_ITI:
                    plt.plot(lags_trunc, lag_data[sl])
                
                plt.xlabel('Lag (samples)')
                plt.ylabel('Normalized cross correlation')
                plt.title('ITI')
                plt.xticks(np.arange(-rng, rng+1, 50))   # adjust tick spacing as needed
                boxoff()
                
        if np.size(IR_appTracesInit) > 1:
            lag_correlations_IR = np.array([norm_xcorr(IR_appTracesInit[t], IR_snout_speed[t]) 
                          for t in range(IR_appTracesInit.shape[0])])
            
            if plot_cros_cor:
                plt.figure()
            
                L = len(lag_correlations_IR[0])
                mid = L // 2
        
                # slice range
                rng = 250        # number of points on each side
                sl = slice(mid - rng, mid + rng + 1)
                
                # full lag vector
                lags = np.arange(-(L//2), L//2 + 1)
                
                # truncated lag vector
                lags_trunc = lags[sl]
                
                for lag_data in lag_correlations_IR:
                    plt.plot(lags_trunc, lag_data[sl])
                
                plt.xlabel('Lag (samples)')
                plt.ylabel('Normalized cross correlation')
                plt.title('IR approach trials')
                plt.xticks(np.arange(-rng, rng+1, 50))   # adjust tick spacing as needed
                boxoff()
        
# %%
    #    syllable data
        mean_dFoF_by_syllable = np.nan
        animal_out_of_view_index = list(map(int, r_log['animal hidden frames'][l].split(',')))
        
        left_turn = [5, 9, 10, 11, 13]
        right_turn = [3, 4, 6, 8, 16]
        if len(animal_out_of_view_index) < 3 and l != 13:
            kpms_first_ttl = int(r_log['first prey'][l])
    
            kpms_path = kpms_general_path + idn + '_' + d.replace("_", "") + '.csv'
            kpms_data = pd.read_csv(kpms_path, sep=None, engine="python", encoding='cp1252')
            kpms_data = kpms_data[kpms_first_ttl-animal_out_of_view_index[1]-setUp:]
            kpms_data = kpms_data[:proper_trim]
            
            dFoF_series = dFoF[:proper_trim]
            dFoF_series = pd.Series(dFoF_series, index=kpms_data.index)
            
            # category column
            cat_col = "syllable"
            
            categories = kpms_data[cat_col].values
            frames = kpms_data.index.values  # keeps original frame index
            
            bouts = []
            
            current_cat = categories[0]
            start_i = 0
            
            for i in range(1, len(categories)):
                if categories[i] != current_cat:
                    end_i = i
                    length = end_i - start_i
            
                    bouts.append({
                        "syllable": current_cat,
                        "start_frame": frames[start_i],
                        "end_frame": frames[end_i-1],
                        "length_frames": length
                    })
            
                    # reset
                    current_cat = categories[i]
                    start_i = i
            
            # add last bout
            bouts.append({
                "syllable": current_cat,
                "start_frame": frames[start_i],
                "end_frame": frames[-1],
                "length_frames": len(categories) - start_i
            })
            
            bout_df = pd.DataFrame(bouts)
            
            bout_df["animal"] = idn
            bout_df["date"] = d
            all_bouts.append(bout_df)

            # print(bout_df.head())
            # print("Total bouts:", len(bout_df))
            
            # summary = bout_df.groupby("syllable")["length_frames"].describe()
            # print(summary)
            
            # by frame
            valid_syllables = list(range(0,21)) # should automate this
            mean_dFoF_by_syllable = dFoF_series.groupby(kpms_data['syllable']).mean()  
            mean_dFoF_by_syllable = mean_dFoF_by_syllable[mean_dFoF_by_syllable.index.isin(valid_syllables)]
            
            # # --- Extract mean dFoF values for each group ---
            left_mask = kpms_data["syllable"].isin(left_turn)
            right_mask = kpms_data["syllable"].isin(right_turn)
            
            # Extract dFoF values from ALL frames in each group
            left_turn_mean = dFoF_series[left_mask].mean()
            left_turn_SD = dFoF_series[left_mask].std()

            right_turn_mean = dFoF_series[right_mask].mean()
            right_turn_SD = dFoF_series[right_mask].std()

            plt.figure()

            plt.bar(
                ["Left Turn", "Right Turn"],
                [left_turn_mean, right_turn_mean],
                yerr=[left_turn_SD, right_turn_SD],
                capsize=6
            )
            plt.axhline(y=0, linestyle='--',
                linewidth=1,
                color='black')
            plt.ylabel("Mean dF/F")
            plt.title(f' {idn} {d} {site} Mean dF/F by frame')
            plt.tight_layout()
            boxoff()
            plt.show()

            
            
            SD_dFoF_by_syllable = dFoF_series.groupby(kpms_data['syllable']).std()
            SD_dFoF_by_syllable = SD_dFoF_by_syllable[SD_dFoF_by_syllable.index.isin(valid_syllables)]

            
            syllables = mean_dFoF_by_syllable.index

            
            plt.figure()
            mean_dFoF_by_syllable.plot(kind='bar')
            plt.errorbar(
                syllables,
                mean_dFoF_by_syllable.values,
                yerr=SD_dFoF_by_syllable,
                fmt='none',
                capsize=4,
                color = 'black'
            )
            plt.axhline(y = 2*np.std(dFoF_series), linestyle='--',
                linewidth=1,
                color='black')
            plt.axhline(y = -2*np.std(dFoF_series), linestyle='--',
                linewidth=1,
                color='black')
            plt.xlabel('Syllable')
            plt.ylabel('Mean dF/F')
            plt.title(f' {idn} {d} {site} Mean dF/F by syllable')
            plt.tight_layout()
            plt.show()
            
            # by bout
            syllable = kpms_data['syllable']

            bout_id = (syllable != syllable.shift()).cumsum()
            
            df = pd.DataFrame({
                'syllable': syllable,
                'dFoF': dFoF_series,
                'bout_id': bout_id
            })
            
            bout_lengths = df.groupby('bout_id').size()
            valid_bouts = bout_lengths[bout_lengths >= 5].index
            
            
            
            bout_means = (
                df
                .groupby(['syllable', 'bout_id'])['dFoF']
                .mean()
                .reset_index()
            )
            
            bout_means = bout_means[bout_means['bout_id'].isin(valid_bouts)]
            
            mean_dFoF_per_syllable_bout = (
                bout_means
                .groupby('syllable')['dFoF']
                .mean()
            )
            mean_dFoF_per_syllable_bout = mean_dFoF_per_syllable_bout[mean_dFoF_per_syllable_bout.index.isin(valid_syllables)].reindex(syllables)


            # grouped syllable bouts
            left_bout_vals = mean_dFoF_per_syllable_bout[
                mean_dFoF_per_syllable_bout.index.isin(left_turn)
            ]
            
            right_bout_vals = mean_dFoF_per_syllable_bout[
                mean_dFoF_per_syllable_bout.index.isin(right_turn)
            ]
            
            # plt.figure()

            # plt.bar(
            #     ["Left Turn", "Right Turn"],
            #     [left_bout_vals.mean(), right_bout_vals.mean()],
            #     yerr=[left_bout_vals.sem(), right_bout_vals.sem()],
            #     capsize=6
            # )
            
            # plt.ylabel("Mean dF/F per bout")
            # plt.title("Bout-Based dF/F Comparison: Left vs Right Turn")
            # plt.tight_layout()
            # plt.show()


            # all bouts
            SD_dFoF_per_syllable = (
                bout_means
                .groupby('syllable')['dFoF']
                .std()
            )          
            SD_dFoF_per_syllable = SD_dFoF_per_syllable[SD_dFoF_per_syllable.index.isin(valid_syllables)].reindex(syllables)

            
            n_bouts = bout_means.groupby('syllable').size()
            # syllables = mean_dFoF_per_syllable_bout.index

            plt.figure()
            plt.bar(
                syllables,
                mean_dFoF_per_syllable_bout.values
            )
            plt.errorbar(
                syllables,
                mean_dFoF_per_syllable_bout.values,
                yerr=SD_dFoF_per_syllable,
                fmt='none',
                capsize=4,
                color = 'black'
            )
            plt.axhline(y = 2*np.std(dFoF_series), linestyle='--',
                linewidth=1,
                color='black')
            plt.axhline(y = -2*np.std(dFoF_series), linestyle='--',
                linewidth=1,
                color='black')
            plt.xlabel('Syllable')
            plt.ylabel('Mean dF/F (per bout)')
            plt.title(f' {idn} {d} {site} Mean dF/F per syllable bout')
            plt.tight_layout()
            plt.show()

           
            print(f"dif between proper_trim and kpms_data is {proper_trim-len(kpms_data)} frames") # i think this could be improved by using camera's ttl (first frame when laser appears on last prey trial)
            
        else:
            print('Multiple clippings in file, skipping for now. Write code for this later')

# %%
        # Save data
        sesdat = {
            'session': d,
            'mouse': idn,
            'conversion': sr,
            'lp_normDat': lp_normDatG,
            
            'ZdFoF': ZdFoF,
            # 'ZdFoFinit': ZdFoFinit,
            'ZdFoFITI': ZdFoFITI,
            
            # preylater onset locked
            'ZdFoFApproach_trialOnset': ZdFoFApproach_trialOnset if len(app_idx) > 0  else np.nan,
            'IR_ZdFoFApproach_trialOnset': IR_ZdFoFApproach_trialOnset if len(IR_app_idx) > 0  else np.nan,
            'ZdFoFAvoid_trialOnset': ZdFoFAvoid_trialOnset if len(avd_idx) > 0 else np.nan,
            'ZdFoFNR': ZdFoFNR if len(NR_idx) > 0 else np.nan,
            'ZdFoFNR_yoked': ZdFoFNR_yoked if len(NR_idx) > 0 else np.nan,
            
            # movement locked:
            'ZdFoFApproach': ZdFoFApproach if len(app_idx) > 0  else np.nan,
            'IR_ZdFoFApproach': IR_ZdFoFApproach if len(IR_app_idx) > 0  else np.nan,
            'ZdFoFAvoid': ZdFoFAvoid if len(avd_idx) > 0 else np.nan,
            
            'Gdata': Gdata,
            'Idata': Idata,
            
            # 'InitGdata': InitGdata,
            # 'InitIdata': InitIdata,
            
            'ITIGdata': ITIGdata,
            'ITIIdata': ITIIdata,
            
            'channel': ch,         
            'site': site,
            
            'lag_correlations_approach': lag_correlations_approach,
            'lag_correlations_ITI': lag_correlations_ITI,
            'lag_correlations_IR': lag_correlations_IR,

    
            'speedTrials': speedTrials if ch == 1 else np.nan,
            'speedITI': speedITI if ch == 1 else np.nan,
            'speedTrialsMov': speedTrialsMov if ch == 1 else np.nan,
            
            'mean_dFoF_by_syllable': mean_dFoF_by_syllable
            
            
                
                }

        if exp == 'nt':
            save_file = f"{nt_savePath}{d}{idn} Channel {ch}.pkl"
        else:
            save_file = f"{fm_savePath}{d}{idn} Channel {ch}.pkl"
            
        pd.to_pickle(sesdat, save_file)
# %%
master_bout_df = pd.concat(all_bouts, ignore_index=True)
summary = master_bout_df.groupby("syllable")["length_frames"].describe()
print(summary)
median_lengths = master_bout_df.groupby("syllable")["length_frames"].median()
print(median_lengths)

plt.figure(figsize=(12,6))
master_bout_df.boxplot(column="length_frames", by="syllable")
plt.title("Bout Length Distribution Across All Sessions")
plt.suptitle("")
plt.ylabel("Frames")
plt.show()
session_summary = master_bout_df.groupby(
    ["animal", "date", "syllable"]
)["length_frames"].mean().reset_index()

print(session_summary.head())

# np.mean(fm_drift) # -10.662857142856318
# np.mean(fm_drift_ratio) # -0.00041931890844608486

# col1 = driftTable[:, 0].reshape(-1, 1)  # First column
# col3 = driftTable[:, 2].reshape(-1, 1)  # Third column

# col2 = driftTable[:, 1].reshape(-1, 1)  # Third column
# col4 = driftTable[:, 3].reshape(-1, 1)  # Third column

# # Stack them vertically
# xx = np.vstack((col1, col3))
# yy = np.vstack((col2, col4))


# xx = np.delete(xx, np.argmax(xx))
# yy = np.delete(yy, np.argmax(yy))


# plt.figure()
# plt.scatter(xx, yy, color = 'black')
# plt.plot(np.unique(xx), np.poly1d(np.polyfit(xx, yy, 1))(np.unique(xx)))





# go to combineData. next


# to do later, stamp within laser trials based on action, direction, other behavior
