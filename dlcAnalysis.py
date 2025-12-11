# -*- coding: utf-8 -*-
"""
Created on Wed May 14 12:40:35 2025

@author: conrad

this code corrects fish-eye distortion from dlc data then converts it to real-
world coordinates (adapted from Alexander Heimel's matlab code)

future goal of this code is to get measures from dlc data such as animal location, speed,
angle of head relative to body, tail capture angle?

and possibly automatic behavior detection such as rearing, grooming, jumping,
scratching, gnawing (at object or wall)
"""

def dlcAnalysis(ttlFile, eventTS, r_log, fIdx, driftTable, trialClass, setUp): 
    import numpy as np
    import pandas as pd
    import scipy.io
    import glob
    import matplotlib.pyplot as plt
    
    ttl = pd.read_csv(ttl_file)

    # initialize paramaters that may be needed later
    avoidTrials = np.full(len(ttl), np.nan)
    iti_idx = [] 
    speed_iti = []
    
    # paramters i should return later
    trial_idx_init
    speed_trials
    ttl, 
    mov_trial_idx, 
    app_idx
    speed_trials_mov
    approachTrials 
    initTrace
    
    debug = False
    
    def cart2pol(x, y):
        phi = np.arctan2(y, x)
        rho = np.sqrt(x**2 + y**2)
        return(phi, rho)
    
    def pol2cart(phi, rho):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return(x, y)
    
    
    def nt_change_overhead_to_camera_coordinates(overhead_x,overhead_y,params):
        distort = params['overhead_camera_distortion']
    
        overhead_x = overhead_x - params['overhead_camera_width']/2 + params['overhead_camera_image_offset'][0]
        overhead_y = overhead_y - params['overhead_camera_height']/2 + params['overhead_camera_image_offset'][1]
        distance_neurotar_center_to_camera_mm = distort[0]
        focal_distance_pxl = distort[1]
    
        theta,overhead_r = cart2pol(overhead_x, overhead_y)
    
        if np.any(overhead_r>focal_distance_pxl):
            print('Point outside camera view')
            overhead_r[overhead_r > focal_distance_pxl] = focal_distance_pxl
            camera_x = np.nan
            camera_y = np.nan
            
        camera_r = distance_neurotar_center_to_camera_mm * np.tan(np.arcsin(overhead_r / focal_distance_pxl))
        camera_x, camera_y = pol2cart(theta, camera_r)
        return(camera_x, camera_y)
    
    def nt_change_camera_to_arena_coordinates(camera_x,camera_y,params):
        # invert overhead_center_position
        camera_center_x, camera_center_y = nt_change_overhead_to_camera_coordinates(
            params['overhead_arena_center'][0],
            params['overhead_arena_center'][1],
            params
            )
    
    
        # move center of neurotar to center position in camera coordinates
        camera_x = camera_x - camera_center_x
        camera_y = camera_y - camera_center_y
        
        alpha = -params['overhead_camera_angle']
        rotation = np.array([
            [np.cos(alpha),  np.sin(alpha)],
            [-np.sin(alpha), np.cos(alpha)]
        ])
        p = rotation @ np.array([camera_x, camera_y])
        
        arena_x = p[0, :]
        arena_y = p[1, :]
        
        return arena_x, arena_y
    
    def nt_change_overhead_to_arena_coordinates(overhead_x,overhead_y,params):
        camera_x, camera_y = nt_change_overhead_to_camera_coordinates(overhead_x,overhead_y,params)
        arena_x,arena_y = nt_change_camera_to_arena_coordinates(camera_x,camera_y,params)
        return arena_x, arena_y
    
    def timestamp_to_frame(ts):
        minutes, seconds, frames = map(int, ts.split(':'))
        return (minutes * 60 + seconds) * 30 + frames
        
    
    # W for my pc, Z for surf cloud
    #dlc_filePath = 'W:\\vs03.herseninstituut.knaw.nl\\VS03-CSF-1\\Conrad\\Innate_approach\\Data_analysis\\24.35.01\\DLC'
    dlc_filePath = 'W:\\Conrad\\Innate_approach\\Data_analysis\\24.35.01\\DLC'
    dlc_savePath = 'W:\\Conrad\\Innate_approach\\Data_analysis\\24.35.01\\DLC'
    
    
    fp_filePath = 'W:\\Conrad\\Innate_approach\\Data_collection\\24.35.01\\'
    fp_savePath = 'W:\\Conrad\\Innate_approach\\Data_analysis\\24.35.01\\'
    
    metadata_filepath = 'W:\\Conrad\\Innate_approach\\Data_analysis\\db_FP.mat'
    metadata = scipy.io.loadmat(metadata_filepath, struct_as_record=False, squeeze_me=True)
    db = metadata['db']
    # test = db[38]
    # filter for freelymovingLaser & temporarily just the days that i analyzed for now
    
    
    # r_log = pd.read_csv(f"{fp_filePath}\\recordinglog.csv", sep=";", encoding='cp1252')
    # filter for freelymovingLaser & temporarily just the days that i analyzed for now
    # if debug == False:
    #     r_log = r_log[33:49].reset_index()
        
    # else: 
    #     r_log = r_log[33:34].reset_index()
       
    
    
    # animalIDs = r_log['ID'].unique()
    # dates = r_log['Date'].unique()
    
    # initialize paramters, need to change on a file by file basis
    # maybe load param file for immutable?
    params = {
        'overhead_camera_distortion' : [320, 340],
        'overhead_camera_distortion_method' : 'fisheye_orthographic',
        'overhead_camera_image_offset' : [-4, -4],
        'overhead_camera_width' : 752,
        'overhead_camera_height' : 582,
        'arena_radius_mm' : 175,
        'overhead_arena_center' : [],
        'overhead_camera_angle' : 0,
    
        #not sure if i need these
        'picamera_time_multiplier' : 1.0002,
        'laser_time_multiplier' : 1.0002,
        'arena_shape' : 'circular'
        
        }
    
    # first we need to correct fish-eye distrotion in dlc data and convert to real world coordinates

    record = [entry for entry in db if str(r_log['ID']) in str(entry.subject)
               and str(r_log['Date']).replace('_','-')[:-1] in str(entry.date)][0]
    params['overhead_arena_center'] = record.measures.overhead_arena_center
    
    dlcAnimal = f"{dlc_filePath}\\{str(r_log['ID'])}_{str(r_log['Date']).replace('_', '')}*fmLaserMouseFP*.csv"
    dlcPrey = f"{dlc_filePath}\\{str(r_log['ID'])}_{str(r_log['Date']).replace('_', '')}*prey*.csv"
    dlcIR = f"{dlc_filePath}\\{str(r_log['ID'])}_{str(r_log['Date']).replace('_', '')}*IR*.csv"
    
    df_pathList = [dlcAnimal, dlcPrey, dlcIR]
    df_list = [0, 0, 0]
    

    
    # clean data for merging doesnt really work
    for i in range(len(df_list)):
        
        df_path = glob.glob(df_pathList[i])
        df = pd.read_csv(df_path[0], header =None)
        df = df.iloc[:,1:]     # removes pointless 1st column
        df_list[i] = df 
        
    # make indices
    preyTrial_times = (r_log['prey trial times']).split(',') 
    preyTrial_idx = [timestamp_to_frame(ts) for ts in preyTrial_times]
    preyTrial_idx_filled = []
    for i in range(0, len(preyTrial_idx), 2):
        start, end = preyTrial_idx[i], preyTrial_idx[i+1]
        preyTrial_idx_filled.extend(range(start, end + 1))
    

        
    if len(preyTrial_idx_filled) != len(df_list[1])-3:
        raise ValueError(f"Index length {len(preyTrial_idx_filled)} does not match prey length {len(df_list[1])}")
    
    IRTrial_times = (r_log['IR trial times']).split(',') 
    IRTrial_idx = [timestamp_to_frame(ts) for ts in IRTrial_times]
    IRTrial_idx_filled = []
    for i in range(0, len(IRTrial_idx), 2):
        start, end = IRTrial_idx[i], IRTrial_idx[i+1]
        IRTrial_idx_filled.extend(range(start, end + 1))
        
# =============================================================================
#     plt.plot(np.arange(len(df_list[1].iloc[3:,1].values)), pd.to_numeric(df_list[1].iloc[3:,1].values))
#         
# =============================================================================
    if len(IRTrial_idx_filled) != len(df_list[2])-3:
        raise ValueError(f"Index length {len(IRTrial_idx_filled)} does not match IR length {len(df_list[2])}")
        
        
    if len(preyTrial_idx)%2 == 1 or len(IRTrial_idx)%2 == 1:
        print('trial start or end missing')
        

    
    raw_dlc = np.zeros([len(df_list[0]), 
                       np.shape(df_list[0])[1] + 
                       np.shape(df_list[1])[1] +
                       np.shape(df_list[2])[1]])
    
    # fisheye correct and merge laser position with animal position
    for i in range(len(df_list)):
        raw_dlc = df_list[i]
        
        raw_dlc.columns = pd.MultiIndex.from_arrays([raw_dlc.iloc[1], raw_dlc.iloc[2]])
        raw_dlc = raw_dlc[3:]
        
        if i == 1:
            raw_dlc.index = preyTrial_idx_filled
        elif i == 2:
            raw_dlc.index = IRTrial_idx_filled
        
        # raw_dlc = raw_dlc[3:].reset_index(drop = True)
        
        # passes data through distortion correction
        #cor_dlc = pd.DataFrame(np.zeros(raw_dlc.shape))
        
        # Initialize a new DataFrame to hold transformed values
        cor_dlc = pd.DataFrame(index=raw_dlc.index)
    
        for bodypart in raw_dlc.columns.get_level_values(1).unique():
        
            try:
                # Get overhead coordinates
                overhead_x = pd.to_numeric(raw_dlc[bodypart]['x'], errors='coerce').values
                overhead_y = pd.to_numeric(raw_dlc[bodypart]['y'], errors='coerce').values
                
                
        
                # Transform coordinates
                arena_x, arena_y = nt_change_overhead_to_arena_coordinates(overhead_x, overhead_y, params)
                
                
                # insert interpolation and filtering here?
                # probably should plot body part data
        
                # Add transformed coordinates
                cor_dlc[(bodypart, 'x')] = arena_x
                cor_dlc[(bodypart, 'y')] = arena_y
        
                cor_dlc[(bodypart, 'likelihood')] = raw_dlc[bodypart]['likelihood'].values
        
            except KeyError:
                print(f"Skipping {bodypart}: missing expected columns")
                
        df_list[i] = cor_dlc
                
    # calculate snout to laser distance (s2l) for prey
    # TO DO: angle of head to laser, and angle of head to body
    snout_displacement = (np.array([df_list[0][('snout','x')][:-1], 
                          df_list[0][('snout','y')]][:-1]) - 
                 np.array([df_list[0][('snout','x')][1:], 
                          df_list[0][('snout','y')][1:]]))
    
    snout_distance = np.sqrt(snout_displacement[0]**2 + 
                             snout_displacement[1]**2)
    frame_to_sec = 1/sr
    snout_speed = snout_distance/frame_to_sec/1000 # speed is m/s
    # plt.plot(snout_speed[3000:])
    
    dx = df_list[1][('preyLaser','x')] - df_list[0][('snout','x')]
    dy = df_list[1][('preyLaser','y')] - df_list[0][('snout','y')]
    
    s2prey_distance = np.sqrt(dx**2 + dy**2)
    s2prey_distance[np.isnan(dx) | np.isnan(dy)] = np.nan
    
    d_prey = s2prey_distance.reset_index(drop = True)
    
    # and for IR...
    dx = df_list[2][('IR','x')] - df_list[0][('snout','x')]
    dy = df_list[2][('IR','y')] - df_list[0][('snout','y')]
    
    s2IR_distance = np.sqrt(dx**2 + dy**2)
    s2IR_distance[np.isnan(dx) | np.isnan(dy)] = np.nan
    
    # Assume s2prey_distance is a 1D numpy array of distances
    d_IR = s2IR_distance.reset_index(drop = True)
    
    distances_trials = [d_prey, d_IR]
    
    for t in range(2):
        distance = distances_trials[t]
        
        distance = np.array(distance) # need to convert for boolean mask in next step
        valid = ~np.isnan(distance)
        
        # Detect where trials start and end
        trial_starts = np.where(np.diff(valid.astype(int)) == 1)[0] + 1
        trial_ends   = np.where(np.diff(valid.astype(int)) == -1)[0] + 1
        
        # Handle case where data starts or ends with a trial
        if valid[0]:
            trial_starts = np.r_[0, trial_starts]
        if valid[-1]:
            trial_ends = np.r_[trial_ends, len(distance)]
        
        # Plot
        plt.figure(figsize=(12, 6))
        for i, (start, end) in enumerate(zip(trial_starts, trial_ends), 1):
            if end-start > 600: # in case of IR trials lasting longer than 20 seconds
                end = start + 600
            trial_dist = distance[start:end]
            trial_time = np.arange(len(trial_dist))  # trial-relative time
            plt.plot(trial_time, trial_dist, label=f"Trial {i}")
            plt.plot(len(trial_time), trial_dist[-1], marker = 'D', color = 'k')
        
        plt.xlabel("Time (frames)")
        plt.ylabel("Distance (mm)")
        if t == 0:
            plt.title("Snout to Prey Laser Distance per Trial\n  " +
                      r_log['Date'] + str(r_log['ID']))
        else: 
            plt.title("Snout to IR Laser Distance per Trial\n  " + 
                      r_log['Date'] + str(r_log['ID']))

        plt.legend()
        plt.show()
        
        plt.figure(figsize=(12, 6))
        for i, (start, end) in enumerate(zip(trial_starts, trial_ends), 1):
            if end-start > 600: # in case of IR trials lasting longer than 20 seconds
                end = start + 600
            trial_speed = snout_speed[start:end] # may be off by 1 frame
            trial_time = np.arange(len(trial_speed))  # trial-relative time
            plt.plot(trial_time, trial_speed, label=f"Trial {i}")
            plt.plot(len(trial_time), trial_speed[-1], marker = 'D', color = 'k')
        
        plt.xlabel("Time (frames)")
        plt.ylabel("Speed (m/s)")
        
        if t == 0:
            plt.title("Snout Speed per Trial\n  " +
                      r_log['Date'] + str(r_log['ID']))
        else: 
            plt.title("Snout Speed per Trial\n  " + 
                      r_log['Date'] + str(r_log['ID']))

        plt.legend()
        plt.show()
    
    
    
    df_list.append(d_prey)
    df_list.append(d_IR)
    
    # app_idx = np.where(~np.isnan(approachTrials))[0]
    avd_idx = np.where(~np.isnan(avoidTrials))[0]

    dlcDat = {
        'session': r_log['Date'],
        'mouse': r_log['ID'],    
        
        'data': df_list
        
        # 'site': site,
            }


    save_file = f"{dlc_savePath}{r_log['Date']}{r_log['ID']} DLC.pkl"
    pd.to_pickle(dlcDat, save_file)
    
    return (iti_idx, speed_iti, trial_idx_init, speed_trials, ttl, mov_trial_idx, app_idx, avd_idx, speed_trials_mov, approachTrials, avoidTrials, initTrace)