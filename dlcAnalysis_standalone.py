# -*- coding: utf-8 -*-
"""
Created on Wed May 14 12:40:35 2025

@author: conrad

this code corrects fish-eye distortion from dlc data then converts it to real-
world coordinates (adapted from Alexander Heimel's matlab code)
                   
data is then filtered out if there dlc gives low probability, if a point jumps 
to physiological improbable speeds, or if point jumps out of arena. 

future goal of this code is to get measures from dlc data such as animal location, speed,
angle of head relative to body, tail capture angle?

and possibly automatic behavior detection such as rearing, grooming, jumping,
scratching, gnawing (at object or wall)
"""
import numpy as np
import pandas as pd
import scipy.io
import glob
import matplotlib.pyplot as plt
# from scipy.interpolate import interp1d
# from scipy.interpolate import CubicSpline
from scipy.interpolate import PchipInterpolator as pchip
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
from boxoff import boxoff


debug = False
plot_need = True
sr = 30
stim_dur = 20
setUp = 120

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

def find_distance(item1, item2):
    dx = item1.iloc[:,0]-item2.iloc[:,0]
    dy = item1.iloc[:,1]-item2.iloc[:,1]
    distance = np.sqrt(dx**2 + dy**2)
    distance[np.isnan(dx) | np.isnan(dy)] = np.nan
    distance = distance.reset_index(drop = True)
    return distance

def my_heatplot(data, title, bins=100):
    colors = [(0, 0, 1), (0, 1, 1), (0, 1, 0.75), (0, 1, 0), (0.75, 1, 0),
              (1, 1, 0), (1, 0.8, 0), (1, 0.7, 0), (1, 0, 0)]
    cm = LinearSegmentedColormap.from_list('sample', colors)
    x = data.iloc[:, 0]
    y = data.iloc[:, 1]

    # Bin coordinates into a 2D histogram
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)

    plt.imshow(
        heatmap.T, origin='lower',
        cmap=cm,
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect='auto'
    )
    plt.colorbar(label="Counts")
    plt.title(title)
    plt.xlabel("X position")
    plt.ylabel("Y position")
    boxoff()
    plt.show()
    
def my_trajectory(data, title):
    x = data.iloc[:, 0]
    y = data.iloc[:, 1]
    
    fig, ax = plt.subplots()

    my_circle = Circle(xy=(0, 0), radius=175, facecolor='none', edgecolor='black')
    
    # Add the circle to the axes
    ax.add_patch(my_circle)
    plt.plot(x, y, color="blue", alpha=0.6)
    # plt.xlim((0, 582))
    # plt.ylim((0,752))
    
    # Ensure the aspect ratio is equal so it looks like a circle
    ax.set_aspect('equal', adjustable='box')
    plt.title(title)
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.axis("equal")  # keep aspect ratio    boxoff()
    boxoff()
    plt.show()

# def trial_index_finder(ttt_syncpi, ttl_from_camera, ttl_check):
#     anchor_point = np.where(abs(ttl_check) == np.min(abs(ttl_check)))[0][0]
    
    
    

# W for my pc, Z for surf cloud
#dlc_filePath = 'W:\\vs03.herseninstituut.knaw.nl\\VS03-CSF-1\\Conrad\\Innate_approach\\Data_analysis\\24.35.01\\DLC'
dlc_filePath = 'W:\\Conrad\\Innate_approach\\Data_analysis\\24.35.01\\DLC'
dlc_savePath = 'W:\\Conrad\\Innate_approach\\Data_analysis\\24.35.01\\DLC'


fp_filePath = 'W:\\Conrad\\Innate_approach\\Data_collection\\24.35.01\\'
fp_savePath = 'W:\\Conrad\\Innate_approach\\Data_analysis\\24.35.01\\'

ntFilePth = 'W:\\Conrad\\Innate_approach\\Data_collection\\Neurotar\\'

metadata_filepath = 'W:\\Conrad\\Innate_approach\\Data_analysis\\db_FP.mat'
metadata = scipy.io.loadmat(metadata_filepath, struct_as_record=False, squeeze_me=True)
db = metadata['db']
# test = db[38]
# filter for freelymovingLaser & temporarily just the days that i analyzed for now


ntFilePth = 'W:\\Conrad\\Innate_approach\\Data_collection\\Neurotar\\'


r_log = pd.read_csv(f"{fp_filePath}\\recordinglog.csv", sep=None, engine="python", encoding='cp1252')
# filter for freelymovingLaser & temporarily just the days that i analyzed for now
if debug == False:
    r_log = r_log[r_log['Exp']=='fm laser'].reset_index()
    r_log = r_log[:-1] #temp filter for bad recording session
else: 
    r_log = r_log[35:36].reset_index()
   


animalIDs = r_log['ID'].unique()
dates = r_log['Date'].unique()

combined_snout_distance = []
ttl_pi_drift = []

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
for l in range(len(r_log)):
    
 
    record = [entry for entry in db if str(r_log['ID'][l]) in str(entry.subject)
               and str(r_log['Date'][l]).replace('_','-')[:-1] in str(entry.date)][0]
    params['overhead_arena_center'] = record.measures.overhead_arena_center
    
    animal_id = str(r_log['ID'][l])
    date = str(r_log['Date'][l]).replace('_', '')
    
    #accidentally indexed wrong on second batch, this fixes it
    if int(date) >= 20250512:
        index_correction_factor = 0
    else:
        index_correction_factor = 1

    print(f"________________________________________________\nCurrent run: {date}{animal_id}\n------------------------------------------------")
    
    dlcAnimal = f"{dlc_filePath}\\{str(r_log['ID'][l])}_{str(r_log['Date'][l]).replace('_', '')}*fmLaserMouseFP*.csv"
    dlcPrey = f"{dlc_filePath}\\{str(r_log['ID'][l])}_{str(r_log['Date'][l]).replace('_', '')}*prey*.csv"
    dlcIR = f"{dlc_filePath}\\{str(r_log['ID'][l])}_{str(r_log['Date'][l]).replace('_', '')}*IR*.csv"
    
    ttlFile = f"{ntFilePth}{str(r_log['ID'][l])}_{str(r_log['Date'][l]).replace('_', '')}_01_ttl"
    ttl = pd.read_csv(ttlFile)
    
    if ttl['Event'][0] == 'Received trigger':
        
        #subtracts neurotar ttl input from first laser ttl output
        ttl['Time'] = (pd.to_datetime(ttl['DateTime']) - pd.to_datetime(ttl['DateTime'][0])).dt.total_seconds() + (ttl['Milliseconds'] / 1000) - ttl['Milliseconds'][0] / 1000
        
        ttl = ttl.loc[ttl['Event'] !='Received trigger'].reset_index(drop = True) # added because received trigger sometimes added twice
        ttl = ttl.loc[ttl['Event'] !='sync'].reset_index(drop = True)
        ttl['Time'] = ttl['Time'] - stim_dur/sr
    
    elif ttl['Event'][0] == 'sync':
        ttl['Time'] = (pd.to_datetime(ttl['DateTime']) - pd.to_datetime(ttl['DateTime'][0])).dt.total_seconds() + (ttl['Milliseconds'] / 1000) - ttl['Milliseconds'][0] / 1000
        # ttl = ttl.loc[ttl['Event'] !='Received trigger'].reset_index(drop = True) # added because received trigger sometimes added twice
        ttl = ttl.loc[ttl['Event'] !='sync'].reset_index(drop = True)
        ttl['Time'] = ttl['Time'] - stim_dur/sr
    else:
        print('No sync pulse detected at start, something has gone wrong')
        
    fIdx = np.array([])    # code for removing failed laser presentations
    if isinstance(r_log['Failed'][l], int):
        fIdx = np.array(int(r_log['Failed'][l]))
        # eventTS = eventTS.delete(np.array(int(r_log['Failed'][l])))
    elif isinstance(r_log['Failed'][l], float) and  np.isnan(r_log['Failed'][l]) == False:
        fIdx = np.array(list(map(int, str(r_log['Failed'][l]).split('.'))))
        if fIdx[1] == 0:
            fIdx = np.delete(fIdx, 1)
        # eventTS = eventTS.delete(fIdx)
        
    elif isinstance(r_log['Failed'][l], str) and len(r_log['Failed'][l]) == 1:
        fIdx = np.array(int(r_log['Failed'][l]))

    if fIdx.size == 1:
        expanded_fIdx = [fIdx*3 +offset for offset in range(3)]
    else: 
        expanded_fIdx = [i + offset for i in fIdx*3 for offset in range(3)]
    
    ttl = ttl.drop(expanded_fIdx, axis = 0).reset_index(drop = True)    
        
    ttl = ttl['Time'][::3]
            
    

    cam_ttl_file = f"{fp_filePath}{animal_id}\\{animal_id}_{date}_001\\{animal_id}_{date}_001_pioverhead_triggers.csv"
    start_frame = pd.read_csv(cam_ttl_file).index[1]
    
    df_pathList = [dlcAnimal, dlcPrey, dlcIR]
    df_list = [0, 0, 0]
    

    
    # clean data for merging doesnt really work
    for i in range(len(df_list)):
        
        df_path = glob.glob(df_pathList[i])
        df = pd.read_csv(df_path[0], header =None, low_memory=False)
        df = df.iloc[:,1:]     # removes pointless 1st column
        df_list[i] = df 
        
    # make indices
    # y = pd.to_numeric(df_list[1][1][3:], errors='coerce')  # safely convert to float
    # plt.figure()
    # plt.plot(y.index, y.values)
    # plt.show()    
    
    preyTrial_times = (r_log['prey trial times'][l]).split(',') 
    preyTrial_idx = [timestamp_to_frame(ts) for ts in preyTrial_times]
    preyTrial_idx_filled = []
    for i in range(0, len(preyTrial_idx), 2):
        start, end = preyTrial_idx[i], preyTrial_idx[i+1]
        preyTrial_idx_filled.extend(range(start, end + index_correction_factor))
    

        
    if len(preyTrial_idx_filled) != len(df_list[1])-3:
        # raise ValueError(f"Index length {len(preyTrial_idx_filled)} does not match prey length {len(df_list[1])}")
        print(f"Index length {len(preyTrial_idx_filled)} does not match prey length {len(df_list[1])-3}")
        print('Either DLC dropped frames (rerun) or you need to check video times in recording log')
        continue
    
    IRTrial_times = (r_log['IR trial times'][l]).split(',') 
    IRTrial_idx = [timestamp_to_frame(ts) for ts in IRTrial_times]
    IRTrial_idx_filled = []
    for i in range(0, len(IRTrial_idx), 2):
        start, end = IRTrial_idx[i], IRTrial_idx[i+1]
        IRTrial_idx_filled.extend(range(start, end + index_correction_factor))
        

    if len(IRTrial_idx_filled) != len(df_list[2])-3:
        # raise ValueError(f"Index length {len(IRTrial_idx_filled)} does not match IR length {len(df_list[2])}")
        print(f"Index length {len(IRTrial_idx_filled)} does not match IR length {len(df_list[2])-3}")
        print('Either DLC dropped frames (rerun) or you need to check video times in recording log')
        continue
        
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
        raw_dlc = raw_dlc[3:].reset_index(drop = True) # clips nonnumeric info 

        # data clipping:
        if i == 0: 
            raw_dlc = raw_dlc[start_frame:].reset_index(drop = True) # clips to nt start pulse
            trim_start = preyTrial_idx_filled[0] - start_frame - setUp*30
            # trim_start = 0
            trim_end = len(raw_dlc)-5*sr
            raw_dlc = raw_dlc[trim_start:trim_end].reset_index(drop = True)
        
        
        if i == 1:
            trim_factor = preyTrial_idx_filled[0] - 120*sr
            raw_dlc.index = [x - trim_factor for x in preyTrial_idx_filled]
        elif i == 2:
            raw_dlc.index = [x - trim_factor for x in IRTrial_idx_filled]
            
            
        
        # raw_dlc = raw_dlc[3:].reset_index(drop = True)
        
        # passes data through distortion correction
        #cor_dlc = pd.DataFrame(np.zeros(raw_dlc.shape))
        
        # Initialize a new DataFrame to hold transformed values
        cor_dlc = pd.DataFrame(index=raw_dlc.index)
    
        for bodypart in raw_dlc.columns.get_level_values(1).unique():
            if bodypart in ['pawFL', 'pawBL', 'pawFR', 'pawBR']: # these body parts arent often visible overhead, not good data
                continue
            
            try:
                # Get overhead coordinates
                overhead_x = pd.to_numeric(raw_dlc[bodypart]['x'], errors='coerce').values
                overhead_y = pd.to_numeric(raw_dlc[bodypart]['y'], errors='coerce').values
        
                # Transform coordinates
                arena_x, arena_y = nt_change_overhead_to_arena_coordinates(overhead_x, overhead_y, params)
               
                
                # probably should plot body part data   

                
                # filter out low likelihood
                
                
                ###
                # i need to add something that detects long periods of low likelihood and flags it, eg for periods where i take animal out to fix cables
                ###
                
                low_prob = np.array(raw_dlc[bodypart]['likelihood'].values, dtype=float) < 0.5
                arena_x[low_prob] = np.nan
                arena_y[low_prob] = np.nan
                
                # num_nans_added = np.sum(low_prob)
                # print(f"Number of low probs: {num_nans_added}")
                
                
                
                # filter out out of arena, with some leeway (3cm)
                dist_from_center = np.sqrt(arena_x**2 + 
                                           arena_y**2)
                
                outside_circle = dist_from_center > params['arena_radius_mm'] + 30
                
                # num_nans_added = np.sum(outside_circle)
                # print(f"Number of out of arena: {num_nans_added}")
                
                arena_x[outside_circle] = np.nan
                arena_y[outside_circle] = np.nan
                
                # filter out jumps 
                dx = np.diff(arena_x)
                dy = np.diff(arena_y)
                distances = np.sqrt(dx**2 + dy**2)
                bad_steps = distances > 70 # roughly equates to 2 meters per second
                mask = np.zeros_like(arena_x, dtype=bool)
                mask[1:] = bad_steps  # offset by one since diff reduces length by 1
                
                # num_nans_added = np.sum(mask)
                # print(f"Number of jumps: {num_nans_added}")
                
                # Replace both x and y values with NaN where the jump is too large
                arena_x[mask] = np.nan
                arena_y[mask] = np.nan
                
                # plt.figure()
                # plt.plot(arena_y)
                # plt.title('y before interpolation')
                
                # if snout, some restriction on space relative to head ring/ ears?
                
                # interpolate
                
                #####
                # should change interpolation code for laser trials
                #####
                
                ts= np.arange(len(arena_x))
                mask = ~np.isnan(arena_x)
                # arena_x_func = interp1d(ts[mask], arena_x[mask], kind= 'slinear', fill_value="extrapolate")
                arena_x_func = pchip(ts[mask], arena_x[mask], extrapolate = True)
                arena_x = arena_x_func(ts)
          
                # arena_y_func = interp1d(ts[mask], arena_y[mask], kind= 'slinear', fill_value="extrapolate")
                arena_y_func = pchip(ts[mask], arena_y[mask], extrapolate = True)
                arena_y = arena_y_func(ts)


                # plt.figure()
                # plt.plot(arena_y)
                # plt.title('y after interpolation: ' + bodypart)

                # Add transformed and filtered coordinates
                cor_dlc[(bodypart, 'x')] = arena_x
                cor_dlc[(bodypart, 'y')] = arena_y
                # cor_dlc[(bodypart, 'likelihood')] = raw_dlc[bodypart]['likelihood'].values
                
        
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
    
    
    # snout to prey laser distance
    snout_pos =  df_list[0][[('snout','x'), ('snout','y')]]
    preyLaser_pos = df_list[1][[('preyLaser','x'), ('preyLaser','y')]]
    d_prey = find_distance(snout_pos, preyLaser_pos)
    
    # and for IR...
    IRLaser_pos = df_list[2][[('IR','x'), ('IR','y')]]
    d_IR = find_distance(snout_pos, IRLaser_pos)

    distances_trials = [d_prey, d_IR]  

    
    # calculate angle from head to snout 
    # create triangle
    
    hrL_pos = df_list[0][[('hrL','x'), ('hrL','y')]]
    hrR_pos = df_list[0][[('hrR','x'), ('hrR','y')]]
    
    earL_pos = df_list[0][[('earL','x'), ('earL','y')]]
    earR_pos = df_list[0][[('earR','x'), ('earR','y')]]

    hr_midpoint = pd.DataFrame((hrL_pos.values + hrR_pos.values) / 2)

    
    # my_heatplot(snout_pos.iloc[30:, :], 'snout_pos')
    # my_heatplot(hrL_pos.iloc[30:, :], 'hrL position')
    # my_trajectory(hr_midpoint.iloc[30:, :], 'midpoint position')
    my_trajectory(snout_pos.iloc[30:, :], f' {date} {animal_id} snout position')

    
        
    snout2left_distance = find_distance(hrL_pos, snout_pos)
    snout2right_distance = find_distance(hrR_pos, snout_pos)
    hr_dis = find_distance(hrL_pos, hrR_pos)
    ear_dis = find_distance(earL_pos, earR_pos)

    if (snout2left_distance > 30).any() or (snout2right_distance > 30).any():
        print("snout distance too far, you need to correct it post interpolation")
        
    # create vector that passes through snout and mid point between ears
    
    
    # calculate angle between snout vector and vector to laser
    
    for t in range(2):
        distance = distances_trials[t]
        
        distance = np.array(distance) # need to convert for boolean mask in next step
        valid = ~np.isnan(distance)
        
        # Detect where trials start and end
        trial_starts = np.where(np.diff(valid.astype(int)) == 1)[0] + 1
        trial_ends   = np.where(np.diff(valid.astype(int)) == -1)[0] + 1
        
     
        
        if t == 0:
            # check how off manually elected trial times are off from recorded tll pulses
            predicted = ttl.reset_index(drop = True)*30
            
            ttl_file_diff = np.array(np.diff(ttl)*30, dtype = int)
            ttl_select_diff = np.diff(trial_starts)
            ttl_check = ttl_file_diff - ttl_select_diff
            
            ttl_pi_drift.append(ttl_check[-1]-ttl_check[0]) # no drift between sync pi and camera :)
            
            anchor_point = np.where(abs(ttl_check) == np.min(abs(ttl_check)))[0][0] # this is where prey laser has same start time for both cam and sync pi systems, and should be used later for syncing rwd data
            print(f'Anchor point is {np.min(abs(ttl_check))} frames off')
            for index, value in enumerate(ttl_check):
                # ttl_correction.append(value)
                if abs(value) > 3:
                    print(f"'mismatch of {value} frames detected! Trial {index} ")
                    
            
            # collect snout2prey laser distances per trial
            for i, (start, end) in enumerate(zip(trial_starts, trial_ends), 1):
                if end-start < 500: # all approach trials, edit this later
                   combined_snout_distance.append(distance[start:end])
                  
        
        # Handle case where data starts or ends with a trial
        if valid[0]:
            trial_starts = np.r_[0, trial_starts]
        if valid[-1]:
            trial_ends = np.r_[trial_ends, len(distance)]
        
        # Plot
        if plot_need == True:
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
                          r_log['Date'][l] + str(r_log['ID'][l]))
            else: 
                plt.title("Snout to IR Laser Distance per Trial\n  " + 
                          r_log['Date'][l] + str(r_log['ID'][l]))
    
            plt.legend()
            boxoff()
            plt.show()
    
    
            # snout speed
            # plt.figure(figsize=(12, 6))
            # for i, (start, end) in enumerate(zip(trial_starts, trial_ends), 1):
            #     if end-start > 600: # in case of IR trials lasting longer than 20 seconds
            #         end = start + 600
            #     trial_speed = snout_speed[start:end] # may be off by 1 frame
            #     trial_time = np.arange(len(trial_speed))  # trial-relative time
            #     plt.plot(trial_time, trial_speed, label=f"Trial {i}")
            #     plt.plot(len(trial_time), trial_speed[-1], marker = 'D', color = 'k')
            
            # plt.xlabel("Time (frames)")
            # plt.ylabel("Speed (m/s)")
            # plt.legend()
            # if t == 0:
            #     plt.title("Snout Speed per Prey Trial\n  " +
            #               r_log['Date'][l] + str(r_log['ID'][l]))
            # else: 
            #     plt.title("Snout Speed per IR Trial\n  " + 
            #               r_log['Date'][l] + str(r_log['ID'][l]))
    
            
            # plt.show()
            
            
    df_list.append(d_prey)
    df_list.append(d_IR)
    df_list.append(snout_speed)

    dlcDat = {
        'session': r_log['Date'][l],
        'mouse': r_log['ID'][l],    
        
        'data': df_list
        
        # 'site': site,
            }


    save_file = f"{dlc_savePath}\\{r_log['Date'][l]}{r_log['ID'][l]} DLC.pkl"
    pd.to_pickle(dlcDat, save_file)
    
    

# comb_save_file = f"{dlc_savePath}\\snout_distance_combined_DLC.pkl"
# pd.to_pickle(combined_snout_distance, comb_save_file)