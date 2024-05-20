import pandas as pd
import numpy as np
import os
import json
from operator import itemgetter
from datetime import datetime, timezone
from app.dataIO import get_obd_data 
import requests
import time
import matplotlib.pyplot as plt


def active_regeneration_shift(OBD_data, active_regeneration_start_time):
        # extracting the relevant parameter-ID
        Time = []
        soot_load_Value = []
        for pac_idx in range(len(OBD_data)):
                if "pids" in OBD_data[pac_idx]:
                    if len(OBD_data[pac_idx]['pids'])>0:
                        for sub_pid_cnt in range(0,len(OBD_data[pac_idx]['pids'])):
                                State = OBD_data[pac_idx]['pids'][sub_pid_cnt]
                                # extracting Soot-Load
                                if 'spn_5466_avg' in State:
                                        Time.append(State['spn_5466_avg']['timestamp'])
                                        soot_load_Value.append(State['spn_5466_avg']['value'][0])
        
        # if soot_load values not available
        if len(soot_load_Value)==0:
               return active_regeneration_start_time
        
        # if soot load values are available --> find the point where maximum negative slope present
        slope = np.array(soot_load_Value[1:-1]) - np.array(soot_load_Value[0:-2])
        min_index = np.argmin(slope)

        # if the point of maximum drop is greater than 10 minutes
        # search for the maximum drop wthin 10 minutes prior to active-regeneration time
        if (Time[min_index-1] - active_regeneration_start_time)>(10*60*1000):
               active_regen_idx = (np.abs(np.asarray(Time) - active_regeneration_start_time)).argmin()
               idx_10_mins_before = (np.abs(np.asarray(Time) - (active_regeneration_start_time - 10*60*1000))).argmin()
               min_index = np.argmin(slope[idx_10_mins_before: active_regen_idx])   
             
 

        return Time[min_index-1]
                                      


# function to be called for generation regeneration evidence
def regeneration_evidence(COUNTRY_FLAG, OBD_data,
                            active_regeneration_start_time, active_regeneration_end_time, 
                            burn_quality_percentage):
    
    
                
        # getting the active-regeneration duration in minutes
        active_regeneration_duration = (active_regeneration_end_time - active_regeneration_start_time)//(60*1000)
        
        # corner case
        # if active_regeneration_start_time >= active_regeneration_end_time 
        # active_regeneration_duration happens to be less that 10 minutes
        if (active_regeneration_start_time >= active_regeneration_end_time) or (active_regeneration_duration <= 10):
                # check the status of burn_quality
                # if burn_quality == high  --> Set the speed status sufficient (1)
                # else --> Set the speed status insufficient (0)
                if burn_quality_percentage>=0.6 and burn_quality_percentage!=2:
                        speed_status = 1
                else:
                        speed_status = 0        
                return speed_status, burn_quality_percentage

        
        
        # if FLAG is set to "US" use the following RPM and Speed thresholds
        if COUNTRY_FLAG == 'US':
                RPM_RANGE = [800, 2000]
                SPEED_THRESHOLD = 80

        # if FLAG is set to "IN" use the following RPM and Speed thresholds
        elif COUNTRY_FLAG == 'IN':
                RPM_RANGE = [1000, 2000]
                SPEED_THRESHOLD = 40
 

        # extracting the relevant parameter-ID
        Time = []
        dp_inst_Value = []
        rpm_inst_Value = []
        speed_inst_Value =[]
        for pac_idx in range(len(OBD_data)):
                if "pids" in OBD_data[pac_idx]:
                    if len(OBD_data[pac_idx]['pids'])>0:
                        for sub_pid_cnt in range(0,len(OBD_data[pac_idx]['pids'])):
                                State = OBD_data[pac_idx]['pids'][sub_pid_cnt]
                                # extracting DPFDP
                                if 'spn_3251_avg' in State:
                                        Time.append(State['spn_3251_avg']['timestamp'])
                                        dp_inst_Value.append(State['spn_3251_avg']['value'][0])
                                # extracting RPM        
                                if 'spn_190_avg' in State:
                                        rpm_inst_Value.append(State['spn_190_avg']['value'][0])
                                # extracting SPEED       
                                if 'spn_84_avg' in State:
                                        speed_inst_Value.append(State['spn_84_avg']['value'][0])    

                                             
                                
        # applying the RPM constraint
        # getting the correspnding values of different varaibles after applying constraints
        # Time1 constitutes of timestamps where the RPM conditions are met
        dp_rpm_constrained = []
        speed_rpm_constrained = []
        Time1 = []
        rpm_constrained = []

        # iterating through each timestamp to make the RPM constraint check
        for i in range(len(Time)):
            if rpm_inst_Value[i]>=RPM_RANGE[0] and rpm_inst_Value[i]<=RPM_RANGE[-1]:
                    dp_rpm_constrained.append(dp_inst_Value[i])
                    Time1.append(Time[i])
                    rpm_constrained.append(rpm_inst_Value[i])
                    speed_rpm_constrained.append(speed_inst_Value[i])
        
        # after applying the rpm-constraint check which are the nearest active regeneration start and end indices
        nearest_ar_start_idx = (np.abs(np.array(Time1) -  active_regeneration_start_time)).argmin()
        nearest_ar_end_idx = (np.abs(np.array(Time1) -  active_regeneration_end_time)).argmin()
        
        # if the nearest start and end indices comes out to be same (which implies after applying the rpm-constraint we are 
        # not left with any valid point)
        if (nearest_ar_start_idx == nearest_ar_end_idx):
                if burn_quality_percentage>=0.6 and burn_quality_percentage!=2:
                        speed_status = 1
                else:
                        speed_status = 0
                return speed_status, burn_quality_percentage
                


        # calculating the mid-region of the active regeneration zone
        mid_idx = (nearest_ar_start_idx + nearest_ar_end_idx)//2

        # bringing both the pre and post window to same sizes
        pre_start_idx = 0
        post_end_idx = len(Time1)
        # if length of the pre-regeneration window > length of the post-regeneration window
        # bring the pre-window to same size as the post window
        if (mid_idx) > (len(Time1)- nearest_ar_end_idx):
                pre_start_idx = mid_idx - (len(Time1)- nearest_ar_end_idx)
        # if length of the pre-regeneration window < length of the post-regeneration window  
        # bring the post-window to same size as the pre window
        elif (mid_idx) < (len(Time1)- nearest_ar_end_idx):
                post_end_idx = nearest_ar_end_idx + mid_idx 
        
        # fixing the pre_dp, pre_rpm and the pre_speed window
        pre_dp_window = dp_rpm_constrained[pre_start_idx:mid_idx]
        pre_rpm_window = rpm_constrained[pre_start_idx:mid_idx]
        pre_speed_window = speed_rpm_constrained[pre_start_idx:mid_idx]

        # fixing the post_dp, post_rpm and the post_speed window
        post_dp_window = dp_rpm_constrained[nearest_ar_end_idx:post_end_idx]
        post_rpm_window = rpm_constrained[nearest_ar_end_idx:post_end_idx]
        post_speed_window = speed_rpm_constrained[nearest_ar_end_idx:post_end_idx]
        
        # quantifying the burn_quality basis the burn_quality_percentage
        #   0 <= burn_quality_percentage < 0.33 --> low burn_quality
        if (burn_quality_percentage>=0 and burn_quality_percentage<0.33):
                burn_quality = 'low'
        #   0.33 <= burn_quality_percentage < 0.66 --> medium burn_quality        
        elif (burn_quality_percentage>=0.33 and burn_quality_percentage<0.66):
                burn_quality = 'medium' 
        #   0.66 <= burn_quality_percentage < 1 --> high burn_quality        
        elif (burn_quality_percentage>=0.66 and burn_quality_percentage<1):
                burn_quality = 'high' 
        #  burn_quality_percentage == 2 --> failed burn            
        elif burn_quality_percentage == 2:
                burn_quality = 'failed'

        # reconcilation logic for low and high burn-quality        
        if (burn_quality == 'low') or (burn_quality == 'high'):
            
            # getting the partition for breaking the rpm-bins into two equal bins 
            bin_size = (RPM_RANGE[-1]-RPM_RANGE[0])//2       
            # getting the columns (we are capturing the count and mean of the dp values in each of the rpm-bins)
            columns = [
                    'RPM Bin', 'Speed Bin', 'Count', 
                    'Mean'
                    ]

            # capturing the decriptive statistics, and their corresponding rpm and speed bin
            df_pre_statistics_all_vehicles = pd.DataFrame(columns = columns)
            df_post_statistics_all_vehicles = pd.DataFrame(columns = columns)

            pre_descriptive_statistics_all_bins = []
            post_descriptive_statistics_all_bins = []
            pre_rpm_bin_considered = []
            post_rpm_bin_considered = []
            pre_speed_bin_considered = []
            post_speed_bin_considered = []

            # getting the RPM_RANGE and creating bin on those
            for bin in range(RPM_RANGE[0],  RPM_RANGE[-1], bin_size):
                    # creating the low and high speed bins for each of the rpm-bins
                    # getting the pre-bins
                    pre_dp_in_rpm_low_speed_bin = []
                    pre_dp_in_rpm_high_speed_bin = []
                    # getting the post-bins
                    post_dp_in_rpm_low_speed_bin = []
                    post_dp_in_rpm_high_speed_bin = []


                    # parsing through all the enteries of the dp-window
                    for j in range(len(pre_dp_window)):  
                            
                            # if the rpm values are between the bin & (bin+bin_size)

                            # checking the above condition for pre-rpm window
                            if pre_rpm_window[j]>= bin and pre_rpm_window[j] < (bin+bin_size):
                                    # getting the dp in the low-speed zone of specific rpm-zone
                                    if int(pre_speed_window[j]) <= SPEED_THRESHOLD:
                                            pre_dp_in_rpm_low_speed_bin.append(pre_dp_window[j])
                                    # getting the dp in the high-speed zone of specific rpm-zone    
                                    else:
                                            pre_dp_in_rpm_high_speed_bin.append(pre_dp_window[j])   
                            
                            # checking the above condition for the post-window
                            if post_rpm_window[j]>= bin and post_rpm_window[j] < (bin + bin_size):  
                                    # getting the dp in the low-speed zone of specific rpm-zone
                                    if int(post_speed_window[j]) <= SPEED_THRESHOLD:
                                            post_dp_in_rpm_low_speed_bin.append(post_dp_window[j])
                                    # getting the dp in the high-speed zone of specific rpm-zone    
                                    else:
                                            post_dp_in_rpm_high_speed_bin.append(post_dp_window[j])             

                    # if rpm with lower-speed bin
                    if len(pre_dp_in_rpm_low_speed_bin) > 0:
                            summary_statistics =  pd.DataFrame(pre_dp_in_rpm_low_speed_bin).describe()  
                            pre_descriptive_statistics_all_bins.append([
                                                            summary_statistics.loc['count'][0],
                                                            summary_statistics.loc['mean'][0], 
                                                            ]) 
                            pre_rpm_bin_considered.append((bin, (bin+bin_size)))
                            pre_speed_bin_considered.append("<="+str(SPEED_THRESHOLD))
                                            
                    if len(post_dp_in_rpm_low_speed_bin) > 0:
                            summary_statistics =  pd.DataFrame(post_dp_in_rpm_low_speed_bin).describe()  
                            post_descriptive_statistics_all_bins.append([
                                                            summary_statistics.loc['count'][0],
                                                            summary_statistics.loc['mean'][0] 
                                                    ]) 
                            post_rpm_bin_considered.append((bin, (bin+bin_size)))
                            post_speed_bin_considered.append("<="+str(SPEED_THRESHOLD))


                    # if rpm with higher-speed bin                
                    if len(pre_dp_in_rpm_high_speed_bin) > 0:
                            summary_statistics =  pd.DataFrame(pre_dp_in_rpm_high_speed_bin).describe()  
                            pre_descriptive_statistics_all_bins.append([
                                                            summary_statistics.loc['count'][0],
                                                            summary_statistics.loc['mean'][0]]) 
                            pre_rpm_bin_considered.append((bin, (bin+bin_size)))
                            pre_speed_bin_considered.append(">"+str(SPEED_THRESHOLD))

                    if len(post_dp_in_rpm_high_speed_bin) > 0:
                            summary_statistics =  pd.DataFrame(post_dp_in_rpm_high_speed_bin).describe()  
                            post_descriptive_statistics_all_bins.append([
                                                                    summary_statistics.loc['count'][0],
                                                                    summary_statistics.loc['mean'][0]]) 
                            post_rpm_bin_considered.append((bin, (bin+bin_size)))
                            post_speed_bin_considered.append(">"+str(SPEED_THRESHOLD))   
                                                            
                                    
            if len(pre_descriptive_statistics_all_bins):
                for j in range(len(pre_descriptive_statistics_all_bins)):  
                        pre_row_data = []
                        pre_row_data.append(pre_rpm_bin_considered[j]) #'RPM Bin'
                        pre_row_data.append(pre_speed_bin_considered[j]) #'Speed Bin'
                        pre_row_data.extend(pre_descriptive_statistics_all_bins[j]) 
                        df_pre_statistics_all_vehicles.loc[len(df_pre_statistics_all_vehicles)] = pre_row_data
        
            if len(post_descriptive_statistics_all_bins):
                for j in range(len(post_descriptive_statistics_all_bins)):
                        post_row_data = []  
                        post_row_data.append(post_rpm_bin_considered[j]) #'RPM Bin'
                        post_row_data.append(post_speed_bin_considered[j]) #'Speed Bin'
                        post_row_data.extend(post_descriptive_statistics_all_bins[j]) 
                        df_post_statistics_all_vehicles.loc[len(df_post_statistics_all_vehicles)] = post_row_data

            # dropping off duplicates from pre and post dataframe
            df_pre_statistics_all_vehicles = df_pre_statistics_all_vehicles.drop_duplicates()
            df_post_statistics_all_vehicles = df_post_statistics_all_vehicles.drop_duplicates()

            # renaming the columns for pre and post statistical dataframes
            df_pre_statistics_all_vehicles.columns = ["Pre RPM Bin", "Pre Speed Bin", 
                                                    "Pre Count", "Pre Mean"]
                    
            df_post_statistics_all_vehicles.columns = ["Post RPM Bin", "Post Speed Bin", 
                                                    "Post Count", "Post Mean"]
                    
            # combining pre and post dataframes basis RPM and SPEED bin values 
            df_statistics = pd.merge(df_pre_statistics_all_vehicles, df_post_statistics_all_vehicles, how='inner', 
                                    left_on=["Pre RPM Bin", "Pre Speed Bin"],
                                    right_on = ["Post RPM Bin", "Post Speed Bin"])
                    
            # In case any duplicates generated while combining the two dataframes drop them
            df_statistics = df_statistics.drop_duplicates()

            # dropping all those statistical bins wherein the Pre or Post Count of values are less than 5
            indices_remove = df_statistics[(df_statistics['Pre Count']<5) | (df_statistics['Post Count']<5)].index
            df_statistics.drop(indices_remove, inplace=True)
         
            fraction_good_bins = df_statistics[(df_statistics["Post Mean"]<df_statistics["Pre Mean"])].shape[0]/df_statistics.shape[0]
            
    
            # for "high" soot burn
            # the logic goes something like this
            # if 0/4 bins has the post-dp < pre-dp, classify as low
            # if 1/4 or 2/4 has the post-dp < pre-dp, classify as medium 
            if burn_quality == 'high' and fraction_good_bins == 0:
                    burn_quality = 'low'
                    burn_quality_percentage = burn_quality_percentage/3
            elif burn_quality == 'high' and fraction_good_bins <= 0.5 and fraction_good_bins != 0:
                    burn_quality = 'medium'
                    burn_quality_percentage = burn_quality_percentage/2   

            # for "low" soot burn
            # the logic goes something like this
            # if 3/4 or 4/4 has the post-dp < pre-dp, classify as medium 
            if burn_quality == 'low' and fraction_good_bins > 0.5:
                    burn_quality = 'medium'
                    burn_quality_percentage = burn_quality_percentage * 2
        high_speed_count = 0
        # giving the duration and speed status for each of the moderate and low burn_quality
        if burn_quality != 'high':     
                # if for atleast 50% of the time; the vehicle is running at required speed  
                # then set the speed status to sufficient i.e. 1        
                for speed in speed_inst_Value:
                        if speed >= SPEED_THRESHOLD:
                                high_speed_count += 1
                # if for atleast 50% of the time speed happens to be high during regeneration                
                if high_speed_count/len(speed_inst_Value) > 0.5:    
                        speed_status = 1
                        
                elif  high_speed_count/len(speed_inst_Value) <= 0.5: 
                        speed_status = 0
        # if burn_quality is high ; keep the speed status sufficient               
        elif burn_quality == 'high':
                speed_status = 1  

                        

        # return the speed, duration status and the time for which the function is executed
        return speed_status, burn_quality_percentage 


def REGENERATION_EVIDENCE_MSTR(vehicle_id, COUNTRY_FLAG, active_regeneration_start_time, active_regeneration_end_time, 
                            burn_quality_percentage, SPEED_FLAG):
    
        # extracting the OBD-data 
        
        Start_TS = active_regeneration_start_time - 60*60*1000
        End_TS = active_regeneration_end_time + 60*60*1000

        OBD_data = get_obd_data(vehicle_id, Start_TS, End_TS)  

        if SPEED_FLAG == 1:
                # if the OBD_data is not empty
                if len(OBD_data): 
                        # mapping the actual-regeneration time
                        actual_regeneration_time = active_regeneration_shift(OBD_data, active_regeneration_start_time)
                        speed_status, burn_quality_percentage = regeneration_evidence(COUNTRY_FLAG, OBD_data,
                                                                        active_regeneration_start_time, active_regeneration_end_time, 
                                                                        burn_quality_percentage)
                else:
                        actual_regeneration_time = active_regeneration_start_time
                        # if burn_quality is high --> speed_status should be sufficient --> 1
                        if burn_quality_percentage > 0.6:
                                speed_status = 1
                        # if burn quality anything apart from high --> speed status should be insufficient --> 0         
                        else:
                                speed_status = 0
        elif SPEED_FLAG == 0:
                actual_regeneration_time = active_regeneration_shift(OBD_data, active_regeneration_start_time)
                speed_status = 1

        return speed_status, burn_quality_percentage, actual_regeneration_time 









                