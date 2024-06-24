import numpy as np
from app.dataIO import get_obd_data 
import json
import requests


# function to be called for making the active-regeneration point shift
def active_regeneration_shift(OBD_data, active_regeneration_start_time, spec_obj):
        
        # extracting the relevant soot-load parameter-ID
        
        if 'dpf_soot_loading_pid' in spec_obj['additional_info']:
                soot_load_pid = spec_obj['additional_info']['dpf_soot_loading_pid']
        
        Time = []
        soot_load_Value = []
        for pac_idx in range(len(OBD_data)):
                if "pids" in OBD_data[pac_idx]:
                    if len(OBD_data[pac_idx]['pids'])>0:
                        for sub_pid_cnt in range(0,len(OBD_data[pac_idx]['pids'])):
                                State = OBD_data[pac_idx]['pids'][sub_pid_cnt]
                                # extracting Soot-Load
                                if soot_load_pid in State:
                                        Time.append(State[soot_load_pid]['timestamp'])
                                        soot_load_Value.append(State[soot_load_pid]['value'][0])
        
        # if soot_load values not available
        if len(soot_load_Value)==0:
               return active_regeneration_start_time
        
        # if soot load values are available --> find the point where maximum negative slope present
        slope = np.array(soot_load_Value[1:-1]) - np.array(soot_load_Value[0:-2])

        adjusted_index = np.argmin(slope)

        # if the point of maximum drop is greater than 10 minutes
        # search for the maximum drop wthin 10 minutes prior to active-regeneration time
        if (active_regeneration_start_time - Time[adjusted_index-1])>(10*60*1000):
                idx_10_mins_before = (np.abs(np.asarray(Time) - (active_regeneration_start_time - 10*60*1000))).argmin()
                adjusted_index = idx_10_mins_before
        if (Time[adjusted_index-1]-active_regeneration_start_time)>(10*60*1000):
                idx_10_mins_after = (np.abs(np.asarray(Time) - (active_regeneration_start_time + 10*60*1000))).argmin()
                adjusted_index = idx_10_mins_after  

        if slope[adjusted_index]<=-20:
               actual_regeneration_time =  Time[adjusted_index-1]
        else:
               actual_regeneration_time =  active_regeneration_start_time                
                        

        return actual_regeneration_time
                                      



# function to get the regeneration evidence for regeneration instance
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
                if burn_quality_percentage>=66:
                        speed_status = 1
                else:
                        speed_status = 0        
                return speed_status, burn_quality_percentage

        
        
        # if FLAG is set to "US" use the following RPM and Speed thresholds
        if  COUNTRY_FLAG == 'US':
                RPM_RANGE = [800, 2000]
                SPEED_THRESHOLD = 80

        # if FLAG is set to "IN" use the following RPM and Speed thresholds
        elif COUNTRY_FLAG == 'IN':
                RPM_RANGE = [1000, 2000]
                SPEED_THRESHOLD = 40
 

        # extracting the relevant parameter-ID
        dp_pid = 'spn_3251_avg'
        rpm_pid = 'spn_190_avg'
        speed_pid = 'spn_84_avg'
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
                                if dp_pid in State:
                                        Time.append(State[dp_pid]['timestamp'])
                                        dp_inst_Value.append(State[dp_pid]['value'][0])
                                # extracting RPM        
                                if rpm_pid in State:
                                        rpm_inst_Value.append(State[rpm_pid]['value'][0])
                                # extracting SPEED       
                                if speed_pid in State:
                                        speed_inst_Value.append(State[speed_pid]['value'][0])    
                              
        # applying the RPM constraint
        # getting the corresponding values of different varaibles after applying constraints
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
                if burn_quality_percentage>66:
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
        #   0 < burn_quality_percentage <= 33 --> low burn_quality
        if (burn_quality_percentage>0 and burn_quality_percentage<=33):
                burn_quality = 'low'
        #   33 < burn_quality_percentage <=66 --> medium burn_quality        
        elif (burn_quality_percentage>33 and burn_quality_percentage<=66):
                burn_quality = 'medium' 
        #   66 < burn_quality_percentage < 100 --> high burn_quality        
        elif (burn_quality_percentage>66 and burn_quality_percentage<100):
                burn_quality = 'high' 
        elif burn_quality_percentage == 0:
               burn_quality = 'failed' 
                     
        

        # reconcilation logic for low and high burn-quality        
        if (burn_quality == 'low') or (burn_quality == 'high'):
            
            # getting the partition for breaking the rpm-bins into two equal bins 
            bin_size = (RPM_RANGE[-1]-RPM_RANGE[0])//2       
            
        
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
                            
                            pre_descriptive_statistics_all_bins.append([
                                                            len(pre_dp_in_rpm_low_speed_bin),
                                                            np.mean(pre_dp_in_rpm_low_speed_bin) 
                                                            ]) 
                            pre_rpm_bin_considered.append((bin, (bin+bin_size)))
                            pre_speed_bin_considered.append("<="+str(SPEED_THRESHOLD))
                                            
                    if len(post_dp_in_rpm_low_speed_bin) > 0:
                            
                            post_descriptive_statistics_all_bins.append([
                                                             len(post_dp_in_rpm_low_speed_bin),
                                                             np.mean(post_dp_in_rpm_low_speed_bin) 
                                                    ]) 
                            post_rpm_bin_considered.append((bin, (bin+bin_size)))
                            post_speed_bin_considered.append("<="+str(SPEED_THRESHOLD))


                    # if rpm with higher-speed bin                
                    if len(pre_dp_in_rpm_high_speed_bin) > 0:
                            
                            pre_descriptive_statistics_all_bins.append([
                                                            len(pre_dp_in_rpm_high_speed_bin),
                                                            np.mean(pre_dp_in_rpm_high_speed_bin)])
                            pre_rpm_bin_considered.append((bin, (bin+bin_size)))
                            pre_speed_bin_considered.append(">"+str(SPEED_THRESHOLD))

                    if len(post_dp_in_rpm_high_speed_bin) > 0:
                            
                            post_descriptive_statistics_all_bins.append([
                                                                   len(post_dp_in_rpm_high_speed_bin),
                                                                   np.mean(post_dp_in_rpm_high_speed_bin)]) 
                            post_rpm_bin_considered.append((bin, (bin+bin_size)))
                            post_speed_bin_considered.append(">"+str(SPEED_THRESHOLD))   
                                                            
                                    
            pre_descriptive_statistics_all_bins = np.array(pre_descriptive_statistics_all_bins)
            post_descriptive_statistics_all_bins = np.array(post_descriptive_statistics_all_bins)

            # Create a function to create a 2D array from input arrays
            def create_2d_array(arr1, arr2, arr3):
                return np.column_stack((arr1, arr2, arr3))

            
            if len(pre_descriptive_statistics_all_bins):
                pre_statistics = create_2d_array(pre_rpm_bin_considered, pre_speed_bin_considered, pre_descriptive_statistics_all_bins)
        
            if len(post_descriptive_statistics_all_bins):
                post_statistics = create_2d_array(post_rpm_bin_considered, post_speed_bin_considered, post_descriptive_statistics_all_bins)
                

            # Function to remove duplicates from a 2D array
            def remove_duplicates(arr):
                _, unique_indices = np.unique(arr, axis=0, return_index=True)
                return arr[np.sort(unique_indices)]

            # Remove duplicates from pre and post statistics arrays
            if pre_descriptive_statistics_all_bins.size > 0:
                pre_statistics = remove_duplicates(pre_statistics)

            if post_descriptive_statistics_all_bins.size > 0:
                post_statistics = remove_duplicates(post_statistics)


            # Function to merge pre and post statistics arrays based on RPM and Speed bins
            def merge_statistics(pre_arr, post_arr):
                merged_array = []

                if pre_arr.shape[0] >= post_arr.shape[0]:
                        for i in range(post_arr.shape[0]):
                                idx_ls = np.where((pre_arr[:, 0] == post_arr[i, 0]) & (pre_arr[:, 1] == post_arr[i, 1]) &  (pre_arr[:, 2] == post_arr[i, 2]))[0]
                                if len(idx_ls) > 0:
                                        idx = idx_ls[0]
                                        merged_array.append(np.concatenate((pre_arr[idx, :], post_arr[i, 3:])))
                elif  post_arr.shape[0] >= pre_arr.shape[0]:
                        for i in range(pre_arr.shape[0]):
                                idx_ls = np.where((post_arr[:, 0] == pre_arr[i, 0]) & (post_arr[:, 1] == pre_arr[i, 1]) &  (post_arr[:, 2] == pre_arr[i, 2]))[0]
                                if len(idx_ls) > 0:
                                        idx = idx_ls[0]
                                        merged_array.append(np.concatenate((pre_arr[i, :], post_arr[idx, 3:])))
                                      

                return np.array(merged_array)

            # Merge pre and post statistics arrays
            if pre_descriptive_statistics_all_bins.size > 0 and post_descriptive_statistics_all_bins.size > 0:
                df_statistics = merge_statistics(pre_statistics, post_statistics)

                fraction_good_bins = 0
                relevant_bins = 0
                for i in range(df_statistics.shape[0]):
                       if float(df_statistics[i, 3]) >= 5 and float(df_statistics[i, 5]) >= 5:
                              relevant_bins += 1
                              if float(df_statistics[i, 4]) >  float(df_statistics[i, 6]):
                                     fraction_good_bins += 1

                if relevant_bins!=0:                   
                      fraction_good_bins =  fraction_good_bins/relevant_bins
                else:
                       fraction_good_bins = 0                      

    
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
                # if for atleast 50% of the regeneration time; the vehicle is running at required speed  
                # then set the speed status to sufficient i.e. 1  
                #  getting the index where active regeneration-starts & active-regeneration ends
                ar_start_idx = (np.abs(np.array(Time) -  active_regeneration_start_time)).argmin()
                ar_end_idx = (np.abs(np.array(Time) -  active_regeneration_end_time)).argmin()  
           
                for speed in speed_inst_Value[ar_start_idx:(ar_end_idx+1)]:
                        if speed >= SPEED_THRESHOLD:
                                high_speed_count += 1
                # if for atleast 50% of the time speed happens to be high during regeneration                
                if high_speed_count/len(speed_inst_Value[ar_start_idx:(ar_end_idx+1)]) > 0.5:    
                        speed_status = 1
                        
                elif  high_speed_count/len(speed_inst_Value[ar_start_idx:(ar_end_idx+1)]) <= 0.5: 
                        speed_status = 0
        # if burn_quality is high ; keep the speed status sufficient               
        elif burn_quality == 'high':
                speed_status = 1  

                        

        # return the speed, duration status and the time for which the function is executed
        return speed_status, burn_quality_percentage 



# This fuction has to be called out
# It calls three modules
# get_obd_data to get the obd_data between the start and end time-stamp
# active_regeneration_shift to calibrate the active-regeneration start time corresponding to the soot-load signal
# regeneration_evidence for evidence generation for each regeneration instance

# the values returned are:  actual_regeneration_start_time : 'Actual Time Of Start Of Regeneration'
#                           duration_status :  0 --> if regen-duration (0, 10], 1 --> if regen-duration (10, 25], 2 --> if regen-duration (25, infinity)
#                           speed_status : 1 ---> if speed sufficient during regen, 0 --> if speed insufficient 
#                           burn_quality_percentage -- > modified burn-quality percentage after reconcilation
def REGENERATION_EVIDENCE_MSTR(vehicle_id: str, COUNTRY_FLAG: str, active_regeneration_start_time: int, active_regeneration_end_time: int, 
                            burn_quality_percentage: float, spec_obj: dict):
    
        # extracting the OBD-data 
        # Start_TS is 1 hour prior to active-regen start time
        # End_TS is 1 hour after the active-regen end time
        Start_TS = active_regeneration_start_time - 70*60*1000 
        End_TS = active_regeneration_end_time + 70*60*1000

        active_regeneration_duration = (active_regeneration_end_time - active_regeneration_start_time)

        # duration_status logging for the active-regeneration incident
        if active_regeneration_duration//(60*1000) <= 10:
               duration_status = 0 # insufficient
        elif 10 < active_regeneration_duration//(60*1000) <= 25:
               duration_status = 1   # moderate   
        else:
               duration_status = 2  # sufficient       

        # getting the obd-data
        OBD_data = get_obd_data(vehicle_id, Start_TS, End_TS)
            
        # if the OBD_data is not empty
        if len(OBD_data): 
                
                # getting the burn_evidence for ecah active regeneration instance
                speed_status, burn_quality_percentage = regeneration_evidence(COUNTRY_FLAG, OBD_data,
                                                               active_regeneration_start_time, active_regeneration_end_time, 
                                                               burn_quality_percentage)
        else:
               
                #if burn_quality is high --> speed_status should be sufficient --> 1
                if burn_quality_percentage > 66:
                       speed_status = 1
                #if burn quality anything apart from high --> speed status should be insufficient --> 0         
                else:
                       speed_status = 0
                

        
        return active_regeneration_start_time, duration_status, speed_status, burn_quality_percentage 
        























                