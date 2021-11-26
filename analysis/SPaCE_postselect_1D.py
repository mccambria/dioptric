# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 09:55:03 2021

@author: agard
"""



import utils.tool_belt as tool_belt
import numpy
import matplotlib.pyplot as plt

dx_max= 0.01 #um
dy_max= 0.01 #um
dz_max= 0.1 #V

def do_postselect(file, folder, x_unit, y_unit, z_unit):
    data = tool_belt.get_raw_data(file, folder)
    drift_list_master = data['drift_list_master']
    readout_counts_array = numpy.flipud(numpy.rot90(data['readout_counts_array']))
    ps_readout_counts_array = []
    
    drift_x = []
    drift_y = []
    drift_z = []
    
    # create a single list of all the drifts
    for i in range(len(drift_list_master)):
        drifts_for_run = drift_list_master[i]
        for j in range(len(drifts_for_run)):
            #just take the first optimize, which should ocur at regular intervals
            # based on the time elapsed and the number of runs
            drift_x.append(drifts_for_run[j][0])
            drift_y.append(drifts_for_run[j][1])
            drift_z.append(drifts_for_run[j][2])
    
    # create a signle lists of the change in drift from point n-1 to point n, 
    # starting at 1. Put in a 0th index with value of 0
    dx_drift = [0]
    dy_drift = [0]
    dz_drift = [0]
    
    for i in range(len(drift_x)-1):
        i = i+1
        dx = drift_x[i] - drift_x[i-1]
        dx_drift.append(dx)
        dy = drift_y[i] - drift_y[i-1]
        dy_drift.append(dy)
        dz = drift_z[i] - drift_z[i-1]
        dz_drift.append(dz)

    #now go throguh the lsit of drifts, keep track of what change in drift we are on,
    # and if the change in drift during a run is too large, exclude that data.
    #the change in drift corresponds to the current index, so if it is too large, we throw out that index
    flattened_d_ind = 0
    for n in range(len(drift_list_master)):
        num_opti = len(drift_list_master[n])
        drift_too_large = False
        # check each of the drifts in the run. Right now, we're tossing out the whole run if 
        #one of the change in drifts was too large. Could get more sophisticated in future.
        for i in range(num_opti):
            d = flattened_d_ind + i # the index we want to consider in the flattened change-in-drift lists
            if dx_drift[d] > dx_max or dy_drift[d] > dy_max or dz_drift[d] > dz_max:
                drift_too_large = True #if the change in any axis is too large, flag it
        if not drift_too_large:
            ps_readout_counts_array.append(readout_counts_array[n]) #only keep the data if the changes were small
        flattened_d_ind = flattened_d_ind + num_opti

    print(len(ps_readout_counts_array))
folder = 'pc_rabi/branch_CFMIII/SPaCE_digital/2021_11'
file = '2021_11_16-09_49_09-johnson-nv3_2021_11_08'
do_postselect(file, folder, 'um', 'um', 'V')

