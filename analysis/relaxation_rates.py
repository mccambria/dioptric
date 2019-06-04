# -*- coding: utf-8 -*-
"""
Created on Fri May 31 11:06:46 2019

This routine takes the sets of data we take for relaxation measurments (prepare
in +1, readout in -1, etc) and calculates the relaxation rates, omega and
gamma. It calculates the values for each run of the data (num_runs). It will
then allow us to average the value for the relaxation rate and take a standard
deviation.

@author: Aedan
"""
import os
import numpy
import json
from scipy import asarray as ar,exp
from scipy.optimize import curve_fit

def relaxation_rate_analysis(folder_name):
    
    directory = 'G:/Team Drives/Kolkowitz Lab Group/nvdata/t1_double_quantum/' 
    
    file_list = []
    for file in os.listdir('{}/{}'.format(directory, folder_name)):
        if file.endswith(".txt"):
            file_list.append(file)

#    print(file_list)
    
    for file in file_list:
        with open('{}/{}/{}'.format(directory, folder_name, file)) as json_file:
            data = json.load(json_file)
            init_state = data['init_state']
            read_state = data['read_state']
            
            relaxation_time_range = data['relaxation_time_range']
            num_steps = data['num_steps']
            num_runs = data['num_runs']
            
            if 
            print('({}, {})'.format(init_state, read_state))
            
            # numpy.concatenate((a, b), axis=1)
        
    # import arrays to work with
    # import data, reading in the readout out states to sort them
    # subreact the relative data to get the two functions
    
    # split up the num_runs into various amounts
    # fit each bin
    # average and st dev
if __name__ == '__main__':
    
    relaxation_rate_analysis('2019-04-30-NV2_29MHzSplitting_important_data')