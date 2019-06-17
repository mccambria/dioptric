# -*- coding: utf-8 -*-
"""
Created on Fri May 31 11:06:46 2019

This routine takes the sets of data we take for relaxation measurments (prepare
in +1, readout in -1, etc) and calculates the relaxation rates, omega and
gamma, via the modified functions used in the Myer's paper ([0,0] - [0,1] and 
[1,1] - [1,-1]). It calculates the values for the whole data set and the 
standard deviation of the complete data set.

The main of this file uses the 
relaxation_rate_binning.main function to caluclate the average and standard 
deviation of the gamma and omega values. It either calculates the factors of 
the experiment's  num_runs for the bin sizes or takes a list of bin sizes. It 
then fits the  standard deviation values vs number of bins to a square root fit 
to extract the standard deviation of one single bin. It will report the 'value' 
of omega and gamma as the value found from the fit of the whole averaged data
set.

This file only works if all the experiments in a folder have the same number
of num_runs, and can only handle two data sets of the same experiment (ie +1 to
+1, a short and a long run).

@author: Aedan
"""

# %% Imports

import numpy
import json
from scipy import asarray as ar, exp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import utils.tool_belt as tool_belt
import analysis.relaxation_rate_binning as relaxation_rate_binning

# %% Constants

data_folder = 't1_double_quantum'

# %% Functions
    
# Calculate the functions 
def factors(number):
    factor_list = []
    for n in range(1, number + 1):
        if number % n == 0:
            factor_list.append(n)
       
    return factor_list

# %% Main 
    
def main(folder_name, num_bins_list = None):
    
    # If the list for number of bins is not passed through, use the factors of 
    # the num_runs
    if num_bins_list == None:
        
        # Get the file list from this folder
        file_list = tool_belt.get_file_list(data_folder, '.txt', folder_name)
          
        # Get the number of runs to create the empty arrays from the first file in 
        # the list. This requires all the relaxation measurements to have the same
        # num_runs
        file = file_list[0]
        for file in file_list:
            with open('{}/{}/{}'.format(directory, folder_name, file)) as json_file:
                try:
                    data = json.load(json_file)
                    num_runs = data['num_runs']
                    
                except Exception:
                    continue
        
        # Get the num_bins to use based on the factors of the number of runs
        
        num_bins_list = factors(num_runs)
    
    # Set up lists to save relavent data to
    
    omega_value_list = []
    omega_stdev_list = []
    gamma_value_list = []
    gamma_stdev_list = []
    
    # Create lists to put the fit_failed information in. We will fill each
    # element of the list with the list given by the analysis routine
    omega_fit_failed_list = [None] * len(num_bins_list)
    gamma_fit_failed_list = [None] * len(num_bins_list)
    
    
    # Step through the various bin sizes and compute the average and standard
    # deviation
    for num_bins_ind in range(len(num_bins_list)):
        num_bins = num_bins_list[num_bins_ind]
        retvals = relaxation_rate_binning.main(folder_name, num_bins,
                        False, False)
        
        # Save the data to the lists
        omega_value_list.append(retvals[0])
        omega_stdev_list.append(retvals[1])
        gamma_value_list.append(retvals[2])
        gamma_stdev_list.append(retvals[3])
        splitting_MHz = retvals[4]
            
        omega_fit_failed_list[num_bins_ind] = retvals[5]
        gamma_fit_failed_list[num_bins_ind] = retvals[6]

        
        # Save the calculated value of omega and gamma for the data for one bin
        if num_bins == 1:
            omega_value_one_bin = retvals[0]
            gamma_value_one_bin = retvals[2]
    
    # Take the average over the different values found using the different bin
    # sizes to compare to the value found using one bin        
    omega_value_avg = numpy.average(omega_value_list)
    gamma_value_avg = numpy.average(gamma_value_list)
     
    # Plot the data to visualize it. THis plot is not saved
    plt.loglog(num_bins_list, gamma_stdev_list, 'go', label = 'gamma standard deviation')
    plt.loglog(num_bins_list, omega_stdev_list, 'bo', label = 'omega standard deviation')
    plt.xlabel('Number of bins for num_runs')
    plt.ylabel('Standard Deviation (kHz)')
    plt.legend()
    
    # Fit the data to sqrt and extract the standadr deviation value for one bin
    def sqrt_root(x, amp):
        return amp * (x)**(1/2)
    print(omega_stdev_list)
    opti_params, cov_arr = curve_fit(sqrt_root, num_bins_list, 
                                     omega_stdev_list, p0 = (0.1))
    omega_stdev = sqrt_root(1, opti_params[0])
    print('Omega Value = {}, std dev = {}'.format(omega_value_one_bin, omega_stdev))
    
    opti_params, cov_arr = curve_fit(sqrt_root, num_bins_list, 
                                     gamma_stdev_list, p0 = (1))
    gamma_stdev = sqrt_root(1, opti_params[0])
    print('Gamma Value = {}, std dev = {}'.format(gamma_value_one_bin, gamma_stdev))
    
    time_stamp = tool_belt.get_time_stamp()
    raw_data = {'time_stamp': time_stamp,
                'splitting_MHz': splitting_MHz,
                'splitting_MHz-units': 'MHz',
                'omega_value_one_bin': omega_value_one_bin,
                'omega_value-units': 'kHz',
                'omega_stdev': omega_stdev,
                'omega_stdev-units': 'kHz',
                'gamma_value_one_bin': gamma_value_one_bin,
                'gamma_value-units': 'kHz',
                'gamma_stdev': gamma_stdev,
                'gamma_stdev-units': 'kHz',
                'omega_value_avg': omega_value_avg,
                'omega_value_avg-units': 'kHz',
                'gamma_value_avg': gamma_value_avg,
                'gamma_value_avg-units': 'kHz',      
                'num_bins_list': num_bins_list,
                'omega_fit_failed_list': omega_fit_failed_list,
                'gamma_fit_failed_list': gamma_fit_failed_list,
                'omega_value_list': omega_value_list,
                'omega_value_list-units': 'kHz',
                'omega_stdev_list': omega_stdev_list,
                'omega_stdev_list-units': 'kHz',
                'gamma_value_list': gamma_value_list,
                'gamma_value_list-units': 'kHz',
                'gamma_stdev_list': gamma_stdev_list,
                'gamma_stdev_list-units': 'kHz'
                }
    
    file_name = time_stamp + '_' + str('%.1f'%splitting_MHz) + \
                '_MHz_splitting_rate_analysis' 
    file_path = '{}/{}/{}'.format(directory, folder_name, file_name)
    
    with open(file_path + '.txt', 'w') as file:
        json.dump(raw_data, file, indent=2)
        
        
# %% Run the file
    
if __name__ == '__main__':
 
    
    # Set the file to pull data from here. These should be files in our 
    # Double_Quantum nvdata folder, filled with the 6 relevant experiments
    
    folder = 'nv13_2019_06_10_72MHz'
    
    '''
    MAIN: this will calculate the value and standard deviation of gamma and
        omega for the whole data set. 
        
        It's important to check that the values
        make sense: occaionally when the bins get too small the data is too noisy
        to accurately fit. Both check that the standard deviation is smaller than
        than the value (we've been seeing a stdev ~ 20-5%), and check the saved 
        txt file for the list of values. If need be, the bins to run through
        can be specified
        
    '''
    
#    # Specify the number of bins
    num_bins_list = [1,2,4, 5, 8, 10]
    main(folder, num_bins_list)
    
    # Use the factors of the num_runs for the num_bins
#    main(folder)
    
    
        
        
        