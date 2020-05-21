# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 09:51:31 2019

Plot the (0,0) data

@author: Aedan
"""

# %% Imports

import numpy
from scipy import exp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import utils.tool_belt as tool_belt

# %% Functions

# The exponential function used to fit the data

def exp_eq(t, offset, rate, amp):
    return offset+amp * exp(- rate * t)

# %%
    
# Area A1 
data = tool_belt.get_raw_data('t1_double_quantum/data_folders/other_data/bachman-ensemble-B5/', 
                              '2020_05_20-08_31_13-bachman-ensemble')

relaxation_time_range = numpy.array(data['relaxation_time_range'])/10**6
num_steps = data['num_steps']
norm_avg_sig_A1 = data['norm_avg_sig']
taus_A1 = numpy.linspace(relaxation_time_range[0], relaxation_time_range[1], num_steps)

# manipulate the data to normalize 
first_point = norm_avg_sig_A1[0]
last_point = norm_avg_sig_A1[-1]

norm_avg_sig_A1 = (numpy.array(norm_avg_sig_A1) - last_point)/ (first_point - last_point)

offset = 0.9
amplitude = 0.1
decay = 0.6*3 # inverse ns

popt_A1, pcov = curve_fit(exp_eq, taus_A1, norm_avg_sig_A1,
                          p0=[offset, decay, amplitude])

linspace_tau_A1 = numpy.linspace(relaxation_time_range[0], relaxation_time_range[1], 1000)

    
# Area B5 scc
data = tool_belt.get_raw_data('t1_double_quantum_scc_readout/branch_Spin_to_charge/2020_05/', 
                              '2020_05_20-17_35_53-bachman-B5')

relaxation_time_range = numpy.array(data['relaxation_time_range'])/10**6
num_steps = data['num_steps']
norm_avg_sig_B1 = data['norm_avg_sig']
taus_B1 = numpy.linspace(relaxation_time_range[0], relaxation_time_range[1], num_steps)

# manipulate the data to normalize 
first_point = norm_avg_sig_B1[0]
last_point = norm_avg_sig_B1[-1]

norm_avg_sig_B1 = (numpy.array(norm_avg_sig_B1) - last_point)/(first_point - last_point)

offset = 0.9
amplitude = 0.1
decay = 0.6*3 # inverse ns

popt_B1, pcov = curve_fit(exp_eq, taus_B1, norm_avg_sig_B1,
                          p0=[offset, decay, amplitude])

linspace_tau_B1 = numpy.linspace(relaxation_time_range[0], relaxation_time_range[1], 1000)
    
fig, ax = plt.subplots(1,1, figsize=(10, 8))
ax.plot(taus_A1, norm_avg_sig_A1,'ko',
                    label = 'B5, normal readout')
#ax.plot(linspace_tau_A1,
#                    exp_eq(linspace_tau_A1, *popt_A1),
#                    'k-', label = 'B5 fit')
ax.plot(taus_B1, norm_avg_sig_B1, 'ro',
                    label = 'B5, scc readout')
#ax.plot(linspace_tau_B1,
#                    exp_eq(linspace_tau_B1, *popt_B1),
#                    'r-', label = 'B1 fit')
ax.set_xlabel('Wait time (ms)')
ax.set_ylabel('Normalized signal Counts')
ax.set_title('Lifetime of ms = 0')
ax.legend()


