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
    
# Area A1 240
data = tool_belt.get_raw_data('t1_double_quantum/data_folders/other_data/bachman/bachman-A1-ensemble-B1-232MHz/', 
                              '2020_06_16-11_58_29-Bachman-A1-B1')
# Area A1 140
#data = tool_belt.get_raw_data('t1_double_quantum/data_folders/other_data/bachman/bachman-A1-ensemble-B1-138MHz/', 
#                              '2020_06_10-16_47_39-Bachman-A1-B1')

relaxation_time_range = numpy.array(data['relaxation_time_range'])/10**6
num_steps = data['num_steps']
#norm_avg_sig_A1 = data['norm_avg_sig']
num_runs = data['num_runs']
sig_counts = data["sig_counts"]
ref_counts = data["ref_counts"]
# Calculate the average signal counts over the runs, and st. error)
avg_sig_counts = numpy.average(sig_counts[:num_runs], axis=0)
ste_sig_counts = numpy.std(sig_counts[:num_runs], axis=0, ddof = 1) / numpy.sqrt(num_runs)

# Assume reference is constant and can be approximated to one value
avg_ref = numpy.average(ref_counts[:num_runs])

# Divide signal by reference to get normalized counts and st error
norm_avg_sig_A1 = avg_sig_counts / avg_ref
norm_avg_sig_ste_A1 = ste_sig_counts / avg_ref
taus_A1 = numpy.linspace(relaxation_time_range[0], relaxation_time_range[1], num_steps)
# manipulate the data to normalize 
first_point = norm_avg_sig_A1[0]
last_point = norm_avg_sig_A1[-1]

norm_avg_sig_A1 = (numpy.array(norm_avg_sig_A1) - last_point)/ (first_point - last_point)

offset = 0.9
amplitude = 0.1
decay = 0.6*3 # inverse ns

#popt_A1, pcov = curve_fit(exp_eq, taus_A1, norm_avg_sig_A1,
#                          p0=[offset, decay, amplitude])

linspace_tau_A1 = numpy.linspace(relaxation_time_range[0], relaxation_time_range[1], 1000)

    
# Area A5 240 MHz
data = tool_belt.get_raw_data('t1_double_quantum/data_folders/other_data/bachman/bachman-A5-ensemble-B1-234MHz/', 
                              '2020_06_01-07_06_15-bachman-A1')
# Area A5 140 MHz
#data = tool_belt.get_raw_data('t1_double_quantum/data_folders/other_data/bachman/bachman-A5-ensemble-B1-138MHz/', 
#                              '2020_06_02-15_04_58-bachman-B1')

relaxation_time_range = numpy.array(data['relaxation_time_range'])/10**6
num_steps = data['num_steps']
#norm_avg_sig_B1 = data['norm_avg_sig']
num_runs = data['num_runs']
sig_counts = data["sig_counts"]
ref_counts = data["ref_counts"]
# Calculate the average signal counts over the runs, and st. error)
avg_sig_counts = numpy.average(sig_counts[:num_runs], axis=0)
ste_sig_counts = numpy.std(sig_counts[:num_runs], axis=0, ddof = 1) / numpy.sqrt(num_runs)

# Assume reference is constant and can be approximated to one value
avg_ref = numpy.average(ref_counts[:num_runs])

# Divide signal by reference to get normalized counts and st error
norm_avg_sig_B1 = avg_sig_counts / avg_ref
norm_avg_sig_ste_B1 = ste_sig_counts / avg_ref
taus_B1 = numpy.linspace(relaxation_time_range[0], relaxation_time_range[1], num_steps)

# manipulate the data to normalize 
first_point = norm_avg_sig_B1[0]
last_point = norm_avg_sig_B1[-1]

norm_avg_sig_B1 = (numpy.array(norm_avg_sig_B1) - last_point)/(first_point - last_point)

offset = 0.9
amplitude = 0.1
decay = 0.6*3 # inverse ns

#popt_B1, pcov = curve_fit(exp_eq, taus_B1, norm_avg_sig_B1,
#                          p0=[offset, decay, amplitude])

linspace_tau_B1 = numpy.linspace(relaxation_time_range[0], relaxation_time_range[1], 1000)
    
fig, ax = plt.subplots(1,1, figsize=(10, 8))
ax.plot(taus_A1, norm_avg_sig_A1, 'yo',
                    label = 'Area A1')
#ax.plot(linspace_tau_A1,
#                    exp_eq(linspace_tau_A1, *popt_A1),
#                    'k-', label = 'B5 fit')
ax.plot(taus_B1, norm_avg_sig_B1, 'ro',
                    label = 'Area A5')
#ax.plot(linspace_tau_B1,
#                    exp_eq(linspace_tau_B1, *popt_B1),
#                    'r-', label = 'B1 fit')
#ax.set_yscale('log')
ax.set_xlabel('Wait time (ms)')
ax.set_ylabel('Normalized signal Counts')
ax.set_title('Lifetime of ms = 0')
#ax.set_xlim([-0.2,2.5])
#ax.set_ylim([0.05,1.2])
ax.legend()


