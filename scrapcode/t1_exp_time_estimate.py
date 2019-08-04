# -*- coding: utf-8 -*-
"""

Calculate the estimated time for the t1 experiment, how often we optimize, and 
the expected error

Created on Thu Aug  1 17:32:46 2019

@author: agardill
"""
# %%
from utils.tool_belt import States
import numpy
# %%

def expected_st_dev_norm(ref_counts, expected_contrast):
    sig_counts = expected_contrast * ref_counts
    rel_std_sig = numpy.sqrt(sig_counts) / sig_counts
    rel_std_ref = numpy.sqrt(ref_counts) / ref_counts
    # Propogate the error
    error = expected_contrast * numpy.sqrt((rel_std_sig**2) + (rel_std_ref**2))
    
    return error
    
def t1_exp_times(exp_array, contrast, exp_count_rate, readout_window):
    total_exp_time_list = []
    
    for line in exp_array:
        exp = line[0]
        relaxation_time_s = line[1][1] * 10**-9
        num_steps = line[2]
        num_reps = line[3]
        num_runs = line[4]
        
        exp_time_s = relaxation_time_s * num_steps / 2 * num_reps * num_runs # seconds
        exp_time_m = exp_time_s / 60
        exp_time_h = exp_time_m / 60
        
        opti_time = exp_time_m / num_runs
        
        total_exp_time_list.append(exp_time_h)
        
        ref_counts = (exp_count_rate * 10 ** 3) * (readout_window * 10**-9) * num_reps * num_runs
        
        exp_error_percent = expected_st_dev_norm(ref_counts, contrast) * 100
        
        print('{}: {} hours, optimize every {} minutes, expected error {}%'.format(exp, '%.1f'%exp_time_h, '%.2f'%opti_time, '%.1f'%exp_error_percent))
        
    total_exp_time = sum(total_exp_time_list)    
    
    
    print('Total exp time: {} hrs'.format('%.1f'%total_exp_time))
        
# %%
        
num_runs = 30

t1_exp_array = numpy.array([[[States.HIGH, States.LOW],  [0, 50*10**3], 51, 8*10**4, num_runs], # 0.9 hrs, optimize every 1.7 min
                                [[States.HIGH, States.LOW],  [0, 700*10**3], 26, 5*10**4, num_runs], # 1.7 hrs, optimize every 3.5 min
                                [[States.HIGH, States.HIGH],  [0, 50*10**3], 51, 8*10**4, num_runs], # 0.9 hrs, optimize every 1.7 min
                                [[States.HIGH, States.HIGH],  [0, 700*10**3], 26, 5*10**4, num_runs], # 1.7 hrs, optimize every 3.5 min
                                [[States.ZERO, States.HIGH],  [0, 2*10**6], 26, 2*10**4, num_runs], # 4.3 hrs, optimize every 9 min
                                [[States.ZERO, States.ZERO],  [0, 2*10**6], 26, 2*10**4, num_runs]]) # 4.3 hrs, optimize every 9 min

contrast = 0.75
exp_count_rate = 22 # kcps
readout_window = 450 # ns

t1_exp_times(t1_exp_array, contrast, exp_count_rate, readout_window)