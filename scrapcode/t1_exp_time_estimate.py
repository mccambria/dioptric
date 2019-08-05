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
        extra_seq_time = 20 * 10**-6
        optimize_time = 0.5
        
        sequence_time = (relaxation_time_s + extra_seq_time) * num_reps
        exp_time_s = (sequence_time * num_steps / 2 + optimize_time) * num_runs # seconds
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
        
num_runs = 40

t1_exp_array = numpy.array([[[States.HIGH, States.LOW],  [0, 50*10**3], 51, 6*10**4, num_runs], # 0.8 hrs, optimize every 1.3 min
                                [[States.HIGH, States.LOW],  [0, 400*10**3], 26, 3.75*10**4, num_runs], # 3.8 hrs, optimize every 5.5 min
                                [[States.HIGH, States.HIGH],  [0, 50*10**3], 51, 6*10**4, num_runs], # 0.8 hrs, optimize every 1.3 min
                                [[States.HIGH, States.HIGH],  [0, 400*10**3], 26, 3.75*10**4, num_runs], # 3.8 hrs, optimize every 5.5 min
                                [[States.ZERO, States.HIGH],  [0, 2*10**6], 26, 1.5*10**4, num_runs], # 4.3 hrs, optimize every 6.5 min
                                [[States.ZERO, States.ZERO],  [0, 2*10**6], 26, 1.5*10**4, num_runs]]) # 4.3 hrs, optimize every 6.5 min

contrast = 0.75
exp_count_rate = 22 # kcps
readout_window = 450 # ns

t1_exp_times(t1_exp_array, contrast, exp_count_rate, readout_window)