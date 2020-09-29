# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 17:35:30 2020

@author: matth
"""


# %% Imports


import numpy as np
from numpy import exp
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.optimize import brute
import matplotlib
import matplotlib.pyplot as plt
import utils.tool_belt as tool_belt
import json
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
    
ms = 5
lw = 1.75


# %% Constants


mixed_pl = 0.85   # relative pl at even mixture
contrasts = [1.00, 0.8, 0.86]  # 0, -1, +1

# infidelity_low = 3/20
# infidelity_high = 2/14
infidelity_low = 0
infidelity_high = 0

# Calculated from solve_initials
pol_eff_calc = 0.680343644216234
state_pls_calc = [1.0677351438344787, 0.6834924587267566, 0.7987723974387646]


# %% Functions


def calc_lambda_parts(omega_minus, omega_plus, gamma):
    
    part_1 = omega_minus + omega_plus + gamma
    part_2 = omega_plus**2 + omega_minus**2 + gamma**2
    part_3 = -(omega_plus*gamma) - (omega_minus*gamma) - (omega_minus*omega_plus)
    
    return part_1, part_2, part_3


def calc_lambda_plus(omega_minus, omega_plus, gamma):
    
    part_1, part_2, part_3 = calc_lambda_parts(omega_minus, omega_plus, gamma)
    return -part_1 + np.sqrt(part_2 + part_3)


def calc_lambda_minus(omega_minus, omega_plus, gamma):
    
    part_1, part_2, part_3 = calc_lambda_parts(omega_minus, omega_plus, gamma)
    return -part_1 - np.sqrt(part_2 + part_3)


def concat_three_level(time, omega_minus, omega_plus, gamma):
    
    # minus/low and plus/high are used interchangeably here
    
    return_signal = []
    
    lambda_plus = calc_lambda_plus(omega_minus, omega_plus, gamma)
    lambda_minus = calc_lambda_minus(omega_minus, omega_plus, gamma)
    alpha = pol_eff_calc
    c_1 = (1/(lambda_minus-lambda_plus))*((1/6)-(alpha/2)+((alpha-(1/3))*((2*omega_minus)+gamma+lambda_minus)))
    c_2 = alpha - c_1 - (1/3)
    
    len_time = len(time)
    for ind in range(len_time):
        
        # The dynamics are all the same until the pi pulses at the end so
        # so calculate the dynamics here, then apply the pi pulses. 
        pop_zero = (c_1 * exp(lambda_plus*time[ind])) + (c_2 * exp(lambda_minus*time[ind])) + (1/3)
        part_1 = c_1 * ((omega_minus-gamma)/(omega_minus+(2*gamma)+lambda_plus)) * exp(lambda_plus*time[ind])
        part_2 = c_2 * ((omega_minus-gamma)/(omega_minus+(2*gamma)+lambda_minus)) * exp(lambda_minus*time[ind])
        pop_low = part_1 + part_2 + (1/3)
        pop_high = 1 - pop_zero - pop_low
        pops = [pop_zero, pop_low, pop_high]
        
        # Apply pi pulses
        
        # None
        if ind < len_time / 3:
            pops_pi_pulse = pops
        
        # Low
        elif ind < 2 * len_time / 3:
            pops_pi_pulse = [(pops[0]*infidelity_low)+(pops[1]*(1-infidelity_low)),
                              (pops[1]*infidelity_low)+(pops[0]*(1-infidelity_low)),
                              pops[2]]
            # pops_pi_pulse = [pops[1], pops[0], pops[2]]
        
        # High
        else:
            pops_pi_pulse = [(pops[0]*infidelity_high)+(pops[2]*(1-infidelity_high)),
                              pops[1],
                              (pops[2]*infidelity_high)+(pops[0]*(1-infidelity_high))]
            # pops_pi_pulse = [pops[2], pops[1], pops[0]]
            
        # Append the photoluminescence calculated from the populations
        return_signal.append(np.dot(pops_pi_pulse, state_pls_calc))

    return return_signal


def process_raw_data(data, ref_range=None):
    """Pull the relaxation signal and ste out of the raw data."""

    num_runs = data['num_runs']
    num_steps = data['num_steps']
    sig_counts = np.array(data['sig_counts'])
    ref_counts = np.array(data['ref_counts'])
    time_range = np.array(data['relaxation_time_range'])

    # Calculate time arrays in ms
    min_time, max_time = time_range / 10**6
    times = np.linspace(min_time, max_time, num=num_steps)

    # Calculate the average signal counts over the runs, and ste
    avg_sig_counts = np.average(sig_counts[::], axis=0)
    ste_sig_counts = np.std(sig_counts[::], axis=0, ddof = 1) / np.sqrt(num_runs)

    # Assume reference is constant and can be approximated to one value
    avg_ref = np.average(ref_counts[::])
    # avg_ref = np.average(ref_counts[::], axis=0)  # test norm per point

    # Divide signal by reference to get normalized counts and st error
    norm_avg_sig = avg_sig_counts / avg_ref
    norm_avg_sig_ste = ste_sig_counts / avg_ref

    # Normalize to the reference range
    if ref_range is not None:
        diff = ref_range[1] - ref_range[0]
        norm_avg_sig = (norm_avg_sig - ref_range[0]) / diff

    return norm_avg_sig, norm_avg_sig_ste, times


def calculate_pls(pol_zero, A_zero, A_low):
    
    # infidelity_high = 0.10
    # pol_eff = 0.6
    pol_rem = (1 - pol_zero) / 2
    pol_low = pol_rem
    pol_high = 1 - pol_zero - pol_low
    
    # pops = np.array([[pol_eff, pol_rem+pol_imb_low, pol_rem-pol_imb_low],
    #                  [pol_rem+pol_imb_low, pol_eff, pol_rem-pol_imb_low],
    #                  [pol_rem-pol_imb_low, pol_rem+pol_imb_low, pol_eff],
    #                  [1/3, 1/3, 1/3]])
    
    # pops = np.array([[pol_eff, pol_rem, pol_rem],
    #                  [(pol_eff*infidelity_low)+(pol_rem*(1-infidelity_low)),
    #                   (pol_rem*infidelity_low)+(pol_eff*(1-infidelity_low)),
    #                   pol_rem],
    #                  [(pol_eff*infidelity_high)+(pol_rem*(1-infidelity_high)),
    #                   pol_rem,
    #                   (pol_rem*infidelity_high)+(pol_eff*(1-infidelity_high))],
    #                  [1/3, 1/3, 1/3]])
    
    # pops = np.array([[pol_eff, pol_rem, pol_rem],
    #                  [pol_rem, pol_eff, pol_rem],
    #                  [pol_rem, pol_rem, pol_eff],
    #                  [1/3, 1/3, 1/3]])
    
    # pops = np.array([[pol_zero, pol_low, pol_high],
    #                  [pol_low, pol_zero, pol_high],
    #                  [pol_high, pol_low, pol_zero],
    #                  [1/3, 1/3, 1/3]])
    
    pops = np.array([[pol_zero, pol_low, pol_high],
                     [pol_low, pol_zero, pol_high],
                     [pol_high, pol_low, pol_zero]])
    
    # mixed_pl = (1/3)(A_zero + A_low + A_high)
    A_high = (3*mixed_pl) - A_zero - A_low
    
    pure_pls = np.array([A_zero, A_low, A_high])
    
    calculated_pls = np.matmul(pops, pure_pls)
    
    return calculated_pls


def solve_initials_objective(args, measured_pls):
    
    calculated_pls = calculate_pls(*args)
    diff = calculated_pls-measured_pls
    
    return np.dot(diff, diff)


def solve_initials_brute(pol_zero, A_zero, A_low, measured_pls):
    
    calculated_pls = calculate_pls(pol_zero, A_zero, A_low)
    diff = calculated_pls-measured_pls
    
    return np.dot(diff, diff)


def solve_initials(measured_pls):
    
    # brute_obj = lambda pol_zero, A_zero, A_low: solve_initials_objective(pol_zero, A_zero, A_low, measured_pls)
    ret_vals = brute(solve_initials_brute, 
                     ((0.4, 0.8), (1.0, 1.4), (0.45, 0.85)),
                     Ns=5, args=measured_pls)
    
    guess = ret_vals[0]
    print(guess)
    res = minimize(solve_initials_objective, guess, args=measured_pls)
    
    pol_zero, A_zero, A_low = res.x
    pol_low = (1 - pol_zero) / 2
    pol_high = (1 - pol_zero) / 2
    A_high = (3*mixed_pl) - A_zero - A_low
    pols = [pol_zero, pol_low, pol_high]
    pls = [A_zero, A_low, A_high]
    
    print(pols)
    print(pls)
    print(calculate_pls(*res.x))
    
    # test = [0.15, 1.12272697, 0.75909075, 0.66818257]
    # print(calculate_pls(*test))
            

# %% Main


def main(folder, labels, files):

    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    ax.set_xlabel(r'Wait time $\tau$ (ms)')
    ax.set_ylabel('Fluorescence (arb. units)')
    
    source = 't1_double_quantum/data_folders/paper_data/bulk_dq/'
    path = source + folder
    
    signals = []
    stes = []
    times = []
    
    for ind in range(3):
        
        label = labels[ind]
        file = files[ind]
        
        data = tool_belt.get_raw_data(path, file)
        signal, ste, time = process_raw_data(data)
        ax.errorbar(time, signal, yerr=ste, label=label ,
                    marker='o', linestyle='None', ms=ms, lw=lw)
        
        signals.extend(signal)
        stes.extend(ste)
        times.extend(time)
    
    popt, pcov = curve_fit(concat_three_level, times, signals,
                           sigma=stes, absolute_sigma=True,
                           p0=(0.060, 0.061, 0.150))
    print(popt)
    
    # popt = (0.060, 0.061, 0.150)
    num_points = 1000
    smooth_times = np.linspace(max(times), min(times), num_points)
    omega_low, omega_high, gamma = popt
    smooth_times_tile = np.tile(smooth_times, 3)
    calc_signal = concat_three_level(smooth_times_tile,
                                     omega_low, omega_high, gamma)
    ax.plot(smooth_times, calc_signal[0:num_points], label='0 fit')
    ax.plot(smooth_times, calc_signal[num_points:2*num_points], label='low fit')
    ax.plot(smooth_times, calc_signal[2*num_points:3*num_points], label='high fit')
    
    # print('coeffs: {}, {}, {}'.format(A_0, A_1, A_2))
    # print('rates: {}, {}, {}'.format(rate_0, rate_1, rate_2))
    
    ax.legend()
    

# %% Run


if __name__ == '__main__':
    
    plt.rcParams['text.latex.preamble'] = [
        r'\usepackage{physics}',
        r'\usepackage{sfmath}',
        r'\usepackage{upgreek}',
        r'\usepackage{helvet}',
       ]  
    plt.rcParams.update({'font.size': 15})
    plt.rcParams.update({'font.family': 'sans-serif'})
    plt.rcParams.update({'font.sans-serif': ['Helvetica']})
    plt.rc('text', usetex=True)
    
    folder = 'goeppert_mayer-nv7_2019_11_27-1662MHz-7deg'
    
    labels = ['0,0', '0,-1', '0,+1']
    files = [
        '2020_02_02-06_18_20-goeppert_mayer-nv7_2019_11_27',  # 0,0 run
        '2020_02_03-12_05_08-goeppert_mayer-nv7_2019_11_27',  # 0,-1 run
        '2020_02_01-00_31_35-goeppert_mayer-nv7_2019_11_27',  # 0,+1 run
        ]
       
    # main(folder, labels, files)
    
    # 0, -1, +1, mix
    # measured_pls = np.array([1.00, 0.80, 0.86, 0.85])
    measured_pls = np.array([1.00, 0.80, 0.86])
    solve_initials(measured_pls)
