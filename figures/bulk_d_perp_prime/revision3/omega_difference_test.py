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
    
ms = 7
lw = 1.75


# %% Constants


mixed_pl = 0.85   # relative pl at even mixture
contrasts = [1.00, 0.8, 0.86]  # 0, -1, +1

# infidelity_low = 3/20
# infidelity_high = 2/14
# infidelity_low = 0
# infidelity_high = 0
infidelity = 0.3333691048338927

# Calculated from solve_initials
# pol_eff_calc = 0.680343644216234
state_pls_calc = [1.0000030411789322, 0.7299961980701497, 0.8200007607509179]


# %% Functions


def exp_eq_omega(t, rate, amp):
    return  amp * exp(- rate * t)


def calc_lambda_parts(omega_minus, omega_plus, gamma):
    
    part_1 = omega_minus + omega_plus + gamma
    part_2 = omega_plus**2 + omega_minus**2 + gamma**2
    part_3 = (omega_plus*gamma) + (omega_minus*gamma) + (omega_minus*omega_plus)
    
    return part_1, part_2, part_3


def calc_lambda_plus(omega_minus, omega_plus, gamma):
    
    part_1, part_2, part_3 = calc_lambda_parts(omega_minus, omega_plus, gamma)
    return np.real(-part_1 + np.sqrt(part_2 - part_3))


def calc_lambda_minus(omega_minus, omega_plus, gamma):
    
    part_1, part_2, part_3 = calc_lambda_parts(omega_minus, omega_plus, gamma)
    return np.real(-part_1 - np.sqrt(part_2 - part_3))


def concat_three_level(time, omega_minus, omega_plus, gamma):
    
    # minus/low and plus/high are used interchangeably here
    
    return_signal = []
    
    lambda_plus = calc_lambda_plus(omega_minus, omega_plus, gamma)
    lambda_minus = calc_lambda_minus(omega_minus, omega_plus, gamma)
    print(lambda_plus)
    print(lambda_minus)
    # alpha = 0
    # c_1 = (1/(lambda_minus-lambda_plus))*((1/6)-(alpha/2)+((alpha-(1/3))*((2*omega_minus)+gamma+lambda_minus)))
    # c_2 = alpha - c_1 - (1/3)
    c_1 = -((3*omega_minus)+lambda_minus) / (3*(omega_minus-gamma)*(lambda_minus-lambda_plus))
    c_2 = ((3*omega_minus)+lambda_plus) / (3*(omega_minus-gamma)*(lambda_minus-lambda_plus))
    print(c_1)
    print(c_2)
    return
    
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
        print(time[ind])
        print(pops)
        return
        # Apply pi pulses
        
        # None
        if ind < len_time / 3:
            pops_pi_pulse = pops
        
        # Low
        elif ind < 2 * len_time / 3:
            pops_pi_pulse = [(pops[0]*infidelity)+(pops[1]*(1-infidelity)),
                              (pops[1]*infidelity)+(pops[0]*(1-infidelity)),
                              pops[2]]
            # pops_pi_pulse = [pops[1], pops[0], pops[2]]
        
        # High
        else:
            pops_pi_pulse = [(pops[0]*infidelity)+(pops[2]*(1-infidelity)),
                              pops[1],
                              (pops[2]*infidelity)+(pops[0]*(1-infidelity))]
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


def calculate_pls(infidelity, A_zero, A_low):
    
    # infidelity_high = 0.10
    # pol_eff = 0.6
    pol_zero = 1.0
    pol_rem = (1 - pol_zero) / 2
    pol_low = pol_rem
    pol_high = 1 - pol_zero - pol_low
    
    # pops = np.array([[pol_eff, pol_rem+pol_imb_low, pol_rem-pol_imb_low],
    #                  [pol_rem+pol_imb_low, pol_eff, pol_rem-pol_imb_low],
    #                  [pol_rem-pol_imb_low, pol_rem+pol_imb_low, pol_eff],
    #                  [1/3, 1/3, 1/3]])
    
    # pops = np.array([[pol_eff, pol_rem, pol_rem],
    #                   [(pol_eff*infidelity_low)+(pol_rem*(1-infidelity_low)),
    #                   (pol_rem*infidelity_low)+(pol_eff*(1-infidelity_low)),
    #                   pol_rem],
    #                   [(pol_eff*infidelity_high)+(pol_rem*(1-infidelity_high)),
    #                   pol_rem,
    #                   (pol_rem*infidelity_high)+(pol_eff*(1-infidelity_high))],
    #                   [1/3, 1/3, 1/3]])
    
    pops = np.array([[pol_zero, pol_rem, pol_rem],
                      [(pol_zero*infidelity)+(pol_rem*(1-infidelity)),
                      (pol_rem*infidelity)+(pol_zero*(1-infidelity)),
                      pol_rem],
                      [(pol_zero*infidelity)+(pol_rem*(1-infidelity)),
                      pol_rem,
                      (pol_rem*infidelity)+(pol_zero*(1-infidelity))]])
    
    # pops = np.array([[pol_eff, pol_rem, pol_rem],
    #                  [pol_rem, pol_eff, pol_rem],
    #                  [pol_rem, pol_rem, pol_eff],
    #                  [1/3, 1/3, 1/3]])
    
    # pops = np.array([[pol_zero, pol_low, pol_high],
    #                  [pol_low, pol_zero, pol_high],
    #                  [pol_high, pol_low, pol_zero],
    #                  [1/3, 1/3, 1/3]])
    
    # pops = np.array([[pol_zero, pol_low, pol_high],
    #                  [pol_low, pol_zero, pol_high],
    #                  [pol_high, pol_low, pol_zero]])
    
    # mixed_pl = (1/3)(A_zero + A_low + A_high)
    A_high = (3*mixed_pl) - A_zero - A_low
    
    pure_pls = np.array([A_zero, A_low, A_high])
    
    calculated_pls = np.matmul(pops, pure_pls)
    
    return calculated_pls


def solve_initials_objective(args, measured_pls):
    
    calculated_pls = calculate_pls(*args)
    diff = calculated_pls-measured_pls
    
    return np.dot(diff, diff)


def solve_initials(measured_pls):
    
    guess = brute(solve_initials_objective, 
                  ((0.0, 0.5), (0.9, 1.1), (0.3, 0.8)),
                  Ns=10, args=(measured_pls,))
    
    # print(guess)
    res = minimize(solve_initials_objective, guess, args=(measured_pls,))
    
    infidelity, A_zero, A_low = res.x
    # pol_low = (1 - pol_zero) / 2
    # pol_high = (1 - pol_zero) / 2
    A_high = (3*mixed_pl) - A_zero - A_low
    # pols = [pol_zero, pol_low, pol_high]
    pls = [A_zero, A_low, A_high]
    print(infidelity)
    print(pls)
    print(calculate_pls(*res.x))
    
    # test = [0.15, 1.12272697, 0.75909075, 0.66818257]
    # print(calculate_pls(*test))
    
    
def solve_initials_plot(measured_pls):
    
    num_points = 50
    pol_zero_linspace = np.linspace(0.5, 0.9, num_points)
    pol_low_linspace = np.linspace(0.0, 0.5, num_points)
    
    result = np.empty((num_points, num_points))
    
    for ind1 in range(num_points):
        for ind2 in range(num_points):
            
            pol_zero = pol_zero_linspace[ind1]
            pol_low = pol_low_linspace[ind2]
            
            pol_high = 1 - pol_zero - pol_low
            if (pol_high < 0) or (pol_high > 1):
                result[ind1, ind2] = None
                continue
                
            coeffs = [[pol_zero, pol_low, pol_high],
                      [pol_low, pol_zero, pol_high],
                      [pol_high, pol_low, pol_zero]]
            try:
                sol = np.linalg.solve(coeffs, measured_pls)
            except Exception:
                result[ind1, ind2] = None
                continue
            
            result[ind1, ind2] = np.mean(sol)
            
    fig, ax = plt.subplots()
    im = ax.pcolormesh(pol_zero_linspace, pol_low_linspace, result, vmax=1)
    fig.colorbar(im, ax=ax)
    
            

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
    
    
def supp_figure(folder, files):
    """
    This is adapted from Aedan's function of the same name in
    analysis/Paper Figures\Magnetically Forbidden Rate\supplemental_figures.py
    """
    
    source = 't1_double_quantum/data_folders/paper_data/bulk_dq/'
    path = source + folder
    
    marker_ind_offset = 1
    markers = ['^', 'o', 's', 'D', 'X']
    colors = ['#009E73', '#E69F00', '#0072B2', '#CC79A7', '#D55E00',]
    markerfacecolors = ['#ACECDB', '#f5b11d', '#72b5db', '#f0a3cd', '#fcbd8b',]
    labels = [r'$P_{0,0}-P_{0,-1}$', r'$P_{0,0}-P_{0,+1}$']
    
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    
    for ind in range(2):
        
        file = files[ind]
        
        data = tool_belt.get_raw_data(path, file)
        
        zero_relaxation_counts = data['zero_relaxation_counts']
        zero_relaxation_ste = np.array(data['zero_relaxation_ste'])
        zero_zero_time = data['zero_zero_time']
        
        plus_relaxation_counts = data['plus_relaxation_counts']
        plus_relaxation_ste = np.array(data['plus_relaxation_ste'])
        plus_plus_time = data['plus_plus_time']

        omega_opti_params = data['omega_opti_params']
        gamma_opti_params = data['gamma_opti_params']
        manual_offset_gamma = data['manual_offset_gamma']
                
        zero_zero_time = np.array(zero_zero_time)
        marker = markers[ind+marker_ind_offset]
        color = colors[ind+marker_ind_offset]
        facecolor = markerfacecolors[ind+marker_ind_offset]
        label = labels[ind]
        
        ax.errorbar(zero_zero_time, zero_relaxation_counts, yerr=zero_relaxation_ste,
                   label=label, zorder=5, marker=marker, ms=ms,
                   color=color, markerfacecolor=facecolor, linestyle='none')
        zero_time_linspace = np.linspace(0, 25.0, num=1000)
        ax.plot(zero_time_linspace, exp_eq_omega(zero_time_linspace, *omega_opti_params),
                color=color, linewidth=lw)
        
        omega_patch = mlines.Line2D([], [], label=r'$F_{\Omega}$', linewidth=lw,
                                   marker='^', markersize=ms,
                                   color=color, markerfacecolor=facecolor)
        
        # ax.tick_params(which = 'both', length=8, width=2, colors='k',
        #             direction='in',grid_alpha=0.7)
        ax.set_xlabel(r'Wait time $\tau$ (ms)')
        ax.set_ylabel(r'$F_{\Omega}$ (arb. units)')
        ax.set_xlim(-0.5, 20.5)
        # ax.set_yscale('log')
        
        ax.legend(handleheight=1.6, handlelength=0.6)
    

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
    
    # labels = ['0,0', '0,-1', '0,+1']
    # files = [
    #     '2020_02_02-06_18_20-goeppert_mayer-nv7_2019_11_27',  # 0,0 run
    #     '2020_02_03-12_05_08-goeppert_mayer-nv7_2019_11_27',  # 0,-1 run
    #     '2020_02_01-00_31_35-goeppert_mayer-nv7_2019_11_27',  # 0,+1 run
    #     ]
       
    # main(folder, labels, files)
    
    
    files = [
        '1662MHz_splitting_rate_analysis-minus',
        '1662MHz_splitting_rate_analysis-plus',
        ]
    
    supp_figure(folder, files)
    
    # 0, -1, +1, mix
    # measured_pls = np.array([1.00, 0.80, 0.86, 0.85])
    # measured_pls = np.array([1.00, 0.82, 0.88])
    # solve_initials(measured_pls)
    # solve_initials_plot(measured_pls)
    
    # print(calculate_pls(0.8, 1.5, 0.3))
