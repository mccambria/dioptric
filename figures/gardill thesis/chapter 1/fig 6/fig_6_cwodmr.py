# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 13:42:44 2023

@author: kolkowitz
"""


import utils.tool_belt as tool_belt
import numpy as np
import os
import time
import labrad
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from random import shuffle
from utils.tool_belt import States
import utils.kplotlib as kpl
from utils.kplotlib import KplColors
from utils.kplotlib import Size
import copy
import matplotlib.pyplot as plt
import majorroutines.optimize as optimize
import json


def gaussian(freq, constrast, sigma, center):
    return constrast * np.exp(-((freq - center) ** 2) / (2 * (sigma**2)))


def doub_gaus(freqs,  c1, a1, s1, c2, a2, s2):
    gauss1 = gaussian(freqs, a1, s1, c1)
    gauss2 = gaussian(freqs, a2, s2, c2)
    return 1.0 - gauss1  -gauss2

def calculate_freqs(freq_range, freq_center, num_steps):
    half_freq_range = freq_range / 2
    freq_low = freq_center - half_freq_range
    freq_high = freq_center + half_freq_range
    return np.linspace(freq_low, freq_high, num_steps)

def fit_resonance(
    freq_range,
    freq_center,
    num_steps,
    norm_avg_sig,
    guess_centers,
    norm_avg_sig_ste=None,
    ref_counts=None,
    do_single_gaussian= False
):
    print(do_single_gaussian)
    if not do_single_gaussian:
        fit_func = lambda freqs, c1, a1, s1, c2, a2, s2: doub_gaus(freqs,  c1, a1, s1, c2, a2, s2)
    
    
        guess_params = [guess_centers[0], 0.2, 0.01,
                        guess_centers[1], 0.2, 0.01,
                        ]
    else:
        fit_func = lambda freqs, a1, s1, c1 : gaussian(freqs,  a1, s1, c1)
    
    
        guess_params = [2.87, 0.01, 2.87
                        ]
    freqs = calculate_freqs(freq_range, freq_center, num_steps)

    popt, pcov = curve_fit(
        fit_func,
        freqs,
        norm_avg_sig,
        p0=guess_params,
    )
    # popt=guess_params
    # print(popt)
    # print('{:.5f}'.format(popt[0]))
    # print(popt[1])
    # print('{:.5f}'.format(np.sqrt(pcov[0][0])))

    return fit_func, popt, pcov

def do_plot(file_list, guess_centers_list):
    
        
    kpl.init_kplotlib()

    fig, ax = plt.subplots(1, 1, figsize=(4,4))
    
    
    ni = 0
    for i in range(len(file_list)):
        file = file_list[i]
        do_single_gaussian=False
        if file == '023_04_10-14_35_56-johnson-nv0_2023_04_06': # for the single resonance, just fit a single gaussian
            do_single_gaussian = True
            
        guess_centers = guess_centers_list[i]
            
        file_name = file + '.txt'
        with open(file_name) as f:
            data = json.load(f)
            
        
        freq_center = data['freq_center']
        freq_range = data['freq_range']
        num_steps = data['num_steps']
        norm_avg_sig = np.array(data['norm_avg_sig'])
        # norm_avg_sig_ste=  np.array(data['norm_avg_sig_ste'])
        
        half_freq_range = freq_range / 2
        freq_low = freq_center - half_freq_range
        freq_high = freq_center + half_freq_range
        freqs = np.linspace(freq_low, freq_high, num_steps)
        
        smooth_freqs = np.linspace(freq_low, freq_high, 1000)
        
        fit_func, popt , _= fit_resonance(
            freq_range,
            freq_center,
            num_steps,
            norm_avg_sig,
            guess_centers,
            # norm_avg_sig_ste,
            do_single_gaussian
            )
        
            

        kpl.plot_line(
            ax, smooth_freqs, fit_func(smooth_freqs,*popt) + ni, color=KplColors.RED
        )
        
        # kpl.plot_points(ax, freqs, norm_avg_sig + ni,  color=KplColors.BLUE, size=Size.TINY)
        ax.plot(freqs, norm_avg_sig + ni,  '.', color = KplColors.BLUE)
        ni = ni+0.2
        
    ax.set_ylabel("Normalized fluorescence")
    ax.set_xlabel(r"Microwave frequency, $\nu$ (GHz)")
    
file_160='2023_04_10-16_29_00-johnson-nv0_2023_04_06'
file_118='2023_04_10-16_37_42-johnson-nv0_2023_04_06'
file_100='2023_04_10-16_46_23-johnson-nv0_2023_04_06'
file_84='2023_04_10-16_55_01-johnson-nv0_2023_04_06'
file_70='2023_04_10-17_03_40-johnson-nv0_2023_04_06'

pesr_file_list = [file_70, file_84, file_100,file_118,  file_160,  ]
# pesr_file_list = [file_70, file_925,  file_160,  ]

guess_centers_list = [[2.87, 2.87], [2.85, 2.883], [2.83, 2.9], [2.83, 2.91],  [2.82, 2.92],]
do_plot(pesr_file_list, guess_centers_list)