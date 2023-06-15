# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 19:05:21 2023

@author: kolkowitz
"""

import utils.tool_belt as tool_belt
import numpy as numpy
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
from scipy.signal import find_peaks



def extract_oscillations(norm_avg_sig, precession_time_range, num_steps, detuning):
    kpl.init_kplotlib()
    # Create an empty array for the frequency arrays
    FreqParams = numpy.empty([3])

    # Perform the fft
    time_step = (precession_time_range[1]/1e3 - precession_time_range[0]/1e3) \
                                                    / (num_steps - 1)

    transform = numpy.fft.rfft(norm_avg_sig)
#    window = max_precession_time - min_precession_time
    freqs = numpy.fft.rfftfreq(num_steps, d=time_step)
    transform_mag = numpy.absolute(transform)
    # Plot the fft
    fig_fft, ax= plt.subplots(1, 1, figsize=(3, 3))
    kpl.plot_line(ax,freqs[1:35], transform_mag[1:35], color = KplColors.BLACK)  # [1:] excludes frequency 0 (DC component)
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('FFT magnitude')
    ax.set_xticks([0,4,8,12, 16])
    fig_fft.canvas.draw()
    fig_fft.canvas.flush_events()
    


    # Guess the peaks in the fft. There are parameters that can be used to make
    # this more efficient
    freq_guesses_ind = find_peaks(transform_mag[1:]
                                  , prominence = 0.5
#                                  , height = 0.8
#                                  , distance = 2.2 / freq_step
                                  )


    # Check to see if there are three peaks. If not, try the detuning passed in
    if len(freq_guesses_ind[0]) != 3:
        print('Number of frequencies found: {}'.format(len(freq_guesses_ind[0])))
#        detuning = 3 # MHz

        FreqParams[0] = detuning - 2.2
        FreqParams[1] = detuning
        FreqParams[2] = detuning + 2.2
    else:
        FreqParams[0] = freqs[freq_guesses_ind[0][0]]
        FreqParams[1] = freqs[freq_guesses_ind[0][1]]
        FreqParams[2] = freqs[freq_guesses_ind[0][2]]

    ax.axvline(detuning - 2.2)
    ax.axvline(detuning)
    ax.axvline(detuning + 2.2)
    return fig_fft, FreqParams

def fit_ramsey(norm_avg_sig,taus,  precession_time_range, FreqParams):

    kpl.init_kplotlib()
    print(FreqParams)
    taus_us = numpy.array(taus)/1e3
    # Guess the other params for fitting
    amp_1 = -0.04
    amp_2 = -0.03
    amp_3 = -0.04
    decay = 16
    offset = .91

    guess_params = (offset, decay, amp_1, FreqParams[0],
                        amp_2, FreqParams[1],
                        amp_3, FreqParams[2])
    # guess_params_double = (offset, decay,
    #                 # amp_1, FreqParams[0],
    #                     amp_2, FreqParams[1],
    #                     amp_3, FreqParams[2])

    # guess_params_fixed_freq = (offset, decay, amp_1,
    #                    amp_2,
    #                    amp_3, )
    # cosine_sum_fixed_freq = lambda t, offset, decay, amp_1,amp_2,  amp_3:tool_belt.cosine_sum(t, offset, decay, amp_1, FreqParams[0], amp_2, FreqParams[1], amp_3, FreqParams[2])

   # Try the fit to a sum of three cosines

    fit_func = tool_belt.cosine_sum
    init_params = guess_params

    # fit_func = cosine_sum_fixed_freq
    # init_params = guess_params_fixed_freq

   # fit_func = tool_belt.cosine_double_sum
   # init_params = guess_params_double

    popt,pcov = curve_fit(fit_func, taus_us, norm_avg_sig,
                     p0=init_params,
                       # bounds=([0, 0, -numpy.infty, -15,
                       #             # -numpy.infty, -15,
                       #             -numpy.infty, -15, ]
                       #         , [numpy.infty, numpy.infty,
                       #            numpy.infty, 15,
                       #             # numpy.infty, 15,
                       #             numpy.infty, 15, ]
                               # )
                      )
    print(popt)
    print(numpy.sqrt(numpy.diag(pcov)))
    # popt=init_params
    # popt=[ 0.8457807, -0.04351987, -0.03723193, -0.0392098 ]
    taus_us_linspace = numpy.linspace(precession_time_range[0]/1e3, precession_time_range[1]/1e3,
                          num=1000)

    fig_fit, ax = plt.subplots(1, 1, figsize=(4, 3))
    kpl.plot_line(
        ax, taus_us_linspace, fit_func(taus_us_linspace,*popt), color=KplColors.RED
    )
    ax.plot(taus_us, norm_avg_sig,'b.')
    ax.set_xlabel(r'Free precesion time, $\tau$ ($\mu$s)')
    ax.set_ylabel("Normalized fluorescence")



    return fig_fit




# file = '2023_04_09-14_12_48-johnson-nv0_2023_04_06'
# file = '2023_04_08-23_21_33-johnson-nv0_2023_04_06'
# file='2021_10_15-10_37_22-johnson-nv0_2021_10_08'
# file = '2023_01_24-12_41_54-E6-nv1'
# file = '2023_01_31-16_05_52-E6-nv1'
# file='2023_03_06-19_52_44-E6-nv1'
file='2022_07_30-11_35_20-johnson-nv1'
file_name = file + '.txt'
with open(file_name) as f:
    data = json.load(f)
    
norm_avg_sig = data['norm_avg_sig']
precession_time_range = data['precession_time_range']
num_steps = data['num_steps']
detuning = data['detuning']
taus = numpy.linspace(
    precession_time_range[0],
    precession_time_range[-1],
    num=num_steps,
)
_,freq_fft =extract_oscillations(norm_avg_sig, precession_time_range, num_steps, detuning)
# print(freq_fft)

FreqParams = [detuning-2.2 ,detuning,  detuning+2.2]
fit_ramsey(norm_avg_sig, taus,  precession_time_range, FreqParams)