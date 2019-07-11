# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 10:42:07 2019

This script fits a sum of three cosines to the data taken from a ramsey 
measruement.

@author: gardill
"""
# %% Imports

import numpy
import json
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from numpy import exp

# %% Define the fitting function
    
def cosine_sum(t, offset, decay, amp_1, freq_1, amp_2, freq_2, amp_3, freq_3):
    two_pi = 2*numpy.pi
    
    return offset + numpy.exp(-t / abs(decay)) * (
                amp_1 * numpy.cos(two_pi * freq_1 * t) +
                amp_2 * numpy.cos(two_pi * freq_2 * t) +
                amp_3 * numpy.cos(two_pi * freq_3 * t))

# %%
        
def fit_ramsey(save_file_type):
    
    FreqGuess = numpy.empty([3])
    
    detuning = 3 # MHz
    
    FreqGuess[0] = detuning - 2.2
    FreqGuess[1] = detuning
    FreqGuess[2] = detuning + 2.2
    
    
    folder_dir = 'E:/Shared drives/Kolkowitz Lab Group/nvdata/ramsey/branch_ramsey2/'
    file_name = '2019-07-11_13-51-17_johnson1.txt'
    
    open_file_name = '{}{}'.format(folder_dir, file_name)
    
    with open(open_file_name) as json_file:
        data = json.load(json_file)
        
        # Get information about the frequency
        precession_time_range = data["precession_time_range"]
        num_steps = data["num_steps"]
        
        # Get the averaged, normalized counts from the ESR 
        norm_avg_counts = numpy.array(data["norm_avg_sig"])  
#        sig_counts = numpy.array(data['sig_counts'])
#        ref_counts = numpy.array(data['ref_counts'])
#        norm_avg_counts = sig_counts[1,:] / ref_counts[1,:]  
       
# %% Fourier transform the data to obtain the frequencies
    
    # Calculate some of the necessary values about the frequency scanned 
    min_precession_time = precession_time_range[0] / 10**3
    max_precession_time = precession_time_range[1] / 10**3
    
    taus = numpy.linspace(min_precession_time, max_precession_time,
                          num=num_steps)
       
    
    # Guess the params for fitting           
    amp_1 = 0.3
    amp_2 = amp_1
    amp_3 = amp_1
    decay = 1
    offset = 1
    
    guess_params = (offset, decay, amp_1, FreqGuess[0], 
                        amp_2, FreqGuess[1], 
                        amp_3, FreqGuess[2])
    
    
    try:
        popt,pcov = curve_fit(cosine_sum, taus, norm_avg_counts, 
                      p0=guess_params)
    except Exception: 
        print('Something went wrong!')
        popt = guess_params
    
    taus_linspace = numpy.linspace(min_precession_time, max_precession_time,
                          num=1000)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.plot(taus, norm_avg_counts,'b',label='data')
    ax.plot(taus_linspace, cosine_sum(taus_linspace,*popt),'r-',label='fit')
    ax.set_xlabel('Free precesion time (us)')
    ax.set_ylabel('Contrast (arb. units)')
    ax.legend()
    text1 = "\n".join(("Decay=" + "%.3f"%(popt[1]) + 'us',
                      "Frequency 1 =" + "%.2f"%(popt[3]) + " MHz",
                      "Frequency 2 =" + "%.2f"%(popt[5]) + " MHz", 
                      "Frequency 3 =" + "%.2f"%(popt[7]) + " MHz"))
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    
    ax.text(0.05, 0.15, text1, transform=ax.transAxes, fontsize=12,
                            verticalalignment="top", bbox=props)
        
   
        
# %% Plot the data itself and the fitted curve
    
    fig.canvas.draw()
    fig.set_tight_layout(True)
    fig.canvas.flush_events()
    
    # Save the file in the same file directory
    no_ext = open_file_name.split('.')[0]
    fig.savefig('{}_fit.{}'.format(no_ext, save_file_type))
    
# %%
    
if __name__ == "__main__":
    
    fit_ramsey('svg')
    
