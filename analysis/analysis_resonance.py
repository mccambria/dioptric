# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 10:42:07 2019

This script plots ESR data to a Gaussian fit. It guesses the frequencies of the
resonances, and uses those to curve fit to two gaussians.

It will not fit if it does not find one or two minimums.

To fix: fine tuning the width parameter. Scale the width parameter with the 
frequency range

@author: gardill
"""
# %% Imports

import os
import numpy
import json
from scipy import asarray as ar,exp
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# %% Call in file and pass in data

# %% Call the file and define the file format for saving in the end
open_file_name = '2019-04-30_16-44-41_ayrton12.txt'
save_file_type = ".png"

# %% Open the file with JSON


def fit_resonance(save_file_type):
        
    print('Select file \n...')
    
    from tkinter import Tk
    from tkinter import filedialog 
    
    root = Tk()
    root.withdraw()
    root.focus_force()
    open_file_name = filedialog.askopenfilename(initialdir = "G:/Team Drives/Kolkowitz Lab Group/nvdata/resonance", 
                title = 'choose file to replot', filetypes = (("svg files","*.svg"),("all files","*.*")) ) 
    
    if open_file_name == '':
        print('No file selected')
    else: 
        file_name_base = open_file_name[:-4]
        
        open_file_name = file_name_base + '.txt'  
        print('File selected: ' + file_name_base + '.svg')
        
    
    with open(open_file_name) as json_file:
        data = json.load(json_file)
        
        # Get information about the frequency
        freq_center = data["freq_center"]
        freq_range = data["freq_range"]
        num_steps = data["num_steps"]
        
        # Get the averaged, normalized counts from the ESR 
        norm_avg_counts = numpy.array(data["norm_avg_sig"])    
       
# %% Frequency array
    
    # Calculate some of the necessary values about the frequency scanned 
    half_freq_range = freq_range / 2
    freq_low = freq_center - half_freq_range
    freq_high = freq_center + half_freq_range
    freqs = numpy.linspace(freq_low, freq_high, num_steps)
       
# %% Define the gaussian function
    
    def gaus(x,offset,a1,center1,sigma1, a2, center2, sigma2):
            return offset - a1*exp(-(x-center1)**2/(2*sigma1**2)) - a2*exp(-(x-center2)**2/(2*sigma2**2))
        
## %% Guess the locations of the minimums
#            
#    minFreqGuess = []
#        
#    findPeaks = find_peaks(-norm_avg_counts + 1, width = 8, prominence = (0.07, None))
#    
#    minGuess = findPeaks[0]
#    
#    print(minGuess)
#    
#    if len(minGuess) < 1:
#        print("No local minimums found, cannot fit automatically")
#        
#    elif len(minGuess) > 2:
#        print("Too many minimums found, cannot fit automatically")
#        
#    else:
#        for i in minGuess:
#            
#            minFreqGuess.append(freq[i])
#    
#        print(minFreqGuess)
    
# %% If there are 1 or 2 guesses for the minimum, then fit a curve
        
    # Guess the st dev, contrast, and veritcal offset for the fitting           
    sigma = 0.01
    contrast = 0.1
    offset = 1
    minFreqGuess = [2.85, 2.88]
        
#    if len(minGuess) == 1:
#        popt,pcov = curve_fit(gaus, freq, norm_avg_counts, 
#                      p0=[offset, contrast, minFreqGuess[0], sigma,
#                          contrast, minFreqGuess[0], sigma])
#    elif len(minGuess) == 2:
    popt,pcov = curve_fit(gaus, freqs, norm_avg_counts, 
                      p0=[offset, contrast, minFreqGuess[0], sigma,
                          contrast, minFreqGuess[1], sigma])
   
        
# %% Plot the data itself and the fitted curve
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.plot(freqs, norm_avg_counts,'b',label='data')
    ax.plot(freqs, gaus(freqs,*popt),'r-',label='fit')
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('Contrast (arb. units)')
    ax.set_title('ESR (60\N{DEGREE SIGN})')
    ax.legend()
    text1 = "\n".join(("Fluorescent Contrast=" + "%.3f"%(popt[1]),
                      "Center Frequency=" + "%.3f"%(popt[2]) + " GHz",
                      "Std Deviation=" + "%.3f"%(popt[3]) + " GHz"))
    text2 = "\n".join(("Fluorescent Contrast=" + "%.3f"%(popt[4]),
                      "Center Frequency=" + "%.3f"%(popt[5]) + " GHz",
                      "Std Deviation=" + "%.3f"%(popt[6]) + " GHz"))
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    
    ax.text(0.05, 0.15, text1, transform=ax.transAxes, fontsize=12,
                            verticalalignment="top", bbox=props)
    ax.text(0.55, 0.15, text2, transform=ax.transAxes, fontsize=12,
                            verticalalignment="top", bbox=props)

    
    fig.canvas.draw()
    fig.set_tight_layout(True)
    fig.canvas.flush_events()
    
    # Save the file in the same file directory
#    fig.savefig(open_file_name + 'replot' + save_file_type)
    
# %%
    
if __name__ == "__main__":
    
    fit_resonance('png')
    
