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
open_file_name = '2019-05-06_14-22-00_ayrton12.txt'
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
        
    #    def gaussians(x,offset,a1,center1,sigma1, a2, center2, sigma2):
    #            return offset - a1*exp(-(x-center1)**2/(2*sigma1**2)) - a2*exp(-(x-center2)**2/(2*sigma2**2))
            
        plt.plot(freqs,norm_avg_counts)
        plt.show()  
        
        def func(freqs, *params):
            norm_avg_counts = numpy.zeros_like(freqs)
            for i in range(0, len(params), 3):
                ctr = params[i]
                amp = params[i+1]
                wid = params[i+2]
                norm_avg_counts = norm_avg_counts + amp * numpy.exp( -((freqs - ctr)/wid)**2)
            return norm_avg_counts
    
# %%
    
if __name__ == "__main__":
    
    fit_resonance('png')
    
    
