# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 12:00:14 2019

@author: Aedan
"""

import numpy
import matplotlib.pyplot as plt
import utils.tool_belt as tool_belt
from scipy.optimize import curve_fit

# %%

def txt_read(file_name):
    folder = 'E:/Shared drives/Kolkowitz Lab Group/nvdata/spectra/Brar/Y2O3/Graphene1'

    wavelength_list = []
    counts_list = []

# Read in the wavelengths and counts, save to lists
    f = open(folder + '/' + file_name + '.txt', 'r')
    
    f_lines = f.readlines()
    for line in f_lines:
        wavelength, counts = line.split()
        wavelength_list.append(float(wavelength))
        counts_list.append(float(counts))
        
    return numpy.array(wavelength_list), numpy.array(counts_list)
    

def creat_fig(low_wavelength_ind, upper_wavelength_ind, y_lim,
              wavelengths, data_arrays, labels, number_of_subplots):
    
    fig, axes = plt.subplots(1, number_of_subplots, figsize=(15, 8))
    
    for i in range(number_of_subplots):
        ax = axes[i]
        counts = data_arrays[i]
        if i % 2 == 0 :
            format = 'r-'
        else:
            format = 'b-'
        ax.plot(wavelengths[low_wavelength_ind:upper_wavelength_ind],
                counts[low_wavelength_ind:upper_wavelength_ind], format, 
                label = labels[i])
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Counts (arb.)')
        ax.set_ylim(y_lim)
        ax.legend()
        
    return fig
        
        
# %%

if __name__ == '__main__':
    a_1= '2019_11_13_18_38_48-Y2O3-high-550-1'
    b_1 = '2019_11_13_18_33_59-Y2O3-high-670-1'
    
    a_2= '2019_11_13-18_45_56-Y2O3-low-550-2'
    b_2 = '2019_11_13-18_50_33-Y2O3-low-670-2'
    
    a_3= '2019_11_13-18_58_01-Y2O3-high-550-3'
    b_3 = '2019_11_13-19_02_29-Y2O3-high-670-3'
    
    a_4= '2019_11_13-19_16_29-Y2O3-low-550-4'
    b_4 = '2019_11_13-19_10_28-Y2O3-low-670-4'
    
    bckd_550 = '550_background_01'
    bckd_670 = '670_background_01'
    
    wavelength_550, bckd_550_counts = txt_read(bckd_550)
    wavelength_670, bckd_670_counts = txt_read(bckd_670)
    
    _, a_1_counts = txt_read(a_1)
    _, b_1_counts = txt_read(b_1)
    _, a_2_counts = txt_read(a_2)
    _, b_2_counts = txt_read(b_2)
    _, a_3_counts = txt_read(a_3)
    _, b_3_counts = txt_read(b_3)
    _, a_4_counts = txt_read(a_4)
    _, b_4_counts = txt_read(b_4)
    
    a_1_norm = (a_1_counts - bckd_550_counts)*1
    b_1_norm = b_1_counts - bckd_670_counts
    
    a_2_norm = (a_2_counts - bckd_550_counts)*1.08
    b_2_norm = b_2_counts - bckd_670_counts
    
    a_3_norm = (a_3_counts - bckd_550_counts)*0.994
    b_3_norm = b_3_counts - bckd_670_counts
    
    a_4_norm = (a_4_counts - bckd_550_counts)*0.887
    b_4_norm = b_4_counts - bckd_670_counts

    a_counts = [a_1_norm, a_2_norm, a_3_norm, a_4_norm]
    b_counts = [b_1_norm,b_2_norm,b_3_norm,b_4_norm]
    
    labels = ['4.2 V (1)', '-3.5 V (2)', '4.2 V (3)', '-3.5 V (4)']
    
    fig_550 = creat_fig(170, 510, [-20,175],
              wavelength_550, a_counts, labels, 4)
    
    fig_550.tight_layout()
#    
    fig_670 = creat_fig(170, 510, [-10,60],
              wavelength_670, b_counts, labels, 4)
    
    fig_670.tight_layout()
    
#    init_guess = [10, 662, 1, 5]
#    popt,pcov = curve_fit(tool_belt.gaussian, wavelength_670[346:401], 
#                          b_4_norm[346:401], p0=init_guess)
#    linspace_wavelength = numpy.linspace(660,664,1000)
#    
#    print(popt[0]**2 + popt[3])
#    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
#    
#    ax.plot(wavelength_670[346:401], b_4_norm[346:401], 'r.')
#    ax.plot(linspace_wavelength, tool_belt.gaussian(linspace_wavelength,*popt), 
#            'b-')
    
