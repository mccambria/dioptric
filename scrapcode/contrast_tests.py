# -*- coding: utf-8 -*-
"""
Predicts a contrast drop based on eigenstate mixing.

Created on Wed Jan 22 16:59:26 2020

@author: matth
"""


# %% Imports


import numpy
from numpy import pi
import matplotlib.pyplot as plt
import analysis.extract_hamiltonian as eh


# %% Constants


gmuB = 2.8/100  # gyromagnetic ratio in GHz / mT


# %% Main


# def main(mixing, zero_count_rate, contrast):
#     pm_count_rate = (1-contrast) * zero_count_rate
    
#     mixed_zero_count_rate = (1-2*mixing) * zero_count_rate
#     mixed_pm_count_rate = (1+mixing) * pm_count_rate
    
#     mixed_diff = mixed_zero_count_rate - mixed_pm_count_rate
#     mixed_contrast = mixed_diff / mixed_zero_count_rate
#     print(mixed_contrast)
    

def main(k_47, k_57, popt, polarization_eff):
    
    plot_mag_Bs = numpy.linspace(0.001, 500, 1000)  # mT
    mag_Bs = plot_mag_Bs * gmuB  # GHz
    
    contrasts = []
    
    # transition = 1  # Driving to LOW
    transition = 2  # Driving to HIGH

    for mag_B in mag_Bs:
        
        # vecs are returned zero, low, high
        # components are ordered +1, 0, -1
        vecs = eh.calc_eigenvectors(mag_B, *popt)
        
        zero_zero_comp = numpy.abs(vecs[0,1])**2
        zero_plus_comp = numpy.abs(vecs[0,0])**2
        zero_minus_comp = numpy.abs(vecs[0,2])**2
        zero_pm_comp = zero_plus_comp + zero_minus_comp
        
        tran_zero_comp = numpy.abs(vecs[transition,1])**2
        tran_plus_comp = numpy.abs(vecs[transition,0])**2
        tran_minus_comp = numpy.abs(vecs[transition,2])**2
        tran_pm_comp = tran_plus_comp + tran_minus_comp
    
        k_zero = zero_zero_comp * k_47 + zero_pm_comp * k_57
        k_tran = tran_zero_comp * k_47 + tran_pm_comp * k_57
        
        k_ref = polarization_eff * k_zero + (1-polarization_eff) * k_tran
        k_sig = (1-polarization_eff) * k_zero + polarization_eff * k_tran
        
        contrasts.append(1 - (k_ref / k_sig))
    
    fig, ax = plt.subplots(1, 1, figsize=(6,4))
    fig.set_tight_layout(True)
    contrasts = numpy.array(contrasts)
    ax.plot(plot_mag_Bs, contrasts * 100)
    ax.set_xlim(0, 500)
    ax.set_ylim(0, None)
    ax.set_xlabel('B field magnitude (mT)')
    ax.set_ylabel('Contrast (%)')


# %% Run the file


if __name__ == '__main__':
    
    # 0 and +1 shelving rates in us**-1
    k_47 = 10.8
    k_57 = 60.7
    popt = [74*(pi/180), 0, 0, 0, 0]
    polarization_eff = 0.7
    
    main(k_47, k_57, popt, polarization_eff)
