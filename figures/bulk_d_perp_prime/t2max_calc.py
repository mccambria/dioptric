# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 08:41:19 2020

Simple program to calculate T2, max and minimum B sensitivity

or bulk relaxation paper

@author: Aedan
"""

import numpy
# %%
# Calculated average values from 4 measurements at low axial-B field
gamma =  0.116863931 * 10**3  # Hz
omega = 0.058978022 * 10**3 # Hz

gamma_unc = 0.005112134 * 10**3 # Hz
omega_unc = 0.00209657 * 10**3 # Hz

#gamma =  0.118* 10**3  # Hz
#omega = 0.061 * 10**3 # Hz
#
#gamma_unc = 0.006 * 10**3 # Hz
#omega_unc = 0.003 * 10**3 # Hz

h_bar = 1.055 * 10**-34 # Js
mu_b = 2*9.27* 10 **-24 #J/T
g = 2

# %%

def calc_T2max_qubit(gamma, gamma_unc, omega, omega_unc):
    T2max = 2/(3*omega + gamma)
    
    T2max_unc = T2max * numpy.sqrt((3*omega_unc)**2 + gamma_unc**2) / (3*omega + gamma)
    
    return T2max, T2max_unc #s

def calc_T2max_qutrit(gamma, gamma_unc, omega, omega_unc):
    T2max = 2/(2*omega + 2*gamma)
    
    T2max_unc = T2max * numpy.sqrt((2*omega_unc)**2 + (2*gamma_unc)**2) / (2*omega + 2*gamma)
    
    return T2max, T2max_unc #s

def B_sensitivity(T2max, T2max_unc):
    B_sens = h_bar / (g * mu_b * numpy.sqrt(T2max))
    
    B_sens_unc = B_sens * 0.5 * T2max_unc / T2max
    
    return B_sens, B_sens_unc #T/sqrt(Hz)

# %%

T2max, T2max_unc = calc_T2max_qutrit(gamma, gamma_unc, omega, omega_unc)
#print(T2max*10**3)
#print(T2max_unc*10**3)

B_sens, B_sens_unc = B_sensitivity(T2max, T2max_unc)
print(B_sens * 10**12)
print(B_sens_unc* 10**12)

