# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 11:24:37 2019

This file plots the center frequency for the nv13_2019_06_10 vs the splitting.

@author: Aedan
"""
# %%
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# %%
def fit_eq(f, offset, amp, power):
    return offset + amp * f**(power)

# %%
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

splitting_list = [29.8, 51.9, 72.4, 112.9, 164.1]

cent_freq = [2.8396, 2.84335, 2.8444, 2.85125, 2.86775 ]

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.plot(splitting_list, cent_freq, 'bo')


ax.set_xlabel('Splitting (MHz)')
ax.set_ylabel('Center Frequency (GHz)')
ax.set_title('Center frequency shift vs splitting, nv13_2019_06_10')
ax.legend()