# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 10:06:12 2019

@author: Aedan
"""
import matplotlib.pyplot as plt


nv2_rates = [33.0, 32.3, 35.0, 28.9, 30, 33, 32.9, 28.9, 30.4]
nv2_error = [0.7, 0.9, 1.1, 1.0, 2, 1, 0.7, 0.7, 0.9]
time = [0, 5, 10, 15, 20, 25, 30, 35, 40]

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.errorbar(time, nv2_rates, yerr = nv2_error, 
            label = r'$\gamma$', fmt='o', markersize = 10,color='blue')

ax.tick_params(which = 'both', length=6, width=2, colors='k',
                grid_alpha=0.7, labelsize = 18)

ax.tick_params(which = 'major', length=12, width=2)

ax.grid()

plt.xlabel('Time (hours)', fontsize=18)
plt.ylabel('Relaxation Rate (kHz)', fontsize=18)
plt.title(r'NV2', fontsize=18)
ax.legend(fontsize=18)