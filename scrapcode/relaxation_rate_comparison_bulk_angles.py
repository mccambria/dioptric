# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 14:31:49 2020

@author: Aedan
"""

import utils.tool_belt as tool_belt
import numpy
import matplotlib.pyplot as plt

base_file_path = 't1_double_quantum/paper_data/bulk_dq'

folder_48 = 'goeppert_mayer-nv7_2019_11_27-48deg'
folder_0 = 'goeppert_mayer-nv7_2019_11_27-0deg'

file_48 = '121MHz_splitting_rate_analysis'
file_0 = '187MHz_splitting_rate_analysis'



data_48 = tool_belt.get_raw_data(base_file_path + '/' + folder_48, file_48)

omega_time_values = data_48['zero_zero_time']
gamma_time_values = data_48['plus_plus_time']

omega_counts_48 = data_48['zero_relaxation_counts']
omega_ste_48 = data_48['zero_relaxation_ste']
gamma_counts_48 = data_48['plus_relaxation_counts']
gamma_ste_48 = data_48['plus_relaxation_ste']

data_0 = tool_belt.get_raw_data(base_file_path + '/' + folder_0, file_0)

omega_counts_0 = data_0['zero_relaxation_counts']
omega_ste_0 = data_0['zero_relaxation_ste']
gamma_counts_0 = data_0['plus_relaxation_counts']
gamma_ste_0 = data_0['plus_relaxation_ste']

fig, axes = plt.subplots(1,2, figsize=(10, 5))

ax = axes[0]

ax.errorbar(omega_time_values, omega_counts_48,
                        yerr = omega_ste_48, 
                        label = '48 deg',  fmt = 'o', color = 'blue')

ax.errorbar(omega_time_values, omega_counts_0,
                        yerr = omega_ste_0, 
                        label = '0 deg',  fmt = 'o', color = 'red')
ax.set_xlabel('Wait time (ms)')
ax.set_ylabel('Normalized counts')
ax.set_title('Omega')
ax.legend()

ax = axes[1]

ax.errorbar(gamma_time_values, gamma_counts_48,
                        yerr = gamma_ste_48, 
                        label = '48 deg',  fmt = 'o', color = 'blue')

ax.errorbar(gamma_time_values, gamma_counts_0,
                        yerr = gamma_ste_0, 
                        label = '0 deg',  fmt = 'o', color = 'red')

ax.set_xlabel('Wait time (ms)')
ax.set_ylabel('Normalized counts')
ax.set_title('Gamma')
ax.legend()

fig.canvas.draw()
fig.canvas.flush_events()
fig.tight_layout()


