# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 09:18:36 2021

@author: kolkowitz
"""

import utils.tool_belt as tool_belt
import majorroutines.optimize as optimize
import numpy
import matplotlib.pyplot as plt
from random import shuffle
import majorroutines.image_sample as image_sample
import copy
import scipy.stats as stats


file_base = 'pc_rabi/branch_Spin_to_charge/moving_target_siv_init/2021_03'
file_none = '2021_03_02-16_47_29-goeppert-mayer-nv5_2021_03_01'
file_dark = '2021_03_02-22_23_50-goeppert-mayer-nv5_2021_03_01'
file_bright = '2021_03_02-22_23_55-goeppert-mayer-nv5_2021_03_01_1'

data = tool_belt.get_raw_data(file_base, file_none)
counts_none = data['readout_counts_avg']
counts_ste_none = data['readout_counts_ste']
data = tool_belt.get_raw_data(file_base, file_dark)
counts_dark = data['readout_counts_avg']
counts_ste_dark = data['readout_counts_ste']
rad_dist = numpy.array(data['rad_dist'])
pulse_time = data['pulse_time']


data = tool_belt.get_raw_data(file_base, file_bright)
pulse_coords_list = data['pulse_coords_list']
readout_counts_array = data['readout_counts_array']

zipped_coords = list(zip(pulse_coords_list,readout_counts_array ))
zipped_coords.sort(key=lambda x: x[0][0])

sorted_coords, sorted_readout_array = list(zip(*zipped_coords))

bright_counts_array = []
for i in range(len(sorted_readout_array)):
    bright_counts_array.append(sorted_readout_array[i][:-5])
print(bright_counts_array)
counts_bright = numpy.average(bright_counts_array, axis=1)
counts_ste_bright = stats.sem(bright_counts_array, axis=1)

fig1, ax = plt.subplots(1, 1, figsize=(10, 10))
#    ax.errorbar(rad_dist*35, counts_none,yerr = counts_ste_none, fmt = 'k--', label = 'No reset')
ax.errorbar(rad_dist*35, counts_dark, yerr = counts_ste_dark,fmt = 'b-', label = 'SiV dark reset')
ax.errorbar(rad_dist*35, counts_bright, yerr = counts_ste_bright, fmt='r-', label = 'SiV bright reset')
ax.set_xlabel('Distance from readout point (um)')
ax.set_ylabel('Counts')
ax.set_title('Moving target measurement with and without resetting SiV state')
ax.legend()

fig2, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.plot(rad_dist*35, counts_none, 'k--', label = 'No reset')
ax.plot(rad_dist*35, counts_dark, 'b-', label = 'SiV dark reset')
ax.plot(rad_dist*35, counts_bright, 'r-', label = 'SiV bright reset')
ax.set_xlabel('Distance from readout point (um)')
ax.set_ylabel('Counts')
ax.set_title('Moving target measurement with and without resetting SiV state')
ax.legend()