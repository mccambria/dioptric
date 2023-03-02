# -*- coding: utf-8 -*-
"""
Created on Feb 24 2023

@author: agardill
"""


import utils.tool_belt as tool_belt
import utils.kplotlib as kpl
from utils.kplotlib import KplColors
from numpy import pi
import numpy
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from utils.tool_belt import NormStyle

file_m30 = '2023_02_25-09_37_07-siena-nv0_2023_02_24'
file_0 = '2023_02_25-07_03_47-siena-nv0_2023_02_24'
file_30='2023_02_25-08_19_20-siena-nv0_2023_02_24'

data=tool_belt.get_raw_data(file_m30)
taus_m30 = data['taus']
sig_m30 = data['pulse_error_list']
sig_err_m30 = data['pulse_ste_list']

data=tool_belt.get_raw_data(file_0)
taus_0 = data['taus']
sig_0 = data['pulse_error_list']
sig_err_0 = data['pulse_ste_list']

data=tool_belt.get_raw_data(file_30)
taus_30 = data['taus']
sig_30 = data['pulse_error_list']
sig_err_30 = data['pulse_ste_list']

kpl.init_kplotlib()

    

# Plot setup
fig, ax = plt.subplots(1, 1)
ax.set_xlabel('Timing between compositi pulses (ns)')
ax.set_ylabel("Error")
# Plotting
kpl.plot_points(ax,  taus_m30, sig_m30, yerr = sig_err_m30, label = '-30 deg', color=KplColors.BLACK)
kpl.plot_points(ax,  taus_0, sig_0, yerr = sig_err_0, label = '0 deg', color=KplColors.RED)
kpl.plot_points(ax,  taus_30, sig_30, yerr = sig_err_30, label = '30 deg', color=KplColors.BLUE)


ax.legend()
    