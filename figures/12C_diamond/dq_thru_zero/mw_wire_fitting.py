# -*- coding: utf-8 -*-
"""
Created on Jan 3 2023

@author: agardill
"""


import utils.tool_belt as tool_belt
import utils.kplotlib as kpl
from utils.kplotlib import KplColors
from numpy import pi
import numpy
import time
import matplotlib.pyplot as plt
from random import shuffle
import labrad
from utils.tool_belt import States
from scipy.optimize import curve_fit
from numpy.linalg import eigvals
import majorroutines.optimize as optimize
from utils.tool_belt import NormStyle

def cosine_fit(x, offset, amp, freq, phase):
    return offset + amp * numpy.cos(x* freq + phase)


def plot(x_data, y_data, title):
    kpl.init_kplotlib()
    
    # x_data = (10 - numpy.array(x_data))*18
    
    x_smooth = numpy.linspace(x_data[0], x_data[-1], 1000)
    
    # fit_func = lambda x, offset, amp, freq,phase: cosine_fit(x, offset, amp, freq, phase)
    # init_params = [ 0.5, 1, numpy.pi/180,  1]
    # popt, pcov = curve_fit(
    #     fit_func,
    #       x_data,
    #     y_data,
    #     # sigma=t2_sq_unc,
    #     # absolute_sigma=True,
    #     p0=init_params,
    # )
    # print(popt)
        

    # Plot setup
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('Y value (V)')
    ax.set_ylabel("Rabi period (ns)")
    ax.set_title(title)

    # Plotting
    kpl.plot_points(ax,  x_data, y_data, label = 'data', color=KplColors.BLACK)
    
    # kpl.plot_line(ax, x_smooth, fit_func(x_smooth,*popt ), label = 'fit', color=KplColors.RED)
    
    ax.legend()
            
    

z_vals =  [0.437, 0.387, 0.254, 0.140, 0.045, -0.178, -0.209, -0.414, -0.582] 
rabi_p = [128, 138, 129, 185, 178, 181, 184, 173, 194]

plot(z_vals, rabi_p, '',)


