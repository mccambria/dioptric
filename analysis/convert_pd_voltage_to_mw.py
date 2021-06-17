# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 10:40:42 2021

A file to take measured power data and corresponding voltages on the photodiode
and cast them into a linear conversion. This is used in 
tool_belt.calc_optical_power_mW

@author: gardill
"""

import numpy
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def linear(x, m ,b):
    return m*x + b
    
def plot(pd_voltage_list, mw_power_list, m ,b, color):
    fig, ax = plt.subplots(1,1, figsize = (8,8))
    pd_linspace = numpy.linspace(pd_voltage_list[0], pd_voltage_list[-1], 100)
    ax.plot(pd_voltage_list,mw_power_list, '{}o'.format(color))
    ax.plot(pd_linspace, linear(pd_linspace, m, b), '{}-'.format(color))
    ax.set_xlabel('photodiode voltage (V)')
    ax.set_ylabel('measured power (mW)')
    
    text = '\n'.join((r'$y = m*x + b$',
                      r'$m = $' + '%.3f'%(m),
                      r'$b = $' + '%.3f'%(b)))

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.25, 0.85, text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    return

def linear_fit(pd_voltage_list, mw_power_list):
    
    
    opti_params, cov_arr = curve_fit(linear, pd_voltage_list,
                                         mw_power_list)
    
    return opti_params

# %% Run the file


if __name__ == '__main__':
    
    # green
#    color = 'g'
#    mw_power_list = [0.0063, 0.0074, 0.0089, 0.011, 0.017, 0.09, 0.43,
#                           0.71, 0.99, 1.2, 1.5, 1.75, 2, 2.25, 2.5, 3.25, 6.89]
#    pd_voltage_list = [-0.0041, -0.0039, -.0039, -.0041, -.0037, -.00197, 0.0095, 
#                             0.021, 0.045, 0.043, 0.069, 0.068, 0.098, 0.109, 0.14, 0.172, 0.487]
    
    # red
#    color = 'r'
#    mw_power_list= [0.005, 0.014, 0.25, 1.4, 4.3, 4.9, 6.5, 7.4, 7.6, 7.7, 7.9, 7.0]
#    pd_voltage_list = [-0.0042, -0.0031, 0.0128, 0.092, 0.356, 0.459, 0.670, 0.868, 1.017, 1.107, 1.193, 1.053]
    
    # yellow
    color = 'y'
    mw_power_list =numpy.array([0.1, 0.7, 2.2, 4.3,5.7, 7.3, 7.7, 7.8,
                                0.3, 3.4, 11, 21, 31, 36, 37, 39,
                                1, 11, 39, 77, 105, 125, 134, 135, 
                                4, 40, 138, 255, 370, 430, 460, 470,])/10**3
    pd_voltage_list = [-0.0049, -0.005, -0.0049, -0.0046, -0.0049, -0.0046, -0.0046, -0.0046,
                       -0.0047, -0.0048, -0.0042, -0.0041, -0.0037, -0.0037, -0.0036, -0.0036,
                       -0.0046, -0.0042, -0.0034, -0.002, -0.0018, -0.0005, -0.00036, -0.0005,
                       -0.0045, -0.0034, -0.0007, 0.003, 0.006, 0.0081, 0.0093, 0.0089]
    
    m, b = linear_fit(pd_voltage_list, mw_power_list)
    
    plot(pd_voltage_list, mw_power_list, m ,b, color)