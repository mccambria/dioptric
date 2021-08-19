# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 14:34:37 2021

@author: agard
"""

from scipy.special import j1
import numpy
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import utils.tool_belt as tool_belt

wavelength = 638
NA = 1.3
pi = numpy.pi
v = 50 / (30 * NA**2 * pi /wavelength**2)**2
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

# %%

def plot_peak(peak, P, t, do_plot = False):
    dr = 150
    lin_r = numpy.linspace(peak - dr, peak + dr, 100)
    data = eta(lin_r,v, P, t)
    
    if do_plot:
        fig, ax = plt.subplots(1, 1)
        ax.plot(lin_r, data, 'bo')
        ax.set_xlabel('r (nm)')
        ax.set_ylabel('NV- probability')
    else:
        ax = []
    
    return lin_r, data, ax

def fit_gaussian_peak(peak, P, t, do_plot = False):
    ret_vals = plot_peak(peak, P, t, do_plot)
    lin_r, data, ax = ret_vals
    
    init_guess = [0.5, peak, peak/10, 0]
    fit_params, _ = curve_fit( tool_belt.gaussian, lin_r, data,
                                      p0 = init_guess)
    
    if do_plot:
        ax.plot(lin_r, tool_belt.gaussian(lin_r, *fit_params), 'r-')
    
    
    
    return fit_params

def vary_powers(peak, P_range, t):
    
    width_list = []
    power_list = []
    
    for P in numpy.linspace(P_range[0], P_range[1], 100):
        failed = True
        try:
            fit_params = fit_gaussian_peak(peak, P, t)
            failed = False
        except Exception:
            continue
        
        if not failed:
            width_list.append(fit_params[2])
            power_list.append(P)
            
    init_guess = [50, -0.5]
    fit_params, _ = curve_fit(power_law, power_list, width_list,
                                      p0 = init_guess)
    print(fit_params)
        
    
    fig, ax = plt.subplots(1, 1)
    lin_powers = numpy.linspace(power_list[0], power_list[-1], 100)
    ax.plot(power_list, width_list, 'bo')
    ax.plot(lin_powers, power_law(lin_powers, *fit_params), 'r-')
    ax.set_xlabel('Power (mW)')
    ax.set_ylabel(r'Gaussian fit width, $\sigma$ (nm)')
    
    eq_text = 'a * x ^ b'
    ax.text(0.75, 0.95, eq_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    text_fit = 'a={:.3f} \nb={:.3f}'.format(*fit_params)
    ax.text(0.05, 0.15, text_fit, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    return

def vary_duration(peak, P, t_range):
    width_list = []
    duration_list = []
    
    for t in numpy.linspace(t_range[0], t_range[1], 100):
        failed = True
        try:
            fit_params = fit_gaussian_peak(peak, P, t)
            failed = False
        except Exception:
            continue
        
        if not failed:
            width_list.append(fit_params[2])
            duration_list.append(t)
            
    init_guess = [50, -0.25]
    fit_params, _ = curve_fit(power_law, duration_list, width_list,
                                      p0 = init_guess)
    print(fit_params)
        
    
    fig, ax = plt.subplots(1, 1)
    lin_durations = numpy.linspace(duration_list[0], duration_list[-1], 100)
    ax.plot(duration_list, width_list, 'bo')
    ax.plot(lin_durations, power_law(lin_durations, *fit_params), 'r-')
    ax.set_xlabel('Duration (ms)')
    ax.set_ylabel(r'Gaussian fit width, $\sigma$ (nm)')
    
    eq_text = 'a * x ^ b'
    ax.text(0.75, 0.95, eq_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    text_fit = 'a={:.3f} \nb={:.3f}'.format(*fit_params)
    ax.text(0.05, 0.15, text_fit, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    return
# %%
def power_law(x, a, b):
    return a*x**b
    
    
def intensity_scaling(P):
    I = P*NA**2*pi/wavelength**2
    return I


def radial_scaling(r):
    x = 2*pi*NA*r/wavelength
    return x

def intensity_airy_func(r, P):
    I = intensity_scaling(P)
    x = radial_scaling(r)
    
    return I * (2*j1(x) / x)**2

def eta(r,v, P, t):
    return numpy.exp(-v*t*intensity_airy_func(r, P)**2)

# %%

# There is a fair bit of finess to the values to test. Too low, and the Gaussian 
# width is overestimated. Too low, and the peak is not there and the fit doesn't pick it up

vary_powers(300, [40, 340], 70)
vary_duration(300, 50, [50, 500]) # as we go to shorter times, the fits over estimate the width

# fit_gaussian_peak(300, 50, 70, do_plot = True)
