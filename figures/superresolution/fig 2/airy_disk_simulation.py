# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 14:34:37 2021

time in ms, power in mW

@author: agard
"""

from scipy.special import j1
from scipy.special import jv
import numpy
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import utils.tool_belt as tool_belt
import random

wavelength = 638
NA = 1.3  
pi = numpy.pi
# v = 50 / (30 * NA**2 * pi /wavelength**2)**2 # I should measure the actual value here
v = 1.6e10 #nm^4 / (ms mW^2)
fwhm =2.355

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)


# %%

def exp_func(t,g):
    return numpy.exp(-g*t)
def quad_func(I, v):
    return v*I**2
    
def estimate_ion_rate():    # try to fit to data

    n_1 = [0.9, 0.5, 0.2, 0.01] # near the first dip
    t_1 = [1, 5, 10, 50]
    r_1 = 350 # nm 
    
    n_2 = [0.8, 0.2, 0.1] # first dip
    t_2 = [1, 5, 10]
    r_2 = 401.476 # nm 5.14
    
    n_3 = [0.9, 0.8, 0.4,0.01] # second dip
    t_3 = [10, 50, 100, 500]
    r_3 = 657.67 # nm 8.42
    
    P = 18.4
    # print(intensity_airy_func(r_1, P))
    
    n_list = [n_1, n_2, n_3]
    t_list = [t_1, t_2, t_3]
    r_list = [r_1, r_2, r_3]
    
    
    intensity_list = []
    g_list = []
    
    do_fit = True
    
    if do_fit:
        for i in range(len(r_list)):
            r= r_list[i]
            n = n_list[i]
            t = t_list[i]
            
            init_fit = [1e-1]
            t_linspace = numpy.linspace(t[0], t[-1], 100)
            
            # eta_n = lambda t, v: eta(r,v, P, t)
            opti_params, cov_arr = curve_fit(exp_func,
                  t ,n,p0=init_fit)
            print(opti_params)
            I = intensity_airy_func(r, P, 0)
            intensity_list.append(I)
            g_list.append(opti_params[0])
            
            fig, ax = plt.subplots()
            ax.plot(t_linspace, exp_func(t_linspace, *opti_params))
            ax.plot(t, n, 'ro')
            ax.set_xlabel('time (ms)')
            ax.set_ylabel('NV- population')
        
        init_fit = [1e10]
        opti_params, cov_arr = curve_fit(quad_func,
              intensity_list ,g_list, p0=init_fit)
        i_min = min(intensity_list)
        i_max = max(intensity_list)
        int_linspace = numpy.linspace(i_min, i_max, 100)
        
        fig, ax = plt.subplots()
        ax.plot(int_linspace, quad_func(int_linspace, *opti_params))
        ax.plot(intensity_list, g_list, 'ro')
        ax.set_xlabel(r'Intensity (mW/nm$^2$)')
        ax.set_ylabel('NV- ionization rate (1/ms)')
        
        text = 'NV- ionization rate scaling, v = {:.3f}\n'.format(opti_params[0]) + r'mW$^2$/(ms nm$^4$)'
        ax.text(0.2, 0.2, text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)
    
# %%

# def plot_peak(peak, P, t, do_plot = False):
#     dr = 150
#     lin_r = numpy.linspace(peak - dr, peak + dr, 100)
#     data = eta(lin_r,v, P, t)
    
#     if do_plot:
#         fig, ax = plt.subplots(1, 1)
#         ax.plot(lin_r, data, 'bo')
#         ax.set_xlabel('r (nm)')
#         ax.set_ylabel('NV- probability')
#     else:
#         ax = []
    
#     return lin_r, data, ax

def plot_broadened_peak(peak, P, t, broadened_val = 0, I_bkgd_perc = 0, do_plot = False):
    dr = 150
    lin_r = numpy.linspace(peak - dr, peak + dr, dr+1)
    data_original = eta(lin_r,v, P, t, I_bkgd_perc)
    num_rand = 100 #number of times it repeats the data
    repeatability = broadened_val #max +/- shift to the data
    

    if broadened_val != 0:
        # calc random shifts in the data
        rand_vals = numpy.random.normal(0, repeatability, num_rand)
        rand_shifts = numpy.round(rand_vals)
        data_out = numpy.array(data_original)*0
        for i in rand_shifts: #for each random shift, either add empty rows to the start or end of data list
            data_shift = data_original.tolist()
            if i > 0:
                data_shift = [0]*int(i) + data_shift[:-int(i)]
            elif i < 0:
                data_shift = data_shift[-int(i):] + [0]*-int(i)
            
            data_new = data_out + data_shift
            data_out = data_new
            
        data_out = data_out / num_rand
    else:
        data_out = data_original
    # i = 0
    # shift_list = [0 for el in range(i)]
    # shifted_data = shift_list + data.tolist()
    # data = data.tolist() + shift_list
    # new_data = (numpy.array(data) + numpy.array(shifted_data))/2
    
    if do_plot:
        fig, ax = plt.subplots(1, 1)
        ax.plot(lin_r, data_out, 'bo')
        ax.set_xlabel('r (nm)')
        ax.set_ylabel('NV- probability')
    else:
        ax = []
    
    return lin_r, data_out, ax

def fit_gaussian_peak(peak, P, t, broadened_val = 0, I_bkgd_perc = 0, do_plot = False):
    # ret_vals = plot_peak(peak, P, t, do_plot)
    ret_vals = plot_broadened_peak(peak, P, t, broadened_val, I_bkgd_perc, do_plot)
    lin_r, data, ax = ret_vals
    
    init_guess = [0.5, peak, peak/10, 0]
    fit_params, cov_arr = curve_fit( tool_belt.gaussian, lin_r, data,
                                      p0 = init_guess)
    
    if do_plot:
        print(abs(fit_params[2]*fwhm))
        print(fit_params)
        ax.plot(lin_r, tool_belt.gaussian(lin_r, *fit_params), 'r-')
        
        eq_text = 'FWHM = {:.3f} +/- {:.3f} nm'.format(abs(fit_params[2]*fwhm), cov_arr[2][2]*fwhm)
        ax.text(0.5, 0.95, eq_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)
        
    
    
    
    return fit_params

def vary_powers(peak, P_range, t, broadened_val = 0,):
    
    width_list = []
    power_list = []
    
    # for P in numpy.linspace(P_range[0], P_range[1], 50):
    for P in numpy.logspace(numpy.log10(P_range[0]), numpy.log10(P_range[1]), 50):
        failed = True
        try:
            fit_params = fit_gaussian_peak(peak, P, t, broadened_val, False)
            failed = False
        except Exception:
            continue
        
        if not failed:
            width_list.append(abs(fit_params[2]))
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
    ax.set_yscale('log')
    ax.set_xscale('log')
    
    eq_text = 'a * x ^ b'
    ax.text(0.75, 0.95, eq_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    text_fit = 'a={:.3f} \nb={:.3f}'.format(*fit_params)
    ax.text(0.05, 0.15, text_fit, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    return power_list, width_list

def vary_duration_width(peak, P, t_range, broadened_val = 0, I_bkgd_perc = 0):
    width_list = []
    duration_list = []
    
    # for t in numpy.linspace(t_range[0], t_range[1], 100):
    for t in numpy.logspace(numpy.log10(t_range[0]), numpy.log10(t_range[1]), 100):
        failed = True
        try:
            fit_params = fit_gaussian_peak(peak, P, t, broadened_val, I_bkgd_perc)
            failed = False
        except Exception:
            continue
        
        if not failed:
            width_list.append(abs(fit_params[2]))
            duration_list.append(t)
            
    fig, ax = plt.subplots(1, 1)
    lin_durations = numpy.linspace(duration_list[0], duration_list[-1], 100)
    ax.plot(duration_list, numpy.array(width_list)*fwhm, 'bo')
    
    ### Fitting ###
    ### Modified scaling with background intensity and broadening
    C = bessel_scnd_der(radial_scaling(peak)) # get the constant term based on bessel functions
    bound_vals = (0, numpy.infty)
    
    # fit function for background, no broadening
    if broadened_val ==0 and I_bkgd_perc != 0:
        fit_func = lambda t, e, a: width_scaling_w_mods(t, C, e, a, 0)
        init_guess = [I_bkgd_perc, 28]
        text = r'Modified $t^{-1/4}$ scaling without broadening'
        bound_vals = ([0,0], [1,numpy.infty])
    # fit function for broadening, no background
    elif broadened_val !=0 and I_bkgd_perc == 0:
        fit_func = lambda t, a, R: width_scaling_w_mods(t, C, 0, a, R)
        init_guess = [28, numpy.sqrt(broadened_val)]
        text = r'Modified $t^{-1/4}$ scaling without nonperfect zero'
        bound_vals = ([0, 0], [numpy.infty, numpy.infty])
    # fit function for broadening and background
    elif broadened_val !=0 and I_bkgd_perc != 0:
        fit_func = lambda t, e, a, R: width_scaling_w_mods(t, C, e, a, R)
        init_guess = [I_bkgd_perc, 28, numpy.sqrt(broadened_val)]
        text = r'Modified $t^{-1/4}$ scaling with broadening and nonperfect zero'
        bound_vals = ([0,0, 0], [1,numpy.infty, numpy.infty])
    # fit function for no broadening or background (inverse quarter scaling)
    elif broadened_val ==0 and I_bkgd_perc == 0:
        fit_func = lambda x, a: power_law(x, a, (-1/4))
        init_guess = [50]
        text = r'$t^{-1/4}$ scaling'

    
    
    print(fit_params)
    if broadened_val !=0 and I_bkgd_perc != 0: #print some info in test case
        print('Fit value for broadening: {} nm'.format(fit_params[2]**2))
        print('Input value for broadening: {} nm'.format(broadened_val))
        print('Fit value for background intensity: {}'.format(fit_params[0]))
        print('Input value for background intensity: {}'.format(I_bkgd_perc))
    
    if broadened_val ==0 and I_bkgd_perc != 0: #print some info in test case
        print('Fit value for background intensity: {}'.format(fit_params[0]))
        print('Input value for background intensity: {}'.format(I_bkgd_perc))
    ax.plot(lin_durations, fit_func(lin_durations, 0.005, 30)*fwhm, 'r-')
    
    #inverse quarter
    fit_func = lambda x, a: power_law(x, a, (-1/4))
    init_guess = [50]
    text = r'$t^{-1/4}$ scaling'
    fit_params, _ = curve_fit(fit_func, duration_list, width_list,
                                      p0 = init_guess)
    ax.plot(lin_durations, fit_func(lin_durations, *fit_params)*fwhm, 'k-')
    
    
    ax.set_xlabel('Duration (ms)')
    ax.set_ylabel(r'FWHM (nm)')
    ax.set_yscale('log')
    ax.set_xscale('log')
    
    ax.text(0.05, 0.10, text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    return duration_list, width_list


def vary_duration_height(peak, P, t_range, broadened_val = 0, I_bkgd_perc = 0):
    height_list = []
    duration_list = []
    
    # for t in numpy.linspace(t_range[0], t_range[1], 100):
    for t in numpy.linspace(t_range[0], t_range[1], 100):
        failed = True
        try:
            fit_params = fit_gaussian_peak(peak, P, t, broadened_val, I_bkgd_perc)
            failed = False
        except Exception:
            continue
        
        if not failed:
            height_list.append(fit_params[0]**2)
            duration_list.append(t)
            
    fig, ax = plt.subplots(1, 1)
    lin_durations = numpy.linspace(duration_list[0], duration_list[-1], 100)
    ax.plot(duration_list, height_list, 'bo')
    
    if I_bkgd_perc != 0:
        fit_func = lambda t,  d: exp_scaling(t, 1, 1, d)
        init_guess = [1]
    fit_params,cov_arr = curve_fit(fit_func, duration_list, height_list,
                                      p0 = init_guess)
    print(fit_params)
    ax.plot(lin_durations, fit_func(lin_durations, *fit_params), 'k-')
    
    I_0 = intensity_scaling(P)
    epsilon = numpy.sqrt(1/(fit_params[0] * v * I_0**2))
    print('fit value for background percentage: ', epsilon)
    
    ax.set_xlabel('Duration (ms)')
    ax.set_ylabel(r'Peak height')
    ax.set_yscale('log')
    # ax.set_xscale('log')
    
    return duration_list, height_list

def simulation_2D(P, t,  img_range, num_steps, I_bkgd_perc = 0):
    img_array = numpy.zeros((num_steps,num_steps))
    
    half_img_range = img_range/2
    x_vals = numpy.linspace(-half_img_range, half_img_range, num_steps)
    y_vals = numpy.linspace(-half_img_range, half_img_range, num_steps)
    
    for j in range(num_steps):
        for i in range(num_steps):
            r = numpy.sqrt(x_vals[i]**2 + y_vals[j]**2)
            val = eta(r,v, P, t, I_bkgd_perc)
            img_array[i][j] = val
            
    half_px = (x_vals[1]-x_vals[0] )/2
    img_extent = [-(half_img_range+half_px), (half_img_range+half_px),
                  -(half_img_range+half_px), (half_img_range+half_px)]
    
    
    fig, ax = plt.subplots(dpi=300)    
    fig.set_figwidth(3)
    fig.set_figheight(3)
    
    img = ax.imshow(img_array, cmap='inferno', 
                    extent=tuple(img_extent))
    clb = plt.colorbar(img)
    clb.set_label("NV- pop", rotation=270)
    
    ax.set_xlabel('x (nm)')
    ax.set_ylabel('y (nm)')
    # Draw the canvas and flush the events to the backend
    fig.canvas.draw()
    plt.tight_layout()
    fig.canvas.flush_events()
            
# %%
def bessel_scnd_der(x):
    term_1 = 24*j1(x)**2
    term_2 = 16*j1(x)*(jv(0,x) - jv(2,x))
    term_3 = 4* (0.5* (jv(0,x) - jv(2,x))**2 + j1(x)* (0.5*(jv(3,x) - j1(x)) - j1(x)))
    
    return term_1/x**4 - term_2/x**3 + term_3/x**2

def power_law(x, a, b):
    return a*x**b
    
def exp_scaling(x, a, b, d):
    return a*numpy.exp(-x**b/d)

def width_scaling_w_mods(t, C, e, a, R):
    return numpy.sqrt((-e*C + numpy.sqrt(e**2*C**2 + 4*C**2*a/t)) / (2*C**2)) + R**2

def intensity_scaling(P):
    I = P*NA**2*pi/wavelength**2
    return I


def radial_scaling(r):
    x = 2*pi*NA*r/wavelength
    return x

def intensity_airy_func(r, P, I_bkgd_perc):
    I = intensity_scaling(P)
    x = radial_scaling(r)
    return I * (2*j1(x) / x)**2  + I_bkgd_perc*I 


def eta(r,v, P, t, I_bkgd_perc = 0):
    return numpy.exp(-v*t*intensity_airy_func(r, P, I_bkgd_perc)**2)


# %%

if __name__ == '__main__':
    # vary_powers(300, [40, 340], 70) # -0.5
    # vary_duration_width(300, 70, [0.25, 100],broadened_val = 0,  I_bkgd_perc = 0.005) # -0.25
    
    # vary_duration_height(300, 20, [0.25, 100],broadened_val = 0,  I_bkgd_perc = 0.006) # -0.25
    
    # fit_gaussian_peak(300, 70, 100, broadened_val = 0, I_bkgd_perc = 0.0, do_plot = True)

    # simulation_2D(40, 1,  1500, 100, I_bkgd_perc = 0.005)
    estimate_ion_rate()
    # fig, ax = plt.subplots(1, 1)
    # r = numpy.linspace(-700 , 700, 701)
    # P = 70
    # t = 300
    # ax.plot(r, intensity_airy_func(r, P, 0))
