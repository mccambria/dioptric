# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 09:25:26 2021
Plotting for the paper

@author: agardill
"""

import utils.tool_belt as tool_belt
import numpy
import matplotlib.pyplot as plt
import airy_disk_simulation 
from scipy.optimize import curve_fit
from scipy.special import j1
from scipy.special import jv
import copy
# %%


fig_tick_l = 3
fig_tick_w = 0.75

f_size = 8

mu = u"\u03BC"
superscript_minus = u"\u207B" 

NA = 1.3
wavelength = 638
# fwhm =2.355
fwhm =1.825 # 2* (ln(2))^1/4
scale = 0.99e3

red = '#ee3542'
orange = '#faa51a'
purple = '#a12964'
blue = '#00b3ff'
green = '#00a651'
orange_ = '#f89522'
dark_purple = '#190d3c'


color_list =  [  '#f9b91d', '#ef6f23','#c43d4e']
color_list =  [  '#2bace2', 'orange','green']
shape_list = ['^', 's', 'd']
shape_list = ['o']*3
# %%

def width_scaling_w_mods(t, e, alpha, R):
    x0=7.0156 #Position of this Airy disk (n2), in dimensionless units
    C = bessel_scnd_der(x0) #Calculate a constant to use in fit
    return numpy.sqrt(4/C* (-e + numpy.sqrt(e**2+ alpha/t)) + R**2) 
    
def bessel_scnd_der(x):
    term_1 = 24*j1(x)**2
    term_2 = 16*j1(x)*(jv(0,x) - jv(2,x))
    term_3 = 4* (0.5* (jv(0,x) - jv(2,x))**2 + j1(x)* (0.5*(jv(3,x) - j1(x)) - j1(x)))
    
    return term_1/x**4 - term_2/x**3 + term_3/x**2

def exp_decay(t, A, alpha, e):
    return A * numpy.exp(-t* (numpy.log(2)/alpha) * e**2)

def gaussian_quad(x,  *params):
    """
    Calculates the value of a gaussian with a x^4 in the exponent
    for the given input and parameters

    Params:
        x: float
            Input value
        params: tuple
            The parameters that define the Gaussian
            0: coefficient that defines the peak height
            1: mean, defines the center of the Gaussian
            2: standard deviation-like parameter, defines the width of the Gaussian
            3: constant y value to account for background
    """

    coeff, mean, stdev, offset = params
    var = stdev ** 4  # variance squared
    centDist = x - mean  # distance from the center
    return offset + coeff ** 2 * numpy.exp(-(centDist ** 4) / (var))

# %%

def plot_1D_SpaCE(file_name, file_path, threshold = None, do_plot = True, do_fit = False,
                  do_save = False):
    data = tool_belt.get_raw_data( file_name, file_path)
    timestamp = data['timestamp']
    counts = data['readout_counts_avg']
    
    if threshold:
        # convert single shot measurements to NV- population
        raw_counts = numpy.array(data['readout_counts_array'])
        for r in range(len(raw_counts)):
            row = raw_counts[r]
            for c in range(len(row)):
                current_val = raw_counts[r][c]
                if current_val < threshold:
                    set_val = 0
                elif current_val >= threshold:
                    set_val = 1
                raw_counts[r][c] = set_val
        counts = numpy.average(raw_counts, axis = 1)
        
    nv_sig = data['nv_sig']
    coords = nv_sig['coords']
    num_steps = data['num_steps_a']
    try:
        rad_dist = numpy.array(data['rad_dist'])*scale
    except Exception:
        

        coords_voltages = data['coords_voltages']
        x_voltages = numpy.array([el[0] for el in coords_voltages]) -coords[0]
        rad_dist = x_voltages*( scale)
    
    opti_params = []
    # fit_func = tool_belt.gaussian
    fit_func = gaussian_quad 

    if do_plot:
        fig_w =1.5
        fig_l = fig_w * 1
        fig, ax = plt.subplots()
        fig.set_figwidth(fig_w)
        fig.set_figheight(fig_l)

        ax.plot(rad_dist, counts, 'b.', markersize = 1,)
        ax.set_xlabel('r (nm)', fontsize = f_size)
        ax.set_ylabel('Average counts', fontsize = f_size)
        ax.tick_params(which = 'both', length=fig_tick_l, width=fig_tick_w,
                        direction='in',grid_alpha=0.7, labelsize = f_size)

    if do_fit:
        init_fit = [2, rad_dist[int(num_steps/2)], 15, 7]
        try:
            opti_params, cov_arr = curve_fit(fit_func,
                  rad_dist,
                  counts,
                  p0=init_fit
                  )
            # print(opti_params[2]*fwhm, numpy.sqrt(cov_arr[2][2])*fwhm)
            if do_plot:
                lin_radii = numpy.linspace(rad_dist[0],
                                rad_dist[-1], 100)
                ax.plot(lin_radii,
                       fit_func(lin_radii, *opti_params), 'r-',  linestyle = 'solid' , linewidth = 1)
        except Exception:
            text = 'Peak could not be fit'
            print(text)

    if do_plot and do_save:
        filePath = tool_belt.get_file_path('SPaCE', timestamp,
                                                nv_sig['name'])
        tool_belt.save_figure(fig, filePath + '-gaussian_fit')
        
    return rad_dist, counts, opti_params, cov_arr

def plot_width_vs_dur(file_list,  path, threshold, e):
    widths_master_list = []
    widths_error = []
    t_vals =[]
    
    # fit the data with file "plot_1D_SpaCE
    # Currently fitting to quadratic gaussian
    # Returns the width of the fit (related to FWHM with scaling at deginning of file)
    # and corresponding uyncertainty with fit. Values in nanometers
    # Also saves the pulse time, in ms
    for f in range(len(file_list)):
        file_name = file_list[f]
        data = tool_belt.get_raw_data( file_name, path)
        nv_sig=data['nv_sig']
    
        ret_vals = plot_1D_SpaCE(file_name, path, threshold, do_plot = False, do_fit = True)
        widths_master_list.append(ret_vals[2][2])
        widths_error.append(numpy.sqrt(ret_vals[3][2][2]))
        t_vals.append(nv_sig['CPG_laser_dur']/10**6)
        

    #### modified inverse quarter####
    # Estimate the lower limit and convert to dimentionless units below
    R_guess = 5 #nm 
    fit_func = lambda t, alpha, R: width_scaling_w_mods(t, e, alpha, R) 
    init_fit = [  0.000005, 2*numpy.pi*NA*R_guess/wavelength] 
    
    #convert extracted value for sigma of peaks to fwhm:
    widths_master_list_fwhm = numpy.array(widths_master_list)*fwhm
    # convert fwhm to dimensionless unit.
    widths_master_list_x = 2*numpy.pi*NA*widths_master_list_fwhm/wavelength
    
    #also convert uncertianty
    widths_error_fwhm = numpy.array(widths_error)*fwhm
    widths_error_fwhm_x = 2*numpy.pi*NA*widths_error_fwhm/wavelength
    
    opti_params, cov_arr = curve_fit(fit_func,
          t_vals,widths_master_list_x,p0=init_fit,
          sigma = widths_error_fwhm_x,
          absolute_sigma = True
          )
    
    print()
    print('constrained e = {:.7f} '.format(e))
    print('alpha = {:.14f} +/- {:.14f}'.format(opti_params[0], numpy.sqrt(cov_arr[0][0])))
    R_val_conv = (opti_params[1])*wavelength/(2*numpy.pi*NA)
    R_err_conv = numpy.sqrt(cov_arr[1][1])*wavelength/(2*numpy.pi*NA)
    print('R = {:.5f} +/- {:.5f}'.format(R_val_conv, R_err_conv))
        
    return opti_params[0], R_val_conv, R_err_conv


def plot_heights_vs_dur(file_list,  path, threshold, Alpha):
    heights_master_list = []
    heights_error = []
    t_vals =[]
    for f in range(len(file_list)):
        file_name = file_list[f]
        data = tool_belt.get_raw_data( file_name, path)
        nv_sig=data['nv_sig']
    
        ret_vals = plot_1D_SpaCE(file_name, path, threshold, do_plot = False, do_fit = True)
        heights_master_list.append(ret_vals[2][0]**2)
        heights_error.append(numpy.sqrt(ret_vals[3][0][0])**2)
        t_vals.append(nv_sig['CPG_laser_dur']/10**6)

    
    #### exp decay###
    init_fit = [0.5, 0.001]
    fit_func = lambda t, A, e: exp_decay(t, A, Alpha, e) 
    opti_params, cov_arr = curve_fit(fit_func,
          t_vals,heights_master_list,p0=init_fit,
          sigma = heights_error,
          absolute_sigma = True)
    print()
    print('constrained alpha: ' + str(Alpha))
    print('A: ' + str(opti_params[0]) + ' +/-' + str(numpy.sqrt(cov_arr[0][0])))
    print('epsilon: ' + str(opti_params[1]) + ' +/- ' + str(numpy.sqrt(cov_arr[1][1])))

    return opti_params[1]
 
def recursive_fit(file_list, path, threshold, num_reps):
    alpha = 0.000005 # first guess of the value alpha
    
    e_list =[]
    R_list = []
    for i in range(num_reps):
        print()
        print('*** Fitting round {} ***'.format(i))
                
        e = plot_heights_vs_dur(file_list, path, threshold, alpha)
        e_list.append(e)
        alpha, R, R_err = plot_width_vs_dur(file_list,  path, threshold, e)
        R_list.append(R)
    
    fig, axes = plt.subplots(1,2)
    ax = axes[0]
    ax.plot(e_list)
    ax.set_xlabel('Num fitting')
    ax.set_ylabel('Epsilon (perc. of I0)')
    ax = axes[1]
    ax.plot(R_list)
    ax.set_xlabel('Num fitting')
    ax.set_ylabel('Repeatability (nm)')
    
# %%

path = 'pc_rabi/branch_CFMIII/SPaCE_digital/2021_11'
file_list= [  #boxed data
            '2021_11_23-20_00_39-johnson-nv1_2021_11_17',
            '2021_11_24-00_45_14-johnson-nv1_2021_11_17',
            '2021_11_24-13_55_24-johnson-nv1_2021_11_17',
            '2021_11_24-16_24_33-johnson-nv1_2021_11_17', #
            '2021_11_24-18_55_35-johnson-nv1_2021_11_17',
            '2021_11_24-18_55_36-johnson-nv1_2021_11_17',
            '2021_11_25-02_10_27-johnson-nv1_2021_11_17',
            '2021_11_25-04_39_56-johnson-nv1_2021_11_17', #
            '2021_11_25-07_10_10-johnson-nv1_2021_11_17',
            '2021_11_25-09_45_56-johnson-nv1_2021_11_17',
            '2021_11_25-12_21_52-johnson-nv1_2021_11_17',
            '2021_11_25-16_38_19-johnson-nv1_2021_11_17', #
            '2021_11_26-00_00_21-johnson-nv1_2021_11_17', 
            ]

threshold  = 5





###################
# plot_width_vs_dur(file_list, [], path, threshold)
# plot_heights_vs_dur(file_list, [], path, threshold)

num_reps = 6
recursive_fit(file_list, path, threshold, num_reps)
