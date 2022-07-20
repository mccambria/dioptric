# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 09:25:26 2021
Plotting for the paper

@author: agardill
"""

import utils.tool_belt as tool_belt
import numpy
import matplotlib.pyplot as plt
# import airy_disk_simulation 
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
fwhm =1.825 
scale = 0.99e3


epsilon = 8.7e-4
alpha = 3.1e-6
R_nm = 6

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

def width_scaling_w_mods(t, C, e, a, R):
    return numpy.sqrt(4/C* (-e + numpy.sqrt(e**2+ a/t)) + R**2)
    
def bessel_scnd_der(x):
    term_1 = 24*j1(x)**2
    term_2 = 16*j1(x)*(jv(0,x) - jv(2,x))
    term_3 = 4* (0.5* (jv(0,x) - jv(2,x))**2 + j1(x)* (0.5*(jv(3,x) - j1(x)) - j1(x)))
    
    return term_1/x**4 - term_2/x**3 + term_3/x**2
    
def inverse_quarter(x, a):
    return a*x**(-1/4)

def inverse_sqrt(x, a):
    return a*x**(-1/2)

def exp_decay(x, a, d):
    return a * numpy.exp(-x/d)

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
    fit_func = gaussian_quad # ???

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
            print(opti_params[2]*fwhm, numpy.sqrt(cov_arr[2][2])*fwhm)
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

def plot_width_vs_dur(file_list, t_vals, path, threshold):
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
        
    # Get a linear list of time values for the fit
    t_min = min(t_vals)
    t_max = max(t_vals)
    # lin_x_vals = numpy.linspace(t_min,
    #                 t_max, 100)
    lin_x_vals = numpy.logspace(numpy.log10(t_min), numpy.log10(t_max), 100)

    # Start plot
    fig_w =3.3
    fig_l = fig_w * 0.9
    fig, ax = plt.subplots()
    fig.set_figwidth(fig_w)
    fig.set_figheight(fig_l)
    
    
    
    init_fit = [8]
    opti_params, cov_arr = curve_fit(inverse_quarter,
          t_vals,widths_master_list,p0=init_fit)
    print('Opti params: ' + str(opti_params))
    ax.plot(lin_x_vals, inverse_quarter(lin_x_vals, opti_params)*fwhm, 
            color = 'orange', linestyle =  (0,(2,2)) , linewidth = 1, label = r'$\Delta x$')
    
    
    
    x0=7.0156 #Position of this Airy disk, in dimensionless units
    C = bessel_scnd_der(x0) #Calculate a constant to use in fit
    #convert extracted value for sigma of peaks to fwhm:
    widths_master_list_fwhm = numpy.array(widths_master_list)*fwhm
    # convert fwhm to dimensionless unit.
    widths_master_list_x = 2*numpy.pi*NA*widths_master_list_fwhm/wavelength
    
    fit_func = lambda t, e, a: width_scaling_w_mods(t, C, e, a, 0) 
    funct_vals = [ epsilon, alpha]
    # opti_params, cov_arr = curve_fit(fit_func,
    #       t_vals,widths_master_list_x,p0=init_fit,
    #        bounds=(0, [1, numpy.inf])
    #       )
    ax.plot(lin_x_vals, fit_func(lin_x_vals, *funct_vals)*wavelength/(2*numpy.pi*NA), 
            color = 'blue',  linestyle = (0,(6,3))   , linewidth = 1, label = r"$\Delta x'$")
    
    
    #### modified inverse quarter####
    # Estimate the lower limit and convert to dimentionless units below
    fit_func = lambda t, e, a, R: width_scaling_w_mods(t, C, e, a, R) 
    func_vals = [ epsilon, alpha, 2*numpy.pi*NA*R_nm/wavelength]
    
    
    # opti_params, cov_arr = curve_fit(fit_func,
    #       t_vals,widths_master_list_x,p0=init_fit,
    #        bounds=(0, [1, numpy.inf, numpy.inf])
    #       )
    
    # print(opti_params)
    # print('e = {:.7f} +/- {:.7f}'.format(opti_params[0], numpy.sqrt(cov_arr[0][0])))
    # print('a = {:.14f} +/- {:.14f}'.format(opti_params[1], numpy.sqrt(cov_arr[1][1])))
    # R_val = opti_params[2]
    # R_val_conv = (opti_params[2])*wavelength/(2*numpy.pi*NA)
    # R_err_conv = numpy.sqrt(cov_arr[2][2])*wavelength/(2*numpy.pi*NA)
    # print('R = {:.5f} +/- {:.5f}'.format(R_val_conv, R_err_conv))
    # print('Opti params: ' + str(opti_params))
    # print(cov_arr)
    # print((opti_params[2])**2*wavelength/(2*numpy.pi*NA))
    ax.plot(lin_x_vals, fit_func(lin_x_vals, *func_vals)*wavelength/(2*numpy.pi*NA), 
            color = 'red',  linestyle = 'solid' , linewidth = 1, label = r"$\Delta x''$")
    
    
        
    # fit_func = lambda t, a: width_scaling_w_mods(t, C, 0, a, R_val) 
    # init_fit = [ 10]
    # opti_params, cov_arr = curve_fit(fit_func,
    #       t_vals,widths_master_list_x,p0=init_fit,
    #       )
    # print(opti_params)
    # ax.plot(lin_x_vals, fit_func(lin_x_vals, *opti_params)*wavelength/(2*numpy.pi*NA)*fwhm, 
    #         color = 'red',  linestyle = 'dashdot' , linewidth = 1, label = 'Eq. 2, assumption #2')
    
    
    #plot all data points
    ax.errorbar(t_vals, numpy.array(widths_master_list)*fwhm,  
                yerr = numpy.array(widths_error)*fwhm, fmt='o', color = 'black', 
                linewidth = 1, markersize = 5, mfc='#d6d6d6',
                label = 'data')
    
        
    ax.set_xlabel(r'Depletion pulse duration, $\tau$ (ms)', fontsize = f_size)
    ax.set_ylabel('FWHM (nm)', fontsize = f_size)
    ax.tick_params(which = 'both', length=fig_tick_l, width=fig_tick_w,
                        direction='in',grid_alpha=0.7, labelsize = f_size)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize = f_size)


 
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


plot_width_vs_dur(file_list, [], path, threshold)




