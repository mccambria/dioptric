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
fwhm =1.825 # ???
scale = 34.5e3 #updated 3/8/2022

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

def width_scaling_w_mods(P, C, e, a, R):
    # term_1 = e**2*C**2 - 4*C**2*(-a/P**2)
    # term_2 = -e*C + numpy.sqrt(term_1)
    # term_3 = 2*C**2
    
    # # return numpy.sqrt(term_2 / term_3) + R**2
    # return numpy.sqrt((numpy.sqrt(term_2 / term_3))**2 + R**2)

    return numpy.sqrt(4/C* (-e + numpy.sqrt(e**2+ 1*a/P**2)) + R**2)
    
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

# %%
def plot_1D_inset(file_name, folder, threshold):
    
    fig_w =1.2
    fig_l = fig_w * 1
    fig, ax = plt.subplots()
    fig.set_figwidth(fig_w)
    fig.set_figheight(fig_l)

    ax.set_xlabel(r'$\Delta$x (nm)', fontsize = f_size)
    ax.set_ylabel(r'$\eta$', fontsize = f_size)
    ax.tick_params(which = 'both', length=fig_tick_l, width=fig_tick_w,
                    direction='in',grid_alpha=0.7, labelsize = f_size)
    
    marker_sizes = [2]
    
    data = tool_belt.get_raw_data( file_name, folder)
    nv_sig = data['nv_sig']
    print(nv_sig['CPG_laser_measured_power'])
    
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
    # rad_dist = numpy.array(data['rad_dist'])*scale
    num_steps = data['num_steps']
    offset_2D = [0,0,0]#data['offset_2D']
    coords_voltages = data['coords_voltages']
    x_voltages = numpy.array([el[0] for el in coords_voltages]) -offset_2D[0] - coords[0]
    rad_dist = +x_voltages*scale

    ax.plot(rad_dist, counts, 
            shape_list[0],  color= color_list[2],  markersize = marker_sizes[0],
            markeredgewidth=0.0, 
            )
    

    opti_params = []
    fit_func = gaussian_quad 


    init_fit = [0.6, rad_dist[int(num_steps/2)], abs(rad_dist[0]-rad_dist[-1])/10, 0]
    try:
        opti_params, cov_arr = curve_fit(fit_func,
              rad_dist,
              counts,
              p0=init_fit
              )
        lin_radii = numpy.linspace(rad_dist[0],
                        rad_dist[-1], 100)
        ax.plot(lin_radii,
               fit_func(lin_radii, *opti_params), color= 'black',#color_list[f],
               linestyle = 'dashed' , linewidth = 1)

        print(opti_params)
    except Exception:
        text = 'Peak could not be fit'
        print(text)
    
    ax.set_xticks([500, 750, 1000])
    # ax.legend(fontsize = f_size)


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
    num_steps = data['num_steps']
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
        init_fit = [0.6, rad_dist[int(num_steps/2)], abs(rad_dist[0]-rad_dist[-1])/10, 0]
        try:
            opti_params, cov_arr = curve_fit(fit_func,
                  rad_dist,
                  counts,
                  p0=init_fit
                  )
            # print(opti_params)
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
    spec_points = []#[4, 7, 11]
    widths_master_list = []
    widths_error = []
    p_vals =[]
    
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
        p_vals.append(nv_sig['CPG_laser_measured_power'])
        
    # return
    # Get a linear list of time values for the fit
    p_min = min(p_vals)
    p_max = max(p_vals)
    print(p_vals)
    print(widths_master_list)
    # lin_x_vals = numpy.linspace(t_min,
    #                 t_max, 100)
    lin_x_vals = numpy.logspace(numpy.log10(p_min), numpy.log10(p_max), 100)

    # Start plot
    # fig_w =3.75
    # fig_l = 2.5
    fig_w =3.3
    fig_l = fig_w * 0.9
    fig, ax = plt.subplots()
    fig.set_figwidth(fig_w)
    fig.set_figheight(fig_l)
    
    #### modified inverse quarter####
    x0=7.0156 #Position of this Airy disk (n2), in dimensionless units
    C = bessel_scnd_der(x0) #Calculate a constant to use in fit
    # print(C)
    # print(t_vals)
    # Estimate the lower limit and convert to dimentionless units below
    R_guess = 10 #nm 
    fit_func = lambda P, e, a: width_scaling_w_mods(P, C, e, a, 0) 
    init_fit = [ 0.006, 10,]# 2*numpy.pi*NA*R_guess/wavelength]
    
    #convert extracted value for sigma of peaks to fwhm:
    widths_master_list_fwhm = numpy.array(widths_master_list)*fwhm
    # convert fwhm to dimensionless unit.
    widths_master_list_x = 2*numpy.pi*NA*widths_master_list_fwhm/wavelength
    
    opti_params, cov_arr = curve_fit(fit_func,
          p_vals,widths_master_list_x,p0=init_fit,
           # bounds=(0, [1, numpy.inf, numpy.inf])
          )
    
    print('e = {:.7f} +/- {:.7f}'.format(opti_params[0], numpy.sqrt(cov_arr[0][0])))
    print('a = {:.14f} +/- {:.14f}'.format(opti_params[1], numpy.sqrt(cov_arr[1][1])))
    # R_val_conv = (opti_params[2])*wavelength/(2*numpy.pi*NA)*fwhm
    # R_err_conv = numpy.sqrt(cov_arr[2][2])*wavelength/(2*numpy.pi*NA)*fwhm
    # print('R = {:.5f} +/- {:.5f}'.format(R_val_conv, R_err_conv))
    # print('Opti params: ' + str(opti_params))
    # print(cov_arr)
    # print((opti_params[2])**2*wavelength/(2*numpy.pi*NA)*fwhm)
    ax.plot(lin_x_vals, fit_func(lin_x_vals, *opti_params)*wavelength/(2*numpy.pi*NA), 
            color = 'blue',  linestyle = (0,(6,3))   , linewidth = 1,)
    
        
    # init_fit = [8]
    # opti_params, cov_arr = curve_fit(inverse_sqrt,
    #       p_vals,widths_master_list,p0=init_fit)
    # print('Opti params: ' + str(opti_params))
    # ax.plot(lin_x_vals, inverse_sqrt(lin_x_vals, opti_params)*fwhm, 
    #         color = 'orange', linestyle = 'dashed' , linewidth = 1)
    
    #plot all data points
    p_vals_exc = copy.deepcopy(p_vals)
    widths_master_exc = copy.deepcopy(widths_master_list)
    widths_error_exc = copy.deepcopy(widths_error)
    for i in reversed(spec_points):
        del p_vals_exc[i]
        del widths_master_exc[i]
        del widths_error_exc[i]
    
    ax.errorbar(p_vals_exc, numpy.array(widths_master_exc)*fwhm,  
                yerr = numpy.array(widths_error_exc)*fwhm, fmt='o', color = 'black', 
                linewidth = 1, markersize = 5, mfc='#d6d6d6')
    
    # plot specific points to highlight inset
    marker_sizes = [5]*3
    for i in range(len(spec_points)):
        n = spec_points[i]
        n_color = color_list[i]
        ax.errorbar(p_vals[n], widths_master_list[n]*fwhm, yerr = widths_error[n]*fwhm, 
                 marker='o',#shape_list[i], 
                color = 'black', markersize = marker_sizes[i], linewidth = 1, #markeredgewidth=0.0,
                mfc=n_color
                )
        
    ax.set_xlabel(r'Depletion pulse power, $P$ (mW)', fontsize = f_size)
    ax.set_ylabel('FWHM (nm)', fontsize = f_size)
    ax.tick_params(which = 'both', length=fig_tick_l, width=fig_tick_w,
                        direction='in',grid_alpha=0.7, labelsize = f_size)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    
    ax.set_yticks([70, 80, 90, 100, 140])
    # ax.set_xticks([0.008, 0.009, 0.01, 
    #                0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 
    #                0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 
    #                2, 3, 4, 5, 6, 7, 8, 9, 10])

 
# %%

path = 'pc_rabi/branch_master/SPaCE/2021_08/2021_08_05_vary_pulse_power'
file_list= [  #cfmB data
            '2021_08_05-23_11_42-johnson-nv2_2021_08_04',
            '2021_08_05-22_46_43-johnson-nv2_2021_08_04',
            '2021_08_05-22_25_21-johnson-nv2_2021_08_04',
            '2021_08_05-17_59_06-johnson-nv2_2021_08_04',
            '2021_08_05-20_06_21-johnson-nv2_2021_08_04',
            '2021_08_05-20_55_56-johnson-nv2_2021_08_04',
            # '2021_08_05-21_17_41-johnson-nv2_2021_08_04',
            # '2021_08_05-21_41_06-johnson-nv2_2021_08_04',
            # '2021_08_05-22_01_58-johnson-nv2_2021_08_04'
            ]
inset_file= '2021_08_05-22_25_21-johnson-nv2_2021_08_04' #18.6 mW
# label_list = ['5000 {}s'.format(mu), '500 {}s'.format(mu), '50 {}s'.format(mu)]

threshold  = 7





###################
plot_width_vs_dur(file_list, [], path, threshold)


plot_1D_inset(inset_file, path, threshold)
