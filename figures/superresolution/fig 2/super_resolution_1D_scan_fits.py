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

def width_scaling_w_mods(t, C, e, a, R):
    # term_1 = e**2*C**2 - 4*C**2*(-a/t)
    # term_2 = -e*C + numpy.sqrt(term_1)
    # term_3 = 2*C**2
    
    # return numpy.sqrt(term_2 / term_3) + R**2
    
    
    # return numpy.sqrt((2*numpy.sqrt(term_2 / term_3))**2 + R**2)
    
    # return numpy.sqrt(2/C* (-e + numpy.sqrt(e**2+ 4*a/t)) + R**2) #previous
    return numpy.sqrt(4/C* (-e + numpy.sqrt(e**2+ a/t)) + R**2) #current
    
def bessel_scnd_der(x):
    term_1 = 24*j1(x)**2
    term_2 = 16*j1(x)*(jv(0,x) - jv(2,x))
    term_3 = 4* (0.5* (jv(0,x) - jv(2,x))**2 + j1(x)* (0.5*(jv(3,x) - j1(x)) - j1(x)))
    
    return term_1/x**4 - term_2/x**3 + term_3/x**2
    
def inverse_quarter(x, a):
    return a*x**(-1/4)

def inverse_sqrt(x, a):
    return a*x**(-1/2)

def exp_decay(x, a, B):
    return a * B* numpy.exp(-x*B)

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
def plot_3_plots(file_list, folder, label_list, threshold):
    
    fig_w =0.9
    fig_l = fig_w * 1
    fig, ax = plt.subplots()
    fig.set_figwidth(fig_w)
    fig.set_figheight(fig_l)

    ax.set_xlabel(r'$\Delta$x (nm)', fontsize = f_size)
    ax.set_ylabel(r'$\eta$', fontsize = f_size)
    ax.tick_params(which = 'both', length=fig_tick_l, width=fig_tick_w,
                    direction='in',grid_alpha=0.7, labelsize = f_size)
    
    marker_sizes = [2.3, 1.9, 2.3]
    marker_sizes = [2]*3
    for f in range(len(file_list)):
        file_name = file_list[f]
        data = tool_belt.get_raw_data( file_name, folder)
        
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
        num_steps = data['num_steps_a']
        offset_2D = data['offset_2D']
        coords_voltages = data['coords_voltages']
        x_voltages = numpy.array([el[0] for el in coords_voltages]) -offset_2D[0] - coords[0]
        rad_dist = -x_voltages*scale

        ax.plot(numpy.flip(rad_dist), numpy.flip(counts), 
                shape_list[f],  color= color_list[f],  markersize = marker_sizes[f],
                markeredgewidth=0.0, 
                )
        
        ax.plot([],[], marker = shape_list[f], color= color_list[f], markersize = 1.5,
                linewidth = 0.5, label = label_list[f])

        opti_params = []
        fit_func = gaussian_quad #???


        init_fit = [2, rad_dist[int(num_steps/2)], 15, 7]
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
            
    ax.legend(fontsize = f_size)


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

def plot_width_vs_dur(file_list, t_vals, path, threshold):
    spec_points = [4, 7, 11]
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
    # print(t_vals)
    # print(2*numpy.pi*NA*numpy.array(widths_error)/wavelength*fwhm)
    # lin_x_vals = numpy.linspace(t_min,
    #                 t_max, 100)
    lin_x_vals = numpy.logspace(numpy.log10(t_min), numpy.log10(t_max), 100)

    # Start plot
    # fig_w =3.75
    # fig_l = 2.5
    fig_w =2.1
    fig_l = 2.63
    fig, ax = plt.subplots()
    fig.set_figwidth(fig_w)
    fig.set_figheight(fig_l)
    
    #### modified inverse quarter####
    x0=7.0156 #Position of this Airy disk (n2), in dimensionless units
    C = bessel_scnd_der(x0) #Calculate a constant to use in fit
    # print(C) 
    # print(t_vals)
    # Estimate the lower limit and convert to dimentionless units below
    R_guess = 5 #nm 
    e = 0.0008723151593412476
    fit_func = lambda t, a, R: width_scaling_w_mods(t, C, e, a, R) 
    init_fit = [  0.005, 2*numpy.pi*NA*R_guess/wavelength]
    
    #convert extracted value for sigma of peaks to fwhm:
    widths_master_list_fwhm = numpy.array(widths_master_list)*fwhm
    # convert fwhm to dimensionless unit.
    widths_master_list_x = 2*numpy.pi*NA*widths_master_list_fwhm/wavelength
    
    widths_error_fwhm = numpy.array(widths_error)*fwhm
    widths_error_fwhm_x = 2*numpy.pi*NA*widths_error_fwhm/wavelength
    
    # print('widths_master_list_x: ' + str(widths_master_list_x))
    opti_params, cov_arr = curve_fit(fit_func,
          t_vals,widths_master_list_x,p0=init_fit,
          sigma = widths_error_fwhm_x,
          absolute_sigma = True,
            # bounds=(0, [numpy.inf, 2*numpy.pi*NA*20/wavelength]) #1
          )
    
    # init_fit = [ 0.0000035 , 2*numpy.pi*NA*(4.7-8) /wavelength]
    # opti_params = init_fit
    # print('e = {:.7f} +/- {:.7f}'.format(opti_params[2], numpy.sqrt(cov_arr[2][2])))
    print('a = {:.14f} +/- {:.14f}'.format(opti_params[0], numpy.sqrt(cov_arr[0][0])))
    R_val_conv = (opti_params[1])*wavelength/(2*numpy.pi*NA)
    R_err_conv = numpy.sqrt(cov_arr[1][1])*wavelength/(2*numpy.pi*NA)
    print('R = {:.5f} +/- {:.5f}'.format(R_val_conv, R_err_conv))
    # print('Opti params: ' + str(opti_params))
    # print(cov_arr)
    # print((opti_params[2])**2*wavelength/(2*numpy.pi*NA)*fwhm)
    ax.plot(lin_x_vals, fit_func(lin_x_vals, *opti_params)*wavelength/(2*numpy.pi*NA), 
            color = 'red',  linestyle = 'dashed' , linewidth = 1)
    
        
    # init_fit = [8]
    # opti_params, cov_arr = curve_fit(inverse_quarter,
    #       t_vals,widths_master_list,p0=init_fit)
    # print('Opti params: ' + str(opti_params))
    # ax.plot(lin_x_vals, inverse_quarter(lin_x_vals, opti_params)*fwhm, 
    #         color = 'orange', linestyle = 'dashed' , linewidth = 1)
    
    #plot all data points
    t_vals_exc = copy.deepcopy(t_vals)
    widths_master_exc = copy.deepcopy(widths_master_list)
    widths_error_exc = copy.deepcopy(widths_error)
    for i in reversed(spec_points):
        del t_vals_exc[i]
        del widths_master_exc[i]
        del widths_error_exc[i]
    
    ax.errorbar(t_vals_exc, numpy.array(widths_master_exc)*fwhm,  
                yerr = numpy.array(widths_error_exc)*fwhm, fmt='o', color = 'black', 
                linewidth = 1, markersize = 5, mfc='#d6d6d6')
    
    # plot specific points to highlight inset
    marker_sizes = [5]*3
    for i in range(len(spec_points)):
        n = spec_points[i]
        n_color = color_list[i]
        ax.errorbar(t_vals[n], widths_master_list[n]*fwhm, yerr = widths_error[n]*fwhm, 
                 marker='o',#shape_list[i], 
                color = 'black', markersize = marker_sizes[i], linewidth = 1, #markeredgewidth=0.0,
                mfc=n_color
                )
        
    ax.set_xlabel(r'Depletion pulse duration, $\tau$ (ms)', fontsize = f_size)
    ax.set_ylabel('FWHM (nm)', fontsize = f_size)
    ax.tick_params(which = 'both', length=fig_tick_l, width=fig_tick_w,
                        direction='in',grid_alpha=0.7, labelsize = f_size)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks([0.008, 0.009, 0.01, 
                   0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 
                   0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 
                   2, 3, 4, 5, 6, 7, 8, 9, 10])


def fit_data_points(x_data, y_data):

        
    # Get a linear list of time values for the fit
    t_min = min(x_data)
    t_max = max(x_data)
    # lin_x_vals = numpy.linspace(t_min,
    #                 t_max, 100)
    lin_x_vals = numpy.logspace(numpy.log10(t_min), numpy.log10(t_max*2), 100)

    # Start plot
    fig_w =3.3
    fig_l = fig_w * 0.9
    fig, ax = plt.subplots()
    fig.set_figwidth(fig_w)
    fig.set_figheight(fig_l)
    
    #### modified inverse quarter####
    x0=7.0156 #Position of this Airy disk, in dimensionless units
    C = bessel_scnd_der(x0) #Calculate a constant to use in fit
    # R_guess = 10 #nm 
    fit_func = lambda t, e, a,: width_scaling_w_mods(t, C, e, a, 0) 
    init_fit = [ 0.0005, 0.000040]
    
    # convert widths to dimensionless unit. Not using FWHM because that would be farther from
    # mean value of peak, so less approximate. However, can recover same fit values by scaling 
    # opti_params with the fwhm (e*fwhm**2, a*fwhm**4, R is converted normally)
    widths_master_list_x = 2*numpy.pi*NA*numpy.array(y_data)/wavelength
    
    d = 0#5
    print(x_data[d:])
    opti_params, cov_arr = curve_fit(fit_func,
          x_data[d:],widths_master_list_x[d:],p0=init_fit,
            bounds=(0, [1, numpy.inf])
          )
    
    print('e = {:.7f} +/- {:.7f}'.format(opti_params[0], numpy.sqrt(cov_arr[0][0])))
    print('a = {:.14f} +/- {:.14f}'.format(opti_params[1], numpy.sqrt(cov_arr[1][1])))
    
    # R_val_conv = (opti_params[2])*wavelength/(2*numpy.pi*NA)*fwhm
    # R_err_conv = numpy.sqrt(cov_arr[2][2])*wavelength/(2*numpy.pi*NA)*fwhm
    # print('R = {:.5f} +/- {:.5f}'.format(R_val_conv, R_err_conv))
    
    ax.plot(lin_x_vals, fit_func(lin_x_vals, *opti_params)*wavelength/(2*numpy.pi*NA), 
            color = 'red',  linestyle = 'dashed' , linewidth = 1)
    
        
    # init_fit = [8]
    
    ax.plot(x_data, y_data,  'ko', 
                linewidth = 1, markersize = 5, mfc='#d6d6d6')
    
        
    ax.set_xlabel(r'Depletion pulse duration, $\tau$ (ms)', fontsize = f_size)
    ax.set_ylabel('FWHM (nm)', fontsize = f_size)
    ax.tick_params(which = 'both', length=fig_tick_l, width=fig_tick_w,
                        direction='in',grid_alpha=0.7, labelsize = f_size)
    
    ax.set_xscale('log')
    ax.set_yscale('log')

    
    
    
def plot_heights_vs_dur(file_list, t_vals, path, threshold):
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


    print('t_vals = ' + str(t_vals))
    print('heights_error = ' + str(heights_error))
    t_min = min(t_vals)
    t_max = max(t_vals)
    lin_x_vals = numpy.linspace(t_min,
                    t_max, 100)

    # Sigma
    fig_w =3.3
    fig_l = fig_w * 0.9
    fig, ax = plt.subplots()
    fig.set_figwidth(fig_w)
    fig.set_figheight(fig_l)
    
    #### exp decay###
    init_fit = [10, 1]
    opti_params, cov_arr = curve_fit(exp_decay,
          t_vals,heights_master_list,p0=init_fit,
          sigma = heights_error,
          absolute_sigma = True)
    Beta = opti_params[1]
    x0=7.0156 #Position of this Airy disk, in dimensionless units
    j = j1(x0)
    # v1 = 1.6e13#2.3e12
    # P = 20
    # NA = 1.3
    # wavelength = 638
    # I0 = P*NA**2*numpy.pi/wavelength**2
    alpha = numpy.log(2) / 0.00000309393011#alpha from fwhm fit
    # epsilon = numpy.sqrt(Beta/(v1*I0**2)) - (2*j/x0)**2
    epsilon = numpy.sqrt(Beta/(alpha)) - (2*j/x0)**2
    epsilon_err = numpy.sqrt(numpy.sqrt(cov_arr[1][1])/(alpha)) - (2*j/x0)**2
    # print(j)
    print('Opti params: ' + str(opti_params))
    print('A: ' + str(opti_params[0]) + ' +/-' + str(numpy.sqrt(cov_arr[0][0])))
    print('epsilon: ' + str(epsilon) + ' +/- ' + str(epsilon_err))
    ax.plot(lin_x_vals, exp_decay(lin_x_vals, *opti_params), 
            color = 'red', linestyle = 'dashed' , linewidth = 1)
    
        
    #plot all data points
    ax.errorbar(t_vals, heights_master_list,  
                yerr = heights_error, fmt='o', color = 'black', 
                linewidth = 1, markersize = 5, mfc='#d6d6d6')
        
    ax.set_xlabel(r'Depletion pulse duration, $\tau$ (ms)', fontsize = f_size)
    ax.set_ylabel('Peak height (counts)', fontsize = f_size)
    ax.tick_params(which = 'both', length=fig_tick_l, width=fig_tick_w,
                        direction='in',grid_alpha=0.7, labelsize = f_size)
    
    ax.set_yscale('log')
 
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
file_list_dur = [
               '2021_11_24-16_24_33-johnson-nv1_2021_11_17',
                '2021_11_25-04_39_56-johnson-nv1_2021_11_17',
                '2021_11_25-16_38_19-johnson-nv1_2021_11_17',
              ]
label_list = ['5000 {}s'.format(mu), '500 {}s'.format(mu), '50 {}s'.format(mu)]

threshold  = 5





###################
plot_width_vs_dur(file_list, [], path, threshold)
# plot_heights_vs_dur(file_list, [], path, threshold)


# plot_3_plots(file_list_dur, path, label_list, threshold)

####################
# data from Lumerial modeling of the experiment. However, the fit is not good. Returns a value for epsilon of 0
x_data = [1, 1.5, 2, 4, 6, 8, 10]
y_data = [58.7993, 50.8695, 46.4504, 36.5816, 31.1147,
          27.6861, 25.3146]

x_data = [
    0.01, 
          0.1, 
          0.25, 
          0.5, 
          0.75, 
          1, 
          1.5, 
          2,
          4, 
          6, 
          8, 
          10]
y_data = [
    148.663, 
          97.4929, 
          80.8762,
          68.7151,
          62.7879, 
          58.7177, 
           52.9568, 
          49.2224,
          41.3943, 
          36.8599, 
          33.9284, 
          29.9457]

# x_data = [10.0, 11.0, 7.5, 5.0, 2.5, 1.0, 0.75, 0.5, 0.25, 0.1, 0.075, 0.05, 0.01]
# y_data = [9.261184383601464, 10.063912926838842, 11.586827675766282, 
#           11.876802373199931, 16.92319211433264, 25.480131991881052, 
#           24.89691165527999, 29.78063302437111, 37.196010226453026, 47.62212211327829, 
#           52.361128554997414, 58.40307029410666, 101.01389593470778]
# fit_data_points(x_data, y_data)



