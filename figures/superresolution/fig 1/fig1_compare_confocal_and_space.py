# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 10:44:13 2021

@author: agard
"""


import utils.tool_belt as tool_belt
import numpy
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

fig_w =2
fig_l = fig_w * 1
        
fig_tick_l = 3
fig_tick_w = 0.75

f_size = 8

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

fit_func_conf = tool_belt.gaussian
fwhm_conf =2.355

fit_func_sr = gaussian_quad
fwhm_sr = 1.825
    
purple = '#9a2765'
light_purple = '#eecde2'
blue = '#2e3192'
light_blue = '#368ab4'#'#66ccff'
red = '#ed1c24'
green='#06a64f' 
orange = '#e65e2b'#'#f7941d'

# def double_gaussian(
#     x,
#     C_1,
#     x0_1,
#     s_1,
#     O_1,
#     C_2,
#     x0_2,
#     s_2,
#     O_2,
# ):
#     low_gauss = fit_func(x, C_1, x0_1, s_1, O_1)
#     high_gauss = fit_func(x,  C_2, x0_2, s_2, O_2)
#     return low_gauss + high_gauss

# %%
def do_plot_2nd_ring(conf_file_name, conf_folder, space_file_name_1, 
            space_file_name_2,space_folder, threshold, scale):
    
    fig, ax = plt.subplots()
    fig.set_figwidth(fig_w)
    fig.set_figheight(fig_l)
        
    
    # prep the confocal line cut
    data = tool_belt.get_raw_data( conf_file_name, conf_folder)
    nv_sig=data['nv_sig']
    readout = nv_sig['imaging_readout_dur']
    readout_sec = readout / 1e9
    x_scan_vals = numpy.array(data['x_scan_vals'])*scale
    try:
        num_steps = data['num_steps']
    except Exception:
        num_steps = len(x_scan_vals)
    x_scan_vals = x_scan_vals - x_scan_vals[int(num_steps/2)]
    x_counts = (numpy.array(data['x_counts']) / 1000) / readout_sec
    
    ax.plot(x_scan_vals, numpy.flip(x_counts), 'g.', color = light_blue, markersize = 3,
                # linewidth = 0.5,
                # mfc = light_blue
                )
    ax.set_ylabel('kcps', fontsize = f_size)
    # ax.set_ylim([-3.6, 40.6]) #5 ms
    ax.set_ylim([-4.3, 40.6]) #10 ms
    # ax.set_ylim([-4.6, 40.6]) #20 ms
    
    ax.tick_params(which = 'both', length=fig_tick_l, width=fig_tick_w,
                    direction='in',grid_alpha=0.7, labelsize = f_size)
    
    init_fit = [2, 0, 100, 7]
    opti_params, cov_arr = curve_fit(fit_func_conf,
          x_scan_vals[0:80],
          numpy.flip(x_counts)[0:80],
          p0=init_fit
          )
    print(opti_params[2]*fwhm_conf, numpy.sqrt(cov_arr[2][2])*fwhm_conf)
    lin_radii = numpy.linspace(x_scan_vals[0],
                    x_scan_vals[-1], 100)
    ax.plot(lin_radii,
           numpy.flip(fit_func_conf(lin_radii, *opti_params)), 'k-', color=light_blue,
           linestyle = 'solid' , linewidth = 1)


    
    ax2=ax.twinx()
    sr_list = [space_file_name_1, space_file_name_2]
    for file in sr_list:
    # prep the super resolution line cut
        data = tool_belt.get_raw_data( file, space_folder)
        offset_2D = data['offset_2D']
        # sr_counts = data['readout_counts_avg']
        
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
        sr_counts = numpy.average(raw_counts, axis = 1)
    
        num_steps = data['num_steps_a']
        coords_voltages = data['coords_voltages']
        nv_sig = data['nv_sig']
        coords = nv_sig['coords']
        x_voltages = numpy.array([el[0] for el in coords_voltages]) -offset_2D[0] +35e-3 - coords[0]
        rad_dist = x_voltages*scale
        # print(x_voltages)

        
        ax2.plot(-rad_dist, numpy.flip(sr_counts), 'b.', color = purple, markersize = 3,
                # linewidth = 0.5,
                # mfc = light_purple
                )
        ax2.set_xlabel('r (nm)', fontsize = f_size)
        ax2.set_ylabel('NV- pop', fontsize = f_size)
        ax2.tick_params(which = 'both', length=fig_tick_l, width=fig_tick_w,
                        direction='in',grid_alpha=0.7, labelsize = f_size)
        ax2.set_xticks([-1000,-500,0,500, 1000])
        # ax2.set_xlim([-550,550])
        # ax2.set_xlim([-950,950])
        # ax2.set_xlim([-650,650])
        ax2.set_xlim([-1110, 1110])
    
    
        init_fit = [2, rad_dist[int(num_steps/2)], 15, 7]
        opti_params, cov_arr = curve_fit(fit_func_sr,
              rad_dist,
              sr_counts,
              p0=init_fit
              )
        print(opti_params[0], cov_arr[0][0])
        print(opti_params[2]*fwhm_sr, numpy.sqrt(cov_arr[2][2])*fwhm_sr)
        
        lin_radii = numpy.linspace(min(rad_dist)-50,
                        max(rad_dist)+50, 100) 
        
        ax2.plot(-lin_radii,
                numpy.flip(fit_func_sr(lin_radii, *opti_params)), 'r-', color=purple,
                linestyle = 'solid' , linewidth = 1)


    return 


# %%
def do_plot_1st_ring(conf_file_name, conf_folder, space_file_name_1, 
            space_file_name_2,space_folder, threshold, scale):
    
    fig, ax = plt.subplots()
    fig.set_figwidth(fig_w)
    fig.set_figheight(fig_l)
        
    
    # prep the confocal line cut
    data = tool_belt.get_raw_data( conf_file_name, conf_folder)
    nv_sig=data['nv_sig']
    readout = nv_sig['imaging_readout_dur']
    readout_sec = readout / 1e9
    scan_vals = numpy.array(data['y_scan_vals'])*scale
    try:
        num_steps = data['num_steps']
    except Exception:
        num_steps = len(scan_vals)
    scan_vals = scan_vals - scan_vals[int(num_steps/2)]
    counts = (numpy.array(data['y_counts']) / 1000) / readout_sec
    
    ax.plot(scan_vals, numpy.flip(counts), 'g.', color = light_blue, markersize = 3,
                # linewidth = 0.5,
                # mfc = light_blue
                )
    ax.set_ylabel('kcps', fontsize = f_size)
    # ax.set_ylim([-3.6, 40.6]) #5 ms
    # ax.set_ylim([-4.3, 40.6]) #10 ms
    # ax.set_ylim([-4.6, 40.6]) #20 ms
    
    ax.tick_params(which = 'both', length=fig_tick_l, width=fig_tick_w,
                    direction='in',grid_alpha=0.7, labelsize = f_size)
    
    init_fit = [2, 0, 100, 7]
    opti_params, cov_arr = curve_fit(fit_func_conf,
          scan_vals,
          numpy.flip(counts),
          p0=init_fit
          )
    print(opti_params[2]*fwhm_conf, numpy.sqrt(cov_arr[2][2])*fwhm_conf)
    lin_radii = numpy.linspace(scan_vals[0],
                    scan_vals[-1], 100)
    ax.plot(lin_radii,
           fit_func_conf(lin_radii, *opti_params), 'k-', color=light_blue,
           linestyle = 'solid' , linewidth = 1)


    
    ax2=ax.twinx()
    sr_list = [space_file_name_1, space_file_name_2]
    for file in sr_list:
    # prep the super resolution line cut
        data = tool_belt.get_raw_data( file, space_folder)
        offset_2D = data['offset_2D']
        # sr_counts = data['readout_counts_avg']
        
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
        sr_counts = numpy.average(raw_counts, axis = 1)
    
        num_steps = data['num_steps_a']
        coords_voltages = data['coords_voltages']
        nv_sig = data['nv_sig']
        coords = nv_sig['coords']
        x_voltages = numpy.array([el[0] for el in coords_voltages]) -offset_2D[0] +35e-3 - coords[0]
        rad_dist = x_voltages*scale
        # print(x_voltages)

        
        ax2.plot(-rad_dist, numpy.flip(sr_counts), 'b.', color = purple, markersize = 3,
                # linewidth = 0.5,
                # mfc = light_purple
                )
        ax2.set_xlabel('r (nm)', fontsize = f_size)
        ax2.set_ylabel('NV- pop', fontsize = f_size)
        ax2.tick_params(which = 'both', length=fig_tick_l, width=fig_tick_w,
                        direction='in',grid_alpha=0.7, labelsize = f_size)
        ax2.set_xticks([-500, -250,0,250,500])
        # ax2.set_xlim([-505, 505])
    
    
        init_fit = [2, rad_dist[int(num_steps/2)], 15, 7]
        opti_params, cov_arr = curve_fit(fit_func_sr,
              rad_dist,
              sr_counts,
              p0=init_fit
              )
        print(opti_params[0], cov_arr[0][0])
        print(opti_params[2]*fwhm_sr, numpy.sqrt(cov_arr[2][2])*fwhm_sr)
        
        lin_radii = numpy.linspace(min(rad_dist)-25,
                        max(rad_dist)+25, 100) 
        
        ax2.plot(-lin_radii,
                numpy.flip(fit_func_sr(lin_radii, *opti_params)), 'r-', color=purple,
                linestyle = 'solid' , linewidth = 1)


    return 

# %%
conf_folder = 'pc_rabi/branch_CFMIII/optimize_digital/2021_12'
conf_file_name ='2021_12_28-00_26_28-johnson-nv0_2021_12_22'

# space_folder = 'pc_rabi/branch_CFMIII/SPaCE_digital/2021_12'
# space_file_name_1 = '2021_12_30-09_08_12-johnson-nv0_2021_12_22'
# space_file_name_2 = '2021_12_30-09_07_02-johnson-nv0_2021_12_22' #20 ms
# threshold = 9

# space_folder = 'pc_rabi/branch_CFMIII/SPaCE_digital/2021_12'
# space_file_name_1 = '2021_12_30-09_13_27-johnson-nv0_2021_12_22'
# space_file_name_2 = '2021_12_30-09_14_08-johnson-nv0_2021_12_22' # 5 ms
# threshold = 9

space_folder = 'pc_rabi/branch_CFMIII/SPaCE_digital/2022_01'
space_file_name_1 = '2022_01_04-10_01_52-johnson-nv0_2021_12_22'
space_file_name_2 = '2022_01_04-10_02_22-johnson-nv0_2021_12_22' # 10 ms
threshold = 14

scale = 0.99e3
do_plot_2nd_ring(conf_file_name, conf_folder, space_file_name_1, 
        space_file_name_2, space_folder, threshold,scale)

# first ring
conf_folder = 'pc_rabi/branch_CFMIII/optimize_digital/2021_12'
conf_file_name ='2021_12_12-22_10_59-johnson-nv0_2021_12_10'

space_folder = 'pc_rabi/branch_CFMIII/SPaCE_digital/2021_12'
space_file_name_1 = '2021_12_13-01_54_20-johnson-nv0_2021_12_10'
space_file_name_2 = '2021_12_13-01_05_43-johnson-nv0_2021_12_10'
threshold = 5

scale = 0.99e3
do_plot_1st_ring(conf_file_name, conf_folder, space_file_name_1, 
        space_file_name_2, space_folder, threshold,scale)

    