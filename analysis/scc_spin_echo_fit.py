# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 10:08:38 2021

@author: agardill
"""

# %% Imports


import utils.tool_belt as tool_belt
import numpy
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


h_bar = 1.0545718e-34 # J s
gamma_C13 = 1.071e-3 # MHz/G
mu_B = 9.274e-28 # J/G


def t2(tau,  A, B, C, D, T2):
    exp_part = numpy.exp(-((tau / T2) ** 3))
    sin_part = (numpy.sin(B*tau))**4
    return  A + C * numpy.exp(-D*sin_part)* exp_part

# def t2_val(tau,  A, C ,B_nuc, B_dc, T2):
#     exp_part = numpy.exp(-((tau / T2) ** 3))
#     sin_part = (numpy.sin(2*numpy.pi*B_dc*gamma_C13*tau/4))**4
#     revival_amp = -8*(( 2*mu_B * B_nuc) / (2*numpy.pi*B_dc*gamma_C13*h_bar))**2
#     return  A + C * numpy.exp(revival_amp*sin_part)* exp_part

# %%

def do_average_files_together(file_list, folder):

    norm_counts_combined = []
    tot_num_runs = 0
    for file in file_list:
        data = tool_belt.get_raw_data(file, folder)
        taus = data['taus']
        norm_avg_sig = data['norm_avg_sig']
        norm_counts_combined.append(norm_avg_sig)
        num_runs = data['num_runs']
        tot_num_runs+= num_runs
        
    norm_counts_combined = numpy.average(norm_counts_combined, axis = 0)
    nv_sig = data['nv_sig']
    uwave_pi_on_2_pulse = data['uwave_pi_on_2_pulse']
    uwave_pi_pulse = data['uwave_pi_pulse']
    state = data['state']
    num_reps = data['num_reps']
    
    timestamp = tool_belt.get_time_stamp()
    rawData = {'timestamp': timestamp,
        'nv_sig': nv_sig,
        'nv_sig-units': tool_belt.get_nv_sig_units(),
        'file_list': file_list,
        'folder': folder,
        "uwave_pi_pulse": uwave_pi_pulse,
        "uwave_pi_pulse-units": "ns",
        "uwave_pi_on_2_pulse": uwave_pi_on_2_pulse,
        "uwave_pi_on_2_pulse-units": "ns",
        'state': state,
        'num_reps': num_reps,
        'num_runs': tot_num_runs,
        'taus': taus,
        "norm_avg_sig": norm_counts_combined.tolist(),
        "norm_avg_sig-units": "arb",
        }
    
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.plot(taus, norm_counts_combined, 'b-')
    ax.set_xlabel('Taus (us)')
    ax.set_ylabel('Contrast (arb. units)')
    ax.legend(loc='lower right')
    
    
    name = nv_sig['name']
    folder_split = folder.split('/')
    folder_dir = folder_split[-2]
    filePath = tool_belt.get_file_path(folder_dir, timestamp, name)
    tool_belt.save_figure(fig, filePath)
    tool_belt.save_raw_data(rawData, filePath)


def combine_revivals(file_list, folder):
    
    norm_counts_tot = []
    taus_tot = []
    for file in file_list:
        data = tool_belt.get_raw_data(file, folder)
        taus = data['taus']
        norm_avg_sig = data['norm_avg_sig']
        norm_counts_tot = norm_counts_tot + norm_avg_sig
        taus_tot = taus_tot + taus
    nv_sig = data['nv_sig']
    uwave_pi_on_2_pulse = data['uwave_pi_on_2_pulse']
    uwave_pi_pulse = data['uwave_pi_pulse']
    state = data['state']
    num_reps = data['num_reps']
    num_runs = ['num_runs']
    
    timestamp = tool_belt.get_time_stamp()
    rawData = {'timestamp': timestamp,
        'nv_sig': nv_sig,
        'nv_sig-units': tool_belt.get_nv_sig_units(),
        "uwave_pi_pulse": uwave_pi_pulse,
        "uwave_pi_pulse-units": "ns",
        "uwave_pi_on_2_pulse": uwave_pi_on_2_pulse,
        "uwave_pi_on_2_pulse-units": "ns",
        'state': state,
        'num_reps': num_reps,
        'num_runs': num_runs,
        'taus': taus_tot,
        "norm_avg_sig": norm_counts_tot,
        "norm_avg_sig-units": "arb",
        }
    
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.plot(numpy.array(taus_tot)/1e3, norm_counts_tot, 'bo')
    ax.set_xlabel('Taus (us)')
    ax.set_ylabel('Contrast (arb. units)')
    ax.legend(loc='lower right')
    
    
    name = nv_sig['name']
    folder_split = folder.split('/')
    folder_dir = folder_split[-2]
    filePath = tool_belt.get_file_path(folder_dir, timestamp, name)
    tool_belt.save_figure(fig, filePath)
    tool_belt.save_raw_data(rawData, filePath)
            
    return

# %%
def do_fit(file, folder):
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
        
    data = tool_belt.get_raw_data(file, folder)
    norm_avg_sig = numpy.array(data['norm_avg_sig'])
    uwave_pi_pulse = data['uwave_pi_pulse']
    
    try:
        taus = data['taus']
    except Exception:
        precession_time_range = data['precession_time_range']
        num_steps = data['num_steps']
        min_precession_time = int(precession_time_range[0])
        max_precession_time = int(precession_time_range[1])
    
        taus = numpy.linspace(
            min_precession_time,
            max_precession_time,
            num=num_steps,)
    plot_taus = (numpy.array(taus) + uwave_pi_pulse) / 1000
    
        
    # Fit data
    lin_taus = numpy.linspace(plot_taus[0], plot_taus[-1], 500)
    
    fit_func = t2
    #       offset, revival rate, amplitude, amplitude in exp, T2 (us)      
    init_params = [1.2, 0.1, -0.18, 45, 4e2]
    # init_params = [1.2, -0.2, 1, 33, 3e2]
    
    popt, pcov = curve_fit(
        fit_func,
        plot_taus,
        norm_avg_sig ,
        p0=init_params,
    )
    
    print(popt)
    ax.plot(lin_taus, fit_func(lin_taus, *popt), 'r-')
    
    text_eq = r"A + C * e$^{-D sin^4(w t)}$ * e$^{-(\tau / T_2)^3}$"
    
    text_popt = "\n".join(
        ( #A, B, C, D, T2
            r"A=%.3f" % (popt[0]),
            r"C=%.3f" % (popt[2]),
            r"D=%.3f" % (popt[3]),
            r"w=%.5f MHz" % (popt[1]),
            r"T$_2$=%.3f (us)" % (popt[4]),
        )
    )
    
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(
        0.05,
        0.15,
        text_popt,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=props,
    )
    ax.text(
        0.6,
        0.95,
        text_eq,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=props,
    )
    
    ax.plot(plot_taus,norm_avg_sig , 'bo')
    ax.set_xlabel('Taus (us)')
    ax.set_ylabel('Contrast (arb. units)')
    ax.legend(loc='lower right')

# %%


# folder = 'pc_rabi/branch_master/scc_spin_echo/2021_10'
folder = 'pc_rabi/branch_master/super_resolution_spin_echo/2021_10'

# file = '2021_10_08-13_39_25-johnson-dnv5_2021_09_23' #A
file = '2021_10_08-13_40_09-johnson-dnv5_2021_09_23' #B

do_fit(file, folder)


##### combine similar data and average data together
# file_list = ['2021_10_05-21_15_13-johnson-dnv5_2021_09_23',
#              '2021_10_06-07_32_54-johnson-dnv5_2021_09_23'
#     ]
# file_list = ['2021_10_05-21_53_26-johnson-dnv5_2021_09_23',
#              '2021_10_06-08_10_59-johnson-dnv5_2021_09_23'
#     ]

# do_average_files_together(file_list, folder)

##### Combine data over different taus into one data
# file_list_A = ['2021_10_08-13_29_07-johnson-dnv5_2021_09_23',
#              '2021_10_05-19_18_00-johnson-dnv5_2021_09_23',
#              '2021_10_05-23_09_33-johnson-dnv5_2021_09_23',
#              '2021_10_06-01_44_05-johnson-dnv5_2021_09_23',
#              '2021_10_06-09_31_43-johnson-dnv5_2021_09_23',
#              '2021_10_06-13_34_33-johnson-dnv5_2021_09_23',
             
#     ]
# file_list_B = ['2021_10_08-13_29_45-johnson-dnv5_2021_09_23',
#                '2021_10_05-20_31_21-johnson-dnv5_2021_09_23',
#                '2021_10_06-00_25_35-johnson-dnv5_2021_09_23',
#                '2021_10_06-03_02_34-johnson-dnv5_2021_09_23',
#                '2021_10_06-10_52_10-johnson-dnv5_2021_09_23',
#                '2021_10_06-14_57_46-johnson-dnv5_2021_09_23'
    # ]

# combine_revivals(file_list_B, folder)


###################
# file_list = ['2021_10_08-13_39_25-johnson-dnv5_2021_09_23',
#              '2021_10_08-13_40_09-johnson-dnv5_2021_09_23'
#     ]

# fmt_list = ['b-', 'r-']
# label_list = ['A', 'B']
# fig, ax = plt.subplots(figsize=(8.5, 8.5))

# for f in range(len(file_list)):
#     file = file_list[f]
#     data = tool_belt.get_raw_data(file, folder)
#     taus = data['taus']
#     norm_avg_sig = numpy.array(data['norm_avg_sig'])
#     uwave_pi_pulse = data['uwave_pi_pulse']
#     plot_taus = (numpy.array(taus) + uwave_pi_pulse) / 1000

#     ax.plot(plot_taus, norm_avg_sig, fmt_list[f], label = label_list[f])
# ax.set_ylabel('Contrast (arb. units)')
# ax.set_xlabel('Taus (us)')
# ax.legend(loc='lower right')