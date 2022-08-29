# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 10:33:08 2022

This file assumes that the first point is the max contrast
@author: kolkowitz
"""
import numpy
import utils.tool_belt as tool_belt
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def stretch_exp(x, o, a, d, n):
    return o + a*numpy.exp(-(x/d)**n)
    
folder = 'pc_rabi/branch_master/dynamical_decoupling_xy4/2022_08'
file1 = '2022_08_26-11_14_02-rubin-nv1_2022_08_10' #xy4-1
file2 = '2022_08_26-12_57_55-rubin-nv1_2022_08_10' #xy4-2
file3 = '2022_08_29-02_03_35-rubin-nv1_2022_08_10' #xy4-3
file4 = '2022_08_29-04_43_19-rubin-nv1_2022_08_10' #xy4-4
file5 = '2022_08_29-08_00_57-rubin-nv1_2022_08_10' #xy4-5
file6 = '2022_08_29-11_56_25-rubin-nv1_2022_08_10' #xy4-6

file_list = [file1, file2, file3, file4, file5,file6]
# file_list = [file2, file3,]
# contrast = 0.167*2

fig, ax = plt.subplots()

for file in file_list:
    data = tool_belt.get_raw_data(file, folder)
    sig_counts = numpy.array(data['sig_counts'])
    ref_counts = numpy.array(data['ref_counts'])
    precession_time_range = data['precession_time_range']
    num_xy4_reps = data['num_xy4_reps']
    num_runs = data['num_runs']
    num_steps = data['num_steps']
    taus = numpy.array(data['taus'])
    # print(taus)
    
    # calc taus
    # min_precession_time = int(precession_time_range[0])
    # max_precession_time = int(precession_time_range[1])
    
    # taus = numpy.linspace(
    #     min_precession_time,
    #     max_precession_time,
    #     num=num_steps,
    # )
    plot_taus = (taus * 2 *4* num_xy4_reps) / 1000
    taus_linspace = numpy.linspace(plot_taus[0], plot_taus[-1], 100     
                      )
    # calc contrast from data 
    ref_avg = numpy.average(ref_counts)
    ref_ste = numpy.std(
        ref_counts, ddof=1
    ) / numpy.sqrt(num_runs*num_steps)
    
    sig_zero = numpy.average(sig_counts, axis=0)[0]
    sig_ste = numpy.std(
        sig_counts, axis=0, ddof=1
    ) / numpy.sqrt(num_runs)
    sig_zero_ste = sig_ste[0]
    
    contrast =   sig_zero / ref_avg
    contrast_err = contrast*numpy.sqrt((ref_ste/ref_avg)**2 + (sig_zero_ste/sig_zero)**2)
    # print(contrast)
    
    # calc norm sig and ste
    norm_sig = sig_counts / ref_counts
    norm_avg_sig = numpy.average(norm_sig, axis=0)
    # print(norm_avg_sig)
    norm_avg_sig_ste = numpy.std(
        norm_sig, axis=0, ddof=1
    ) / numpy.sqrt(num_runs)
    
    # convert norm sig to coherence
    norm_avg_coherence = (1- norm_avg_sig)/(1-contrast)
    norm_avg_coherence_ste = norm_avg_coherence* numpy.sqrt((norm_avg_sig_ste/norm_avg_sig)**2 + \
                                                            (contrast_err/contrast)**2)
    
    
    ax.errorbar(
            plot_taus,
            norm_avg_coherence,
            yerr=norm_avg_coherence_ste,
            fmt="o-",
            # color="blue",
            # label="data",
            label = 'N = {}'.format(num_xy4_reps)
        )    
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.set_xlabel("Coherence time, T (us)")
    ax.set_ylabel("Normalized signal Counts")
    ax.set_title('XY4-N')
    ax.legend()
    
    if False:
        # fit to stretched exponential
        fit_func = lambda x, d:stretch_exp(x, 0.5, 1, d, 3)
        init_params = [100]
        opti_params, cov_arr = curve_fit(
            fit_func,
            plot_taus,
            norm_avg_coherence,
            p0=init_params,
            sigma=norm_avg_coherence_ste,
            absolute_sigma=True,
            )
    
        ax.plot(
            taus_linspace,
            fit_func(taus_linspace, *opti_params),
            "r",
            label="fit",
        )
                      
        text = "d = {:.2f} us\nn= {:.2f}".format(opti_params[0], opti_params[1])
        
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        ax.text(
            0.55,
            0.9,
            text,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=props,
        )
