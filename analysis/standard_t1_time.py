

import utils.tool_belt as tool_belt
import utils.kplotlib as kpl
from utils.kplotlib import KplColors
import utils.positioning as positioning
import utils.common as common
import majorroutines.optimize as optimize
import numpy
import os
import time
from random import shuffle
import matplotlib.pyplot as plt
import labrad
from utils.tool_belt import States
import shutil
# import analysis.relaxation_rate_analysis as relaxation_rate_analysis
from pathlib import Path
from scipy.optimize import curve_fit
from utils.tool_belt import NormStyle


kpl.init_kplotlib()


def fit_t1(file, rabi_amp):
    fig, ax = plt.subplots()
    data = tool_belt.get_raw_data(file)
    
    sig_counts = data['sig_counts']
    ref_counts= data['ref_counts']
    relaxation_time_range = data['relaxation_time_range']
    num_steps = data['num_steps']
    num_reps = data['num_reps']
    nv_sig = data['nv_sig']
    readout = nv_sig['spin_readout_dur']
    
    ret_vals = tool_belt.process_counts(sig_counts, ref_counts, num_reps, readout, NormStyle.SINGLE_VALUED)
    (
        sig_counts_avg_kcps,
        ref_counts_avg_kcps,
        norm_avg_sig,
        norm_avg_sig_ste,
    ) = ret_vals
    
    taus = numpy.linspace(
        relaxation_time_range[0],
        relaxation_time_range[1],
        num=num_steps,
    )
    taus_ms = taus/1e6
    smooth_taus_ms = numpy.linspace(taus_ms[0], taus_ms[-1], 1000)
    
    amp=rabi_amp*2*2/3
    fit_func = lambda x,  decay: tool_belt.exp_decay(x, amp, decay, 1-amp)
    init_params = [4]
    popt, pcov = curve_fit(fit_func, taus_ms, norm_avg_sig,
                        p0=init_params,
                        sigma=norm_avg_sig_ste,
                        absolute_sigma=True)
    print(popt[0])    
    print(numpy.sqrt(pcov[0][0]))  
    
    T1=  popt[0]
    T1_unc = numpy.sqrt(pcov[0][0])   
    
    omega = 1/(3*T1*1e-3)
    omega_ste = omega * T1_unc/ T1
    print('Omega {} +/- {} s^-1'.format(omega, omega_ste))
    
    kpl.plot_points(ax, taus_ms, norm_avg_sig, yerr=norm_avg_sig_ste)
    
    kpl.plot_line(
        ax,
        smooth_taus_ms,
        fit_func(smooth_taus_ms, *popt),
        color=KplColors.RED,)
    
    ax.legend()
    text = "T1 = {:.2f} +/- {:.2f} ms".format(T1, T1_unc)
    kpl.anchored_text(ax, text, kpl.Loc.LOWER_LEFT, size=kpl.Size.SMALL)
        
        
    ax.set_xlabel('Wait time (ms)')
    ax.set_ylabel("Normalized fluorescence")
    ax.set_title('Relaxation from ms=0 to ms=0')
        
def compare_t1(file_list, label_list):
    fig, ax = plt.subplots()
    for ind in range(len(file_list)):
        file = file_list[ind]
        data = tool_belt.get_raw_data(file)
        
        sig_counts = data['sig_counts']
        ref_counts= data['ref_counts']
        relaxation_time_range = data['relaxation_time_range']
        num_steps = data['num_steps']
        num_reps = data['num_reps']
        nv_sig = data['nv_sig']
        readout = nv_sig['spin_readout_dur']
        
        ret_vals = tool_belt.process_counts(sig_counts, ref_counts, num_reps, readout, NormStyle.SINGLE_VALUED)
        (
            sig_counts_avg_kcps,
            ref_counts_avg_kcps,
            norm_avg_sig,
            norm_avg_sig_ste,
        ) = ret_vals
        
        taus = numpy.linspace(
            relaxation_time_range[0],
            relaxation_time_range[1],
            num=num_steps,
        )
        taus_ms = taus/1e6
        
        kpl.plot_points(ax, taus_ms, norm_avg_sig, yerr=norm_avg_sig_ste, label = label_list[ind])
        
        
        ax.legend()
            
            
        ax.set_xlabel('Wait time (ms)')
        ax.set_ylabel("Normalized fluorescence")
        ax.set_title('Relaxation from ms=0 to ms=0')
        
        #%%
        
    
file_0 = '2023_05_04-07_39_28-rubin-nv0_2023_05_03'
file_3= '2023_05_03-10_22_27-rubin-nv3_2023_05_01'

file_list= [file_0, file_3]
label_list = ['Area B', 'Area, A']

fit_t1(file_0, 0.103)
# compare_t1(file_list, label_list)