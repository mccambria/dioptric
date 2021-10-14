# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 10:09:33 2021

@author: agard
"""



# %% Imports


import utils.tool_belt as tool_belt
import numpy
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# %%
def t2(tau,  a1, a2,  A, C):
    # exp_part = numpy.exp(-((tau / decay_time) ** 3))
    sin_part = A - C * numpy.sin(a1 * tau)**2 * numpy.sin(a2 * tau)**2
    return  sin_part

# %%
def do_fit(file, folder):
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
        
    data = tool_belt.get_raw_data(file, folder)
    norm_avg_sig = numpy.array(data['norm_avg_sig'])
    uwave_pi_pulse = data['uwave_pi_pulse']
    
    taus = data['taus']
    plot_taus = (numpy.array(taus) + uwave_pi_pulse) / 1000
    
        
    # Fit data
    lin_taus = numpy.linspace(plot_taus[0], plot_taus[-1], 500)
    
    fit_func = t2
    #       offset, revival rate, amplitude, amplitude in exp, T2 (us)      
    init_params = [28, 0.5, 1.01, -0.18]
    # init_params = [1.2, -0.2, 1, 33, 3e2]
    
    popt, pcov = curve_fit(
        fit_func,
        plot_taus,
        norm_avg_sig ,
        p0=init_params,
    )
    
    print(popt)
    ax.plot(lin_taus, fit_func(lin_taus, *popt), 'r-')
    
    # text_eq = r"A + C * e$^{-D sin^4(w t)}$ * e$^{-(\tau / T_2)^3}$"
    
    # text_popt = "\n".join(
    #     ( #A, B, C, D, T2
    #         r"A=%.3f" % (popt[0]),
    #         r"C=%.3f" % (popt[2]),
    #         r"D=%.3f" % (popt[3]),
    #         r"w=%.5f MHz" % (popt[1]),
    #         r"T$_2$=%.3f (us)" % (popt[4]),
    #     )
    # )
    
    # props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    # ax.text(
    #     0.05,
    #     0.15,
    #     text_popt,
    #     transform=ax.transAxes,
    #     fontsize=12,
    #     verticalalignment="top",
    #     bbox=props,
    # )
    # ax.text(
    #     0.6,
    #     0.95,
    #     text_eq,
    #     transform=ax.transAxes,
    #     fontsize=12,
    #     verticalalignment="top",
    #     bbox=props,
    # )
    
    ax.plot(plot_taus,norm_avg_sig , 'b.-')
    ax.set_xlabel('Taus (us)')
    ax.set_ylabel('Contrast (arb. units)')
    ax.legend(loc='lower right')
    
# %%
folder = 'pc_rabi/branch_master/super_resolution_spin_echo/2021_10'

###################
file_list = [
    # '2021_10_13-01_15_35-johnson-dnv5_2021_09_23',
                '2021_10_13-08_43_23-johnson-dnv5_2021_09_23'
    ]

fmt_list = ['b-', 'r-']
label_list = ['A', 'B']
fig, ax = plt.subplots()

for f in range(len(file_list)):
    file = file_list[f]
    data = tool_belt.get_raw_data(file, folder)
    taus = data['taus']
    norm_avg_sig = numpy.array(data['norm_avg_sig'])
    uwave_pi_pulse = data['uwave_pi_pulse']
    plot_taus = (numpy.array(taus) + uwave_pi_pulse) / 1000
    
    ft = numpy.fft.fft(norm_avg_sig)
    freqs = numpy.fft.fftfreq(len(plot_taus), d = (plot_taus[1] - plot_taus[0]))

    ax.plot(freqs, ft.real, 'b-')
# ax.set_ylabel('Contrast (arb. units)')
# ax.set_xlabel('Taus (us)')
# ax.legend(loc='lower right')

# do_fit(file_list[1], folder)


