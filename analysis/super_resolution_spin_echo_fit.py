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

two_pi = 2*numpy.pi
# %%
def t2(tau,  a1, a2,  d, Offset, C, A):
    exp_part = numpy.exp(-((tau / d) ** 4))
    sin_part = A - C * numpy.sin(two_pi *a1 * tau/2)**2 * numpy.sin(two_pi*a2 * tau/2)**2
    return  Offset - exp_part *sin_part 

# %%
def do_fit(file, folder):
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
        
    data = tool_belt.get_raw_data(file, folder)
    norm_avg_sig = numpy.array(data['norm_avg_sig'])
    uwave_pi_pulse = data['uwave_pi_pulse']
    
    taus = data['taus']
    plot_taus = (numpy.array(taus) + uwave_pi_pulse) / 1000
    
        
    # Fit data
    lin_taus = numpy.linspace(plot_taus[0], plot_taus[-1], 1000)
    
    fit_func = t2
    #       w0, w1, Tc, Offset, oscillation amplitude, decay amplitude     
    # init_params = [20, 0.08,  4, 1.08, 0.02, 0.06] # A   
    init_params = [9, 0.08,  3, 1.08, 0.1, 0.05] # B
    
    # popt, pcov = curve_fit(
    #     fit_func,
    #     plot_taus,
    #     norm_avg_sig ,
    #     p0=init_params,
    # )
    
    # print(popt)
    ax.plot(lin_taus, fit_func(lin_taus, *init_params), 'r-')
    
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
    
    ax.plot(plot_taus,norm_avg_sig , 'b-')
    ax.set_xlabel('Taus (us)')
    ax.set_ylabel('Contrast (arb. units)')
    ax.legend(loc='lower right')
    

# %%
folder = 'pc_rabi/branch_master/super_resolution_spin_echo/2021_10'

###################
file_list = [
    # '2021_10_14-10_39_50-johnson-dnv5_2021_09_23',
                '2021_10_15-09_38_47-johnson-dnv5_2021_09_23'
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
    
    ft = numpy.fft.rfft(norm_avg_sig)
    freqs = numpy.fft.rfftfreq(len(plot_taus), d = (plot_taus[1] - plot_taus[0]))
    ft_mag = numpy.absolute(ft)

    ax.plot(freqs[1:], ft_mag[1:], 'b-')
    
ax.set_ylabel('FFT Amplitude')
ax.set_xlabel('Frequency (MHz)')

# ax.set_ylabel('Contrast (arb. units)')
# ax.set_xlabel('Taus (us)')
# ax.legend(loc='lower right')


# do_fit(file_list[1], folder)


