# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 09:32:09 2022

@author: agard
"""



import utils.tool_belt as tool_belt
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy
from utils.tool_belt import NormStyle

def decay_sin(t, amp, offset, decay, freq):
    return offset + amp * numpy.cos(t*freq * 2 * numpy.pi) * numpy.exp(-t/decay)

def decay_double_sin(t, amp1, amp2, offset, decay, freq1, freq2):
    return offset + (amp1 * numpy.cos(t*freq1 * 2 * numpy.pi) + amp2 * numpy.cos(t*freq2 * 2 * numpy.pi)) * numpy.exp(-t/decay)

folder = "pc_rabi/branch_master/ramsey/2023_01"

# mode = 'SQ'
mode = 'DQ'

if mode == 'DQ':
    file1 = '2023_01_03-15_56_19-siena-nv8_2022_12_22'
    file2 = '2023_01_03-16_08_26-siena-nv8_2022_12_22'
    file3 = '2023_01_03-16_22_29-siena-nv8_2022_12_22'
    file4 = '2023_01_03-16_38_17-siena-nv8_2022_12_22'
    file5 = '2023_01_03-16_55_56-siena-nv8_2022_12_22'
    file6 = '2023_01_03-17_15_20-siena-nv8_2022_12_22'
    file7 = '2023_01_03-17_43_23-siena-nv8_2022_12_22'
    file8 = '2023_01_03-18_20_05-siena-nv8_2022_12_22'
    file9 = '2023_01_03-19_05_35-siena-nv8_2022_12_22'
    file_list = [file1,
                  file2,
                  file3,
                  file4,
                  file5,
                  file6,
                  file7,
                  file8,
                  file9,
        ]
    freq_g = 4.4
    title = "Ramsey, DQ"

elif mode == 'SQ':
    file1 = '2023_01_03-21_27_18-siena-nv8_2022_12_22'
    file2 = '2023_01_03-21_39_33-siena-nv8_2022_12_22'
    file3 = '2023_01_03-21_53_29-siena-nv8_2022_12_22'
    file4 = '2023_01_03-22_09_08-siena-nv8_2022_12_22'
    file5 = '2023_01_03-22_26_32-siena-nv8_2022_12_22'
    file6 = '2023_01_03-22_45_39-siena-nv8_2022_12_22'
    file7 = '2023_01_03-23_13_28-siena-nv8_2022_12_22'
    file8 = '2023_01_03-23_49_57-siena-nv8_2022_12_22'
    file9 = '2023_01_04-00_35_08-siena-nv8_2022_12_22'
    file_list = [file1,
                  file2,
                  file3,
                  # file4,
                  file5,
                  file6,
                  file7,
                   file8,
                  file9,
        ]
    freq_g = 2.4
    title = "Ramsey, SQ"

norm_avg_sig_master = []
norm_avg_sig_ste_master = []
taus_master = []

avg = 0.923

for file in file_list:
    
    data = tool_belt.get_raw_data(file, folder)
    # detuning= data['detuning']
    nv_sig = data['nv_sig']
    sig_counts = data['sig_counts']
    ref_counts = data['ref_counts']
    num_reps = data['num_reps']
    readout = nv_sig['spin_readout_dur']
    norm_style = NormStyle.SINGLE_VALUED
    
    ret_vals = tool_belt.process_counts(sig_counts, ref_counts, num_reps, readout, norm_style)
    (
        sig_counts_avg_kcps,
        ref_counts_avg_kcps,
        norm_avg_sig,
        norm_avg_sig_ste,
    ) = ret_vals
    
    
    norm_avg = numpy.average(norm_avg_sig)
    shift = avg - norm_avg
    norm_avg_sig = numpy.array(norm_avg_sig) + shift
    
    taus = data['taus']
    norm_avg_sig_master = norm_avg_sig_master + list(norm_avg_sig)
    norm_avg_sig_ste_master = norm_avg_sig_ste_master + list(norm_avg_sig_ste)
    taus_master = taus_master + taus


taus_us = numpy.array(taus_master)/1e3
# Guess the other params for fitting
amp = -0.05
offset = .927
decay = 10

guess_params = (amp, offset, decay, freq_g )
fit_func = lambda t, amp, offset, decay, freq: decay_sin(t, amp, offset, decay, freq)
init_params = guess_params

init_params = guess_params

popt,pcov = curve_fit(fit_func, taus_us, norm_avg_sig_master,
              p0=init_params,
              sigma=norm_avg_sig_ste_master,
              absolute_sigma=True,
                # bounds=([-numpy.infty, 0, 0, 0], [numpy.infty, numpy.infty, numpy.infty,  30])
                )
              
# popt = guess_params
print(popt)
print(numpy.sqrt(numpy.diag(pcov)))

taus_us_linspace = numpy.linspace(taus_us[0] ,taus_us[-1],
                      num=10000)

fig_fit, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.errorbar(taus_us, norm_avg_sig_master, label='data', yerr=norm_avg_sig_ste_master, marker = 'o', linestyle = "", color = 'b')
ax.plot(taus_us_linspace, fit_func(taus_us_linspace,*popt),'r',label='fit')
ax.set_xlabel(r'Free precesion time ($\mu$s)')
ax.set_ylabel('Contrast (arb. units)')
ax.legend()
ax.set_title(title)


text1 = "\n".join((r'$C + A e^{-t/d} \mathrm{cos}(2 \pi \nu t)$',
                    r'$d = $' + '%.2f'%(abs(popt[2])) + ' us',
                    r'$\nu = $' + '%.2f'%(popt[3]) + ' MHz',
                    ))

props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
# print(text1)

ax.text(0.70, 0.25, text1, transform=ax.transAxes, fontsize=12,
                            verticalalignment="top", bbox=props)