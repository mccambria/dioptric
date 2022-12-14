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

folder = "pc_rabi/branch_master/ramsey/2022_12"

mode = 'SQ'
# mode = 'DQ'

if mode == 'DQ':
    file1 = '2022_12_13-15_52_32-siena-nv1_2022_10_27'
    file10 = '2022_12_13-16_33_04-siena-nv1_2022_10_27'
    file2 = '2022_12_13-16_41_54-siena-nv1_2022_10_27'
    file20 = '2022_12_13-16_50_30-siena-nv1_2022_10_27'
    file3 = '2022_12_13-16_59_46-siena-nv1_2022_10_27'
    file4 = '2022_12_13-17_09_43-siena-nv1_2022_10_27'
    file5 = '2022_12_13-18_44_55-siena-nv1_2022_10_27'
    file6 = '2022_12_13-19_36_48-siena-nv1_2022_10_27'
    file7 = '2022_12_13-20_30_53-siena-nv1_2022_10_27'
    file8 = '2022_12_13-21_23_20-siena-nv1_2022_10_27'
    file_list = [file1,
                  file10,
                  file2,
                  file20,
                  file3,
                  file4,
                  file5,
                  file6,
                  file7,
                  file8
        ]
    freq_g = 4.4
    title = "Ramsey, DQ"

elif mode == 'SQ':
    file1 = '2022_12_13-23_10_37-siena-nv1_2022_10_27'
    file2 = '2022_12_13-23_37_08-siena-nv1_2022_10_27'
    file3 = '2022_12_14-00_05_18-siena-nv1_2022_10_27'
    file4 = '2022_12_14-00_40_19-siena-nv1_2022_10_27'
    file5 = '2022_12_14-08_34_15-siena-nv1_2022_10_27'
    file6 = '2022_12_14-01_24_06-siena-nv1_2022_10_27'
    file7 = '2022_12_14-02_16_36-siena-nv1_2022_10_27'
    file_list = [file1,
                  file2,
                   # file3,
                    file4,
                    # file5,
                    file6,
                    file7,
        ]
    freq_g = 2.4
    title = "Ramsey, SQ"

norm_avg_sig_master = []
norm_avg_sig_ste_master = []
taus_master = []

avg = 0.907

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
offset = .907
decay = 50

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