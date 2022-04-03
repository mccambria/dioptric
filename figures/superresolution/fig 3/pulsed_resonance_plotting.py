# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 11:55:37 2021

@author: agard
"""
# %%
import utils.tool_belt as tool_belt
import numpy
import matplotlib.pyplot as plt
import time
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# %%

marker_size = 2.5
fit_linewidth = 0.75
marker_line = 1

f_size = 7

fig_tick_l = 3
fig_tick_w = 0.75

green = '#00a651'
light_green = '#c2dfa2'
orange = '#f7941d'
light_orange = '#ffdcb4'
gray = '#d6d6d6'
# %%

def lorentzian(x, x0, A, L):
    x_center = x - x0
    return 1 + A * 0.5*L / (x_center**2 + (0.5*L)**2)


def double_lorentzian(x, x0_1, A_1, L_1, x0_2, A_2, L_2):
    return lorentzian(x, x0_1, A_1, L_1) + \
        lorentzian(x, x0_2, A_2, L_2) - 1
        
        
def quad_lorentzian(x, x0_1, A_1, L_1,
                    x0_2, A_2, L_2,
                    x0_3, A_3, L_3,
                    x0_4, A_4, L_4,):
    return double_lorentzian(x, x0_1, A_1, L_1, x0_2, A_2, L_2) +\
        double_lorentzian(x, x0_3, A_3, L_3, x0_4, A_4, L_4) - 1

#%%

def compare_sr_esr(file_A, file_B, file_C, folder):
    fig_w = 2.25
    # fig_l = 4.5
    fig_l = 3.1
    data_A = tool_belt.get_raw_data(file_A, folder)
    freqs_A = data_A['freqs']
    norm_sig_A = numpy.array(data_A['norm_avg_sig'])
    
    data_B = tool_belt.get_raw_data(file_B, folder)
    freqs_B = data_B['freqs']
    norm_sig_B = numpy.array(data_B['norm_avg_sig'])
    
    data_C = tool_belt.get_raw_data(file_C, folder)
    freqs_C= data_C['freqs']
    norm_sig_C = numpy.array(data_C['norm_avg_sig'])
    
    freqs_lin_A = numpy.linspace(freqs_A[0], freqs_A[-1], 1000)
    freqs_lin_B = numpy.linspace(freqs_B[0], freqs_B[-1], 1000)
    freqs_lin_C = numpy.linspace(freqs_C[0], freqs_C[-1], 1000)
    
        
    fig, ax = plt.subplots(dpi=300)
    fig.set_figwidth(fig_w)
    fig.set_figheight(fig_l)
    ax.set_ylim([0.953, 1.295])
    
    init_guess_A = [2.8366, 0.0012, 0.01,
                  2.9133, 0.0012, 0.01]
    
    init_guess_B = [2.8622, 0.0012, 0.01,
                  2.8844, 0.0012, 0.01]
    
    fit_function = double_lorentzian
    
    params_A, cov_arr = curve_fit(fit_function, freqs_A, norm_sig_A, init_guess_A)
    params_B, cov_arr = curve_fit(fit_function, freqs_B, norm_sig_B, init_guess_B)
        
    
    ax.plot(freqs_lin_A, fit_function(freqs_lin_A, *params_A),'b-',  color = green, 
            linewidth = fit_linewidth)
    ax.plot(freqs_A, norm_sig_A, 'b^',  color = green, markersize = marker_size, 
            linewidth = marker_line, mfc = light_green)
    ax.plot([], [], 'b-o', label = 'NV A', markersize = marker_size, linewidth = fit_linewidth)
    
    ax.plot(freqs_lin_B, fit_function(freqs_lin_B, *params_B), 'r-', color=orange, 
            linewidth = fit_linewidth)
    ax.plot(freqs_B, norm_sig_B, 'rs', color = orange, markersize = marker_size, 
            linewidth = marker_line, mfc = light_orange)
    ax.plot([], [], 'r-o', label = 'NV B', markersize = marker_size, linewidth = fit_linewidth)
    
    # +++++++++++++++++++++
    
        
    # init_guess_C= init_guess_A + init_guess_B
    # params, cov_arr = curve_fit(quad_lorentzian, freqs_C, norm_sig_C, init_guess_C)
    # ax.plot(freqs_C, norm_sig_C, 'go', markersize = marker_size)
    # ax.plot(freqs_lin_C, quad_lorentzian(freqs_lin_C, *params), 'g-', linewidth = fit_linewidth,
    #         label = 'NV A and NV B')
    
    
    # +++++++++++++++++++++
    ax.set_xlabel(r'Frequency, $\nu$ (GHz)',  fontsize = f_size)
    ax.set_ylabel(r'Normalized contrast, $\mathcal{C}$',  fontsize = f_size)
    ax.tick_params(which = 'both', length=fig_tick_l, width=fig_tick_w,
                            direction='in',grid_alpha=0.7, labelsize = f_size)
    # ax.legend(fontsize = f_size)
    
    
    return

# %%
def single_scc_esr(file, folder):
    data = tool_belt.get_raw_data(file, folder)
    freqs = data['freqs']
    norm_sig = numpy.array(data['norm_avg_sig'])
    
    
    freqs_lin = numpy.linspace(freqs[0], freqs[-1], 1000)
    
    fig_w = 1.6
    fig_l = 2
        
    fig, ax = plt.subplots()
    fig.set_figwidth(fig_w)
    fig.set_figheight(fig_l)
    
    init_guess = [2.8366, 0.0012, 0.01,
                    2.8622, 0.0012, 0.01,
                  2.8844, 0.0012, 0.01,
                  2.9133, 0.0012, 0.01]
    
    fit_function = quad_lorentzian
    
    params, cov_arr = curve_fit(fit_function, freqs, norm_sig, init_guess)
        
    
    ax.plot(freqs_lin, fit_function(freqs_lin, *params), 'k-', linewidth = 0.75)
    ax.plot(freqs, norm_sig, 'ko', markersize = 4,#1.2, 
            linewidth = 0.1, mfc = gray)
    
    
    ax.set_xlim([2.7858, 2.9586])
    # ax.set_ylim([0.951,1.248])
    ax.set_xticks([2.8, 2.85, 2.9, 2.95])
    ax.set_yticks([1, 1.05, 1.1])
    # ax.set_xlabel(r'$\nu$ (GHz)',  fontsize = f_size)
    # ax.set_ylabel(r'$\mathcal{C}$',  fontsize = f_size)
    ax.tick_params(which = 'both', length=fig_tick_l*2, width=fig_tick_w,
                            direction='in',grid_alpha=0.7, labelsize = f_size)
    
    return


# %%

if __name__ == '__main__':
    
    folder = 'pc_rabi/branch_master/super_resolution_pulsed_resonance/2021_09'
    
    file_A = '2021_09_29-02_09_19-johnson-dnv7_2021_09_23'
    file_B = '2021_09_29-12_39_34-johnson-dnv7_2021_09_23'
    file_C = '2021_09_30-09_20_45-johnson-dnv7_2021_09_23'
    compare_sr_esr(file_A,file_B , file_C,  folder)
    
    
    file = '2021_09_28-10_04_05-johnson-dnv7_2021_09_23-compilation'
    folder_scc = 'pc_rabi/branch_master/scc_pulsed_resonance/2021_09'
    single_scc_esr(file, folder_scc)