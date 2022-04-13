# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 10:36:45 2022

@author: agard
"""


import utils.tool_belt as tool_belt
import numpy
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


fig_tick_l = 3
fig_tick_w = 0.75

f_size = 8


def calculate_freqs(freq_range, freq_center, num_steps):
    half_freq_range = freq_range / 2
    freq_low = freq_center - half_freq_range
    freq_high = freq_center + half_freq_range
    return numpy.linspace(freq_low, freq_high, num_steps)

def gaussian(freq, constrast, sigma, center):
    return constrast * numpy.exp(-((freq - center) ** 2) / (2 * (sigma ** 2)))


def double_gaussian_dip(
    freq,
    low_constrast,
    low_sigma,
    low_center,
    high_constrast,
    high_sigma,
    high_center,
):
    low_gauss = gaussian(freq, low_constrast, low_sigma, low_center)
    high_gauss = gaussian(freq, high_constrast, high_sigma, high_center)
    return 1.0 - low_gauss - high_gauss


def single_gaussian_dip(freq, constrast, sigma, center):
    return 1.0 - gaussian(freq, constrast, sigma, center)


def lorentzian(x, x0, A, L):
    x_center = x - x0
    return 1 - A * 0.5*L / (x_center**2 + (0.5*L)**2)


def double_lorentzian(x, x0_1, A_1, L_1, x0_2, A_2, L_2):
    return lorentzian(x, x0_1, A_1, L_1) + \
        lorentzian(x, x0_2, A_2, L_2) - 1
        
def create_fit_figure(
    freq_range, freq_center, num_steps, norm_avg_sig, fit_func, popt, magnet_angle
):

    freqs = calculate_freqs(freq_range, freq_center, num_steps)
    smooth_freqs = calculate_freqs(freq_range, freq_center, 1000)

    fig_w =3
    fig_l = fig_w * 0.8
    fig, ax = plt.subplots()
    fig.set_figwidth(fig_w)
    fig.set_figheight(fig_l)
    ax.plot(smooth_freqs, fit_func(smooth_freqs, *popt), "r-", label="fit")
    ax.plot(freqs, norm_avg_sig, "bo", linewidth = 1, markersize = 3, mfc='white')
    ax.set_xlabel("Frequency (GHz)", fontsize = f_size)
    ax.set_ylabel("Normalized fluorescence", fontsize = f_size)
    
    # ax.set_xticks([2.7, 2.8, 2.9, 3.0])
    ax.set_xticks([2.75, 2.875, 3])
    # ax.set_ylim([0.67, 1.18])
    
    ax.tick_params(which = 'both', length=fig_tick_l, width=fig_tick_w,
                        direction='in',grid_alpha=0.7, labelsize = f_size)
    
    text = "{:.0f}".format(magnet_angle) + u"\N{DEGREE SIGN}"

    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(
        0.8,
        0.15,
        text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=props,
    )
    

    fig.canvas.draw()
    fig.set_tight_layout(True)
    fig.canvas.flush_events()

    return fig


def fit_resonance(
    freq_range,
    freq_center,
    num_steps,
    norm_avg_sig,
    guess_params,
    norm_avg_sig_ste=None,
    ref_counts=None,
):

    freqs = calculate_freqs(freq_range, freq_center, num_steps)
    
    fit_func = double_lorentzian
    try:
        popt, pcov = curve_fit(
            fit_func, freqs, norm_avg_sig, p0=guess_params,
            bounds = ([2.6, 0, 0, 2.87, 0, 0],[2.88, 0.4, 1, 3.2, 0.4, 1])
        )
        print(popt)
        #x0_1, A_1, L_1, x0_2, A_2, L_2
    except Exception as e:
        print(e)
        popt = guess_params
        pcov = None

    return fit_func, popt, pcov


if __name__ == "__main__":

    # folder = "pc_rabi/branch_master/pulsed_resonance/2021_09"
    # # file = '2021_09_15-13_30_13-johnson-dnv0_2021_09_09'
    # file_list = ["2021_09_27-13_52_00-johnson-dnv7_2021_09_23"]
    # label_list = ["Point A", "Point B", "Point C"]

    # fig, ax = plt.subplots(figsize=(8.5, 8.5))
    # for f in range(len(file_list)):
    #     file = file_list[f]
    #     data = tool_belt.get_raw_data(file, folder)

    #     freq_center = data["freq_center"]
    #     freq_range = data["freq_range"]
    #     num_steps = data["num_steps"]
    #     num_runs = data["num_runs"]
    #     norm_avg_sig = data["norm_avg_sig"]

    #     freqs = calculate_freqs(freq_range, freq_center, num_steps)

    #     ax.plot(freqs, norm_avg_sig, label=label_list[f])
    #     ax.set_xlabel("Frequency (GHz)")
    #     ax.set_ylabel("Contrast (arb. units)")
    #     ax.legend(loc="lower right")

    # fit_func, popt, pcov = fit_resonance(freq_range, freq_center, num_steps,
    #                                       norm_avg_sig, norm_avg_sig_ste)


    file_list = [
        '2021_09_30-21_56_35-johnson-dnv5_2021_09_23',
        '2021_09_30-22_16_41-johnson-dnv5_2021_09_23',
              '2021_09_30-22_36_48-johnson-dnv5_2021_09_23',
              '2021_09_30-22_56_56-johnson-dnv5_2021_09_23',
                '2021_09_30-23_17_05-johnson-dnv5_2021_09_23',
                '2021_09_30-23_37_12-johnson-dnv5_2021_09_23'
                 ]
    file = "2021_09_30-22_36_48-johnson-dnv5_2021_09_23"
    
    for f in range(len(file_list)):
        file = file_list[f]
        folder = 'pc_rabi/branch_master/pulsed_resonance/2021_09'
        data = tool_belt.get_raw_data(file, folder)
        nv_sig = data['nv_sig']
        freq_center = data["freq_center"]
        freq_range = data["freq_range"]
        num_steps = data["num_steps"]
        num_runs = data["num_runs"]
        norm_avg_sig = numpy.array(data["norm_avg_sig"])
        ref_counts = numpy.array(data["ref_counts"])
        magnet_angle = nv_sig['magnet_angle']
    
    #x0_1, A_1, L_1, x0_2, A_2, L_2 
        low_freqs = [2.79,2.79,2.79,2.84,2.8626,2.82,]
        high_freqs = [2.95,2.95,2.95,2.91,2.8849,2.92,]
    
        guess_params =[low_freqs[f], 0.0015, 0.01 ,
                       high_freqs[f],0.0014, 0.01, 
            ]
        
        
        fit_func, popt, pcov = fit_resonance(
            freq_range, freq_center, num_steps, norm_avg_sig, 
            guess_params,
        )
    
    
        create_fit_figure(
            freq_range, freq_center, num_steps, norm_avg_sig, fit_func, popt, magnet_angle
                )
