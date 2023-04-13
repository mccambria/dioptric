# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 21:04:38 2023

@author: kolkowitz
"""



import utils.tool_belt as tool_belt
import numpy as np
import matplotlib.pyplot as plt
import utils.kplotlib as kpl
from utils.kplotlib import KplColors
from utils.kplotlib import Size
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json
import random
import copy
import majorroutines.charge_majorroutines.photonstatistics as model


def gaussian(freq, constrast, sigma, center):
    return 1+constrast * np.exp(-((freq - center) ** 2) / (2 * (sigma**2)))

def calculate_freqs(freq_range, freq_center, num_steps):
    half_freq_range = freq_range / 2
    freq_low = freq_center - half_freq_range
    freq_high = freq_center + half_freq_range
    return np.linspace(freq_low, freq_high, num_steps)

def fit_resonance(
    freq_range,
    freq_center,
    num_steps,
    norm_avg_sig,
    norm_avg_sig_ste=None,
    ref_counts=None,
):
    fit_func = lambda freqs, a1, s1, c1 : gaussian(freqs,  a1, s1, c1)


    guess_params = [0.1, 0.01,  2.833]
    freqs = calculate_freqs(freq_range, freq_center, num_steps)
    popt, pcov = curve_fit(
        fit_func,
        freqs,
        norm_avg_sig,
        p0=guess_params,
    )
    # popt=guess_params
    # pcov=0
    return fit_func, popt, pcov

def calc_histogram(nv0, nvm, dur, bins=None):

    # Counts are in us, readout is in ns
    dur_us = dur / 1e3
    # print(nv0)
    nv0_counts = [
        np.count_nonzero(np.array(rep) < dur_us) for rep in nv0
    ]  
    nvm_counts = [
        np.count_nonzero(np.array(rep) < dur_us) for rep in nvm
    ]  
    # print(nv0_counts)
    max_0 = max(nv0_counts)
    max_m = max(nvm_counts)
    if bins == None:

        occur_0, bin_edges_0 = np.histogram(
            nv0_counts, np.linspace(0, max_0, max_0 + 1)  # 200)  #
        )
        occur_m, bin_edge_m = np.histogram(
            nvm_counts, np.linspace(0, max_m, max_m + 1)  # 200)  #
        )
    elif bins != None:
        occur_0, bin_edges_0 = np.histogram(
            nv0_counts, bins  # np.linspace(0, max_0, max_0 + 1) #200)  #
        )
        occur_m, bin_edge_m = np.histogram(
            nvm_counts, bins  # np.linspace(0, max_m,  max_m + 1) #200)  #
        )

    # Histogram returns bin edges. A bin is defined with the first point
    # inclusive and the last exclusive - eg a count a 2 will fall into
    # bin [2,3) - so just drop the last bin edge for our x vals
    x_vals_0 = bin_edges_0[:-1]
    x_vals_m = bin_edge_m[:-1]

    return occur_0, x_vals_0, occur_m, x_vals_m



def calculate_threshold_with_model(
    readout_time, nv0_array, nvm_array, max_x_val, power, nd_filter=None,plot_model_hists=True
):
    """
    Using the histograms of the NV- and NV0 measurement, and modeling them as
    an NV perfectly prepared in either NV- or NV0, detemines the optimum
    value of single shot counts to determine either NV- or NV0.

    the fit finds
    mu_0 = the mean counts of NV0
    mu_m = the mean counts of NV-
    fidelity = given the threshold, tthe fidelity is related to how accurate
        we can identify the charge state from a single shot measurement.
        Best to shoot for values of > 80%
    threshold = the number of counts that, above this value, identify the charge
        state as NV-. And below this value, identify as NV0.
    """
    tR = readout_time / 10 ** 6
    tR_us = readout_time / 10 ** 3
    fit_rate = single_nv_photon_statistics_model(tR, nv0_array, nvm_array,do_plot=plot_model_hists)
    max_x_val = int(max_x_val)
    x_data = np.linspace(0, 100, 101)
    threshold_list, fidelity_list, thresh_para = model.calculate_threshold(tR, x_data, fit_rate)
    mu_0 = fit_rate[3] * tR
    mu_m = fit_rate[2] * tR
    fidelity = thresh_para[1]
    threshold = thresh_para[0]
    # print(title_text)
    # print(threshold_list)
    print("Threshold: {} counts, fidelity: {:.3f}".format(threshold, fidelity))

    if plot_model_hists:

        plot_x_data = np.linspace(0, max_x_val, max_x_val + 1)
        fig3, ax = plt.subplots()
        ax.plot(
            plot_x_data,
            model.get_PhotonNV0_list(plot_x_data, tR, fit_rate, 0.5),
            "-o",
        )
        ax.plot(
            plot_x_data,
            model.get_PhotonNVm_list(plot_x_data, tR, fit_rate, 0.5),
            "-o",
        )
        plt.axvline(x=thresh_para[0], color="red")
        # mu_0 = fit_rate[3] * tR
        # mu_m = fit_rate[2] * tR
        textstr = "\n".join(
            (
                r"$\mu_0=%.2f$" % (mu_0),
                r"$\mu_-=%.2f$" % (mu_m),
                r"$fidelity =%.2f$" % (thresh_para[1]),
                r"$threshold = %.1f$" % (thresh_para[0],),
            )
        )
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        ax.text(
            0.65,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=props,
        )
        if nd_filter:
            title_text = "{} us readout, {} V, {}".format(
                int(tR_us), power, nd_filter
            )
        else:
            title_text = "{} us readout, {} V".format(int(tR_us), power)
        ax.set_title(title_text)
        plt.xlabel("Number of counts")
        plt.ylabel("Probability Density")
        return threshold_list,fidelity_list, threshold, fidelity, mu_0, mu_m, fig3

    else:
        # print('i made it here too')
        return threshold_list,fidelity_list, threshold, fidelity, mu_0, mu_m, ''


def single_nv_photon_statistics_model(readout_time, NV0, NVm, do_plot=True):
    """
    A function to take the NV histograms after red and green initialization,
    and use a model to plot the expected histograms if the NV is perfectly
    initialized in NV- or NV0

    for the fit,
    g0 =  Ionization rate from NV- to NV0
    g1 = recombination rate from NV0 to NV-
    y1 = fluorescnece rate of NV1
    y0 = Fluorescence rate of NV0
    """
    NV0_hist = np.array(NV0)
    NVm_hist = np.array(NVm)
    tR = readout_time
    combined_hist = NVm_hist.tolist() + NV0_hist.tolist()
    random.shuffle(combined_hist)

    # fit = [g0,g1,y1,y0]
    guess = [10 * 10 ** -4, 100 * 10 ** -4, 1000 * 10 ** -4, 500 * 10 ** -4]
    fit, dev = model.get_curve_fit(tR, 0, combined_hist, guess)

    if do_plot:
        u_value0, freq0 = model.get_Probability_distribution(NV0_hist.tolist())
        u_valuem, freqm = model.get_Probability_distribution(NVm_hist.tolist())
        u_value2, freq2 = model.get_Probability_distribution(combined_hist)
        curve = model.get_photon_distribution_curve(
            tR, u_value2, fit[0], fit[1], fit[2], fit[3]
        )

        A1, A1pcov = model.get_curve_fit_to_weight(
            tR, 0, NV0_hist.tolist(), [0.5], fit
        )
        A2, A2pcov = model.get_curve_fit_to_weight(
            tR, 0, NVm_hist.tolist(), [0.5], fit
        )

        nv0_curve = model.get_photon_distribution_curve_weight(
            u_value0, tR, fit[0], fit[1], fit[2], fit[3], A1[0]
        )
        nvm_curve = model.get_photon_distribution_curve_weight(
            u_valuem, tR, fit[0], fit[1], fit[2], fit[3], A2[0]
        )
        fig4, ax = plt.subplots()
        ax.plot(u_value0, 0.5 * np.array(freq0), "-ro")
        ax.plot(u_valuem, 0.5 * np.array(freqm), "-go")
        ax.plot(u_value2, freq2, "-bo")
        ax.plot(u_value2, curve)
        ax.plot(u_valuem, 0.5 * np.array(nvm_curve), "green")
        ax.plot(u_value0, 0.5 * np.array(nv0_curve), "red")
        textstr = "\n".join(
            (
                r"$g_0(s^{-1}) =%.2f$" % (fit[0] * 10 ** 3,),
                r"$g_1(s^{-1})  =%.2f$" % (fit[1] * 10 ** 3,),
                r"$y_0 =%.2f$" % (fit[3],),
                r"$y_1 =%.2f$" % (fit[2],),
            )
        )
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        ax.text(
            0.6,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=props,
        )
        plt.xlabel("Number of counts")
        plt.ylabel("Probability Density")
        plt.show()
    return fit


def do_plot_hist(data): 
    # DOESNT WORK
    nv0_counts = data['nv0']
    nvm_counts = data['nvm']
    nv_sig = data['nv_sig']
    readout_dur = nv_sig['charge_readout_dur']
    bins =None
    
    occur_0, x_vals_0, occur_m, x_vals_m = calc_histogram(
        nv0_counts, nvm_counts, readout_dur,bins,
    )
    # print(occur_m)

    max_x_val = max(list(x_vals_0) + list(x_vals_m)) + 10

    num_reps = len(nv0_counts)
    mean_0 = sum(occur_0 * x_vals_0) / num_reps
    mean_m = sum(occur_m * x_vals_m) / num_reps

    # print('i made it here')
    dur_us = readout_dur / 1e3
    nv0_counts_list = [
        np.count_nonzero(np.array(rep) < dur_us) for rep in nv0_counts
    ]
    nvm_counts_list = [
        np.count_nonzero(np.array(rep) < dur_us) for rep in nvm_counts
    ]
    threshold_list, fidelity_list, threshold, fidelity, mu_0, mu_m, fig = calculate_threshold_with_model(
        readout_dur,
        nv0_counts_list,
        nvm_counts_list,
        max_x_val,
        0.2,
        plot_model_hists = True
    )


    return threshold_list, fidelity_list, threshold, fidelity, nv0_counts,nvm_counts

def do_plot_scc_esr(data):
    sig_counts_avg = data['sig_counts_avg']
    ref_count_raw = data['ref_count_raw']
    freqs = data['test_pulse_dur_list']
    norm_avg_sig = sig_counts_avg /np.average(ref_count_raw)
    comb=list(zip(norm_avg_sig,freqs))
    comb=sorted(comb, key=lambda i:i[1])
    norm_avg_sig,freqs  = [[i for i, j in comb],
        [j for i, j in comb]]
    
    freq_range = freqs[-1]-freqs[0]
    freq_center = freqs[0] + freq_range/2
    num_steps = len(freqs)
    
    smooth_freqs = np.linspace(freqs[0], freqs[-1], 1000)
    
    fit_func, popt , _= fit_resonance(
        freq_range,
        freq_center,
        num_steps,
        norm_avg_sig,
        )
    
        
    fig_tick_l = 3
    fig_tick_w = 0.75
    f_size = 8
    fig_w = 1.7
    fig_l = fig_w * 0.75


    fig3, ax = plt.subplots()
    fig3.set_figwidth(fig_w)
    fig3.set_figheight(fig_l)
    kpl.plot_line(
        ax, smooth_freqs, fit_func(smooth_freqs,*popt), color=KplColors.RED
    )
    
    kpl.plot_points(ax, freqs, norm_avg_sig,  color=KplColors.BLUE, size=Size.TINY)
    # ax.plot(freqs, norm_avg_sig,  '.', color = KplColors.BLUE)
    
    ax.tick_params(which = 'both', #length=fig_tick_l, width=fig_tick_w,
                    colors='k',
                        direction='in',grid_alpha=0.7, labelsize = f_size)
    ax.set_ylabel("Norm. fluor.", fontsize = f_size)
    ax.set_xlabel(r"MW frequency, $\nu$ (GHz)", fontsize = f_size)
                          
    from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)                                                                                                                                                                                                   
    ax.xaxis.set_major_locator(MultipleLocator(0.02)) # Intervals for major x-ticks                                                                                                                                                                                                                     
    ax.xaxis.set_minor_locator(MultipleLocator(0.005)) # Minor ticks : Automatic filling based on the ytick range                                                                                                                                                                                                                        
    ax.yaxis.set_major_locator(MultipleLocator(0.2))  # For y-ticks                                                                                                                                                                                                                     
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))  
    
#%%
    
file_hist = '2021_09_18-08_24_20-johnson-nv1_2021_09_07'
file_name = file_hist + '.txt'
with open(file_name) as f:
    data = json.load(f)
# do_plot_hist(data) 

file_scc = '2021_09_21-10_36_20-johnson-nv1_2021_09_07-esr_test'
file_name = file_scc + '.txt'
with open(file_name) as f:
    data = json.load(f)
do_plot_scc_esr(data)