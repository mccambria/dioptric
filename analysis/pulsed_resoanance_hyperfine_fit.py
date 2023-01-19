# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 12:20:06 2022

@author: kolkowitz
"""


import utils.tool_belt as tool_belt
import utils.kplotlib as kpl
import majorroutines.optimize as optimize
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import labrad
from utils.tool_belt import States, NormStyle
from random import shuffle
import sys

def process_counts(ref_counts, sig_counts, norm_style=NormStyle.SINGLE_VALUED):
    """Extract the normalized average signal at each data point.
    Since we sometimes don't do many runs (<10), we often will have an
    insufficient sample size to run stats on for norm_avg_sig calculation.
    We assume Poisson statistics instead.
    """

    ref_counts = np.array(ref_counts)
    sig_counts = np.array(sig_counts)

    num_runs, num_points = ref_counts.shape

    # Find the averages across runs
    sig_counts_avg = np.average(sig_counts, axis=0)
    single_ref_avg = np.average(ref_counts)
    ref_counts_avg = np.average(ref_counts, axis=0)

    sig_counts_ste = np.sqrt(sig_counts_avg) / np.sqrt(num_runs)
    single_ref_ste = np.sqrt(single_ref_avg) / np.sqrt(num_runs * num_points)
    ref_counts_ste = np.sqrt(ref_counts_avg) / np.sqrt(num_runs)

    if norm_style == NormStyle.SINGLE_VALUED:
        norm_avg_sig = sig_counts_avg / single_ref_avg
        norm_avg_sig_ste = norm_avg_sig * np.sqrt(
            (sig_counts_ste / sig_counts_avg) ** 2
            + (single_ref_ste / single_ref_avg) ** 2
        )
    elif norm_style == NormStyle.POINT_TO_POINT:
        norm_avg_sig = sig_counts_avg / ref_counts_avg
        norm_avg_sig_ste = norm_avg_sig * np.sqrt(
            (sig_counts_ste / sig_counts_avg) ** 2
            + (ref_counts_ste / ref_counts_avg) ** 2
        )

    return (
        ref_counts_avg,
        sig_counts_avg,
        norm_avg_sig,
        ref_counts_ste,
        sig_counts_ste,
        norm_avg_sig_ste,
    )

def calculate_freqs(freq_range, freq_center, num_steps):
    half_freq_range = freq_range / 2
    freq_low = freq_center - half_freq_range
    freq_high = freq_center + half_freq_range
    return np.linspace(freq_low, freq_high, num_steps)

def create_fit_figure(
    freq_range,
    freq_center,
    num_steps,
    norm_avg_sig,
    fit_func,
    popt,
    pcov,
    norm_avg_sig_ste=None,
):

    freqs = calculate_freqs(freq_range, freq_center, num_steps)
    smooth_freqs = calculate_freqs(freq_range, freq_center, 1000)

    fig, ax = plt.subplots()
    if norm_avg_sig_ste is not None:
        kpl.plot_points(ax, freqs, norm_avg_sig, yerr=norm_avg_sig_ste)
    else:
        kpl.plot_line(ax, freqs, norm_avg_sig)
    kpl.plot_line(
        ax,
        smooth_freqs,
        fit_func(smooth_freqs, *popt),
        color=kpl.KplColors.RED,
    )
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Normalized fluorescence")
    # ax.legend(loc="lower right")

    text = "f(mi=0) = {:.6f} +/- {:.6f} GHz"
    
    # text1 = text.format(*popt[0:3])
    # text2 = text.format(*popt[3:6])
    # text3 = text.format(*popt[6:9])

    # kpl.text(ax, 0.03, 0.8, text.format(popt[0], np.sqrt(pcov[0][0])) )
    kpl.anchored_text(ax, text.format(popt[0], np.sqrt(pcov[0][0])), kpl.Loc.LOWER_LEFT, size=kpl.Size.SMALL)
    # kpl.anchored_text(ax, text.format(popt[1], np.sqrt(pcov[1][1])), kpl.Loc.UPPER_RIGHT, size=kpl.Size.SMALL)
    # kpl.text(ax, 0.35, 0.8, text2)
    # kpl.text(ax, 0.7, 0.8, text3)

#    kpl.tight_layout(fig)

    return fig



def gaussian(freq, constrast, sigma, center):
    return constrast * np.exp(-((freq - center) ** 2) / (2 * (sigma**2)))

def tri_gaus(freqs,  center, hyperfine, a1, s1, a2, s2, a3, s3):
    c2 = center + hyperfine
    c3 = center - hyperfine
    gauss1 = gaussian(freqs, a1, s1, center)
    gauss2 = gaussian(freqs, a2, s2, c2)
    gauss3 = gaussian(freqs, a3, s3, c3)
    return 1.0 - gauss1 - gauss2 - gauss3

def six_gaus(freqs,  center1, center2, hyperfine, a1, s1, a2, s2, a3, s3, a4, s4, a5, s5, a6, s6):
    c12 = center1 + hyperfine
    c13 = center1 - hyperfine
    c22 = center2 + hyperfine
    c23 = center2 - hyperfine
    gauss1 = gaussian(freqs, a1, s1, center1)
    gauss2 = gaussian(freqs, a2, s2, c12)
    gauss3 = gaussian(freqs, a3, s3, c13)
    gauss4 = gaussian(freqs, a4, s4, center2)
    gauss5 = gaussian(freqs, a5, s5, c22)
    gauss6 = gaussian(freqs, a6, s6, c23)
    return 1.0 - gauss1 - gauss2 - gauss3- gauss4 - gauss5 - gauss6



def fit_resonance(
    freq_range,
    freq_center,
    num_steps,
    norm_avg_sig,
    norm_avg_sig_ste=None,
    ref_counts=None,
):

    hyperfine = 2.189288*1e-3
    fit_func = lambda freqs, center, a1, s1, a2, s2, a3, s3: tri_gaus(freqs,center, hyperfine, a1, s1, a2, s2, a3, s3)
    
    # hyperfine = 2.189288*1e-3
    # fit_func = lambda freqs, center1, center2,a1, s1, a2, s2, a3, s3,a4, s4, a5, s5, a6, s6: six_gaus(freqs,center1, center2, hyperfine, a1, s1, a2, s2, a3, s3,
    #                                                                    a4, s4, a5, s5, a6, s6)
    
    
    guess_params = [freq_center,
                    0.1, 0.0002, 
                    0.1, 0.0002,
                    0.1, 0.0002,
                    ]

    freqs = calculate_freqs(freq_range, freq_center, num_steps)
    
    popt, pcov = curve_fit(
        fit_func,
        freqs,
        norm_avg_sig,
        p0=guess_params,
    )
    # popt=guess_params
    # print(popt)
    print('{:.5f}'.format(popt[0]))
    # print(popt[1])
    print('{:.5f}'.format(np.sqrt(pcov[0][0])))

    return fit_func, popt, pcov


# %% Run the file


if __name__ == "__main__":
    kpl.init_kplotlib()
    
    file = "2023_01_18-16_52_19-siena-nv4_2023_01_16"
    data = tool_belt.get_raw_data(file)
    freq_center = data["freq_center"]
    freq_range = data["freq_range"]
    num_steps = data["num_steps"]
    ref_counts = data["ref_counts"]
    sig_counts = data["sig_counts"]
    num_reps = data["num_reps"]
    nv_sig = data["nv_sig"]
    # norm_style = NormStyle.point_to_point
    norm_style = NormStyle.SINGLE_VALUED

    ret_vals = process_counts(ref_counts, sig_counts, norm_style)
    (
        avg_ref_counts,
        avg_sig_counts,
        norm_avg_sig,
        ste_ref_counts,
        ste_sig_counts,
        norm_avg_sig_ste,
    ) = ret_vals
    fit_func, popt, pcov = fit_resonance(
        freq_range,
        freq_center,
        num_steps,
        norm_avg_sig,
        norm_avg_sig_ste,
        ref_counts,
    )
    create_fit_figure(
        freq_range,
        freq_center,
        num_steps,
        norm_avg_sig,
        fit_func,
        popt,
        pcov,
        norm_avg_sig_ste=norm_avg_sig_ste,
    )