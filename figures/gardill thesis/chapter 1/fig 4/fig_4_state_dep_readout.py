# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 12:08:27 2023

@author: kolkowitz
"""

import copy
import json
import os
import time
from random import shuffle

import labrad
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

import majorroutines.targeting as targeting
import utils.kplotlib as kpl
import utils.tool_belt as tool_belt
from utils.kplotlib import KplColors
from utils.tool_belt import States


def plot_readout_duration_optimization(max_readout, num_reps, sig_tags, ref_tags):
    """Generate two plots: 1, the total counts vs readout duration for each of
    the spin states; 2 the SNR vs readout duration
    """

    kpl.init_kplotlib()

    fig, axes_pack = plt.subplots(2, 1, figsize=(6, 10))

    num_points = 101
    readouts_with_zero = np.linspace(0, max_readout, num_points + 1)
    readouts = readouts_with_zero[1:]  # Exclude 0 ns

    # Integrate up the tags that fell before each readout duration under test
    integrated_sig_tags = []
    integrated_ref_tags = []
    zip_iter = zip(
        (sig_tags, ref_tags), (integrated_sig_tags.append, integrated_ref_tags.append)
    )
    for sorted_tags, integrated_append in zip_iter:
        current_readout_ind = 0
        current_readout = readouts[current_readout_ind]
        for ind in range(len(sorted_tags)):
            # Cycle through readouts until the tag falls within the readout or
            # we run out of readouts
            while sorted_tags[ind] > current_readout:
                integrated_append(ind)
                current_readout_ind += 1
                if current_readout_ind == num_points:
                    break
                current_readout = readouts[current_readout_ind]
            if current_readout_ind == num_points:
                break
        # If we got to the end of the counts and there's still readouts left
        # (eg the experiment went dark for some reason), pad with the final ind
        while current_readout_ind < num_points:
            integrated_append(ind)
            current_readout_ind += 1

    # Calculate the snr per readout for each readout duration
    snr_per_readouts = []
    for sig, ref in zip(integrated_sig_tags, integrated_ref_tags):
        # Assume Poisson statistics on each count value
        sig_noise = np.sqrt(sig)
        ref_noise = np.sqrt(ref)
        snr = (ref - sig) / np.sqrt(sig_noise**2 + ref_noise**2)
        # print(snr)
        snr_per_readouts.append(snr / np.sqrt(num_reps))
    # print(snr_per_readouts)

    sig_hist, bin_edges = np.histogram(sig_tags, bins=readouts_with_zero)
    ref_hist, bin_edges = np.histogram(ref_tags, bins=readouts_with_zero)
    readout_window = round(readouts_with_zero[1] - readouts_with_zero[0])
    readout_window_sec = readout_window * 10**-9
    sig_rates = sig_hist / (readout_window_sec * num_reps * 1000)
    # print(integrated_sig_tags)
    ref_rates = ref_hist / (readout_window_sec * num_reps * 1000)
    # print(integrated_ref_tags)
    bin_centers = (readouts_with_zero[:-1] + readouts) / 2

    ax = axes_pack[0]
    kpl.plot_line(
        ax, bin_centers, sig_rates, color=KplColors.GREEN, label=r"$m_{s}=\pm 1$"
    )
    kpl.plot_line(ax, bin_centers, ref_rates, color=KplColors.RED, label=r"$m_{s}=0$")
    ax.set_ylabel("Count rate (kcps)")
    ax.set_xlabel("Time since readout began (ns)")
    ax.legend()

    # ax2=ax.twinx()

    ax = axes_pack[1]
    kpl.plot_line(ax, readouts, snr_per_readouts)
    # ax.set_xlabel("Readout duration (ns)")
    ax.set_ylabel("SNR per sqrt(readout)")
    max_snr = tool_belt.round_sig_figs(max(snr_per_readouts), 3)
    optimum_readout = round(readouts[np.argmax(snr_per_readouts)])
    text = f"Max SNR: {max_snr} at {optimum_readout} ns"
    # kpl.anchored_text(ax, text, kpl.Loc.LOWER_LEFT)

    return fig


file = "2023_02_16-20_15_12-siena-nv0_2023_02_16"

file_name = file + ".txt"
with open(file_name) as f:
    data = json.load(f)

sig_tags = data["sig_tags"]
ref_tags = data["ref_tags"]
num_reps = data["num_reps"]
nv_sig = data["nv_sig"]
max_readout = nv_sig["spin_readout_dur"]
plot_readout_duration_optimization(max_readout, num_reps, sig_tags, ref_tags)
