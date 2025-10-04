# -*- coding: utf-8 -*-
"""
Pulsed electron spin resonance on multiple NVs with spin-to-charge
conversion readout imaged onto a camera

Created on October 3rd, 2025

@author: saroj chand 
"""

import os
import sys
import time
import traceback
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np

from majorroutines.pulsed_resonance import fit_resonance, gaussian, norm_voigt, voigt
from majorroutines.widefield import base_routine
from utils import common
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils import widefield as widefield
from utils.constants import NVSig, NVSpinState
from utils.positioning import get_scan_1d as calculate_freqs


def create_raw_data_figure(nv_list, freqs, counts, counts_errs):
    fig, ax = plt.subplots()
    widefield.plot_raw_data(ax, nv_list, freqs, counts, counts_errs)
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Fraction in NV$^{-}$")
    return fig


def reformat_counts(counts):
    counts = np.array(counts)
    num_nvs = counts.shape[1]
    num_steps = counts.shape[3]
    adj_num_steps = num_steps // 4
    exp_ind = 0  # Everything, signal and ref, are under the same exp_rep for resonance

    sig_counts_0 = counts[exp_ind, :, :, 0:adj_num_steps, :]
    sig_counts_1 = counts[exp_ind, :, :, adj_num_steps : 2 * adj_num_steps, :]
    sig_counts = np.append(sig_counts_0, sig_counts_1, axis=3)
    ref_counts_0 = counts[exp_ind, :, :, 2 * adj_num_steps : 3 * adj_num_steps, :]
    ref_counts_1 = counts[exp_ind, :, :, 3 * adj_num_steps :, :]
    ref_counts = np.empty((num_nvs, num_runs, adj_num_steps, 2 * num_reps))
    ref_counts[:, :, :, 0::2] = ref_counts_0
    ref_counts[:, :, :, 1::2] = ref_counts_1

    reformatted_counts = np.stack((sig_counts, ref_counts))
    return reformatted_counts


def create_fit_figure(
    nv_list,
    freqs,
    norm_counts,
    norm_counts_ste,
    axes_pack=None,
    layout=None,
    no_legend=True,
    nv_inds=None,
    split_esr=None,
):
    ### Do the fitting

    num_nvs = len(nv_list)
    num_freqs = len(freqs)
    half_num_freqs = num_freqs // 2
    if nv_inds is None:
        nv_inds = list(range(num_nvs))
    num_nvs = len(nv_inds)

    def constant(freq):
        norm = 1
        if isinstance(freq, list):
            return [norm] * len(freq)
        if isinstance(freq, np.ndarray):
            return np.array([norm] * len(freq))
        else:
            return norm

    fit_fns = []
    pcovs = []
    popts = []
    center_freqs = []
    center_freq_errs = []

    do_fit = True
    if do_fit:
        for nv_ind in nv_inds:
            nv_counts = norm_counts[nv_ind]
            nv_counts_ste = norm_counts_ste[nv_ind]

            # Pre-processing

            guess_params = [1, 6, None, 1, 6, None]

            if nv_ind in split_esr:
                guess_params.append(0.002)

                def fit_fn(freq, *args):
                    half_splitting = args[-1] / 2
                    split_line_1 = gaussian(
                        freq, args[0], args[1], args[2] - half_splitting
                    ) + gaussian(freq, args[0], args[1], args[2] + half_splitting)
                    split_line_2 = gaussian(
                        freq, args[3], args[4], args[5] - half_splitting
                    ) + gaussian(freq, args[3], args[4], args[5] + half_splitting)
                    return split_line_1 + split_line_2
            else:

                def fit_fn(freq, *args):
                    return gaussian(freq, *args[0:3]) + gaussian(freq, *args[3:6])

            low_freq_guess = freqs[np.argmax(nv_counts[:half_num_freqs])]
            high_freq_guess = 2 * 2.87 - low_freq_guess
            guess_params[2] = low_freq_guess
            guess_params[5] = high_freq_guess
            num_params = len(guess_params)
            bounds = [[0] * num_params, [np.inf] * num_params]
            # Linewidth limits
            for ind in [1, 4]:
                bounds[0][ind] = 3
                bounds[1][ind] = 30
            if nv_ind in split_esr:
                bounds[1][-1] = 0.03

            # Do the fit

            # if num_resonances == 0:
            #     fit_fns.append(constant)
            #     popts.append([])
            # else:
            _, popt, pcov = fit_resonance(
                freqs,
                nv_counts,
                nv_counts_ste,
                fit_func=fit_fn,
                guess_params=guess_params,
                bounds=bounds,
            )

            # Tracking for plotting
            fit_fns.append(fit_fn)
            popts.append(popt)
            pcovs.append(pcov)

            # if num_resonances == 1:
            #     center_freqs.append(popt[2])
            #     center_freq_errs.append(np.sqrt(pcov[2, 2]))
            # elif num_resonances == 2:
            #     center_freqs.append((popt[2], popt[5]))
            center_freqs.append((popt[2], popt[5]))
    else:
        fit_fns = None
        popts = None

    # print(center_freqs)
    # print(center_freq_errs)
    nvb_freqs = []
    nva_freqs = []
    for ind in range(num_nvs):
        center_freq_pair = center_freqs[ind]
        if center_freq_pair[0] > 2.82:
            nvb_freqs.append(center_freqs[ind])
        else:
            nva_freqs.append(center_freqs[ind])
    nvb_mean_freqs = np.mean(nvb_freqs, axis=0)
    nva_mean_freqs = np.mean(nva_freqs, axis=0)
    # print(nvb_mean_freqs)
    # print(nva_mean_freqs)

    ### Make the figure

    if axes_pack is None:
        figsize = kpl.double_figsize
        figsize[1] = 7
        # figsize = [6.5, 4.0]
        # layout = kpl.calc_mosaic_layout(num_nvs, num_cols=6)
        layout = kpl.calc_mosaic_layout(6 * 19, num_cols=6, num_rows=19)
        layout[0] = [".", ".", ".", layout[0][3], layout[0][4], "."]
        layout[1] = [layout[1][0], ".", ".", *layout[1][3:]]
        fig, axes_pack = plt.subplot_mosaic(
            layout,
            figsize=figsize,
            sharex=True,
            sharey=True,
            gridspec_kw={"hspace": 0.015},
        )
    axes_pack_flat = list(axes_pack.values())

    widefield.plot_fit(
        axes_pack_flat,
        [nv_list[ind] for ind in nv_inds],
        freqs,
        norm_counts[nv_inds],
        norm_counts_ste[nv_inds],
        fit_fns,
        popts,
        no_legend=no_legend,
        # linestyle="solid",
        nv_inds=nv_inds,
    )

    ax = axes_pack[layout[-1, 0]]
    # ax = axes_pack[layout[-1, 3]]
    kpl.set_shared_ax_xlabel(ax, "Frequency (GHz)")
    # ax = axes_pack[layout[10, 0]]
    kpl.set_shared_ax_ylabel(ax, "NV$^{-}$ population (arb. units)")
    # ax = axes_pack[layout[-1, 0]]
    ax.set_xticks([2.80, 2.94])
    ax.set_xticks([2.87], minor=True)
    ax.set_yticks([0, 1], [None, None])
    gap = 0.008
    ax.set_xlim([np.min(freqs) - gap, np.max(freqs) + gap])
    ax.set_ylim([-0.2, 1.2])
    # ax.set_ylim([-0.3, 2])

    for ax in axes_pack_flat:
        # ax.tick_params(labelsize=kpl.FontSize.SMALL.value)
        # ax.tick_params(which="both", direction="in", labelsize=kpl.FontSize.SMALL.value)
        ax.tick_params(which="both", direction="in")

    for key in axes_pack.keys():
        ax = axes_pack[key]
        if key[1] in ["a", "b", "c"]:
            ax.axvline(nvb_mean_freqs[0], color=kpl.KplColors.LIGHT_GRAY, zorder=-50)
            ax.axvline(nvb_mean_freqs[1], color=kpl.KplColors.LIGHT_GRAY, zorder=-50)
        else:
            ax.axvline(nva_mean_freqs[0], color=kpl.KplColors.LIGHT_GRAY, zorder=-50)
            ax.axvline(nva_mean_freqs[1], color=kpl.KplColors.LIGHT_GRAY, zorder=-50)

    # ax = axes_pack[layout[-1, 0]]
    # ax.set_xlabel(" ")
    # fig.text(0.55, 0.01, "Frequency (GHz)", ha="center")
    # ax.set_ylabel(" ")
    # # label = "Normalized fraction in NV$^{-}$"
    # label = "Change in NV$^{-}$ fraction"
    # fig.text(0.005, 0.55, label, va="center", rotation="vertical")
    # # ax.set_ylim([0.945, 1.19])
    # # ax.set_yticks([1.0, 1.1, 1.2])
    # # ax.set_xticks([2.83, 2.87, 2.91])
    # x_buffer = 0.05 * (np.max(freqs) - np.min(freqs))
    # ax.set_xlim(np.min(freqs) - x_buffer, np.max(freqs) + x_buffer)
    # y_buffer = 0.05 * (np.max(norm_counts) - np.min(norm_counts))
    # ax.set_ylim(np.min(norm_counts) - y_buffer, np.max(norm_counts) + y_buffer)
    # return fig

def main(
    nv_list,
    num_steps,
    num_reps,          # this is your "signal reps"
    num_runs,
    freqs,
    uwave_ind_list=[0, 1],
    num_reps_ref=None, # if None → use max(1, num_reps//4)
    save_images=False,
):
    # ---- normalize uwave list to always have 2 entries ----
    if len(uwave_ind_list) == 1:
        uwave_ind_list = [uwave_ind_list[0], uwave_ind_list[0]]
    elif len(uwave_ind_list) > 2:
        uwave_ind_list = uwave_ind_list[:2]

    # ---- repetition counts for signal vs reference ----
    num_reps_sig = int(num_reps)
    if num_reps_ref is None:
        num_reps_ref = max(1, num_reps_sig // 4)
    num_reps_ref = int(num_reps_ref)

    # ---- setup ----
    pulse_gen = tb.get_server_pulse_gen()
    original_num_steps = int(num_steps)
    num_steps = int(num_steps) * 4  # Q1 sig0, Q2 sig1, Q3 ms=0 ref, Q4 ms=±1 ref
    seq_file = "resonance_dualgen.py"

    # ---- run_fn loads sequence with per-block reps inside QUA ----
    def run_fn(step_inds):
        seq_args = [
            widefield.get_base_scc_seq_args(nv_list, uwave_ind_list),
            step_inds,          # full 4-quarter step indices from base_routine
            num_reps_sig,       # signal quarters looping count (inside QUA)
            num_reps_ref,       # reference quarters looping count (inside QUA)
        ]
        seq_args_string = tb.encode_seq_args(seq_args)
        # stream_load with outer num_reps=1 because we loop inside QUA
        pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

    # ---- step_fn: which source is on/off at each quarter ----
    def step_fn(step_ind):
        quarter = num_steps // 4
        half    = num_steps // 2
        three_q = 3 * quarter

        if step_ind < quarter:
            # Q1: sweep SRC0
            freq = freqs[step_ind % original_num_steps]
            src0 = uwave_ind_list[0]
            uw0d = tb.get_virtual_sig_gen_dict(src0)
            g0   = tb.get_server_sig_gen(src0)
            g0.set_amp(uw0d["uwave_power"])
            g0.set_freq(freq)
            g0.uwave_on()

            # ensure SRC1 off
            src1 = uwave_ind_list[1]
            g1   = tb.get_server_sig_gen(src1)
            if src1 != src0:
                g1.uwave_off()

        elif step_ind < half:
            # Q2: sweep SRC1 (or the same SRC0 if duplicated)
            freq = freqs[step_ind % original_num_steps]
            src1 = uwave_ind_list[1]
            uw1d = tb.get_virtual_sig_gen_dict(src1)
            g1   = tb.get_server_sig_gen(src1)
            g1.set_amp(uw1d["uwave_power"])
            g1.set_freq(freq)
            g1.uwave_on()

            # ensure SRC0 off
            src0 = uwave_ind_list[0]
            g0   = tb.get_server_sig_gen(src0)
            if src1 != src0:
                g0.uwave_off()

        elif step_ind < three_q:
            # Q3: ms=0 ref → both off
            for u in set(uwave_ind_list[:2]):
                tb.get_server_sig_gen(u).uwave_off()

        else:
            # Q4: ms=±1 ref → fixed tones ON (both paths)
            for u in set(uwave_ind_list[:2]):
                uwd = tb.get_virtual_sig_gen_dict(u)
                g   = tb.get_server_sig_gen(u)
                g.set_amp(uwd["uwave_power"])
                g.set_freq(uwd["frequency"])
                g.uwave_on()

    # ---- run base routine (unchanged external API) ----
    raw_data = base_routine.main(
        nv_list,
        num_steps,
        num_reps=1,           # outer reps is 1; inner loops are inside QUA
        num_runs=num_runs,
        run_fn=run_fn,
        step_fn=step_fn,
        uwave_ind_list=uwave_ind_list,
        save_images=save_images,
        num_exps=1,
        ref_by_rep_parity=False,
    )

    # ---- save metadata ----
    timestamp = dm.get_time_stamp()
    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    repr_nv_name = repr_nv_sig.name
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)

    raw_data |= {
        "timestamp": timestamp,
        "freqs": np.asarray(freqs),
        "num_steps": num_steps,
        "original_num_steps": original_num_steps,
        "num_runs": num_runs,
        "num_reps_outer": 1,
        "num_reps_sig": num_reps_sig,
        "num_reps_ref": num_reps_ref,
        "uwave_ind_list": uwave_ind_list,           # normalized to 2 entries
        "sig_block_sources": [uwave_ind_list[0], uwave_ind_list[1]],
    }
    dm.save_raw_data(raw_data, file_path)

    # ---- optional plotting (keep as you had) ----
    try:
        pass
    except Exception:
        print(traceback.format_exc())

    tb.reset_cfm()
    kpl.show()


if __name__ == "__main__":
    kpl.init_kplotlib()
    # fmt: off
    exclude_inds1= [72, 64, 55, 96, 112, 87, 89, 114, 17, 12, 99, 116, 32, 107, 58, 36]
    exclude_inds2 = [12, 14, 11, 13, 52, 61, 116, 31, 32, 26, 87, 101, 105]
    # exclude_inds = exclude_inds1[:5] + exclude_inds2[:7]
    exclude_inds = exclude_inds1[:5]
    exclude_inds = list(set(exclude_inds))
    nv_inds = [ind for ind in range(117) if ind not in exclude_inds]
    nv_inds = None
    nva_inds = [0,1,2,6,8,9,10, 13, 19,20,23,25,28,31,32,33,35,36,38,39,42,43,44,46,48,50,56,57,61,62,63,64, 67,68,69,75,77,80,81,82, 85,86,87,88,90,91,92,95, 99,100,101,102,103,106,107,108,112, 113,114,116]  # Larger splitting
    nvb_inds = [3, 4, 5, 7, 11, 12, 14, 15, 16, 17, 18, 21, 22, 24, 26, 27, 29, 30, 34, 37, 40, 41, 45, 47, 49, 51, 52, 53, 54, 55, 58, 59, 60, 65, 66, 70, 71, 72, 73, 74, 76, 78, 79, 83, 84, 89, 93, 94, 96, 97, 98, 104, 105, 109, 110, 111, 115]  # Smaller splitting
    split_esr = [12, 13, 14, 61, 116] 
    broad_esr = [52, 11] 
    # weak_esr = [72, 64, 55, 96, 112, 87, 89, 114, 17, 12, 99, 116, 32, 107, 58, 36] 
    # weak_esr = weak_esr[:6]
    weak_esr = [72, 64, 55, 96, 112, 87, 12, 58, 36]
    # weak_esr = [72, 64, 55, 96, 112, 87]
    # weak_esr = []
    # split_esr = []
    # nv_inds = nva_inds
    for ind in weak_esr:
        for nv_list in [nva_inds, nvb_inds]:
            if ind in nv_list:
                nv_list.remove(ind)
    for issue_list in [broad_esr, split_esr]:
        for ind in issue_list:
            for nv_list in [nva_inds, nvb_inds]:
                if ind in nv_list:
                    nv_list.remove(ind)
                    nv_list.append(ind)
    # nv_inds = nva_inds + nvb_inds
    chunk_size = 3
    nv_inds = []
    max_length = max(len(nva_inds), len(nvb_inds))
    # Handle jagged
    for ind in range(2):
        nv_inds.append(nva_inds.pop(0))
    nv_inds.append(nvb_inds.pop(0))
    for ind in range(3):
        nv_inds.append(nva_inds.pop(0))
    nv_inds
    for ind in range(0, max_length, chunk_size):
        nv_inds.extend(nvb_inds[ind:ind + chunk_size])
        nv_inds.extend(nva_inds[ind:ind + chunk_size])  
    # nv_inds[-3:] = 
    # fmt: on

    file_id = "2025_09_23-12_36_13-rubin-nv0_2025_09_08"
    data = dm.get_raw_data(file_stem=file_id, load_npz=True, use_cache=True)
    nv_list = data["nv_list"]
    num_nvs = len(nv_list)
    num_steps = data["num_steps"]
    num_runs = data["num_runs"]
    num_reps = data["num_reps"]
    freqs = data["freqs"]

    # Manipulate the counts into the format expected for normalization
    counts = np.array(data.pop("counts"))
    reformatted_counts = reformat_counts(counts)
    sig_counts = reformatted_counts[0]
    ref_counts = reformatted_counts[1]

    norm_counts, norm_counts_ste = widefield.process_counts(
        nv_list, sig_counts, ref_counts, threshold=True
    )

    for nv_ind in split_esr:
        contrast = np.max(norm_counts[nv_ind])
        norm_counts[nv_ind] /= contrast
        norm_counts_ste[nv_ind] /= contrast

    # raw_fig = create_raw_data_figure(nv_list, freqs, avg_counts, avg_counts_ste)
    fit_fig = create_fit_figure(
        nv_list,
        freqs,
        norm_counts,
        norm_counts_ste,
        nv_inds=nv_inds,
        split_esr=split_esr,
    )

    kpl.show(block=True)
