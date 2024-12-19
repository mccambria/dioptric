# -*- coding: utf-8 -*-
"""
Pulsed electron spin resonance on multiple NVs with spin-to-charge
conversion readout imaged onto a camera

Created on November 19th, 2023

@author: mccambria
"""

import os
import sys
import time
import traceback
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np

from majorroutines.pulsed_resonance import fit_resonance, norm_voigt, voigt, voigt_split
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
    counts,
    counts_ste,
    norms,
    axes_pack=None,
    layout=None,
    no_legend=True,
):
    ### Do the fitting

    num_nvs = len(nv_list)
    num_freqs = len(freqs)
    half_num_freqs = num_freqs // 2

    def constant(freq):
        norm = 1
        if isinstance(freq, list):
            return [norm] * len(freq)
        if isinstance(freq, np.ndarray):
            return np.array([norm] * len(freq))
        else:
            return norm

    norms_ms0_newaxis = norms[0][:, np.newaxis]
    norms_ms1_newaxis = norms[1][:, np.newaxis]
    contrast = norms_ms1_newaxis - norms_ms0_newaxis
    contrast = np.where(contrast > 0.05, contrast, 0.05)
    norm_counts = (counts - norms_ms0_newaxis) / contrast
    norm_counts_ste = counts_ste / contrast
    #
    # norm_counts = counts - norms_ms0_newaxis
    # norm_counts_ste = counts_ste
    #
    # norm_counts = (counts / norms_ms0_newaxis) - 1
    # norm_counts_ste = counts_ste / norms_ms0_newaxis

    fit_fns = []
    pcovs = []
    popts = []
    center_freqs = []
    center_freq_errs = []

    do_fit = False
    if do_fit:
        for nv_ind in range(num_nvs):
            nv_counts = norm_counts[nv_ind]
            nv_counts_ste = norm_counts_ste[nv_ind]
            amp_guess = 1 - np.max(nv_counts)

            # if nv_ind in [3, 5, 7, 10, 12]:
            #     num_resonances = 1
            # if nv_ind in [0, 1, 2, 4, 6, 11, 14]:
            #     num_resonances = 2
            # else:
            #     num_resonances = 0
            # num_resonances = 1
            num_resonances = 2

            if num_resonances == 1:
                guess_params = [amp_guess, 5, 5, np.median(freqs)]
                bounds = [[0] * 4, [np.inf] * 4]
                # Limit linewidths
                for ind in [1, 2]:
                    bounds[1][ind] = 10

                def fit_fn(freq, g_width, l_width, center):
                    return norm_voigt(freq, g_width, l_width, center)
            elif num_resonances == 2:
                # Find peaks in left and right halves
                low_freq_guess = freqs[np.argmax(nv_counts[:half_num_freqs])]
                high_freq_guess = freqs[
                    np.argmax(nv_counts[half_num_freqs:]) + half_num_freqs
                ]
                # low_freq_guess = freqs[num_steps * 1 // 3]
                # high_freq_guess = freqs[num_steps * 2 // 3]
                # low_freq_guess = 2.85
                # high_freq_guess = 2.89

                guess_params = [5, 5, low_freq_guess]
                guess_params.extend([5, 5, high_freq_guess])
                num_params = len(guess_params)
                bounds = [[0] * num_params, [np.inf] * num_params]
                # Limit linewidths
                for ind in [0, 1, 3, 4]:
                    bounds[1][ind] = 10

                def fit_fn(freq, *args):
                    return norm_voigt(freq, *args[:3]) + norm_voigt(freq, *args[3:])
                    # return 1 + voigt(freq, *args[:4]) + voigt(freq, *args[4:])

            if num_resonances == 0:
                fit_fns.append(constant)
                popts.append([])
            else:
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

            if num_resonances == 1:
                center_freqs.append(popt[2])
                center_freq_errs.append(np.sqrt(pcov[2, 2]))
            elif num_resonances == 2:
                center_freqs.append((popt[2], popt[5]))
    else:
        fit_fns = None
        popts = None

    # print(center_freqs)
    # print(center_freq_errs)

    ### Make the figure

    if axes_pack is None:
        # figsize = [6.5, 6.0]
        figsize = [6.5, 5.0]
        figsize[0] *= 3
        figsize[1] *= 3
        # figsize = [6.5, 4.0]
        layout = kpl.calc_mosaic_layout(num_nvs, num_rows=None)
        fig, axes_pack = plt.subplot_mosaic(
            layout, figsize=figsize, sharex=True, sharey=True
        )
    axes_pack_flat = list(axes_pack.values())

    widefield.plot_fit(
        axes_pack_flat,
        nv_list,
        freqs,
        norm_counts,
        norm_counts_ste,
        fit_fns,
        popts,
        no_legend=no_legend,
        # linestyle="solid",
    )

    ax = axes_pack[layout[-1, 0]]
    kpl.set_shared_ax_xlabel(ax, "Frequency (GHz)")
    kpl.set_shared_ax_ylabel(ax, "Normalized NV$^{-}$ population")
    # kpl.set_shared_ax_ylabel(ax, "Norm. NV$^{-}$ pop.")
    # kpl.set_shared_ax_ylabel(ax, "Relative change in fluorescence")
    ax.set_xticks([2.80, 2.95])
    ax.set_yticks([0, 1])
    ax.set_ylim([-0.3, 1.3])

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
    nv_list: list[NVSig],
    num_steps,
    num_reps,
    num_runs,
    freq_center,
    freq_range,
    uwave_ind_list=[0, 1],
):
    ### Some initial setup
    pulse_gen = tb.get_server_pulse_gen()
    freqs = calculate_freqs(freq_center, freq_range, num_steps)
    original_num_steps = num_steps
    num_steps *= 4  # For sig, ms=0 ref, and ms=+/-1 ref

    seq_file = "resonance_ref2.py"

    ### Collect the data

    def run_fn(step_inds):
        seq_args = [widefield.get_base_scc_seq_args(nv_list, uwave_ind_list), step_inds]
        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

    # def step_fn(step_ind):
    #     freq = freqs[step_ind]
    #     sig_gen.set_freq(freq)

    def step_fn(step_ind):
        if step_ind < (1 / 2) * num_steps:
            freq = freqs[step_ind % original_num_steps]

            uwave_ind = uwave_ind_list[0]
            uwave_dict = tb.get_virtual_sig_gen_dict(uwave_ind)
            sig_gen = tb.get_server_sig_gen(uwave_ind)
            sig_gen.set_amp(uwave_dict["uwave_power"])
            sig_gen.set_freq(freq)
            sig_gen.uwave_on()

            uwave_ind = uwave_ind_list[1]
            sig_gen = tb.get_server_sig_gen(uwave_ind)
            sig_gen.uwave_off()

        elif step_ind < (3 / 4) * num_steps:  # ms=0 ref
            for uwave_ind in uwave_ind_list:
                sig_gen = tb.get_server_sig_gen(uwave_ind)
                sig_gen.uwave_off()
        else:  # ms=+/-1 ref
            for uwave_ind in uwave_ind_list:
                uwave_dict = tb.get_virtual_sig_gen_dict(uwave_ind)
                sig_gen = tb.get_server_sig_gen(uwave_ind)
                sig_gen.set_amp(uwave_dict["uwave_power"])
                sig_gen.set_freq(uwave_dict["frequency"])
                sig_gen.uwave_on()

    raw_data = base_routine.main(
        nv_list,
        num_steps,
        num_reps,
        num_runs,
        run_fn,
        step_fn,
        uwave_ind_list=uwave_ind_list,
        save_images=True,
        num_exps=1,
        ref_by_rep_parity=False,
    )

    ### Process and plot

    try:
        counts = data["counts"]
        reformatted_counts = reformat_counts(counts)
        sig_counts = reformatted_counts[0]
        ref_counts = reformatted_counts[1]

        avg_counts, avg_counts_ste, norms = widefield.process_counts(
            nv_list, sig_counts, ref_counts, threshold=True
        )

        # raw_fig = create_raw_data_figure(nv_list, freqs, avg_counts, avg_counts_ste)
        fit_fig = create_fit_figure(nv_list, freqs, avg_counts, avg_counts_ste, norms)
    except Exception:
        print(traceback.format_exc())
        raw_fig = None
        fit_fig = None

    ### Clean up and return

    tb.reset_cfm()
    kpl.show()

    timestamp = dm.get_time_stamp()
    raw_data |= {
        "timestamp": timestamp,
        "freqs": freqs,
        "freq-units": "GHz",
        "freq_range": freq_range,
        "freq_center": freq_center,
    }

    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    repr_nv_name = repr_nv_sig.name
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    if "img_arrays" in raw_data:
        keys_to_compress = ["img_arrays"]
    else:
        keys_to_compress = None
    dm.save_raw_data(raw_data, file_path, keys_to_compress)
    if raw_fig is not None:
        dm.save_figure(raw_fig, file_path)
    if fit_fig is not None:
        file_path = dm.get_file_path(__file__, timestamp, repr_nv_name + "-fit")
        dm.save_figure(fit_fig, file_path)


if __name__ == "__main__":
    kpl.init_kplotlib()

    file_id = 1729211906249

    data = dm.get_raw_data(file_id=file_id, load_npz=True, use_cache=False)
    img_arrays = np.array(data.pop("img_arrays"), dtype=np.float16)

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

    avg_counts, avg_counts_ste, norms = widefield.process_counts(
        nv_list, sig_counts, ref_counts, threshold=True
    )

    # raw_fig = create_raw_data_figure(nv_list, freqs, avg_counts, avg_counts_ste)
    fit_fig = create_fit_figure(nv_list, freqs, avg_counts, avg_counts_ste, norms)

    ###

    # pixel_drifts = data["pixel_drifts"]
    # img_arrays = np.array(data.pop("img_arrays"), dtype=np.float16)
    # base_pixel_drift = [15, 45]
    # # base_pixel_drift = [24, 74]
    # num_reps = 1

    # buffer = 30
    # img_array_size = 250
    # cropped_size = img_array_size - 2 * buffer
    # proc_img_arrays = np.empty(
    #     (2, num_runs, 2 * adj_num_steps, num_reps, cropped_size, cropped_size)
    # )
    # for run_ind in range(num_runs):
    #     pixel_drift = pixel_drifts[run_ind]
    #     offset = [
    #         pixel_drift[1] - base_pixel_drift[1],
    #         pixel_drift[0] - base_pixel_drift[0],
    #     ]
    #     for step_ind in range(2 * adj_num_steps):
    #         img_array = img_arrays[0, run_ind, step_ind, 0]
    #         cropped_img_array = widefield.crop_img_array(img_array, offset, buffer)
    #         proc_img_arrays[0, run_ind, step_ind, 0, :, :] = cropped_img_array

    sig_img_arrays = np.mean(img_arrays[:, :, 0 : num_steps // 2, :], axis=(0, 1, 3))
    ref_img_array = np.mean(
        img_arrays[:, :, num_steps // 2 : 3 * num_steps // 4, :], axis=(0, 1, 2, 3)
    )
    proc_img_arrays = sig_img_arrays - ref_img_array

    # downsample_factor = 1
    # proc_img_arrays = [
    #     widefield.downsample_img_array(el, downsample_factor) for el in proc_img_arrays
    # ]
    # proc_img_arrays = np.array(proc_img_arrays)

    # Nice still
    # fig, ax = plt.subplots()
    # kpl.imshow(ax, proc_img_arrays[17])
    # ax.axis("off")
    # scale = widefield.get_camera_scale()
    # kpl.scale_bar(ax, scale, "1 µm", kpl.Loc.LOWER_RIGHT)

    widefield.animate(
        freqs,
        nv_list,
        avg_counts,
        avg_counts_ste,
        norms,
        proc_img_arrays,
        cmin=np.percentile(proc_img_arrays, 60),
        cmax=np.percentile(proc_img_arrays, 99.9),
        # scale_bar_length_factor=downsample_factor,
        just_movie=True,
    )

    ###

    kpl.show(block=True)
