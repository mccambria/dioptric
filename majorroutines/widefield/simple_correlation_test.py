# -*- coding: utf-8 -*-
"""
Optimize SCC parameters

Created on December 6th, 2023

@author: mccambria
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from matplotlib.ticker import MaxNLocator
from scipy.optimize import curve_fit

from majorroutines.widefield import base_routine
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils import widefield as widefield
from utils.constants import NVSig


def process_and_plot_experimental(data):
    nv_list = data["nv_list"]
    # nv_list = nv_list[::-1]
    counts = np.array(data["counts"])
    # counts = np.array(data["states"])
    states = np.array(data["states"])
    num_runs = counts.shape[2]
    # counts = counts[:, :, :]
    # states = states[:, :, :]

    start = 1000
    window = 500
    counts = counts[:, :, start : start + window, :, :]
    states = states[:, :, start : start + window, :, :]
    # counts = counts[:, :, ::10, :, :]
    # states = states[:, :, ::10, :, :]

    # exclude_inds = (6, 9, 13)
    exclude_inds = ()
    num_nvs = len(nv_list)
    nv_list = [nv_list[ind] for ind in range(num_nvs) if ind not in exclude_inds]
    num_nvs = len(nv_list)
    counts = np.delete(counts, exclude_inds, axis=1)

    # Break down the counts array
    # experiment, nv, run, step, rep
    sig_counts = np.array(counts[0])
    ref_counts = np.array(counts[1])

    sig_counts, ref_counts = widefield.threshold_counts(
        nv_list, sig_counts, ref_counts, None, dynamic_thresh=True
    )

    i_counts = ref_counts[5]
    j_counts = ref_counts[6]
    i_counts_m = ma.masked_invalid(i_counts)
    j_counts_m = ma.masked_invalid(j_counts)
    mask = ~i_counts_m.mask & ~j_counts_m.mask
    fig, ax = plt.subplots()
    ref_ccounts = np.array(counts[1])
    i_counts = ref_ccounts[5]
    j_counts = ref_ccounts[6]
    kpl.histogram(ax, i_counts[mask])
    kpl.histogram(ax, j_counts[mask])

    fig, ax = plt.subplots()
    sig_ccounts = np.array(counts[0])
    kpl.histogram(ax, sig_ccounts[0].flatten())
    ax.set_xlabel("Integrated counts")
    ax.set_ylabel("Number of occurences")
    ax.axvline(nv_list[0].threshold)
    ax.axvline(31.85)

    # Calculate the correlations
    flattened_sig_counts = [sig_counts[ind].flatten() for ind in range(num_nvs)]
    flattened_ref_counts = [ref_counts[ind].flatten() for ind in range(num_nvs)]

    # flattened_sig_counts = flattened_sig_counts[::-1]
    # flattened_ref_counts = flattened_ref_counts[::-1]

    # ref_states = states[1]
    # flattened_ref_states = [ref_states[ind].flatten() for ind in range(num_nvs)]
    # flattened_ref_counts = np.array(flattened_ref_counts)
    # flattened_ref_states = np.array(flattened_ref_states)

    num_shots = len(flattened_ref_counts[0])
    # sig_corr_coeffs = np.corrcoef(flattened_sig_counts)
    # ref_corr_coeffs = np.corrcoef(flattened_ref_counts)
    sig_corr_coeffs = tb.nan_corr_coef(flattened_sig_counts)
    ref_corr_coeffs = tb.nan_corr_coef(flattened_ref_counts)

    # MCC
    # flattened_ref_counts = np.where(flattened_ref_states, flattened_ref_counts, np.nan)
    # # flattened_ref_counts = np.where(
    # #     np.logical_not(flattened_ref_states), flattened_ref_counts, np.nan
    # # )
    # # fmt: off
    # coords = [(5.464, 5.386),(6.728, 4.389),(5.631, 4.334),(4.584, 3.664),(6.007, 6.824),(7.247, 3.271),(7.128, 2.078),(7.104, 5.525),(5.709, 3.111),(2.443, 5.817)]
    # # fmt: on
    # coords = [np.array(el) for el in coords]
    # fig, ax = plt.subplots()
    # max_corr = 0
    # max_corr_inds = None
    # for ind in range(num_nvs):
    #     for jnd in range(num_nvs):
    #         if jnd <= ind:
    #             continue
    #         dist = np.sqrt(np.sum((coords[ind] - coords[jnd]) ** 2))
    #         i_counts = flattened_ref_counts[ind]
    #         j_counts = flattened_ref_counts[jnd]
    #         i_counts_m = ma.masked_invalid(i_counts)
    #         j_counts_m = ma.masked_invalid(j_counts)
    #         mask = ~i_counts_m.mask & ~j_counts_m.mask
    #         corr = np.corrcoef(i_counts[mask], j_counts[mask])[0, 1]
    #         if corr > max_corr:
    #             max_corr = corr
    #             max_corr_inds = [ind, jnd]
    #         kpl.plot_points(ax, dist, corr, color=kpl.KplColors.BLUE)
    # ax.set_xlabel("Distance between NVs (μm)")
    # ax.set_ylabel("Count correlation | both NVs in NV$^{-}$")
    # print(max_corr_inds)
    # return

    diff_corr_coeffs = np.cov(flattened_sig_counts) - np.cov(flattened_ref_counts)
    # stddev = np.sqrt(np.diag(sig_corr_coeffs).real + np.diag(ref_corr_coeffs).real)
    # diff_corr_coeffs /= stddev[:, None]
    # diff_corr_coeffs /= stddev[None, :]
    # diff_corr_coeffs = sig_corr_coeffs - ref_corr_coeffs

    spin_flips = np.array([-1 if nv.spin_flip else +1 for nv in nv_list])
    # spin_flips = np.array(
    #     [-1 if ind in [0, 1, 4, 6] else +1 for ind in range(num_nvs)]
    # )  # MCC
    ideal_sig_corr_coeffs = np.outer(spin_flips, spin_flips)
    ideal_sig_corr_coeffs = ideal_sig_corr_coeffs.astype(float)

    # Replace diagonals (Cii=1) with nan so they don't show
    vals = [sig_corr_coeffs, diff_corr_coeffs, ref_corr_coeffs, ideal_sig_corr_coeffs]
    for val in vals:
        np.fill_diagonal(val, np.nan)

    print(np.nanmean(ref_corr_coeffs) / np.nanmean(np.abs(sig_corr_coeffs)))

    # MCC
    # fig, ax = plt.subplots()
    # mean_diff_corr_coeffs = [
    #     np.nanmean(np.abs(diff_corr_coeffs[ind])) for ind in range(num_nvs)
    # ]
    # kpl.plot_points(ax, range(num_nvs), mean_diff_corr_coeffs)
    # ax.set_xlabel("NV index")
    # ax.set_ylabel("Mean abs val of diff covariances")

    ### Plot

    # Make the colorbar symmetric about 0
    sig_max = np.nanmax(np.abs(sig_corr_coeffs))
    ref_max = np.nanmax(np.abs(ref_corr_coeffs))
    diff_max = np.nanmax(np.abs(diff_corr_coeffs))

    figs = []
    titles = ["Signal", "Difference", "Reference", "Ideal signal"]
    cbar_maxes = [sig_max, diff_max, sig_max, 1]
    for ind in range(len(vals)):
        coors = vals[ind]  # Replace diagonals (Cii=1) with nan so they don't show
        np.fill_diagonal(coors, np.nan)
        fig, ax = plt.subplots()
        cbar_max = cbar_maxes[ind]
        # cbar_max = 0.032
        kpl.imshow(
            ax,
            vals[ind],
            title=titles[ind],
            cbar_label="Covariance" if ind == 1 else "Correlation coefficient",
            cmap="RdBu_r",
            vmin=-cbar_max,
            vmax=cbar_max,
            nan_color=kpl.KplColors.GRAY,
        )
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        figs.append(fig)

    # MCC
    # for ind in [0, 1, 3]:
    #     plt.close(figs[ind])

    return figs
    ### Spurious correlations offset

    # offsets = np.array(range(15000))
    offsets = list(range(1000))
    # offsets = [500]
    spurious_vals = []
    for offset in offsets:
        ref_corr_coeffs = np.array(
            [[None for ind in range(num_nvs)] for ind in range(num_nvs)],
            dtype=float,
        )
        for ind in range(num_nvs):
            for jnd in range(num_nvs):
                if jnd <= ind:
                    continue
                val = np.corrcoef(
                    [
                        flattened_ref_counts[ind][: num_shots - offset],
                        flattened_ref_counts[jnd][offset:],
                    ]
                )[0, 1]
                ref_corr_coeffs[ind, jnd] = val
                ref_corr_coeffs[jnd, ind] = val
        ref_corr_coeffs = np.array(ref_corr_coeffs)
        np.fill_diagonal(ref_corr_coeffs, np.nan)
        spurious_vals.append(np.nanmean(ref_corr_coeffs))

    fig, ax = plt.subplots()
    kpl.plot_points(ax, offsets, spurious_vals, label="Data")
    ax.set_xlabel("Shot offset")
    ax.set_ylabel("Average spurious correlation")
    window = 20
    avg = tb.moving_average(spurious_vals, window)
    avg_x_vals = np.array(range(len(avg))) + window // 2
    kpl.plot_line(
        ax,
        avg_x_vals,
        avg,
        color=kpl.KplColors.RED,
        zorder=10,
        linewidth=3,
        label="Moving average",
    )

    def fit_fn(offset, amp1, amp2, d1, d2):
        return (
            amp1 * np.exp(-offset / d1) + amp2 * np.exp(-offset / d2)
            # + amp3 * np.exp(offset / d3)
        )

    # # popt, pcov = curve_fit(fit_fn, avg_x_vals, avg, p0=(0.001, 20))
    # popt, pcov = curve_fit(fit_fn, avg_x_vals, avg, p0=(0.001, 0.0015, 20, 3000))
    # kpl.plot_line(
    #     ax,
    #     offsets,
    #     fit_fn(offsets, *popt),
    #     color=kpl.KplColors.ORANGE,
    #     zorder=10,
    #     linewidth=3,
    #     label="Fit",
    # )
    # print(popt)
    # ax.legend()

    return figs


def process_and_plot(
    data, ax=None, sig_or_ref=True, no_cbar=False, cbar_max=None, no_labels=False
):
    ### Unpack

    nv_list = data["nv_list"]
    counts = np.array(data["counts"])
    num_nvs = len(nv_list)

    passed_cbar_max = cbar_max
    passed_ax = ax

    # Break down the counts array
    # experiment, nv, run, step, rep
    sig_counts = np.array(counts[0])
    ref_counts = np.array(counts[1])

    num_runs = data["num_runs"]
    # sig_counts = sig_counts[:, round(0.5 * num_runs) :]
    # ref_counts = ref_counts[:, round(0.5 * num_runs) :]
    # sig_counts = sig_counts[:, : round(0.5 * num_runs)]
    # ref_counts = ref_counts[:, : round(0.5 * num_runs)]
    # sig_counts = sig_counts[:, round(0.25 * num_runs) : round(0.75 * num_runs)]
    # ref_counts = ref_counts[:, round(0.25 * num_runs) : round(0.75 * num_runs)]

    sig_counts, ref_counts = widefield.threshold_counts(
        nv_list, sig_counts, ref_counts, dynamic_thresh=False
    )

    ### Calculate the correlations
    flattened_sig_counts = [sig_counts[ind].flatten() for ind in range(num_nvs)]
    flattened_ref_counts = [ref_counts[ind].flatten() for ind in range(num_nvs)]

    sig_corr_coeffs = tb.nan_corr_coef(flattened_sig_counts)
    ref_corr_coeffs = tb.nan_corr_coef(flattened_ref_counts)

    spin_flips = np.array([-1 if nv.spin_flip else +1 for nv in nv_list])
    if -1 not in spin_flips:
        spin_flips[0] = -1
        spin_flips[1] = -1
        spin_flips[4] = -1
        spin_flips[6] = -1
    ideal_sig_corr_coeffs = np.outer(spin_flips, spin_flips)
    ideal_sig_corr_coeffs = ideal_sig_corr_coeffs.astype(float)

    ideal_ref_corr_coeffs = np.outer([0] * num_nvs, [0] * num_nvs)
    ideal_ref_corr_coeffs = ideal_ref_corr_coeffs.astype(float)

    ### Plot

    figsize = kpl.figsize.copy()

    # figsize[0] *= 1.4
    # figsize[1] *= 0.85
    # titles = ["Ideal signal", "Signal"]
    # vals = [ideal_sig_corr_coeffs, sig_corr_coeffs]
    # titles = ["Ideal reference", "Reference"]
    # vals = [ideal_ref_corr_coeffs, ref_corr_coeffs]

    figsize[0] *= 2
    figsize[1] *= 0.85
    titles = ["Ideal signal", "Signal", "Reference"]
    vals = [ideal_sig_corr_coeffs, sig_corr_coeffs, ref_corr_coeffs]

    if passed_ax is None:
        num_plots = len(vals)
        fig, axes_pack = plt.subplots(ncols=num_plots, figsize=figsize)

    # Replace diagonals (Cii=1) with nan so they don't show
    for val in [
        ideal_ref_corr_coeffs,
        ideal_sig_corr_coeffs,
        sig_corr_coeffs,
        ref_corr_coeffs,
    ]:
        np.fill_diagonal(val, np.nan)

    # Make the colorbar symmetric about 0
    sig_max = np.nanmax(np.abs(sig_corr_coeffs))
    ref_max = np.nanmax(np.abs(ref_corr_coeffs))

    print(f"Sig mean mag: {np.nanmean(np.abs(sig_corr_coeffs))}")
    print(f"Ref mean: {np.nanmean(ref_corr_coeffs)}")
    print(f"Ref std: {np.nanstd(ref_corr_coeffs)}")
    print()

    # cbar_maxes = [sig_max, sig_max, 1]
    cbar_max = sig_max if passed_cbar_max is None else passed_cbar_max
    for ind in range(len(vals)):
        if passed_ax is None:
            # fig, ax = plt.subplots()
            # figs.append(fig)
            ax = axes_pack[ind]
        else:
            if sig_or_ref and ind != 1:
                continue
            if not sig_or_ref and ind != 2:
                continue
            ax = passed_ax
            ret_val = vals[ind]
        # if passed_cbar_max is not None:
        #     cbar_max = passed_cbar_max
        # else:
        #     cbar_max = cbar_maxes[ind]
        kpl.imshow(
            ax,
            vals[ind],
            title=titles[ind],
            cbar_label="Correlation coefficient",
            cmap="RdBu_r",
            vmin=-cbar_max,
            vmax=cbar_max,
            nan_color=kpl.KplColors.GRAY,
            no_cbar=no_cbar or ind < num_plots - 1,
        )
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_yticks([0, 2, 4, 6, 8])
        ax.set_xticks([0, 2, 4, 6, 8])
        # ax.tick_params(labelsize=16)
        if not no_labels:
            ax.set_xlabel("NV index")
            ax.set_ylabel("NV index")

        # import os
        # output_dir = f'data/correlation_matrix/orientation_1540558251818'
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)
        # np.save(os.path.join(output_dir, 'sig_corr_coeffs.npy'), sig_corr_coeffs)
        # np.save(os.path.join(output_dir, 'ref_corr_coeffs.npy'), ref_corr_coeffs)
        # np.save(os.path.join(output_dir, 'ideal_sig_corr_coeffs.npy'), ideal_sig_corr_coeffs)
        # np.save(os.path.join(output_dir, 'ideal_sig_corr_coeffs.npy'), ideal_ref_corr_coeffs)

        # # for fig, title in zip(figs, titles):
        # #     fig.savefig(os.path.join(output_dir, f"{title.replace(' ', '_')}.png"))

        # print(f"Data and figures saved to {output_dir}")

    if passed_ax is not None:
        return ret_val
    # return figs


def main(nv_list, num_reps, num_runs):
    ### Some initial setup
    uwave_ind_list = [0, 1]
    seq_file = "simple_correlation_test.py"
    num_steps = 1

    pulse_gen = tb.get_server_pulse_gen()

    ### Collect the data

    def run_fn(shuffled_step_inds):
        seq_args = [widefield.get_base_scc_seq_args(nv_list, uwave_ind_list)]
        # print(seq_args)
        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

    raw_data = base_routine.main(
        nv_list,
        num_steps,
        num_reps,
        num_runs,
        run_fn=run_fn,
        uwave_ind_list=uwave_ind_list,
    )

    ### Process and plot

    # process_and_print(nv_list, counts)
    try:
        figs = process_and_plot(raw_data)
    except Exception:
        figs = None

    ### Clean up and save data

    tb.reset_cfm()

    kpl.show()

    timestamp = dm.get_time_stamp()
    raw_data |= {
        "timestamp": timestamp,
    }

    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    repr_nv_name = repr_nv_sig.name
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    dm.save_raw_data(raw_data, file_path)

    if figs is not None:
        for ind in range(len(figs)):
            fig = figs[ind]
            file_path = dm.get_file_path(__file__, timestamp, f"{repr_nv_name}-{ind}")
            dm.save_figure(fig, file_path)


if __name__ == "__main__":
    kpl.init_kplotlib()

    # data = dm.get_raw_data(file_id=1540048047866)  # Straight block
    # process_and_plot(data)
    # data = dm.get_raw_data(file_id=1541938921939)  # reverse block
    # process_and_plot(data)
    # data = dm.get_raw_data(file_id=1538271354881)  # checkerboard
    # process_and_plot(data)
    # data = dm.get_raw_data(file_id=1540558251818)  # orientation
    # data = dm.get_raw_data(file_id=1655494429496)  # orientation
    data = dm.get_raw_data(file_id=1653570783798) 
    process_and_plot(data)

    plt.show(block=True)
