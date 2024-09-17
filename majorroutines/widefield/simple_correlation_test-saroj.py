# -*- coding: utf-8 -*-
"""

Created on December 16th, 2023

@author: Saroj Chand
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
        nv_list, sig_counts, ref_counts, dynamic_thresh=True
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


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def clip_correlation_coeffs(corr_matrix, min_val=-0.01, max_val=0.01):
    """
    Clip the correlation coefficients to be within the specified range.
    """
    return np.clip(corr_matrix, min_val, max_val)

def remove_nans_from_data(sig_counts, ref_counts, nv_list):
    """
    Remove NVs that contain any NaN values in their signal or reference counts.
    """
    valid_indices = [
        i for i in range(len(nv_list)) 
        if not (np.isnan(sig_counts[i]).any() or np.isnan(ref_counts[i]).any())
    ]
    
    # Filter the signal and reference counts and the NV list
    sig_counts_filtered = sig_counts[valid_indices]
    ref_counts_filtered = ref_counts[valid_indices]
    nv_list_filtered = [nv_list[i] for i in valid_indices]

    return sig_counts_filtered, ref_counts_filtered, nv_list_filtered

def reshuffle_by_spin(nv_list, sig_counts, ref_counts, sig_corr_coeffs, ref_corr_coeffs):
    """
    Reshuffle the NV list, counts, and correlation matrices such that
    spin +1 NVs are placed first and spin -1 NVs are placed at the end.
    """
    # Identify indices of NVs with spin +1 and spin -1
    spin_plus_indices = [i for i, nv in enumerate(nv_list) if nv.spin_flip == False]
    spin_minus_indices = [i for i, nv in enumerate(nv_list) if nv.spin_flip == True]

    # Reshuffle the NV list, counts, and correlation matrices
    reshuffled_indices = spin_plus_indices + spin_minus_indices
    nv_list_reshuffled = [nv_list[i] for i in reshuffled_indices]
    sig_counts_reshuffled = sig_counts[reshuffled_indices]
    ref_counts_reshuffled = ref_counts[reshuffled_indices]

    # Reshuffle correlation matrices by rows and columns
    sig_corr_reshuffled = sig_corr_coeffs[np.ix_(reshuffled_indices, reshuffled_indices)]
    ref_corr_reshuffled = ref_corr_coeffs[np.ix_(reshuffled_indices, reshuffled_indices)]

    return nv_list_reshuffled, sig_counts_reshuffled, ref_counts_reshuffled, sig_corr_reshuffled, ref_corr_reshuffled

def reshuffle_by_corr(nv_list, sig_counts, ref_counts, sig_corr_coeffs, ref_corr_coeffs):
    """
    Reshuffle the NV list, counts, and correlation matrices based on
    the average correlation coefficient of each NV, from positive to negative.
    """
    # Calculate the average correlation coefficient for each NV
    avg_corr_coeffs = np.nanmean(sig_corr_coeffs, axis=1)

    # Get the indices that would sort the NVs by their average correlation coefficients (descending order)
    reshuffled_indices = np.argsort(avg_corr_coeffs)[::-1]

    # Reshuffle the NV list, counts, and correlation matrices according to the sorted indices
    nv_list_reshuffled = [nv_list[i] for i in reshuffled_indices]
    sig_counts_reshuffled = sig_counts[reshuffled_indices]
    ref_counts_reshuffled = ref_counts[reshuffled_indices]

    # Reshuffle the correlation matrices (both rows and columns)
    sig_corr_reshuffled = sig_corr_coeffs[np.ix_(reshuffled_indices, reshuffled_indices)]
    ref_corr_reshuffled = ref_corr_coeffs[np.ix_(reshuffled_indices, reshuffled_indices)]

    return nv_list_reshuffled, sig_counts_reshuffled, ref_counts_reshuffled, sig_corr_reshuffled, ref_corr_reshuffled

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
    sig_counts = np.array(counts[0])
    ref_counts = np.array(counts[1])

    # Remove NVs with NaN values in their signal or reference counts
    sig_counts, ref_counts, nv_list = remove_nans_from_data(sig_counts, ref_counts, nv_list)
    num_nvs = len(nv_list)  # Update the number of NVs after filtering

    # Thresholding counts with dynamic thresholds
    sig_counts, ref_counts = widefield.threshold_counts(
        nv_list, sig_counts, ref_counts, dynamic_thresh=False
    )

    ### Calculate the correlations
    flattened_sig_counts = [sig_counts[ind].flatten() for ind in range(num_nvs)]
    flattened_ref_counts = [ref_counts[ind].flatten() for ind in range(num_nvs)]

    sig_corr_coeffs = tb.nan_corr_coef(flattened_sig_counts)
    ref_corr_coeffs = tb.nan_corr_coef(flattened_ref_counts)

    # Reshuffle NVs by spin: spin +1 first, spin -1 at the end
    # nv_list, sig_counts, ref_counts, sig_corr_coeffs, ref_corr_coeffs = reshuffle_by_corr(
    #     nv_list, sig_counts, ref_counts, sig_corr_coeffs, ref_corr_coeffs
    # )

    ### Ideal correlation matrix calculation after reshuffling
    spin_flips = np.array([-1 if nv.spin_flip else +1 for nv in nv_list])
    ideal_sig_corr_coeffs = np.outer(spin_flips, spin_flips).astype(float)
    ideal_ref_corr_coeffs = np.zeros((num_nvs, num_nvs), dtype=float)

    ### Plot

    ### Calculate average of positive and negative correlation coefficients
    positive_corrs = sig_corr_coeffs[sig_corr_coeffs > 0]
    negative_corrs = sig_corr_coeffs[sig_corr_coeffs < 0]

    avg_positive = np.nanmean(positive_corrs) if positive_corrs.size > 0 else 0
    avg_negative = np.nanmean(negative_corrs) if negative_corrs.size > 0 else 0

    vmin = avg_negative  # Set lower limit to average of negative correlations
    vmax = avg_positive  # Set upper limit to average of positive correlations

    ### Plot

    figsize = kpl.figsize.copy()
    figsize[0] *= 2
    figsize[1] *= 0.85
    titles = ["Ideal signal", "Signal", "Reference"]
    vals = [ideal_sig_corr_coeffs, sig_corr_coeffs, ref_corr_coeffs]

    if passed_ax is None:
        num_plots = len(vals)
        fig, axes_pack = plt.subplots(ncols=num_plots, figsize=figsize)

    # Replace diagonals with NaNs for cleaner visualization
    for val in vals:
        np.fill_diagonal(val, np.nan)

    # Set colorbar limits based on the average of positive and negative correlations
    for ind in range(len(vals)):
        if passed_ax is None:
            ax = axes_pack[ind]
        else:
            if sig_or_ref and ind != 1:
                continue
            if not sig_or_ref and ind != 2:
                continue
            ax = passed_ax
            ret_val = vals[ind]

        # Plot correlation matrix with calculated vmin and vmax
        kpl.imshow(
            ax,
            vals[ind],
            title=titles[ind],
            cbar_label="Correlation coefficient",
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            nan_color=kpl.KplColors.GRAY,
            no_cbar=no_cbar or ind < num_plots - 1,
        )

        # Dynamically set the ticks and tick intervals based on num_nvs
        max_ticks = 6  # Maximum number of ticks (adjustable)
        tick_interval = max(1, num_nvs // max_ticks)

        ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=max_ticks))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=max_ticks))
        
        # Set dynamic tick locations based on the number of NVs
        ax.set_xticks(np.arange(0, num_nvs, tick_interval))
        ax.set_yticks(np.arange(0, num_nvs, tick_interval))

        if not no_labels:
            ax.set_xlabel("NV index")
            ax.set_ylabel("NV index")

    if passed_ax is not None:
        return ret_val


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

    data = dm.get_raw_data(file_id=1648658056651)  # Block

    # Check if data is fetched successfully
    if data is not None:
        # Process and plot the data
        figs = process_and_plot(data)

        # Display the figures
        kpl.show()

        repr_nv_name = "nv0"
        timestamp = dm.get_time_stamp()

        # Save the figures if any were generated
        if figs is not None:
            for ind, fig in enumerate(figs):
                file_path = dm.get_file_path(__file__, timestamp, f"{repr_nv_name}")
                dm.save_figure(fig, file_path)

        # Show the plot with blocking to prevent the script from exiting immediately
        plt.show(block=True)
    else:
        print("Error: Failed to fetch the raw data.")
