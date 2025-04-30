# -*- coding: utf-8 -*-
"""
Widefield Rabi experiment

Created on November 29th, 2023

@author: mccambria
@author: sbchand
"""

import sys
import time
import traceback

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from majorroutines.widefield import base_routine
from utils import common
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils import widefield as widefield
from utils.constants import NVSig


def create_fit_figure(nv_list, phis, norm_counts, norm_counts_ste):
    # fit function
    def cos_func(phi, amp, phase_offset, offset):
        return amp * np.cos(phi - phase_offset) + offset

    num_nvs = len(nv_list)
    fit_fns = []
    popts = []
    phi_degrees = []
    for nv_ind in range(num_nvs):
        nv_counts = norm_counts[nv_ind]
        nv_counts_ste = norm_counts_ste[nv_ind]
        guess_params = [1.0, 0.0, 0.5]  # amp, phase_offset, baseline offset

        try:
            popt, _ = curve_fit(
                cos_func,
                phis,
                nv_counts,
                p0=guess_params,
                sigma=nv_counts_ste,
                absolute_sigma=True,
            )
        except Exception:
            popt = None

        fit_fns.append(cos_func if popt is not None else None)
        popts.append(popt)

        # Create new figure for this NV
        fig, ax = plt.subplots(figsize=(6, 5))

        # Plot data points
        ax.errorbar(
            phis,
            nv_counts,
            yerr=abs(nv_counts_ste),
            fmt="o",
            label=f"NV {nv_ind}",
            capsize=3,
        )

        # Plot fit if successful
        if popt is not None:
            phi_fit = np.linspace(min(phis), max(phis), 200)
            fit_vals = cos_func(phi_fit, *popt)
            # ax.plot(phi_fit, fit_vals, "-", label="Fit")
            residuals = cos_func(phis, *popt) - nv_counts
            chi_sq = np.sum((residuals / nv_counts_ste) ** 2)
            red_chi_sq = chi_sq / (len(nv_counts) - len(popt))
            # print(f"NV {nv_ind} - Reduced chi²: {red_chi_sq:.3f}")
            peak = popt[0]  # phase_offset
            peak_phi = popt[1]  # phase_offset
            offset = popt[2]  # cpitms
            phi_degree = np.degrees(peak_phi)
            phi_degrees.append(phi_degree)
            print(
                f"Peak Amp = {peak:.2f} Peak occurs at φ ≈ {peak_phi:.2f} rad ≈ {np.degrees(peak_phi):.1f}°, offset = {offset:2f}"
            )
    # phi_degrees = np.degrees(phi_degrees)
    mean_phase_offset = np.mean(phi_degrees)
    # Suggest correction
    correction_angle = -mean_phase_offset

    print(f"\nAverage Phase Offset = {mean_phase_offset:.2f}°")
    print(f"Suggested Phase Correction = {correction_angle:.2f}°")

    # good_offsets = [phi for phi in phi_degrees if abs(phi) < 0.3]
    # avg_phase_offset = np.mean(good_offsets)
    # print(f"Suggested global IQ phase correction: {np.degrees(avg_phase_offset):.1f}°")
    # ax.set_xlabel("Phase (rad)")
    # ax.set_ylabel("Normalized Counts")
    # ax.set_title(f"Cosine Fit for NV {nv_ind}")
    # # plt.title(f"Fit: A={A_fit:.2f}, δ={np.rad2deg(delta_fit):.1f}°, C={C_fit:.2f}")
    # ax.legend()
    # ax.grid(True)
    # ax.spines["right"].set_visible(False)
    # ax.spines["top"].set_visible(False)
    # plt.tight_layout()
    # plt.show(block=True)
    # A_fit, delta_fit, C_fit = popt

    # plt.plot(phi_vals, signal, 'o', label="Data")
    # plt.plot(phi_vals, cos_func(phi_vals, *popt), '-', label="Fit")
    # plt.xlabel("Phase φ (deg)")
    # plt.ylabel("Signal")
    # plt.title(f"Fit: A={A_fit:.2f}, δ={np.rad2deg(delta_fit):.1f}°, C={C_fit:.2f}")
    # plt.legend()
    # plt.show()


def main(nv_list, num_steps, num_reps, num_runs, phi_list, uwave_ind_list):
    ### Some initial setup
    pulse_gen = tb.get_server_pulse_gen()
    seq_file = "spin_echo_phase_scan.py"

    ### Collect the data

    def run_fn(shuffled_step_inds):
        shuffled_taus = [phi_list[ind] for ind in shuffled_step_inds]
        seq_args = [
            widefield.get_base_scc_seq_args(nv_list, uwave_ind_list),
            shuffled_taus,
        ]
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
        save_images=False,
        load_iq=True,
    )

    ### save the raw data
    timestamp = dm.get_time_stamp()
    raw_data |= {
        "timestamp": timestamp,
        "phis": phi_list,
        "phi-units": "radian",
    }

    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    repr_nv_name = repr_nv_sig.name
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    dm.save_raw_data(raw_data, file_path)

    ### Clean up
    tb.reset_cfm()

    ### Process and plot
    try:
        raw_fig = None
        fit_fig = None
        counts = raw_data["counts"]
        sig_counts = counts[0]
        ref_counts = counts[1]
        norm_counts, norm_counts_ste = widefield.process_counts(
            nv_list, sig_counts, ref_counts, threshold=True
        )
        fit_fig = create_fit_figure(nv_list, phi_list, norm_counts, norm_counts_ste)
    except Exception:
        print(traceback.format_exc())
        raw_fig = None
        fit_fig = None
    kpl.show()

    # if raw_fig is not None:
    #     dm.save_figure(raw_fig, file_path)
    # if fit_fig is not None:
    #     file_path = dm.get_file_path(__file__, timestamp, repr_nv_name + "-fit")
    #     dm.save_figure(fit_fig, file_path)


if __name__ == "__main__":
    kpl.init_kplotlib()
    # file_id = 1817334208399
    file_id = 1825020210830  #
    # file_id = 1825070485845
    file_id = 1837498410890  # file_name = "2025_04_04-17_38_13-rubin-nv0_2025_02_26"

    # data = dm.get_raw_data(file_id=file_id, load_npz=False, use_cache=True)
    data = dm.get_raw_data(
        file_stem="2025_04_04-17_38_13-rubin-nv0_2025_02_26",
        load_npz=False,
        use_cache=True,
    )
    nv_list = data["nv_list"]
    num_nvs = len(nv_list)
    num_steps = data["num_steps"]
    num_runs = data["num_runs"]
    phis = data["phis"]

    counts = np.array(data["counts"])
    sig_counts = counts[0]
    ref_counts = counts[1]

    norm_counts, norm_counts_ste = widefield.process_counts(
        nv_list, sig_counts, ref_counts, threshold=True
    )
    # file_name = dm.get_file_name(file_id=file_id)
    # print(f"{file_name}_{file_id}")
    num_nvs = len(nv_list)
    phi_step = phis[1] - phis[0]
    num_steps = len(phis)
    fit_fig = create_fit_figure(nv_list, phis, norm_counts, norm_counts_ste)

    kpl.show(block=True)
