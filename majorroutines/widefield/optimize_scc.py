# -*- coding: utf-8 -*-
"""
Optimize SCC parameters

Created on December 6th, 2023

@author: mccambria
"""


import copy
from random import shuffle
import sys
import matplotlib.pyplot as plt
import numpy as np
from majorroutines.widefield import optimize
from utils import tool_belt as tb
from utils import data_manager as dm
from utils import common
from utils import widefield as widefield
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import data_manager as dm
from utils.constants import LaserKey, NVSpinState
import os
import time
from utils.positioning import get_scan_1d as calculate_freqs
from majorroutines.pulsed_resonance import fit_resonance, voigt_split, voigt


def create_raw_data_figure(nv_list, taus, snrs, snrs_errs):
    fig, ax = plt.subplots()
    widefield.plot_raw_data(ax, nv_list, taus, counts, counts_errs)
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Counts")
    return fig


def main(nv_list, uwave_nv, state, min_tau, max_tau, num_steps, num_reps, num_runs):
    with common.labrad_connect() as cxn:
        main_with_cxn(
            cxn,
            nv_list,
            uwave_nv,
            state,
            min_tau,
            max_tau,
            num_steps,
            num_reps,
            num_runs,
        )


def main_with_cxn(
    cxn, nv_list, uwave_nv, state, min_tau, max_tau, num_steps, num_reps, num_runs
):
    ### Some initial setup

    tb.reset_cfm(cxn)

    # First NV to represent the others
    repr_nv_ind = 0
    repr_nv_sig = nv_list[repr_nv_ind]
    pos.set_xyz_on_nv(cxn, repr_nv_sig)
    num_nvs = len(nv_list)
    nv_list_mod = copy.deepcopy(nv_list)

    camera = tb.get_server_camera(cxn)
    pulse_gen = tb.get_server_pulse_gen(cxn)
    sig_gen = tb.get_server_sig_gen(cxn, state)
    sig_gen_name = sig_gen.name

    taus = np.linspace(min_tau, max_tau, num_steps)

    uwave_dict = uwave_nv[state]
    uwave_duration = tb.get_pi_pulse_dur(uwave_dict["rabi_period"])
    uwave_power = uwave_dict["uwave_power"]
    freq = uwave_dict["frequency"]
    sig_gen.set_amp(uwave_power)
    sig_gen.set_freq(freq)

    seq_file = "resonance_ref.py"

    ### Data tracking

    sig_counts = np.empty((num_nvs, num_runs, num_steps, num_reps))
    ref_counts = np.empty((num_nvs, num_runs, num_steps, num_reps))
    tau_ind_master_list = [[] for ind in range(num_runs)]
    tau_ind_list = list(range(0, num_steps))

    ### Collect the data

    for run_ind in range(num_runs):
        shuffle(tau_ind_list)

        camera.arm()
        sig_gen.uwave_on()

        for tau_ind in tau_ind_list:
            pixel_coords_list = [
                widefield.get_nv_pixel_coords(nv) for nv in nv_list
            ]
            tau_ind_master_list[run_ind].append(tau_ind)
            tau = taus[tau_ind]
            for nv in nv_list_mod:
                nv[LaserKey.IONIZATION]["duration"] = tau

            seq_args = widefield.get_base_scc_seq_args(nv_list_mod)
            seq_args.extend([sig_gen_name, uwave_duration])
            seq_args_string = tb.encode_seq_args(seq_args)
            pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

            # Try 5 times then give up
            num_attempts = 5
            attempt_ind = 0
            while True:
                try:
                    pulse_gen.stream_start()
                    for rep_ind in range(num_reps):
                        for sig_ref_ind in range(2):
                            img_str = camera.read()
                            img_array = widefield.img_str_to_array(img_str)
                            for nv_ind in range(num_nvs):
                                pixel_coords = pixel_coords_list[nv_ind]
                                counts_val = widefield.integrate_counts_from_adus(
                                    img_array, pixel_coords
                                )
                                counts = (
                                    sig_counts if sig_ref_ind == 0 else ref_counts
                                )
                                counts[
                                    nv_ind, run_ind, tau_ind, rep_ind
                                ] = counts_val
                    break
                except Exception as exc:
                    print(exc)
                    camera.arm()
                    attempt_ind += 1
                    if attempt_ind == num_attempts:
                        raise RuntimeError("Maxed out number of attempts")
            if attempt_ind > 0:
                print(f"{attempt_ind} crashes occurred")

        camera.disarm()
        sig_gen.uwave_off()
        optimize.optimize_pixel_with_cxn(cxn, repr_nv_sig)

    ### Process and plot

    avg_sig_counts, avg_sig_counts_ste = widefield.process_counts(sig_counts)
    avg_ref_counts, avg_ref_counts_ste = widefield.process_counts(ref_counts)
    avg_snr = (avg_sig_counts - avg_ref_counts) / (avg_sig_counts + avg_ref_counts)

    kpl.init_kplotlib()

    sig_fig, sig_ax = plt.subplots()
    widefield.plot_raw_data(sig_ax, nv_list, taus, avg_sig_counts, avg_sig_counts_ste)
    sig_ax.set_xlabel("Ionization pulse duration (ns)")
    sig_ax.set_ylabel("Signal counts")

    ref_fig, ref_ax = plt.subplots()
    widefield.plot_raw_data(ref_ax, nv_list, taus, avg_ref_counts, avg_ref_counts_ste)
    ref_ax.set_xlabel("Ionization pulse duration (ns)")
    ref_ax.set_ylabel("Reference counts")

    snr_fig, snr_ax = plt.subplots()
    widefield.plot_raw_data(snr_ax, nv_list, taus, avg_snr)
    snr_ax.set_xlabel("Ionization pulse duration (ns)")
    snr_ax.set_ylabel("SNR")

    ### Clean up and return

    tb.reset_cfm(cxn)

    timestamp = dm.get_time_stamp()
    raw_data = {
        "timestamp": timestamp,
        "nv_list": nv_list,
        "state": state,
        "num_reps": num_reps,
        "num_steps": num_steps,
        "num_runs": num_runs,
        "uwave_nv": uwave_nv,
        "taus": taus,
        "tau-units": "ns",
        "tau_ind_master_list": tau_ind_master_list,
        "max_tau": max_tau,
        "sig_counts": sig_counts,
        "ref_counts": ref_counts,
        "counts-units": "photons",
    }

    repr_nv_name = repr_nv_sig["name"]
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    dm.save_raw_data(raw_data, file_path)
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name + "-sig")
    dm.save_figure(sig_fig, file_path)
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name + "-ref")
    dm.save_figure(ref_fig, file_path)
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name + "-snr")
    dm.save_figure(snr_fig, file_path)


if __name__ == "__main__":
    kpl.init_kplotlib()

    # file_name = "2023_11_27-19_31_32-johnson-nv0_2023_11_25"
    # data = dm.get_raw_data(file_name)
    data = dm.get_raw_data(file_id=1379809150970)

    nv_list = data["nv_list"]
    num_nvs = len(nv_list)
    num_steps = data["num_steps"]
    num_runs = data["num_runs"]
    avg_img_arrays = np.average(img_arrays, axis=1)
    freqs = data["freqs"]
    counts = np.array(data["counts"])

    avg_counts, avg_counts_ste = widefield.process_counts(counts)

    plt.show(block=True)
