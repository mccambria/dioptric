# -*- coding: utf-8 -*-
"""
Widefield Rabi experiment

Created on November 29th, 2023

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


def extract_error_params(norm_counts, seq_names):
    """
    Extracts pulse error parameters from median signal values across NVs.
    See PRL 105, 077601 (2010), Table I.
    """
    norm_counts = np.array(norm_counts)
    medians = np.median(norm_counts, axis=0)
    error_dict = {}

    for i, seq in enumerate(seq_names):
        val = medians[i]
        if seq == "pi_2_X":
            error_dict["phi_prime"] = -0.5 * val
        elif seq == "pi_2_Y":
            error_dict["chi_prime"] = -0.5 * val
        elif seq == "pi_2_X_pi_X":
            error_dict["phi"] = 0.5 * val - error_dict.get("phi_prime", 0)
        elif seq == "pi_2_Y_pi_Y":
            error_dict["chi"] = 0.5 * val - error_dict.get("chi_prime", 0)
        elif seq == "pi_Y_pi_2_X":
            error_dict["vz"] = -0.5 * val + error_dict.get("phi_prime", 0)
        elif seq == "pi_X_pi_2_Y":
            error_dict["ez"] = 0.5 * val - error_dict.get("chi_prime", 0)
        elif seq == "pi_2_Y_pi_2_X":
            error_dict["vx"] = -0.5 * val + error_dict.get("phi_prime", 0)
        elif seq == "pi_2_X_pi_2_Y":
            error_dict["ex"] = 0.5 * val - error_dict.get("chi_prime", 0)
        elif seq == "pi_2_X_pi_X_pi_2_Y":
            error_dict["ey"] = 0.5 * val - error_dict.get("phi_prime", 0)
        elif seq == "pi_2_Y_pi_X_pi_2_X":
            error_dict["vy"] = -0.5 * val + error_dict.get("phi_prime", 0)
        elif seq == "pi_2_X_pi_Y_pi_2_Y":
            error_dict["vz_alt"] = -0.5 * val + error_dict.get("chi_prime", 0)
        elif seq == "pi_2_Y_pi_Y_pi_2_X":
            error_dict["ez_alt"] = 0.5 * val - error_dict.get("phi_prime", 0)

    return error_dict


def plot_pulse_errors(error_dict, title="Extracted Pulse Errors"):
    # Sort keys for consistent ordering
    keys = sorted(error_dict.keys())
    values = [error_dict[k] for k in keys]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(keys, values)
    plt.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    plt.xticks(rotation=45)
    plt.ylabel("Error Amplitude")
    plt.title(title)
    plt.tight_layout()

    # Annotate bars with values
    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.05,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.show()


def main(nv_list, num_reps, num_runs, uwave_ind_list):
    ### Some initial setup
    pulse_gen = tb.get_server_pulse_gen()
    seq_file = "bootstrapped_pulse_error_tomography.py"
    num_steps = 1
    seq_names = [
        "pi_2_X",
        "pi_2_Y",
        "pi_2_X_pi_X",
        "pi_2_Y_pi_Y",
        "pi_Y_pi_2_X",
        "pi_X_pi_2_Y",
        "pi_2_Y_pi_2_X",
        "pi_2_X_pi_2_Y",
        "pi_2_X_pi_X_pi_2_Y",
        "pi_2_Y_pi_X_pi_2_X",
        "pi_2_X_pi_Y_pi_2_Y",
        "pi_2_Y_pi_Y_pi_2_X",
    ]

    num_exps = len(seq_names) + 1  # last exp is reference

    ### Collect the data
    def run_fn(shuffled_step_inds):
        seq_args = [
            widefield.get_base_scc_seq_args(nv_list, uwave_ind_list),
            seq_names,
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
        num_exps=num_exps,
        load_iq=True,
    )

    ### save the raw data
    timestamp = dm.get_time_stamp()
    raw_data |= {"timestamp": timestamp, "bootstrap_sequence_names": seq_names}

    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    repr_nv_name = repr_nv_sig.name
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    dm.save_raw_data(raw_data, file_path)

    ### Clean up
    tb.reset_cfm()

    ### Process and plot
    # try:
    #     fig = None
    #     counts = raw_data["counts"]
    #     sig_counts_0 = counts[0]  #  "pi_2_X",
    #     rsig_counts_1 = counts[1]
    #     ....
    #     norm_counts, norm_counts_ste = widefield.process_counts(
    #         nv_list, sig_counts_1, threshold=True
    #     )
    #     analyze_bootstrap_data(nv_list, norm_counts, norm_counts_ste)
    # except Exception:
    #     print(traceback.format_exc())
    #     fig = None
    kpl.show()

    # if raw_fig is not None:
    #     dm.save_figure(raw_fig, file_path)
    # if fit_fig is not None:
    #     file_path = dm.get_file_path(__file__, timestamp, repr_nv_name + "-fit")
    #     dm.save_figure(fit_fig, file_path)


if __name__ == "__main__":
    kpl.init_kplotlib()
    file_id = 1837498410890
    data = dm.get_raw_data(file_id=file_id, load_npz=False, use_cache=True)
    nv_list = data["nv_list"]
    seq_names = data["nv_list"]
    num_nvs = len(nv_list)
    counts = np.array(data["counts"])
    norm_counts = []
    for c in len(seq_names):
        count = counts[c]  #
        nc, _ = widefield.process_counts(nv_list, count, threshold=True)
        norm_counts.append(nc)
    norm_counts = np.array(norm_counts)  # shape: (num_seqs, num_nvs)

    file_name = dm.get_file_name(file_id=file_id)
    print(f"{file_name}_{file_id}")
    file_name = dm.get_file_name(file_id=file_id)
    print(f"{file_name}_{file_id}")
    error_dict = extract_error_params(norm_counts, seq_names)
    plot_pulse_errors(error_dict)

    kpl.show(block=True)
