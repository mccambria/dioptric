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
    error_dict = {}

    for i, seq in enumerate(seq_names):
        val = np.asarray(norm_counts[i]).item()
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


def plot_pulse_errors(error_dict):
    # keys = sorted(error_dict.keys())
    keys = error_dict.keys()
    values = [error_dict[k] for k in keys]

    # Check for problematic values
    for k, v in zip(keys, values):
        if not isinstance(v, (int, float, np.number)) or np.isnan(v):
            raise ValueError(f"Invalid value for {k}: {v}")

    plt.figure(figsize=(6, 5))
    bars = plt.bar(keys, values)
    plt.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    plt.xticks(rotation=45, fontsize=11)
    plt.yticks(fontsize=11)
    plt.ylabel("Error Amplitude", fontsize=12)
    plt.title("Extracted Pulse Errors", fontsize=12)
    plt.tight_layout()

    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.0,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.show()


def spherical_arc(start, end, n_points=100):
    start = start / np.linalg.norm(start)
    end = end / np.linalg.norm(end)
    omega = np.arccos(np.clip(np.dot(start, end), -1, 1))
    if omega == 0:
        return np.tile(start, (n_points, 1)).T
    sin_omega = np.sin(omega)
    t = np.linspace(0, 1, n_points)
    arc = (
        np.sin((1 - t) * omega)[:, None] * start + np.sin(t * omega)[:, None] * end
    ) / sin_omega
    return arc.T, np.degrees(omega)


def Bloch_Sphere_Visualization(pulse_errors):
    # Ideal unit vectors for X and Y pulses
    ideal_X = np.array([1, 0, 0])
    ideal_Y = np.array([0, 1, 0])
    ideal_Z = np.array([0, 0, 1])
    # Actual axes derived from errors
    actual_X = np.array([1, pulse_errors["ey"], pulse_errors["ez"]])
    actual_Y = np.array([pulse_errors["vx"], 1, pulse_errors["vz"]])

    # Normalize to map on Bloch sphere
    actual_X = actual_X / np.linalg.norm(actual_X)
    actual_Y = actual_Y / np.linalg.norm(actual_Y)

    # Bloch sphere setup
    fig = plt.figure(figsize=(6.5, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Draw the Bloch sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color="lightblue", alpha=0.1, edgecolor="gray")

    # Draw ideal and actual vectors
    ax.quiver(0, 0, 0, *ideal_X, color="blue", label="Ideal X")
    ax.quiver(0, 0, 0, *ideal_Y, color="green", label="Ideal Y")
    ax.quiver(0, 0, 0, *actual_X, color="red", linestyle="dashed", label="Exp X")
    ax.quiver(0, 0, 0, *actual_Y, color="orange", linestyle="dashed", label="Exp Y")
    ax.quiver(
        0,
        0,
        0,
        *ideal_Z,
        color="black",
        alpha=0.5,
        linestyle="dotted",
        label="Z reference",
    )

    arc_X, angle_X = spherical_arc(ideal_X, actual_X)
    arc_Y, angle_Y = spherical_arc(ideal_Y, actual_Y)
    ax.plot(*arc_X, color="red", linestyle="--", linewidth=1.5)
    ax.plot(*arc_Y, color="orange", linestyle="--", linewidth=1.5)

    mid_X = arc_X[:, len(arc_X[0]) // 2]
    mid_Y = arc_Y[:, len(arc_Y[0]) // 2]

    ax.text(*mid_X, f"{angle_X:.1f}°", color="red", fontsize=11, ha="center")
    ax.text(*mid_Y, f"{angle_Y:.1f}°", color="orange", fontsize=11, ha="center")

    # Axis settings
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_zlabel("Z", fontsize=12)
    ax.tick_params(labelsize=11)
    ax.set_title("Bloch Sphere Axis Tilt due to Pulse Errors", fontsize=12)
    ax.legend(fontsize=11)
    plt.tight_layout()
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
    # file_id = 1843336828775
    file_ids = [1843336828775, 1843444108428, 1843662119540]
    # data = dm.get_raw_data(file_id=file_id, load_npz=False, use_cache=True)
    data = widefield.process_multiple_files(file_ids=file_ids)
    nv_list = data["nv_list"]
    seq_names = data["bootstrap_sequence_names"]
    num_nvs = len(nv_list)
    counts = np.array(data["counts"])
    print(counts.shape)
    # sys.exit()
    ref_counts = counts[-1]  # last data is ref
    norm_counts = []
    for c in range(len(seq_names)):
        sig_counts = counts[c]  #
        nc, _ = widefield.process_counts(nv_list, sig_counts, threshold=True)
        nc_medians = np.median(nc, axis=0)
        norm_counts.append(nc_medians)
    norm_counts = np.array(norm_counts)  # shape: (num_seqs, num_nvs)
    # file_name = dm.get_file_name(file_id=file_id)
    # print(f"{file_name}_{file_id}")
    error_dict = extract_error_params(norm_counts, seq_names)
    print(error_dict)
    plot_pulse_errors(error_dict)
    # Bloch_Sphere_Visualization(error_dict)
    plt.show(block=True)
