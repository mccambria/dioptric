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

# def extract_error_params(norm_counts, seq_names):
#     """
#     Bootstrap Pulse Error Extraction using Full Linear System
#     Based on PRL 105, 077601 (2010) - Dobrovitski et al.
#     Block 3: Solves for 6 axis error parameters via least-squares
#     Assumes: ε'_y = 0 to fix gauge
#     @author: YourNam

#     Combined extraction function:
#     - Extracts angle and z-axis errors from Blocks 1 & 2 (explicit formulas)
#     - Solves axis tilt errors from Block 3 (least-squares)

#     """
#     norm_counts = np.array(norm_counts)
#     error_dict = {}
#     error_ste = {}
#     block3_signals = []
#     seq_block3 = []
#     for i, seq in enumerate(seq_names):
#         val = np.asarray(norm_counts[i]).item()
#         if seq == "pi_2_X":
#             error_dict["phi_prime"] = -0.5 * val
#         elif seq == "pi_2_Y":
#             error_dict["chi_prime"] = -0.5 * val
#         elif seq == "pi_2_X_pi_X":
#             error_dict["phi"] = 0.5 * val - error_dict.get("phi_prime", 0)
#         elif seq == "pi_2_Y_pi_Y":
#             error_dict["chi"] = 0.5 * val - error_dict.get("chi_prime", 0)
#         elif seq == "pi_Y_pi_2_X":
#             error_dict["vz"] = -0.5 * val + error_dict.get("phi_prime", 0)
#         elif seq == "pi_X_pi_2_Y":
#             error_dict["ez"] = 0.5 * val - error_dict.get("chi_prime", 0)

#         # Collect signals for Block 3
#         elif seq in [
#             "pi_2_Y_pi_2_X",
#             "pi_2_X_pi_2_Y",
#             "pi_2_X_pi_X_pi_2_Y",
#             "pi_2_Y_pi_X_pi_2_X",
#             "pi_2_X_pi_Y_pi_2_Y",
#             "pi_2_Y_pi_Y_pi_2_X",
#         ]:
#             block3_signals.append(val)
#             seq_block3.append(seq)

#     if len(block3_signals) == 6:
#         A = np.array(
#             [
#                 [-1, -1, -1, 0, 0],
#                 [1, -1, 1, 0, 0],
#                 [1, 1, -1, 2, 0],
#                 [-1, 1, 1, 2, 0],
#                 [-1, -1, 1, 0, 2],
#                 [1, -1, -1, 0, 2],
#             ]
#         )
#         block3_names = [
#             "epsilon_z_prime",  # ε′_z
#             "nu_x_prime",  # ν′_x
#             "nu_z_prime",  # ν′_z
#             "epsilon_y",  # ε_y
#             "nu_x",  # ν_x
#         ]

#         block3_signals = np.array(block3_signals)
#         x, residuals, rank, s = np.linalg.lstsq(A, block3_signals, rcond=None)
#         cov_matrix = np.linalg.inv(A.T @ A)
#         std_errors = np.sqrt(np.diag(cov_matrix)) * np.std(block3_signals - A @ x)

#         error_dict.update(dict(zip(block3_names, x)))
#         error_ste.update(dict(zip(block3_names, std_errors)))
#         # predicted_signals = A @ x
#         # error = np.linalg.norm(predicted_signals - block3_signals)
#         # print("Residual norm:", error)
#         predicted_signals = A @ x
#         residuals = block3_signals - predicted_signals
#         residual_norm = np.linalg.norm(residuals)
#         print("Residual norm:", residual_norm)

#         # Plot comparison of measured vs predicted
#         # fig, ax = plt.subplots(figsize=(8, 4))
#         # ax.plot(seq_block3, block3_signals, "o-", label="Measured", color="blue")
#         # ax.plot(seq_block3, predicted_signals, "s--", label="Predicted", color="orange")
#         # ax.set_title("Measured vs Predicted Signals (Block 3)")
#         # ax.set_ylabel("Signal Value")
#         # ax.set_xticks(range(len(seq_block3)))
#         # ax.set_xticklabels(seq_block3, rotation=45, ha="right")
#         # ax.legend()
#         # ax.grid(True)
#         # plt.tight_layout()
#         # plt.show()

#     return error_dict, error_ste


def extract_error_params(norm_counts, seq_names):
    """
    Bootstrap Pulse Error Extraction using Full Linear System
    Implements exact protocol from Dobrovitski et al., PRL 105, 077601 (2010)
    Assumes: ε'_y = 0 to fix gauge
    Returns angle errors (phi, chi) and axis tilt errors (ε and ν)
    """
    norm_counts = np.array(norm_counts)
    error_dict = {}
    error_ste = {}
    block3_signals = []
    seq_block3 = []

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
        elif seq in [
            "pi_2_Y_pi_2_X",
            "pi_2_X_pi_2_Y",
            "pi_2_X_pi_X_pi_2_Y",
            "pi_2_Y_pi_X_pi_2_X",
            "pi_2_X_pi_Y_pi_2_Y",
            "pi_2_Y_pi_Y_pi_2_X",
        ]:
            block3_signals.append(val)
            seq_block3.append(seq)

    if len(block3_signals) == 6:
        # Full 6x6 matrix from paper
        A_full = np.array(
            [
                [-1, -1, -1, -1, 0, 0],
                [-1, 1, -1, 1, 0, 0],
                [-1, 1, 1, -1, 2, 0],
                [-1, -1, 1, 1, 2, 0],
                [1, -1, -1, 1, 0, 2],
                [1, 1, -1, -1, 0, 2],
            ]
        )

        block3_names = [
            "epsilon_y_prime",  # fixed to 0, not solved
            "epsilon_z_prime",  # ε′_z
            "nu_x_prime",  # ν′_x
            "nu_z_prime",  # ν′_z
            "epsilon_y",  # ε_y
            "nu_x",  # ν_x
        ]

        block3_signals = np.array(block3_signals)
        # Manually fix ε'_y = 0
        A = A_full[:, 1:]  # Remove first column

        x = np.linalg.solve(
            A.T @ A, A.T @ block3_signals
        )  # exact solution using pseudo-inverse

        block3_names_used = block3_names[1:]  # skip ε'_y
        error_dict.update(dict(zip(block3_names_used, x)))

        predicted_signals = A @ x
        residuals = block3_signals - predicted_signals
        residual_norm = np.linalg.norm(residuals)

        # x, residuals, rank, s = np.linalg.lstsq(A, block3_signals, rcond=None)
        # cov_matrix = np.linalg.inv(A.T @ A)
        # std_errors = np.sqrt(np.diag(cov_matrix)) * np.std(block3_signals - A @ x)

        # error_dict.update(dict(zip(block3_names, x)))
        # error_ste.update(dict(zip(block3_names, std_errors)))
        print("Residual norm:", residual_norm)

    return error_dict, error_ste


def plot_pulse_errors(error_dict, error_ste=None):
    keys = list(error_dict.keys())
    values = [np.degrees(error_dict[k]) for k in keys]
    errors = [error_ste.get(k, 0) if error_ste else 0 for k in keys]
    for k, v in zip(keys, values):
        if not isinstance(v, (int, float, np.number)) or np.isnan(v):
            raise ValueError(f"Invalid value for {k}: {v}")

    # Mathematical labels using LaTeX
    math_labels = {
        "phi_prime": r"$\phi'$",
        "chi_prime": r"$\chi'$",
        "phi": r"$\phi$",
        "chi": r"$\chi$",
        "vz": r"$\nu_z$",
        "ez": r"$\epsilon_z$",
        "vx": r"$\nu'_x$",
        "ex": r"$\epsilon'_x$",
        "ey": r"$\epsilon_y$",
        "vy": r"$\nu_y$",
        "vz_alt": r"$\nu'_y$",
        "ez_alt": r"$\epsilon'_z$",
        "epsilon_z_prime": r"$\epsilon'_z$",
        "nu_x_prime": r"$\nu'_x$",
        "nu_z_prime": r"$\nu'_z$",
        "epsilon_y": r"$\epsilon_y$",
        "nu_x": r"$\nu_x$",
        "epsilon_z": r"$\epsilon_z$",
    }

    descriptions = {
        "phi_prime": r"Rotation angle error of $\pi/2\,X$ pulse",
        "chi_prime": r"Rotation angle error of $\pi/2\,Y$ pulse",
        "phi": r"Rotation angle error of $\pi\,X$ pulse",
        "chi": r"Rotation angle error of $\pi\,Y$ pulse",
        "vz": r"$Z$-axis tilt of $\pi\,Y$ pulse",
        "ez": r"$Z$-axis tilt of $\pi\,X$ pulse",
        "vx": r"$X$-axis tilt of $\pi/2\,Y$ pulse",
        "ex": r"$X$-axis tilt of $\pi/2\,X$ pulse",
        "ey": r"$Y$-axis tilt of $\pi\,X$ pulse",
        "vy": r"$Y$-axis tilt of $\pi\,Y$ pulse",
        "vz_alt": r"$Y$-axis tilt of $\pi/2\,Y$ pulse",
        "ez_alt": r"$Z$-axis tilt of $\pi/2\,Y$ pulse",
        "epsilon_z_prime": r"$Z$-axis tilt of $\pi/2\,X$ pulse",
        "nu_x_prime": r"$X$-axis tilt of $\pi/2\,Y$ pulse",
        "nu_z_prime": r"$Z$-axis tilt of $\pi/2\,Y$ pulse",
        "epsilon_y": r"$Y$-axis tilt of $\pi\,X$ pulse",
        "nu_x": r"$X$-axis tilt of $\pi\,Y$ pulse",
        "epsilon_z": r"$Z$-axis tilt of $\pi\,X$ pulse",
    }

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6, 6), gridspec_kw={"height_ratios": [2, 1]}
    )
    x_pos = np.arange(len(keys))
    # x_pos = range(len(keys))
    bars = ax1.bar(x_pos, values, yerr=errors, capsize=2)
    ax1.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(
        [math_labels.get(k, k) for k in keys], rotation=0, ha="right", fontsize=11
    )
    ax1.set_ylabel("Errors (degrees)", fontsize=12)
    ax1.set_title("Bootstrap Tomography of Pulse Errors", fontsize=14)
    ax1.tick_params(labelsize=11)

    for i, (bar, val) in enumerate(zip(bars, values)):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax2.axis("off")
    # text_block = ", ".join([f"{math_labels[k]}: {descriptions[k]}" for k in keys])
    # ax2.text(0.01, 0.8, text_block, fontsize=11, wrap=False)
    text_y = 1.0
    line_height = 0.12
    for k in keys:
        symbol = math_labels.get(k, k)
        label = descriptions.get(k, "Unknown")
        ax2.text(0.01, text_y, f"{symbol}: {label}", fontsize=11, va="top")
        text_y -= line_height
    plt.tight_layout()
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

    num_exps = len(seq_names) + 1  # last two exp are reference

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


# def normalize_to_sigma_z(raw_counts, bright_ref, dark_ref):
#     """Return signal mapped to [-1, 1] based on reference levels."""
#     return 2 * (raw_counts - dark_ref) / (bright_ref - dark_ref) - 1
#     # return (raw_counts - dark_ref) / (bright_ref - dark_ref)


def normalize_to_sigma_z_scc(counts, bright_ref, dark_ref):
    norm = (counts - dark_ref) / (bright_ref - dark_ref)
    sigma_z = 2 * norm - 1
    return -1 * sigma_z  # Flip for SCC interpretation


if __name__ == "__main__":
    kpl.init_kplotlib()
    # 32ns gap bewteen pi pulses due to buffer
    # file_ids = [
    #     1843336828775,
    #     1843444108428,
    #     1843662119540,
    # ]  # before correction
    # file_ids = [
    #     1844234382841,
    #     1844135699091,
    #     1844039507259,
    #     1843920112174,
    # ]  # after correction

    # no gap bewteen pi pulses
    # before correction
    # file_ids = [
    #     "2025_04_25-19_40_02-rubin-nv0_2025_02_26",
    #     "2025_04_25-23_30_53-rubin-nv0_2025_02_26",
    #     "2025_04_25-21_33_07-rubin-nv0_2025_02_26",
    #     "2025_04_26-01_32_52-rubin-nv0_2025_02_26",
    # ]
    # file_ids = ["2025_04_27-18_43_06-rubin-nv0_2025_02_26"]  # before
    # file_ids = ["2025_05_01-23_21_40-rubin-nv0_2025_02_26"]  # before
    file_ids = [
        "2025_05_02-02_23_53-rubin-nv0_2025_02_26",
        "2025_05_02-04_10_46-rubin-nv0_2025_02_26",
    ]  # before
    # file_ids = ["2025_05_02-04_10_46-rubin-nv0_2025_02_26"]  # before

    # data = dm.get_raw_data(file_id=file_id, load_npz=False, use_cache=True)
    data = widefield.process_multiple_files(file_ids=file_ids)
    # file_name = widefield.combined_filename(file_ids=file_ids)
    nv_list = data["nv_list"]
    seq_names = data["bootstrap_sequence_names"]
    num_nvs = len(nv_list)
    counts = np.array(data["counts"])
    ref_counts = counts[-1]  # last data is ref
    norm_counts = []
    for c in range(len(seq_names)):
        sig_counts = counts[c]  #
        nc, _ = widefield.process_counts(nv_list, sig_counts, threshold=False)
        bright_ref = np.max(nc)
        dark_ref = np.min(nc)
        nc = normalize_to_sigma_z_scc(nc, bright_ref, dark_ref)
        nc_medians = np.median(nc, axis=0)
        norm_counts.append(nc_medians)
        print(f"{bright_ref}, {dark_ref}")
    norm_counts = np.array(norm_counts)  # shape: (num_seqs, num_nvs)
    # file_name = dm.get_file_name(file_id=file_id)
    # print(f"{file_name}_{file_id}")
    error_dict, error_ste = extract_error_params(norm_counts, seq_names)
    formatted_dict = {k: round(v, 6) for k, v in error_dict.items()}
    print(formatted_dict)
    plot_pulse_errors(error_dict, error_ste)
    # Bloch_Sphere_Visualization(error_dict)
    plt.show(block=True)
