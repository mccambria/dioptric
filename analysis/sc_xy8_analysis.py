# -*- coding: utf-8 -*-
"""
Widefield Rabi experiment

Created on November 29th, 2023

@author: mccambria
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

import utils.tool_belt as tb
from majorroutines.widefield import base_routine
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import widefield
from utils import widefield as widefield


def process_and_plot_xy8(nv_list, taus, norm_counts, norm_counts_ste):
    num_nvs = len(nv_list)
    from scipy.optimize import curve_fit

    def stretched_exp(tau, a, t2, n, b):
        return a * (1 - np.exp(-((tau / t2) ** n))) + b

    T2_list = []
    n_list = []
    nv_indices = []
    chi2_list = []
    for nv_ind in range(num_nvs):
        nv_counts = norm_counts[nv_ind]
        nv_counts_ste = norm_counts_ste[nv_ind]

        # Skip low contrast or high noise
        if np.ptp(nv_counts) < 0.05 or np.mean(nv_counts_ste) > 0.1:
            print(f"NV {nv_ind} skipped: low contrast or noisy")
            continue

        a0 = np.clip(np.ptp(nv_counts), 0.1, 1.0)
        t2_0 = np.median(taus)
        n0 = 1.0
        b0 = np.min(nv_counts)
        p0 = [a0, t2_0, n0, b0]

        bounds = (
            [0, 1e2, 0.1, 0.0],
            [1.5, 1e7, 5.0, 1.0],
        )  # Lower bounds  # Upper bounds

        try:
            popt, _ = curve_fit(
                stretched_exp,
                taus,
                nv_counts,
                p0=p0,
                bounds=bounds,
                sigma=nv_counts_ste,
                absolute_sigma=True,
                maxfev=20000,
            )

            residuals = stretched_exp(taus, *popt) - nv_counts
            red_chi_sq = np.sum((residuals / nv_counts_ste) ** 2) / (
                len(taus) - len(popt)
            )

            if red_chi_sq > 10 or np.isnan(popt).any():
                print(f"NV {nv_ind} rejected: high χ² or NaNs")
                continue
            T2_list.append(popt[1])
            n_list.append(popt[2])
            nv_indices.append(nv_ind)
            chi2_list.append(red_chi_sq)

            T2 = round(popt[1], 1)
            n = round(popt[2], 2)
            print(f"NV {nv_ind} - T2 = {T2} ns, n = {n}, χ² = {red_chi_sq:.2f}")

        except Exception as e:
            print(f"NV {nv_ind} fit failed: {e}")
            continue
        # # plotting
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.errorbar(
            taus,
            nv_counts,
            yerr=np.abs(nv_counts_ste),
            fmt="o",
            capsize=3,
            label=f"NV {nv_ind}",
        )

        if popt is not None:
            tau_fit = np.linspace(min(taus), max(taus), 300)
            fit_vals = stretched_exp(tau_fit, *popt)
            ax.plot(tau_fit, fit_vals, "-", label="Fit")
            # Plot

        ax.set_title(f"XY8 Decay: NV {nv_ind}")
        ax.set_xlabel("τ (ns)")
        ax.set_ylabel("Normalized NV⁻ Population")
        # ax.set_xscale("symlog", linthresh=1e5)
        ax.legend()
        ax.grid(True)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        plt.show(block=True)

    # Convert T2 from ns → µs for plotting
    T2_list_us = [t2 / 1e3 for t2 in T2_list]
    median_T2_us = np.median(T2_list_us)

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(
        nv_indices,
        T2_list_us,
        c=n_list,
        s=40,
        edgecolors="k",
    )

    # Add median line
    ax.axhline(
        median_T2_us,
        color="r",
        linestyle="--",
        linewidth=0.5,
        label=f"Median T₂ ≈ {median_T2_us:.1f} µs",
    )

    # Annotate χ² values if provided
    if chi2_list is not None:
        for idx, chi2 in zip(nv_indices, chi2_list):
            ax.annotate(
                f"χ²={chi2:.2f}",
                (idx, T2_list_us[nv_indices.index(idx)]),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize=6,
                color="gray",
            )

    # Labels and formatting
    ax.set_xlabel("NV Index", fontsize=15)
    ax.set_ylabel("T₂ (µs)", fontsize=15)
    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.set_yscale("log")
    ax.set_title("T₂ per NV from XY8 Fits", fontsize=15)
    ax.grid(True, which="both", ls="--", alpha=0.6)

    cbar = plt.colorbar(scatter, ax=ax, pad=0.01)
    cbar.set_label("Stretching Exponent (n)", fontsize=15)
    cbar.ax.tick_params(labelsize=15)
    ax.legend(fontsize=11)
    plt.show()


def hybrid_tau_spacing(min_tau, max_tau, num_steps, log_frac=0.6):
    N_log = int(num_steps * log_frac)
    N_lin = num_steps - N_log

    log_max = 10 ** (
        np.log10(min_tau) + (np.log10(max_tau) - np.log10(min_tau)) * log_frac
    )
    taus_log = np.logspace(np.log10(min_tau), np.log10(log_max), N_log, endpoint=False)
    taus_lin = np.linspace(log_max, max_tau, N_lin)

    taus = np.unique(np.concatenate([taus_log, taus_lin]))
    taus = [round(tau / 4) * 4 for tau in taus]
    return taus


if __name__ == "__main__":
    kpl.init_kplotlib()

    # file_name = ""
    file_ids = [
        1818535967472,
        1818490062733,
        1818428014990,
        1818371630370,
        1818240906171,
    ]
    combined_filename = widefield.combined_filename(file_ids)
    print(f"File name: {combined_filename}")
    raw_data = widefield.process_multiple_files(file_ids)
    nv_list = raw_data["nv_list"]
    taus = np.array(raw_data["taus"])  # τ values (in ns)
    counts = np.array(raw_data["counts"])  # shape: (2, num_nvs, num_steps)
    sig_counts = counts[0]
    ref_counts = counts[1]

    # Normalize counts
    norm_counts, norm_counts_ste = widefield.process_counts(
        nv_list, sig_counts, ref_counts, threshold=True
    )

    num_nvs = len(nv_list)

    process_and_plot_xy8(nv_list, taus, norm_counts, norm_counts_ste)

    plt.show(block=True)
