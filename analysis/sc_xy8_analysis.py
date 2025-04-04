# -*- coding: utf-8 -*-
"""
Widefield Rabi experiment

Created on November 29th, 2023

@author: Saroj Chand
"""

import time

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
from scipy.optimize import curve_fit, least_squares

import utils.tool_belt as tb
from majorroutines.widefield import base_routine
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import widefield
from utils import widefield as widefield


def stretched_exp(tau, a, t2, n, b):
    n = 1.0
    return a * (1 - np.exp(-((tau / t2) ** n))) + b


def residuals(params, x, y, yerr):
    return (stretched_exp(x, *params) - y) / yerr


def process_and_fit_xy8(nv_list, taus, norm_counts, norm_counts_ste):
    num_nvs = len(nv_list)
    T2_list = []
    n_list = []
    chi2_list = []
    param_errors_list = []
    fit_params = []
    for nv_ind in range(num_nvs):
        nv_counts = norm_counts[nv_ind]
        nv_counts_ste = norm_counts_ste[nv_ind]

        # Optional skip for low contrast
        # if np.ptp(nv_counts) < 0.05 or np.mean(nv_counts_ste) > 0.2:
        #     print(f"NV {nv_ind} skipped: low contrast or noisy")
        #     continue

        # Initial guesses
        a0 = np.clip(np.ptp(nv_counts), 0.1, 1.0)
        t2_0 = np.median(taus)
        n0 = 1.0
        # b0 = np.min(nv_counts)
        b0 = np.mean(nv_counts[-4:])
        p0 = [a0, t2_0, n0, b0]

        bounds = (
            [0, 1e-1, 0.01, -10.0],
            [1.5, 1e4, 11.0, 10.0],
        )
        try:
            popt, pcov = curve_fit(
                stretched_exp,
                taus,
                nv_counts,
                p0=p0,
                bounds=bounds,
                sigma=nv_counts_ste,
                absolute_sigma=True,
                maxfev=20000,
            )

            # χ²
            residuals = stretched_exp(taus, *popt) - nv_counts
            red_chi_sq = np.sum((residuals / nv_counts_ste) ** 2) / (
                len(taus) - len(popt)
            )
            # Parameter uncertainties
            param_errors = np.sqrt(np.diag(pcov))

            T2_list.append(popt[1])
            n_list.append(popt[2])
            fit_params.append(popt)
            chi2_list.append(red_chi_sq)
            param_errors_list.append(param_errors)
            print(
                f"NV {nv_ind} - T2 = {popt[1]:.1f} ns, n = {popt[2]:.2f}, χ² = {red_chi_sq:.2f}"
            )

        except Exception as e:
            print(f"NV {nv_ind} fit failed: {e}")
            continue

    return (
        fit_params,
        T2_list,
        n_list,
        chi2_list,
        param_errors_list,
    )


def process_and_plot_xy8(nv_list, taus, norm_counts, norm_counts_ste):
    num_nvs = len(nv_list)

    T2_list = []
    n_list = []
    nv_indices = []
    chi2_list = []
    T2_errs = []
    fit_params = []
    for nv_ind in range(num_nvs):
        nv_counts = norm_counts[nv_ind]
        nv_counts_ste = norm_counts_ste[nv_ind]
        max_counts = np.max(nv_counts)
        # Skip low contrast or high noise
        # if np.ptp(nv_counts) < 0.05 or np.mean(nv_counts_ste) > 0.1:
        #     print(f"NV {nv_ind} skipped: low contrast or noisy")
        #     continue

        a0 = np.clip(np.ptp(nv_counts), 0.1, 1.0)
        t2_0 = np.median(taus)
        n0 = 1.0
        b0 = np.min(nv_counts)
        p0 = [a0, t2_0, n0, b0]

        bounds = (
            [0, 1e-1, 0.01, -10.0],
            [1.5, 1e4, 11.0, 10.0],
        )  # Lower bounds  # Upper bounds

        try:
            popt, pcov = curve_fit(
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
            # if red_chi_sq > 1.0:
            #     pcov *= red_chi_sq
            param_errors = np.sqrt(np.diag(pcov))
            # if red_chi_sq > 10 or np.isnan(popt).any():
            #     print(f"NV {nv_ind} rejected: high χ² or NaNs")
            #     continue

            ### Manaual fit with least squares
            # try:
            #     result = least_squares(
            #         residuals,
            #         p0,
            #         args=(taus, nv_counts, nv_counts_ste),
            #         bounds=bounds,
            #         jac="2-point",
            #         max_nfev=20000,
            #     )
            #     popt = result.x

            #     # Compute chi-squared
            #     res = residuals(popt, taus, nv_counts, nv_counts_ste)
            #     red_chi_sq = np.sum(res**2) / (len(taus) - len(popt))

            #     # Estimate parameter uncertainties
            #     try:
            #         dof = max(0, len(taus) - len(result.x))
            #         residual_var = np.sum(result.fun**2) / dof
            #         J = result.jac
            #         cov = np.linalg.pinv(J.T @ J) * residual_var
            #         param_errors = np.sqrt(np.diag(cov))
            #     except Exception as e:
            #         print(f"Error estimating param errors for NV {nv_ind}: {e}")
            #         param_errors = np.full_like(result.x, np.nan)

            fit_params.append(popt)
            T2_list.append(popt[1])
            n_list.append(popt[2])
            nv_indices.append(nv_ind)
            chi2_list.append(red_chi_sq)
            T2_errs.append(param_errors[1])
            T2 = round(popt[1], 1)
            n = round(popt[2], 2)
            print(f"NV {nv_ind} - T2 = {T2} us, n = {n}, χ² = {red_chi_sq:.2f}")
        except Exception as e:
            print(f"NV {nv_ind} fit failed: {e}")
            # continue
        # fit funtions
        # fit_funtion = lambda x: stretched_exp(x, *popt)
        # fit_functions.append(fit_funtion)
        # # plotting
        # fig, ax = plt.subplots(figsize=(6, 5))
        # ax.errorbar(
        #     taus,
        #     max_counts - nv_counts,
        #     yerr=np.abs(nv_counts_ste),
        #     fmt="o",
        #     capsize=3,
        #     label=f"NV {nv_ind}",
        # )
        # # fit funtions
        # if popt is not None:
        #     tau_fit = tau_fit = np.logspace(
        #         np.log10(min(taus)), np.log10(max(taus)), 200
        #     )
        #     fit_vals = max_counts - stretched_exp(tau_fit, *popt)
        #     ax.plot(tau_fit, fit_vals, "-", label="Fit")
        # ax.set_title(f"XY8 Decay: NV {nv_ind} - T₂ = {T2} µs, n = {n}", fontsize=15)
        # ax.set_xlabel("τ (µs)", fontsize=15)
        # ax.set_ylabel("Norm. NV⁻ Population", fontsize=15)
        # ax.tick_params(axis="both", labelsize=15)
        # # ax.set_xscale("symlog", linthresh=1e5)
        # ax.set_xscale("log")
        # # ax.set_yscale("log")
        # # ax.legend()
        # # ax.grid(True)
        # # ax.spines["right"].set_visible(False)
        # # ax.spines["top"].set_visible(False)
        # plt.show(block=True)
        # Convert T2 from ns → µs for plotting
    ## plot T2
    # Define outliers
    # outlier_indices = [0, 10, 36]

    # Create a copy of T2_list for plotting with clipped or masked values
    T2_list_plot = T2_list.copy()
    T2_clipped_vals = []

    # Replace outlier T2s with np.nan or a capped value for visualization
    # for i in outlier_indices:
    #     T2_clipped_vals.append(T2_list_plot[i])
    #     T2_list_plot[i] = np.nan  # or use: T2_list_plot[i] = np.percentile(T2_list, 95)

    median_T2_us = np.median(T2_list)
    nv_indices = np.arange(num_nvs)

    ###plot
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(
        nv_indices,
        T2_list_plot,
        c=chi2_list,
        s=40,
        edgecolors="blue",
    )
    # Error bars using errorbar (no color map here)
    if T2_errs is not None:
        ax.errorbar(
            nv_indices,
            T2_list_plot,
            yerr=T2_errs,
            fmt="none",
            ecolor="gray",
            elinewidth=1,
            capsize=3,
            alpha=0.7,
            zorder=1,
        )
    # Add median line
    ax.axhline(
        median_T2_us,
        color="r",
        linestyle="--",
        linewidth=0.8,
        label=f"Median T2 ≈ {median_T2_us:.1f} µs",
    )

    # Annotate χ² values if provided
    if n_list is not None:
        for idx, n in zip(nv_indices, chi2_list):
            ax.annotate(
                f"n={n:.2f}",
                (idx, T2_list[idx]),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize=6,
                color="black",
            )
    # Stretching exponent note
    ax.text(
        0.99,
        0.95,
        "n is the stretching exponent in the fit",
        transform=ax.transAxes,
        fontsize=10,
        color="dimgray",
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="whitesmoke"),
    )
    # Labels and formatting
    ax.set_xlabel("NV Index", fontsize=15)
    ax.set_ylabel("T2 (µs)", fontsize=15)
    ax.tick_params(axis="both", which="major", labelsize=15)
    # ax.set_yscale("log")
    ax.set_title("T2 per NV (XY8-1, 185 MHz Orientation)", fontsize=15)
    ax.grid(True, which="both", ls="--", alpha=0.6)
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Reduced χ²", fontsize=15)
    cbar.ax.tick_params(labelsize=15)
    ax.legend(fontsize=11)
    plt.show()

    ### plot all
    sns.set(style="whitegrid")
    num_cols = 8
    num_nvs = len(nv_list)
    num_rows = int(np.ceil(num_nvs / num_cols))
    # Full plot
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(num_cols * 1.8, num_rows * 3),
        sharex=True,
        sharey=False,
        constrained_layout=True,
        gridspec_kw={"wspace": 0.0, "hspace": 0.0},
    )
    axes = axes.flatten()
    # axes = axes[::-1]
    for nv_idx, ax in enumerate(axes):
        nv_counts = norm_counts[nv_idx]
        if nv_idx >= len(nv_list):
            ax.axis("off")
            continue
        sns.scatterplot(
            x=taus,
            y=max_counts - nv_counts,
            ax=ax,
            color="blue",
            label=f"NV {nv_idx}(T2 = {T2_list[nv_idx]:.2f} ± {T2_errs[nv_idx]:.2f} us)",
            # label=f"NV {nv_idx}(T1 = {T2_list[nv_idx]:.2f} us)",
            s=10,
            alpha=0.7,
        )
        # Plot error bars separately for clarity
        ax.errorbar(
            taus,
            max_counts - norm_counts[nv_idx],
            yerr=norm_counts_ste[nv_idx],
            fmt="o",
            alpha=0.9,
            ecolor="gray",
            markersize=0.1,
        )
        # Plot fitted curve if available
        popt = fit_params[nv_idx]
        if popt is not None:
            taus_fit = tau_fit = np.logspace(
                np.log10(min(taus)), np.log10(max(taus)), 200
            )
            fit_vals = max_counts - stretched_exp(tau_fit, *popt)
            sns.lineplot(
                x=taus_fit,
                y=fit_vals,
                ax=ax,
                # color="blue",
                # label='Fit',
                lw=1,
            )
        ax.legend(fontsize="xx-small")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax.tick_params(labelleft=False)
        ax.set_xscale("log")
        # # Add NV index within the plot at the center
        axes_grid = np.array(axes).reshape((num_rows, num_cols))
        # Loop over each column
        for col in range(num_cols):
            # Go from bottom row upwards
            for row in reversed(range(num_rows)):
                if row * num_cols + col < len(axes):  # Check if subplot exists
                    ax = axes_grid[row, col]

                    # Apply ticks
                    tick_positions = np.logspace(
                        np.log10(taus[0]), np.log10(taus[-1] - 2), 4
                    )
                    ax.set_xticks(tick_positions)
                    ax.set_xticklabels(
                        [f"{tick:.0f}" for tick in tick_positions],
                        rotation=45,
                        fontsize=9,
                    )
                    for label in ax.get_xticklabels():
                        label.set_y(0.08)
                    # Label
                    ax.set_xlim(taus[0] - 0.4, taus[-1])
                    ax.set_xlabel("Time (μs)", fontsize=11, labelpad=1)
                    break  # Done for this column

    fig.text(
        0.005, 0.5, "NV$^{-}$ Population", va="center", rotation="vertical", fontsize=11
    )
    fig.suptitle(
        f"XY8-1 T2 Fits (185 MHz Orientation - {all_file_ids_str})", fontsize=12
    )
    fig.tight_layout(pad=0.4, rect=[0.01, 0.01, 0.99, 0.99])
    plt.show(block=True)


def plot_xy8(
    nv_list,
    taus,
    norm_counts,
    norm_counts_ste,
    T2_list,
    n_list,
    chi2_list,
    fit_funtions,
):
    # plotting
    num_nvs = len(nv_list)
    for nv_ind in range(num_nvs):
        nv_counts = norm_counts[nv_ind]
        nv_counts_ste = norm_counts_ste[nv_ind]
        max_counts = np.max(nv_counts)
        T2 = T2_list[nv_ind]
        n = n_list[nv_ind]
        # plotting
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.errorbar(
            taus,
            max_counts - nv_counts,
            yerr=np.abs(nv_counts_ste),
            fmt="o",
            capsize=3,
            label=f"NV {nv_ind}",
        )
        # fit funtions
        tau_fit = tau_fit = np.logspace(np.log10(min(taus)), np.log10(max(taus)), 200)
        fit_vals = max_counts - fit_funtions[nv_ind](tau_fit)

        ax.plot(tau_fit, fit_vals, "-", label="Fit")
        ax.set_title(f"XY8 Decay: NV {nv_ind} - T₂ = {T2} µs, n = {n}", fontsize=15)
        ax.set_xlabel("τ (µs)", fontsize=15)
        ax.set_ylabel("Norm. NV⁻ Population", fontsize=15)
        ax.tick_params(axis="both", labelsize=15)
        # ax.set_xscale("symlog", linthresh=1e5)
        ax.set_xscale("log")
        # ax.set_yscale("log")
        # ax.legend()
        # ax.grid(True)
        # ax.spines["right"].set_visible(False)
        # ax.spines["top"].set_visible(False)
        plt.show(block=True)

    # Convert T2 from ns → µs for plotting
    median_T2_us = np.median(T2_list)
    nv_indices = np.arange(num_nvs)
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(
        nv_indices,
        T2_list,
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
                (idx, T2_list[nv_indices.index(idx)]),
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


def plot_fitted_data(
    nv_list,
    taus,
    norm_counts,
    norm_counts_ste,
    fit_functions,
    fit_params,
    fit_errors,
    num_cols=8,
    selected_indices=None,
):
    """Plot for raw data with fitted curves using Seaborn style, including NV index labels."""
    fit_params = np.array(fit_params)
    param_errors = np.array(fit_errors)
    rates = fit_params[:, 1]
    rate_errors = param_errors[:, 1]
    T1 = 1 / rates
    T1_err = rate_errors / (rates**2)
    T1, T1_err = list(T1), list(T1_err)

    sns.set(style="whitegrid")
    num_nvs = len(nv_list)
    num_rows = int(np.ceil(num_nvs / num_cols))
    # Full plot
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(num_cols * 1.5, num_rows * 3),
        sharex=True,
        sharey=False,
        constrained_layout=True,
        gridspec_kw={"wspace": 0.0, "hspace": 0.0},
    )
    axes = axes.flatten()
    # axes = axes[::-1]
    for nv_idx, ax in enumerate(axes):
        if nv_idx >= len(nv_list):
            ax.axis("off")
            continue
        if selected_indices is not None:
            nv_idx_label = selected_indices[nv_idx]
        else:
            nv_idx_label = nv_idx
        sns.scatterplot(
            x=taus,
            y=norm_counts[nv_idx],
            ax=ax,
            color="blue",
            label=f"NV {nv_idx_label}(T1 = {T1[nv_idx]:.2f} ± {T1_err[nv_idx]:.2f} ms)",
            s=10,
            alpha=0.7,
        )
        # Plot error bars separately for clarity
        ax.errorbar(
            taus,
            norm_counts[nv_idx],
            yerr=norm_counts_ste[nv_idx],
            fmt="o",
            alpha=0.9,
            ecolor="gray",
            markersize=0.1,
        )

        taus_fit = np.logspace(np.log10(taus[0]), np.log10(taus[-1]), 200)
        # Plot fitted curve if available
        if fit_functions[nv_idx]:
            fit_curve = fit_functions[nv_idx](taus_fit)
            sns.lineplot(
                x=taus_fit,
                y=fit_curve,
                ax=ax,
                # color="blue",
                # label='Fit',
                lw=1,
            )
        ax.legend(fontsize="xx-small")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax.tick_params(labelleft=False)
        ax.set_xscale("log")
        # # Add NV index within the plot at the center
        # for col in range(num_cols):
        #     bottom_row_idx = num_rows * num_cols - num_cols + col
        #     if bottom_row_idx < len(axes):
        #         ax = axes[bottom_row_idx]
        #         # tick_positions = np.linspace(min(taus), max(taus), 5)
        #         tick_positions = np.logspace(np.log10(taus[0]), np.log10(taus[-1]), 6)
        #         ax.set_xticks(tick_positions)
        #         ax.set_xticklabels(
        #             [f"{tick:.2f}" for tick in tick_positions],
        #             rotation=45,
        #             fontsize=9,
        #         )
        #         ax.set_xlabel("Time (ms)")
        #     else:
        #         ax.set_xticklabels([])

        # num_axes = len(axes)
        axes_grid = np.array(axes).reshape((num_rows, num_cols))

        # Loop over each column
        for col in range(num_cols):
            # Go from bottom row upwards
            for row in reversed(range(num_rows)):
                if row * num_cols + col < len(axes):  # Check if subplot exists
                    ax = axes_grid[row, col]

                    # Apply ticks
                    tick_positions = np.logspace(
                        np.log10(taus[0]), np.log10(taus[-1]), 6
                    )
                    ax.set_xticks(tick_positions)
                    ax.set_xticklabels(
                        [f"{tick:.2f}" for tick in tick_positions],
                        rotation=45,
                        fontsize=9,
                    )
                    ax.set_xlabel("Time (ms)")
                    break  # Done for this column

    fig.text(
        0.005,
        0.5,
        "NV$^{-}$ Population",
        va="center",
        rotation="vertical",
        fontsize=12,
    )
    fig.suptitle(f"T1 Relaxation ({all_file_ids_str})", fontsize=15)
    fig.tight_layout(pad=0.4, rect=[0.01, 0.01, 0.99, 0.99])
    plt.show(block=True)


if __name__ == "__main__":
    kpl.init_kplotlib()

    # 68MHz orientation --> spacing is not logrithmic
    # file_ids = [
    #     1818535967472,
    #     1818490062733,
    #     1818428014990,
    #     1818371630370,
    #     1818240906171,
    # ]

    # 185MHz orientation
    file_ids = [
        1818816006504,
        1818985247947,
        1819154094977,
        1819318427055,
        1819466247867,
        1819611450115,
    ]
    # 68MHz orientation
    # file_ids = [
    #     1820856154901,
    #     1820741644537,
    #     1820575030849,
    #     1820447821240,
    #     1820301119952,
    #     1820151354472,
    # ]
    file_path, all_file_ids_str = widefield.combined_filename(file_ids)
    print(f"File name: {file_path}")
    raw_data = widefield.process_multiple_files(file_ids)
    nv_list = raw_data["nv_list"]
    taus = 2 * np.array(raw_data["taus"]) / 1e3  # τ values (in us)
    # taus = 2 * np.array(raw_data["taus"]) / 2e3  # τ values (in us)
    # taus = 2 * 8 * taus
    counts = np.array(raw_data["counts"])  # shape: (2, num_nvs, num_steps)
    sig_counts = counts[0]
    ref_counts = counts[1]
    # Normalize counts
    norm_counts, norm_counts_ste = widefield.process_counts(
        nv_list, sig_counts, ref_counts, threshold=True
    )
    process_and_plot_xy8(nv_list, taus, norm_counts, norm_counts_ste)

    # fit_params, T2_list, n_list, chi2_list, param_errors_list = process_and_fit_xy8(
    #     nv_list, taus, norm_counts, norm_counts_ste
    # )
    # plot_xy8(nv_list, taus, norm_counts, norm_counts_ste, T2_list, n_list, fit_params)
    plt.show(block=True)
