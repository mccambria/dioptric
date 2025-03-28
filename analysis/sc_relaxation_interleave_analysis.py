# -*- coding: utf-8 -*-
"""
Created on Fall, 2024

@author: saroj chand
"""
from utils import data_manager as dm
from utils import kplotlib as kpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
from scipy.optimize import curve_fit, least_squares
from scipy.ndimage import gaussian_filter1d
from utils import widefield as widefield


def exp_decay(tau, norm, rate, offset=0):
    """Exponential decay function."""
    return norm * np.exp(-rate * tau) + offset


def double_exp_decay(tau, norm1, rate1, norm2, rate2, offset=0):
    """Double exponential decay function for more complex fitting scenarios."""
    return norm1 * np.exp(-rate1 * tau) + norm2 * np.exp(-rate2 * tau) + offset


def T1_fit(tau, y, yerr=None, model="single"):
    """Performs fitting using least squares with different loss functions."""
    if not np.isfinite(y).all() or not np.isfinite(tau).all():
        raise ValueError("Input data contains non-finite values.")

    epsilon = 1e-10  # Small value to prevent division by zero

    if model == "single":

        def residuals(params):
            return (exp_decay(tau, *params) - y) / (
                yerr + epsilon if yerr is not None else 1
            )

        initial_guess = [y[0], 1 / (tau[-1] - tau[0]), 0]
    elif model == "double":

        def residuals(params):
            return (double_exp_decay(tau, *params) - y) / (
                yerr + epsilon if yerr is not None else 1
            )

        initial_guess = [
            y[0] / 2,
            1 / (tau[-1] - tau[0]),
            y[0] / 2,
            1 / (tau[-1] - tau[0]),
            0,
        ]
    else:
        raise ValueError("Invalid model specified.")

    # print("Initial guess:", initial_guess)  # Debugging print statement

    result = least_squares(residuals, initial_guess, loss="soft_l1")
    # Estimate parameter standard errors
    try:
        # Degrees of freedom = number of points - number of parameters
        dof = max(0, len(tau) - len(result.x))
        residual_var = np.sum(result.fun**2) / dof
        J = result.jac
        cov = np.linalg.inv(J.T @ J) * residual_var
        param_errors = np.sqrt(np.diag(cov))
    except:
        param_errors = np.full_like(result.x, np.nan)

    return result.x, result.success, param_errors


def process_and_fit_data(data, use_double_fit=False):
    """Processes and fits NV relaxation data with robust fitting."""
    nv_list = data["nv_list"]
    taus = np.array(data["taus"]) / 1e6  # Convert ns to ms
    counts = np.array(data["counts"])
    sig_counts, ref_counts = counts[0], counts[1]

    # Process counts using widefield's process_counts function
    sig_avg_counts, sig_avg_counts_ste = widefield.process_counts(
        nv_list, sig_counts, threshold=False
    )
    ref_avg_counts, ref_avg_counts_ste = widefield.process_counts(
        nv_list, ref_counts, threshold=False
    )
    # Compute the difference between the states
    norm_counts = sig_avg_counts - ref_avg_counts
    norm_counts_ste = np.sqrt(sig_avg_counts_ste**2 + ref_avg_counts_ste**2)

    fit_params, fit_functions, residuals, param_errors = [], [], [], []

    for nv_idx in range(len(nv_list)):
        nv_counts = gaussian_filter1d(
            norm_counts[nv_idx], sigma=1
        )  # Smoothing for stability
        nv_counts_ste = norm_counts_ste[nv_idx]

        # Try single exponential first
        params, success, errors = T1_fit(
            taus, nv_counts, yerr=nv_counts_ste, model="single"
        )
        if not success and use_double_fit:
            params, success = T1_fit(
                taus, nv_counts, yerr=nv_counts_ste, model="double"
            )

        fit_curve = (
            exp_decay(taus, *params)
            if len(params) == 3
            else double_exp_decay(taus, *params)
        )
        fit_params.append(params)
        param_errors.append(errors)
        fit_functions.append(
            lambda t, p=params: (
                exp_decay(t, *p[:3]) if len(p) == 3 else double_exp_decay(t, *p)
            )
        )
        residuals.append(nv_counts - fit_curve)

    return (
        fit_params,
        fit_functions,
        residuals,
        taus,
        norm_counts,
        norm_counts_ste,
        nv_list,
        param_errors,
    )


def plot_fitted_data(
    nv_list,
    taus,
    norm_counts,
    norm_counts_ste,
    fit_functions,
    fit_params,
    fit_errors,
    num_cols=8,
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
        sns.scatterplot(
            x=taus,
            y=norm_counts[nv_idx],
            ax=ax,
            color="blue",
            label=f"NV {nv_idx}(T1 = {T1[nv_idx]:.2f} ± {T1_err[nv_idx]:.2f} ms)",
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
                color="blue",
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
    fig.suptitle(f"T1 Relaxation {all_file_ids_str}", fontsize=16)
    fig.tight_layout(pad=0.4, rect=[0.01, 0.01, 0.99, 0.99])
    plt.show(block=True)


def plot_fitted_data_separately(
    nv_list,
    taus,
    norm_counts,
    norm_counts_ste,
    fit_functions,
    fit_params,
    param_errors,
):
    """Plot separate figures for each NV with fitted curves using Seaborn style."""
    # sns.set(style="whitegrid")
    fit_params = np.array(fit_params)
    param_errors = np.array(param_errors)
    rates = fit_params[:, 1]
    rate_errors = param_errors[:, 1]
    T1 = 1 / rates
    T1_err = rate_errors / (rates**2)
    T1, T1_err = list(T1), list(T1_err)
    for nv_idx in range(len(nv_list)):
        fig, ax = plt.subplots(figsize=(6, 5))
        # Plot data with Seaborn's lineplot for a cleaner look
        sns.scatterplot(
            x=taus,
            y=norm_counts[nv_idx],
            ax=ax,
            color="blue",
            label=f"NV {nv_idx}",
            s=40,  # Size of markers
            alpha=0.7,
        )

        # Plot error bars separately for clarity
        ax.errorbar(
            taus,
            norm_counts[nv_idx],
            yerr=norm_counts_ste[nv_idx],
            fmt="o",
            alpha=0.5,
            ecolor="gray",
        )
        taus_fit = np.logspace(np.log10(taus[0]), np.log10(taus[-1]), 200)
        # Plot fitted curve if available
        if fit_functions[nv_idx]:
            fit_curve = fit_functions[nv_idx](taus_fit)
            sns.lineplot(
                x=taus_fit,
                y=fit_curve,
                ax=ax,
                # color="red",
                # label='Fit',
                lw=1.5,
            )

        # Add titles and labels
        ax.tick_params(axis="both", labelsize=15)

        ax.set_title(
            f"NV {nv_idx} (T1 = {T1[nv_idx]:.2f} ± {T1_err[nv_idx]:.2f} ms)",
            fontsize=15,
        )

        ax.set_xlabel("Relaxation Time (ms)", fontsize=15)
        ax.set_ylabel("Normalized Counts", fontsize=15)
        ax.set_xscale("log")
        # ax.set_yscale("log")
        ax.legend(fontsize="small")
        ax.grid(True, linestyle="--", linewidth=0.5)

        # Save each figure with a unique name
        now = datetime.now()
        date_time_str = now.strftime("%Y%m%d_%H%M%S")
        file_name = f"nv_{nv_idx}_fitted_plot_{date_time_str}.png"  # Example filename
        # fig.savefig(file_name)
        # plt.close(fig)  # Close the figure after saving to avoid display issues
        plt.show(block=True)


def plot_T1_with_errorbars(fit_params, param_errors, nv_list=None):
    fit_params = np.array(fit_params)
    param_errors = np.array(param_errors)
    nv_indices = np.arange(1, len(fit_params) + 1)
    nv_ticks = np.linspace(1, len(fit_params) + 1, 6)

    if fit_params.shape[1] == 3:
        rates = fit_params[:, 1]
        rate_errors = param_errors[:, 1]

        T1 = 1 / rates
        T1_err = rate_errors / (rates**2)

        plt.errorbar(
            nv_indices,
            T1,
            yerr=T1_err,
            fmt="o",
            capsize=5,
            label="T1 (Median: {:.2f} ms)".format(np.median(T1)),
            # color="blue",
            ecolor="gray",
        )

        # plt.xticks(nv_indices, nv_indices, rotation=45, ha="right")
        plt.xticks(nv_ticks, fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlabel("NV Index", fontsize=15)
        plt.ylabel("T1 (ms)", fontsize=15)
        plt.title("T1 Relaxation Times", fontsize=15)
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.legend()
        plt.show()
    else:
        print("Double exponential model not yet supported in this plot.")


# Combine data from multiple file IDs
def process_multiple_files(file_ids):
    """
    Load and combine data from multiple file IDs.
    """
    combined_data = dm.get_raw_data(file_id=file_ids[0])
    for file_id in file_ids[1:]:
        new_data = dm.get_raw_data(file_id=file_id)
        combined_data["num_runs"] += new_data["num_runs"]
        combined_data["counts"] = np.append(
            combined_data["counts"], new_data["counts"], axis=2
        )
    return combined_data


# Usage Example
if __name__ == "__main__":
    kpl.init_kplotlib()
    file_ids = [1811090565552, 1811157726815, 1811214987613, 1811261374947]
    all_file_ids_str = "_".join(map(str, file_ids))
    now = datetime.now()
    date_time_str = now.strftime("%Y%m%d_%H%M%S")
    file_name = dm.get_file_name(file_id=file_ids[0])
    timestamp = dm.get_time_stamp()
    file_path = dm.get_file_path(
        __file__, file_name, f"{all_file_ids_str}_{date_time_str}"
    )

    print(f"File path: {file_path}")

    data = process_multiple_files(file_ids)
    # data = dm.get_raw_data(file_id=1550610460299)  # Example file ID
    (
        fit_params,
        fit_functions,
        residuals,
        taus,
        norm_counts,
        norm_counts_ste,
        nv_list,
        fit_errors,
    ) = process_and_fit_data(data, use_double_fit=True)
    plot_fitted_data(
        nv_list,
        taus,
        norm_counts,
        norm_counts_ste,
        fit_functions,
        fit_params,
        fit_errors,
        num_cols=10,
    )
    # scatter_fitted_parameters(fit_params, nv_list)
    plot_T1_with_errorbars(fit_params, fit_errors, nv_list)
    # plot_fitted_data_separately(
    #     nv_list,
    #     taus,
    #     norm_counts,
    #     norm_counts_ste,
    #     fit_functions,
    #     fit_params,
    #     fit_errors,
    # )
    kpl.show(block=True)
