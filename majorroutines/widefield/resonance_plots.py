import os
import sys
import time
import traceback
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np

from majorroutines.pulsed_resonance import fit_resonance, norm_voigt, voigt, voigt_split
from majorroutines.widefield import base_routine, optimize
from utils import common
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils import widefield as widefield
from utils.constants import NVSig, NVSpinState
from utils.positioning import get_scan_1d as calculate_freqs


def process_data(data):
    """
    Process the raw data for NV centers, including averaging counts, normalization, and
    setting up the parameters for fitting and visualization.

    Args:
        data: The raw data dictionary containing NVs, counts, frequencies, and other metadata.

    Returns:
        avg_counts: Averaged counts for NVs.
        avg_counts_ste: Standard error of averaged counts.
        norms: Normalization data for NVs.
        freqs: Frequency data.
        nv_list: List of NV signatures.
    """
    nv_list = data["nv_list"]
    freqs = data["freqs"]
    counts = np.array(data["counts"])

    # Process the counts using the widefield utilities
    avg_counts, avg_counts_ste, norms = widefield.process_counts(
        nv_list, counts, threshold=False
    )

    return avg_counts, avg_counts_ste, norms, freqs, nv_list


def plot_data(nv_list, freqs, avg_counts, avg_counts_ste):
    """
    Plot the raw data for NV centers.

    Args:
        nv_list: List of NV signatures.
        freqs: Frequency data.
        avg_counts: Averaged counts for NVs.
        avg_counts_ste: Standard error of averaged counts.

    Returns:
        fig: The matplotlib figure object containing the raw data plot.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for nv_idx, nv_sig in enumerate(nv_list):
        ax.errorbar(
            freqs,
            avg_counts[nv_idx],
            yerr=avg_counts_ste[nv_idx],
            label=f"NV {nv_idx+1}",
        )

    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Normalized NV$^{-}$ Population")
    ax.legend(loc="best", ncol=2, fontsize=8)  # Add legend with NV labels

    plt.title("Raw Data for NV Centers")

    # Manually adjust the layout
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)

    return fig


def plot_fit_data(nv_list, freqs, avg_counts, avg_counts_ste, norms):
    """
    Plot the fitted data for NV centers.

    Args:
        nv_list: List of NV signatures.
        freqs: Frequency data.
        avg_counts: Averaged counts for NVs.
        avg_counts_ste: Standard error of averaged counts.
        norms: Normalization data.

    Returns:
        fig: The matplotlib figure object containing the fitted data plot.
    """
    fig, axes_pack = plt.subplots(
        len(nv_list), 1, figsize=(10, len(nv_list) * 2), sharex=True
    )

    for nv_idx, ax in enumerate(axes_pack):
        ax.errorbar(
            freqs,
            avg_counts[nv_idx],
            yerr=avg_counts_ste[nv_idx],
            fmt="o",
            label=f"NV {nv_idx+1}",
        )
        # Plot fitted function (placeholder, you can replace with actual fitting code)
        ax.plot(freqs, avg_counts[nv_idx], label="Fit")
        ax.set_ylabel("Norm. NV$^{-}$ Pop.")
        ax.legend(loc="best", fontsize=8)

    ax.set_xlabel("Frequency (GHz)")

    # Adjust the layout manually
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.05, hspace=0.4)

    return fig


def visualize_large_nv_data(
    nv_list, freqs, avg_counts, avg_counts_ste, norms, batch_size=25
):
    """
    Visualize large NV datasets by batching NVs and plotting them in separate figures or subplots.

    Args:
        nv_list: List of NV signatures.
        freqs: Frequency data.
        avg_counts: Averaged counts for NVs.
        avg_counts_ste: Standard error of averaged counts.
        norms: Normalization data.
        batch_size: Number of NVs to plot in each batch.

    Returns:
        fig_list: List of figure objects for each batch.
    """
    num_nvs = len(nv_list)
    fig_list = []

    for i in range(0, num_nvs, batch_size):
        batch_nv_list = nv_list[i : i + batch_size]
        batch_avg_counts = avg_counts[i : i + batch_size]
        batch_avg_counts_ste = avg_counts_ste[i : i + batch_size]

        fig = plot_fit_data(
            batch_nv_list, freqs, batch_avg_counts, batch_avg_counts_ste, norms
        )
        fig_list.append(fig)

    return fig_list


import matplotlib.patches as patches


def plot_nv_resonance_data(
    nv_list, freqs, avg_counts, avg_counts_ste, file_path, num_cols=4
):
    """
    Plot the NV resonance data in multiple panels (grid) in the same figure and save the figure.

    Args:
        nv_list: List of NV signatures.
        freqs: Frequency data.
        avg_counts: Averaged counts for NVs.
        avg_counts_ste: Standard error of averaged counts.
        file_path: Path where the figure will be saved.
        num_cols: Number of columns for the grid layout.
    """
    # Number of NVs and rows/columns
    num_nvs = len(nv_list)
    num_rows = int(np.ceil(num_nvs / num_cols))  # Calculate the number of rows needed

    # Create a figure with a grid of subplots (num_rows x num_cols)
    fig, axes_pack = plt.subplots(
        num_rows,
        num_cols,
        figsize=(num_cols * 3, num_rows * 1.0),  # Increase height slightly
        sharex=True,
    )
    axes_pack = axes_pack.flatten()  # Flatten the axes array for easy indexing

    # Plot each NV in the corresponding subplot
    for nv_idx, ax in enumerate(axes_pack):
        if nv_idx < num_nvs:
            ax.errorbar(
                freqs,
                avg_counts[nv_idx],
                yerr=avg_counts_ste[nv_idx],
                fmt="o",
                label=f"NV {nv_idx+1}",
                markersize=4,
                color="blue",
            )
            # Auto-scaling of y-axis to fit data better
            ax.set_ylim([min(avg_counts[nv_idx]) - 0.05, max(avg_counts[nv_idx]) + 0.3])

            # Only label the y-axis for the leftmost column
            if nv_idx % num_cols == 0:
                ax.set_ylabel("Norm. NV$^{-}$ Pop.")
            else:
                ax.set_yticklabels([])  # Remove y-axis ticks for non-leftmost subplots

            # Only label the x-axis for the bottom row
            if nv_idx >= (num_rows - 1) * num_cols:
                ax.set_xlabel("Frequency (GHz)")
            else:
                ax.set_xticklabels([])  # Remove x-axis ticks for non-bottom subplots
        else:
            # Hide any unused subplots if the number of NVs is less than the grid size
            ax.axis("off")

    # Adjust layout to ensure nothing overlaps and reduce vertical gaps
    plt.subplots_adjust(
        left=0.1, right=0.95, top=0.95, bottom=0.03, hspace=0.03, wspace=0.03
    )

    # Save the figure to the specified file path
    plt.savefig(file_path, bbox_inches="tight")

    # Close the figure to free up memory
    plt.close(fig)


import matplotlib.pyplot as plt
import seaborn as sns


def plot_nv_resonance_data_sns_with_freq_labels(
    nv_list, freqs, avg_counts, avg_counts_ste, file_path, num_cols=3, title = None
):
    """
    Plot the NV resonance data using Seaborn aesthetics in multiple panels (grid) in the same figure,
    add frequency values at the bottom of each column, and save the figure.

    Args:
        nv_list: List of NV signatures.
        freqs: Frequency data.
        avg_counts: Averaged counts for NVs.
        avg_counts_ste: Standard error of averaged counts.
        file_path: Path where the figure will be saved.
        num_cols: Number of columns for the grid layout.
    """
    # Use Seaborn style
    sns.set(style="whitegrid", palette="muted")

    # Number of NVs and rows/columns
    num_nvs = len(nv_list)
    num_rows = int(np.ceil(num_nvs / num_cols))  # Calculate the number of rows needed

    # Create a figure with a grid of subplots (num_rows x num_cols)
    fig, axes_pack = plt.subplots(
        num_rows,
        num_cols,
        figsize=(
            num_cols * 3,
            num_rows * 1,
        ),  # Adjust the figure size for better visibility
        sharex=True,
        sharey=False,  # Allow y-axis to scale individually for each subplot
    )
    axes_pack = axes_pack.flatten()  # Flatten the axes array for easy indexing

    # Set a color palette
    colors = sns.color_palette("deep", num_nvs)

    # Plot each NV in the corresponding subplot
    for nv_idx, ax in enumerate(axes_pack):
        if nv_idx < num_nvs:
            # Use Seaborn's smooth lines with matplotlib markers
            sns.lineplot(
                x=freqs,
                y=avg_counts[nv_idx],
                ax=ax,
                color=colors[nv_idx % len(colors)],
                lw=2,
                marker="o",
                markersize=5,
                label=f"NV {nv_idx+1}",
            )
            # Add error bars manually using matplotlib
            ax.errorbar(
                freqs,
                avg_counts[nv_idx],
                yerr=avg_counts_ste[nv_idx],
                fmt="none",
                ecolor="gray",
                alpha=0.6,
            )

            # Auto-scale y-axis for better view of resonance
            ax.set_ylim(
                [min(avg_counts[nv_idx]), max(avg_counts[nv_idx])+0.03]
            )

            # Only set y-tick labels for the leftmost column
            if nv_idx % num_cols == 0:
                ax.set_yticks(
                    ax.get_yticks()
                )  # Keep the default y-tick labels for the leftmost column
            else:
                ax.set_yticklabels([])
            #  Add a single y-axis label for the entire figure
            fig.text(
                0.04,
                0.5,
                "Norm. NV$^{-}$ Pop.",
                va="center",
                rotation="vertical",
                fontsize=12,
            )
            # Only label the x-axis for the bottom row
            if nv_idx >= (num_rows - 1) * num_cols:
                ax.set_xlabel("Frequency (GHz)")
            else:
                ax.set_xticklabels([])  # Remove x-axis ticks for non-bottom subplots

            # Add grid for better visualization
            ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        else:
            # Hide any unused subplots if the number of NVs is less than the grid size
            ax.axis("off")

    # Add frequency values at the bottom of each column
    for col in range(num_cols):
        bottom_row_idx = (
            num_rows * num_cols - num_cols + col
        )  # Index of the bottom row in each column

        if bottom_row_idx < len(axes_pack):  # Ensure the index is within bounds
            ax = axes_pack[bottom_row_idx]
            # Set fewer x-ticks (num_ticks) using np.linspace
            tick_positions = np.linspace(min(freqs), max(freqs), 5)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(
                [f"{tick:.2f}" for tick in tick_positions], rotation=45, fontsize=9
            )
            # Set the x-ticks to the frequencies for the bottom row of each column
            # ax.set_xticks(freqs)
            # ax.set_xticklabels([f"{f:.2f}" for f in freqs], rotation=45, fontsize=9)

    # Adjust layout to ensure nothing overlaps and reduce vertical gaps
    plt.subplots_adjust(
        left=0.1, right=0.95, top=0.95, bottom=0.1, hspace=0.01, wspace=0.01
    )

    # Save the figure to the specified file path
    plt.savefig(file_path, bbox_inches="tight")

    # Close the figure to free up memory
    plt.close(fig)


if __name__ == "__main__":
    # Load raw data
    # data = dm.get_raw_data(file_id=1652859661831, load_npz=False, use_cache=True)
    data = dm.get_raw_data(file_id=1657565965228, load_npz=False, use_cache=True)
    nv_list = data["nv_list"]
    num_nvs = len(nv_list)
    counts = np.array(data["counts"])[0]
    num_nvs = len(nv_list)
    num_steps = data["num_steps"]
    num_runs = data["num_runs"]
    num_reps = data["num_reps"]
    freqs = data["freqs"]
    adj_num_steps = num_steps // 4
    sig_counts_0 = counts[:, :, 0:adj_num_steps, :]
    sig_counts_1 = counts[:, :, adj_num_steps : 2 * adj_num_steps, :]
    sig_counts = np.append(sig_counts_0, sig_counts_1, axis=3)
    ref_counts_0 = counts[:, :, 2 * adj_num_steps : 3 * adj_num_steps, :]
    ref_counts_1 = counts[:, :, 3 * adj_num_steps :, :]
    ref_counts = np.empty((num_nvs, num_runs, adj_num_steps, 2 * num_reps))
    ref_counts[:, :, :, 0::2] = ref_counts_0
    ref_counts[:, :, :, 1::2] = ref_counts_1

    # avg_counts, avg_counts_ste, norms = widefield.process_counts(
    #     nv_list, sig_counts, ref_counts, threshold=False
    # )
    avg_counts, avg_counts_ste, norms = widefield.process_counts(
        nv_list, sig_counts, ref_counts, threshold=True, method= "otsu"
    )
    # raw_fig = plot_data(nv_list, freqs, avg_counts, avg_counts_ste)
    # fit_fig = visualize_large_nv_data(nv_list, freqs, avg_counts, avg_counts_ste, norms)

    # Save plot to a file
    # file_path = "nv_data_plot.png"
    # # plot_nv_resonance_data(nv_list, freqs, avg_counts, avg_counts_ste, file_path)
    # file_path = "nv_resonance_plot_60stepspng"
    # plot_nv_resonance_data_sns_with_freq_labels(
    #     nv_list, freqs, avg_counts, avg_counts_ste, file_path, num_cols=5
    # )

    # print(f"Plot saved to {file_path}")
    # plt.show()
    # #  Save plot to a file
    # file_path = "nv_data_plot.png"
    # plot_nv_resonance_data(nv_list, freqs, avg_counts, avg_counts_ste, file_path)
    file_path = "nv_resonance_plot_none.png"
    plot_nv_resonance_data_sns_with_freq_labels(
        nv_list, freqs, avg_counts, avg_counts_ste, file_path, num_cols=5, title = "Rabi duration 128ns"
    )

    print(f"Plot saved to {file_path}")
    plt.show()
