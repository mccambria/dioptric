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
    counts = data["counts"]

    # Process the counts using the widefield utilities
    avg_counts, avg_counts_ste, norms = widefield.process_counts(
        nv_list, counts, threshold=True
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
    plt.tight_layout()

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
    plt.tight_layout()

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


if __name__ == "__main__":
    # Load raw data
    data = dm.get_raw_data(file_id=1647377018086, load_npz=False, use_cache=True)

    # Process the data
    avg_counts, avg_counts_ste, norms, freqs, nv_list = process_data(data)

    # Plot raw data
    raw_fig = plot_data(nv_list, freqs, avg_counts, avg_counts_ste)

    # Visualize large NV datasets (in batches if necessary)
    fig_list = visualize_large_nv_data(
        nv_list, freqs, avg_counts, avg_counts_ste, norms
    )

    # Show plots
    for fig in fig_list:
        fig.show()
