# -*- coding: utf-8 -*-
"""
Created on Fall, 2024

@author: Saroj Chand
"""

import random
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from utils import kplotlib as kpl
import networkx as nx
import numpy.ma as ma
from matplotlib import patches
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as mticker

from utils import data_manager as dm
from utils import widefield as dm

# from utils.tool_belt import nan_corr_coef
from utils.widefield import threshold_counts

# region utility funtions


def nan_corr_coef(arr):
    """
    Calculate Pearson correlation coefficients for a 2D array, ignoring NaN values.

    This version masks NaN values and computes the correlation coefficient between rows.

    """
    arr = np.array(arr)
    # Mask NaN values in the array
    masked_arr = ma.masked_invalid(arr)
    # Compute the correlation coefficient using masked arrays, ignoring NaNs
    corr_coef_arr = np.ma.corrcoef(masked_arr, rowvar=True)

    # Convert masked correlations back to a standard numpy array, filling masked entries with NaN
    corr_coef_arr = corr_coef_arr.filled(np.nan)

    return corr_coef_arr


def remove_nans_from_data(sig_counts, ref_counts, nv_list):
    """Remove NVs that contain any NaN values in their signal or reference counts."""
    valid_indices = [
        i
        for i in range(len(nv_list))
        if not (np.isnan(sig_counts[i]).any() or np.isnan(ref_counts[i]).any())
    ]

    # Filter the signal and reference counts and the NV list
    sig_counts_filtered = sig_counts[valid_indices]
    ref_counts_filtered = ref_counts[valid_indices]
    nv_list_filtered = [nv_list[i] for i in valid_indices]

    return sig_counts_filtered, ref_counts_filtered, nv_list_filtered


# Combine data from multiple file IDs
def process_multiple_files(file_ids):
    """
    Load and combine data from multiple file IDs.
    """
    combined_data = dm.get_raw_data(file_stem=file_ids[0], load_npz=True)
    for file_id in file_ids[1:]:
        new_data = dm.get_raw_data(file_id=file_id)
        combined_data["num_runs"] += new_data["num_runs"]
        combined_data["counts"] = np.append(
            combined_data["counts"], new_data["counts"], axis=2
        )
    return combined_data


def rescale_extreme_values(sig_corr, sigma_threshold=2.0, method="tanh"):
    """
    Rescale extreme values in the correlation matrix based on standard deviation thresholds.

    """
    # Calculate the mean and standard deviation of the correlation matrix
    mean_corr = np.nanmean(sig_corr)
    std_corr = np.nanstd(sig_corr)

    # Identify extreme values that are beyond the threshold (mean ± sigma_threshold * std_corr)
    upper_threshold = mean_corr + sigma_threshold * std_corr
    lower_threshold = mean_corr - sigma_threshold * std_corr

    # Create a copy of the original matrix
    rescaled_corr = np.copy(sig_corr)

    # Rescale extreme values using the selected method
    if method == "tanh":
        # Rescale using the hyperbolic tangent function (smoothly compresses extreme values)
        rescaled_corr = np.tanh(rescaled_corr)
    elif method == "sigmoid":
        # Rescale using a sigmoid function (values closer to 0 are unchanged, extremes are compressed)
        rescaled_corr = 2 / (1 + np.exp(-rescaled_corr)) - 1  # Maps values to [-1, 1]

    # Mask values that are beyond the upper and lower thresholds
    rescaled_corr[rescaled_corr > upper_threshold] = upper_threshold
    rescaled_corr[rescaled_corr < lower_threshold] = lower_threshold

    return rescaled_corr


def plot_correlation_histogram(corr_matrix, bins=50, title="Histogram of Corr. Coeff."):
    """
    Plot a histogram of the correlation coefficients from the correlation matrix,
    with separated bars.
    """
    # Remove the diagonal (which contains 1s for self-correlation) and flatten the matrix
    flattened_corr = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]

    # Calculate histogram data
    counts, bin_edges = np.histogram(flattened_corr, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Midpoints of bins
    bar_width = (bin_edges[1] - bin_edges[0]) * 0.6  # Reduce width for separation

    # Plot histogram with separated bars
    plt.figure(figsize=(8, 6))
    plt.bar(bin_centers, counts, width=bar_width, color="c", edgecolor="k", alpha=0.7)

    # Add labels and title
    plt.xlabel("Correlation Coefficient", fontsize=18)
    plt.ylabel("Frequency", fontsize=18)
    plt.title(title, fontsize=18)

    # Display the plot
    plt.tight_layout()
    plt.grid(True)
    plt.show()


# end region utility funtions


# region heatmaps
def process_and_plot(data, rearrangement="spin_flip", file_path=None):
    """
    Process and plot NV center correlation matrices with creative spin arrangements.

    Parameters:
    data (dict): Dictionary containing 'nv_list' and 'counts'.
    rearrangement (str): Method for rearranging NV centers ('spin_flip', 'checkerboard', 'block', 'spiral', etc.).
    """
    # Seaborn aesthetics
    sns.set(style="whitegrid", context="talk", font_scale=1.2)

    # Unpack data
    nv_list = data["nv_list"]
    counts = np.array(data["counts"])
    # indices_to_remove = [64, 108, 75, 100, 103, 55, 77, 89, 17, 82]
    # indices_to_remove = [64, 108, 75, 100, 103, 55, 77, 89, 17, 82, 38]
    print(counts.shape)
    # nv_list = [
    #     nv_list[ind] for ind in range(len(nv_list)) if ind not in indices_to_remove
    # ]
    # # Use NumPy boolean masking to filter counts
    # mask = np.ones(counts.shape[1], dtype=bool)  # Create a mask for NV indices
    # mask[list(indices_to_remove)] = False  # Set unwanted indices to False
    # counts = counts[:, mask, :, :, :]  # Apply mask along the NV axis (second dimension)
    # print(counts.shape)

    # return
    if len(nv_list) == 0 or counts.size == 0:
        print("Error: Data does not contain NV list or counts.")
        return None

    # Separate signal and reference counts
    sig_counts = np.array(counts[0])
    ref_counts = np.array(counts[1])

    # Remove NVs with NaN values
    num_nvs = len(nv_list)

    # Thresholding counts with dynamic thresholds
    sig_counts, ref_counts = threshold_counts(
        nv_list, sig_counts, ref_counts, dynamic_thresh=True
    )

    # Flatten counts for each NV
    flattened_sig_counts = [sig_counts[ind].flatten() for ind in range(num_nvs)]
    flattened_ref_counts = [ref_counts[ind].flatten() for ind in range(num_nvs)]

    # Calculate correlations
    sig_corr_coeffs = nan_corr_coef(flattened_sig_counts)
    ref_corr_coeffs = nan_corr_coef(flattened_ref_counts)
    diff_corr_coeffs = sig_corr_coeffs - ref_corr_coeffs
    
    # bins = int(np.ceil(np.log2(len(sig_corr_coeffs)) + 1))
    # bin_width = 3.5 * np.nanstd(sig_corr_coeffs) / len(sig_corr_coeffs) ** (1 / 3)
    # bins = int(
    #     np.ceil((np.nanmax(sig_corr_coeffs) - np.nanmin(sig_corr_coeffs)) / bin_width)
    # )

    q75, q25 = np.percentile(sig_corr_coeffs, [75, 25])
    iqr = q75 - q25
    bin_width = 2 * iqr / len(sig_corr_coeffs) ** (1 / 3)
    bins = int(
        np.ceil((np.nanmax(sig_corr_coeffs) - np.nanmin(sig_corr_coeffs)) / bin_width)
    )
    # square root rule
    # bins = int(np.ceil(np.sqrt(len(sig_corr_coeffs))))
    plot_correlation_histogram(sig_corr_coeffs, bins=150, title="Signal Correlation")
    plot_correlation_histogram(ref_corr_coeffs, bins=150, title="Reference Correlation")
    plot_correlation_histogram(
        diff_corr_coeffs, bins=150, title="Difference Correlation"
    )
    # Apply the same rearrangement to signal, reference, and ideal matrices
    if rearrangement == "spin_flip":
        nv_list, sig_corr_coeffs, ref_corr_coeffs = rearrange_spin_flip(
            nv_list,
            sig_corr_coeffs,
            ref_corr_coeffs,
        )
    elif rearrangement == "checkerboard":
        nv_list, sig_corr_coeffs, ref_corr_coeffs, diff_corr_coeffs = (
            rearrange_checkerboard(
                nv_list, sig_corr_coeffs, ref_corr_coeffs, diff_corr_coeffs
            )
        )
    elif rearrangement == "block":
        nv_list, sig_corr_coeffs, ref_corr_coeffs, diff_corr_coeffs = rearrange_block(
            nv_list, sig_corr_coeffs, ref_corr_coeffs, diff_corr_coeffs
        )
    elif rearrangement == "spiral":
        nv_list, sig_corr_coeffs, ref_corr_coeffs = rearrange_spiral(
            nv_list, sig_corr_coeffs, ref_corr_coeffs
        )
    elif rearrangement == "random":
        nv_list, sig_corr_coeffs, ref_corr_coeffs = rearrange_random(
            nv_list, sig_corr_coeffs, ref_corr_coeffs
        )
    elif rearrangement == "alternate_quadrants":
        nv_list, sig_corr_coeffs, ref_corr_coeffs, diff_corr_coeffs = (
            rearrange_alternate_quadrants(
                nv_list,
                sig_corr_coeffs,
                ref_corr_coeffs,
                diff_corr_coeffs,
                num_quadrants=20,
            )
        )
    elif rearrangement == "concentric_circles":
        nv_list, sig_corr_coeffs, ref_corr_coeffs = rearrange_concentric_circles(
            nv_list, sig_corr_coeffs, ref_corr_coeffs
        )
    elif rearrangement == "letter":
        nv_list, sig_corr_coeffs, ref_corr_coeffs, diff_corr_coeffs = (
            rearrange_to_letter(
                nv_list,
                sig_corr_coeffs,
                ref_corr_coeffs,
                diff_corr_coeffs,
                letter="X",
                grid_size=(15, 15),
            )
        )
    elif rearrangement == "radial":
        nv_list, sig_corr_coeffs, ref_corr_coeffs = rearrange_radial(
            nv_list, sig_corr_coeffs, ref_corr_coeffs
        )
    else:
        pass

    # Generate ideal correlation matrix based on spin flips before rearrangement
    spin_flips = np.array([-1 if nv.spin_flip else +1 for nv in nv_list])
    ideal_sig_corr_coeffs = np.outer(spin_flips, spin_flips).astype(float)
    # ideal_ref_corr_coeffs = np.zeros((num_nvs, num_nvs), dtype=float)
    # Calculate min and max correlation values for the actual signal and reference correlations
    # positive_corrs = sig_corr_coeffs[sig_corr_coeffs > 0]
    # negative_corrs = sig_corr_coeffs[sig_corr_coeffs < 0]
    # print(len(positive_corrs))
    # print(len(negative_corrs))
    # Set vmin and vmax separately for signal/reference correlations
    mean_corr = np.nanmean(sig_corr_coeffs)
    std_corr = np.nanstd(sig_corr_coeffs)
    # print(mean_corr, median_corr, std_corr)
    sig_vmax = mean_corr + 1.5 * std_corr
    # sig_vmax = 0.01
    # sig_vmax = np.nanmin(sig_corr_coeffs)
    sig_vmin = -sig_vmax
    sig_vmax = np.percentile(sig_corr_coeffs, 98)
    sig_vmin = np.percentile(sig_corr_coeffs, 0)
    sig_vmin = -sig_vmax
    # For reference correlations

    # For reference correlations
    diff_vmax = np.nanmean(diff_corr_coeffs) + 1.5 * np.nanstd(diff_corr_coeffs)
    diff_vmin = -diff_vmax
    # For the ideal correlation matrix
    ideal_vmin = -1
    ideal_vmax = 1

    # Plotting setup
    # titles = ["Ideal Signal", "Signal", "Reference", "Difference"]
    titles = ["Signal", "Reference"]
    # titles = ["Ideal Signal", "Signal After Reference Subtraction"]
    # vals = [ideal_sig_corr_coeffs, sig_corr_coeffs, ref_corr_coeffs, diff_corr_coeffs]
    vals = [sig_corr_coeffs, ref_corr_coeffs]
    # vals = [ideal_sig_corr_coeffs, sig_corr_coeffs]
    n = len(vals)
    fig, axes_pack = plt.subplots(1, n, figsize=(6*n, 6))
    axes_pack = np.atleast_1d(axes_pack).ravel()
    for ind, (val, title) in enumerate(zip(vals, titles)):
        mat = np.array(val, dtype=float, copy=True)

        # --- 1) Guard against all-NaN after masking the diagonal ---
        side = mat.shape[0]
        # Only mask the diagonal if there are finite off-diagonal entries
        mat_no_diag = mat.copy()
        np.fill_diagonal(mat_no_diag, np.nan)
        if np.isfinite(mat_no_diag).any():
            mat = mat_no_diag
        else:
            # If everything would become NaN, keep the original (don’t mask)
            pass

        ax = axes_pack[ind]

        # --- your vmin/vmax selection unchanged ---
        if title == "Ideal Signal":
            vmin, vmax = ideal_vmin, ideal_vmax
        elif title in ("Signal", "Reference"):
            vmin, vmax = sig_vmin, sig_vmax
        else:
            vmin, vmax = diff_vmin, diff_vmax

        # --- normalization unchanged ---
        if vmin < 0 < vmax:
            norm = plt.matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        else:
            norm = plt.matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

        # --- colormap unchanged ---
        cmap = plt.get_cmap("coolwarm").copy()
        cmap.set_bad(color="#f0f0f0")

        heatmap = sns.heatmap(
            mat,
            ax=ax,
            cmap=cmap,
            norm=norm,
            cbar=True,
            square=True,                # keeps cells square
            mask=np.isnan(mat),
            annot=False,
            cbar_kws={"pad": 0.00, "shrink": 0.7, "aspect": 20},
            rasterized=True,
            linewidths=0.0,
        )

        # --- 2) Force axes to the full data extent (prevents corner shrinking) ---
        ax.set_xlim(0, side)
        ax.set_ylim(side, 0)  # heatmap uses inverted y

        # --- colorbar formatting (unchanged) ---
        cbar = heatmap.collections[0].colorbar
        cbar.set_label("Correlation coefficient", fontsize=16)
        cbar.ax.tick_params(labelsize=13)
        cbar.formatter = ScalarFormatter(useMathText=True)
        cbar.formatter.set_scientific(True)
        cbar.formatter.set_powerlimits((-2, 2))
        cbar.update_ticks()

        # --- 3) Ticks: compute from actual matrix size ---
        max_ticks = 6
        step = max(1, side // (max_ticks - 1))
        ticks = np.arange(0, side, step)

        ax.set_xticks(ticks + 0.5)
        ax.set_yticks(ticks + 0.5)
        ax.set_xticklabels(ticks, rotation=0, fontsize=12)
        ax.set_yticklabels(ticks, rotation=0, fontsize=12)
        ax.tick_params(axis="x", which="both", pad=0)
        ax.tick_params(axis="y", which="both", pad=0)

        ax.set_title(title, fontsize=16, pad=8)
        ax.set_xlabel("NV index", fontsize=14)
        ax.set_ylabel("NV index", fontsize=14, labelpad=0)


# end region heatmaps


# region Rearrangement Functions Based on Spin Flip
def rearrange_spin_flip(nv_list, sig_corr, ref_corr):
    """Rearrange spins based on their flip status: +1 (spin up) followed by -1 (spin down)."""
    spin_plus_indices = [i for i, nv in enumerate(nv_list) if not nv.spin_flip]
    spin_minus_indices = [i for i, nv in enumerate(nv_list) if nv.spin_flip]
    reshuffled_indices = spin_plus_indices + spin_minus_indices
    return apply_rearrangement(nv_list, sig_corr, ref_corr, reshuffled_indices)


def rearrange_checkerboard(nv_list, sig_corr, ref_corr, diff_diff):
    """Checkerboard pattern where alternating spins are up (+1) and down (-1)."""
    spin_plus_indices = [i for i, nv in enumerate(nv_list) if not nv.spin_flip]
    spin_minus_indices = [i for i, nv in enumerate(nv_list) if nv.spin_flip]
    reshuffled_indices = []
    for i in range(len(nv_list)):
        if i % 2 == 0:
            reshuffled_indices.append(
                spin_plus_indices.pop(0)
                if spin_plus_indices
                else spin_minus_indices.pop(0)
            )
        else:
            reshuffled_indices.append(
                spin_minus_indices.pop(0)
                if spin_minus_indices
                else spin_plus_indices.pop(0)
            )
    return apply_rearrangement(
        nv_list, sig_corr, ref_corr, diff_diff, reshuffled_indices
    )


def rearrange_block(nv_list, sig_corr, ref_corr, diff_corr):
    """Block arrangement: first half spin up (+1), second half spin down (-1)."""
    half = len(nv_list) // 2
    spin_plus_indices = [i for i, nv in enumerate(nv_list) if not nv.spin_flip]
    spin_minus_indices = [i for i, nv in enumerate(nv_list) if nv.spin_flip]
    reshuffled_indices = spin_plus_indices[:half] + spin_minus_indices[:half]
    return apply_rearrangement(
        nv_list, sig_corr, ref_corr, diff_corr, reshuffled_indices
    )


def rearrange_spiral(nv_list, sig_corr, ref_corr):
    """Spiral arrangement based on spin flip status. Spins in a spiral-like pattern."""
    spin_plus_indices = [i for i, nv in enumerate(nv_list) if not nv.spin_flip]
    spin_minus_indices = [i for i, nv in enumerate(nv_list) if nv.spin_flip]
    reshuffled_indices = np.argsort([np.sin(i) for i in range(len(nv_list))])
    spin_up_first = [i for i in reshuffled_indices if spin_plus_indices] + [
        i for i in reshuffled_indices if spin_minus_indices
    ]
    return apply_rearrangement(nv_list, sig_corr, ref_corr, spin_up_first)


def rearrange_random(nv_list, sig_corr, ref_corr):
    """Random arrangement of spin-up and spin-down NV centers."""
    np.random.seed(42)
    reshuffled_indices = np.random.permutation(len(nv_list))
    return apply_rearrangement(nv_list, sig_corr, ref_corr, reshuffled_indices)


def rearrange_alternate_quadrants(
    nv_list, sig_corr, ref_corr, diff_corr_coeffs, num_quadrants=8
):
    """
    Divide NV centers into multiple quadrants and alternate spin-up and spin-down in each quadrant.

    """
    num_nvs = len(nv_list)
    spin_plus_indices = [i for i, nv in enumerate(nv_list) if not nv.spin_flip]
    spin_minus_indices = [i for i, nv in enumerate(nv_list) if nv.spin_flip]
    reshuffled_indices = []

    # Determine the size of each quadrant
    quadrant_size = num_nvs // num_quadrants

    # Alternate NVs in each quadrant
    for i in range(num_nvs):
        quadrant_index = i // quadrant_size
        if quadrant_index % 2 == 0:  # Even quadrants: start with spin-up
            reshuffled_indices.append(
                spin_plus_indices.pop(0)
                if spin_plus_indices
                else spin_minus_indices.pop(0)
            )
        else:  # Odd quadrants: start with spin-down
            reshuffled_indices.append(
                spin_minus_indices.pop(0)
                if spin_minus_indices
                else spin_plus_indices.pop(0)
            )

    # Handle remaining NVs (if any)
    reshuffled_indices.extend(spin_plus_indices)
    reshuffled_indices.extend(spin_minus_indices)

    return apply_rearrangement(
        nv_list, sig_corr, ref_corr, diff_corr_coeffs, reshuffled_indices
    )


def rearrange_concentric_circles(nv_list, sig_corr, ref_corr):
    """Concentric circle pattern where inner circles are spin-up, outer circles are spin-down."""
    spin_plus_indices = [i for i, nv in enumerate(nv_list) if not nv.spin_flip]
    spin_minus_indices = [i for i, nv in enumerate(nv_list) if nv.spin_flip]
    reshuffled_indices = []

    # Divide into concentric circles (spiral-like arrangement)
    num_rings = 4  # Define how many concentric circles
    for i in range(num_rings):
        if i % 2 == 0:  # Even rings are spin-up
            reshuffled_indices.extend(
                spin_plus_indices[: len(spin_plus_indices) // num_rings]
            )
            spin_plus_indices = spin_plus_indices[len(spin_plus_indices) // num_rings :]
        else:  # Odd rings are spin-down
            reshuffled_indices.extend(
                spin_minus_indices[: len(spin_minus_indices) // num_rings]
            )
            spin_minus_indices = spin_minus_indices[
                len(spin_minus_indices) // num_rings :
            ]

    return apply_rearrangement(nv_list, sig_corr, ref_corr, reshuffled_indices)


def rearrange_radial(nv_list, sig_corr, ref_corr):
    """
    Arrange NV centers in a radial pattern with alternating spin-up and spin-down per ring.
    """
    spin_plus_indices = [i for i, nv in enumerate(nv_list) if not nv.spin_flip]
    spin_minus_indices = [i for i, nv in enumerate(nv_list) if nv.spin_flip]

    reshuffled_indices = []
    ring_size = 3
    toggle = True  # Start with spin-up
    while spin_plus_indices or spin_minus_indices:
        current_ring = []
        for _ in range(ring_size):
            if toggle and spin_plus_indices:
                current_ring.append(spin_plus_indices.pop(0))
            elif not toggle and spin_minus_indices:
                current_ring.append(spin_minus_indices.pop(0))
        reshuffled_indices.extend(current_ring)
        ring_size += 1
        toggle = not toggle  # Alternate per ring

    return apply_rearrangement(nv_list, sig_corr, ref_corr, reshuffled_indices)


from PIL import Image, ImageDraw, ImageFont


def rearrange_to_letter(
    nv_list, sig_corr, ref_corr, diff_corr_coeffs, letter="A", grid_size=(10, 10)
):
    """
    Arrange NV centers in the shape of a specified letter.
    NVs are assigned to pixels forming the letter in the grid.
    """
    grid_rows, grid_cols = grid_size
    canvas = Image.new(
        "1", (grid_cols * 10, grid_rows * 10), 0
    )  # Larger resolution for clear letter drawing
    draw = ImageDraw.Draw(canvas)

    # Load a font and draw the letter onto the canvas
    try:
        font = ImageFont.truetype("arial.ttf", size=grid_cols * 8)
    except IOError:
        font = ImageFont.load_default()

    text_position = (grid_cols * 2, grid_rows)  # Center the letter
    draw.text(text_position, letter, fill=1, font=font)

    # Convert the canvas to a binary grid (1 for letter pixels, 0 for background)
    letter_grid = np.array(
        canvas.resize(grid_size, Image.Resampling.LANCZOS)
    )  # Downscale to match grid
    letter_indices = np.argwhere(letter_grid > 0)  # Identify pixels forming the letter

    num_nvs = len(nv_list)
    if len(letter_indices) > num_nvs:
        letter_indices = letter_indices[:num_nvs]  # Truncate if too many pixels
    elif len(letter_indices) < num_nvs:
        print("Warning: Not enough letter pixels; some NVs may not be assigned.")

    # Normalize indices to fit NV positions
    min_vals = letter_indices.min(axis=0)
    max_vals = letter_indices.max(axis=0)
    letter_positions = (letter_indices - min_vals) / (
        max_vals - min_vals
    )  # Normalize to [0,1]
    letter_positions *= np.max([grid_rows, grid_cols])  # Scale back

    # Get NV correlation strengths
    corr_strengths = np.diag(sig_corr)
    sorted_indices = np.argsort(
        corr_strengths
    )  # Sort NVs based on correlation strength

    # Assign NVs to letter positions
    assigned_nvs = np.zeros((num_nvs, 2))
    for i, idx in enumerate(letter_positions):
        assigned_nvs[i] = idx  # Assign sorted NVs to letter pixels

    # Assign remaining NVs outside the letter
    remaining_indices = sorted_indices[len(letter_positions) :]
    for i, idx in enumerate(remaining_indices):
        assigned_nvs[i + len(letter_positions)] = np.random.uniform(
            0, grid_rows, size=2
        )

    # Plot NV positions
    fig, ax = plt.subplots(figsize=(6, 6))
    spin_up = assigned_nvs[: num_nvs // 2]
    spin_down = assigned_nvs[num_nvs // 2 :]

    ax.scatter(spin_up[:, 1], spin_up[:, 0], color="blue", label="Spin-Up", alpha=0.8)
    ax.scatter(
        spin_down[:, 1], spin_down[:, 0], color="red", label="Spin-Down", alpha=0.8
    )

    ax.set_xlim(0, grid_cols)
    ax.set_ylim(grid_rows, 0)  # Flip y-axis for proper visualization
    ax.set_title(f"NVs Arranged in Letter '{letter}'", fontsize=14)
    ax.legend()
    plt.show()

    return apply_rearrangement(
        nv_list, sig_corr, ref_corr, diff_corr_coeffs, sorted_indices
    )


# def rearrange_to_letter(
#     nv_list, sig_corr, ref_corr, diff_corr_coeffs, letter="A", grid_size=(10, 10)
# ):
#     """
#     Arrange NV centers in the shape of a specified letter.

#     """
#     from PIL import Image, ImageDraw, ImageFont

#     # Create a blank canvas for the letter
#     grid_rows, grid_cols = grid_size
#     canvas = Image.new("1", (grid_cols, grid_rows), 0)
#     draw = ImageDraw.Draw(canvas)

#     # Load a font and draw the letter onto the canvas
#     try:
#         font = ImageFont.truetype("arial.ttf", size=min(grid_rows, grid_cols))
#     except IOError:
#         font = ImageFont.load_default()  # Fallback font if `arial.ttf` isn't available

#     text_position = (grid_cols // 4, grid_rows // 4)  # Center the letter
#     draw.text(text_position, letter, fill=1, font=font)

#     # Convert the canvas to a binary grid
#     letter_grid = np.array(canvas)

#     # Identify "on" pixels in the grid
#     letter_indices = np.argwhere(letter_grid > 0)

#     # Normalize letter indices to NV list size
#     letter_indices = letter_indices[: len(nv_list)]

#     # Assign NVs to the letter shape
#     reshuffled_indices = []
#     for index in letter_indices:
#         reshuffled_indices.append(index[0])  # Assign NVs in the order of the grid

#     # Add any remaining NVs (not part of the letter) to the reshuffled list
#     remaining_indices = list(set(range(len(nv_list))) - set(reshuffled_indices))
#     reshuffled_indices.extend(remaining_indices)

#     return apply_rearrangement(
#         nv_list, sig_corr, ref_corr, diff_corr_coeffs, reshuffled_indices
#     )


def apply_rearrangement(nv_list, sig_corr, ref_corr, diff_corr, reshuffled_indices):
    nv_list_reshuffled = [nv_list[i] for i in reshuffled_indices]
    sig_corr_reshuffled = sig_corr[np.ix_(reshuffled_indices, reshuffled_indices)]
    ref_corr_reshuffled = ref_corr[np.ix_(reshuffled_indices, reshuffled_indices)]
    diff_corr_reshuffled = diff_corr[np.ix_(reshuffled_indices, reshuffled_indices)]
    return (
        nv_list_reshuffled,
        sig_corr_reshuffled,
        ref_corr_reshuffled,
        diff_corr_reshuffled,
    )


# end region


# region Network graph
def draw_curved_edges(
    G, pos, ax, norm, edges, edge_colors, edge_widths, edge_alphas, curvature=0.2
):
    """
    Draw curved edges for a graph.

    Parameters:
    - G: The graph.
    - pos: Dictionary of node positions.
    - ax: The matplotlib axis to draw on.
    - edges: List of edges.
    - edge_colors: List of edge colors.
    - edge_widths: List of edge widths.
    - edge_alphas: List of edge alpha (transparency).
    - curvature: The curvature of the edges.
    """
    for i, (u, v) in enumerate(edges):
        # Get the positions of the nodes
        x1, y1 = pos[u]
        x2, y2 = pos[v]

        # Calculate the mid-point for curvature
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2

        # Add a small offset to create a curve
        offset_x = curvature * (y2 - y1)
        offset_y = curvature * (x1 - x2)

        # Create a curved edge using a Bezier curve
        curve = patches.FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            connectionstyle=f"arc3,rad={curvature}",
            color=cm.coolwarm(norm(edge_colors[i])),
            lw=edge_widths[i],
            alpha=edge_alphas[i],
            arrowstyle="-",
        )

        # Add the edge to the axis
        ax.add_patch(curve)


def plot_nv_network(data):
    """
    Plot a network graph where NV centers are represented as nodes,
    with spin-up nodes in red and spin-down nodes in blue. Curved edges between nodes
    are colored based on the correlation coefficients, with varying edge widths and transparencies.

    Parameters:
    - data: Dictionary containing 'nv_list' and 'counts'.
    """
    # Seaborn aesthetics
    sns.set(style="whitegrid", context="talk", font_scale=1.2)

    # Unpack data
    nv_list = data.get("nv_list", [])
    counts = np.array(data.get("counts", []))

    # # indices_to_remove = [18, 35, 56]
    # indices_to_remove = [64, 108, 75, 100, 103, 55, 77, 89, 17, 82, 38]
    # print(counts.shape)
    # nv_list = [
    #     nv_list[ind] for ind in range(len(nv_list)) if ind not in indices_to_remove
    # ]
    # Use NumPy boolean masking to filter counts
    # mask = np.ones(counts.shape[1], dtype=bool)  # Create a mask for NV indices
    # mask[list(indices_to_remove)] = False  # Set unwanted indices to False
    # counts = counts[:, mask, :, :, :]  # Apply mask along the NV axis (second dimension)
    # print(counts.shape)
    # # high spurious correlation
    # more_indices_to_remove = [16, 20, 28, 47, 50, 56]
    # more_indices_to_remove = [6, 13, 34, 47, 42, 49, 58, 71, 69, 75, 83, 93, 103, 102]
    # nv_list = [
    #     nv_list[ind] for ind in range(len(nv_list)) if ind not in more_indices_to_remove
    # ]
    # # Use NumPy boolean masking to filter counts
    # mask = np.ones(counts.shape[1], dtype=bool)  # Create a mask for NV indices
    # mask[list(more_indices_to_remove)] = False  # Set unwanted indices to False
    # counts = counts[:, mask, :, :, :]  # Apply mask along the NV axis (second dimension)
    # print(counts.shape)

    if len(nv_list) == 0 or counts.size == 0:
        print("Error: Data does not contain NV list or counts.")
        return None

    # Separate signal and reference counts
    sig_counts = np.array(counts[0])
    ref_counts = np.array(counts[1])

    num_nvs = len(nv_list)

    # Thresholding counts with dynamic thresholds
    sig_counts, ref_counts = threshold_counts(
        nv_list, sig_counts, ref_counts, dynamic_thresh=True
    )

    # Flatten counts for each NV
    flattened_sig_counts = [sig_counts[ind].flatten() for ind in range(num_nvs)]
    flattened_ref_counts = [ref_counts[ind].flatten() for ind in range(num_nvs)]

    # Calculate correlations
    sig_corr_coeffs = nan_corr_coef(flattened_sig_counts)
    ref_corr_coeffs = nan_corr_coef(flattened_ref_counts)
    # difference
    # sig_corr_coeffs = sig_corr_coeffs - ref_corr_coeffs
    sig_corr_coeffs = ref_corr_coeffs
    # print("sig_corr_coeffs shape:", sig_corr_coeffs.shape)

    if not np.isfinite(sig_corr_coeffs).all():
        print("Invalid values found in sig_corr_coeffs.")
        return
    # Identify NV pairs with correlation > 0.02
    from collections import Counter

    spurious_pairs = []
    nv_occurrences = Counter()
    spurious_threshold = 0.01
    for i in range(num_nvs):
        for j in range(i + 1, num_nvs):  # Upper triangle only
            if abs(sig_corr_coeffs[i, j]) > spurious_threshold:
                spurious_pairs.append((i, j))
                nv_occurrences[i] += 1
                nv_occurrences[j] += 1

    # Print NV pairs with high correlation
    print(f"\nSpurious Correlation NV Pairs (Corr > {spurious_threshold}):")
    for pair in spurious_pairs:
        print(f"NV {pair[0]} - NV {pair[1]}")

    # Print all NV occurrences sorted by frequency
    # print("\nNV Occurrences in Spurious Correlations:")
    sorted_occurrences = sorted(
        nv_occurrences.items(), key=lambda x: x[1], reverse=True
    )
    print("nv_occurrences:", nv_occurrences)
    for nv, count in sorted_occurrences:
        print(f"NV {nv}: {count} times spurious corr > {spurious_threshold}.")
    # Extract NV positions
    nv_positions = np.array([nv.coords["pixel"][:2] for nv in nv_list])
    # Compute pairwise distances
    from scipy.spatial.distance import pdist, squareform

    distances = squareform(pdist(nv_positions))  # Creates a symmetric distance matrix

    # Flatten correlation and distance matrices (excluding diagonal)
    corr_values = sig_corr_coeffs[
        np.triu_indices(len(nv_list), k=1)
    ]  # Extract upper triangle (no duplicates)
    distance_values = distances[
        np.triu_indices(len(nv_list), k=1)
    ]  # Same for distances
    # distance_values *= 0.072
    # Scatter plot: Correlation Coefficient vs. Distance
    plt.figure(figsize=(6, 5))
    plt.scatter(
        distance_values,
        corr_values,
        marker="o",
        s=20,
        color="blue",
        edgecolors=None,
        alpha=0.6,
        label="Reference Corr.",
    )
    plt.xlabel("Pair-Wise Distance between NV centers (pixels)", fontsize=15)
    plt.ylabel("Correlation Coefficient", fontsize=15)
    plt.title(f"Pair-Wise Coefficient vs Distance ({num_nvs}NVs)", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # plt.xlim([0, 60])
    # plt.grid(True)
    plt.legend(fontsize=12)
    plt.show()
    # return
    # Initialize a graph
    G = nx.Graph()
    for i, nv in enumerate(nv_list):
        pixel = nv.coords.get("pixel", [])
        # nv.coords["pixel"] = [float(coord) for coord in pixel[:2]]
        nv.coords["pixel"] = [float(pixel[0]), -float(pixel[1])]
        # Add nodes
        G.add_node(i, pos=nv.coords["pixel"], spin_flip=nv.spin_flip)

    pos = nx.get_node_attributes(G, "pos")
    spin_up_nodes = [i for i, nv in enumerate(nv_list) if not nv.spin_flip]
    spin_down_nodes = [i for i, nv in enumerate(nv_list) if nv.spin_flip]

    # Create the plot
    fig, ax = plt.subplots(figsize=(6, 5))
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=spin_up_nodes,
        node_color="red",
        node_size=20,
        label="Group A",
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=spin_down_nodes,
        node_color="blue",
        node_size=20,
        label="Group B",
    )
    # Add edges based on correlation coefficientss
    edges = []
    edge_colors = []
    edge_widths = []
    edge_alphas = []

    for i in range(sig_corr_coeffs.shape[0]):
        for j in range(i + 1, sig_corr_coeffs.shape[1]):
            try:
                edges.append((i, j))
                edge_colors.append(float(sig_corr_coeffs[i, j]))
                # edge_widths.append(20 * abs(sig_corr_coeffs[i, j]))
                edge_widths.append(0.6)
                # edge_alphas.append(0.5 + 0.5 * abs(sig_corr_coeffs[i, j]))
                edge_alphas.append(0.6)
            except Exception as e:
                print(f"Error at edge ({i}, {j}): {e}")
                print(f"sig_corr_coeffs[{i}, {j}] = {sig_corr_coeffs[i, j]}")

    mean_corr = np.nanmean(sig_corr_coeffs)
    std_corr = np.nanstd(sig_corr_coeffs)
    # vmax = mean_corr + 1.5 * std_corr
    vmax = np.percentile(sig_corr_coeffs, 98)
    # vmax = 0.021379929550422862
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)

    # Draw curved edges with color based on the correlation coefficients
    draw_curved_edges(
        G, pos, ax, norm, edges, edge_colors, edge_widths, edge_alphas, curvature=0.0
    )
    ax.set_aspect("equal", adjustable="datalim")
    # Create ScalarMappable for the color bar
    # Create ScalarMappable with manual color bar limits
    sm = plt.cm.ScalarMappable(cmap=cm.coolwarm, norm=norm)
    sm.set_array([])
    # sm.set_array(edge_colors)
    cbar = plt.colorbar(sm, ax=ax, aspect=30, pad=0.02)
    cbar.set_label("Correlation Coefficient", fontsize=15)

    # cbar.ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    # cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1e"))
    # Format the ticks normally but remove the default scientific notation
    # cbar.ax.yaxis.set_major_formatter(
    #     mticker.FuncFormatter(lambda x: f"{x / 1e-2:.0f}")
    # )
    def scale_ticks(x, pos):
        scaled = x / 1e-2
        return f"{int(scaled)}" if scaled.is_integer() else f"{scaled:.1f}"

    cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(scale_ticks))
    # cbar.ax.set_yticklabels([f"{tick:.0f}" for tick in xticks])  # Show as -1 to 1
    cbar.ax.text(
        0.0,
        1.01,
        r"$\mathregular{10^{-2}}$",
        transform=cbar.ax.transAxes,
        fontsize=15,
        ha="left",
    )

    # cbar.ax.yaxis.get_offset_text().set_fontsize(15)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=15)
    # Add scale bar
    scalebar_length_pixels = 20
    scalebar_length_um = scalebar_length_pixels * 0.072
    scalebar_x_start = 20
    scalebar_y_start = 10

    # Draw scale bar
    ax.plot(
        [scalebar_x_start, scalebar_x_start + scalebar_length_pixels],
        [scalebar_y_start, scalebar_y_start],
        color="black",
        linewidth=2,
    )

    # Add label to scale bar
    ax.text(
        scalebar_x_start + scalebar_length_pixels / 2,
        scalebar_y_start + 2,
        f"{scalebar_length_um:.2f} µm",
        ha="center",
        va="bottom",
        fontsize=12,
        color="black",
    )

    # Add labels (optional)
    labels = {i: f"{i}" for i in range(len(nv_list))}
    nx.draw_networkx_labels(G, pos, labels, font_size=5, font_color="white", ax=ax)

    # Set title and legend
    plt.title("Diff. Corr. Network Graph", fontsize=16)
    plt.legend(scatterpoints=1, loc="upper right", fontsize=12)
    # plt.tight_layout()
    # plt.show()
    # Adjust subplots for proper spacing
    # fig.subplots_adjust(left=0.05, right=0.88, bottom=0.05, top=0.95, wspace=0.3)
    # if fig is not None:
    #     dm.save_figure(fig, file_path)
    kpl.show(block=True)


# Combine and process data iteratively
def process_files_incrementally(file_ids):
    """
    Load and process data incrementally from multiple file IDs.
    Perform analysis after each addition.
    """
    combined_data = dm.get_raw_data(file_id=file_ids[0])
    analyses = []  # Store analysis results

    # Perform analysis on the first file
    analysis = analyze_data(combined_data)
    analyses.append(analysis)

    # Append and analyze subsequent files
    for i, file_id in enumerate(file_ids[1:], start=1):
        new_data = dm.get_raw_data(file_id=file_id)
        combined_data["num_runs"] += new_data["num_runs"]
        combined_data["counts"] = np.append(
            combined_data["counts"], new_data["counts"], axis=2
        )

        # Perform analysis
        analysis = analyze_data(combined_data)
        analyses.append(analysis)

    return analyses


def analyze_data(data):
    """
    Perform analysis on the given data.
    """
    # Extract data
    nv_list = data.get("nv_list", [])
    counts = np.array(data.get("counts", []))

    if len(nv_list) == 0 or counts.size == 0:
        raise ValueError("Data does not contain NV list or counts.")

    # Separate signal and reference counts
    sig_counts = np.array(counts[0])
    ref_counts = np.array(counts[1])

    # Thresholding counts with dynamic thresholds
    sig_counts, ref_counts = threshold_counts(
        nv_list, sig_counts, ref_counts, dynamic_thresh=True
    )

    # Flatten counts for each NV
    flattened_sig_counts = [sig_counts[ind].flatten() for ind in range(len(nv_list))]
    flattened_ref_counts = [ref_counts[ind].flatten() for ind in range(len(nv_list))]

    # Calculate correlations
    sig_corr_coeffs = nan_corr_coef(flattened_sig_counts)
    ref_corr_coeffs = nan_corr_coef(flattened_ref_counts)
    sig_corr_coeffs = sig_corr_coeffs - ref_corr_coeffs

    # Fit bimodal distribution
    means, stds, weights, gmm = fit_bimodal_distribution(sig_corr_coeffs)

    # Plot histogram with bimodal fit
    # plot_histogram_and_fit(
    #     sig_corr_coeffs, bins=50, means=means, stds=stds, weights=weights
    # )

    # Return fitted parameters for analysis
    return {
        "means": means,
        "stds": stds,
        "weights": weights,
        "corr_coeffs": sig_corr_coeffs,
    }


def fit_bimodal_distribution(corr_coeffs):
    """
    Fit a bimodal distribution to the correlation coefficients.
    """
    corr_coeffs = corr_coeffs[np.isfinite(corr_coeffs)]  # Remove NaNs
    corr_coeffs = corr_coeffs.reshape(-1, 1)  # Reshape for GaussianMixture

    # Fit a Gaussian Mixture Model
    gmm = GaussianMixture(n_components=2, random_state=0)
    gmm.fit(corr_coeffs)

    # Extract parameters
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_).flatten()
    weights = gmm.weights_.flatten()

    return means, stds, weights, gmm


def plot_histogram_and_fit(corr_coeffs, bins=150, means=None, stds=None, weights=None):
    """
    Plot a histogram of the correlation coefficients with the fitted bimodal distribution.
    """
    # Remove NaNs
    corr_coeffs = corr_coeffs[np.isfinite(corr_coeffs)]

    # Plot histogram
    counts, bin_edges = np.histogram(corr_coeffs, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.bar(
        bin_centers,
        counts,
        width=bin_edges[1] - bin_edges[0],
        alpha=0.6,
        color="c",
        label="Histogram",
    )

    # Plot fitted distributions
    if means is not None and stds is not None and weights is not None:
        x = np.linspace(bin_edges[0], bin_edges[-1], 1000)
        for mean, std, weight in zip(means, stds, weights):
            plt.plot(
                x,
                weight * norm.pdf(x, mean, std),
                label=f"Component: mean={mean:.2f}, std={std:.2f}",
            )

    plt.xlabel("Correlation Coefficient")
    plt.ylabel("Density")
    plt.title("Bimodal Fit of Correlation Coefficients")
    plt.legend()
    plt.grid(True)
    plt.show(block=False)


def plot_fitted_parameters_incrementally(analyses, averaging_times):
    """
    Plot the evolution of the fitted parameters (means, stds, weights) incrementally,
    including in-between variance.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    # Plot standard deviations
    for i in range(len(analyses[0]["stds"])):
        ax.plot(
            averaging_times,
            [analysis["stds"][i] for analysis in analyses],
            marker="o",
            label=f"Std {i+1}",
        )
    ax.set_ylabel("Standard Deviation")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()


def plot_histogram_over_time(corr_matrices, averaging_times):
    """
    Plot histogram evolution of correlation coefficients over averaging time.
    """
    num_plots = len(corr_matrices)
    fig, axes = plt.subplots(1, num_plots, figsize=(16, 4), sharey=True)

    for i, (corr_matrix, avg_time) in enumerate(zip(corr_matrices, averaging_times)):
        flattened_corr = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        axes[i].hist(flattened_corr, bins=50, color="c", alpha=0.7, edgecolor="k")
        axes[i].set_title(f"Time: {avg_time:.1f} h")
        axes[i].set_xlabel("Correlation Coefficient")

    axes[0].set_ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def plot_thresholded_counts(data):
    """
    Plot thresholded signal and reference counts along the measurement index.

    Parameters:
        data (dict): Dictionary containing 'nv_list' and 'counts'.
    """
    nv_list = data["nv_list"]
    counts = np.array(data["counts"])
    num_nvs = len(nv_list)

    # Separate signal and reference counts
    sig_counts = np.array(counts[0])  # Signal counts
    ref_counts = np.array(counts[1])  # Reference counts

    # Apply dynamic thresholding
    thresholded_sig_counts, thresholded_ref_counts = threshold_counts(
        nv_list, sig_counts, ref_counts, dynamic_thresh=True
    )

    # Aggregate counts (sum over all NVs for each timestamp)
    thresholded_sig_counts = np.array(thresholded_sig_counts).reshape((num_nvs, -1))
    thresholded_ref_counts = np.array(thresholded_ref_counts).reshape((num_nvs, -1))
    sig_sum = thresholded_sig_counts.sum(axis=0)
    ref_sum = thresholded_ref_counts.sum(axis=0)

    # Plot signal counts over time
    time_dependent_fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        sig_sum,
        marker="o",
        markersize=2,
        linestyle="-",
        alpha=0.7,
        label="Thresholded Signal Count",
    )
    ax.set_xlabel("Time (s)", fontsize=14)
    ax.set_ylabel("Aggregate Signal Count", fontsize=14)
    ax.set_title("Time-Dependent Thresholded Signal Count", fontsize=16)
    ax.set_xlim(0, 60000)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

    # Plot reference counts over time
    time_dependent_fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        ref_sum,
        marker="x",
        markersize=2,
        linestyle="--",
        alpha=0.7,
        label="Thresholded Reference Count",
    )
    ax.set_xlabel("Time (s)", fontsize=14)
    ax.set_ylabel("Aggregate Reference Count", fontsize=14)
    ax.set_title("Time-Dependent Thresholded Reference Count", fontsize=16)
    ax.set_xlim(0, 60000)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.show()


# end region

if __name__ == "__main__":
    kpl.init_kplotlib()
    # region Process and analyze data from single file
    # file_id = 1667457284652
    # file_id = 1737922643755
    # data = dm.get_raw_data(file_id=file_id)

    # final data set after randomizing the scc order between two groups
    file_ids = ["2025_10_12-10_27_20-rubin-nv0_2025_09_08","2025_10_12-10_27_20-rubin-nv0_2025_09_08"]
    data = widefield.process_multiple_files(file_ids)
    # process_and_plot(data, rearrangement="block")
    plot_nv_network(data)
    # plot_thresholded_counts(data)
    # try:
    #     # print(data.shape)
    #     # Process and plot the heatmaops with a rearrangement pattern
    #     # process_and_plot(data, rearrangement="alternate_quadrants")
    #     process_and_plot(data, rearrangement="block")
    #     # rearrangement (str): ('alternate_quadrants', 'checkerboard', 'block', 'spiral', etc.).
    #     # Process and plot netwrok graph
    #     # plot_nv_network(data)
    #     # plot_nv_network_3d(data)
    # except Exception as e:
    #     print(f"Error occurred: {e}")

    plt.show(block=True)

    # analyses = process_files_incrementally(file_ids)
    # # Generate example averaging times for each step
    # averaging_times = np.linspace(4, 20, len(file_ids))

    # # Plot fitted parameters incrementally
    # plot_fitted_parameters_incrementally(analyses, averaging_times)

    # # try:
    # #     analyses = process_files_incrementally(file_ids)
    # #     # Generate example averaging times for each step
    # #     averaging_times = np.linspace(4, 20, len(file_ids))

    # #     # Plot fitted parameters incrementally
    # #     plot_fitted_parameters_incrementally(analyses, averaging_times)

    # # except Exception as e:
    # #     print(f"Error occurred: {e}")

    # plt.show(block=True)
