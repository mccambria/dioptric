

# -*- coding: utf-8 -*-
"""
Created on Semptember 16th, 2024
@author: Saroj Chand
"""

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.ma as ma
import seaborn as sns
from matplotlib import patches
from matplotlib.ticker import MaxNLocator

from utils import data_manager as dm

# from utils.tool_belt import nan_corr_coef
from utils.widefield import threshold_counts


# Optimized nan_corr_coef function
def nan_corr_coef(arr):
    """
    Calculate Pearson correlation coefficients for a 2D array, ignoring NaN values.

    This version masks NaN values and computes the correlation coefficient between rows.

    Parameters:
    - arr: 2D numpy array where each row represents a different set of data points.

    Returns:
    - corr_coef_arr: Symmetric matrix of correlation coefficients, with NaN values handled.
    """
    arr = np.array(arr)

    # Mask NaN values in the array
    masked_arr = ma.masked_invalid(arr)
    # Compute the correlation coefficient using masked arrays, ignoring NaNs
    corr_coef_arr = np.ma.corrcoef(masked_arr, rowvar=True)

    # Convert masked correlations back to a standard numpy array, filling masked entries with NaN
    corr_coef_arr = corr_coef_arr.filled(np.nan)

    return corr_coef_arr


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
    nv_list = data.get("nv_list", [])
    counts = np.array(data.get("counts", []))

    if len(nv_list) == 0 or counts.size == 0:
        print("Error: Data does not contain NV list or counts.")
        return None

    # Separate signal and reference counts
    sig_counts = np.array(counts[0])
    ref_counts = np.array(counts[1])

    # Remove NVs with NaN values
    num_nvs = len(nv_list)

    # Thresholding counts with dynamic thresholds
    thresh_method = "otsu"
    sig_counts, ref_counts = threshold_counts(nv_list, sig_counts, ref_counts, thresh_method)
    sig_counts, ref_counts, nv_list = remove_nans_from_data(
        sig_counts, ref_counts, nv_list
    )
    num_nvs = len(nv_list)

    # Thresholding counts with dynamic thresholds
    thresh_method = "entropy"
    sig_counts, ref_counts = threshold_counts(
        nv_list, sig_counts, ref_counts, method=thresh_method
    )

    # Flatten counts for each NV
    flattened_sig_counts = [sig_counts[ind].flatten() for ind in range(num_nvs)]
    flattened_ref_counts = [ref_counts[ind].flatten() for ind in range(num_nvs)]

    # Calculate correlations
    sig_corr_coeffs = nan_corr_coef(flattened_sig_counts)
    ref_corr_coeffs = nan_corr_coef(flattened_ref_counts)
    sig_corr_coeffs = rescale_extreme_values(
        sig_corr_coeffs, sigma_threshold=0.3, method="tanh"
    )
    ref_corr_coeffs = rescale_extreme_values(
        ref_corr_coeffs, sigma_threshold=0.3, method="tanh"
    )
    # plot_correlation_histogram(sig_corr_coeffs, bins=50)
    # Apply the same rearrangement to signal, reference, and ideal matrices
    if rearrangement == "spin_flip":
        nv_list, sig_corr_coeffs, ref_corr_coeffs = rearrange_spin_flip(
            nv_list, sig_corr_coeffs, ref_corr_coeffs
        )
    elif rearrangement == "checkerboard":
        nv_list, sig_corr_coeffs, ref_corr_coeffs = rearrange_checkerboard(
            nv_list, sig_corr_coeffs, ref_corr_coeffs
        )
    elif rearrangement == "block":
        nv_list, sig_corr_coeffs, ref_corr_coeffs = rearrange_block(
            nv_list, sig_corr_coeffs, ref_corr_coeffs
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
        nv_list, sig_corr_coeffs, ref_corr_coeffs = rearrange_alternate_quadrants(
            nv_list, sig_corr_coeffs, ref_corr_coeffs
        )
    elif rearrangement == "concentric_circles":
        nv_list, sig_corr_coeffs, ref_corr_coeffs = rearrange_concentric_circles(
            nv_list, sig_corr_coeffs, ref_corr_coeffs
        )
    elif rearrangement == "corr_sign_individual":
        nv_list, sig_corr_coeffs, ref_corr_coeffs = rearrange_by_corr_sign_individual(
            nv_list, sig_corr_coeffs, ref_corr_coeffs
        )
    elif rearrangement == "corr_sign":
        nv_list, sig_corr_coeffs, ref_corr_coeffs = rearrange_by_corr_sign(
            nv_list, sig_corr_coeffs, ref_corr_coeffs
        )
    else:
        pass

    # Generate ideal correlation matrix based on spin flips before rearrangement
    spin_flips = np.array([-1 if nv.spin_flip else +1 for nv in nv_list])
    ideal_sig_corr_coeffs = np.outer(spin_flips, spin_flips).astype(float)
    # ideal_ref_corr_coeffs = np.zeros((num_nvs, num_nvs), dtype=float)
    # Calculate min and max correlation values for the actual signal and reference correlations
    positive_corrs = sig_corr_coeffs[sig_corr_coeffs > 0]
    negative_corrs = sig_corr_coeffs[sig_corr_coeffs < 0]
    # print(len(positive_corrs))
    # print(len(negative_corrs))
    # Set vmin and vmax separately for signal/reference correlations
    mean_corr = np.nanmean(sig_corr_coeffs)
    std_corr = np.nanstd(sig_corr_coeffs)
    sig_vmax = mean_corr + std_corr
    sig_vmin = -sig_vmax

    # For reference correlations (assuming reference should be scaled the same way as signal)
    ref_vmin = sig_vmin
    ref_vmax = sig_vmax

    # Set vmin and vmax separately for the ideal correlation matrix
    ideal_vmin = (
        -1
    )  # Since the ideal matrix is binary (-1 for anti-correlated, +1 for correlated)
    ideal_vmax = 1

    # Plotting setup
    figsize = [15, 5]
    fig, axes_pack = plt.subplots(ncols=3, figsize=figsize)
    titles = ["Ideal Signal", "Signal", "Reference"]
    vals = [ideal_sig_corr_coeffs, sig_corr_coeffs, ref_corr_coeffs]

    # Use Seaborn heatmap for visualization
    for ind, (val, title) in enumerate(zip(vals, titles)):
        np.fill_diagonal(val, np.nan)  # Set diagonal to NaN for cleaner visualization
        ax = axes_pack[ind]

        # Set vmin and vmax depending on whether it's the ideal or actual data
        if title == "Ideal Signal":
            vmin, vmax = ideal_vmin, ideal_vmax
        elif title == "Signal":
            vmin, vmax = sig_vmin, sig_vmax
        else:
            vmin, vmax = ref_vmin, ref_vmax

        heatmap = sns.heatmap(
            val,
            ax=ax,
            cmap="coolwarm",
            cbar=True,
            vmin=vmin,
            vmax=vmax,
            square=True,
            mask=np.isnan(val),
            annot=False,  # Set True if you want the values to appear on the heatmap
            cbar_kws={"shrink": 0.6},  # Shrink colorbar for better fit
        )

        ax.set_title(title, fontsize=16)

        # Add a colorbar label
        cbar = heatmap.collections[0].colorbar
        cbar.set_label("Correlation coefficient", fontsize=16)
        cbar.ax.tick_params(labelsize=16)

        # Set dynamic tick locations based on the number of NVs
        max_ticks = 6  # Maximum number of ticks (adjustable)
        tick_interval = max(1, num_nvs // max_ticks)
        ax.set_xticks(np.arange(0, num_nvs, tick_interval))
        ax.set_yticks(np.arange(0, num_nvs, tick_interval))

        # Set font size for ticks
        ax.tick_params(axis="both", which="major", labelsize=16)

        # Set x and y labels only for Signal and Reference
        ax.set_xlabel("NV index", fontsize=16)
        ax.set_ylabel("NV index", fontsize=16)

    # Adjust subplots for proper spacing
    fig.subplots_adjust(left=0.05, right=0.88, bottom=0.05, top=0.95, wspace=0.3)
    if fig is not None:
        dm.save_figure(fig, file_path)
    plt.show()


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


def rescale_extreme_values(sig_corr, sigma_threshold=2.0, method="tanh"):
    """
    Rescale extreme values in the correlation matrix based on standard deviation thresholds.

    Parameters:
    - sig_corr: Signal correlation coefficients matrix (77x77).
    - sigma_threshold: Threshold in terms of standard deviations to consider values as extreme.
    - method: Method to rescale extreme values ('tanh' or 'sigmoid').

    Returns:
    - rescaled_corr: Correlation matrix with extreme values rescaled.
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


def plot_correlation_histogram(corr_matrix, bins=50):
    """
    Plot a histogram of the correlation coefficients from the correlation matrix.

    Parameters:
    - corr_matrix: The correlation matrix (e.g., sig_corr).
    - bins: Number of bins for the histogram.
    """
    # Remove the diagonal (which contains 1s for self-correlation) and flatten the matrix
    flattened_corr = corr_matrix[
        np.triu_indices_from(corr_matrix, k=1)
    ]  # Only take upper triangle without diagonal

    # Plot histogram
    plt.figure(figsize=(8, 6))
    plt.hist(flattened_corr, bins=bins, color="c", edgecolor="k", alpha=0.7)

    # Add labels and title
    plt.xlabel("Correlation Coefficient", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.title("Histogram of Correlation Coefficients", fontsize=16)

    # Display the plot
    plt.grid(True)
    plt.show()


# Rearrangement Functions Based on Spin Flip
def rearrange_spin_flip(nv_list, sig_corr, ref_corr):
    """Rearrange spins based on their flip status: +1 (spin up) followed by -1 (spin down)."""
    spin_plus_indices = [
        i for i, nv in enumerate(nv_list) if not nv.spin_flip
    ]  # Spin up
    spin_minus_indices = [
        i for i, nv in enumerate(nv_list) if nv.spin_flip
    ]  # Spin down
    reshuffled_indices = spin_plus_indices + spin_minus_indices
    return apply_rearrangement(nv_list, sig_corr, ref_corr, reshuffled_indices)


def rearrange_checkerboard(nv_list, sig_corr, ref_corr):
    """Checkerboard pattern where alternating spins are up (+1) and down (-1)."""
    spin_plus_indices = [
        i for i, nv in enumerate(nv_list) if not nv.spin_flip
    ]  # Spin up
    spin_minus_indices = [
        i for i, nv in enumerate(nv_list) if nv.spin_flip
    ]  # Spin down
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
    return apply_rearrangement(nv_list, sig_corr, ref_corr, reshuffled_indices)


def rearrange_block(nv_list, sig_corr, ref_corr):
    """Block arrangement: first half spin up (+1), second half spin down (-1)."""
    half = len(nv_list) // 2
    spin_plus_indices = [
        i for i, nv in enumerate(nv_list) if not nv.spin_flip
    ]  # Spin up
    spin_minus_indices = [
        i for i, nv in enumerate(nv_list) if nv.spin_flip
    ]  # Spin down
    reshuffled_indices = spin_plus_indices[:half] + spin_minus_indices[:half]
    return apply_rearrangement(nv_list, sig_corr, ref_corr, reshuffled_indices)


def rearrange_spiral(nv_list, sig_corr, ref_corr):
    """Spiral arrangement based on spin flip status. Spins in a spiral-like pattern."""
    spin_plus_indices = [
        i for i, nv in enumerate(nv_list) if not nv.spin_flip
    ]  # Spin up
    spin_minus_indices = [
        i for i, nv in enumerate(nv_list) if nv.spin_flip
    ]  # Spin down
    reshuffled_indices = np.argsort(
        [np.sin(i) for i in range(len(nv_list))]
    )  # Simple spiral pattern
    spin_up_first = [i for i in reshuffled_indices if not nv_list[i].spin_flip] + [
        i for i in reshuffled_indices if nv_list[i].spin_flip
    ]
    return apply_rearrangement(nv_list, sig_corr, ref_corr, spin_up_first)


def rearrange_random(nv_list, sig_corr, ref_corr):
    """Random arrangement of spin-up and spin-down NV centers."""
    np.random.seed(42)  # Ensure reproducibility
    reshuffled_indices = np.random.permutation(
        len(nv_list)
    )  # Random shuffle of indices
    return apply_rearrangement(nv_list, sig_corr, ref_corr, reshuffled_indices)


def rearrange_alternate_quadrants(nv_list, sig_corr, ref_corr):
    """Divide NV centers into four quadrants and alternate spin-up and spin-down in each quadrant."""
    num_nvs = len(nv_list)
    spin_plus_indices = [
        i for i, nv in enumerate(nv_list) if not nv.spin_flip
    ]  # Spin up
    spin_minus_indices = [
        i for i, nv in enumerate(nv_list) if nv.spin_flip
    ]  # Spin down
    reshuffled_indices = []

    # Alternate quadrants
    for i in range(num_nvs):
        if i < num_nvs // 4:
            reshuffled_indices.append(
                spin_plus_indices.pop(0)
                if spin_plus_indices
                else spin_minus_indices.pop(0)
            )
        elif i < num_nvs // 2:
            reshuffled_indices.append(
                spin_minus_indices.pop(0)
                if spin_minus_indices
                else spin_plus_indices.pop(0)
            )
        elif i < 3 * num_nvs // 4:
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

    return apply_rearrangement(nv_list, sig_corr, ref_corr, reshuffled_indices)


def rearrange_concentric_circles(nv_list, sig_corr, ref_corr):
    """Concentric circle pattern where inner circles are spin-up, outer circles are spin-down."""
    spin_plus_indices = [
        i for i, nv in enumerate(nv_list) if not nv.spin_flip
    ]  # Spin up
    spin_minus_indices = [
        i for i, nv in enumerate(nv_list) if nv.spin_flip
    ]  # Spin down
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


def rearrange_by_corr_sign_individual(nv_list, sig_corr, ref_corr):
    """
    Rearrange NV centers based on whether their correlations are predominantly positive or negative,
    using NV coordinates and pairs in the correlation matrix.

    Parameters:
    - nv_list: List of NV center coordinates [(x, y)].
    - sig_corr: Signal correlation coefficients matrix.
    - ref_corr: Reference correlation coefficients matrix.

    Returns:
    - nv_list_reshuffled: Rearranged NV centers by coordinates.
    - sig_corr_reshuffled: Rearranged signal correlation coefficients.
    - ref_corr_reshuffled: Rearranged reference correlation coefficients.
    """
    # Collect NV coordinate pairs of positive and negative correlations
    positive_corr_indices = []
    negative_corr_indices = []

    # Iterate over the upper triangle of the correlation matrix to classify NV pairs
    for i in range(sig_corr.shape[0]):
        for j in range(i + 1, sig_corr.shape[1]):
            if sig_corr[i, j] > 0:
                # Positive correlation: Add the indices of NVs i and j
                positive_corr_indices.append(i)
                positive_corr_indices.append(j)
            elif sig_corr[i, j] <= 0:
                # Negative correlation: Add the indices of NVs i and j
                negative_corr_indices.append(i)
                negative_corr_indices.append(j)

    # Convert lists to unique indices
    positive_corr_indices = list(np.unique(positive_corr_indices))
    negative_corr_indices = list(np.unique(negative_corr_indices))
    # Combine positive and negative NV indices
    reshuffled_indices = positive_corr_indices + negative_corr_indices
    return apply_rearrangement(nv_list, sig_corr, ref_corr, reshuffled_indices)


def rearrange_by_corr_sign(nv_list, sig_corr, ref_corr):
    """
    Rearrange NV centers and correlation matrices from lowest to highest correlation sum along the diagonal.

    Parameters:
    - nv_list: List of NV center coordinates [(x, y)].
    - sig_corr: Signal correlation coefficients matrix.
    - ref_corr: Reference correlation coefficients matrix.

    Returns:
    - nv_list_reshuffled: NV centers rearranged by increasing correlation sum.
    - sig_corr_reshuffled: Signal correlation coefficients matrix rearranged.
    - ref_corr_reshuffled: Reference correlation coefficients matrix rearranged.
    """
    # Calculate the sum of absolute correlations for each NV
    corr_sums = np.nansum(np.abs(sig_corr), axis=1)

    # Get the indices that would sort the correlation sums in ascending order
    sorted_indices = np.argsort(corr_sums)

    # Apply the sorting to rearrange the NV list and correlation matrices
    return apply_rearrangement(nv_list, sig_corr, ref_corr, sorted_indices)


def apply_rearrangement(nv_list, sig_corr, ref_corr, reshuffled_indices):
    nv_list_reshuffled = [nv_list[i] for i in reshuffled_indices]
    sig_corr_reshuffled = sig_corr[np.ix_(reshuffled_indices, reshuffled_indices)]
    ref_corr_reshuffled = ref_corr[np.ix_(reshuffled_indices, reshuffled_indices)]

    return nv_list_reshuffled, sig_corr_reshuffled, ref_corr_reshuffled


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
    # Unpack data
    nv_list = data.get("nv_list", [])
    counts = np.array(data.get("counts", []))

    if len(nv_list) == 0 or counts.size == 0:
        print("Error: Data does not contain NV list or counts.")
        return None

    # Separate signal and reference counts
    sig_counts = np.array(counts[0])
    ref_counts = np.array(counts[1])

    # Remove NVs with NaN values
    sig_counts, ref_counts, nv_list = remove_nans_from_data(
        sig_counts, ref_counts, nv_list
    )
    num_nvs = len(nv_list)

    # Flatten counts for each NV
    flattened_sig_counts = [sig_counts[ind].flatten() for ind in range(num_nvs)]
    flattened_ref_counts = [ref_counts[ind].flatten() for ind in range(num_nvs)]

    # Calculate correlations
    sig_corr_coeffs = nan_corr_coef(flattened_sig_counts)

    # Initialize a graph
    G = nx.Graph()

    # Add nodes to the graph with pixel coordinates as their positions
    for i, nv in enumerate(nv_list):
        G.add_node(i, pos=nv.coords["pixel"], spin_flip=nv.spin_flip)

    # Extract node positions for plotting
    pos = nx.get_node_attributes(G, "pos")

    # Separate the nodes into spin-up and spin-down based on `spin_flip`
    spin_up_nodes = [
        i for i, nv in enumerate(nv_list) if not nv.spin_flip
    ]  # Spin-up: False (not flipped)
    spin_down_nodes = [
        i for i, nv in enumerate(nv_list) if nv.spin_flip
    ]  # Spin-down: True (flipped)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw spin-up nodes in red
    nx.draw_networkx_nodes(
        G, pos, nodelist=spin_up_nodes, node_color="red", node_size=60, label="Spin-up"
    )

    # Draw spin-down nodes in blue
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=spin_down_nodes,
        node_color="blue",
        node_size=60,
        label="Spin-down",
    )

    # Add edges based on correlation coefficients
    edges = []
    edge_colors = []
    edge_widths = []
    edge_alphas = []

    # Threshold for drawing edges
    threshold = (
        0.0  # Only draw edges if the correlation is above this absolute threshold
    )
    for i in range(sig_corr_coeffs.shape[0]):
        for j in range(i + 1, sig_corr_coeffs.shape[1]):
            if abs(sig_corr_coeffs[i, j]) > threshold:
            # if sig_corr_coeffs[i, j] < threshold:  # Only anticorrelations (negative values)
                G.add_edge(i, j)
                edges.append((i, j))
                edge_colors.append(sig_corr_coeffs[i, j])  # Correlation value as the edge color
                # edge_widths.append(0.5)  # Edge width proportional to correlation
                edge_widths.append(5 * abs(sig_corr_coeffs[i, j]))  # Edge width proportional to correlation
                edge_alphas.append(0.5 + 0.5 * abs(sig_corr_coeffs[i, j]))  # Transparency proportional to correlation
                # edge_alphas.append(0.5)     
                # Normalize edge colors between -1 and 1 for the colormap
                # mean_corr = np.nanmean(sig_corr_coeffs<0)
                # std_corr = np.nanstd(sig_corr_coeffs)
                # vmax = mean_corr + 0.1*std_corr
                vmax = 0.01
                edge_colors.append(
                    sig_corr_coeffs[i, j]
                )  # Correlation value as the edge color
                edge_widths.append(
                    5 * abs(sig_corr_coeffs[i, j])
                )  # Edge width proportional to correlation
                edge_alphas.append(
                    0.5 + 0.5 * abs(sig_corr_coeffs[i, j])
                )  # Transparency proportional to correlation

    # Normalize edge colors between -1 and 1 for the colormap
    mean_corr = np.nanmean(sig_corr_coeffs)
    std_corr = np.nanstd(sig_corr_coeffs)
    vmax = mean_corr + 0.5 * std_corr
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)

    # Draw curved edges with color based on the correlation coefficients
    draw_curved_edges(
        G, pos, ax, norm, edges, edge_colors, edge_widths, edge_alphas, curvature=0.6
    )

    # Add a color bar to indicate the correlation values, associated with the correct axis
    sm = plt.cm.ScalarMappable(cmap=cm.coolwarm, norm=norm)
    sm.set_array(edge_colors)
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Correlation Coefficient", fontsize=12)

    # Add labels (optional)
    labels = {i: f"{i}" for i in range(len(nv_list))}
    nx.draw_networkx_labels(G, pos, labels, font_size=6, font_color="white", ax=ax)

    # Set title and legend
    plt.title("NV Center Network Graph", fontsize=16)
    plt.legend(scatterpoints=1)
    
    if fig is not None:
        dm.save_figure(fig, file_path)

    if fig is not None:
        dm.save_figure(fig, file_path)

    # Display the plot
    plt.show()


if __name__ == "__main__":
    sns.set(style="white", context="talk")
    # data = dm.get_raw_data(file_id=1653570783798)  # Fetch data
    # data = dm.get_raw_data(file_id=1540048047866)  # Fetch data
    from datetime import datetime

    now = datetime.now()
    date_time_str = now.strftime("%Y%m%d_%H%M%S")
    # file_id =  1662370749488
    file_id =  1667457284652
    data = dm.get_raw_data(file_id= file_id)
    file_id = 1662370749488
    data = dm.get_raw_data(file_id=file_id)
    file_name = dm.get_file_name(file_id=file_id)
    timestamp = dm.get_time_stamp()
    file_path = dm.get_file_path(__file__, file_name, f"{file_id}_{date_time_str}")

    if data is not None:
        # Process and plot the data with a specific rearrangement pattern
        # process_and_plot(data,  rearrangement="by_corr_sign_alternate_block")
        # process_and_plot(data, rearrangement="checkerboard", file_path=file_path)
        plot_nv_network(data, file_path)
    else:
        print("Error: Failed to fetch the raw data.")
