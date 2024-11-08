# -*- coding: utf-8 -*-
"""
Real-time acquisition, histogram analysis, and SLM weight adjustment.

Created on Oct 26, 2024

@author: sbcahnd
"""

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from skimage.filters import threshold_li, threshold_otsu, threshold_triangle

from utils import data_manager as dm
from utils import widefield as widefield


def threshold_counts(nv_list, sig_counts, ref_counts=None, method="otsu"):
    """Threshold counts for NVs based on the selected method."""

    num_nvs = len(nv_list)
    sig_thresholds, ref_thresholds = [], []

    # Process thresholds based on the selected method
    for nv_ind in range(num_nvs):
        combined_counts = (
            np.append(sig_counts[nv_ind].flatten(), ref_counts[nv_ind].flatten())
            if ref_counts is not None
            else sig_counts[nv_ind].flatten()
        )

        # Choose method for thresholding
        if method == "otsu":
            threshold = threshold_otsu(combined_counts)
        elif method == "triangle":
            threshold = threshold_triangle(combined_counts)
        elif method == "entropy":
            threshold = threshold_li(combined_counts)
        else:
            raise ValueError(f"Unknown thresholding method: {method}")

        # Append threshold to the appropriate list
        sig_thresholds.append(threshold)

        # Optional: Compute ref threshold if needed separately
        if ref_counts is not None:
            ref_threshold = (
                threshold_otsu(ref_counts[nv_ind].flatten())
                if method == "otsu"
                else (
                    threshold_triangle(ref_counts[nv_ind].flatten())
                    if method == "triangle"
                    else threshold_li(ref_counts[nv_ind].flatten())
                )
            )
            ref_thresholds.append(ref_threshold)

    return sig_thresholds, ref_thresholds if ref_counts is not None else None


def calculate_metrics(nv_list, raw_data, method="otsu"):
    """Calculate SNR and fidelity for each NV across multiple datasets."""
    metrics = defaultdict(
        lambda: {
            "snr": [],
            "fidelity": [],
            "threshold": [],
            "probability_nv_minus": [],
            "probability_nv_zero": [],
        }
    )
    num_nvs = len(nv_list)
    counts = np.array(raw_data["counts"])

    # Compute optimal thresholds for all NVs using both sig and ref counts
    sig_thresholds, _ = threshold_counts(nv_list, counts[0], counts[1], method=method)

    for nv_ind in range(num_nvs):
        sig_counts_list = counts[0, nv_ind].flatten()
        ref_counts_list = counts[1, nv_ind].flatten()

        # Calculate probabilities for NV- (|0>) and NV0 (|1>) states based on the threshold
        threshold = sig_thresholds[nv_ind]
        p_nv_minus = np.sum(ref_counts_list >= threshold) / len(
            ref_counts_list
        )  # Probability for NV- (|0>)
        p_nv_zero = np.sum(sig_counts_list >= threshold) / len(
            sig_counts_list
        )  # Probability for NV0 (|1>)

        # Calculate error rates
        epsilon_0 = 1 - p_nv_minus  # False negative rate
        epsilon_1 = p_nv_zero  # False positive rate

        # Calculate Fidelity
        fidelity = 1 - 0.5 * (epsilon_0 + epsilon_1)

        # Calculate SNR
        if epsilon_0 == epsilon_1:
            snr = (
                (2 * fidelity - 1) / np.sqrt(2 * fidelity * (1 - fidelity))
                if fidelity < 1
                else np.inf
            )
        else:
            snr = (p_nv_minus - p_nv_zero) / np.sqrt(
                p_nv_minus * (1 - p_nv_minus) + p_nv_zero * (1 - p_nv_zero)
            )

        # Append metrics
        metrics[nv_ind]["snr"].append(snr)
        metrics[nv_ind]["fidelity"].append(fidelity)
        metrics[nv_ind]["threshold"].append(threshold)
        metrics[nv_ind]["probability_nv_minus"].append(p_nv_minus)
        metrics[nv_ind]["probability_nv_zero"].append(1 - p_nv_minus)

    return metrics


import os

import matplotlib.pyplot as plt


def save_high_res_figure(path, dpi=300, bbox_inches="tight", **kwargs):
    """
    Save a Matplotlib figure to the specified path with high resolution.

    Parameters:
    - path (str): Full file path to save the figure, including the file name and extension.
    - dpi (int, optional): Dots per inch for the figure resolution. Default is 300.
    - bbox_inches (str, optional): Bounding box option for tight layout. Default is 'tight'.
    - **kwargs: Additional keyword arguments to pass to plt.savefig().
    """
    # Get the directory from the path
    directory = os.path.dirname(path)

    # Create the directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the figure
    plt.savefig(path, dpi=dpi, bbox_inches=bbox_inches, **kwargs)
    print(f"Figure saved at {path} with resolution {dpi} DPI.")


def visualize_metrics(metrics, datasets):
    """Visualize SNR, fidelity, threshold, and probabilities for NV⁻ and NV⁰ for each NV across datasets."""
    num_nvs = len(metrics)

    # Convert metrics to arrays for easier plotting
    snr_data = np.array([metrics[i]["snr"] for i in range(num_nvs)])
    fidelity_data = np.array([metrics[i]["fidelity"] for i in range(num_nvs)])
    threshold_data = np.array([metrics[i]["threshold"] for i in range(num_nvs)])
    probability_nv_minus = np.array(
        [metrics[i]["probability_nv_minus"] for i in range(num_nvs)]
    )
    probability_nv_zero = np.array(
        [metrics[i]["probability_nv_zero"] for i in range(num_nvs)]
    )

    # Compact Plotting for SNR, Fidelity, Threshold, and Probability metrics in a single row (4 columns)
    plt.figure(figsize=(6, 6))

    # NV⁻ Probability Heatmap
    plt.subplot(1, 4, 1)
    sns.heatmap(
        probability_nv_minus,
        cmap="Blues",
        annot=False,
        cbar_kws={"label": "Probability (NV⁻)"},
        square=True,
    )
    plt.title("NV⁻ Probability", fontsize=10)
    # plt.xlabel("Datasets", fontsize=9)
    plt.ylabel("NV Index", fontsize=9)
    plt.xticks([], fontsize=8)
    plt.yticks(fontsize=8)  # Hide y-ticks for compactness
    plt.gca().set_aspect(0.2)

    # SNR Heatmap
    plt.subplot(1, 4, 2)
    sns.heatmap(
        snr_data, cmap="coolwarm", annot=False, cbar_kws={"label": "SNR"}, square=True
    )
    plt.title("SNR", fontsize=10)
    # plt.xlabel("Datasets", fontsize=9)
    plt.xticks([], fontsize=8)
    plt.yticks([], fontsize=8)
    plt.gca().set_aspect(0.2)

    # Fidelity Heatmap
    plt.subplot(1, 4, 3)
    sns.heatmap(
        fidelity_data,
        cmap="viridis",
        annot=False,
        cbar_kws={"label": "Fidelity"},
        square=True,
    )
    plt.title("Fidelity", fontsize=10)
    # plt.xlabel("Datasets", fontsize=9)
    plt.xticks([], fontsize=8)
    plt.yticks([], fontsize=8)  # Hide y-ticks for compactness
    plt.gca().set_aspect(0.2)

    # Threshold Heatmap
    plt.subplot(1, 4, 4)
    sns.heatmap(
        threshold_data,
        cmap="YlGnBu",
        annot=False,
        cbar_kws={"label": "Threshold"},
        square=True,
    )
    plt.title("Threshold", fontsize=10)
    # plt.xlabel("Datasets", fontsize=9)
    plt.xticks([], fontsize=8)
    plt.yticks([], fontsize=8)  # Hide y-ticks for compactness
    plt.gca().set_aspect(0.2)
    # Save the current figure with high resolution
    path = r"C:\Users\Saroj Chand\OneDrive\Documents\charge_state.png"
    save_high_res_figure(path)
    # Tight layout adjustment for compact fit
    plt.tight_layout(pad=0.01)
    plt.show()

    # import matplotlib.pyplot as plt
    # Scatter plot of SNR vs Fidelity with NV index labels
    # Scatter plot of SNR vs Fidelity with NV index labels
    cmap = plt.cm.get_cmap("viridis", snr_data.shape[1])
    plt.figure(figsize=(6, 5))
    for i in range(snr_data.shape[1]):
        color = cmap(i)  # Assign a color from the colormap
        plt.scatter(
            snr_data[:, i],
            fidelity_data[:, i],
            label=f"Dataset {i+1}",
            alpha=0.8,
            color=color,
            s=60,
            edgecolors="k",
        )

    # Set labels and title with elegant styling for SNR vs Fidelity
    plt.xlabel("SNR", fontsize=12)
    plt.ylabel("Fidelity", fontsize=12)
    plt.title("SNR vs Fidelity Across Datasets", fontsize=14)

    # Adding equations for SNR and Fidelity with definitions of ε₀ and ε₁
    plt.text(
        0.95,
        0.4,
        r"$\text{SNR} = \frac{p_{0|0} - p_{0|1}}{\sqrt{p_{0|0}(1 - p_{0|0}) + p_{0|1}(1 - p_{0|1})}}$",
        transform=plt.gca().transAxes,
        fontsize=12,
        ha="right",
        color="black",
        bbox=dict(facecolor="white", edgecolor="grey", boxstyle="round,pad=0.3"),
    )

    plt.text(
        0.95,
        0.28,
        r"$\text{If } \epsilon_0 = \epsilon_1, \text{ SNR} = \frac{2\mathcal{F} - 1}{\sqrt{2\mathcal{F}(1 - \mathcal{F})}}$",
        transform=plt.gca().transAxes,
        fontsize=12,
        ha="right",
        color="black",
        bbox=dict(facecolor="white", edgecolor="grey", boxstyle="round,pad=0.3"),
    )

    plt.text(
        0.95,
        0.17,
        r"$\text{Fidelity } \mathcal{F} = 1 - \frac{1}{2} (\epsilon_0 + \epsilon_1)$",
        transform=plt.gca().transAxes,
        fontsize=12,
        ha="right",
        color="black",
        bbox=dict(facecolor="white", edgecolor="grey", boxstyle="round,pad=0.3"),
    )

    plt.text(
        0.95,
        0.07,
        r"$\text{where } \epsilon_0 = 1 - p_{0|0}, \; \epsilon_1 = p_{0|1}$",
        transform=plt.gca().transAxes,
        fontsize=12,
        ha="right",
        color="black",
        bbox=dict(facecolor="white", edgecolor="grey", boxstyle="round,pad=0.3"),
    )

    # Save the SNR vs Fidelity plot
    path_fidelity = r"C:\Users\Saroj Chand\OneDrive\Documents\snr_vs_fidelity.png"
    save_high_res_figure(path_fidelity)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

    # Scatter plot of Probability (NV⁻) vs SNR
    plt.figure(figsize=(6, 5))
    for i in range(snr_data.shape[1]):
        color = cmap(i)
        plt.scatter(
            snr_data[:, i],
            probability_nv_minus[:, i],
            label=f"Dataset {i+1}",
            alpha=0.8,
            color=color,
            s=60,
            edgecolors="k",
        )

    # Set labels and title with elegant styling for Probability vs SNR
    plt.xlabel("SNR", fontsize=12)
    plt.ylabel("Probability (NV⁻)", fontsize=12)
    plt.title("Probability (NV⁻) vs SNR Across Datasets", fontsize=14)

    # Save the Probability vs SNR plot
    path_prob = r"C:\Users\Saroj Chand\OneDrive\Documents\prob_vs_snr.png"
    save_high_res_figure(path_prob)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

    # Scatter plot of Probability (NV⁻) vs Fidelity
    plt.figure(figsize=(6, 5))
    for i in range(snr_data.shape[1]):
        color = cmap(i)
        plt.scatter(
            fidelity_data[:, i],
            probability_nv_minus[:, i],
            label=f"Dataset {i+1}",
            alpha=0.8,
            color=color,
            s=60,
            edgecolors="k",
        )

    # Set labels and title with elegant styling for Probability vs Fidelity
    plt.xlabel("Fidelity", fontsize=12)
    plt.ylabel("Probability (NV⁻)", fontsize=12)
    plt.title("Probability (NV⁻) vs Fidelity Across Datasets", fontsize=14)

    # Save the Probability vs Fidelity plot
    path_prob_fidelity = r"C:\Users\Saroj Chand\OneDrive\Documents\prob_vs_fidelity.png"
    save_high_res_figure(path_prob_fidelity)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

    # # Scatter plot of SNR vs Fidelity with NV index labels
    # cmap = plt.cm.get_cmap('viridis', snr_data.shape[1])
    # plt.figure(figsize=(6, 5))
    # for i in range(snr_data.shape[1]):
    #     color = cmap(i)  # Assign a color from the colormap
    #     plt.scatter(snr_data[:, i], fidelity_data[:, i], label=f"Dataset {i+1}", alpha=0.8, color=color, s=60, edgecolors="k")

    # # Set labels and title with elegant styling
    # plt.xlabel("SNR", fontsize=12)
    # plt.ylabel("Fidelity", fontsize=12)
    # plt.title("SNR vs Fidelity Across Datasets", fontsize=14)

    # # Adding equations for SNR and Fidelity
    # plt.text(
    #     0.95, 0.33, r"$\text{SNR} = \frac{p_{0|0} - p_{0|1}}{\sqrt{p_{0|0}(1 - p_{0|0}) + p_{0|1}(1 - p_{0|1})}}$",
    #     transform=plt.gca().transAxes, fontsize=12, ha="right", color="black",
    #     bbox=dict(facecolor="white", edgecolor="grey", boxstyle="round,pad=0.3")
    # )

    # plt.text(
    #     0.95, 0.21, r"$\text{If } \epsilon_0 = \epsilon_1, \text{ SNR} = \frac{2\mathcal{F} - 1}{\sqrt{2\mathcal{F}(1 - \mathcal{F})}}$",
    #     transform=plt.gca().transAxes, fontsize=12, ha="right", color="black",
    #     bbox=dict(facecolor="white", edgecolor="grey", boxstyle="round,pad=0.3")
    # )

    # plt.text(
    #     0.95, 0.1, r"$\text{Fidelity } \mathcal{F} = 1 - \frac{1}{2} (\epsilon_0 + \epsilon_1)$",
    #     transform=plt.gca().transAxes, fontsize=12, ha="right", color="black",
    #     bbox=dict(facecolor="white", edgecolor="grey", boxstyle="round,pad=0.3")
    # )
    # path = r"C:\Users\Saroj Chand\OneDrive\Documents\snrvsfidality.png"
    # save_high_res_figure(path)
    # plt.grid(True, linestyle='--', alpha=0.6)
    # plt.show()


def select_best_dataset(metrics, snr_threshold=1.0, fidelity_threshold=0.8):
    """Selects the best dataset based on SNR and fidelity thresholds."""
    best_dataset = None
    max_valid_nvs = 0

    for dataset_idx in range(len(metrics[0]["snr"])):
        valid_nvs = sum(
            1
            for nv in metrics
            if metrics[nv]["snr"][dataset_idx] > snr_threshold
            and metrics[nv]["fidelity"][dataset_idx] > fidelity_threshold
        )

        if valid_nvs > max_valid_nvs:
            max_valid_nvs = valid_nvs
            best_dataset = dataset_idx

    print(
        f"Best dataset based on thresholds: Dataset {best_dataset + 1} with {max_valid_nvs} NVs meeting criteria."
    )
    return best_dataset


def process_multiple_datasets(dataset_ids):
    """Process multiple datasets using dm to load data by ID."""
    all_metrics = []

    for dataset_id in dataset_ids:
        # Load data from dm
        raw_data = dm.get_raw_data(file_id=dataset_id)

        # Extract nv_list from raw_data
        nv_list = raw_data["nv_list"]

        # Calculate metrics for the loaded dataset
        metrics = calculate_metrics(nv_list, raw_data)
        all_metrics.append(metrics)

    # Combine metrics across datasets
    combined_metrics = defaultdict(
        lambda: {
            "snr": [],
            "fidelity": [],
            "threshold": [],
            "probability_nv_minus": [],
            "probability_nv_zero": [],
        }
    )
    for dataset_metrics in all_metrics:
        for nv_ind, nv_metrics in dataset_metrics.items():
            combined_metrics[nv_ind]["snr"].extend(nv_metrics["snr"])
            combined_metrics[nv_ind]["fidelity"].extend(nv_metrics["fidelity"])
            combined_metrics[nv_ind]["threshold"].extend(nv_metrics["threshold"])
            combined_metrics[nv_ind]["probability_nv_minus"].extend(
                nv_metrics["probability_nv_minus"]
            )
            combined_metrics[nv_ind]["probability_nv_zero"].extend(
                nv_metrics["probability_nv_zero"]
            )

    # Visualize metrics
    visualize_metrics(combined_metrics, dataset_ids)

    # Select the best dataset based on thresholds
    best_dataset_index = select_best_dataset(combined_metrics)
    return best_dataset_index


# Example usage with dataset IDs
dataset_ids = [
    1688554695897,
    # 1688505772462,
    # Add more dataset IDs as needed
]


# # Create a figure for displaying equations only
# plt.figure(figsize=(8, 6))

# # Adding equations for SNR and Fidelity with definitions of ε₀ and ε₁
# plt.text(
#     0.5, 0.8, r"$\text{SNR} = \frac{p_{\text{NV}^- | \text{NV}^-} - p_{\text{NV}^- | \text{NV}^0}}{\sqrt{p_{\text{NV}^- | \text{NV}^-}(1 - p_{\text{NV}^- | \text{NV}^-}) + p_{\text{NV}^- | \text{NV}^0}(1 - p_{\text{NV}^- | \text{NV}^0})}}$",
#     ha="center", color="black", fontsize=14,
#     bbox=dict(facecolor="white", edgecolor="grey", boxstyle="round,pad=0.3")
# )

# plt.text(
#     0.5, 0.6, r"$\text{If } \epsilon_{\text{NV}^-} = \epsilon_{\text{NV}^0}, \text{ SNR} = \frac{2\mathcal{F} - 1}{\sqrt{2\mathcal{F}(1 - \mathcal{F})}}$",
#     ha="center", color="black", fontsize=14,
#     bbox=dict(facecolor="white", edgecolor="grey", boxstyle="round,pad=0.3")
# )

# plt.text(
#     0.5, 0.4, r"$\text{Fidelity } \mathcal{F} = 1 - \frac{1}{2} (\epsilon_{\text{NV}^-} + \epsilon_{\text{NV}^0})$",
#     ha="center", color="black", fontsize=14,
#     bbox=dict(facecolor="white", edgecolor="grey", boxstyle="round,pad=0.3")
# )

# plt.text(
#     0.5, 0.2, r"$\text{where } \epsilon_{\text{NV}^-} = 1 - p_{\text{NV}^- | \text{NV}^-}, \; \epsilon_{\text{NV}^0} = p_{\text{NV}^- | \text{NV}^0}$",
#     ha="center", color="black", fontsize=14,
#     bbox=dict(facecolor="white", edgecolor="grey", boxstyle="round,pad=0.3")
# )

# # Remove axis
# plt.axis('off')

# # Save the figure with equations
# path_equations = r"C:\Users\Saroj Chand\OneDrive\Documents\equations_only.png"
# plt.savefig(path_equations, dpi=300, bbox_inches='tight')
# plt.show()


# Process the datasets
best_dataset = process_multiple_datasets(dataset_ids)
print(f"Proceed with Dataset {best_dataset}")
