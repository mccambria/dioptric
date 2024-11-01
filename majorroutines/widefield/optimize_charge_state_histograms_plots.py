# -*- coding: utf-8 -*-
"""
Real-time acquisition, histogram analysis, and SLM weight adjustment.

Created on Oct 26, 2024

@author: sbcahnd
"""


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import defaultdict
from utils import data_manager as dm
from utils import widefield as widefield

import numpy as np
from skimage.filters import threshold_otsu, threshold_triangle, threshold_li

import numpy as np
from collections import defaultdict
from skimage.filters import threshold_otsu, threshold_triangle, threshold_li

def threshold_counts(nv_list, sig_counts, ref_counts=None, method='otsu'):
    """Threshold counts for NVs based on the selected method."""
    
    num_nvs = len(nv_list)
    sig_thresholds, ref_thresholds = [], []

    # Process thresholds based on the selected method
    for nv_ind in range(num_nvs):
        combined_counts = np.append(
            sig_counts[nv_ind].flatten(), ref_counts[nv_ind].flatten()
        ) if ref_counts is not None else sig_counts[nv_ind].flatten()

        # Choose method for thresholding
        if method == 'otsu':
            threshold = threshold_otsu(combined_counts)
        elif method == 'triangle':
            threshold = threshold_triangle(combined_counts)
        elif method == 'entropy':
            threshold = threshold_li(combined_counts)
        else:
            raise ValueError(f"Unknown thresholding method: {method}")

        # Append threshold to the appropriate list
        sig_thresholds.append(threshold)
        
        # Optional: Compute ref threshold if needed separately
        if ref_counts is not None:
            ref_threshold = threshold_otsu(ref_counts[nv_ind].flatten()) if method == 'otsu' else (
                threshold_triangle(ref_counts[nv_ind].flatten()) if method == 'triangle' else
                threshold_li(ref_counts[nv_ind].flatten())
            )
            ref_thresholds.append(ref_threshold)

    return sig_thresholds, ref_thresholds if ref_counts is not None else None

def calculate_metrics(nv_list, raw_data, method='otsu'):
    """Calculate SNR, fidelity, and ETX for each NV across multiple datasets."""
    metrics = defaultdict(lambda: {"snr": [], "fidelity": [], "etx": []})
    num_nvs = len(nv_list)
    counts = np.array(raw_data["counts"])

    # Compute optimal thresholds for all NVs using both sig and ref counts
    sig_thresholds, _ = threshold_counts(nv_list, counts[0], counts[1], method=method)

    for nv_ind in range(num_nvs):
        sig_counts_list = counts[0, nv_ind].flatten()
        ref_counts_list = counts[1, nv_ind].flatten()

        # Calculate SNR
        noise = np.sqrt(np.var(ref_counts_list) + np.var(sig_counts_list))
        signal = np.mean(ref_counts_list) - np.mean(sig_counts_list)
        snr = signal / noise if noise != 0 else 0

        # Calculate fidelity using the combined threshold
        threshold = sig_thresholds[nv_ind]
        fidelity = np.sum(sig_counts_list < threshold) / len(sig_counts_list)

        # Append metrics
        metrics[nv_ind]["snr"].append(snr)
        metrics[nv_ind]["fidelity"].append(fidelity)
        metrics[nv_ind]["etx"].append(threshold)

    return metrics



def visualize_metrics(metrics, datasets):
    """Visualize SNR, fidelity, and ETX for each NV across datasets."""
    num_nvs = len(metrics)

    # Convert metrics to arrays for easier plotting
    snr_data = np.array([metrics[i]["snr"] for i in range(num_nvs)])
    fidelity_data = np.array([metrics[i]["fidelity"] for i in range(num_nvs)])
    etx_data = np.array([metrics[i]["etx"] for i in range(num_nvs)])

    # Plotting SNR and Fidelity across NVs and datasets
    plt.figure(figsize=(14, 5))
    
    # SNR Heatmap
    plt.subplot(1, 3, 1)
    sns.heatmap(snr_data, cmap="coolwarm", annot=False, cbar_kws={"label": "SNR"})
    plt.title("SNR Across NVs and Datasets")
    plt.xlabel("Datasets")
    plt.ylabel("NV Index")

    # Fidelity Heatmap
    plt.subplot(1, 3, 2)
    sns.heatmap(fidelity_data, cmap="viridis", annot=False, cbar_kws={"label": "Fidelity"})
    plt.title("Fidelity Across NVs and Datasets")
    plt.xlabel("Datasets")

    # ETX Heatmap
    plt.subplot(1, 3, 3)
    sns.heatmap(etx_data, cmap="YlGnBu", annot=False, cbar_kws={"label": "ETX"})
    plt.title("ETX Across NVs and Datasets")
    plt.xlabel("Datasets")

    plt.tight_layout()
    plt.show()

    # Scatter plot of SNR vs Fidelity for dataset selection
    plt.figure(figsize=(10, 6))
    for i in range(snr_data.shape[1]):
        plt.scatter(snr_data[:, i], fidelity_data[:, i], label=f"Dataset {i+1}", alpha=0.7)
    plt.xlabel("SNR")
    plt.ylabel("Fidelity")
    plt.legend()
    plt.title("SNR vs Fidelity Across Datasets")
    plt.show()


def select_best_dataset(metrics, snr_threshold=1.0, fidelity_threshold=0.9):
    """Selects the best dataset based on SNR and fidelity thresholds."""
    best_dataset = None
    max_valid_nvs = 0

    for dataset_idx in range(len(metrics[0]["snr"])):
        valid_nvs = sum(
            1 for nv in metrics if metrics[nv]["snr"][dataset_idx] > snr_threshold
            and metrics[nv]["fidelity"][dataset_idx] > fidelity_threshold
        )

        if valid_nvs > max_valid_nvs:
            max_valid_nvs = valid_nvs
            best_dataset = dataset_idx

    print(f"Best dataset based on thresholds: Dataset {best_dataset + 1} with {max_valid_nvs} NVs meeting criteria.")
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
    combined_metrics = defaultdict(lambda: {"snr": [], "fidelity": [], "etx": []})
    for dataset_metrics in all_metrics:
        for nv_ind, nv_metrics in dataset_metrics.items():
            combined_metrics[nv_ind]["snr"].extend(nv_metrics["snr"])
            combined_metrics[nv_ind]["fidelity"].extend(nv_metrics["fidelity"])
            combined_metrics[nv_ind]["etx"].extend(nv_metrics["etx"])

    # Visualize metrics
    visualize_metrics(combined_metrics, dataset_ids)

    # Select the best dataset based on thresholds
    best_dataset_index = select_best_dataset(combined_metrics)
    return best_dataset_index


# Example usage with dataset IDs
dataset_ids = [
    1688554695897,
    1688505772462,
    # Add more dataset IDs as needed
]

# Process the datasets
best_dataset = process_multiple_datasets(dataset_ids)
print(f"Proceed with Dataset {best_dataset + 1}")

