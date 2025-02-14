import os
from datetime import datetime

import matplotlib.patches as Circle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import curve_fit
from skimage.draw import disk

from utils import data_manager as dm
from utils import kplotlib as kpl


def filter_and_reorder_nv_coords(
    nv_coordinates, integrated_intensities, reference_nv, min_distance=3
):
    """
    Filters NV coordinates based on distance from each other and reorders based on distance from a reference NV.

    Parameters:
    - nv_coordinates: List of NV coordinates to filter and reorder.
    - integrated_intensities: Corresponding intensities for each NV coordinate.
    - reference_nv: Reference NV coordinate to include and base reordering on.
    - min_distance: Minimum allowable distance between NVs.

    Returns:
    - reordered_coords: Filtered and reordered NV coordinates.
    - reordered_intensities: Corresponding reordered intensities.
    """
    nv_coords = [reference_nv]  # Initialize with the reference NV

    # Filter NV coordinates based on minimum distance
    for coord in nv_coordinates:
        keep_coord = True
        for existing_coord in nv_coords:
            distance = np.linalg.norm(np.array(existing_coord) - np.array(coord))
            if distance < min_distance:
                keep_coord = False
                break
        if keep_coord:
            nv_coords.append(coord)

    # Reorder based on distance to the reference NV
    distances = [
        np.linalg.norm(np.array(coord) - np.array(reference_nv)) for coord in nv_coords
    ]
    sorted_indices = np.argsort(distances)

    reordered_coords = [nv_coords[idx] for idx in sorted_indices]
    reordered_intensities = [integrated_intensities[idx] for idx in sorted_indices]

    return reordered_coords, reordered_intensities


# Save the results to a file
def save_results(nv_coordinates, spot_weights, updated_spot_weights, filename):
    """
    Save NV data results to an .npz file.

    nv_coordinates: list or array of NV coordinates
    integrated_intensities: array of integrated intensities
    spot_weights: array of weights (inverse of integrated intensities)
    filename: string, the name of the file to save results
    """
    # Ensure the directory exists
    path = os.path.dirname(filename)
    if not os.path.exists(path):
        os.makedirs(path)  # Create the directory if it doesn't exist

    # Save the data to a .npz file
    np.savez(
        filename,
        nv_coordinates=nv_coordinates,
        spot_weights=spot_weights,
        updated_spot_weights=updated_spot_weights,
    )


def load_nv_coords(
    # file_path="slmsuite/nv_blob_detection/nv_blob_filtered_77nvs_new.npz",
    file_path="slmsuite/nv_blob_detection/nv_blob_filtered_240nvs.npz",
):
    data = np.load(file_path)
    print(data.keys())
    nv_coordinates = data["nv_coordinates"]
    # spot_weights = data["spot_weights"]
    spot_weights = data["updated_spot_weights"]
    # spot_weights = data["integrated_counts"]
    # spot_weights = data["integrated_counts"]
    return nv_coordinates, spot_weights


def adjust_aom_voltage_for_slm(aom_voltages, power_law_params):
    aom_voltages = np.array(aom_voltages)
    a, b, c = power_law_params
    nv_powers = a * (aom_voltages**b) + c
    scaled_nv_powers = nv_powers / (len(nv_powers))
    # Normalize powers across all spots
    total_power = np.sum(scaled_nv_powers)
    nv_weights = nv_powers / total_power
    # Compute adjusted AOM voltage for the total power
    adjusted_aom_voltage = ((total_power - c) / a) ** (1 / b)
    return nv_weights, adjusted_aom_voltage


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_adjustment(x, median, increase=True):
    adjustment = 1 / (1 + np.exp(-(x - median)))
    return adjustment if increase else 1 - adjustment


# Main section of the code
if __name__ == "__main__":
    kpl.init_kplotlib()
    # data = dm.get_raw_data(file_id=1711487774016, load_npz=True)
    # img_array = np.array(data["ref_img_array"])
    # file_id = 1713167190804
    file_id = 1713212456618
    data = dm.get_raw_data(file_id=file_id, load_npz=True)  # optimal spotweights
    nv_entries = data["nv_data"]
    optimal_aom_values = [entry["optimal_step_value"] for entry in nv_entries]
    print(optimal_aom_values)
    nv_coordinates, spot_weights = load_nv_coords(
        # file_path="slmsuite/nv_blob_detection/nv_blob_filtered_160nvs.npz"
        # file_path="slmsuite/nv_blob_detection/nv_blob_filtered_160nvs_reordered.npz"
        file_path="slmsuite/nv_blob_detection/nv_blob_filtered_160nvs_reordered_manual_updated.npz"
    )
    nv_coordinates = nv_coordinates.tolist()
    spot_weights = spot_weights.tolist()
    print(spot_weights)
    # # Filter and reorder NV coordinates based on reference NV
    sigma = 2.0
    reference_nv = [106.923, 120.549]
    filtered_reordered_coords, filtered_reordered_spot_weights = (
        filter_and_reorder_nv_coords(
            nv_coordinates, spot_weights, reference_nv, min_distance=3
        )
    )

    # aom_voltage = 0.39  # Current AOM voltage
    power_law_params = [3.7e5, 6.97, 8e-14]  # Example power-law fit parameters
    updated_spot_weights, adjusted_aom_voltage = adjust_aom_voltage_for_slm(
        optimal_aom_values, power_law_params
    )
    indices_to_update_reaout = [27, 41, 45, 55, 130]
    indices_to_update_prep = [
        2,
        4,
        6,
        18,
        45,
        60,
        61,
        67,
        69,
        71,
        84,
        99,
        102,
        103,
        108,
        111,
        129,
        130,
        133,
        135,
        136,
        138,
        148,
        150,
        155,
        158,
        159,
    ]
    # smoothing with sigmoid
    # smoothed_spot_weights = sigmoid(
    #     updated_spot_weights - np.median(updated_spot_weights)
    # )
    indices_to_update_readout = [27, 41, 45, 55, 130]
    indices_to_update_prep = [
        2,
        4,
        6,
        18,
        45,
        60,
        61,
        67,
        69,
        71,
        84,
        99,
        102,
        103,
        108,
        111,
        129,
        130,
        133,
        135,
        136,
        138,
        148,
        150,
        155,
        158,
        159,
    ]

    # Copy of spot weights for updates
    adjusted_spot_weights = spot_weights.copy()

    # Apply sigmoid smoothing for power adjustment
    smoothed_spot_weights = sigmoid(np.array(spot_weights) - np.median(spot_weights))
    #  Calculate the median of the weights
    # median_weight = np.mean(spot_weights)
    median_weight = 0.15

    # Adjust weights for readout and prep indices
    for idx in indices_to_update_readout:
        adjusted_spot_weights[idx] = sigmoid_adjustment(
            spot_weights[idx], median=median_weight, increase=True
        )

    for idx in indices_to_update_prep:
        adjusted_spot_weights[idx] = sigmoid_adjustment(
            spot_weights[idx], median=median_weight, increase=False
        )
    # smoothed_spot_weights = spot_weights.copy()
    # Update spot weights using the sigmoid function
    # smoothed_spot_weights_1 = sigmoid(np.array(spot_weights) - np.median(spot_weights))

    # Update only the specified indices
    # for idx in indices_to_update:
    #     smoothed_spot_weights[idx] = smoothed_spot_weights_1[idx]

    # # Print adjusted weights for verification
    # print("NV Index | Coords    | Original Weight | Updated Weight")
    # print("-" * 60)
    # for idx in indices_to_update:
    #     print(
    #         f"{idx:<8} | {nv_coordinates[idx]} | {spot_weights[idx]:.3f} | {smoothed_spot_weights[idx]:.3f}"
    #     )

    # Save the updated results
    # save_results(
    #     nv_coordinates,
    #     spot_weights,
    #     smoothed_spot_weights,
    #     filename="slmsuite/nv_blob_detection/updated_nv_blob_filtered_160nvs_reordered.npz",
    # )
    print(adjusted_spot_weights)
    # # Prepare the raw data as a list of dictionarie

    # save_results(
    #     nv_coordinates,
    #     spot_weights,
    #     adjusted_spot_weights,
    #     filename="slmsuite/nv_blob_detection/nv_blob_filtered_160nvs_reordered_manual_updated.npz",
    # )

    # # Plot the original image with circles around each NV
    # fig, ax = plt.subplots()
    # title = "50ms, Ref"
    # kpl.imshow(ax, img_array, title=title, cbar_label="Photons")
    # # Draw circles and index numbers
    # for idx, coord in enumerate(filtered_reordered_coords):
    #     circ = plt.Circle(coord, sigma, color="lightblue", fill=False, linewidth=0.5)
    #     ax.add_patch(circ)
    #     # Place text just above the circle
    #     ax.text(
    #         coord[0],
    #         coord[1] - sigma - 1,
    #         str(idx + 1),
    #         color="white",
    #         fontsize=6,
    #         ha="center",
    #     )
    # plt.show(block=True)
