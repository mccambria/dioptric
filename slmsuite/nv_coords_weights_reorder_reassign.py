import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import curve_fit
from skimage.draw import disk

# from tabulate import tabulate
from utils import data_manager as dm
from utils import kplotlib as kpl


# Define the 2D Gaussian function
def gaussian_2d(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = xy
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta) ** 2) / (2 * sigma_x**2) + (np.sin(theta) ** 2) / (
        2 * sigma_y**2
    )
    b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (4 * sigma_y**2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x**2) + (np.cos(theta) ** 2) / (
        2 * sigma_y**2
    )
    g = offset + amplitude * np.exp(
        -(a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo) ** 2))
    )
    return g.ravel()


# Function to fit a 2D Gaussian around NV coordinates
def fit_gaussian(image, coord, window_size=2):
    x0, y0 = coord
    img_shape_y, img_shape_x = image.shape

    # Ensure the window is within image bounds
    x_min = max(int(x0 - window_size), 0)
    x_max = min(int(x0 + window_size + 1), img_shape_x)
    y_min = max(int(y0 - window_size), 0)
    y_max = min(int(y0 + window_size + 1), img_shape_y)

    if (x_max - x_min) <= 1 or (y_max - y_min) <= 1:
        print(
            f"Invalid cutout for NV at ({x0}, {y0}): Region too small or out of bounds"
        )
        return x0, y0

    # Extract cutout and mesh grid
    x_range = np.arange(x_min, x_max)
    y_range = np.arange(y_min, y_max)
    x, y = np.meshgrid(x_range, y_range)
    image_cutout = image[y_min:y_max, x_min:x_max]

    # Check for valid cutout size
    if image_cutout.size == 0:
        print(f"Zero-size cutout for NV at ({x0}, {y0})")
        return x0, y0

    # Normalize the image cutout
    image_cutout = (image_cutout - np.min(image_cutout)) / (
        np.max(image_cutout) - np.min(image_cutout)
    )

    # Initial guess parameters
    initial_guess = (1, x0, y0, 3, 3, 0, 0)  # Amplitude normalized to 1

    try:
        # Apply bounds to avoid unreasonable parameter values
        bounds = (
            (0, x_min, y_min, 0, 0, -np.pi, 0),  # Lower bounds
            (np.inf, x_max, y_max, np.inf, np.inf, np.pi, np.inf),  # Upper bounds
        )

        # Perform the Gaussian fit
        popt, _ = curve_fit(
            gaussian_2d, (x, y), image_cutout.ravel(), p0=initial_guess, bounds=bounds
        )
        amplitude, fitted_x, fitted_y, _, _, _, _ = popt

        return fitted_x, fitted_y, amplitude

    except Exception as e:
        print(f"Fit failed for NV at ({x0}, {y0}): {e}")
        return x0, y0, 0


def integrate_intensity(image_array, nv_coords, sigma):
    """
    Integrate the intensity around each NV coordinate within a circular region
    defined by sigma, with a Gaussian weighting if needed.
    """
    intensities = []
    for coord in nv_coords:
        # Define a larger radius to ensure full capture of intensity around bright spots
        rr, cc = disk((coord[0], coord[1]), radius=sigma, shape=image_array.shape)

        # Integrate (sum) the intensity values within the disk
        intensity = np.sum(image_array[rr, cc])

        # Append integrated intensity to the list
        intensities.append(intensity)
    return intensities


def remove_outliers(intensities, nv_coords):
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = np.percentile(intensities, 25)
    Q3 = np.percentile(intensities, 75)
    IQR = Q3 - Q1

    # Define bounds for identifying outliers
    lower_bound = Q1 - 1.0 * IQR
    upper_bound = Q3 + 6.5 * IQR
    # lower_bound = 10
    # upper_bound = 100

    # Filter out the outliers and corresponding NV coordinates
    filtered_intensities = []
    filtered_nv_coords = []

    for intensity, coord in zip(intensities, nv_coords):
        if lower_bound <= intensity <= upper_bound:
            filtered_intensities.append(intensity)
            filtered_nv_coords.append(coord)

    return filtered_intensities, filtered_nv_coords


def remove_manual_indices(nv_coords, indices_to_remove):
    """Remove NVs based on manually specified indices"""
    return [
        coord for idx, coord in enumerate(nv_coords) if idx not in indices_to_remove
    ]


def filter_and_reorder_nv_coords(
    nv_coordinates, integrated_intensities, reference_nv, min_distance=3
):
    """
    Filters NV coordinates based on distance from each other and reorders based on distance from a reference NV.

    """
    nv_coords = [reference_nv]  # Store as list for later operations
    # Find the closest NV to the reference_nv in case it's not an exact match
    distances_to_ref = np.linalg.norm(
        np.array(nv_coordinates) - np.array(reference_nv), axis=1
    )
    closest_index = np.argmin(distances_to_ref)  # Get the index of the closest match
    reference_nv = nv_coordinates[closest_index]  # Use this as the reference
    included_indices = [closest_index]  # Track included indices

    # Filter NV coordinates based on minimum distance
    for idx, coord in enumerate(nv_coordinates):
        keep_coord = True
        for existing_coord in nv_coords:
            distance = np.linalg.norm(np.array(existing_coord) - np.array(coord))
            if distance < min_distance:
                keep_coord = False
                break
        if keep_coord:
            nv_coords.append(coord)
            included_indices.append(idx)
            # intensities.append(integrated_intensities[idx])  # Store matching intensity
    print(included_indices)
    # Reorder based on distance to the reference NV
    distances = [
        np.linalg.norm(np.array(coord) - np.array(reference_nv)) for coord in nv_coords
    ]
    sorted_indices = np.argsort(distances)
    reordered_coords = [nv_coords[idx] for idx in sorted_indices]
    reordered_intensities = [integrated_intensities[idx] for idx in sorted_indices]

    return reordered_coords, reordered_intensities, included_indices


def sigmoid_weights(intensities, threshold, beta=1):
    weights = np.exp(beta * (intensities - threshold))
    return weights / np.max(weights)  # Normalize the weights


def linear_weights(intensities, alpha=1):
    weights = 1 / np.power(intensities, alpha)
    weights = weights / np.max(weights)  # Normalize to avoid extreme values
    return weights


def non_linear_weights_adjusted(intensities, alpha=1, beta=0.5, threshold=0.5):
    # Normalize the intensities between 0 and 1
    norm_intensities = intensities / np.max(intensities)

    # Apply a non-linear transformation to only the lower intensities
    weights = np.where(
        norm_intensities > threshold,
        1,  # Keep bright NVs the same
        1
        / (1 + np.exp(-beta * (norm_intensities - threshold)))
        ** alpha,  # Non-linear scaling for low intensities
    )

    # Ensure that the weights are normalized
    weights = weights / np.max(weights)

    return weights


# Save the results to a file
def save_results(nv_coordinates, updated_spot_weights, filename):
    # Ensure the directory exists
    path = os.path.dirname(filename)
    if not os.path.exists(path):
        os.makedirs(path)  # Create the directory if it doesn't exist

    # Save the data to a .npz file
    np.savez(
        filename,
        nv_coordinates=nv_coordinates,
        # integrated_counts=integrated_intensities,
        # spot_weights=spot_weights,
        # nv_powers=nv_powers,
        updated_spot_weights=updated_spot_weights,
    )


def filter_by_snr(snr_list, threshold=0.5):
    """Filter out NVs with SNR below the threshold."""
    return [i for i, snr in enumerate(snr_list) if snr >= threshold]


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


def load_nv_weights(file_path="optimal_separation_and_goodness.txt"):
    # Load data, skipping the header row
    data = np.loadtxt(file_path, delimiter=",", skiprows=1)
    # Extract the step values for separation
    nv_weights = data[:, 2]  # Step Val (Separation) is the 3rd column (index 2)
    return nv_weights


def sigmoid_weight_update(
    fidelities, spot_weights, intensities, alpha=1, beta=10, fidelity_threshold=0.90
):
    # Normalize intensities between 0 and 1
    norm_intensities = intensities / np.max(intensities)

    # Initialize updated weights as 1 (i.e., no change for high-fidelity NVs)
    updated_weights = np.copy(spot_weights)

    # Loop over each NV and update weights for those with fidelity < fidelity_threshold
    for i, fidelity in enumerate(fidelities):
        if fidelity < fidelity_threshold:
            # Use a sigmoid to adjust the weight based on intensity
            updated_weights[i] = (
                1 / (1 + np.exp(-beta * (norm_intensities[i]))) ** alpha
            )

    # Normalize the updated weights to avoid extreme values
    updated_weights = updated_weights / np.max(updated_weights)

    return updated_weights


def manual_sigmoid_weight_update(
    spot_weights, intensities, alpha, beta, update_indices
):
    updated_spot_weights = (
        spot_weights.copy()
    )  # Make a copy to avoid mutating the original list
    norm_intensities = intensities / np.max(intensities)
    for idx in update_indices:
        print(f"NV Index {idx}: Weight before update: {updated_spot_weights[idx]}")

        # Apply the sigmoid weight update for the specific NV
        weight_update = 1 / (1 + np.exp(-beta * (norm_intensities[idx]))) ** alpha
        updated_spot_weights[idx] = weight_update  # Update weight for this NV

        print(f"NV Index {idx}: Weight after update: {updated_spot_weights[idx]}")

    return updated_spot_weights


# Adjust weights based on SNR values
def adjust_weights_sigmoid(spot_weights, snr_values, alpha=1.0, beta=0.001):
    """Apply sigmoid adjustment to spot weights based on SNR values."""
    updated_weights = np.copy(spot_weights)
    for i, value in enumerate(snr_values):
        if value < 0.9:
            # Sigmoid-based weight adjustment
            updated_weights[i] = 1 / (1 + np.exp(-beta * (value - alpha)))
    return updated_weights


def filter_by_peak_intensity(fitted_data, threshold=0.5):
    filtered_coords = []
    filtered_intensities = []

    for x, y, intensity in fitted_data:
        if intensity >= threshold:
            filtered_coords.append((x, y))
            filtered_intensities.append(intensity)

    return filtered_coords, filtered_intensities


def adjust_aom_voltage_for_slm(nv_amp, aom_voltage, power_law_params):
    nv_amp = np.array(nv_amp)
    a, b, c = power_law_params

    aom_voltages = nv_amp * aom_voltage

    nv_powers = a * (aom_voltages**b) + c
    scaled_nv_powers = nv_powers / (len(nv_powers))
    # Normalize powers across all spots
    total_power = np.sum(scaled_nv_powers)
    nv_weights = nv_powers / total_power
    # Compute adjusted AOM voltage for the total power
    adjusted_aom_voltage = ((total_power - c) / a) ** (1 / b)
    return nv_weights, adjusted_aom_voltage


def curve_extreme_weights_simple(weights, scaling_factor=1.0):
    median = np.median(weights)

    curved_weights = [1 / (1 + np.exp(-scaling_factor * (w - median))) for w in weights]

    return curved_weights


def curve_inverse_counts(counts, scaling_factor=0.5):
    median_count = np.median(counts)
    adjusted_weights = np.exp(-scaling_factor * (counts / median_count))
    adjusted_weights /= np.max(adjusted_weights)
    return adjusted_weights


def select_half_left_side_nvs_and_plot(nv_coordinates):
    # Filter NVs on the left side (x < median x)
    median_x = np.median(nv_coordinates[:, 0])
    left_side_indices = [
        i for i, coord in enumerate(nv_coordinates) if coord[0] < median_x
    ]

    # Randomly select half of the NVs from the left side
    print(f"Selected {len(left_side_indices)} NVs from the left side.")

    # Plot distribution
    plt.figure(figsize=(10, 7))

    # Plot all NVs
    plt.scatter(
        nv_coordinates[:, 0], nv_coordinates[:, 1], color="gray", label="All NVs"
    )

    # Highlight left-side NVs
    left_coords = nv_coordinates[left_side_indices]
    plt.scatter(
        left_coords[:, 0], left_coords[:, 1], color="blue", label="Left Side NVs"
    )

    # Add median line
    plt.axvline(
        median_x, color="green", linestyle="--", label=f"Median X = {median_x:.2f}"
    )

    # Labels and legend
    plt.title("NV Distribution with Left Side Selection", fontsize=16)
    plt.xlabel("X Coordinate", fontsize=14)
    plt.ylabel("Y Coordinate", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

    return


# Main section of the code
if __name__ == "__main__":
    kpl.init_kplotlib()
    # Parameters
    remove_outliers_flag = False  # Set this flag to enable/disable outlier removal
    reorder_coords_flag = True  # Set this flag to enable/disable reordering of NVs
    data = dm.get_raw_data(
        file_stem="2025_10_07-15_16_25-rubin-nv0_2025_09_08", load_npz=True
    )
    img_array = np.array(data["ref_img_array"])
    # img_array = data["img_array"]
    nv_coordinates, spot_weights = load_nv_coords(
        # file_path="slmsuite/nv_blob_detection/nv_blob_327nvs.npz"
        # file_path="slmsuite/nv_blob_detection/nv_blob_308nvs_reordered.npz"
        # file_path="slmsuite/nv_blob_detection/nv_blob_254nvs_reordered.npz"
        file_path="slmsuite/nv_blob_detection/nv_blob_151nvs_reordered.npz"
    )
    # Convert coordinates to a standard format (lists of lists)
    # nv_coordinates = [[coord[0] - 3, coord[1] + 3] for coord in nv_coordinates]
    nv_coordinates = [list(coord) for coord in nv_coordinates]
    # Filter NV coordinates: Keep only those where both x and y are in [0, 250]
    nv_coordinates_filtered = [
        coord
        for coord in nv_coordinates
        if isinstance(coord, (list, tuple))
        and len(coord) == 2
        and all(2 <= x <= 248 for x in coord)
    ]

    # Ensure spot weights are filtered accordingly
    spot_weights_filtered = [
        weight
        for coord, weight in zip(nv_coordinates, spot_weights)
        if isinstance(coord, (list, tuple))
        and len(coord) == 2
        and all(2 <= x <= 248 for x in coord)
    ]

    # Replace original lists with filtered versions
    nv_coordinates = nv_coordinates_filtered
    spot_weights = spot_weights_filtered

    print(f"After filtering: {len(spot_weights)} NVs")

    # Filter and reorder NV coordinates based on reference NV
    # integrated_intensities = []
    sigma = 3
    reference_nv = [124.195, 127.341]
    filtered_reordered_coords, filtered_reordered_spot_weights, include_indices = (
        filter_and_reorder_nv_coords(
            nv_coordinates, spot_weights, reference_nv, min_distance=6
        )
    )
    print(len(filtered_reordered_coords))
    # filtered_reordered_coords = [
    #     [coord[0] - 5, coord[1] - 0] for coord in filter_and_reorder_nv_coords
    # ]

    # Integration over disk region around each NV coordinate
    # filtered_reordered_counts = []
    # integration_radius = 3.0
    # for coord in filtered_reordered_coords:
    #     x, y = coord[:2]  # Assuming `coord` contains at least two elements (y, x)
    #     rr, cc = disk((y, x), integration_radius, shape=img_array.shape)
    #     sum_value = np.sum(img_array[rr, cc])
    #     filtered_reordered_counts.append(sum_value)

    # # calcualte spot weight  based on
    # calcualted_spot_weights = linear_weights(filtered_reordered_counts, alpha=0.3)
    # filtered_reordered_spot_weights = calcualted_spot_weights
    # Manually remove NVs with specified indices
    indices_to_remove = []
    filtered_reordered_coords_0 = [
        coord
        for i, coord in enumerate(filtered_reordered_coords)
        if i not in indices_to_remove
    ]
    filtered_reordered_spot_weights_0 = [
        count
        for i, count in enumerate(filtered_reordered_spot_weights)
        if i not in indices_to_remove
    ]
    filtered_reordered_coords = filtered_reordered_coords_0
    filtered_reordered_spot_weights = filtered_reordered_spot_weights_0

    # print(filtered_reordered_coords)
    # print("Filter:", filtered_reordered_counts)
    # print("Filtered and Reordered NV Coordinates:", filtered_reordered_coords)
    # print("Filtered and Reordered NV Coordinates:", integrated_intensities)

    # Initialize lists to store the results
    # fitted_amplitudes = []
    # for coord in filtered_reordered_coords:
    #     fitted_x, fitted_y, amplitude = fit_gaussian(img_array, coord, window_size=2)
    #     fitted_amplitudes.append(amplitude)

    # fmt: off
    #308NVs
    pol_duration_list = [336, 336, 308, 308, 428, 428, 504, 504, 816, 816, 528, 528, 372, 372, 1060, 1060, 852, 852, 852, 852, 612, 612, 484, 484, 1120, 1120, 852, 852, 404, 404, 812, 812, 672, 672, 560, 560, 644, 644, 352, 352, 380, 380, 852, 852, 400, 400, 620, 620, 628, 628, 292, 292, 528, 528, 392, 392, 524, 524, 680, 680, 504, 504, 396, 396, 324, 324, 428, 428, 240, 240, 504, 504, 540, 540, 852, 852, 1188, 1188, 764, 764, 976, 976, 820, 820, 444, 444, 1100, 1100, 488, 488, 604, 604, 972, 972, 380, 380, 352, 352, 660, 660, 592, 592, 416, 416, 452, 452, 620, 620, 576, 576, 316, 316, 660, 660, 660, 660, 720, 720, 620, 620, 1024, 1024, 320, 320, 852, 852, 1396, 1396, 464, 464, 416, 416, 624, 624, 1008, 1008, 460, 460, 508, 508, 668, 668, 448, 448, 440, 440, 668, 668, 852, 852, 852, 852, 844, 844, 1048, 1048, 320, 320, 780, 780, 492, 492, 1476, 1476, 656, 656, 1064, 1064, 456, 456, 344, 344, 852, 852, 540, 540, 352, 352, 524, 524, 852, 852, 1156, 1156, 1388, 1388, 308, 308, 852, 852, 1360, 1360, 572, 572, 204, 204, 316, 316, 696, 696, 504, 504, 1332, 1332, 1012, 1012, 708, 708, 852, 852, 912, 912, 804, 804, 608, 608, 948, 948, 596, 596, 1256, 1256, 808, 808, 852, 852, 392, 392, 568, 568, 872, 872, 1268, 1268, 780, 780, 852, 852, 476, 476, 508, 508, 640, 640, 392, 392, 512, 512, 700, 700, 700, 700, 932, 932, 840, 840, 852, 852, 1248, 1248, 852, 852, 852, 852, 1444, 1444, 620, 620, 852, 852, 852, 852, 660, 660, 752, 752, 1052, 1052, 592, 592, 852, 852, 852, 852, 1248, 1248, 860, 860, 520, 520, 1320, 1320, 1096, 1096, 568, 568, 488, 488, 852, 852, 556, 556, 420, 420, 1192, 1192, 552, 552, 1032, 1032, 508, 508, 1268, 1268, 872, 872, 852, 852, 852, 852, 560, 560, 328, 328, 1232, 1232, 1288, 1288, 500, 500, 356, 356, 836, 836, 852, 852, 392, 392, 940, 940, 1252, 1252, 1428, 1428, 896, 896, 1260, 1260, 1260, 1260, 852, 852, 776, 776, 796, 796, 368, 368, 1164, 1164, 1276, 1276, 1472, 1472, 448, 448, 1000, 1000, 504, 504, 1096, 1096, 612, 612, 584, 584, 660, 660, 776, 776, 684, 684, 1424, 1424, 852, 852, 416, 416, 1452, 1452, 996, 996, 668, 668, 484, 484, 364, 364, 548, 548, 472, 472, 852, 852, 1080, 1080, 852, 852, 1276, 1276, 1188, 1188, 852, 852, 852, 852, 324, 324, 1124, 1124, 300, 300, 512, 512, 884, 884, 852, 852, 1140, 1140, 852, 852, 1124, 1124, 852, 852, 1144, 1144, 852, 852, 824, 824, 852, 852, 1080, 1080, 1000, 1000, 1296, 1296, 852, 852, 1284, 1284, 852, 852, 852, 852, 1196, 1196, 432, 432, 1112, 1112, 696, 696, 400, 400, 852, 852, 852, 852, 440, 440, 852, 852, 1260, 1260, 808, 808, 572, 572, 852, 852, 772, 772, 428, 428, 940, 940, 852, 852, 480, 480, 1196, 1196, 1020, 1020, 492, 492, 1012, 1012, 852, 852, 964, 964, 1284, 1284, 852, 852, 852, 852, 852, 852, 852, 852, 820, 820, 852, 852, 944, 944, 1180, 1180, 852, 852, 528, 528, 1432, 1432, 852, 852, 976, 976, 764, 764, 1048, 1048, 852, 852, 852, 852, 852, 852, 352, 352, 852, 852, 1408, 1408, 564, 564, 852, 852, 852, 852, 1460, 1460, 1072, 1072, 548, 548, 852, 852, 688, 688, 852, 852, 852, 852, 488, 488, 1028, 1028, 540, 540, 1400, 1400, 852, 852, 852, 852, 1000, 1000, 852, 852, 892, 892, 852, 852, 852, 852, 1056, 1056, 852, 852, 1496, 1496, 852, 852, 852, 852, 1316, 1316, 1396, 1396, 1172, 1172, 852, 852, 852, 852, 852, 852, 708, 708]
    scc_duration_list = [142, 142, 142, 142, 64, 142, 142, 142, 136, 142, 142, 142, 142, 142, 142, 142, 80, 142, 142, 142, 142, 196, 88, 108, 108, 142, 142, 142, 72, 142, 142, 142, 142, 142, 64, 142, 142, 142, 142, 142, 104, 160, 84, 142, 36, 36, 92, 142, 142, 142, 48, 56, 48, 172, 142, 142, 80, 142, 142, 48, 142, 142, 142, 142, 142, 142, 142, 142, 142, 142, 72, 112, 142, 142, 142, 140, 142, 142, 52, 72, 72, 142, 142, 36, 36, 68, 142, 142, 142, 142, 36, 48, 142, 142, 142, 152, 142, 142, 104, 72, 68, 124, 112, 108, 164, 168, 142, 142, 142, 142, 142, 142, 64, 64, 142, 132, 142, 142, 72, 142, 152, 142, 164, 164, 164, 142, 142, 142, 156, 142, 142, 142, 142, 142, 142, 142, 142, 142, 124, 142, 142, 142, 64, 142, 108, 108, 142, 142, 142, 142, 142, 142, 140, 142, 142, 142, 100, 142, 142, 142, 188, 188, 76, 142, 142, 100, 142, 160, 160, 124, 142, 142, 136, 142, 142, 142, 142, 142, 142, 64, 142, 142, 142, 142, 132, 172, 56, 142, 64, 64, 196, 68, 142, 92, 92, 142, 142, 142, 142, 142, 142, 142, 48, 142, 142, 144, 142, 142, 142, 142, 142, 142, 142, 142, 196, 142, 142, 142, 76, 142, 142, 142, 64, 142, 142, 136, 136, 142, 142, 142, 142, 100, 142, 142, 142, 142, 142, 96, 142, 142, 124, 124, 124, 142, 142, 142, 142, 142, 56, 142, 142, 142, 142, 142, 88, 142, 142, 196, 142, 120, 120, 142, 142, 142, 142, 142, 142, 142, 142, 142, 142, 72, 142, 142, 142, 116, 116, 48, 36, 36, 142, 142, 142, 36, 104, 56, 40, 142, 142, 142, 142, 40, 142, 142, 142, 142, 92, 142, 192, 142, 68, 142, 142, 142, 142, 168, 142, 142]
    #254NVs
    pol_duration_list = [336, 336, 308, 308, 428, 428, 504, 528, 372, 1060, 1060, 852, 852, 852, 612, 484, 1120, 852, 404, 404, 812, 672, 672, 560, 644, 644, 352, 352, 380, 380, 852, 852, 620, 620, 628, 628, 292, 292, 528, 528, 392, 392, 524, 524, 680, 504, 396, 396, 324, 324, 428, 240, 240, 504, 540, 540, 852, 1188, 1188, 764, 764, 976, 976, 820, 820, 444, 444, 1100, 488, 488, 604, 972, 380, 380, 352, 352, 660, 592, 592, 416, 416, 452, 452, 620, 620, 576, 576, 316, 316, 660, 660, 660, 720, 720, 620, 1024, 852, 852, 1396, 1396, 464, 464, 416, 416, 624, 624, 1008, 1008, 460, 460, 508, 508, 668, 668, 448, 448, 440, 668, 668, 852, 852, 852, 852, 844, 1048, 1048, 320, 320, 780, 780, 492, 1476, 1476, 656, 656, 1064, 1064, 456, 344, 344, 852, 540, 352, 524, 852, 852, 1156, 1388, 308, 852, 852, 1360, 1360, 572, 572, 204, 696, 504, 504, 1332, 1332, 1012, 1012, 708, 708, 852, 912, 912, 804, 804, 608, 608, 948, 948, 596, 596, 1256, 1256, 808, 808, 852, 392, 392, 568, 568, 872, 872, 1268, 1268, 780, 780, 852, 852, 476, 476, 508, 508, 640, 640, 392, 392, 512, 700, 700, 700, 700, 932, 932, 840, 840, 852, 852, 1248, 1248, 852, 852, 852, 1444, 1444, 620, 620, 852, 852, 752, 752, 1052, 1052, 852, 852, 852, 852, 1248, 860, 860, 520, 520, 1320, 1320, 1096, 1096, 568, 568, 488, 488, 852, 852, 556, 556, 420, 420, 1192, 1192, 552, 552]
    scc_duration_list = [142, 142, 142, 142, 64, 142, 142, 142, 142, 142, 142, 142, 142, 142, 196, 108, 108, 142, 72, 142, 142, 142, 142, 64, 142, 142, 142, 142, 104, 160, 84, 142, 92, 142, 142, 142, 48, 56, 48, 172, 142, 142, 80, 142, 48, 142, 142, 142, 142, 142, 142, 142, 142, 72, 142, 142, 142, 142, 142, 52, 72, 72, 142, 142, 36, 36, 68, 142, 142, 142, 36, 142, 142, 152, 142, 142, 104, 68, 124, 112, 108, 164, 168, 142, 142, 142, 142, 142, 142, 64, 64, 132, 142, 142, 72, 142, 164, 142, 142, 142, 156, 142, 142, 142, 142, 142, 142, 142, 142, 142, 124, 142, 142, 142, 64, 142, 108, 142, 142, 142, 142, 142, 142, 140, 142, 142, 100, 142, 142, 142, 188, 76, 142, 142, 100, 142, 160, 124, 142, 142, 142, 142, 142, 64, 142, 142, 142, 132, 56, 64, 64, 196, 68, 142, 92, 92, 142, 142, 142, 48, 142, 142, 144, 142, 142, 142, 142, 142, 142, 142, 196, 142, 142, 142, 76, 142, 142, 142, 64, 142, 136, 136, 142, 142, 142, 142, 100, 142, 142, 142, 142, 142, 96, 142, 142, 124, 124, 124, 142, 142, 142, 142, 56, 142, 142, 142, 142, 142, 88, 142, 142, 196, 142, 120, 120, 142, 142, 142, 142, 142, 142, 142, 72, 142, 116, 116, 48, 142, 142, 142, 36, 56, 40, 142, 142, 142, 142, 40, 142, 142, 142, 142, 92, 142, 192, 142, 68, 142, 142, 142, 142, 168, 142, 142]
 
    # spot_weights = [0.9570329772823472, 0.6324213363758899, 1.2166713508521962, 0.540454796802809, 0.8404440151871955, 1.0823562017418076, 1.1483738073293428, 1.1483738073293428, 0.9570329772823472, 0.540454796802809, 1.0823562017418076, 1.2166713508521962, 1.0185866267061978, 1.2166713508521962, 0.7853434550996824, 1.0823562017418076, 0.6324213363758899, 1.2166713508521962, 1.2166713508521962, 1.1483738073293428, 1.0185866267061978, 1.2166713508521962, 0.6324213363758899, 1.2166713508521962, 0.8976629455943622, 0.7323283136331619, 0.7323283136331619, 1.1483738073293428, 1.1483738073293428, 1.2166713508521962, 0.9570329772823472, 1.2166713508521962, 0.49736426748690293, 1.2166713508521962, 1.2166713508521962, 0.540454796802809, 1.1483738073293428, 0.7853434550996824, 0.7853434550996824, 0.8404440151871955, 0.41679654445031256, 0.7853434550996824, 0.7853434550996824, 1.2166713508521962, 1.0823562017418076, 0.6813654117927326, 1.1483738073293428, 1.2166713508521962, 0.5854624326807136, 1.2166713508521962, 1.1483738073293428, 0.8976629455943622, 1.0185866267061978, 1.0185866267061978, 0.8404440151871955, 1.1483738073293428, 1.1483738073293428, 1.2166713508521962, 0.9570329772823472, 1.1483738073293428, 1.0823562017418076, 1.1483738073293428, 1.0185866267061978, 0.6813654117927326, 1.0185866267061978, 0.9570329772823472, 0.6324213363758899, 1.1483738073293428, 1.1483738073293428, 1.0185866267061978, 1.2166713508521962, 1.2166713508521962, 0.9570329772823472, 1.0823562017418076, 0.6813654117927326, 1.2166713508521962, 1.2166713508521962, 0.9570329772823472, 1.1483738073293428, 1.0823562017418076, 0.9570329772823472, 1.2166713508521962, 0.540454796802809, 0.9570329772823472, 1.2166713508521962, 1.2166713508521962, 1.1483738073293428, 1.2166713508521962, 1.1483738073293428, 1.1483738073293428, 1.0823562017418076, 0.7853434550996824, 1.0185866267061978, 1.2166713508521962, 1.0823562017418076, 1.0823562017418076, 1.0823562017418076, 0.9570329772823472, 0.8976629455943622, 0.8976629455943622, 1.1483738073293428, 1.1483738073293428, 1.2166713508521962, 1.1483738073293428, 1.1483738073293428, 0.8976629455943622, 0.9570329772823472, 0.7853434550996824, 1.2166713508521962, 1.2166713508521962, 1.2166713508521962, 1.0823562017418076, 0.9570329772823472, 1.1483738073293428, 0.9570329772823472, 1.1483738073293428, 1.0185866267061978, 1.0823562017418076, 1.0185866267061978, 1.0185866267061978, 0.8976629455943622, 1.0185866267061978, 0.9570329772823472, 0.41679654445031256, 1.1483738073293428, 0.8976629455943622, 1.2166713508521962, 1.2166713508521962, 0.5854624326807136, 0.7853434550996824, 1.0823562017418076, 1.2166713508521962, 0.6324213363758899, 0.7323283136331619, 1.2166713508521962, 1.0185866267061978, 0.7853434550996824, 1.0823562017418076, 1.0185866267061978, 0.9570329772823472, 1.1483738073293428, 1.2166713508521962, 1.0185866267061978, 1.0823562017418076, 1.0823562017418076, 1.2166713508521962, 0.8976629455943622, 1.2166713508521962, 0.7853434550996824, 1.1483738073293428, 1.0823562017418076, 0.8404440151871955, 0.8404440151871955, 1.2166713508521962, 0.540454796802809, 1.2166713508521962, 0.9570329772823472, 0.540454796802809, 1.2166713508521962, 1.1483738073293428, 0.9570329772823472, 0.49736426748690293, 1.2166713508521962, 1.0823562017418076, 1.2166713508521962, 0.7853434550996824, 0.8976629455943622, 1.0823562017418076, 1.1483738073293428, 1.2166713508521962, 0.6324213363758899, 1.1483738073293428, 1.1483738073293428, 1.1483738073293428, 1.2166713508521962, 1.0185866267061978, 0.9570329772823472, 1.1483738073293428, 1.0185866267061978, 1.2166713508521962, 0.8404440151871955, 1.1483738073293428, 0.4561564174952019, 1.1483738073293428, 1.2166713508521962, 0.9570329772823472, 0.6813654117927326, 0.8404440151871955, 0.49736426748690293, 0.8404440151871955, 1.1483738073293428, 1.1483738073293428, 1.0185866267061978, 1.1483738073293428, 0.9570329772823472, 0.9570329772823472, 0.9570329772823472, 0.8976629455943622, 0.7853434550996824, 0.4561564174952019, 0.49736426748690293, 1.1483738073293428, 0.7853434550996824, 0.6813654117927326, 0.8976629455943622, 0.49736426748690293, 1.1483738073293428, 1.0185866267061978, 1.0185866267061978, 1.0823562017418076, 0.6324213363758899, 1.0823562017418076, 0.8976629455943622, 1.1483738073293428, 1.1483738073293428, 1.2166713508521962, 0.7853434550996824, 1.1483738073293428, 1.2166713508521962, 1.2166713508521962, 1.0185866267061978, 1.0185866267061978, 1.1483738073293428, 1.2166713508521962, 1.2166713508521962, 0.41679654445031256, 1.2166713508521962, 1.1483738073293428, 1.1483738073293428, 1.2166713508521962, 1.2166713508521962, 1.0185866267061978, 1.0185866267061978, 0.49736426748690293, 1.0185866267061978, 1.2166713508521962, 1.2166713508521962, 0.41679654445031256, 0.8976629455943622, 1.1483738073293428, 1.0185866267061978, 0.9570329772823472, 1.1483738073293428, 1.0185866267061978, 0.9570329772823472, 0.9570329772823472, 1.1483738073293428, 1.2166713508521962, 1.0823562017418076, 1.1483738073293428, 1.1483738073293428, 1.2166713508521962, 1.2166713508521962, 0.8404440151871955, 0.49736426748690293, 0.6813654117927326, 0.7323283136331619, 1.0823562017418076, 0.8976629455943622, 1.1483738073293428, 1.0823562017418076, 1.1483738073293428, 0.49736426748690293, 0.7853434550996824, 1.1483738073293428, 0.8404440151871955, 0.8404440151871955, 1.1483738073293428, 1.0823562017418076, 1.2166713508521962, 0.7323283136331619, 1.1483738073293428, 1.2166713508521962, 0.7853434550996824, 1.1483738073293428, 1.0823562017418076, 0.9570329772823472, 1.2166713508521962, 0.41679654445031256, 1.1483738073293428, 1.1483738073293428, 1.0185866267061978, 1.2166713508521962, 1.0185866267061978, 1.0185866267061978, 1.0185866267061978, 0.6324213363758899, 1.1483738073293428, 1.1483738073293428, 1.0823562017418076, 0.6324213363758899, 1.0823562017418076, 1.1483738073293428, 1.2166713508521962, 0.9570329772823472, 0.41679654445031256, 0.540454796802809, 1.0823562017418076, 1.0185866267061978, 1.2166713508521962, 1.2166713508521962, 1.0823562017418076, 1.2166713508521962, 0.9570329772823472, 0.9570329772823472, 1.2166713508521962, 1.2166713508521962, 1.2166713508521962]
    # spot_weights = [0.7575309157761969, 1.429510264972462, 0.6517549399458059, 1.429510264972462, 1.130885325349257, 0.8138242955463341, 0.8138242955463341, 1.2750267318102078, 1.3509573548979483, 1.2750267318102078, 0.5547985537038624, 0.7035207955157136, 1.201681697203128, 1.130885325349257, 1.201681697203128, 1.062600457113157, 0.5547985537038624, 1.130885325349257, 0.7035207955157136, 1.130885325349257, 1.201681697203128, 0.9334153886415295, 1.429510264972462, 0.6021940582534366, 0.7575309157761969, 1.201681697203128, 1.429510264972462, 0.9967896934605812, 0.8724396430231868, 1.201681697203128, 1.2750267318102078, 1.130885325349257, 1.062600457113157, 1.062600457113157, 0.9334153886415295, 0.9967896934605812, 1.062600457113157, 1.429510264972462, 0.5547985537038624, 0.7575309157761969, 0.6021940582534366, 1.2750267318102078, 1.429510264972462, 0.8724396430231868, 1.130885325349257, 0.7035207955157136, 0.7575309157761969, 0.6021940582534366, 1.3509573548979483, 1.2750267318102078, 0.8724396430231868, 1.201681697203128, 0.9967896934605812, 0.8724396430231868, 0.8724396430231868, 1.429510264972462, 1.2750267318102078, 1.130885325349257, 1.429510264972462, 0.7575309157761969, 1.429510264972462, 0.8724396430231868, 1.2750267318102078, 1.201681697203128, 1.429510264972462, 1.2750267318102078, 1.2750267318102078, 1.2750267318102078, 1.130885325349257, 0.6517549399458059, 0.9334153886415295, 1.201681697203128, 0.6517549399458059, 1.130885325349257, 1.2750267318102078, 0.8724396430231868, 0.8138242955463341, 1.3509573548979483, 1.2750267318102078, 1.201681697203128, 1.3509573548979483, 1.201681697203128, 1.2750267318102078, 0.7035207955157136, 1.130885325349257, 0.5547985537038624, 0.5095285131085532, 1.130885325349257, 1.130885325349257, 0.6021940582534366, 0.9967896934605812, 0.7035207955157136, 0.9334153886415295, 1.201681697203128, 1.130885325349257, 1.429510264972462, 0.5095285131085532, 0.8724396430231868, 0.9967896934605812, 0.6021940582534366, 0.9967896934605812, 0.7575309157761969, 1.2750267318102078, 0.8138242955463341, 0.9967896934605812, 1.3509573548979483, 0.9967896934605812, 1.429510264972462, 0.7575309157761969, 0.9334153886415295, 1.2750267318102078, 1.429510264972462, 0.5547985537038624, 1.201681697203128, 0.9334153886415295, 0.5547985537038624, 1.2750267318102078, 1.201681697203128, 1.130885325349257, 0.9334153886415295, 0.5547985537038624, 0.5547985537038624, 1.130885325349257, 0.9334153886415295, 1.2750267318102078, 0.8138242955463341, 0.8138242955463341, 1.201681697203128, 1.429510264972462, 0.8724396430231868, 1.201681697203128, 0.5547985537038624, 1.429510264972462, 1.2750267318102078, 0.8138242955463341, 0.9334153886415295, 0.7035207955157136, 0.5095285131085532, 0.7035207955157136, 1.3509573548979483, 0.6021940582534366, 1.3509573548979483, 0.9334153886415295, 0.7035207955157136, 0.9334153886415295, 0.8138242955463341, 0.7035207955157136, 0.7035207955157136, 1.429510264972462, 1.429510264972462, 0.7035207955157136, 0.7035207955157136, 1.3509573548979483, 1.130885325349257, 0.6021940582534366, 0.5547985537038624, 1.2750267318102078, 0.8724396430231868, 1.062600457113157, 0.5095285131085532, 0.9967896934605812, 0.6517549399458059, 0.8724396430231868, 0.9967896934605812, 0.9334153886415295, 0.7035207955157136, 0.8138242955463341, 1.062600457113157, 1.2750267318102078, 1.3509573548979483, 1.130885325349257, 1.201681697203128, 1.429510264972462, 1.130885325349257, 1.429510264972462, 0.8138242955463341, 1.2750267318102078, 1.3509573548979483, 1.130885325349257, 1.3509573548979483, 0.7575309157761969, 0.6517549399458059, 1.3509573548979483, 1.429510264972462, 0.9967896934605812, 0.5547985537038624, 1.3509573548979483, 1.2750267318102078, 0.8724396430231868, 1.201681697203128, 0.8724396430231868, 0.7035207955157136, 0.5095285131085532, 1.201681697203128, 1.201681697203128, 0.6021940582534366, 1.062600457113157, 1.062600457113157, 0.8138242955463341, 1.2750267318102078, 0.6021940582534366, 1.429510264972462, 1.130885325349257, 0.7035207955157136, 0.5547985537038624, 0.9334153886415295, 0.9334153886415295, 0.8724396430231868, 0.7035207955157136, 0.8724396430231868, 0.9334153886415295, 1.3509573548979483, 1.429510264972462, 1.062600457113157, 1.2750267318102078, 1.3509573548979483, 0.6021940582534366, 0.7035207955157136, 1.3509573548979483, 0.5547985537038624, 0.5547985537038624, 1.2750267318102078, 1.130885325349257, 0.5095285131085532, 0.7575309157761969, 0.9967896934605812, 0.8138242955463341, 0.6021940582534366, 0.6517549399458059, 1.2750267318102078, 0.5547985537038624, 0.9334153886415295, 0.6517549399458059, 0.8724396430231868, 1.062600457113157, 0.6517549399458059, 0.8138242955463341, 0.7575309157761969, 1.130885325349257, 1.3509573548979483, 0.7575309157761969, 1.429510264972462, 0.9967896934605812, 1.429510264972462, 0.7035207955157136, 1.2750267318102078, 0.9967896934605812, 0.9967896934605812, 1.062600457113157, 0.7035207955157136, 0.7035207955157136, 0.6517549399458059, 0.9334153886415295, 1.2750267318102078, 1.201681697203128, 0.9967896934605812, 0.6517549399458059, 0.6021940582534366, 0.8138242955463341, 0.8138242955463341, 0.8724396430231868, 1.2750267318102078, 0.7035207955157136, 0.6021940582534366, 0.8138242955463341, 0.8724396430231868, 0.9967896934605812, 0.7035207955157136, 1.201681697203128, 1.3509573548979483, 1.3509573548979483, 1.201681697203128, 0.9967896934605812, 1.062600457113157, 0.8724396430231868, 0.6517549399458059, 0.8138242955463341, 0.9967896934605812, 0.9967896934605812, 0.5547985537038624, 0.9334153886415295, 0.7575309157761969, 0.7575309157761969, 0.8138242955463341, 1.429510264972462, 1.3509573548979483, 1.3509573548979483, 0.9967896934605812, 1.2750267318102078, 0.6517549399458059, 0.7035207955157136, 0.8138242955463341, 1.130885325349257, 0.6517549399458059, 1.3509573548979483, 0.6021940582534366, 1.429510264972462, 1.062600457113157, 1.130885325349257, 0.8724396430231868, 0.8724396430231868, 1.429510264972462, 1.429510264972462, 1.429510264972462, 1.429510264972462, 0.8138242955463341, 1.3509573548979483, 1.201681697203128]
    readout_fidelity_list = [0.8439059562261625, 0.9092516766755109, 0.83634436194217, 0.8707938551100962, 0.827787147977106, 0.6795621762395407, 0.846493111572296, 0.8041797866477087, 0.8309415714294462, 0.8165074457725303, 0.8610397547410755, 0.8328701985190865, 0.8360119079022252, 0.7382406342667385, 0.8192839008699937, 0.8888361051273321, 0.7414023425150913, 0.8644016682421367, 0.8689478407500897, 0.8158293776556018, 0.7042994973744343, 0.8211549506182853, 0.8473070257603573, 0.9085554706296406, 0.8324085922616652, 0.7498896595789045, 0.8481256096156068, 0.840096371327024, 0.9468243588074932, 0.9047347579934848, 0.7334974232436706, 0.8894864610506845, 0.9170790791745599, 0.8173236167414496, 0.9601368075851286, 0.8568407528341493, 0.9085744730240701, 0.9455281566763, 0.7095446072520907, 0.7227589576855181, 0.8933403419919588, 0.8807374041336493, 0.8442881973211361, 0.8015952558892406, 0.8397046497077106, 0.8594182715898053, 0.8805876055145699, 0.7682628283879553, 0.9372629757015056, 0.8665772293631172, 0.6753956879691871, 0.7960447370936736, 0.8818515836163718, 0.6556022021095937, 0.8853912039674572, 0.9134052914395794, 0.8082054456260116, 0.883630567820449, 0.7635895203554444, 0.8104380079862179, 0.9135245918131436, 0.851623910060725, 0.9159970770003365, 0.7953620188835608, 0.8340772261402369, 0.9094409268376751, 0.9606014290687352, 0.8134180521555835, 0.8800426043319532, 0.7143354254208665, 0.8623162753776119, 0.8833319686716878, 0.8564883217067567, 0.9258761194074335, 0.8519537144588103, 0.9273340793356175, 0.832957718426953, 0.9302156811129381, 0.8908678185022101, 0.8945626404162732, 0.7861950816694436, 0.8655149020876061, 0.7440241593251659, 0.6715824927189824, 0.8214613381675588, 0.82111480100841, 0.8977351393883233, 0.7974600181380282, 0.7926393685125018, 0.6678457554253023, 0.6871613058073955, 0.7862167617963615, 0.7890232161675668, 0.8634413760457884, 0.7523360163126428, 0.8384125234823413, 0.7986171518297067, 0.8504695093666119, 0.8754746917717782, 0.933702686474571, 0.8699848290082262, 0.903531508374223, 0.8074042043531022, 0.8427427373291523, 0.6299725239795916, 0.9306270844434006, 0.7827284646396486, 0.747311941413062, 0.9217602590995679, 0.7917557423654791, 0.8937636805555005, 0.8413182492695681, 0.7196514446512327, 0.8334694156818139, 0.8341259023056063, 0.7740417457309892, 0.7800788845931144, 0.7764647756846802, 0.8227415160378531, 0.6951120069719291, 0.8283086089268121, 0.7767201111366027, 0.6093404236883638, 0.7927219255586185, 0.7956282576294438, 0.7356850022188652, 0.8012319686429621, 0.7851070690566277, 0.8889521686225141, 0.8216635247692663, 0.8428865699939609, 0.7642585690522736, 0.6745099349208614, 0.9079303610370669, 0.8802580129185784, 0.888061162887289, 0.7466716658066535, 0.8196724491172587, 0.8469678816375075, 0.9322243085474935, 0.8707841028953507, 0.6977752765109828, 0.766282863756687, 0.8307021256089109, 0.7939501741727019, 0.8379844998132839, 0.786783284197331, 0.8873157532045342, 0.8305693513682948, 0.8636678846744668, 0.9117681451024926, 0.8235297194564768, 0.889173429652443, 0.8388071330859509, 0.8386524618743856, 0.7870359912170275, 0.9469720427546674, 0.9079920955542512, 0.8280123053733557, 0.943718115142337, 0.6913230176566288, 0.9016982892862869, 0.8533639687027277, 0.7756408332536411, 0.8433932866353974, 0.8562151908353284, 0.8861458631203407, 0.6983490357717914, 0.892866572802159, 0.8869150795427004, 0.9238852294347414, 0.8552573414488992, 0.822465050704539, 0.874120324650245, 0.9604453185449038, 0.9065286512228254, 0.8477816186248177, 0.9498837491099068, 0.923520913209408, 0.817270422487357, 0.8189854905115936, 0.7857395688290298, 0.7052935863218135, 0.8719895605316077, 0.8133930324632948, 0.8695317663359939, 0.9503741374575527, 0.8455554465653841, 0.8060255102027347, 0.8842257036390702, 0.7511938596331651, 0.8150796872015591, 0.7212549059298221, 0.8227027801464071, 0.7497407863049288, 0.883271162846587, 0.8197276075544457, 0.7729543855265983, 0.8569157898231241, 0.9318299608576697, 0.7639075974739191, 0.8132240063984888, 0.7935293852911943, 0.7849229164518408, 0.7417251998777201, 0.7934208701040113, 0.7260933675761322, 0.6836805008186106, 0.6890740151193439, 0.7696307115565337, 0.8452811948045122, 0.8238744444924848, 0.7503059634450235, 0.8971329939753709, 0.6783649313302331, 0.7713679105235205, 0.6671265549188841, 0.8308592849825557, 0.9042362120145664, 0.8456418330231599, 0.7602305215183252, 0.7774040660763271, 0.8336218989824515, 0.8876357362335492, 0.8553135890735359, 0.9387722327780835, 0.8035997617384467, 0.78034820451149, 0.7497224129196838, 0.8047897539914357, 0.7267829374477748, 0.862207442130311, 0.7943689106382995, 0.8483711257751093, 0.884506050555706, 0.7904556018811926, 0.767269451980138, 0.7904059797554965, 0.86495953451807, 0.7587091180080967, 0.8757513536167029, 0.7924722046992426, 0.824990098067957, 0.787557501901105, 0.8862583276848568, 0.880105358422536, 0.7067164842790965, 0.8973098424412447, 0.7642502970366717, 0.7737987068753585, 0.8166195793675317, 0.8362266055390203, 0.8209328231383217, 0.7399179025863674, 0.8447612467821835, 0.8486693626246639, 0.8645443028904113, 0.872881982453072, 0.9299619764510627, 0.6734022990729245, 0.8084681531084563, 0.8721132766721423, 0.9017309599281076, 0.8304229408271588, 0.7953651680695478, 0.7419919787514773, 0.884472244621832, 0.8851449672720411, 0.8676946173712492, 0.8848814757859875, 0.8955151357665765, 0.8784914905632037, 0.7699013336443894, 0.9021100962803654, 0.7970510921844141, 0.920559910212773, 0.8254983531094995, 0.7709836578516909, 0.8030231183189858, 0.7379369912698307, 0.8597690659879869, 0.7940505587989988, 0.7019766806687832, 0.7879788401696246, 0.8812788068731174, 0.759138675053101, 0.7506597788258432, 0.8681532634910625, 0.8856714311775515, 0.8011265077202014, 0.8351471676023916, 0.7627506919416979, 0.7228369267415855, 0.8687289357780981, 0.8497201515132289, 0.8442586417178991, 0.7506177761613693, 0.922360608474637, 0.7645892359714783]
    prep_fidelity_list = [0.6407601329396244, 0.538833877747348, 0.7466120017935529, 0.729950887654123, 0.697352905902533, 0.7376695182921822, 0.6546699635771233, 0.6132468326381302, 0.579808924887143, 0.6999984619912284, 0.6620315893370687, 0.6597404152731849, 0.6652872418179596, 0.572951229915774, 0.5951134263805575, 0.6114952659310944, 0.6911464138516635, 0.5540692991367563, 0.583737523875151, 0.8170498217739275, 0.7386770262649247, 0.722389484386739, 0.7052300676656993, 0.7137140961902765, 0.5763689340177016, 0.7839908307280843, 0.6775603720668653, 0.5527216425975862, 0.5970838592486301, 0.7533939047779763, 0.7014860813863226, 0.7213218052962784, 0.5614863823033583, 0.6216081147215924, 0.4392874017218047, 0.5444882695388847, 0.577717459410563, 0.5777689417335348, 0.5220849875297934, 0.6433241708363584, 0.5488285202974947, 0.42369014487861745, 0.6117968579018744, 0.36824366847676626, 0.6006060669547335, 0.7663491264495194, 0.5615173080058697, 0.5640576487891742, 0.4844007557980231, 0.5861025199594301, 0.6417907027760759, 0.5921768784565269, 0.5520179308098809, 0.6533303434837003, 0.5679942009626251, 0.5895859106982345, 0.7, 0.7656965197170964, 0.6634921119665942, 0.611783426107181, 0.4429426078989367, 0.5298939519242443, 0.7051837269142827, 0.6387103341007624, 0.6930430603632405, 0.4823085678639252, 0.6987292003594459, 0.6482322881549569, 0.5886002411626927, 0.6942211052762463, 0.40918142219770426, 0.6748518439470833, 0.7227576307653494, 0.5008069037873728, 0.7038862248354654, 0.7354059679785112, 0.6529126396824776, 0.6483375456088886, 0.5448445091814911, 0.8150206920753063, 0.6770788333765851, 0.6270857270000706, 0.647406145061295, 0.5382968347841475, 0.7074546513272533, 0.641555141682572, 0.6690243847019317, 0.5990394974997127, 0.6161155439430547, 0.5232520486435814, 0.8163585385156019, 0.7433008204025919, 0.5433570833539891, 0.7, 0.5526199749156968, 0.593999429926944, 0.5336832456732641, 0.6153427008392094, 0.43595637426141653, 0.44414670844291004, 0.6389352431939969, 0.6856977162998811, 0.5863688368918758, 0.7160458705728434, 0.6235887105129726, 0.6705150440050838, 0.7730156892514259, 0.6747036822158731, 0.5490104816637816, 0.3968874092660123, 0.9463528770420643, 0.587004211344306, 0.6887735947946554, 0.6584403099258103, 0.7773346240182575, 0.611820200388959, 0.6851293870056161, 0.518998940466318, 0.5635996148392914, 0.6561382190626939, 0.6279452628560842, 0.6579203136798359, 0.6748042422107434, 0.6731681420889679, 0.4743363931001596, 0.4697428090499034, 0.6146349515908291, 0.6734025670614401, 0.6042906101559419, 0.6749664122903162, 0.5631545260060975, 0.6426720640215954, 0.7, 0.6531186963479778, 0.6588895714145787, 0.6498250120383813, 0.6848316055955831, 0.7405426439968017, 0.6573747986505594, 0.5320081813048763, 0.5200218313310676, 0.5591797873230663, 0.5378191752702794, 0.6443668356303153, 0.656115864484368, 0.6977881154182, 0.6346001345104413, 0.6299223226649175, 0.6353697027463019, 0.49106570920986603, 0.6300454201602531, 0.6472323377480453, 0.6515091326556048, 0.6101227938442129, 0.6427935471680872, 0.692005291575172, 0.6140898185664839, 0.6672551991160236, 0.3903851956658204, 0.69504954021056, 0.6318568667235835, 0.6295995553462752, 0.7099771936975394, 0.5989968212487382, 0.6008050472021178, 0.6376709890239276, 0.5182387050920214, 0.5847385709606442, 0.43851800442312994, 0.7475526459815092, 0.6229051688129418, 0.7867235012861452, 0.6685722011503049, 0.7358378366388513, 0.7559822931830524, 0.47015203529613847, 0.6815444968656943, 0.7040645903262697, 0.576603130521228, 0.6675924508113993, 0.6148237933683376, 0.34576041212573505, 0.5387874026839201, 0.5571052956980729, 0.5468002217306241, 0.5749734511857139, 0.5376236954606632, 0.6051812815149482, 0.5134448504292679, 0.7809197582170688, 0.5037030437819461, 0.6352903502973124, 0.5293707316819296, 0.5589229970026053, 0.5612043688560822, 0.594026365770213, 0.35188237566025427, 0.7399120822893599, 0.3926849029398286, 0.5264758596766845, 0.7680975968579034, 0.5279241219113744, 0.6682564086528049, 0.7840798783665741, 0.5587899262978651, 0.6688910018446126, 0.6374877828975651, 0.542729900233669, 0.7312219317234556, 0.6635552142631784, 0.5779985688384093, 0.5572457479570592, 0.5714119319376247, 0.5897115318667888, 0.6309913451482356, 0.7078084464895129, 0.5908557493388251, 0.6096934209742964, 0.6920440802564993, 0.5327763180866518, 0.5095190021157817, 0.5228810372240021, 0.6592137213111002, 0.6326545836620499, 0.5704272354690394, 0.7680522797187016, 0.6721005226204627, 0.5568200691891765, 0.7532179967105745, 0.5002964283859377, 0.7741235092107822, 0.6675218976812355, 0.6511292684811916, 0.6580503751429292, 0.5429432565511725, 0.6994118539372571, 0.543615171619065, 0.6716229463239642, 0.5189182576183067, 0.5230229269149442, 0.6795235607736613, 0.37609303802325567, 0.5800910416045467, 0.5925428620146602, 0.5574424725119095, 0.6861697930047335, 0.43632073395555326, 0.726431740930428, 0.5888058725888705, 0.4769365290017267, 0.4390841729916787, 0.6128643435721605, 0.674286294239103, 0.5960898578264615, 0.5652236711571212, 0.6042222046081609, 0.679586403280342, 0.48804842043689256, 0.6086931939162232, 0.5018260139569402, 0.6118910599693197, 0.6072435378421215, 0.6442590075526455, 0.6724832143790572, 0.6662994032034222, 0.40707988506536064, 0.4727642507740548, 0.49976556551385587, 0.6366246887866285, 0.5456827811214051, 0.5072981091682227, 0.48863438365828415, 0.6405285950221367, 0.629124080492995, 0.7465126661021009, 0.49255766682293634, 0.5258365164990046, 0.743242640344814, 0.6184614931985257, 0.5638852273267185, 0.6160283294215385, 0.5582778721006247, 0.5665784460021457, 0.6302099231367273, 0.636511231644459, 0.5897696554028605, 0.5022932595589811, 0.4926244711628113, 0.38798267356929517, 0.5656770774479482, 0.7539234671964767, 0.5410036225739419, 0.4384410551215202, 0.6934994294402257, 0.6441454598149815, 0.4202171832167332, 0.5412647395892292, 0.6121825932597593, 0.6427120853740056, 0.7035936130376136, 0.6324776458987109, 0.7050525502484414, 0.5850362171172101, 0.6577418769537399, 0.6916314355950108, 0.7209046699934817, 0.586813682523665, 0.5250139815328914, 0.49999985526788304, 0.6383712930688102, 0.6579288072071785, 0.49071896321015107, 0.3918278780197474, 0.6606808836845484, 0.5828449962339644, 0.5377330852406832, 0.49150958219193863, 0.6709686081652605, 0.4732990908297705, 0.5956282153557184, 0.7440279666747744, 0.66843159105784, 0.5248543510106787, 0.5637105515139142, 0.4197269156995933, 0.4439782185375183, 0.32283042224210545, 0.5657900958430006]

    include_indices = [
        i
        for i, (v1, v2) in enumerate(zip(readout_fidelity_list, prep_fidelity_list))
        if (v1 is not None and v2 is not None)
        and all(isinstance(x, (int, float)) for x in (v1, v2))
        and not (math.isnan(v1) or math.isnan(v2))
        and v1 >= 0.7 and v2 >= 0.2
    ]
    # include_indices = [0, 1, 2, 3, 4, 5, 6, 10, 12, 14, 15, 17, 18, 19, 21, 23, 24, 26, 28, 29, 31, 32, 33, 34, 36, 37, 38, 39, 40, 41, 42, 43, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 62, 63, 64, 65, 66, 68, 69, 70, 72, 73, 74, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 87, 88, 89, 90, 92, 94, 95, 96, 97, 98, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 115, 116, 117, 118, 121, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 145, 146, 147, 148, 149, 150, 151, 152, 154, 155, 156, 157, 158, 159, 160, 162, 163, 164, 165, 166, 167, 169, 170, 171, 173, 174, 177, 179, 180, 181, 182, 184, 186, 188, 189, 190, 191, 192, 193, 194, 198, 200, 201, 202, 203, 204, 205, 206, 207, 208, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 264, 265, 266, 267, 270, 271, 274, 275, 276, 277, 280, 281, 282, 283, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307]
    ## 2 oruen
    # include_indices = [0, 1, 2, 4, 9, 10, 12, 14, 15, 17, 18, 20, 22, 23, 24, 27, 28, 29, 30, 33, 34, 37, 40, 41, 42, 45, 46, 49, 50, 51, 54, 55, 57, 58, 59, 60, 62, 65, 67, 68, 69, 72, 73, 75, 78, 79, 80, 81, 82, 85, 86, 87, 88, 90, 94, 95, 98, 99, 101, 102, 104, 106, 107, 111, 113, 114, 116, 117, 119, 122, 123, 125, 126, 127, 128, 130, 131, 133, 134, 135, 136, 137, 142, 143, 144, 145, 146, 148, 149, 151, 153, 155, 158, 161, 163, 164, 165, 166, 167, 170, 172, 173, 174, 175, 178, 181, 183, 185, 186, 187, 191, 192, 193, 195, 196, 197, 199, 200, 201, 203, 205, 207, 210, 211, 212, 214, 216, 218, 220, 221, 223, 225, 226, 227, 228, 229, 230, 233, 235, 237, 238, 239, 242, 244, 245, 246, 247, 249, 250, 252, 253]
    # include_indices = [i for i, val in enumerate(prep_fidelity_list) if val >= 0.4 or val is None]
    # import math
    # include_indices = [
    #     i for i, val in enumerate(prep_fidelity_list)
    #     if (val is None) or (isinstance(val, (int, float)) and not math.isnan(val) and val >= 0.5)
    # ]

    # print(np.sort(list(include_indices)))

    indices = np.sort(list(include_indices))
    print(", ".join(str(i) for i in indices))
    # print("[" + ", ".join(map(str, np.sort(list(include_indices)))) + "]")

    # include_indices = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 40, 41, 43, 46, 48, 49, 50, 51, 53, 55, 58, 60, 61, 62, 63, 64, 65, 68, 69, 70, 71, 73, 76, 77, 78, 79, 80, 82, 83, 88, 90, 91, 93, 94, 96, 98, 99, 102, 103, 104, 105, 107, 108, 109, 110, 113, 114, 115, 116, 117, 118, 121, 123, 125, 126, 127, 128]
    # nv_powers = [val for ind, val in enumerate(nv_powers) if ind not in drop_indices]
    # print(len(include_indices))
    # fmt: on
    # filtered_reordered_coords = [filtered_reordered_coords[i] for i in include_indices]
    # print(f"len filtered_reordered_coords: {len(filtered_reordered_coords)}")
    # # # select_half_left_side_nvs_and_plot(nv_coordinates_filtered)
    # spot_weights_filtered = np.array(
    #     [weight for i, weight in enumerate(spot_weights) if i in include_indices]
    # )
    # filtered_pol_durs = [pol_duration_list[i] for i in include_indices]
    # filtered_scc_durs = [scc_duration_list[i] for i in include_indices]
    # print(filtered_pol_durs)
    # print(filtered_scc_durs)
    # sys.exit()
    # print(len(spot_weights_filtered))
    # # sys.exit()
    # aom_voltage = 0.2861
    aom_voltage = 0.2675 # 244 nvs 
    # a, b, c = [3.7e5, 6.97, 8e-14]
    # a, b, c = 161266.751, 6.617, -19.492
    a, b, c = 1.5133e04, 2.6976, -38.63  # UPDATED 2025-09-17

    total_power = a * (aom_voltage) ** b + c
    # print(total_power)
    normalized_spot_weigths = spot_weights / np.sum(spot_weights)
    nv_powers = total_power * normalized_spot_weigths
    # print(nv_powers)
    # print(nv_powers)
    # calcualted_spot_weights = linear_weights(filtered_reordered_counts, alpha=0.3)
    # updated_spot_weights = linear_weights(filtered_reordered_counts, alpha=0.6)
    nv_powers_filtered = np.array(
        [power for i, power in enumerate(nv_powers) if i in include_indices]
    )
    print(nv_powers_filtered)
    # Create a copy or initialize spot weights for modification
    # updated_spot_weights = curve_extreme_weights_simple(
    #     spot_weights, scaling_factor=1.0
    # )
    # filtered_reordered_spot_weights = np.array(
    #     [
    #         weight
    #         for i, weight in enumerate(updated_spot_weights)
    #         if i in include_indices
    #     ]
    # )
    # updated_spot_weights = spot_weights
    # print(filter_and_reorder_nv_coords)
    # updated_spot_weights = np.array(
    #     [w for i, w in enumerate(updated_spot_weights_0) if i in include_indices]
    # )
    # updated_spot_weights = spot_weights
    # updated_spot_weights = curve_extreme_weights_simple(nv_powers)
    # updated_spot_weights = curve_inverse_counts(filtered_reordered_spot_weights)
    # drop_indices = [150, 161, 392, 403]
    # updated_spot_weights = [
    #     val for ind, val in enumerate(updated_spot_weights) if ind not in drop_indices
    # ]
    # filtered_reordered_coords = [
    #     filtered_reordered_coords[ind]
    #     for ind in range(len(filtered_reordered_coords))
    #     if ind not in drop_indices
    # ]
    # nv_powers = [val for ind, val in enumerate(nv_powers) if ind not in drop_indices]

    ### new weithgs for 75 NVs
    # fmt: off
    # spot_weights = [1.5095766644650288, 0.9618852387815993, 1.0430775922714643, 0.4436039796130322, 1.0430775922714643, 1.5095766644650288, 0.30376599435099677, 1.310540492774533, 1.4079312952605532, 0.6747045661156417, 0.4964211566919598, 0.7410295032188732, 1.0430775922714643, 1.0430775922714643, 0.6118852419676627, 0.552486087697023, 0.552486087697023, 0.3939475438077535, 0.7410295032188732, 0.9618852387815993, 0.8845355774031071, 0.3939475438077535, 0.8109450107566616, 1.1281958109476427, 1.5095766644650288, 1.4079312952605532, 1.4079312952605532, 0.552486087697023, 0.4436039796130322, 0.3939475438077535, 1.2173226567254087, 0.7410295032188732, 0.8845355774031071, 0.552486087697023, 1.0430775922714643, 0.4964211566919598, 1.5095766644650288, 1.4079312952605532, 0.6747045661156417, 0.6118852419676627, 1.5095766644650288, 0.6118852419676627, 0.3939475438077535, 1.1281958109476427, 0.8845355774031071, 1.2173226567254087, 1.4079312952605532, 0.6118852419676627, 1.310540492774533, 1.5095766644650288, 1.1281958109476427, 1.310540492774533, 0.7410295032188732, 1.310540492774533, 1.5095766644650288, 0.8845355774031071, 1.5095766644650288, 0.7410295032188732, 0.9618852387815993, 0.3473642711816343, 0.6747045661156417, 0.6747045661156417, 1.310540492774533, 0.6118852419676627, 0.6118852419676627, 1.4079312952605532, 0.3939475438077535, 0.6118852419676627, 1.2173226567254087, 1.4079312952605532, 1.4079312952605532, 0.6747045661156417, 1.310540492774533, 1.0430775922714643, 1.0430775922714643, 1.0430775922714643, 1.4079312952605532, 1.2173226567254087, 1.4079312952605532, 0.3473642711816343, 1.2173226567254087, 0.8109450107566616, 0.9618852387815993, 1.5095766644650288, 1.4079312952605532, 1.2173226567254087, 0.8109450107566616, 1.310540492774533, 0.552486087697023, 1.4079312952605532, 1.0430775922714643, 0.7410295032188732, 0.4964211566919598, 0.7410295032188732, 0.6747045661156417, 1.310540492774533, 0.9618852387815993, 1.5095766644650288, 0.8845355774031071, 0.8109450107566616, 1.310540492774533, 1.4079312952605532, 0.8109450107566616, 1.310540492774533, 1.1281958109476427, 1.4079312952605532, 1.2173226567254087, 0.8845355774031071, 0.7410295032188732, 0.6118852419676627, 1.1281958109476427, 0.8109450107566616, 1.1281958109476427, 0.7410295032188732, 1.1281958109476427, 1.5095766644650288, 0.6118852419676627, 1.5095766644650288, 1.2173226567254087, 0.7410295032188732, 0.6118852419676627, 0.7410295032188732, 1.310540492774533, 1.0430775922714643, 1.4079312952605532, 0.7410295032188732, 1.2173226567254087, 1.5095766644650288, 1.310540492774533, 0.9618852387815993, 1.310540492774533, 0.6747045661156417, 0.7410295032188732, 1.0430775922714643, 1.4079312952605532, 0.6747045661156417, 0.3939475438077535, 0.6118852419676627, 0.7410295032188732, 1.310540492774533, 1.5095766644650288, 1.2173226567254087, 1.2173226567254087, 0.9618852387815993, 0.6747045661156417, 1.2173226567254087, 1.2173226567254087, 0.4964211566919598, 1.2173226567254087, 0.6118852419676627, 0.8109450107566616, 1.0430775922714643, 1.4079312952605532, 0.3939475438077535, 0.7410295032188732, 1.4079312952605532, 0.9618852387815993, 1.310540492774533, 1.1281958109476427, 1.4079312952605532, 1.5095766644650288, 1.0430775922714643, 0.7410295032188732, 1.4079312952605532, 1.5095766644650288, 1.4079312952605532, 1.4079312952605532, 0.552486087697023, 0.3939475438077535, 0.8845355774031071, 0.552486087697023, 0.6118852419676627, 0.6747045661156417, 0.6747045661156417, 0.552486087697023, 1.4079312952605532, 1.5095766644650288, 1.4079312952605532, 1.1281958109476427, 0.8109450107566616, 0.8845355774031071, 1.1281958109476427, 0.9618852387815993, 1.1281958109476427, 1.0430775922714643, 1.310540492774533, 0.6747045661156417, 1.0430775922714643, 1.5095766644650288, 1.2173226567254087, 1.4079312952605532, 1.1281958109476427, 1.0430775922714643, 0.9618852387815993, 0.8109450107566616, 0.552486087697023, 0.30376599435099677, 1.310540492774533, 0.4964211566919598, 0.6118852419676627, 0.7410295032188732, 0.7410295032188732, 1.2173226567254087, 0.8845355774031071, 0.8109450107566616, 1.5095766644650288, 0.8845355774031071, 0.9618852387815993, 1.2173226567254087, 1.2173226567254087, 1.1281958109476427, 0.8109450107566616, 0.552486087697023, 1.2173226567254087, 1.2173226567254087, 0.9618852387815993, 0.6118852419676627, 1.310540492774533, 0.9618852387815993, 1.4079312952605532, 0.8109450107566616, 0.6118852419676627, 0.8109450107566616, 1.4079312952605532, 0.7410295032188732, 1.2173226567254087, 1.310540492774533, 0.7410295032188732, 0.4436039796130322, 1.0430775922714643, 1.5095766644650288, 1.1281958109476427, 0.8109450107566616, 0.8109450107566616, 0.7410295032188732, 1.0430775922714643, 1.310540492774533, 1.2173226567254087, 0.8109450107566616, 0.7410295032188732, 1.4079312952605532, 1.4079312952605532, 1.2173226567254087, 0.4964211566919598, 1.4079312952605532, 0.9618852387815993, 1.5095766644650288, 1.2173226567254087, 0.6118852419676627, 1.0430775922714643, 0.7410295032188732, 0.9618852387815993, 0.6118852419676627, 0.4964211566919598, 1.5095766644650288, 0.8845355774031071, 1.1281958109476427, 1.4079312952605532, 0.9618852387815993, 0.6747045661156417, 1.310540492774533, 0.8845355774031071, 0.4436039796130322, 1.2173226567254087, 0.6747045661156417, 0.3939475438077535, 0.6118852419676627, 0.9618852387815993, 0.6747045661156417, 0.7410295032188732, 1.4079312952605532, 0.8845355774031071, 0.7410295032188732, 0.9618852387815993, 1.1281958109476427, 1.2173226567254087, 0.8845355774031071, 0.7410295032188732, 0.552486087697023, 1.1281958109476427, 1.1281958109476427, 1.4079312952605532, 1.0430775922714643, 0.8845355774031071, 0.552486087697023, 0.8109450107566616, 0.6747045661156417, 1.5095766644650288, 1.310540492774533, 1.1281958109476427, 0.8109450107566616, 1.2173226567254087, 1.0430775922714643, 1.2173226567254087, 0.7410295032188732, 1.4079312952605532, 1.5095766644650288, 1.0430775922714643, 1.1281958109476427, 1.4079312952605532, 1.310540492774533, 0.9618852387815993, 1.4079312952605532, 1.5095766644650288, 0.6118852419676627, 0.8109450107566616, 1.5095766644650288, 1.310540492774533, 0.6747045661156417, 1.310540492774533, 0.552486087697023, 1.1281958109476427, 1.5095766644650288, 0.6747045661156417, 0.9618852387815993, 0.6747045661156417, 1.310540492774533, 0.8845355774031071, 1.4079312952605532, 0.6118852419676627, 1.2173226567254087, 0.8109450107566616, 0.552486087697023, 1.5095766644650288, 0.6747045661156417, 0.7410295032188732, 1.310540492774533, 0.6118852419676627, 0.4436039796130322, 1.310540492774533, 0.9618852387815993, 1.2173226567254087, 1.310540492774533, 1.310540492774533, 1.310540492774533, 1.310540492774533, 1.310540492774533, 0.7410295032188732, 0.9618852387815993, 1.4079312952605532, 1.310540492774533, 0.8845355774031071, 0.9618852387815993, 0.8845355774031071, 1.4079312952605532, 0.4964211566919598, 1.0430775922714643, 1.4079312952605532, 0.6747045661156417, 0.7410295032188732, 0.7410295032188732, 1.2173226567254087, 1.2173226567254087, 0.552486087697023, 1.310540492774533, 0.7410295032188732, 1.4079312952605532, 1.0430775922714643, 0.6118852419676627, 1.310540492774533, 0.8109450107566616, 1.5095766644650288, 0.3939475438077535, 0.8845355774031071, 0.9618852387815993, 0.8109450107566616, 0.8845355774031071, 1.1281958109476427, 1.4079312952605532, 0.8109450107566616] 
    #fmt: off
    # spot_weights_med = np.median(spot_weights)
    # spot_weights = [spot_weights_med if spot_weights[ind] > 3 else spot_weights[ind] for ind in range(len(spot_weights))]
    # updated_spot_weights = curve_extreme_weights_simple(
    # spot_weights_filtered , scaling_factor=1.0
    # )
    updated_spot_weights = spot_weights_filtered
    # Update weights for the specified indices using the calculated weights
    # fmt: off
    # selected_indices_68MHz = [0, 7, 8, 9, 11, 14, 18, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 35, 38, 44, 45, 46, 47, 48, 49, 53, 55, 57, 58, 60, 62, 64, 66, 67, 68, 69, 70, 71, 72, 73]
    # selected_indices_185MHz  =[0, 1, 2, 3, 4, 5, 6, 10, 12, 13, 15, 16, 17, 19, 20, 21, 23, 29, 34, 36, 39, 40, 41, 42, 43, 50, 51, 52, 54, 56, 59, 61, 63, 65, 74]
    # selected_indices = selected_indices_185MHz
    # fmt: on
    # updated_spot_weights = [
    #     val for ind, val in enumerate(updated_spot_weights) if ind in selected_indices
    # ]
    # filtered_reordered_coords = [
    #     filtered_reordered_coords[ind]
    #     for ind in range(len(filtered_reordered_coords))
    #     if ind in selected_indices
    # ]
    # nv_powers = [val for ind, val in enumerate(nv_powers) if ind in selected_indices]

    ####
    filtered_total_power = np.sum(nv_powers_filtered)
    print(total_power)
    adjusted_aom_voltage = ((filtered_total_power - c) / a) ** (1 / b)
    print("Adjusted Voltages (V):", adjusted_aom_voltage)
    # sys.exit()

    filtered_reordered_spot_weights = updated_spot_weights
    print("filtered_reordered_spot_weights_len:", len(filtered_reordered_spot_weights))
    print("filtered_reordered_coords_len:", len(filtered_reordered_coords))
    print("filtered_nv_power_len:", len(nv_powers_filtered))
    print("NV Index | Coords    |   previous weights")
    print("-" * 60)
    for idx, (coords, weight) in enumerate(
        zip(filtered_reordered_coords, filtered_reordered_spot_weights)
    ):
        print(f"{idx + 1:<8} | {coords} | {weight:.3f}")

    print(adjusted_aom_voltage)

    # print(np.max(filtered_reordered_spot_weights))
    # print(np.median(filter_and_reorder_nv_coords))
    # sys.exit()
    # print(len(spot_weights))
    # updated_spot_weights = filtered_reordered_counts
    # spot_weights = updated_spot_weights
    # spot_weights = linear_weights(filtered_reordered_counts, alpha=0.9)
    # spot_weights = non_linear_weights_adjusted(
    #     filtered_intensities, alpha=0.9, beta=0.3, threshold=0.9
    # )
    # spot_weights = sigmoid_weights(filtered_intensities, threshold=0, beta=0.005)
    # Print some diagnostics
    # Update spot weights for NVs with low fidelity

    # Calculate the spot weights based on the integrated intensities
    # spot_weights = non_linear_weights(filtered_intensities, alpha=0.9)

    # Save the filtered results
    # save_results(
    #     filtered_reordered_coords,
    #     filtered_reordered_spot_weights,
    #     filename="slmsuite/nv_blob_detection/nv_blob_151nvs_reordered.npz",
    # )

    # # Plot the original image with circles around each NV

    fig, ax = plt.subplots()
    title = "12ms, INTI_520_Combined_Image"
    kpl.imshow(ax, img_array, title=title, cbar_label="Photons")
    # Draw circles and index numbers
    for idx, coord in enumerate(filtered_reordered_coords):
        circ = plt.Circle(coord, sigma, color="lightblue", fill=False, linewidth=0.5)
        ax.add_patch(circ)
        # Place text just above the circle
        ax.text(
            coord[0],
            coord[1] - sigma - 1,
            str(idx),
            color="white",
            fontsize=8,
            ha="center",
        )

    plt.show(block=True)
