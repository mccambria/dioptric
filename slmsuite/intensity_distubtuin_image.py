import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import curve_fit
from skimage.draw import disk
from tabulate import tabulate

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


def sigmoid_weights(intensities, threshold, beta=1):
    """
    Compute the weights using a sigmoid function.

    intensities: array of intensities
    threshold: intensity value at which the function starts transitioning
    beta: steepness parameter (higher beta makes the transition steeper)
    """
    weights = np.exp(beta * (intensities - threshold))
    return weights / np.max(weights)  # Normalize the weights


def linear_weights(intensities, alpha=1):
    weights = 1 / np.power(intensities, alpha)
    weights = weights / np.max(weights)  # Normalize to avoid extreme values
    return weights


def non_linear_weights_adjusted(intensities, alpha=1, beta=0.5, threshold=0.5):
    """
    Adjust weights such that bright NVs keep the same weight and low-intensity NVs get scaled.

    Parameters:
    - intensities: Array of intensities for NV centers.
    - alpha: Controls the non-linearity for low intensities.
    - beta: Controls the sharpness of the transition around the threshold.
    - threshold: Intensity value above which weights are not changed.

    Returns:
    - weights: Adjusted weights where bright NVs have weight ~1, and low-intensity NVs are scaled.
    """
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
def save_results(
    nv_coordinates, integrated_intensities, spot_weights, updated_spot_weights, filename
):
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
        integrated_counts=integrated_intensities,
        spot_weights=spot_weights,
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
    """
    Update the spot weights using a sigmoid function. Low-fidelity NVs are adjusted more,
    while high-fidelity NVs remain largely unchanged.

    Parameters:
    - fidelities: Array of fidelity values for NV centers.
    - intensities: Array of integrated intensities for NV centers.
    - alpha: Controls the non-linearity of the weight update for low-fidelity NVs.
    - beta: Controls the steepness of the sigmoid function transition.
    - fidelity_threshold: Fidelity value below which weights should be updated.

    Returns:
    - updated_weights: Array of updated spot weights.
    """
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
    """
    Update spot weights only for NVs with specified indices using a sigmoid function.
    Prints the weights before and after updates.

    Parameters:
    - spot_weights: Current spot weights for each NV.
    - intensities: Integrated intensities for each NV.
    - alpha, beta: Parameters for the sigmoid function.
    - update_indices: List of NV indices to update the weights.

    Returns:
    - updated_spot_weights: List of updated spot weights for each NV.
    """
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
    """
    Filter NVs based on peak intensity.

    Args:
        fitted_data: List of tuples (x, y, peak_intensity) from Gaussian fitting.
        threshold: Minimum peak intensity required to keep the NV.

    Returns:
        Filtered NV coordinates and their intensities.
    """
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


# Main section of the code
if __name__ == "__main__":
    kpl.init_kplotlib()

    # Parameters
    remove_outliers_flag = False  # Set this flag to enable/disable outlier removal
    reorder_coords_flag = True  # Set this flag to enable/disable reordering of NVs

    # data = dm.get_raw_data(file_id=1648773947273, load_npz=True)
    # data = dm.get_raw_data(file_id=1651663986412, load_npz=True)
    # data = dm.get_raw_data(file_id=1680236956179, load_npz=True)
    # data = dm.get_raw_data(file_id=1681853425454, load_npz=True)
    # data = dm.get_raw_data(file_id=1688298946808, load_npz=True)
    # data = dm.get_raw_data(file_id=1688328009205, load_npz=True)
    # data = dm.get_raw_data(file_id=1688554695897, load_npz=True)
    # data = dm.get_raw_data(file_id=1693166192526, load_npz=True)
    # data = dm.get_raw_data(file_id=1693412457124, load_npz=True)
    # data = dm.get_raw_data(file_id=1698496302146, load_npz=True)
    # data = dm.get_raw_data(file_id=1699573772441, load_npz=True)
    # data = dm.get_raw_data(file_id=1700650667777, load_npz=True)
    # data = dm.get_raw_data(file_id=1700668458198, load_npz=True)
    data = dm.get_raw_data(file_id=1700710358100, load_npz=True)
    # data = dm.get_raw_data(file_id=1715452021340, load_npz=True)

    img_array = np.array(data["ref_img_array"])
    nv_coordinates, spot_weights = load_nv_coords(
        # file_path="slmsuite/nv_blob_detection/nv_blob_filtered_144nvs.npz"
        # file_path="slmsuite/nv_blob_detection/nv_blob_filtered_160nvs.npz"
        file_path="slmsuite/nv_blob_detection/nv_blob_filtered_160nvs_reordered.npz"
    )
    nv_amps = load_nv_weights().tolist()
    # print(nv_weights)

    nv_coordinates = nv_coordinates.tolist()
    # integrated_intensities = integrated_intensities.tolist()
    spot_weights = spot_weights.tolist()

    # # Display NV data in a table
    # print(f"Filtered NV coordinates: {len(nv_coordinates)} NVs")
    # data = [
    #     [idx + 1, coords, round(weight, 3)]
    #     for idx, (coords, weight) in enumerate(zip(nv_coordinates, nv_weights))
    # ]
    # print(
    #     tabulate(data, headers=["NV Index", "Coordinates", "Weight"], tablefmt="grid")
    # )
    # print(integrated_intensities)
    # integration_radius = 3
    # for coord in nv_coordinates:
    #     x, y = coord[:2]  # Assuming `coord` contains at least two elements (y, x)
    #     rr, cc = disk((y, x), integration_radius, shape=img_array.shape)
    #     sum_value = np.sum(img_array[rr, cc])
    #     integrated_intensities.append(sum_value)  # Append to the list
    # Filter NV coordinates based on x and y ranges (0 to 240)
    # filtered_coords = []
    # filtered_intensities = []
    # for coord, intensity in zip(nv_coordinates, integrated_intensities):
    #     x, y = coord
    #     if 0 <= x <= 248 and 0 <= y <= 248:
    #         filtered_coords.append(coord)
    #         filtered_intensities.append(intensity)

    # # Filter and reorder NV coordinates based on reference NV
    integrated_intensities = []
    sigma = 2.0
    reference_nv = [106.923, 120.549]
    filtered_reordered_coords, filtered_reordered_spot_weights = (
        filter_and_reorder_nv_coords(
            nv_coordinates, spot_weights, reference_nv, min_distance=3
        )
    )

    # Integration over disk region around each NV coordinate
    # for coord, count in zip(nv_coordinates, integrated_intensities):
    #     x, y = coord
    #     if 0 <= x <= 248 and 0 <= y <= 248:
    #         filtered_reordered_coords.append(coord)
    #         filtered_reordered_counts.append(count)
    filtered_reordered_counts = []
    integration_radius = 2.5
    # for coord in filtered_reordered_coords:
    #     x, y = coord[:2]  # Assuming `coord` contains at least two elements (y, x)
    #     rr, cc = disk((y, x), integration_radius, shape=img_array.shape)
    #     sum_value = np.sum(img_array[rr, cc])
    #     filtered_reordered_counts.append(sum_value)  # Append to the list

    # Manually remove NVs with specified indices
    # indices_to_remove = [1, 137, 161]  # Example indices to remove
    # filtered_reordered_coords = [
    #     coord
    #     for i, coord in enumerate(filtered_reordered_coords)
    #     if i not in indices_to_remove
    # ]
    # filtered_reordered_counts = [
    #     count
    #     for i, count in enumerate(filtered_reordered_counts)
    #     if i not in indices_to_remove
    # ]
    # print("Filter:", filtered_reordered_counts)
    # print("Filtered and Reordered NV Coordinates:", filtered_reordered_coords)
    # print("Filtered and Reordered NV Coordinates:", integrated_intensities)

    # integrated_intensities = integrate_intensity(img_array, reordered_nv_coords, sigma)
    # integrated_intensities = np.array(integrated_intensities)
    # Initialize lists to store the results
    # fitted_amplitudes = []
    # for coord in filtered_reordered_coords:
    #     fitted_x, fitted_y, amplitude = fit_gaussian(img_array, coord, window_size=2)
    #     fitted_amplitudes.append(amplitude)

    # Calculate weights based on the fitted intensities
    # indices = [4, 27, 41, 82, 86, 89, 109, 117, 138, 139, 148, 149]
    spot_weights = filtered_reordered_spot_weights
    indices = [4, 27, 30, 41, 117, 130, 139, 155]
    # Apply linear weights to all counts
    # calcualted_spot_weights = linear_weights(filtered_reordered_counts, alpha=0.3)
    # updated_spot_weights = linear_weights(filtered_reordered_counts, alpha=0.6)

    # Create a copy or initialize spot weights for modification
    updated_spot_weights = (
        spot_weights.copy()
    )  # Assuming 'spot_weights' is your original weight array

    # Update weights for the specified indices using the calculated weights
    # for idx in indices:
    #     if 0 <= idx < len(updated_spot_weights):  # Ensure index is within valid range
    #         updated_spot_weights[idx] = calcualted_spot_weights[idx]

    aom_voltage = 0.39  # Current AOM voltage
    power_law_params = [3.7e5, 6.97, 8e-14]  # Example power-law fit parameters

    nv_weights, adjusted_aom_voltage = adjust_aom_voltage_for_slm(
        nv_amps, aom_voltage, power_law_params
    )

    # Print adjusted voltages
    print("Adjusted Voltages (V):", adjusted_aom_voltage)
    print("nv_weights:", nv_weights)
    print("NV Index | Coords    |   previous weights")
    print("-" * 60)
    for idx, (coords, weight) in enumerate(
        zip(
            nv_coordinates,
            nv_weights,
        )
    ):
        print(f"{idx+1:<8} | {coords} | {weight:.3f}")
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
    # Define the NV indices you want to update
    # indices_to_update = [11, 14, 18, 19, 25, 30, 36, 42, 43, 45, 48, 56, 72]
    # indices_to_update = np.arange(0, 77).tolist()
    # Calculate the updated spot weights
    # updated_spot_weights = manual_sigmoid_weight_update(
    #     spot_weights,
    #     integrated_intensities,
    #     alpha=0.9,
    #     beta=6.0,
    #     update_indices=indices_to_update,
    # )

    # updated_spot_weights = adjust_weights_sigmoid(
    #     spot_weights, snr, alpha=0.0, beta=0.3
    # )

    # Get indices of NVs that meet the SNR threshold
    # threshold = 0.0
    # filtered_indices = filter_by_snr(snr, threshold)

    # Filter NV coordinates and associated data based on these indices
    # filtered_nv_coords = [reordered_nv_coords[i] for i in filtered_indices]
    # filtered_spot_weights = [spot_weights[i] for i in filtered_indices]
    # filtered_integrated_intensities = [
    #     integrated_intensities[i] for i in filtered_indices
    # ]
    print(f"Filtered NV coordinates: {len(filtered_reordered_coords)} NVs")
    print("NV Index | Coords    |    Counts |   previous weights |   updated weights")
    print("-" * 60)
    for idx, (coords, counts, weight, updated_weight) in enumerate(
        zip(
            nv_coordinates,
            filtered_reordered_counts,
            nv_weights,
            updated_spot_weights,
        )
    ):
        print(
            f"{idx+1:<8} | {coords} | {counts:.3f} | {weight:.3f} | {updated_weight:.3f}"
        )

    # print("NV Index | Spot Weight | Updated Spot Weight | Counts")
    # print("-" * 50)
    # for idx, (weight, updated_weight, counts) in enumerate(
    #     zip(spot_weights, updated_spot_weights, filtered_reordered_counts)
    # ):
    #     print(f"{idx:<8} | {weight:.3f} | {updated_weight:.3f} | {counts:.3f}")
    # print(f"NV Coords: {filtered_nv_coords}")
    # print(f"Filtered integrated intensities: {filtered_intensities}")
    # print(f"Normalized spot weights: {spot_weights}")
    # print(f"Normalized spot weights: {updated_spot_weights}")
    # print(f"Number of NVs detected: {len(filtered_nv_coords)}")

    # Save the filtered results
    # save_results(
    #     filtered_reordered_coords,
    #     filtered_reordered_counts,
    #     spot_weights,
    #     updated_spot_weights,
    #     filename="slmsuite/nv_blob_detection/nv_blob_filtered_160nvs_reordered.npz",
    # )
    # save_results(
    #     nv_coordinates,
    #     filtered_reordered_counts,
    #     nv_weights,
    #     updated_spot_weights,
    #     filename="slmsuite/nv_blob_detection/nv_blob_filtered_160nvs_reordered.npz",
    # )

    # # Plot the original image with circles around each NV
    fig, ax = plt.subplots()
    title = "50ms, Ref"
    kpl.imshow(ax, img_array, title=title, cbar_label="Photons")
    # Draw circles and index numbers
    for idx, coord in enumerate(filtered_reordered_coords):
        circ = plt.Circle(coord, sigma, color="lightblue", fill=False, linewidth=0.5)
        ax.add_patch(circ)
        # Place text just above the circle
        ax.text(
            coord[0],
            coord[1] - sigma - 1,
            str(idx + 1),
            color="white",
            fontsize=6,
            ha="center",
        )

    # indices_to_circle = [4, 29, 41, 89, 102, 118, 139, 144, 148, 149]
    # indices_to_circle = [4, 27, 41, 82, 86, 89, 102, 109, 117, 138, 139, 148, 149]
    # indices_to_circle = [
    #     i + 1 for i in [4, 27, 41, 82, 86, 89, 102, 109, 117, 138, 139, 148, 149]
    # ]
    # fig, ax = plt.subplots()
    # title = "50ms, Ref"
    # kpl.imshow(ax, img_array, title=title, cbar_label="Photons")

    # # Draw circles and index numbers for selected indices only
    # for idx, coord in enumerate(filtered_reordered_coords):
    #     if idx + 1 in indices_to_circle:  # +1 to match your indices (assuming 1-based)
    #         circ = Circle(coord, sigma, color="lightblue", fill=False, linewidth=0.5)
    #         ax.add_patch(circ)
    #         # Place text just above the circle
    #         ax.text(
    #             coord[0],
    #             coord[1] - sigma - 1,
    #             str(idx + 1),
    #             color="white",
    #             fontsize=8,
    #             ha="center",
    #         )
    # Plot histogram of the filtered integrated intensities using Seaborn
    # sns.set(style="whitegrid")

    # plt.figure(figsize=(6, 5))
    # sns.histplot(filtered_counts, bins=45, kde=False, color="blue")

    # plt.xlabel("Integrated Intensity")
    # plt.ylabel("Frequency")
    # plt.title("Histogram of Filtered Integrated Counts")
    plt.show(block=True)
