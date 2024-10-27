import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Circle
from scipy.optimize import curve_fit
from skimage.draw import disk

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
    intensities = []
    for coord in nv_coords:
        # Create a circular mask around the NV coordinate with the given sigma as radius
        rr, cc = disk(coord, sigma, shape=image_array.shape)
        # Integrate (sum) the intensity values within the disk
        intensity = np.sum(image_array[rr, cc])
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


def reorder_coords(nv_coords):
    # Calculate Euclidean distances from the first NV coordinate
    distances = [
        np.linalg.norm(np.array(coord) - np.array(nv_coords[0])) for coord in nv_coords
    ]
    # Get sorted indices based on distances
    sorted_indices = np.argsort(distances)
    # Reorder NV coordinates based on sorted distances
    reordered_coords = [nv_coords[idx] for idx in sorted_indices]
    return reordered_coords


def sigmoid_weights(intensities, threshold, beta=1):
    """
    Compute the weights using a sigmoid function.

    intensities: array of intensities
    threshold: intensity value at which the function starts transitioning
    beta: steepness parameter (higher beta makes the transition steeper)
    """
    weights = np.exp(beta * (intensities - threshold))
    return weights / np.max(weights)  # Normalize the weights


def non_linear_weights(intensities, alpha=1):
    weights = 1 / np.power(intensities, alpha)
    weights = weights / np.max(weights)  # Normalize to avoid extreme values
    return weights


def linear_weights(intensities, alpha=1):
    weights = 1 / np.power(intensities, alpha)
    weights = weights / np.max(weights)  # Normalize to avoid extreme values
    return weights


def non_linear_weights_adjusted(intensities, alpha=1, beta=0.5, threshold=0.7):
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
def save_results(nv_coordinates, integrated_intensities, spot_weights, filename):
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
        integrated_intensities=integrated_intensities,
        spot_weights=spot_weights,
    )


def filter_by_snr(snr_list, threshold=0.5):
    """Filter out NVs with SNR below the threshold."""
    return [i for i, snr in enumerate(snr_list) if snr >= threshold]


def load_nv_coords(
    # file_path="slmsuite/nv_blob_detection/nv_blob_filtered_162nvs_ref.npz",
    # file_path="slmsuite/nv_blob_detection/nv_blob_filtered_77nvs_new.npz",
    file_path="slmsuite/nv_blob_detection/nv_blob_filtered_240nvs.npz",
):
    data = np.load(file_path)
    print(data.keys)
    nv_coordinates = data["nv_coordinates"]
    spot_weights = data["spot_weights"]
    # spot_weights = data["integrated_counts"]
    return nv_coordinates, spot_weights


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
    data = dm.get_raw_data(file_id=1683023192746, load_npz=True)

    img_array = np.array(data["ref_img_array"])
    # img_array = -np.array(data["diff_img_array"])
    nv_coordinates, spot_weights = load_nv_coords(
        file_path="slmsuite/nv_blob_detection/nv_blob_filtered_116nvs_updated.npz"
    )
    nv_coordinates = nv_coordinates.tolist()
    # spot_weights = spot_weights.tolist()
    # spot_weights = np.array(spot_weights)
    # print(spot_weights)

    # Start merged coordinates with the reference NV
    # Reference NV to append as the first coordinate
    reference_nv = [122.502, 159.336]
    # # # Start with the reference NV as the first element
    nv_coords = [reference_nv]

    # # Iterate through the rest of the NV coordinates
    # Iterate through the rest of the NV coordinates
    for coord in nv_coordinates:
        # Check if the new NV coordinate is far enough from all accepted NVs
        keep_coord = True  # Assume the coordinate is valid

        for existing_coord in nv_coords:
            # Calculate the distance between the current NV and each existing NV
            distance = np.linalg.norm(np.array(existing_coord) - np.array(coord))

            if distance < 6:
                keep_coord = False  # If too close, mark it for exclusion
                break  # No need to check further distances

        # Ensure both x and y are within the valid range [0, 249]
        x, y = int(coord[0]), int(coord[1])
        if not (0 <= x < 250 and 0 <= y < 250):
            print(f"Skipping NV at ({coord[0]:.2f}, {coord[1]:.2f}) - Out of bounds")
            continue  # Skip this coordinate

        # Check the pixel value at the NV coordinate
        # pixel_value = img_array[y, x]
        # if pixel_value >= pixel_value_threshold:
        #     print(f"Excluding NV at ({coord[0]:.2f}, {coord[1]:.2f}) - Pixel value: {pixel_value}")
        #     keep_coord = False
        # If the coordinate passed the distance check, add it to the list
        if keep_coord:
            nv_coords.append(coord)
    # Fit 2D Gaussian to each NV and get optimized coordinates
    # optimal_nv_coords = []
    # for coord in nv_coords:
    #     optimized_x, optimized_y = fit_gaussian(img_array, coord)
    #     optimal_nv_coords.append((optimized_x, optimized_y))

    # Fit 2D Gaussian and collect peak intensities
    fitted_data = []
    for coord in nv_coords:
        optimized_x, optimized_y, peak_intensity = fit_gaussian(img_array, coord)
        fitted_data.append((optimized_x, optimized_y, peak_intensity))

    # Filter NVs by peak intensity
    filtered_nv_coords, filtered_counts = filter_by_peak_intensity(
        fitted_data, threshold=0.6
    )
    # # Round each coordinate to 3 decimal places
    rounded_nv_coords = [(round(x, 3), round(y, 3)) for x, y in filtered_nv_coords]
    filtered_nv_coords = rounded_nv_coords
    # Print NV index and corresponding coordinates
    print("NV Index | Coordinates (x, y)")
    print("-" * 30)
    for idx, (x, y) in enumerate(rounded_nv_coords):
        print(f"{idx:<8} | [{x:.3f}, {y:.3f}]")
    # print(filtered_nv_coords)
    # manual removal of the nvs
    # manual_removal_indices = [115]
    # filtered_nv_coords = remove_manual_indices(
    #     filtered_nv_coords, manual_removal_indices
    # )

    # Remove NVs with SNR < 0.5
    # filtered_nv_indices = filter_by_snr(SNR, threshold=0.7)
    # nv_coordinates = [nv_coordinates[i] for i in filtered_nv_indices]

    # Integrate intensities for each NV coordinate
    sigma = 2
    integrated_intensities = integrate_intensity(img_array, filtered_nv_coords, sigma)
    integrated_intensities = np.array(integrated_intensities)

    # if remove_outliers_flag:
    #     # Remove outliers and corresponding NV coordinates
    #     filtered_intensities, filtered_nv_coords = remove_outliers(
    #         integrated_intensities, nv_coordinates
    #     )
    # else:
    #     filtered_intensities = integrated_intensities
    #     filtered_nv_coords = nv_coordinates

    # Reorder NV coordinates based on Euclidean distances from the first NV
    if reorder_coords_flag:
        filtered_nv_coords = reorder_coords(filtered_nv_coords)

    # Convert filtered intensities to a NumPy array for element-wise operations
    filtered_counts = np.array(filtered_counts)

    # updated_spot_weights = linear_weights(integrated_intensities, alpha=0.9)
    # spot_weights = linear_weights(filtered_intensities, alpha=0.2)
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
    snr = [
        0.859,
        0.886,
        0.837,
        0.872,
        0.726,
        0.824,
        1.005,
        1.007,
        0.984,
        0.641,
        0.822,
        1.153,
        0.934,
        1.161,
        0.729,
        0.742,
        0.965,
        0.629,
        0.826,
        0.866,
        0.887,
        0.567,
        0.825,
        0.792,
        1.127,
        0.878,
        0.791,
        0.811,
        1.057,
        0.977,
        1.02,
        0.928,
        1.374,
        0.929,
        0.88,
        0.6,
        0.983,
        1.379,
        0.816,
        0.922,
        1.093,
        0.837,
        0.895,
        0.974,
        0.906,
        0.802,
        0.901,
        1.095,
        0.922,
        1.348,
        0.541,
        0.891,
        0.788,
        0.944,
        0.867,
        0.939,
        0.191,
        0.935,
        0.991,
        1.107,
        1.195,
        0.644,
        0.907,
        1.288,
        0.87,
        0.918,
        0.841,
        0.95,
        1.326,
        0.911,
        0.762,
        0.924,
        1.272,
        0.867,
        0.897,
        0.926,
        0.841,
        0.827,
        1.03,
        1.053,
        0.809,
        1.525,
        1.051,
        0.858,
        1.123,
        0.935,
        1.223,
        0.969,
        1.068,
        0.955,
        0.939,
        0.903,
        1.03,
        0.521,
        0.83,
        0.821,
        1.235,
        1.036,
        0.968,
        0.872,
        0.811,
        0.825,
        0.895,
        0.297,
        0.523,
        1.092,
        0.827,
        0.844,
        0.927,
        0.974,
        1.067,
        0.886,
        0.554,
        0.873,
        0.953,
        1.159,
    ]
    updated_spot_weights = adjust_weights_sigmoid(
        spot_weights, snr, alpha=0.0, beta=0.3
    )

    print(f"Filtered NV coordinates: {len(filtered_nv_coords)} NVs")
    print("NV Index | Spot Weight | Updated Spot Weight | Counts")
    print("-" * 50)
    for idx, (weight, updated_weight, counts) in enumerate(
        zip(spot_weights, updated_spot_weights, integrated_intensities)
    ):
        print(f"{idx:<8} | {weight:.3f} | {updated_weight:.3f} | {counts:.3f}")
    # print(f"NV Coords: {filtered_nv_coords}")
    # print(f"Filtered integrated intensities: {filtered_intensities}")
    # print(f"Normalized spot weights: {spot_weights}")
    # print(f"Normalized spot weights: {updated_spot_weights}")
    # print(f"Number of NVs detected: {len(filtered_nv_coords)}")

    # Save the filtered results
    # save_results(
    #     filtered_nv_coords,
    #     filtered_counts,
    #     updated_spot_weights,
    #     filename="slmsuite/nv_blob_detection/nv_blob_filtered_116nvs_updated.npz",
    # )

    # Plot the original image with circles around each NV
    fig, ax = plt.subplots()
    title = "24ms, Ref"
    kpl.imshow(ax, img_array, title=title, cbar_label="Photons")
    # Draw circles and index numbers
    for idx, coord in enumerate(filtered_nv_coords):
        circ = Circle(coord, sigma, color="lightblue", fill=False, linewidth=0.5)
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

    # Plot histogram of the filtered integrated intensities using Seaborn
    sns.set(style="whitegrid")

    plt.figure(figsize=(6, 5))
    sns.histplot(filtered_counts, bins=45, kde=False, color="blue")

    plt.xlabel("Integrated Intensity")
    plt.ylabel("Frequency")
    plt.title("Histogram of Filtered Integrated Counts")
    plt.show(block=True)
