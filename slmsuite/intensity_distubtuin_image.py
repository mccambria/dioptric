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
def save_results(nv_coordinates, updated_spot_weights, filename):
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
    spot_weights = data["spot_weights"]
    # spot_weights = data["updated_spot_weights"]
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


def curve_extreme_weights_simple(weights, scaling_factor=1.0):
    median = np.median(weights)

    curved_weights = [1 / (1 + np.exp(-scaling_factor * (w - median))) for w in weights]

    return curved_weights


def select_half_left_side_nvs_and_plot(nv_coordinates):
    """
    Select half of the NVs on the left side of the coordinate space
    based on their 'pixel' x-coordinates, mark them with spin_flip = True,
    and plot the distribution.

    Parameters:
    - nv_list: List of NV objects, each with a 'coords' dictionary containing 'pixel'.

    Returns:
    - selected_indices: List of indices of NVs selected from the left side.
    """

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
    # data = dm.get_raw_data(file_id=1700710358100, load_npz=True)
    # data = dm.get_raw_data(file_id=1733432867671, load_npz=True)
    # data = dm.get_raw_data(file_id=1732420670067, load_npz=True)
    # data = dm.get_raw_data(file_id=1751170993877, load_npz=True)
    # data = dm.get_raw_data(file_id=1752794666146, load_npz=True)
    # data = dm.get_raw_data(file_id=1764727515943, load_npz=True)  # comniened
    data = dm.get_raw_data(file_id=1766632456770, load_npz=True)

    img_array = data["ref_img_array"]
    # img_array = data["img_array"]
    # print(img_arrays)
    # sys.exit()
    nv_coordinates, spot_weights = load_nv_coords(
        # file_path="slmsuite/nv_blob_detection/nv_blob_filtered_144nvs.npz"
        # file_path="slmsuite/nv_blob_detection/nv_blob_filtered_160nvs.npz"
        # file_path="slmsuite/nv_blob_detection/nv_blob_shallow_326nvs.npz"
        file_path="slmsuite/nv_blob_detection/nv_blob_shallow_149nvs.npz"
        # file_path="slmsuite/nv_blob_detection/nv_blob_filtered_160nvs_reordered.npz"
    )
    # nv_amps = load_nv_weights().tolist()

    # Convert coordinates to a standard format (lists of lists)
    nv_coordinates = [list(coord) for coord in nv_coordinates]

    # Debug: Print before filtering
    print(f"Before filtering: {len(nv_coordinates)} NVs")
    print(f"Sample NV coordinates: {nv_coordinates[:10]}")  # Check first 10 NVs

    # Filter NV coordinates: Keep only those where both x and y are in [0, 250]
    nv_coordinates_filtered = [
        coord
        for coord in nv_coordinates
        if isinstance(coord, (list, tuple))
        and len(coord) == 2
        and all(0 <= x <= 250 for x in coord)
    ]

    # Debug: Print after filtering

    # Ensure spot weights are filtered accordingly
    spot_weights_filtered = [
        weight
        for coord, weight in zip(nv_coordinates, spot_weights)
        if isinstance(coord, (list, tuple))
        and len(coord) == 2
        and all(0 <= x <= 250 for x in coord)
    ]

    # Replace original lists with filtered versions
    nv_coordinates = nv_coordinates_filtered
    spot_weights = spot_weights_filtered
    print(f"After filtering: {len(nv_coordinates_filtered)} NVs")
    # Debug: Print the filtered NV coordinates
    # print("Filtered NV coordinates:")
    # for idx, coord in enumerate(nv_coordinates):
    #     print(f"{idx}: {coord}")

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
    # [107.51, 120.42]
    # reference_nv = [106.923, 120.549]
    reference_nv = [117.516, 129.595]
    # reference_nv = [116.765, 129.256]
    filtered_reordered_coords, filtered_reordered_spot_weights = (
        filter_and_reorder_nv_coords(
            nv_coordinates, spot_weights_filtered, reference_nv, min_distance=3
        )
    )

    # # Convert to numpy arrays
    # filtered_reordered_coords = np.array(filtered_reordered_coords)
    # intensities = np.array(filtered_reordered_spot_weights)

    # # Remove extreme intensity values (e.g., outside 1 standard deviation)
    # mean_intensity = np.mean(intensities)
    # std_intensity = np.std(intensities)
    # lower_bound = mean_intensity - 2 * std_intensity
    # upper_bound = mean_intensity + 2 * std_intensity

    # intensity_mask = (intensities >= lower_bound) & (intensities <= upper_bound)

    # # Filter based on spatial coordinates
    # x_coords, y_coords = (
    #     filtered_reordered_coords[:, 0],
    #     filtered_reordered_coords[:, 1],
    # )
    # spatial_mask = (x_coords > 0) & (x_coords < 250) & (y_coords > 0) & (y_coords < 250)

    # # Combine both filters
    # final_mask = intensity_mask & spatial_mask

    # # Apply final filtering
    # filtered_reordered_coords = filtered_reordered_coords[final_mask]
    # filtered_reordered_spot_weights = intensities[final_mask]

    # # Plot filtered NV positions
    # plt.figure(figsize=(6, 6))
    # plt.scatter(
    #     filtered_reordered_coords[:, 0],
    #     filtered_reordered_coords[:, 1],
    #     c=filtered_reordered_spot_weights,
    #     cmap="viridis",
    #     edgecolors="black",
    # )
    # plt.colorbar(label="Intensity")
    # plt.xlabel("X Coordinate")
    # plt.ylabel("Y Coordinate")
    # plt.title("Filtered NV Positions")
    # plt.show()

    # Integration over disk region around each NV coordinate
    # for coord, count in zip(nv_coordinates, integrated_intensities):
    #     x, y = coord
    #     if 0 <= x <= 248 and 0 <= y <= 248:
    #         filtered_reordered_coords.append(coord)
    #         filtered_reordered_counts.append(count)
    # filtered_reordered_counts = []
    # integration_radius = 2.0
    # for coord in filtered_reordered_coords:
    #     x, y = coord[:2]  # Assuming `coord` contains at least two elements (y, x)
    #     rr, cc = disk((y, x), integration_radius, shape=img_array.shape)
    #     sum_value = np.sum(img_array[rr, cc])
    #     filtered_reordered_counts.append(sum_value)  # Append to the list

    # Manually remove NVs with specified indices
    indices_to_remove = [2]  # Example indices to remove
    filtered_reordered_coords = [
        coord
        for i, coord in enumerate(filtered_reordered_coords)
        if i not in indices_to_remove
    ]
    filtered_reordered_spot_weights = [
        count
        for i, count in enumerate(filtered_reordered_spot_weights)
        if i not in indices_to_remove
    ]
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
    # spot_weights = filtered_reordered_spot_weights
    # fmt: off
    # spots weights for 117 NVs before Birge shutdown
    # spot_weights = [0.7466728351068052, 0.7466728351068052, 0.6384167684513203, 1.1708211389400445, 1.5561808283609733, 1.1708211389400445, 0.6384167684513203, 0.7466728351068052, 1.1708211389400445, 0.7466728351068052, 0.7466728351068052, 0.6384167684513203, 1.1708211389400445, 0.7466728351068052, 0.7466728351068052, 0.7466728351068052, 0.8702844369863524, 0.7466728351068052, 0.8702844369863524, 0.6384167684513203, 0.8702844369863524, 0.6384167684513203, 1.0110233833356788, 0.6384167684513203, 0.6384167684513203, 0.8702844369863524, 1.5561808283609733, 0.8702844369863524, 0.8702844369863524, 1.3517793512995497, 0.7466728351068052, 0.8702844369863524, 1.5561808283609733, 0.7466728351068052, 0.8702844369863524, 0.7466728351068052, 1.0110233833356788, 0.7466728351068052, 0.5438939973931943, 0.7466728351068052, 3.023897571362437, 0.7466728351068052, 0.8702844369863524, 0.8702844369863524, 0.7466728351068052, 1.0110233833356788, 0.6384167684513203, 0.7466728351068052, 0.7466728351068052, 1.0110233833356788, 0.5438939973931943, 0.7466728351068052, 1.5561808283609733, 1.5561808283609733, 1.0110233833356788, 0.7466728351068052, 1.3517793512995497, 0.6384167684513203, 1.0110233833356788, 0.7466728351068052, 0.7466728351068052, 0.8702844369863524, 1.0110233833356788, 1.5561808283609733, 1.1708211389400445, 1.0110233833356788, 0.6384167684513203, 0.8702844369863524, 0.8702844369863524, 0.7466728351068052, 0.6384167684513203, 0.7466728351068052, 0.8702844369863524, 1.0110233833356788, 0.8702844369863524, 0.8702844369863524, 0.7466728351068052, 1.3517793512995497, 1.5561808283609733, 1.0110233833356788, 0.8702844369863524, 0.7466728351068052, 0.8702844369863524, 1.0110233833356788, 0.8702844369863524, 1.0110233833356788, 1.1708211389400445, 1.5561808283609733, 0.8702844369863524, 0.8702844369863524, 1.3517793512995497, 0.7466728351068052, 1.0110233833356788, 1.0110233833356788, 1.5561808283609733, 2.0454197052578524, 0.7466728351068052, 1.7865009761020008, 2.0454197052578524, 1.1708211389400445, 1.1708211389400445, 1.1708211389400445, 1.0110233833356788, 1.5561808283609733, 0.8702844369863524, 1.3517793512995497, 1.0110233833356788, 0.7466728351068052, 2.0454197052578524, 0.7466728351068052, 0.8702844369863524, 0.7466728351068052, 1.1708211389400445, 1.7865009761020008, 0.7466728351068052, 0.8702844369863524, 1.1708211389400445]
    # spots weights for 117 NVs after Birge shutdown
    # spot_weights = [0.88616979, 0.76754119, 1.17115504, 1.02016682, 0.48948736, 1.02016682, 1.17115504, 0.66278449, 0.66278449, 0.88616979, 1.02016682, 0.88616979, 0.88616979, 0.88616979, 2.24760926, 0.88616979, 0.88616979, 0.88616979, 0.35655407, 1.74435458, 1.34089129, 2.54271689, 0.88616979, 0.66278449, 1.34089129, 1.17115504, 0.57052194, 0.25583134, 1.34089129, 0.88616979, 1.02016682, 0.76754119, 0.76754119, 1.17115504, 0.88616979, 1.02016682, 0.76754119, 0.88616979, 0.66278449, 1.17115504, 0.76754119, 1.17115504, 0.88616979, 0.57052194, 0.88616979, 1.02016682, 0.88616979, 0.66278449, 0.76754119, 1.02016682, 0.76754119, 1.9823403, 1.17115504, 1.17115504, 0.76754119, 1.17115504, 0.48948736, 0.88616979, 0.76754119, 0.41851923, 1.34089129, 0.88616979, 0.66278449, 0.76754119, 1.34089129, 1.34089129, 1.02016682, 1.02016682, 0.66278449, 1.02016682, 1.02016682, 0.66278449, 0.57052194, 1.34089129, 2.54271689, 0.88616979, 1.17115504, 1.17115504, 0.66278449, 1.34089129, 1.17115504, 0.76754119, 1.02016682, 1.17115504, 2.24760926, 0.66278449, 0.76754119, 0.41851923, 1.02016682, 1.34089129, 1.02016682, 1.17115504, 1.02016682, 1.74435458, 0.66278449, 0.66278449, 1.17115504, 0.48948736, 0.88616979, 0.76754119, 1.17115504, 0.76754119, 1.17115504, 1.02016682, 1.9823403, 0.41851923, 0.57052194, 1.9823403, 1.02016682, 0.76754119, 0.76754119, 1.02016682, 1.17115504, 0.57052194, 1.02016682, 1.02016682, 1.17115504]
    # spot_weights = [0.79982543, 0.59451918, 0.59451918, 0.79982543, 1.59568291, 1.22041605, 0.92344378, 0.69066246, 0.92344378, 0.69066246, 0.51007613, 0.79982543, 1.06307698, 1.06307698, 1.39729173, 0.69066246, 0.79982543, 0.37155141, 0.79982543, 0.79982543, 0.79982543, 0.92344378, 0.79982543, 0.79982543, 0.43612295, 1.06307698, 1.59568291, 0.69066246, 0.92344378, 0.92344378, 0.69066246, 0.92344378, 1.06307698, 0.79982543, 1.22041605, 0.69066246, 0.69066246, 0.92344378, 0.92344378, 0.79982543, 2.06572131, 0.69066246, 0.59451918, 0.26659209, 0.69066246, 0.59451918, 0.69066246, 0.92344378, 0.51007613, 1.22041605, 0.59451918, 1.81772545, 1.81772545, 1.81772545, 1.06307698, 0.69066246, 0.92344378, 0.79982543, 0.69066246, 0.59451918, 0.79982543, 0.69066246, 0.59451918, 0.92344378, 1.81772545, 1.39729173, 0.51007613, 0.79982543, 0.92344378, 0.92344378, 0.79982543, 0.69066246, 0.79982543, 1.59568291, 2.34214798, 0.69066246, 1.22041605, 0.92344378, 0.92344378, 1.22041605, 1.06307698, 0.92344378, 0.43612295, 1.22041605, 0.31534891, 0.69066246, 0.92344378, 2.06572131, 0.69066246, 0.79982543, 0.92344378, 1.06307698, 0.92344378, 1.59568291, 1.59568291, 1.39729173, 0.51007613, 1.81772545, 1.81772545, 0.79982543, 1.59568291, 1.22041605, 2.06572131, 2.6496684, 0.79982543, 0.69066246, 1.39729173, 0.92344378, 1.81772545, 0.79982543, 0.69066246, 0.69066246, 1.22041605, 1.59568291, 1.06307698, 0.69066246, 1.06307698]
    #fidelity
    # spot_weights = [0.56184079, 1.45107477, 0.85728647, 0.64867703, 0.22151819, 0.64867703, 0.85728647, 0.74676297, 0.41762254, 0.85728647, 0.56184079, 0.74676297, 0.56184079, 2.10113985, 1.27686902, 0.56184079, 0.56184079, 0.56184079, 0.74676297, 0.85728647, 0.64867703, 0.41762254, 0.64867703, 0.98153355, 2.36701149, 0.98153355, 0.56184079, 0.41762254, 0.41762254, 0.74676297, 2.66119108, 0.4851588, 0.35830516, 0.85728647, 0.85728647, 0.64867703, 0.56184079, 0.4851588, 1.27686902, 0.64867703, 0.74676297, 2.10113985, 0.98153355, 0.41762254, 0.4851588, 0.85728647, 2.36701149, 0.41762254, 0.74676297, 1.64525187, 0.74676297, 0.98153355, 2.66119108, 0.74676297, 1.12089429, 0.64867703, 0.22151819, 0.85728647, 1.45107477, 0.98153355, 0.64867703, 1.86127091, 1.64525187, 0.64867703, 0.74676297, 0.56184079, 0.18726876, 1.45107477, 0.74676297, 0.74676297, 0.74676297, 0.56184079, 0.35830516, 0.98153355, 0.98153355, 0.74676297, 0.74676297, 2.66119108, 1.45107477, 1.86127091, 0.64867703, 2.66119108, 0.85728647, 1.27686902, 1.27686902, 1.12089429, 0.64867703, 1.27686902, 1.27686902, 1.86127091, 0.56184079, 1.27686902, 0.85728647, 0.56184079, 1.64525187, 0.64867703, 0.85728647, 0.74676297, 1.27686902, 2.66119108, 0.22151819, 0.56184079, 0.15765944, 0.4851588, 1.45107477, 0.56184079, 0.56184079, 0.4851588, 0.85728647, 2.10113985, 0.74676297, 0.74676297, 0.98153355, 0.41762254, 2.36701149, 2.36701149, 1.64525187, 1.45107477, 0.64867703, 0.64867703, 2.66119108, 0.56184079, 0.41762254, 0.74676297, 1.45107477, 0.56184079, 1.86127091, 0.41762254, 0.74676297, 0.98153355, 0.74676297, 1.86127091, 0.74676297, 0.74676297, 1.45107477, 2.36701149, 0.56184079, 0.74676297, 0.56184079, 0.74676297, 0.56184079, 0.98153355, 0.85728647, 0.30635643, 0.85728647, 0.4851588, 0.4851588, 2.10113985, 0.30635643, 1.45107477, 0.98153355, 1.64525187, 0.85728647, 0.64867703, 0.74676297, 0.35830516, 1.45107477, 2.66119108, 1.12089429, 2.36701149]
    # spot_weights = [0.9946230241426821, 0.8752156373168588, 0.7683044943185703, 0.5876174577844024, 1.6224416389703067, 0.7683044943185703, 0.5876174577844024, 1.4402028887333522, 0.6727812317954857, 1.1277195536713025, 0.8752156373168588, 0.5876174577844024, 0.9946230241426821, 0.9946230241426821, 1.4402028887333522, 0.7683044943185703, 0.5876174577844024, 0.511860354092575, 1.6224416389703067, 1.2757873939965607, 0.3325466922344282, 0.5876174577844024, 0.5876174577844024, 1.1277195536713025, 0.7683044943185703, 0.6727812317954857, 0.8752156373168588, 0.5876174577844024, 1.6224416389703067, 0.5876174577844024, 0.9946230241426821, 1.2757873939965607, 0.8752156373168588, 0.7683044943185703, 0.7683044943185703, 1.2757873939965607, 0.5876174577844024, 0.6727812317954857, 0.8752156373168588, 0.44462844214574804, 0.7683044943185703, 0.8752156373168588, 1.2757873939965607, 0.7683044943185703, 0.6727812317954857, 1.8240837639143594, 0.7683044943185703, 0.6727812317954857, 0.6727812317954857, 1.1277195536713025, 0.7683044943185703, 0.8752156373168588, 1.6224416389703067, 0.7683044943185703, 1.8240837639143594, 0.9946230241426821, 0.24559627956065677, 1.2757873939965607, 1.8240837639143594, 0.6727812317954857, 0.8752156373168588, 1.2757873939965607, 1.8240837639143594, 1.2757873939965607, 0.9946230241426821, 1.8240837639143594, 0.44462844214574804, 1.8240837639143594, 0.5876174577844024, 1.2757873939965607, 0.7683044943185703, 0.5876174577844024, 0.6727812317954857, 0.9946230241426821, 1.4402028887333522, 0.6727812317954857, 0.9946230241426821, 1.6224416389703067, 0.44462844214574804, 0.7683044943185703, 0.511860354092575, 1.1277195536713025, 0.8752156373168588, 1.6224416389703067, 1.8240837639143594, 1.4402028887333522, 1.8240837639143594, 1.8240837639143594, 1.1277195536713025, 1.2757873939965607, 0.7683044943185703, 1.2757873939965607, 0.6727812317954857, 0.5876174577844024, 0.7683044943185703, 0.3325466922344282, 0.6727812317954857, 0.5876174577844024, 1.6224416389703067, 1.2757873939965607, 0.7683044943185703, 0.6727812317954857, 0.38510750925271625, 0.511860354092575, 0.7683044943185703, 0.9946230241426821, 1.4402028887333522, 0.511860354092575, 0.6727812317954857, 1.1277195536713025, 1.2757873939965607, 0.44462844214574804, 0.38510750925271625, 0.511860354092575, 1.8240837639143594, 1.2757873939965607, 1.8240837639143594, 1.8240837639143594, 0.511860354092575, 0.511860354092575, 0.6727812317954857, 0.5876174577844024, 0.5876174577844024, 1.4402028887333522, 1.6224416389703067, 0.511860354092575, 1.8240837639143594, 0.9946230241426821, 0.6727812317954857, 0.8752156373168588, 0.8752156373168588, 0.9946230241426821, 1.1277195536713025, 1.4402028887333522, 1.2757873939965607, 1.8240837639143594, 0.9946230241426821, 0.8752156373168588, 1.6224416389703067, 0.7683044943185703, 0.9946230241426821, 0.6727812317954857, 0.9946230241426821, 0.6727812317954857, 1.1277195536713025, 0.6727812317954857, 0.5876174577844024, 1.1277195536713025, 0.8752156373168588, 1.1277195536713025, 1.1277195536713025, 1.1277195536713025, 0.7683044943185703, 0.7683044943185703, 1.1277195536713025, 1.8240837639143594, 1.4402028887333522, 1.8240837639143594, 0.9946230241426821, 1.4402028887333522]
    # shallow NVs
    spot_weights = [0.4562224464789493, 1.6258105373893232, 1.2836565287202242, 1.4460863942325919, 1.6258105373893232, 1.6258105373893232, 1.6258105373893232, 0.4562224464789493, 0.2551395620216251, 1.6258105373893232, 1.6258105373893232, 0.4562224464789493, 0.5996516375232777, 1.1371125765502164, 0.684791766418767, 1.6258105373893232, 1.2836565287202242, 1.4460863942325919, 1.6258105373893232, 1.6258105373893232, 0.13533286406058123, 0.8865100525378773, 1.2836565287202242, 0.8865100525378773, 1.6258105373893232, 1.4460863942325919, 0.7800819423907052, 0.684791766418767, 0.8865100525378773, 1.4460863942325919, 1.6258105373893232, 1.6258105373893232, 1.0051393306874856, 0.5237449473097014, 1.6258105373893232, 0.684791766418767, 0.18716337758948745, 1.4460863942325919, 1.4460863942325919, 0.4562224464789493, 0.5237449473097014, 1.6258105373893232, 0.15945232430172335, 1.1371125765502164, 0.8865100525378773, 1.2836565287202242, 1.2836565287202242, 1.0051393306874856, 1.6258105373893232, 1.1371125765502164, 1.4460863942325919, 1.0051393306874856, 0.2551395620216251, 0.8865100525378773, 0.39629847091687376, 1.4460863942325919, 1.6258105373893232, 1.0051393306874856, 0.8865100525378773, 0.5996516375232777, 1.0051393306874856, 0.5996516375232777, 0.684791766418767, 1.2836565287202242, 0.684791766418767, 1.1371125765502164, 0.5996516375232777, 1.4460863942325919, 1.4460863942325919, 1.2836565287202242, 0.8865100525378773, 1.1371125765502164, 1.0051393306874856, 0.684791766418767, 1.4460863942325919, 0.5237449473097014, 0.2963997197412046, 1.2836565287202242, 0.7800819423907052, 1.1371125765502164, 0.8865100525378773, 0.2551395620216251, 0.2551395620216251, 0.4562224464789493, 0.39629847091687376, 0.8865100525378773, 0.2963997197412046, 1.6258105373893232, 0.4562224464789493, 0.684791766418767, 0.5996516375232777, 1.1371125765502164, 0.2551395620216251, 0.11440873934329801, 0.5996516375232777, 0.4562224464789493, 1.6258105373893232, 0.11440873934329801, 1.2836565287202242, 1.4460863942325919, 0.34324731076342113, 0.684791766418767, 0.5996516375232777, 0.5237449473097014, 0.5237449473097014, 0.5237449473097014, 0.8865100525378773, 0.5996516375232777, 1.6258105373893232, 1.6258105373893232, 1.2836565287202242, 0.4562224464789493, 0.7800819423907052, 1.1371125765502164, 0.18716337758948745, 1.2836565287202242, 0.4562224464789493, 1.0051393306874856, 1.4460863942325919, 1.4460863942325919, 0.684791766418767, 0.8865100525378773, 1.2836565287202242, 0.7800819423907052, 1.4460863942325919, 1.0051393306874856, 1.6258105373893232, 1.6258105373893232, 0.684791766418767, 1.1371125765502164, 1.4460863942325919, 0.7800819423907052, 1.6258105373893232, 1.1371125765502164, 1.1371125765502164, 0.7800819423907052, 0.2963997197412046, 0.34324731076342113, 1.6258105373893232, 0.684791766418767, 0.684791766418767, 1.0051393306874856, 1.1371125765502164, 1.6258105373893232, 1.6258105373893232, 1.1371125765502164, 1.4460863942325919, 1.6258105373893232, 1.2836565287202242, 1.1371125765502164, 1.6258105373893232, 1.6258105373893232, 1.6258105373893232, 0.2963997197412046, 1.0051393306874856, 1.4460863942325919, 0.7800819423907052, 1.0051393306874856, 0.34324731076342113, 1.0051393306874856, 0.8865100525378773, 0.4562224464789493, 1.6258105373893232, 1.2836565287202242, 1.2836565287202242, 0.18716337758948745, 1.0051393306874856, 1.1371125765502164, 1.0051393306874856]
    # fmt: on
    # fmt: off
    # print(spot_weights)
    # sys.exit()
    # spot_weights = curve_extreme_weights_simple(spot_weights)
    # print(f"median: {np.median(spot_weights)}")
    # print(f"mx: {np.max(spot_weights)}")
    # print(f"mn: {np.min(spot_weights)}")
    # norm_spot_weights = spot_weights / np.sum(spot_weights)
    # norm_spot_weights = np.array(norm_spot_weights)
    # aom_voltage = 0.4129
    # a, b, c = [3.7e5, 6.97, 8e-14]
    # total_power = a * (aom_voltage**b) + c
    # print(total_power)
    # sys.exit()
    # nv_powers = norm_spot_weights * total_power
    # drop_indices = [17, 55, 64, 72, 87, 89, 96, 99, 112, 114, 116]
    # spot_weights = [
    #     val for ind, val in enumerate(spot_weights) if ind not in drop_indices
    # ]
    # nv_powers = [val for ind, val in enumerate(nv_powers) if ind not in drop_indices]
    # print(np.sum(nv_powers))
    # Indices to exclude (zero-based indexing)
    # 117 NVs
    # include_indices =[0, 1, 2, 3, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 28, 29, 31, 32, 33, 34, 36, 37, 39, 42, 44, 45, 46, 47, 48, 49, 51, 52, 53, 55, 56, 57, 58, 60, 61, 62, 64, 65, 66, 68, 69, 70, 71, 72, 73, 74, 75, 77, 79, 83, 84, 85, 88, 89, 90, 91, 92, 94, 95, 96, 97, 99, 100, 101, 102, 103, 105, 106, 107, 108, 109, 110, 111, 113, 114, 116, 117, 118, 120, 122, 123, 124, 125, 128, 131, 132, 134, 136, 137, 138, 140, 141, 142, 145, 146, 147, 148, 149, 152, 153, 154, 155, 156, 157, 158, 159]
    # 117NVs
    # include_indices = list(range(160))
    # final_drop_inds = [23, 73, 89, 99, 117, 120, 132, 137, 155, 157, 159]
    # include_indices = [ind for ind in include_indices if ind not in final_drop_inds]
    # fmt: on
    # Filter nv_coordinates and spot_weights to exclude the specified indices
    # nv_coordinates_filtered = np.array(
    #     [coord for i, coord in enumerate(nv_coordinates) if i in include_indices]
    # )
    # print(f"len nv_powers: {len(nv_powers)}")
    # print(f"len nv_powers: {len(nv_coordinates_filtered)}")
    # select_half_left_side_nvs_and_plot(nv_coordinates_filtered)
    # spot_weights_filtered = np.array(
    #     [weight for i, weight in enumerate(spot_weights) if i in include_indices]
    # )
    # nv_powers_filtered = np.array(
    #     [power for i, power in enumerate(nv_powers) if i in include_indices]
    # )

    # Apply linear weights to all counts
    # calcualted_spot_weights = linear_weights(filtered_reordered_counts, alpha=0.3)
    # updated_spot_weights = linear_weights(filtered_reordered_counts, alpha=0.6)

    # Create a copy or initialize spot weights for modification
    updated_spot_weights = curve_extreme_weights_simple(spot_weights)
    # drop_indices = [17, 55, 64, 72, 87, 89, 96, 99, 112, 114, 116]
    # updated_spot_weights = [
    #     val for ind, val in enumerate(updated_spot_weights) if ind not in drop_indices
    # ]

    # Update weights for the specified indices using the calculated weights
    # for idx in indices:
    #     if 0 <= idx < len(updated_spot_weights):  # Ensure index is within valid range
    #         updated_spot_weights[idx] = calcualted_spot_weights[idx]

    aom_voltage = 0.38  # Current AOM voltage
    # aom_voltage = 0.4118  # Current AOM voltage
    power_law_params = [3.7e5, 6.97, 8e-14]  # Example power-law fit parameters
    a, b, c = [3.7e5, 6.97, 8e-14]
    total_power = a * (aom_voltage) ** b + c
    print(total_power)
    sys.exit()
    # nv_weights, adjusted_aom_voltage = adjust_aom_voltage_for_slm(
    #     nv_amps_filtered, aom_voltage, power_law_params
    # )
    # filtered_total_power = np.sum(nv_powers)
    # adjusted_aom_voltage = ((filtered_total_power - c) / a) ** (1 / b)
    # print("Adjusted Voltages (V):", adjusted_aom_voltage)
    filtered_reordered_spot_weights = updated_spot_weights
    # print("nv_weights:", spot_weights)
    print("NV Index | Coords    |   previous weights")
    print("-" * 60)
    for idx, (coords, weight) in enumerate(
        zip(filtered_reordered_coords, filtered_reordered_spot_weights)
    ):
        print(f"{idx + 1:<8} | {coords} | {weight:.3f}")

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
    #     filename="slmsuite/nv_blob_detection/nv_blob_shallow_148nvs_reordered.npz",
    # )
    # save_results(
    #     nv_coordinates,
    #     filtered_reordered_counts,
    #     nv_weights,
    #     updated_spot_weights,
    #     filename="slmsuite/nv_blob_detection/nv_blob_filtered_160nvs_reordered_selected_117nvs_updated.npz",
    # filename="slmsuite/nv_blob_detection/nv_blob_filtered_160nvs_reordered.npz",
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
            str(idx),
            color="white",
            fontsize=6,
            ha="center",
        )
    # Plot histogram of the filtered integrated intensities using Seaborn
    # sns.set(style="whitegrid")

    # plt.figure(figsize=(6, 5))
    # sns.histplot(filtered_counts, bins=45, kde=False, color="blue")

    # plt.xlabel("Integrated Intensity")
    # plt.ylabel("Frequency")
    # plt.title("Histogram of Filtered Integrated Counts")
    plt.show(block=True)
