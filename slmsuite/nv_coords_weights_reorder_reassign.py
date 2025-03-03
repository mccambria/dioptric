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
    nv_coords = [reference_nv]  # Initialize with the reference NV
    # intensities = [integrated_intensities[nv_coordinates.index(reference_nv)]]

    # Ensure the reference NV is in the list or find the closest one
    # distances_to_ref = np.linalg.norm(nv_coordinates - np.array(reference_nv), axis=1)
    # closest_index = np.argmin(distances_to_ref)  # Index of the closest NV
    # reference_nv = nv_coordinates[closest_index]  # Use closest NV as reference
    # print(closest_index)
    # Initialize with reference NV and its corresponding intensity
    # nv_coords = [reference_nv]
    # intensities = [integrated_intensities[closest_index]]
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
            # intensities.append(integrated_intensities[idx])  # Store matching intensity

    # Reorder based on distance to the reference NV
    distances = [
        np.linalg.norm(np.array(coord) - np.array(reference_nv)) for coord in nv_coords
    ]
    sorted_indices = np.argsort(distances)
    print(sorted_indices)
    reordered_coords = [nv_coords[idx] for idx in sorted_indices]
    reordered_intensities = [integrated_intensities[idx] for idx in sorted_indices]

    return reordered_coords, reordered_intensities


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
    # data = dm.get_raw_data(file_id=1700710358100, load_npz=True)
    # data = dm.get_raw_data(file_id=1733432867671, load_npz=True)
    # data = dm.get_raw_data(file_id=1732420670067, load_npz=True)
    # data = dm.get_raw_data(file_id=1751170993877, load_npz=True)
    # data = dm.get_raw_data(file_id=1752794666146, load_npz=True)
    # data = dm.get_raw_data(file_id=1764727515943, load_npz=True)
    data = dm.get_raw_data(file_id=1791296034768, load_npz=True)

    img_array = np.array(data["ref_img_array"])
    # img_array = data["img_array"]

    # sys.exit()
    nv_coordinates, spot_weights = load_nv_coords(
        file_path="slmsuite/nv_blob_detection/nv_blob_shallow_rubin_310nvs.npz"
        # file_path="slmsuite/nv_blob_detection/nv_blob_rubin_shallow_312nvs_reordered.npz"
        # file_path="slmsuite/nv_blob_detection/nv_blob_filtered_160nvs_reordered.npz"
    )
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
        and all(1 <= x <= 249 for x in coord)
    ]

    # Ensure spot weights are filtered accordingly
    spot_weights_filtered = [
        weight
        for coord, weight in zip(nv_coordinates, spot_weights)
        if isinstance(coord, (list, tuple))
        and len(coord) == 2
        and all(1 <= x <= 249 for x in coord)
    ]

    # Replace original lists with filtered versions
    nv_coordinates = nv_coordinates_filtered
    spot_weights = spot_weights_filtered
    print(f"After filtering: {len(nv_coordinates_filtered)} NVs")

    # Filter and reorder NV coordinates based on reference NV
    # integrated_intensities = []
    sigma = 2.0
    reference_nv = [109.077, 120.824]
    # reference_nv = [117.596, 129.217]
    filtered_reordered_coords, filtered_reordered_spot_weights = (
        filter_and_reorder_nv_coords(
            nv_coordinates, spot_weights_filtered, reference_nv, min_distance=8
        )
    )

    # Integration over disk region around each NV coordinate
    filtered_reordered_counts = []
    integration_radius = 2.0
    for coord in filtered_reordered_coords:
        x, y = coord[:2]  # Assuming `coord` contains at least two elements (y, x)
        rr, cc = disk((y, x), integration_radius, shape=img_array.shape)
        sum_value = np.sum(img_array[rr, cc])
        filtered_reordered_counts.append(sum_value)

    # calcualte spot weight  based on
    calcualted_spot_weights = linear_weights(filtered_reordered_spot_weights, alpha=0.3)
    filtered_reordered_spot_weights = calcualted_spot_weights
    # Manually remove NVs with specified indices
    # indices_to_remove = [2]  # Example indices to remove
    # filtered_reordered_coords = [
    #     coord
    #     for i, coord in enumerate(filtered_reordered_coords)
    #     if i not in indices_to_remove
    # ]
    # filtered_reordered_spot_weights = [
    #     count
    #     for i, count in enumerate(filtered_reordered_spot_weights)
    #     if i not in indices_to_remove
    # ]
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
    # spots weights for 117 NVs before Birge shutdown
    # spot_weights = [0.7466728351068052, 0.7466728351068052, 0.6384167684513203, 1.1708211389400445, 1.5561808283609733, 1.1708211389400445, 0.6384167684513203, 0.7466728351068052, 1.1708211389400445, 0.7466728351068052, 0.7466728351068052, 0.6384167684513203, 1.1708211389400445, 0.7466728351068052, 0.7466728351068052, 0.7466728351068052, 0.8702844369863524, 0.7466728351068052, 0.8702844369863524, 0.6384167684513203, 0.8702844369863524, 0.6384167684513203, 1.0110233833356788, 0.6384167684513203, 0.6384167684513203, 0.8702844369863524, 1.5561808283609733, 0.8702844369863524, 0.8702844369863524, 1.3517793512995497, 0.7466728351068052, 0.8702844369863524, 1.5561808283609733, 0.7466728351068052, 0.8702844369863524, 0.7466728351068052, 1.0110233833356788, 0.7466728351068052, 0.5438939973931943, 0.7466728351068052, 3.023897571362437, 0.7466728351068052, 0.8702844369863524, 0.8702844369863524, 0.7466728351068052, 1.0110233833356788, 0.6384167684513203, 0.7466728351068052, 0.7466728351068052, 1.0110233833356788, 0.5438939973931943, 0.7466728351068052, 1.5561808283609733, 1.5561808283609733, 1.0110233833356788, 0.7466728351068052, 1.3517793512995497, 0.6384167684513203, 1.0110233833356788, 0.7466728351068052, 0.7466728351068052, 0.8702844369863524, 1.0110233833356788, 1.5561808283609733, 1.1708211389400445, 1.0110233833356788, 0.6384167684513203, 0.8702844369863524, 0.8702844369863524, 0.7466728351068052, 0.6384167684513203, 0.7466728351068052, 0.8702844369863524, 1.0110233833356788, 0.8702844369863524, 0.8702844369863524, 0.7466728351068052, 1.3517793512995497, 1.5561808283609733, 1.0110233833356788, 0.8702844369863524, 0.7466728351068052, 0.8702844369863524, 1.0110233833356788, 0.8702844369863524, 1.0110233833356788, 1.1708211389400445, 1.5561808283609733, 0.8702844369863524, 0.8702844369863524, 1.3517793512995497, 0.7466728351068052, 1.0110233833356788, 1.0110233833356788, 1.5561808283609733, 2.0454197052578524, 0.7466728351068052, 1.7865009761020008, 2.0454197052578524, 1.1708211389400445, 1.1708211389400445, 1.1708211389400445, 1.0110233833356788, 1.5561808283609733, 0.8702844369863524, 1.3517793512995497, 1.0110233833356788, 0.7466728351068052, 2.0454197052578524, 0.7466728351068052, 0.8702844369863524, 0.7466728351068052, 1.1708211389400445, 1.7865009761020008, 0.7466728351068052, 0.8702844369863524, 1.1708211389400445]
    spot_weights = [0.3268459239007881, 1.7928139326333443, 1.2539168761234627, 1.7928139326333443, 0.6612479042019478, 0.5030856545812062, 0.3268459239007881, 1.5946285103642925, 1.4155138353948802, 0.577544072380617, 0.577544072380617, 0.3785057034159371, 1.7928139326333443, 1.2539168761234627, 1.7928139326333443, 0.5030856545812062, 1.5946285103642925, 1.7928139326333443, 1.1083873273376117, 0.577544072380617, 1.7928139326333443, 0.3268459239007881, 1.1083873273376117, 0.6612479042019478, 0.3785057034159371, 1.1083873273376117, 0.8602120251720561, 1.7928139326333443, 0.43700628320563345, 0.43700628320563345, 0.3785057034159371, 0.577544072380617, 0.9775724397515285, 0.20638881548889623, 1.2539168761234627, 1.5946285103642925, 0.2813475193074132, 0.7551336343037693, 0.6612479042019478, 0.2813475193074132, 1.7928139326333443, 0.20638881548889623, 0.577544072380617, 0.43700628320563345, 1.7928139326333443, 0.7551336343037693, 0.43700628320563345, 0.20638881548889623, 1.5946285103642925, 1.4155138353948802, 0.9775724397515285, 1.2539168761234627, 1.1083873273376117, 0.577544072380617, 0.8602120251720561, 1.5946285103642925, 0.3268459239007881, 0.43700628320563345, 1.2539168761234627, 1.1083873273376117, 0.2813475193074132, 1.1083873273376117, 0.6612479042019478, 0.24138608133564415, 0.24138608133564415, 1.2539168761234627, 1.2539168761234627, 1.5946285103642925, 1.7928139326333443, 1.5946285103642925, 0.2813475193074132, 0.9775724397515285, 0.7551336343037693, 1.4155138353948802, 0.3785057034159371, 1.5946285103642925, 0.6612479042019478, 0.8602120251720561, 1.7928139326333443, 0.5030856545812062, 0.5030856545812062, 1.5946285103642925, 0.3785057034159371, 0.6612479042019478, 1.5946285103642925, 0.577544072380617, 1.5946285103642925, 0.9775724397515285, 1.7928139326333443, 0.43700628320563345, 0.3268459239007881, 1.7928139326333443, 0.43700628320563345, 1.7928139326333443, 0.577544072380617, 1.5946285103642925, 0.43700628320563345, 1.7928139326333443, 1.7928139326333443, 0.7551336343037693, 1.4155138353948802, 0.9775724397515285, 1.1083873273376117, 1.7928139326333443, 0.43700628320563345, 0.20638881548889623, 1.7928139326333443, 1.7928139326333443, 0.43700628320563345, 0.2813475193074132, 1.7928139326333443, 0.43700628320563345, 1.7928139326333443, 0.24138608133564415, 0.43700628320563345, 0.2813475193074132, 0.9775724397515285, 1.1083873273376117, 0.5030856545812062, 0.7551336343037693, 1.5946285103642925, 0.5030856545812062, 0.6612479042019478, 0.8602120251720561, 0.3785057034159371, 0.3268459239007881, 1.2539168761234627, 1.7928139326333443, 1.2539168761234627, 0.17583128047499216, 0.43700628320563345, 0.3785057034159371, 0.3268459239007881, 0.6612479042019478, 0.5030856545812062, 1.1083873273376117, 0.17583128047499216, 1.7928139326333443, 0.9775724397515285, 0.2813475193074132, 1.7928139326333443, 0.3268459239007881, 1.7928139326333443, 0.43700628320563345, 1.2539168761234627, 0.3785057034159371, 1.7928139326333443, 0.9775724397515285, 1.7928139326333443, 0.3785057034159371, 1.7928139326333443, 0.43700628320563345, 1.7928139326333443, 1.4155138353948802, 0.2813475193074132, 0.20638881548889623, 1.7928139326333443, 1.7928139326333443, 1.2539168761234627, 1.5946285103642925, 1.4155138353948802, 1.7928139326333443, 0.9775724397515285, 1.5946285103642925, 1.5946285103642925, 0.43700628320563345, 0.7551336343037693, 1.4155138353948802, 0.3268459239007881, 1.7928139326333443, 1.5946285103642925, 0.9775724397515285, 0.43700628320563345, 1.7928139326333443, 1.5946285103642925, 1.1083873273376117, 0.43700628320563345, 0.3268459239007881, 1.7928139326333443, 1.7928139326333443, 1.5946285103642925, 0.3268459239007881, 0.5030856545812062, 0.20638881548889623, 1.4155138353948802, 0.3785057034159371, 0.5030856545812062, 0.5030856545812062, 1.5946285103642925, 0.1492342672477611, 0.1492342672477611, 1.7928139326333443, 1.1083873273376117, 0.9775724397515285, 0.43700628320563345, 1.5946285103642925, 1.4155138353948802, 0.7551336343037693, 1.7928139326333443, 0.6612479042019478, 1.7928139326333443, 1.7928139326333443, 1.2539168761234627, 1.7928139326333443, 1.2539168761234627, 0.9775724397515285, 0.2813475193074132, 1.7928139326333443, 1.1083873273376117, 0.8602120251720561, 1.1083873273376117, 0.9775724397515285, 0.3268459239007881, 0.577544072380617, 1.5946285103642925, 1.7928139326333443, 1.2539168761234627, 1.7928139326333443, 0.3785057034159371, 1.7928139326333443, 0.43700628320563345, 0.6612479042019478, 0.5030856545812062, 0.3785057034159371, 1.7928139326333443, 1.2539168761234627, 1.5946285103642925, 0.17583128047499216, 1.5946285103642925, 0.2813475193074132, 1.1083873273376117, 0.577544072380617, 0.24138608133564415, 0.3785057034159371, 0.5030856545812062, 1.2539168761234627, 1.5946285103642925, 1.7928139326333443, 0.8602120251720561, 0.8602120251720561, 0.7551336343037693, 0.6612479042019478, 0.577544072380617, 0.7551336343037693, 1.7928139326333443, 1.1083873273376117, 1.7928139326333443, 0.1492342672477611, 1.7928139326333443, 1.7928139326333443, 0.20638881548889623, 0.3268459239007881, 0.3268459239007881, 1.7928139326333443, 1.7928139326333443, 1.5946285103642925, 0.8602120251720561, 0.2813475193074132, 0.577544072380617, 1.2539168761234627, 0.10621335884311792, 1.7928139326333443, 0.3268459239007881, 1.7928139326333443, 1.1083873273376117, 0.2813475193074132, 1.4155138353948802, 1.2539168761234627, 0.5030856545812062, 1.1083873273376117, 0.20638881548889623, 1.7928139326333443, 0.577544072380617, 1.7928139326333443, 0.6612479042019478, 0.3268459239007881, 1.7928139326333443, 0.43700628320563345, 1.7928139326333443, 1.5946285103642925, 0.8602120251720561, 0.9775724397515285, 0.20638881548889623, 0.6612479042019478, 1.4155138353948802, 1.5946285103642925, 0.20638881548889623, 0.6612479042019478, 1.7928139326333443, 1.7928139326333443, 0.20638881548889623, 1.4155138353948802, 1.1083873273376117, 1.7928139326333443, 0.3268459239007881, 1.1083873273376117, 1.1083873273376117, 0.12616081467834883, 0.10621335884311792, 1.7928139326333443, 0.8602120251720561, 1.7928139326333443, 0.5030856545812062, 1.7928139326333443, 1.5946285103642925, 0.9775724397515285, 1.4155138353948802, 1.2539168761234627, 0.3785057034159371]
    # sys.exit()
    # nv_powers = [val for ind, val in enumerate(nv_powers) if ind not in drop_indices]
    # print(np.sum(nv_powers))
    # 117 NVs
    # include_indices =[0, 1, 2, 3, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 28, 29, 31, 32, 33, 34, 36, 37, 39, 42, 44, 45, 46, 47, 48, 49, 51, 52, 53, 55, 56, 57, 58, 60, 61, 62, 64, 65, 66, 68, 69, 70, 71, 72, 73, 74, 75, 77, 79, 83, 84, 85, 88, 89, 90, 91, 92, 94, 95, 96, 97, 99, 100, 101, 102, 103, 105, 106, 107, 108, 109, 110, 111, 113, 114, 116, 117, 118, 120, 122, 123, 124, 125, 128, 131, 132, 134, 136, 137, 138, 140, 141, 142, 145, 146, 147, 148, 149, 152, 153, 154, 155, 156, 157, 158, 159]
    # SHALLOW nvS
    include_indices = [0, 1, 5, 6, 10, 17, 19, 20, 24, 26, 29, 35, 36, 43, 44, 46, 48, 49, 52, 53, 55, 57, 58, 62, 64, 65, 66, 68, 70, 72, 73, 74, 75, 78, 80, 82, 83, 90, 91, 93, 94, 95, 98, 99, 102, 103, 110, 111, 112, 113, 116, 122, 124, 126, 129, 130, 131, 136, 138, 142, 146]
    # include_indices = list(range(160))
    include_indices = [0, 3, 4, 6, 7, 8, 13, 14, 19, 20, 22, 25, 26, 30, 31, 33, 36, 38, 39, 42, 43, 50, 51, 52, 58, 59, 61, 62, 63, 64, 65, 74, 75, 76, 77, 78, 79, 83, 84, 86, 88, 90, 91, 92, 95, 96, 99, 101, 105, 107, 109, 110, 112, 113, 119, 121, 126, 127, 128, 132, 133, 134, 136, 139, 140, 141, 143, 145, 146]

    # include_indices = list(range(147))
    # include_indices = [ind for ind in include_indices if ind not in final_drop_inds]
    # fmt: on

    print(f"len nv_powers: {len(nv_coordinates_filtered)}")
    # select_half_left_side_nvs_and_plot(nv_coordinates_filtered)
    # spot_weights_filtered = np.array(
    #     [weight for i, weight in enumerate(spot_weights) if i in include_indices]
    # )

    aom_voltage = 0.4311  #
    aom_voltage = 0.4524
    a, b, c = [3.7e5, 6.97, 8e-14]
    total_power = a * (aom_voltage) ** b + c
    print(total_power)
    nv_powers = np.array(spot_weights)
    nv_powers = nv_powers * total_power  # Apply linear weights to all counts
    # print(nv_powers)
    # calcualted_spot_weights = linear_weights(filtered_reordered_counts, alpha=0.3)
    # updated_spot_weights = linear_weights(filtered_reordered_counts, alpha=0.6)
    nv_powers_filtered = np.array(
        [power for i, power in enumerate(nv_powers) if i in include_indices]
    )
    # print(nv_powers_filtered)
    # Create a copy or initialize spot weights for modification
    updated_spot_weights = curve_extreme_weights_simple(
        spot_weights, scaling_factor=1.0
    )

    # reference_nv = [117.596, 129.217]
    # filtered_reordered_coords, filtered_reordered_spot_weights = (
    #     filter_and_reorder_nv_coords(
    #         filtered_reordered_coords,
    #         updated_spot_weights,
    #         reference_nv,
    #         min_distance=3,
    #     )
    # )
    print(filter_and_reorder_nv_coords)
    # updated_spot_weights = np.array(
    #     [w for i, w in enumerate(updated_spot_weights_0) if i in include_indices]
    # )
    # updated_spot_weights = spot_weights
    # updated_spot_weights = curve_extreme_weights_simple(nv_powers)
    # updated_spot_weights = curve_inverse_counts(filtered_reordered_spot_weights)
    # drop_indices = [17, 55, 64, 72, 87, 89, 96, 99, 112, 114, 116]
    # updated_spot_weights = [
    #     val for ind, val in enumerate(updated_spot_weights) if ind not in drop_indices
    # ]

    # Update weights for the specified indices using the calculated weights
    # for idx in indices:
    #     if 0 <= idx < len(updated_spot_weights):  # Ensure index is within valid range
    #         updated_spot_weights[idx] = calcualted_spot_weights[idx]

    aom_voltage = 0.4  # Current AOM voltage
    # aom_voltage = 0.4118  # Current AOM voltage
    power_law_params = [3.7e5, 6.97, 8e-14]  # Example power-law fit parameters
    a, b, c = [3.7e5, 6.97, 8e-14]
    total_power = a * (aom_voltage) ** b + c
    print(total_power)
    # sys.exit()
    # nv_weights, adjusted_aom_voltage = adjust_aom_voltage_for_slm(
    #     nv_amps_filtered, aom_voltage, power_law_params
    # )
    # adjusted_nv_powers = adjusted_nv_powers * total_power
    filtered_total_power = np.sum(nv_powers_filtered) / len(nv_coordinates_filtered)
    adjusted_aom_voltage = ((filtered_total_power - c) / a) ** (1 / b)
    # print("Adjusted Voltages (V):", adjusted_aom_voltage)
    filtered_reordered_spot_weights = updated_spot_weights
    # print("nv_weights:", spot_weights)
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

    save_results(
        filtered_reordered_coords,
        filtered_reordered_spot_weights,
        filename="slmsuite/nv_blob_detection/nv_blob_rubin_shallow_255nvs_reordered_updated.npz",
    )
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
            fontsize=8,
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
