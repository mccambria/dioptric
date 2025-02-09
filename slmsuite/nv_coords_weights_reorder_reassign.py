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
    # data = dm.get_raw_data(file_id=1764727515943, load_npz=True)  # comniened
    data = dm.get_raw_data(file_id=1766788934308, load_npz=True)

    img_array = data["ref_img_array"]
    # img_array = data["img_array"]

    # sys.exit()
    nv_coordinates, spot_weights = load_nv_coords(
        file_path="slmsuite/nv_blob_detection/nv_blob_shallow_149nvs.npz"
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
        and all(0 <= x <= 250 for x in coord)
    ]

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
    filtered_reordered_counts = []
    integration_radius = 2.0
    for coord in filtered_reordered_coords:
        x, y = coord[:2]  # Assuming `coord` contains at least two elements (y, x)
        rr, cc = disk((y, x), integration_radius, shape=img_array.shape)
        sum_value = np.sum(img_array[rr, cc])
        filtered_reordered_counts.append(sum_value)  # Append to the list

    # Manually remove NVs with specified indices
    indices_to_remove = [2]  # Example indices to remove
    filtered_reordered_coords = [
        coord
        for i, coord in enumerate(filtered_reordered_coords)
        if i not in indices_to_remove
    ]
    filtered_reordered_spot_weights = [
        count
        for i, count in enumerate(filtered_reordered_counts)
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
    # fidelity
    # spot_weights = [0.56184079, 1.45107477, 0.85728647, 0.64867703, 0.22151819, 0.64867703, 0.85728647, 0.74676297, 0.41762254, 0.85728647, 0.56184079, 0.74676297, 0.56184079, 2.10113985, 1.27686902, 0.56184079, 0.56184079, 0.56184079, 0.74676297, 0.85728647, 0.64867703, 0.41762254, 0.64867703, 0.98153355, 2.36701149, 0.98153355, 0.56184079, 0.41762254, 0.41762254, 0.74676297, 2.66119108, 0.4851588, 0.35830516, 0.85728647, 0.85728647, 0.64867703, 0.56184079, 0.4851588, 1.27686902, 0.64867703, 0.74676297, 2.10113985, 0.98153355, 0.41762254, 0.4851588, 0.85728647, 2.36701149, 0.41762254, 0.74676297, 1.64525187, 0.74676297, 0.98153355, 2.66119108, 0.74676297, 1.12089429, 0.64867703, 0.22151819, 0.85728647, 1.45107477, 0.98153355, 0.64867703, 1.86127091, 1.64525187, 0.64867703, 0.74676297, 0.56184079, 0.18726876, 1.45107477, 0.74676297, 0.74676297, 0.74676297, 0.56184079, 0.35830516, 0.98153355, 0.98153355, 0.74676297, 0.74676297, 2.66119108, 1.45107477, 1.86127091, 0.64867703, 2.66119108, 0.85728647, 1.27686902, 1.27686902, 1.12089429, 0.64867703, 1.27686902, 1.27686902, 1.86127091, 0.56184079, 1.27686902, 0.85728647, 0.56184079, 1.64525187, 0.64867703, 0.85728647, 0.74676297, 1.27686902, 2.66119108, 0.22151819, 0.56184079, 0.15765944, 0.4851588, 1.45107477, 0.56184079, 0.56184079, 0.4851588, 0.85728647, 2.10113985, 0.74676297, 0.74676297, 0.98153355, 0.41762254, 2.36701149, 2.36701149, 1.64525187, 1.45107477, 0.64867703, 0.64867703, 2.66119108, 0.56184079, 0.41762254, 0.74676297, 1.45107477, 0.56184079, 1.86127091, 0.41762254, 0.74676297, 0.98153355, 0.74676297, 1.86127091, 0.74676297, 0.74676297, 1.45107477, 2.36701149, 0.56184079, 0.74676297, 0.56184079, 0.74676297, 0.56184079, 0.98153355, 0.85728647, 0.30635643, 0.85728647, 0.4851588, 0.4851588, 2.10113985, 0.30635643, 1.45107477, 0.98153355, 1.64525187, 0.85728647, 0.64867703, 0.74676297, 0.35830516, 1.45107477, 2.66119108, 1.12089429, 2.36701149]
    # spot_weights = [0.9946230241426821, 0.8752156373168588, 0.7683044943185703, 0.5876174577844024, 1.6224416389703067, 0.7683044943185703, 0.5876174577844024, 1.4402028887333522, 0.6727812317954857, 1.1277195536713025, 0.8752156373168588, 0.5876174577844024, 0.9946230241426821, 0.9946230241426821, 1.4402028887333522, 0.7683044943185703, 0.5876174577844024, 0.511860354092575, 1.6224416389703067, 1.2757873939965607, 0.3325466922344282, 0.5876174577844024, 0.5876174577844024, 1.1277195536713025, 0.7683044943185703, 0.6727812317954857, 0.8752156373168588, 0.5876174577844024, 1.6224416389703067, 0.5876174577844024, 0.9946230241426821, 1.2757873939965607, 0.8752156373168588, 0.7683044943185703, 0.7683044943185703, 1.2757873939965607, 0.5876174577844024, 0.6727812317954857, 0.8752156373168588, 0.44462844214574804, 0.7683044943185703, 0.8752156373168588, 1.2757873939965607, 0.7683044943185703, 0.6727812317954857, 1.8240837639143594, 0.7683044943185703, 0.6727812317954857, 0.6727812317954857, 1.1277195536713025, 0.7683044943185703, 0.8752156373168588, 1.6224416389703067, 0.7683044943185703, 1.8240837639143594, 0.9946230241426821, 0.24559627956065677, 1.2757873939965607, 1.8240837639143594, 0.6727812317954857, 0.8752156373168588, 1.2757873939965607, 1.8240837639143594, 1.2757873939965607, 0.9946230241426821, 1.8240837639143594, 0.44462844214574804, 1.8240837639143594, 0.5876174577844024, 1.2757873939965607, 0.7683044943185703, 0.5876174577844024, 0.6727812317954857, 0.9946230241426821, 1.4402028887333522, 0.6727812317954857, 0.9946230241426821, 1.6224416389703067, 0.44462844214574804, 0.7683044943185703, 0.511860354092575, 1.1277195536713025, 0.8752156373168588, 1.6224416389703067, 1.8240837639143594, 1.4402028887333522, 1.8240837639143594, 1.8240837639143594, 1.1277195536713025, 1.2757873939965607, 0.7683044943185703, 1.2757873939965607, 0.6727812317954857, 0.5876174577844024, 0.7683044943185703, 0.3325466922344282, 0.6727812317954857, 0.5876174577844024, 1.6224416389703067, 1.2757873939965607, 0.7683044943185703, 0.6727812317954857, 0.38510750925271625, 0.511860354092575, 0.7683044943185703, 0.9946230241426821, 1.4402028887333522, 0.511860354092575, 0.6727812317954857, 1.1277195536713025, 1.2757873939965607, 0.44462844214574804, 0.38510750925271625, 0.511860354092575, 1.8240837639143594, 1.2757873939965607, 1.8240837639143594, 1.8240837639143594, 0.511860354092575, 0.511860354092575, 0.6727812317954857, 0.5876174577844024, 0.5876174577844024, 1.4402028887333522, 1.6224416389703067, 0.511860354092575, 1.8240837639143594, 0.9946230241426821, 0.6727812317954857, 0.8752156373168588, 0.8752156373168588, 0.9946230241426821, 1.1277195536713025, 1.4402028887333522, 1.2757873939965607, 1.8240837639143594, 0.9946230241426821, 0.8752156373168588, 1.6224416389703067, 0.7683044943185703, 0.9946230241426821, 0.6727812317954857, 0.9946230241426821, 0.6727812317954857, 1.1277195536713025, 0.6727812317954857, 0.5876174577844024, 1.1277195536713025, 0.8752156373168588, 1.1277195536713025, 1.1277195536713025, 1.1277195536713025, 0.7683044943185703, 0.7683044943185703, 1.1277195536713025, 1.8240837639143594, 1.4402028887333522, 1.8240837639143594, 0.9946230241426821, 1.4402028887333522]
    # shallow NVs
    # spot_weights = [0.4562224464789493, 1.6258105373893232, 1.2836565287202242, 1.4460863942325919, 1.6258105373893232, 1.6258105373893232, 1.6258105373893232, 0.4562224464789493, 0.2551395620216251, 1.6258105373893232, 1.6258105373893232, 0.4562224464789493, 0.5996516375232777, 1.1371125765502164, 0.684791766418767, 1.6258105373893232, 1.2836565287202242, 1.4460863942325919, 1.6258105373893232, 1.6258105373893232, 0.13533286406058123, 0.8865100525378773, 1.2836565287202242, 0.8865100525378773, 1.6258105373893232, 1.4460863942325919, 0.7800819423907052, 0.684791766418767, 0.8865100525378773, 1.4460863942325919, 1.6258105373893232, 1.6258105373893232, 1.0051393306874856, 0.5237449473097014, 1.6258105373893232, 0.684791766418767, 0.18716337758948745, 1.4460863942325919, 1.4460863942325919, 0.4562224464789493, 0.5237449473097014, 1.6258105373893232, 0.15945232430172335, 1.1371125765502164, 0.8865100525378773, 1.2836565287202242, 1.2836565287202242, 1.0051393306874856, 1.6258105373893232, 1.1371125765502164, 1.4460863942325919, 1.0051393306874856, 0.2551395620216251, 0.8865100525378773, 0.39629847091687376, 1.4460863942325919, 1.6258105373893232, 1.0051393306874856, 0.8865100525378773, 0.5996516375232777, 1.0051393306874856, 0.5996516375232777, 0.684791766418767, 1.2836565287202242, 0.684791766418767, 1.1371125765502164, 0.5996516375232777, 1.4460863942325919, 1.4460863942325919, 1.2836565287202242, 0.8865100525378773, 1.1371125765502164, 1.0051393306874856, 0.684791766418767, 1.4460863942325919, 0.5237449473097014, 0.2963997197412046, 1.2836565287202242, 0.7800819423907052, 1.1371125765502164, 0.8865100525378773, 0.2551395620216251, 0.2551395620216251, 0.4562224464789493, 0.39629847091687376, 0.8865100525378773, 0.2963997197412046, 1.6258105373893232, 0.4562224464789493, 0.684791766418767, 0.5996516375232777, 1.1371125765502164, 0.2551395620216251, 0.11440873934329801, 0.5996516375232777, 0.4562224464789493, 1.6258105373893232, 0.11440873934329801, 1.2836565287202242, 1.4460863942325919, 0.34324731076342113, 0.684791766418767, 0.5996516375232777, 0.5237449473097014, 0.5237449473097014, 0.5237449473097014, 0.8865100525378773, 0.5996516375232777, 1.6258105373893232, 1.6258105373893232, 1.2836565287202242, 0.4562224464789493, 0.7800819423907052, 1.1371125765502164, 0.18716337758948745, 1.2836565287202242, 0.4562224464789493, 1.0051393306874856, 1.4460863942325919, 1.4460863942325919, 0.684791766418767, 0.8865100525378773, 1.2836565287202242, 0.7800819423907052, 1.4460863942325919, 1.0051393306874856, 1.6258105373893232, 1.6258105373893232, 0.684791766418767, 1.1371125765502164, 1.4460863942325919, 0.7800819423907052, 1.6258105373893232, 1.1371125765502164, 1.1371125765502164, 0.7800819423907052, 0.2963997197412046, 0.34324731076342113, 1.6258105373893232, 0.684791766418767, 0.684791766418767, 1.0051393306874856, 1.1371125765502164, 1.6258105373893232, 1.6258105373893232, 1.1371125765502164, 1.4460863942325919, 1.6258105373893232, 1.2836565287202242, 1.1371125765502164, 1.6258105373893232, 1.6258105373893232, 1.6258105373893232, 0.2963997197412046, 1.0051393306874856, 1.4460863942325919, 0.7800819423907052, 1.0051393306874856, 0.34324731076342113, 1.0051393306874856, 0.8865100525378773, 0.4562224464789493, 1.6258105373893232, 1.2836565287202242, 1.2836565287202242, 0.18716337758948745, 1.0051393306874856, 1.1371125765502164, 1.0051393306874856]
    # spot_weights = [0.23048653978603376, 0.42844245824494454, 1.2814403012925042, 0.3688013647365374, 0.6594644105825402, 0.4961601236932695, 0.42844245824494454, 0.8667896916211687, 0.42844245824494454, 2.350090829666805, 0.6594644105825402, 0.2705425553636888, 0.7570674254168084, 0.42844245824494454, 1.2814403012925042, 0.8667896916211687, 2.350090829666805, 0.6594644105825402, 1.643683428661117, 0.6594644105825402, 0.6594644105825402, 0.8667896916211687, 0.3688013647365374, 0.4961601236932695, 2.090301603930463, 0.31641834427450277, 0.16537654479480055, 0.6594644105825402, 1.4529175874202336, 1.643683428661117, 0.8667896916211687, 0.8667896916211687, 2.350090829666805, 0.4961601236932695, 0.7570674254168084, 0.6594644105825402, 1.2814403012925042, 1.4529175874202336, 2.350090829666805, 2.090301603930463, 0.6594644105825402, 1.8555110618433197, 0.4961601236932695, 0.4961601236932695, 1.643683428661117, 0.4961601236932695, 0.7570674254168084, 1.1275996661608945, 1.1275996661608945, 1.8555110618433197, 0.9898587894972546, 1.643683428661117, 1.1275996661608945, 0.16537654479480055, 1.8555110618433197, 0.31641834427450277, 0.42844245824494454, 0.6594644105825402, 0.4961601236932695, 0.7570674254168084, 1.8555110618433197, 0.4961601236932695, 1.2814403012925042, 1.643683428661117, 0.8667896916211687, 0.31641834427450277, 0.8667896916211687, 0.8667896916211687, 0.9898587894972546, 1.1275996661608945, 0.6594644105825402, 1.4529175874202336, 0.31641834427450277, 0.7570674254168084, 0.9898587894972546, 0.8667896916211687, 0.4961601236932695, 1.4529175874202336, 2.350090829666805, 2.090301603930463, 0.42844245824494454, 2.350090829666805, 0.5728449784857683, 0.42844245824494454, 0.8667896916211687, 0.42844245824494454, 1.1275996661608945, 0.23048653978603376, 1.643683428661117, 0.5728449784857683, 0.31641834427450277, 0.42844245824494454, 0.8667896916211687, 0.8667896916211687, 0.4961601236932695, 0.5728449784857683, 0.4961601236932695, 1.2814403012925042, 0.19562213152586785, 1.1275996661608945, 1.643683428661117, 0.3688013647365374, 0.6594644105825402, 0.31641834427450277, 0.7570674254168084, 1.1275996661608945, 0.8667896916211687, 0.5728449784857683, 0.7570674254168084, 0.8667896916211687, 0.5728449784857683, 1.8555110618433197, 0.6594644105825402, 1.8555110618433197, 0.42844245824494454, 2.350090829666805, 0.5728449784857683, 0.9898587894972546, 0.5728449784857683, 0.23048653978603376, 1.2814403012925042, 0.8667896916211687, 1.1275996661608945, 0.13922863720647466, 0.8667896916211687, 0.7570674254168084, 1.1275996661608945, 1.8555110618433197, 0.9898587894972546, 0.23048653978603376, 1.643683428661117, 0.8667896916211687, 1.2814403012925042, 0.8667896916211687, 0.9898587894972546, 1.1275996661608945, 0.9898587894972546, 1.4529175874202336, 0.7570674254168084, 2.350090829666805, 1.643683428661117, 1.643683428661117, 1.2814403012925042, 2.350090829666805, 0.6594644105825402, 2.350090829666805, 1.1275996661608945, 2.350090829666805]
    # spot_weights = [0.6524865085701085, 0.49091017135308285, 0.6524865085701085, 0.6524865085701085, 1.437544026096126, 0.5667836110123416, 0.9793849299099652, 0.3648990164916802, 0.6524865085701085, 0.8576180465157477, 0.8576180465157477, 0.7490567697595972, 0.6524865085701085, 1.1156683475734799, 0.8576180465157477, 0.6524865085701085, 2.325224129862271, 1.437544026096126, 0.6524865085701085, 0.49091017135308285, 0.3648990164916802, 0.7490567697595972, 0.49091017135308285, 0.49091017135308285, 0.8576180465157477, 0.5667836110123416, 0.49091017135308285, 0.423909037724256, 0.423909037724256, 0.3648990164916802, 1.8358775923721036, 0.2676798998341008, 0.6524865085701085, 1.437544026096126, 0.423909037724256, 0.6524865085701085, 0.7490567697595972, 0.7490567697595972, 2.0681837768959888, 1.6262913424156686, 0.3648990164916802, 1.1156683475734799, 0.8576180465157477, 0.5667836110123416, 0.9793849299099652, 0.5667836110123416, 0.8576180465157477, 1.2678811694975047, 0.8576180465157477, 0.8576180465157477, 1.437544026096126, 1.437544026096126, 2.325224129862271, 0.2676798998341008, 0.49091017135308285, 0.49091017135308285, 0.3648990164916802, 0.5667836110123416, 0.7490567697595972, 0.5667836110123416, 0.5667836110123416, 0.6524865085701085, 0.7490567697595972, 0.9793849299099652, 1.6262913424156686, 0.5667836110123416, 0.49091017135308285, 0.8576180465157477, 0.7490567697595972, 2.325224129862271, 0.8576180465157477, 0.49091017135308285, 0.49091017135308285, 0.6524865085701085, 1.437544026096126, 0.423909037724256, 0.9793849299099652, 0.5667836110123416, 2.0681837768959888, 0.7490567697595972, 0.5667836110123416, 2.0681837768959888, 0.9793849299099652, 0.7490567697595972, 0.6524865085701085, 0.7490567697595972, 1.2678811694975047, 0.8576180465157477, 0.9793849299099652, 2.325224129862271, 0.5667836110123416, 1.437544026096126, 0.7490567697595972, 0.5667836110123416, 0.9793849299099652, 0.8576180465157477, 1.8358775923721036, 0.6524865085701085, 2.0681837768959888, 0.8576180465157477, 0.423909037724256, 0.49091017135308285, 1.8358775923721036, 0.6524865085701085, 1.1156683475734799, 0.49091017135308285, 1.6262913424156686, 0.9793849299099652, 0.7490567697595972, 2.0681837768959888, 0.8576180465157477, 0.8576180465157477, 0.8576180465157477, 2.325224129862271, 2.325224129862271, 1.2678811694975047, 0.7490567697595972, 2.325224129862271, 0.5667836110123416, 0.49091017135308285, 0.7490567697595972, 1.437544026096126, 0.5667836110123416, 0.31307026943399247, 1.2678811694975047, 2.0681837768959888, 1.8358775923721036, 0.7490567697595972, 0.6524865085701085, 0.6524865085701085, 1.1156683475734799, 2.0681837768959888, 2.0681837768959888, 0.8576180465157477, 2.0681837768959888, 1.437544026096126, 0.9793849299099652, 1.1156683475734799, 1.1156683475734799, 1.437544026096126, 0.7490567697595972, 1.1156683475734799, 0.5667836110123416, 1.6262913424156686, 1.6262913424156686, 2.0681837768959888, 1.8358775923721036, 0.9793849299099652]
    spot_weights =[0.707614142342187, 0.45972449425461276, 0.532386455974887, 0.8123434840984062, 1.5590000248960638, 0.45972449425461276, 1.0621317346772978, 0.395728802365764, 0.707614142342187, 0.9300769448433988, 0.6146703319261233, 0.8123434840984062, 0.6146703319261233, 0.9300769448433988, 0.532386455974887, 0.6146703319261233, 2.521678926375928, 1.5590000248960638, 0.707614142342187, 0.532386455974887, 0.395728802365764, 0.6146703319261233, 0.532386455974887, 0.532386455974887, 0.45972449425461276, 0.6146703319261233, 0.532386455974887, 0.45972449425461276, 0.45972449425461276, 0.395728802365764, 1.990988213409228, 0.29029578428899816, 0.707614142342187, 0.45972449425461276, 0.1774512468460802, 0.707614142342187, 0.45972449425461276, 0.8123434840984062, 2.242921608756961, 1.7636943267744782, 0.395728802365764, 1.2099295395955405, 0.9300769448433988, 0.33952112003641327, 1.0621317346772978, 0.6146703319261233, 0.9300769448433988, 0.6146703319261233, 0.9300769448433988, 0.395728802365764, 1.5590000248960638, 1.7636943267744782, 2.521678926375928, 0.29029578428899816, 0.532386455974887, 0.532386455974887, 0.395728802365764, 0.532386455974887, 0.8123434840984062, 0.532386455974887, 0.6146703319261233, 0.707614142342187, 0.8123434840984062, 1.0621317346772978, 1.7636943267744782, 0.6146703319261233, 0.532386455974887, 0.9300769448433988, 0.8123434840984062, 2.521678926375928, 0.9300769448433988, 0.6146703319261233, 0.532386455974887, 0.707614142342187, 1.5590000248960638, 0.45972449425461276, 0.707614142342187, 0.6146703319261233, 2.242921608756961, 0.8123434840984062, 0.45972449425461276, 2.242921608756961, 0.33952112003641327, 0.532386455974887, 0.707614142342187, 0.8123434840984062, 1.375002600914907, 0.9300769448433988, 1.0621317346772978, 2.521678926375928, 0.6146703319261233, 1.5590000248960638, 0.8123434840984062, 0.6146703319261233, 1.0621317346772978, 0.45972449425461276, 1.990988213409228, 0.707614142342187, 2.242921608756961, 0.9300769448433988, 0.45972449425461276, 0.532386455974887, 1.990988213409228, 0.707614142342187, 1.2099295395955405, 1.2099295395955405, 1.375002600914907, 1.0621317346772978, 0.8123434840984062, 2.242921608756961, 0.9300769448433988, 0.45972449425461276, 0.9300769448433988, 2.521678926375928, 2.521678926375928, 1.375002600914907, 0.532386455974887, 2.521678926375928, 0.6146703319261233, 0.29029578428899816, 0.8123434840984062, 0.33952112003641327, 0.2473151432509325, 0.33952112003641327, 0.532386455974887, 1.990988213409228, 1.990988213409228, 0.8123434840984062, 0.707614142342187, 0.29029578428899816, 1.2099295395955405, 1.2099295395955405, 2.242921608756961, 0.9300769448433988, 2.242921608756961, 1.5590000248960638, 1.0621317346772978, 1.2099295395955405, 1.2099295395955405, 0.395728802365764, 0.707614142342187, 1.2099295395955405, 0.29029578428899816, 1.7636943267744782, 1.7636943267744782, 2.242921608756961, 1.7636943267744782, 2.521678926375928]
    # fmt: on
    # fmt: off
    readout_fidelity_list= [0.7910346974771012, 0.8537974445135668, 0.8637193306771253, 0.8374189913074277, 0.9181469574809848, 0.8694464667863699, 0.9384634351876944, 0.8680040448119246, 0.9226210069660886, 0.8988600379278242, 0.9088592816445846, 0.8749668091866413, 0.8973419522408208, 0.8589089679169892, 0.9436857422636902, 0.8624387941974844, 0.8082369291895957, 0.9567664157510902, 0.7905487843557945, 0.8562998860339586, 0.9346587481279328, 0.8942895666642299, 0.927825165018398, 0.776291785472258, 0.9433985702284242, 0.9118307479500294, 0.9208012891357047, 0.9463624257742007, 0.8774087640962681, 0.8738010321854603, 0.9494999812909778, 0.8475754410271904, 0.7949387569458899, 0.8397872260773789, 0.832005814624452, 0.8585698896591569, 0.8698567692780338, 0.8666621787867446, 0.8931440196264964, 0.8549703763113317, 0.8160804162871438, 0.8103793971124559, 0.8279981645576945, 0.94319632658461, 0.884566417253063, 0.8864985256701118, 0.9466785458582161, 0.8681320357914009, 0.9261526903359162, 0.8910726583003923, 0.9024692936513214, 0.9345066800877723, 0.8832701598153546, 0.863264724782739, 0.822744839873732, 0.9235905649517818, 0.7260475302030744, 0.9169310852973075, 0.9124671401536977, 0.950933038336218, 0.9404748534048617, 0.7835340741890058, 0.8415095852128811, 0.8534316938278272, 0.9374260928698699, 0.9105569949526905, 0.8885646817555796, 0.951594968772978, 0.9210775810637293, 0.4999999938946924, 0.9254206699641094, 0.8196441943719386, 0.7174333983658618, 0.8673020961817439, 0.9354136509442872, 0.8930067946120323, 0.8076102613366749, 0.895308319770373, 0.5004148258482037, 0.8598647390017464, 0.8990293462019805, 0.9081463386604317, 0.8573850339298418, 0.9219215655455681, 0.948821684940695, 0.8963014617062968, 0.9108065189029758, 0.9269251753248866, 0.8711315142848008, 0.8624743770973502, 0.8728679337692495, 0.8783379550497221, 0.951329183078171, 0.9064300859002559, 0.9110168607174641, 0.9160033311947926, 0.9327065166785726, 0.825915460881494, 0.9172120280799974, 0.9022801923278877, 0.8822020048966583, 0.8604564163268977, 0.8061321879312529, 0.8465864914666642, 0.8668903485287174, 0.7354624069332731, 0.9051225222841786, 0.9264906178228052, 0.9274468110985926, 0.928692967913262, 0.8977208916080855, 0.8786019565812202, 0.9054702600389162, 0.8119130983782279, 0.9057920755312001, 0.8289234937584826, 0.9074245544254917, 0.8024855557329635, 0.8847867643424019, 0.8781217145523129, 0.8475887403468434, 0.907937126476579, 0.8626352808947708, 0.7963506330315075, 0.8666950633651074, 0.8353894864271256, 0.8106560802514793, 0.9262755020972224, 0.8464502302903539, 0.9288089912268143, 0.859292887201319, 0.8837074382700327, 0.8770799851963816, 0.8932263724648392, 0.7710732787745815, 0.8589687712531706, 0.8925545885939717, 0.8852854659550835, 0.8686785949915612, 0.9036136449691496, 0.8375615166275213, 0.8140214426318657, 0.8099906536531916, 0.8373851687504905, 0.7576182288402435, 0.9244816424573312, 0.9264072385745544, 0.8451131581389844]
    prep_fidelity_list = [0.557810540058108, 0.5813583243317477, 0.5586430845080415, 0.5170293189876699, 0.5125066789568905, 0.607724007548803, 0.5082779486080762, 0.6751090890507605, 0.5099559685794226, 0.4307532745643661, 0.5514982008946254, 0.5037119165066559, 0.6010729526993711, 0.49426520455442513, 0.583996972314135, 0.5899996347096139, 0.5258434245193802, 0.5130950257288316, 0.5806785453182726, 0.6246275183223184, 0.43144251461472716, 0.6169156145616901, 0.42532291735135874, 0.7122870827062884, 0.5959271495550733, 0.543022780406793, 0.5678353725029532, 0.5044311961719382, 0.5520317432234334, 0.5850613753875162, 0.5628255367148665, 0.6954447789152969, 0.4544419884683777, 0.5641229554462979, 0.6880155634495017, 0.6203186950363319, 0.6729507309658898, 0.4847455907157048, 0.6119442209412707, 0.5086049289477561, 0.5895234870256467, 0.651732580260199, 0.6553782418616528, 0.5097560633206857, 0.5661543890289558, 0.5717189596722825, 0.534493820311468, 0.5844638256091127, 0.5959208247329886, 0.5689256398579032, 0.6229253757336897, 0.5753468613055325, 0.5832860254516378, 0.6318012944173708, 0.6470792318598046, 0.5263300352019602, 0.5638446362556075, 0.643909148866486, 0.5765064497765584, 0.41575534145640347, 0.44379756082864774, 0.6649938148089384, 0.6088483521015422, 0.6663727886382607, 0.5192379105774856, 0.5654436234533892, 0.5408950552582996, 0.48307670906998246, 0.45211230418452086, 0.8178040129410739, 0.5589454792042946, 0.5483016320524501, 0.5872504013100013, 0.6557001256164992, 0.5304212883607317, 0.6535426028327838, 0.6202195282362755, 0.6216788330730079, 0.8878456714988634, 0.599728390471843, 0.5857292146992049, 0.5448764879666532, 0.6608098944270349, 0.5680975215730284, 0.5190975686779405, 0.5785198794423521, 0.5473877194950522, 0.5513776236808775, 0.5782168025824734, 0.5315113133495968, 0.4502745990559637, 0.6201154969943081, 0.49312631466058954, 0.6299244018729604, 0.5858883959478696, 0.5892557238188174, 0.5567280441163207, 0.2529337846364633, 0.5868245334685978, 0.6114316867717693, 0.5854521954453655, 0.5546665600179603, 0.7007587656444079, 0.5910874646215306, 0.57555738765054, 0.6616399581999924, 0.46782740296198577, 0.5733196926653079, 0.5796152659042791, 0.5344355747707654, 0.5541904465724738, 0.6113551372024819, 0.5508213839707357, 0.6785358602422213, 0.5128854400705184, 0.5488776699206077, 0.5523948098142695, 0.32875549123501646, 0.46690552016329023, 0.5631255684126881, 0.40699439424808304, 0.647020467595286, 0.4730317324300408, 0.5737230961850276, 0.6355940226918639, 0.5984140305711674, 0.6107608221652161, 0.5519414434237095, 0.6782099838543398, 0.583199727670036, 0.6331672146774081, 0.5298436716098136, 0.39316609637462896, 0.5856898391124996, 0.38595249732663583, 0.34204486101448917, 0.6612313390915756, 0.4351799026244093, 0.5946619203904253, 0.5183201788246048, 0.5851023255483998, 0.6115945044636315, 0.6257037803945935, 0.45782199864224293, 0.6084334406175178, 0.5848473601341064, 0.5846631559105275, 0.6150387514370544]

    indices_to_drop = [ind for ind, (val1, val2) in enumerate(zip(readout_fidelity_list, prep_fidelity_list)) if val1<0.79 or val2<0.53]
    print(len(indices_to_drop))
    # sys.exit()
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

    aom_voltage = 0.4252  #
    power_law_params = [3.7e5, 6.97, 8e-14]  # Example power-law fit parameters
    a, b, c = [3.7e5, 6.97, 8e-14]
    total_power = a * (aom_voltage) ** b + c
    print(total_power)
    nv_powers = np.array(spot_weights) / sum(np.array(spot_weights))
    nv_powers = nv_powers * total_power  # Apply linear weights to all counts
    # calcualted_spot_weights = linear_weights(filtered_reordered_counts, alpha=0.3)
    # updated_spot_weights = linear_weights(filtered_reordered_counts, alpha=0.6)
    # Create a copy or initialize spot weights for modification
    updated_spot_weights = curve_extreme_weights_simple(
        spot_weights, scaling_factor=1.0
    )
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
    # adjusted_nv_powers = adjusted_nv_powers * total_power
    filtered_total_power = np.sum(updated_spot_weights)
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
    #     filename="slmsuite/nv_blob_detection/nv_blob_shallow_148nvs_reordered_updated.npz",
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
