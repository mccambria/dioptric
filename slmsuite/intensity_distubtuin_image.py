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
def fit_gaussian(image, coord, window_size=3):
    x0, y0 = coord
    img_shape_y, img_shape_x = image.shape  # Get image dimensions

    # Ensure the window doesn't go out of bounds
    x_min = max(int(x0 - window_size), 0)
    x_max = min(int(x0 + window_size + 1), img_shape_x)
    y_min = max(int(y0 - window_size), 0)
    y_max = min(int(y0 + window_size + 1), img_shape_y)

    # If the cutout region is too small or invalid, return original coordinates
    if (x_max - x_min) <= 1 or (y_max - y_min) <= 1:
        print(
            f"Invalid cutout for NV at ({x0}, {y0}): Region too small or out of bounds"
        )
        return x0, y0  # Return original coordinates

    # Generate the mesh grid for the fitting region
    x_range = np.arange(x_min, x_max)
    y_range = np.arange(y_min, y_max)
    x, y = np.meshgrid(x_range, y_range)

    # Extract the cutout region from the image
    image_cutout = image[y_min:y_max, x_min:x_max]

    # Check if the cutout is valid and non-zero in size
    if image_cutout.size == 0:
        print(f"Zero-size cutout for NV at ({x0}, {y0})")
        return x0, y0

    # Initial guess parameters for the Gaussian fit
    initial_guess = (np.max(image_cutout), x0, y0, 3, 3, 0, np.min(image_cutout))

    # Perform the 2D Gaussian fit
    try:
        popt, _ = curve_fit(gaussian_2d, (x, y), image_cutout.ravel(), p0=initial_guess)
        _, fitted_x, fitted_y, _, _, _, _ = popt  # Extract fitted X and Y coordinates
        return fitted_x, fitted_y  # Return optimized coordinates
    except Exception as e:
        print(f"Fit failed for NV at ({x0}, {y0}): {e}")
        return x0, y0  # Return original coordinates if fitting fails


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
    # lower_bound = Q1 - 1.5 * IQR
    # upper_bound = Q3 + 2.8 * IQR
    lower_bound = 3.3
    upper_bound = 100

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
    weights = 1 / (1 + np.exp(beta * (intensities - threshold)))
    return weights / np.max(weights)  # Normalize the weights


def non_linear_weights(intensities, alpha=1):
    weights = 1 / np.power(intensities, alpha)
    weights = weights / np.max(weights)  # Normalize to avoid extreme values
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
    file_path="slmsuite/nv_blob_detection/nv_blob_filtered_77nvs.npz",
    # file_path="slmsuite/nv_blob_detection/nv_coords_integras_counts_filtered.npz",
):
    data = np.load(file_path)
    nv_coordinates = data["nv_coordinates"]
    return nv_coordinates


# Main section of the code
if __name__ == "__main__":
    kpl.init_kplotlib()

    # Parameters
    remove_outliers_flag = False  # Set this flag to enable/disable outlier removal
    reorder_coords_flag = True  # Set this flag to enable/disable reordering of NVs

    # data = dm.get_raw_data(file_id=1648773947273, load_npz=True)
    data = dm.get_raw_data(file_id=1651663986412, load_npz=True)
    img_array = np.array(data["ref_img_array"])
    nv_coordinates = load_nv_coords().tolist()
    # Start merged coordinates with the reference NV
    # Reference NV to append as the first coordinate
    # reference_nv = [113.431, 149.95]
    reference_nv = [112.76, 150.887]
    # # # Start with the reference NV as the first element
    nv_coords = [reference_nv]

    # # Iterate through the rest of the NV coordinates
    for coord in nv_coordinates:
        # Calculate the distance between the current NV and the reference NV
        distance = np.linalg.norm(np.array(reference_nv) - np.array(coord))

        # Append NV if it's farther than 3 pixels away from the reference NV
        if distance >= 3:
            nv_coords.append(coord)

    # Fit 2D Gaussian to each NV and get optimized coordinates
    # optimal_nv_coords = []
    # for coord in nv_coords:
    #     optimized_x, optimized_y = fit_gaussian(img_array, coord)
    #     optimal_nv_coords.append((optimized_x, optimized_y))

    # # Round each coordinate to 3 decimal places
    # rounded_nv_coords = [(round(x, 3), round(y, 3)) for x, y in optimal_nv_coords]
    nv_coordinates = nv_coords
    # manual removal of the nvs
    # manual_removal_indices = [41]
    # nv_coordinates = remove_manual_indices(nv_coordinates, manual_removal_indices)

    # Remove NVs with SNR < 0.5
    # filtered_nv_indices = filter_by_snr(SNR, threshold=0.7)
    # nv_coordinates = [nv_coordinates[i] for i in filtered_nv_indices]

    # Integrate intensities for each NV coordinate
    sigma = 3
    integrated_intensities = integrate_intensity(img_array, nv_coordinates, sigma)
    integrated_intensities = np.array(integrated_intensities)

    if remove_outliers_flag:
        # Remove outliers and corresponding NV coordinates
        filtered_intensities, filtered_nv_coords = remove_outliers(
            integrated_intensities, nv_coordinates
        )
    else:
        filtered_intensities = integrated_intensities
        filtered_nv_coords = nv_coordinates

    # Reorder NV coordinates based on Euclidean distances from the first NV
    if reorder_coords_flag:
        filtered_nv_coords = reorder_coords(filtered_nv_coords)

    # Convert filtered intensities to a NumPy array for element-wise operations
    filtered_intensities = np.array(filtered_intensities)

    spot_weights = non_linear_weights(filtered_intensities, alpha=0.9)

    # Print some diagnostics
    print(f"NV Coords: {filtered_nv_coords}")
    print(f"Filtered integrated intensities: {filtered_intensities}")
    print(f"Filtered NV coordinates: {len(filtered_nv_coords)} NVs")
    print(f"Normalized spot weights: {spot_weights}")
    print(f"Number of NVs detected: {len(filtered_nv_coords)}")

    # Save the filtered results
    # save_results(
    #     filtered_nv_coords,
    #     filtered_intensities,
    #     spot_weights,
    #     filename="slmsuite/nv_blob_detection/nv_coords_integras_counts_filtered.npz",
    # )

    # Plot the original image with circles around each NV
    fig, ax = plt.subplots()
    title = "12 ms, Ref"
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
    sns.histplot(filtered_intensities, bins=45, kde=False, color="blue")

    plt.xlabel("Integrated Intensity")
    plt.ylabel("Frequency")
    plt.title("Histogram of Filtered Integrated Counts")
    plt.show(block=True)
