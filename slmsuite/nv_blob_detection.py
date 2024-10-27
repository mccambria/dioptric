import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from skimage.draw import disk
from skimage.feature import blob_log
from skimage.filters import gaussian

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


# Fit a 2D Gaussian to a local region of the image data and return FWHM
def fit_gaussian_2d_local(image, center, size=10):
    x0, y0 = center
    x_min, x_max = int(x0 - size), int(x0 + size)
    y_min, y_max = int(y0 - size), int(y0 + size)

    x_min = max(x_min, 0)
    x_max = min(x_max, image.shape[1])
    y_min = max(y_min, 0)
    y_max = min(y_max, image.shape[0])

    local_image = image[y_min:y_max, x_min:x_max]

    x = np.arange(x_min, x_max)
    y = np.arange(y_min, y_max)
    x, y = np.meshgrid(x, y)

    initial_guess = (
        local_image.max(),
        x0,
        y0,
        1,
        1,
        0,
        np.min(local_image),
    )

    try:
        popt, _ = curve_fit(gaussian_2d, (x, y), local_image.ravel(), p0=initial_guess)
        amplitude, xo, yo, sigma_x, sigma_y, theta, offset = popt

        # Calculate FWHM in pixels for both x and y directions
        fwhm_x = 2.355 * sigma_x  # FWHM = 2.355 * sigma
        fwhm_y = 2.355 * sigma_y

        return (round(xo, 3), round(yo, 3)), (fwhm_x, fwhm_y), popt
    except RuntimeError:
        return center, (None, None), None


# Apply the blob detection algorithm and estimate spot size in pixels
def detect_nv_coordinates_blob(
    img_array,
    sigma=3.0,
    lower_threshold=30.0,
    upper_threshold=None,
    smoothing_sigma=1,
    integration_radius=3,
):
    smoothed_img = gaussian(img_array, sigma=smoothing_sigma)

    blobs = blob_log(
        smoothed_img,
        min_sigma=sigma,
        max_sigma=sigma,
        num_sigma=1,
        threshold=lower_threshold,
    )

    blobs[:, 2] = blobs[:, 2] * np.sqrt(2)

    valid_blobs = []
    optimized_coords = []
    spot_sizes = []  # List to store FWHM sizes for each spot
    integrated_counts = []
    for blob in blobs:
        y, x, r = blob
        rr, cc = disk((y, x), integration_radius, shape=img_array.shape)
        integrated_intensity = np.sum(smoothed_img[rr, cc])

        if integrated_intensity >= lower_threshold and (
            upper_threshold is None or integrated_intensity <= upper_threshold
        ):
            valid_blobs.append(blob)

            # Perform Gaussian fitting and get the FWHM
            optimized_coord, fwhm, _ = fit_gaussian_2d_local(img_array, (x, y), size=2)
            optimized_coords.append(optimized_coord)
            spot_sizes.append(fwhm)  # Append the FWHM for the spot
            integrated_counts.append(integrated_intensity)

    valid_blobs = np.array(valid_blobs)
    optimized_coords = np.array(optimized_coords)

    fig, ax = plt.subplots()
    title = "24ms, Ref"
    cax = kpl.imshow(ax, img_array, title=title, cbar_label="Photons")
    # cax = ax.imshow(img_array, cmap="hot")
    ax.set_title("NV Detection with Blob and Gaussian Fitting")
    ax.axis("off")

    # fig.colorbar(cax, ax=ax, orientation="vertical", label="Intensity")

    for idx, blob in enumerate(valid_blobs, start=1):
        y, x, r = blob
        circ = plt.Circle((x, y), r, color="red", linewidth=1, fill=False)
        ax.add_patch(circ)

        ax.text(
            x, y - r - 2, f"{idx}", color="black", fontsize=8, ha="center", va="center"
        )

    kpl.show(block=True)

    return optimized_coords, integrated_counts, spot_sizes


# Save the results to a file
def save_results(
    nv_coordinates,
    integrated_counts,
    path="slmsuite/nv_blob_detection",
    filename="nv_detection_results_with_gaussian_fit.npz",
):
    if not os.path.exists(path):
        os.makedirs(path)

    full_filepath = os.path.join(path, filename)

    np.savez(
        full_filepath,
        nv_coordinates=nv_coordinates,
        integrated_counts=integrated_counts,
    )


# Calculate the diffraction-limited resolution
def calculate_resolution(wavelength, NA):
    resolution = (0.61 * wavelength) / NA
    return resolution  # in micrometers


# Estimate pixel-to-µm conversion factor using FWHM
def pixel_to_um_conversion_factor(avg_fwhm, resolution):
    # Use the average FWHM to estimate the conversion factor
    conversion_factor = resolution / avg_fwhm  # µm per pixel
    return conversion_factor


# Function to calculate the Euclidean distance between two coordinates
def euclidean_distance(coord1, coord2):
    return np.linalg.norm(np.array(coord1) - np.array(coord2))


# Function to remove duplicates based on the Euclidean distance threshold
def remove_duplicates(coords, threshold=3):
    unique_coords = []
    for coord in coords:
        if not any(
            euclidean_distance(coord, unique_coord) < threshold
            for unique_coord in unique_coords
        ):
            unique_coords.append(coord)
    return unique_coords


# Process multiple images and remove duplicate NV coordinates
def process_multiple_images(
    image_ids, sigma=2.0, lower_threshold=30.0, upper_threshold=None, smoothing_sigma=1
):
    all_nv_coordinates = []
    all_spot_sizes = []

    for image_id in image_ids:
        print(f"Processing image ID: {image_id}")
        data = dm.get_raw_data(file_id=image_id, load_npz=True)
        img_array = np.array(data["img_array"])  # Load image data

        nv_coordinates, spot_sizes = detect_nv_coordinates_blob(
            img_array,
            sigma=sigma,
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold,
            smoothing_sigma=smoothing_sigma,
        )

        # Append new coordinates and spot sizes
        all_nv_coordinates.extend(nv_coordinates)
        all_spot_sizes.extend(spot_sizes)

    # Remove duplicates based on Euclidean distance
    unique_nv_coordinates = remove_duplicates(all_nv_coordinates, threshold=3)
    print(
        f"Total unique NV coordinates after removing duplicates: {len(unique_nv_coordinates)}"
    )

    return unique_nv_coordinates, all_spot_sizes


# Main section of the code
if __name__ == "__main__":
    kpl.init_kplotlib()
    # Load the image data
    # data = dm.get_raw_data(file_id=1646374739142, load_npz=True)
    # img_array = np.array(data["img_array"])

    # data = dm.get_raw_data(file_id=1680868340409, load_npz=True)
    # data = dm.get_raw_data(file_id=1680026835865, load_npz=True)
    # data = dm.get_raw_data(file_id=1681681506476, load_npz=True)
    data = dm.get_raw_data(file_id=1682946042852, load_npz=True)

    img_array = np.array(data["ref_img_array"])
    # img_array = np.array(data["diff_img_array"])
    # img_array = -img_array
    print(type(img_array), img_array.dtype, img_array.shape)

    # # Parameters for detection and resolution
    wavelength = 0.65  # Wavelength in micrometers (650 nm)
    NA = 1.45  # Numerical Aperture of objective

    # # Calculate the resolution (in micrometers)
    resolution = calculate_resolution(wavelength, NA)
    # print(f"Resolution: {round(resolution,3)} µm")

    # Apply the blob detection and Gaussian fitting
    sigma = 2
    lower_threshold = 0.17
    upper_threshold = None
    smoothing_sigma = 0.0

    nv_coordinates, integrated_counts, spot_sizes = detect_nv_coordinates_blob(
        img_array,
        sigma=sigma,
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold,
        smoothing_sigma=smoothing_sigma,
    )

    # List to store valid NV coordinates after filtering
    filtered_nv_coords = []

    # Iterate through detected NV coordinates and apply distance filtering
    for coord in nv_coordinates:
        # Assume the coordinate is valid initially
        keep_coord = True

        # Check distance with all previously accepted NVs
        for existing_coord in filtered_nv_coords:
            distance = np.linalg.norm(np.array(existing_coord) - np.array(coord))

            if distance < 5:
                keep_coord = False  # Mark it for exclusion if too close
                break  # No need to check further distances

        # If the coordinate passes the distance check, add it to the list
        if keep_coord:
            filtered_nv_coords.append(coord)
            spot_weigths = integrated_counts
    print(f"Number of NVs detected: {len(filtered_nv_coords)}")

    # print(f"Detected NV coordinates (optimized): {filtered_nv_coords}")

    # Calculate and print the average FWHM
    # if len(spot_sizes) > 0:
    #     avg_fwhm = np.mean([(fwhm_x + fwhm_y) / 2 for fwhm_x, fwhm_y in spot_sizes])
    #     print(f"Average FWHM (in pixels): {round(avg_fwhm,3)}")

    #     # Estimate and print the pixel-to-µm conversion factor
    #     conversion_factor = pixel_to_um_conversion_factor(avg_fwhm, resolution)
    #     print(
    #         f"Pixel-to-µm conversion factor: {round(conversion_factor,3)} µm per pixel"
    #     )
    # else:
    #     print("No spots detected. Unable to calculate conversion factor.")

    # Save the results
    save_results(
        nv_coordinates,
        integrated_counts,
        filename="nv_blob_filtered_120nvs.npz",
    )

    # image_ids = [
    #     1679871908428,
    #     1679874278209,
    #     1679872688533,
    #     1679871364006,
    #     1679879341220,
    #     1679875544782,
    #     1679879096913,
    #     1679871587183,
    #     1679879168640,
    #     1679874767869,
    # ]  # Add more image IDs as needed
    # # Process multiple images and remove duplicates
    # unique_nv_coordinates, spot_sizes = process_multiple_images(
    #     image_ids,
    #     sigma=sigma,
    #     lower_threshold=lower_threshold,
    #     upper_threshold=upper_threshold,
    #     smoothing_sigma=smoothing_sigma,
    # )
    # print(f"Number of NVs detected: {len(unique_nv_coordinates)}")
    # # print(f"Detected NV coordinates (optimized): {nv_coordinates}")
    # # Save the results
    # save_results(
    #     unique_nv_coordinates,
    #     spot_sizes,
    #     filename="nv_blob_filtered_multiple_1.npz",
    # )
