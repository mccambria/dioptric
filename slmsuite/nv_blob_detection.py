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
    sigma=2.0,
    lower_threshold=15.0,
    upper_threshold=None,
    smoothing_sigma=1,
    integration_radius=2,
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
            optimized_coord, fwhm, _ = fit_gaussian_2d_local(img_array, (x, y), size=3)
            optimized_coords.append(optimized_coord)
            spot_sizes.append(fwhm)  # Append the FWHM for the spot
            integrated_counts.append(integrated_intensity)

    valid_blobs = np.array(valid_blobs)
    optimized_coords = np.array(optimized_coords)

    fig, ax = plt.subplots()
    title = "24ms, Ref"
    cax = kpl.imshow(ax, img_array, title=title, cbar_label="Photons")
    # cax = ax.imshow(img_array, cmap="hot")
    ax.set_title("NV Detection with Blob")
    ax.axis("off")

    # fig.colorbar(cax, ax=ax, orientation="vertical", label="Intensity")

    for idx, blob in enumerate(valid_blobs, start=1):
        y, x, r = blob
        circ = plt.Circle((x, y), r, color="red", linewidth=1, fill=False)
        ax.add_patch(circ)

        ax.text(
            x, y - r - 2, f"{idx}", color="black", fontsize=8, ha="center", va="center"
        )

    # kpl.show(block=True)

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

        nv_coordinates, spot_sizes, *_ = detect_nv_coordinates_blob(
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


# # Remove duplicate NV coordinates
# def remove_duplicates(coords, threshold=3):
#     unique_coords = []
#     for coord in coords:
#         if not any(
#             np.linalg.norm(np.array(coord) - np.array(unique)) < threshold
#             for unique in unique_coords
#         ):
#             unique_coords.append(coord)


# # Apply blob detection & estimate NV spot sizes
# def detect_nv_coordinates(
#     img_array, sigma=3.0, lower_threshold=30.0, integration_radius=3
# ):
#     smoothed_img = gaussian(img_array, sigma=1)
#     blobs = blob_log(
#         smoothed_img,
#         min_sigma=sigma,
#         max_sigma=sigma,
#         num_sigma=1,
#         threshold=lower_threshold,
#     )
#     blobs[:, 2] *= np.sqrt(2)

#     valid_blobs, optimized_coords, spot_sizes = [], [], []
#     for y, x, r in blobs:
#         rr, cc = disk((y, x), integration_radius, shape=img_array.shape)
#         integrated_intensity = np.sum(smoothed_img[rr, cc])

#         if integrated_intensity >= lower_threshold:
#             valid_blobs.append((x, y))
#             optimized_coord, fwhm = fit_gaussian_2d_local(img_array, (x, y), size=2)
#             optimized_coords.append(optimized_coord)
#             spot_sizes.append(fwhm)

#     return np.array(optimized_coords), np.array(spot_sizes)


import matplotlib.pyplot as plt
import numpy as np


def process_scan_file():
    """Processes a saved scan file, extracts NV coordinates from each scan entry,
    and creates a combined image using max projection.
    """
    # raw_data = dm.get_raw_data(file_id=1763690008369, load_npz=True)
    # raw_data = dm.get_raw_data(file_id=1766305651080, load_npz=True)
    # raw_data = dm.get_raw_data(file_id=1764604364429, load_npz=True)
    # raw_data = dm.get_raw_data(file_id=1788823078836, load_npz=True)  # rubin
    raw_data = dm.get_raw_data(file_id=1802697426409, load_npz=True)  # rubin

    # Extract scanned data
    scanned_data = raw_data["scanned_data"]["scanned_data"]
    print(f"Type of scanned_data: {type(scanned_data)}")  # Expected: <class 'list'>

    # Preallocate a list for image processing
    blob_coords, spot_weights, img_arrays = [], [], []

    for index, scan in enumerate(scanned_data):
        img_array = np.array(scan["scan_data"], dtype=np.float64)

        # Detect NVs
        optimized_coords, integrated_counts, _ = detect_nv_coordinates_blob(img_array)

        # Only store detected NVs if not empty
        if optimized_coords.size > 0:
            blob_coords.extend(optimized_coords)  # Ensure list format
            spot_weights.extend(integrated_counts)

        # Normalize image
        img_array = (img_array - 300) / max(1, np.median(img_array))
        # img_array = widefie
        # Store processed image
        img_arrays.append(img_array)

    # Convert to NumPy array if images exist
    if img_arrays:
        img_arrays = np.array(img_arrays)
        combined_img = np.max(img_arrays, axis=0)  # Maximum intensity projection

        # Plot final image
        fig, ax = plt.subplots()
        kpl.imshow(
            ax, combined_img, title="Combined_img_laser_INTI_520", cbar_label="Photons"
        )
        ax.axis("off")
        print(f"Final detected NV count: {len(blob_coords)}")
    else:
        print("No valid images found.")

    # **Save the results (uncomment if needed)**
    save_results(
        blob_coords,
        spot_weights,
        path="slmsuite/nv_blob_detection",
        filename=f"nv_blob_shallow_rubin_{len(blob_coords)}nvs.npz",
    )

    timestamp = dm.get_time_stamp()
    data = {
        "timestamp": timestamp,
        "img_array": combined_img,
    }

    file_path = dm.get_file_path(__file__, timestamp, "combined_image_array")
    # dm.save_raw_data(data, file_path, keys_to_compress=["img_array"])
    kpl.show(block=True)


# Save NV results
def save_results(nv_coordinates, spot_weights, path, filename):
    if not os.path.exists(path):
        os.makedirs(path)

    np.savez(
        os.path.join(path, filename),
        nv_coordinates=nv_coordinates,
        updated_spot_weights=spot_weights,
    )


# Plot NV detection results
def plot_nv_detection(img_array, nv_coords):
    fig, ax = plt.subplots()
    kpl.imshow(ax, img_array, title="NV Detection", cbar_label="Photons")

    for x, y in nv_coords:
        circ = plt.Circle((x, y), 2.4, color="red", linewidth=1, fill=False)
        ax.add_patch(circ)
        ax.text(x, y - 3, f"{x:.1f}, {y:.1f}", color="white", fontsize=8, ha="center")

    kpl.show(block=True)


# Main section of the code
if __name__ == "__main__":
    kpl.init_kplotlib()
    # Load the image data
    # data = dm.get_raw_data(file_id=1646374739142, load_npz=True)
    # img_array = np.array(data["img_array"])

    # data = dm.get_raw_data(file_id=1693686359757, load_npz=True)
    # data = dm.get_raw_data(file_id=1694279622270, load_npz=True)
    # data = dm.get_raw_data(file_id=1697710844436, load_npz=True)
    # data = dm.get_raw_data(file_id=1698496302146, load_npz=True)
    # data = dm.get_raw_data(file_id=1699573772441, load_npz=True)
    # data = dm.get_raw_data(file_id=1764727515943, load_npz=True)  # comind datas
    # data = dm.get_raw_data(file_id=1766351495521, load_npz=True)
    # data = dm.get_raw_data(file_id=1766620461342, load_npz=True)
    # data = dm.get_raw_data(file_id=1791296034768, load_npz=True)
    # data = dm.get_raw_data(file_id=1791776254933, load_npz=True)  # rubin green scan
    data = dm.get_raw_data(file_id=1807103519645, load_npz=True)
    img_array = np.array(data["ref_img_array"])
    # img_array = np.array(data["ref_img_array"]["ref_img_array"])
    # img_array = np.array(data["img_array"])
    # print(img_array)
    # img_array = np.array(data["diff_img_array"])
    # img_array = -img_array
    # print(type(img_array), img_array.dtype, img_array.shape)

    # # Parameters for detection and resolution
    wavelength = 0.7  # Wavelength in micrometers (650 nm)
    NA = 1.45  # Numerical Aperture of objective

    # Calculate the resolution (in micrometers)
    resolution = calculate_resolution(wavelength, NA)
    print(f"Resolution: {round(resolution, 3)} µm")

    # Apply the blob detection and Gaussian fitting
    sigma = 2.0
    lower_threshold = 0.09
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
    filtered_counts = []
    # Iterate through detected NV coordinates and apply distance filtering
    for coord, count in zip(nv_coordinates, integrated_counts):
        # Assume the coordinate is valid initially
        keep_coord = True

        # Check distance with all previously accepted NVs
        for existing_coord in filtered_nv_coords:
            distance = np.linalg.norm(np.array(existing_coord) - np.array(coord))

            if distance < 8:
                keep_coord = False  # Mark it for exclusion if too close
                break  # No need to check further distances

        # If the coordinate passes the distance check, add it to the list
        if keep_coord:
            filtered_nv_coords.append(coord)
            filtered_counts.append(count)

    print(f"Number of NVs detected: {len(filtered_nv_coords)}")
    for idx, (coord, count) in enumerate(
        zip(filtered_nv_coords, filtered_counts), start=1
    ):
        print(f"NV {idx}: {coord}, {count}:.2f")
    # Plotting the results
    # Verify if reversing coordinates resolves the offset
    default_radius = 2.4
    fig, ax = plt.subplots()
    title = "24ms, Ref"
    cax = kpl.imshow(ax, img_array, title=title, cbar_label="Photons")
    ax.set_title("NV Detection with Blob")
    ax.axis("off")

    for idx, (x, y) in enumerate(filtered_nv_coords, start=1):  # Swapped y, x to x, y
        circ = plt.Circle(
            (x, y), default_radius, color="white", linewidth=1, fill=False
        )
        ax.add_patch(circ)
        ax.text(
            x,
            y - default_radius - 2,
            f"{idx}",
            # color="black",
            fontsize=8,
            ha="center",
            va="center",
        )

    kpl.show(block=True)

    print(f"Detected NV coordinates (optimized): {len(filtered_nv_coords)}")

    # Save the results
    # save_results(
    #     filtered_nv_coords,
    #     filtered_counts,
    #     path="slmsuite/nv_blob_detection",
    #     filename="nv_blob_shallow_rubin_81nvs.npz",
    # )

    # full ROI -- multiple images save in the same file
    # process_scan_file()

    # Calculate and print the average FWHM
    # if len(spot_sizes) > 0:
    #     avg_fwhm = np.mean([(fwhm_x + fwhm_y) / 2 for fwhm_x, fwhm_y in spot_sizes])
    #     print(f"Average FWHM (in pixels): {round(avg_fwhm, 3)}")

    #     # Estimate and print the pixel-to-µm conversion factor
    #     conversion_factor = pixel_to_um_conversion_factor(avg_fwhm, resolution)
    #     print(
    #         f"Pixel-to-µm conversion factor: {round(conversion_factor, 3)} µm per pixel"
    #     )
    # else:
    #     print("No spots detected. Unable to calculate conversion factor.")

    # fig, ax = plt.subplots()
    # title = "24ms, Ref"
    # cax = kpl.imshow(ax, img_array, title=title, cbar_label="Photons")
    # ax.set_title("Combined_img_INTE_laser_520")
    # ax.axis("off")
    # kpl.show(block=True)
    # image_ids = [
    #     1693359304252,
    #     1693375930943,
    #     1693374155876,
    #     1693372069315,
    #     1693363033340,
    #     1693376854962,
    #     1693361128042,
    #     1693376833728,
    #     1693379996949,
    # ]  # Add more image IDs as needed

    # # Process multiple images and remove duplicates
    # unique_nv_coordinates, spot_sizes = process_multiple_images(
    #     image_ids,
    #     sigma=sigma,
    #     lower_threshold=lower_threshold,
    #     upper_threshold=upper_threshold,
    #     smoothing_sigma=smoothing_sigma,
    # )
    # # Filter out-of-bounds coordinates and their corresponding spot sizes
    # filtered_nv_coordinates = []
    # filtered_spot_sizes = []

    # for coord, size in zip(unique_nv_coordinates, spot_sizes):
    #     x, y = int(coord[0]), int(coord[1])
    #     if not (0 <= x < 250 and 0 <= y < 250):
    #         print(f"Skipping NV at ({coord[0]:.2f}, {coord[1]:.2f}) - Out of bounds")
    #         continue  # Skip this coordinate and its spot size if out of bounds

    #     filtered_nv_coordinates.append(coord)
    #     filtered_spot_sizes.append(size)

    # # Print the results
    # print(f"Number of NVs detected: {len(filtered_nv_coordinates)}")
    # Optionally print NV coordinates and spot sizes for debugging
    # print(f"Detected NV coordinates (optimized): {filtered_nv_coordinates}")
    # print(f"Corresponding spot sizes: {filtered_spot_sizes}")

    # # Save the results
    # save_results(
    #     filtered_nv_coordinates,
    #     filtered_spot_sizes,
    #     filename="nv_blob_filtered_265nvs.npz",
    # )
