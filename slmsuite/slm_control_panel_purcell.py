# -*- coding: utf-8 -*-
"""
Control panel for the slm

Created on Spring, 2024

@author: saroj chand
"""

import io
import os
import sys
import warnings
from datetime import datetime

# os.environ["QT_QPA_PLATFORM"] = "offscreen"
import cv2
import imageio
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage

# Generate a phase .gif
from IPython.display import Image
from scipy.optimize import curve_fit

from slmsuite import example_library
from slmsuite.hardware.cameras.thorlabs import ThorCam
from slmsuite.hardware.cameraslms import FourierSLM
from slmsuite.hardware.slms.thorlabs import ThorSLM
from slmsuite.holography import analysis, toolbox
from slmsuite.holography.algorithms import FeedbackHologram, SpotHologram
from slmsuite.misc import fitfunctions
from utils import data_manager as dm
from utils import tool_belt as tb

warnings.filterwarnings("ignore")
mpl.rc("image", cmap="Blues")


def plot_phase(phase, angle):
    # Initialize the figure and axes outside the loop
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    blaze_vector = (np.cos(np.radians(angle)), np.sin(np.radians(angle)))

    # Update phase with live rotation
    delta_phase = toolbox.phase.blaze(grid=slm, vector=blaze_vector, offset=0)
    phase = None

    # Display the phase pattern on the SLM
    slm.write(phase, settle=True)

    # Capture image from the camera
    cam.set_exposure(0.0001)
    im = cam.get_image()

    # Clear the axes and plot the phase, delta phase, and camera image
    ax[0].clear()
    ax[0].imshow(phase, cmap="gray")
    ax[0].set_title("Total Phase")

    ax[1].clear()
    ax[1].imshow(delta_phase, cmap="gray")
    ax[1].set_title("Delta Phase")

    ax[2].clear()
    ax[2].imshow(im, cmap="gray")
    ax[2].set_title("Camera Image")

    plt.pause(0.01)


def cam_plot():
    cam.set_exposure(0.0001)
    img = cam.get_image()
    # Plot the result
    plt.figure(figsize=(6, 5))
    plt.imshow(img, cmap="gray")  # Adjust 'cmap' as needed for color maps
    plt.show()

    # # Save the image

    file_path = r"slmsuite\cam_image"
    num_nvs = len(nuvu_pixel_coords)
    now = datetime.now()
    date_time_str = now.strftime("%Y%m%d_%H%M%S")
    filename = f"slm_generated_spots_{num_nvs}nvs_{date_time_str}.npy"
    # Save the phase data
    save(img, file_path, filename)
    print(f"Image saved at {file_path}")


def blaze(vector_deg=(0.2, 0.2)):
    # Get .2 degrees in normalized units.
    vector = toolbox.convert_blaze_vector(vector_deg, from_units="deg", to_units="norm")
    blaze_phase = toolbox.phase.blaze(grid=slm, vector=vector)
    plot_phase(blaze_phase, title="Blaze at {} deg".format(vector_deg))


# region "calibration"
def fourier_calibration():
    cam.set_exposure(0.002)  # Increase exposure because power will be split many ways
    fs.fourier_calibrate(
        array_shape=[20, 12],  # Size of the calibration grid (Nx, Ny) [knm]
        array_pitch=[30, 40],  # Pitch of the calibration grid (x, y) [knm]
        plot=True,
    )
    # cam.set_exposure(0.01)
    # save calibation
    calibration_file = fs.save_fourier_calibration(path="slmsuite/fourier_calibration")
    print("Fourier calibration saved to:", calibration_file)


def test_wavefront_calibration():
    cam.set_exposure(0.001)
    movie = fs.wavefront_calibrate(
        interference_point=(600, 400),
        field_point=(0.25, 0),
        field_point_units="freq",
        superpixel_size=60,
        test_superpixel=(16, 16),  # Testing mode
        autoexposure=False,
        plot=3,  # Special mode to generate a phase .gif
    )
    imageio.mimsave("wavefront.gif", movie)
    Image(filename="wavefront.gif")


def wavefront_calibration():
    cam.set_exposure(0.001)
    fs.wavefront_calibrate(
        interference_point=(600, 400),
        field_point=(0.25, 0),
        field_point_units="freq",
        superpixel_size=40,
        autoexposure=False,
    )
    # save calibation
    calibration_file = fs.save_wavefront_calibration(
        path="slmsuite/wavefront_calibration"
    )
    print("Fourier calibration saved to:", calibration_file)


def load_fourier_calibration():
    calibration_file_path = (
        # "slmsuite/fourier_calibration/26438-SLM-fourier-calibration_00003.h5"
        "slmsuite/fourier_calibration/26438-SLM-fourier-calibration_00006.h5"
    )
    fs.load_fourier_calibration(calibration_file_path)
    print("Fourier calibration loaded from:", calibration_file_path)


def load_wavefront_calibration():
    calibration_file_path = (
        "slmsuite/wavefront_calibration/26438-SLM-wavefront-calibration_00004.h5"
    )
    fs.load_wavefront_calibration(calibration_file_path)
    print("Wavefront calibration loaded from:", calibration_file_path)


def evaluate_uniformity(vectors=None, size=25):
    # Set exposure and capture image
    cam.set_exposure(0.001)
    img = cam.get_image()
    # Extract subimages
    if vectors is None:
        subimages = analysis.take(img, vectors=None, size=size)
    else:
        subimages = analysis.take(img, vectors=vectors, size=size)

    # Plot subimages
    analysis.take_plot(subimages)
    # Normalize subimages and compute powers
    powers = analysis.image_normalization(subimages)
    # Plot histogram of powers
    plt.hist(powers / np.mean(powers))
    plt.show()


# Test pattern
def circles():
    cam.set_exposure(0.1)
    center = (750, 530)  # Center of the circle
    radii = np.linspace(50, 200, num=4)  # Adjust the number of circles as needed
    circle_points = []
    for radius in radii:
        num_points = int(2 * np.pi * radius / 60)

        # Generate points within the circle using polar coordinates
        theta = np.linspace(0, 2 * np.pi, num_points)  # Angle values
        x_circle = center[0] + radius * np.cos(theta)  # X coordinates
        y_circle = center[1] + radius * np.sin(theta)  # Y coordinates

        # Convert to grid format for the current circle
        circle = np.vstack((x_circle, y_circle))

        circle_points.append(circle)

    # Combine the points of all circles
    circles = np.concatenate(circle_points, axis=1)
    hologram = SpotHologram(
        shape=(2048, 2048), spot_vectors=circles, basis="ij", cameraslm=fs
    )

    # # Precondition computationally.
    hologram.optimize(
        "WGS-Kim",
        maxiter=20,
        feedback="computational_spot",
        stat_groups=["computational_spot"],
    )
    phase = hologram.extract_phase()
    # Define the path to save the phase data1
    file_path = r"slmsuite\circles"
    now = datetime.now()
    date_time_str = now.strftime("%Y%m%d_%H%M%S")
    filename = f"slm_phase_circles_{date_time_str}.npy"
    # file_path = dm.get_file_path(__file__, filename)
    # Save the phase data
    save(phase, file_path, filename)
    slm.write(phase, settle=True)

    # cam_plot()
    # evaluate_uniformity(vectors=circle)

    # Hone the result with experimental feedback.
    # hologram.optimize(
    #     "WGS-Kim",
    #     maxiter=20,
    #     feedback="experimental_spot",
    #     stat_groups=["computational_spot", "experimental_spot"],
    #     fixed_phase=False,
    # )
    # phase = hologram.extract_phase()
    # slm.write(phase, settle=True)
    # cam_plot()
    # evaluate_uniformity(vectors=circle)


# region "nv phase calulation"
def calibration_triangle():
    cam.set_exposure(0.1)

    # Define parameters for the equilateral triangle
    # center = (730, 570)  # Center of the triangle
    center = (680, 630)  # Center of the triangle
    side_length = 400  # Length of each side of the triangle\

    # Calculate the coordinates of the three vertices of the equilateral triangle
    theta = np.linspace(0, 2 * np.pi, 4)[:-1]  # Exclude the last point to avoid overlap
    x_triangle = center[0] + side_length * np.cos(theta + np.pi / 6)  # X coordinates
    y_triangle = center[1] + side_length * np.sin(theta + np.pi / 6)  # Y coordinates

    # Combine the coordinates into a grid format
    triangle_points = np.vstack((x_triangle, y_triangle))
    print("thorcam coords:", triangle_points)
    hologram = SpotHologram(
        shape=(2048, 2048), spot_vectors=triangle_points, basis="ij", cameraslm=fs
    )

    # Precondition computationally
    hologram.optimize(
        "WGS-Kim",
        maxiter=20,
        feedback="computational_spot",
        stat_groups=["computational_spot"],
    )

    phase = hologram.extract_phase()
    slm.write(phase, settle=True)
    # file_path = r"slmsuite\calibration"
    # num_nvs = len(nuvu_pixel_coords)
    # now = datetime.now()
    # date_time_str = now.strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
    # filename = f"slm_calibration_{num_nvs}nvs_{date_time_str}.npy"
    # Save the phase data
    # save(phase, file_path, filename)
    # cam_plot()


def nuvu2thorcam_calibration(coords):
    """
    Calibrates and transforms coordinates from the Nuvu camera's coordinate system
    to the Thorlabs camera's coordinate system using an affine transformation.
    """
    cal_coords_thorcam = np.array(
        [[1026.410, 830.0], [333.589, 830.0], [680.0, 230.0]], dtype="float32"
    )

    cal_coords_nuvu = np.array(
        [[229.609, 11.449], [214.144, 240.823], [22.711, 111.838]], dtype="float32"
    )

    # Compute the affine transformation matrix
    M = cv2.getAffineTransform(cal_coords_nuvu, cal_coords_thorcam)
    # Append a column of ones to the input coordinates to facilitate affine transformation
    ones_column = np.ones((coords.shape[0], 1))
    coords_homogeneous = np.hstack((coords, ones_column))
    thorcam_coords = np.dot(coords_homogeneous, M.T)

    return thorcam_coords


def load_nv_coords(
    # file_path="slmsuite/nv_blob_detection/nv_blob_filtered_437nvs_reordered.npz",
    # file_path="slmsuite/nv_blob_detection/nv_blob_shallow_161nvs_reordered.npz",
    # file_path="slmsuite/nv_blob_detection/nv_blob_shallow_148nvs_reordered.npz",
    # file_path="slmsuite/nv_blob_detection/nv_blob_rubin_shallow_240nvs_reordered.npz",
    # file_path="slmsuite/nv_blob_detection/nv_blob_rubin_shallow_154nvs_reordered.npz",
    # file_path="slmsuite/nv_blob_detection/nv_blob_rubin_shallow_81nvs_reordered.npz",
    file_path="slmsuite/nv_blob_detection/nv_blob_rubin_shallow_75nvs_reordered.npz",
    # file_path="slmsuite/nv_blob_detection/nv_blob_rubin_shallow_40nvs_reordered.npz",
    # file_path="slmsuite/nv_blob_detection/nv_blob_rubin_shallow_35nvs_reordered.npz",
    # file_path="slmsuite/nv_blob_detection/nv_blob_shallow_148nvs_reordered_updated.npz",
    # file_path="slmsuite/nv_blob_detection/nv_blob_rubin_shallow_107nvs_reordered_updated.npz",
    # file_path="slmsuite/nv_blob_detection/nv_blob_rubin_shallow_140nvs_reordered_updated.npz",
    # file_path="slmsuite/nv_blob_detection/nv_blob_shallow_89nvs_reordered.npz",
    # file_path="slmsuite/nv_blob_detection/nv_blob_shallow_52nvs_reordered.npz",
    # file_path="slmsuite/nv_blob_detection/nv_blob_filtered_160nvs_reordered.npz",
    # file_path="slmsuite/nv_blob_detection/nv_blob_filtered_160nvs_reordered_updated.npz",  # after shutdownb
    # file_path="slmsuite/nv_blob_detection/nv_blob_filtered_160nvs_reordered_selected_117nvs.npz",
    # file_path="slmsuite/nv_blob_detection/nv_blob_filtered_160nvs_reordered_selected_117nvs_updated.npz",
    # file_path="slmsuite/nv_blob_detection/nv_blob_filtered_160nvs_reordered_selected_106nvs.npz",
):
    data = np.load(file_path, allow_pickle=True)
    nv_coordinates = data["nv_coordinates"]
    # spot_weights = data["spot_weights"]
    spot_weights = data["updated_spot_weights"]
    # spot_weights = data["integrated_counts"]
    # print(len(spot_weights))
    print(f"spot_weights: {spot_weights}")
    # spot_weights = data["integrated_counts"]
    return nv_coordinates, spot_weights


nuvu_pixel_coords, spot_weights = load_nv_coords()
print(f"Total NV coordinates: {len(nuvu_pixel_coords)}")
thorcam_coords = nuvu2thorcam_calibration(nuvu_pixel_coords).T
# sys.exit()


def compute_and_write_nvs_phase():
    hologram = SpotHologram(
        shape=(4096, 2048),
        spot_vectors=thorcam_coords,
        basis="ij",
        spot_amp=spot_weights,
        cameraslm=fs,
    )
    # Precondition computationally
    hologram.optimize(
        "WGS-Kim",
        maxiter=30,
        feedback="computational_spot",
        stat_groups=["computational_spot"],
    )

    initial_phase = hologram.extract_phase()
    # Define the path to save the phase data1
    file_path = r"slmsuite\computed_phase"
    num_nvs = len(nuvu_pixel_coords)
    now = datetime.now()
    date_time_str = now.strftime("%Y%m%d_%H%M%S")
    filename = f"slm_phase_{num_nvs}nvs_{date_time_str}.npy"
    # file_path = dm.get_file_path(__file__, filename)
    # Save the phase data
    save(initial_phase, file_path, filename)
    slm.write(initial_phase, settle=True)
    # cam_plot()


def write_pre_computed_nvs_phase():
    phase = np.load("slmsuite\computed_phase\slm_phase_75nvs_20250429_105705.npy")
    slm.write(phase, settle=True)
    # cam_plot()


# Define the save function
def save(data, path, filename):
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(os.path.join(path, filename), data)


try:
    slm = ThorSLM(serialNumber="00429430")
    cam = ThorCam(serial="26438", verbose=True)
    fs = FourierSLM(cam, slm)
    # cam = tb.get_server_thorcam()
    # slm = tb.get_server_thorslm()
    # fourier_calibration()
    load_fourier_calibration()
    # test_wavefront_calibration()
    # wavefront_calibration()
    # load_wavefront_calibration()
    compute_and_write_nvs_phase()
    # write_pre_computed_nvs_phase()
    # calibration_triangle()
    # circles()
    # smiley()
    # cam_plot()
finally:
    print("Closing")
    slm.close_window()
    slm.close_device()
    cam.close()
# endregions
