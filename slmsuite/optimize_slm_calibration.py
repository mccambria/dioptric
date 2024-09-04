import os
import sys
import warnings

# Add slmsuite to the Python path if not installed via pip.
sys.path.append(os.path.join(os.getcwd(), "c:/Users/Saroj Chand/Documents/dioptric"))

# Suppress warnings
warnings.filterwarnings("ignore")

# Import necessary libraries
import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
from IPython.display import Image
from scipy.optimize import curve_fit

from majorroutines.widefield import image_sample

# Import slmsuite and related modules
from slmsuite.hardware.cameras.thorlabs import ThorCam
from slmsuite.hardware.cameraslms import FourierSLM
from slmsuite.hardware.slms.thorlabs import ThorSLM
from slmsuite.holography import analysis, toolbox
from slmsuite.holography.algorithms import FeedbackHologram, SpotHologram
from slmsuite.misc import fitfunctions
from utils import common, widefield
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils.constants import LaserKey

# Set matplotlib defaults
plt.rc("image", cmap="Blues")


def cam_plot(cam, exposure=0.0001):
    """
    Capture and display an image from the camera.

    Parameters:
        cam (ThorCam): Camera object.
        exposure (float): Exposure time for the camera.
    """
    cam.set_exposure(exposure)
    img = cam.get_image()

    plt.figure(figsize=(12, 9))
    plt.imshow(img)
    plt.show()


def nuvu2thorcam_calibration(coords):
    """
    Calibrates and transforms coordinates from the Nuvu camera's coordinate system
    to the Thorlabs camera's coordinate system using an affine transformation.

    Parameters:
    coords (np.ndarray): An array of shape (N, 2) containing coordinates in the Nuvu camera's system.

    Returns:
    np.ndarray: An array of shape (N, 2) containing transformed coordinates in the Thorlabs camera's system.
    """
    # Define the corresponding points in both coordinate systems
    cal_coords_nuvu = np.array(
        [[131.091, 74.744], [130.013, 141.887], [69.072, 104.945]], dtype="float32"
    )  # Points in (512, 512) coordinate system
    cal_coords_thorcam = np.array(
        [[853.92, 590.0], [646.07, 590.0], [750.0, 410.0]], dtype="float32"
    )  # Corresponding points in (1480, 1020) coordinate system

    # Compute the affine transformation matrix
    M = cv2.getAffineTransform(cal_coords_nuvu, cal_coords_thorcam)

    # Append a column of ones to the input coordinates to facilitate affine transformation
    ones_column = np.ones((coords.shape[0], 1))
    coords_homogeneous = np.hstack((coords, ones_column))

    # Perform the affine transformation
    thorcam_coords = np.dot(coords_homogeneous, M.T)

    return thorcam_coords


def load_calibration(fs, calibration_type):
    """
    Load Fourier or wavefront calibration for the SLM.

    Parameters:
        fs (FourierSLM): The FourierSLM object.
        calibration_type (str): Type of calibration to load ("fourier" or "wavefront").
    """
    calibration_paths = {
        "fourier": r"C:\Users\matth\GitHub\dioptric\slmsuite\fourier_calibration\26438-SLM-fourier-calibration_00003.h5",
        "wavefront": r"C:\Users\matth\GitHub\dioptric\slmsuite\wavefront_calibration\26438-SLM-wavefront-calibration_00004.h5",
    }

    calibration_file_path = calibration_paths.get(calibration_type)
    if calibration_file_path:
        if calibration_type == "fourier":
            fs.load_fourier_calibration(calibration_file_path)
        elif calibration_type == "wavefront":
            fs.load_wavefront_calibration(calibration_file_path)
        print(
            f"{calibration_type.capitalize()} calibration loaded from: {calibration_file_path}"
        )
    else:
        raise ValueError("Invalid calibration type. Choose 'fourier' or 'wavefront'.")


def generate_initial_phase(fs, nv_centers, scanning_range=0.3, num_points=10):
    """
    Generate the initial phase for the SLM based on NV center positions.

    Parameters:
        fs (FourierSLM): The FourierSLM object.
        nv_centers (np.ndarray): Array of NV center coordinates.
        scanning_range (float): Scanning range for phase optimization.
        num_points (int): Number of points in the phase optimization.

    Returns:
        np.ndarray: The initial phase array.
    """
    hologram = SpotHologram(
        shape=(2048, 2048), spot_vectors=nv_centers, basis="ij", cameraslm=fs
    )
    hologram.optimize(
        "WGS-Kim",
        maxiter=20,
        feedback="computational_spot",
        stat_groups=["computational_spot"],
    )
    return hologram.extract_phase()


def shift_phase(phase, shift_x, shift_y):
    """
    Shift the phase by adding a phase gradient.

    Parameters:
        phase (np.ndarray): The current phase array.
        shift_x (float): The shift in the x direction.
        shift_y (float): The shift in the y direction.

    Returns:
        np.ndarray: The shifted phase array.
    """
    y_indices, x_indices = np.indices(phase.shape)
    phase_shift = shift_x * x_indices + shift_y * y_indices
    shifted_phase = phase + phase_shift
    return shifted_phase


def evaluate_counts(image, target_coords, radius=3):
    """
    Evaluate the image by summing counts within a given radius around target coordinates.

    Parameters:
        image (np.ndarray): Image array.
        target_coords (np.ndarray): Array of target coordinates.
        radius (int): Radius for summing counts around target coordinates.

    Returns:
        int: Total counts around the target coordinates.
    """
    if target_coords.ndim == 1:
        target_coords = np.array([target_coords])

    total_counts = 0

    for x, y in np.round(target_coords).astype(int):
        y_indices, x_indices = np.ogrid[: image.shape[0], : image.shape[1]]
        mask = (x_indices - x) ** 2 + (y_indices - y) ** 2 <= radius**2
        total_counts += np.sum(image[mask])

    return total_counts


def collect_counts_for_shifts(
    slm,
    initial_phase,
    target_coords,
    shift_range=(-0.1, 0.1),
    num_steps=10,
):
    """
    Collect counts for different phase shifts.

    Parameters:
        slm (ThorSLM): The SLM object.
        initial_phase (np.ndarray): The initial phase array.
        target_coords (np.ndarray): Array of target coordinates.
        shift_range (tuple): Range of phase shifts.
        num_steps (int): Number of steps for shifting.

    Returns:
        np.ndarray: Phase shifts.
        np.ndarray: Corresponding counts.
    """
    shifts = np.linspace(shift_range[0], shift_range[1], num_steps)
    counts = np.zeros_like(shifts)

    for i, shift in enumerate(shifts):
        shifted_phase = shift_phase(initial_phase, shift, 0)
        slm.write(shifted_phase)
        image = take_image()
        counts[i] = evaluate_counts(image, target_coords)

    return shifts, counts


# Define the save function
def save(data, path, filename):
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(os.path.join(path, filename), data)


def take_image(nv_sig=None, num_reps=20, display_image=False):
    """
    Capture an image using the SLM and camera.

    Parameters:
        fs (FourierSLM): The FourierSLM object.
        nv_sig (NVSig, optional): NV signal information used for positioning and imaging.
        num_reps (int): Number of repetitions for averaging the image.
        display_image (bool): If True, display the captured image.

    Returns:
        np.ndarray: Averaged image array.
    """
    tb.reset_cfm()

    laser_key = LaserKey.WIDEFIELD_IMAGING
    laser_dict = tb.get_optics_dict(laser_key)
    readout_laser = laser_dict["name"]
    readout_duration = laser_dict["duration"]

    if nv_sig is not None:
        tb.get_server_positioning().set_xyz_on_nv(nv_sig)

    camera = tb.get_server_camera()
    pulse_gen = tb.get_server_pulse_gen()

    seq_args = [readout_duration, readout_laser]
    pulse_gen.stream_load(
        "simple_readout-widefield.py", tb.encode_seq_args(seq_args), num_reps
    )

    camera.arm()
    img_array_list = []

    def rep_fn(rep_ind):
        img_str = camera.read()
        sub_img_array, _ = widefield.img_str_to_array(img_str)
        img_array_list.append(sub_img_array)

    widefield.rep_loop(num_reps, rep_fn)
    camera.disarm()

    img_array = np.mean(img_array_list, axis=0)

    if display_image:
        fig, ax = plt.subplots()
        kpl.imshow(ax, img_array, title="Widefield Image", cbar_label="ADUs")
        kpl.show()

    return img_array


def fit_gaussian_to_counts(shifts, counts):
    """
    Fit a Gaussian model to the counts versus phase shift data.

    Parameters:
        shifts (np.ndarray): Phase shifts.
        counts (np.ndarray): Corresponding counts.

    Returns:
        float: Optimal phase shift.
        np.ndarray: Fitted Gaussian curve.
    """

    def gaussian(x, amplitude, mean, stddev, offset):
        return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev**2)) + offset

    # Initial guesses for the Gaussian parameters
    amplitude_guess = np.max(counts) - np.min(counts)
    mean_guess = shifts[np.argmax(counts)]
    stddev_guess = (np.max(shifts) - np.min(shifts)) / 4
    offset_guess = np.min(counts)

    try:
        popt, _ = curve_fit(
            gaussian,
            shifts,
            counts,
            p0=[amplitude_guess, mean_guess, stddev_guess, offset_guess],
        )
        optimal_shift = popt[1]  # The mean of the Gaussian fit

        # Generate the fitted Gaussian curve
        fitted_curve = gaussian(shifts, *popt)

        # Print the fitted parameters
        print(
            f"Fitted parameters: Amplitude = {popt[0]:.4f}, Mean = {popt[1]:.4f}, "
            f"StdDev = {popt[2]:.4f}, Offset = {popt[3]:.4f}"
        )
    except Exception as e:
        print("Error fitting Gaussian:", e)
        optimal_shift = mean_guess
        fitted_curve = np.zeros_like(shifts)

    return optimal_shift, fitted_curve


def optimize_phase_for_nv(
    slm,
    initial_phase,
    target_coord,
    nv_index,
    shift_range_x=(-0.1, 0.1),
    shift_range_y=(-0.1, 0.1),
    num_steps=10,
    num_iterations=3,
    save_path="optimized_phases",
):
    """
    Optimize the SLM phase for a specific NV center.

    Parameters:
        slm (ThorSLM): The SLM object.
        initial_phase (np.ndarray): The initial phase array.
        target_coord (np.ndarray): Coordinate of the NV center.
        nv_index (int): Index of the NV center.
        shift_range_x (tuple): Range of phase shifts in the X direction.
        shift_range_y (tuple): Range of phase shifts in the Y direction.
        num_steps (int): Number of steps for shifting.
        num_iterations (int): Number of iterations for optimization.
        save_path (str): Directory path to save the optimized phases.
    """
    phase = initial_phase.copy()

    for iteration in range(num_iterations):
        # Optimize X direction
        shifts_x, counts_x = collect_counts_for_shifts(
            slm, phase, target_coord, shift_range_x, num_steps
        )
        optimal_shift_x, fitted_curve_x = fit_gaussian_to_counts(shifts_x, counts_x)

        # Apply the optimal X phase shift
        phase = shift_phase(phase, optimal_shift_x, 0)
        slm.write(phase)
        image = take_image()
        counts = evaluate_counts(image, target_coord)

        # Optimize Y direction
        shifts_y, counts_y = collect_counts_for_shifts(
            slm, phase, target_coord, shift_range_y, num_steps
        )
        optimal_shift_y, fitted_curve_y = fit_gaussian_to_counts(shifts_y, counts_y)

        # Apply the optimal Y phase shift
        phase = shift_phase(phase, 0, optimal_shift_y)
        slm.write(phase)
        image = take_image()
        counts = evaluate_counts(image, target_coord)

        # Display progress and fitted Gaussian curves
        plt.figure(figsize=(10, 5))

        # X direction
        plt.subplot(1, 2, 1)
        plt.plot(shifts_x, counts_x, "o", label="Measured Counts X")
        plt.plot(shifts_x, fitted_curve_x, "-", label="Fitted Gaussian X")
        plt.axvline(optimal_shift_x, color="r", linestyle="--", label="Optimal Shift X")
        plt.xlabel("Phase Shift X")
        plt.ylabel("Counts")
        plt.legend()
        plt.title(f"Iteration {iteration + 1} - X direction")

        # Y direction
        plt.subplot(1, 2, 2)
        plt.plot(shifts_y, counts_y, "o", label="Measured Counts Y")
        plt.plot(shifts_y, fitted_curve_y, "-", label="Fitted Gaussian Y")
        plt.axvline(optimal_shift_y, color="r", linestyle="--", label="Optimal Shift Y")
        plt.xlabel("Phase Shift Y")
        plt.ylabel("Counts")
        plt.legend()
        plt.title(f"Iteration {iteration + 1} - Y direction")

        plt.tight_layout()
        plt.show()

    # Save the optimized phase
    save_filename = f"optimized_phase_nv_{nv_index}.npy"
    save(phase, save_path, save_filename)
    print(f"Optimized phase for NV {nv_index} saved to {save_filename}")

    return phase


def main():
    target_coords = np.array([[110.186, 129.281], [128.233, 88.007], [86.294, 103.0]])

    slm = ThorSLM(serialNumber="00429430")
    cam = ThorCam(serial="26438", verbose=True)
    fs = FourierSLM(cam, slm)

    try:
        fourer_calibration = load_calibration(fs, "fourier")
        initial_phase = np.load(
            r"C:\Users\matth\GitHub\dioptric\slmsuite\Initial_phase\initial_phase.npy"
        )

        # Loop over each NV center and optimize its phase separately
        for i, target_coord in enumerate(target_coords):
            print(f"Optimizing phase for NV {i}")
            optimize_phase_for_nv(
                slm,
                initial_phase,
                target_coord,
                nv_index=i,
                save_path=r"C:\path\to\save\optimized_phases",
            )

    finally:
        print("Closing devices.")
        slm.close_window()
        slm.close_device()
        cam.close()


if __name__ == "__main__":
    main()
