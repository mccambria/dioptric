import os
import sys
import time

# Add slmsuite to the python path (for the case where it isn't installed via pip).
sys.path.append(os.path.join(os.getcwd(), "c:/Users/Saroj Chand/Documents/dioptric"))

import warnings

import cv2
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage

warnings.filterwarnings("ignore")
import io

import imageio
import matplotlib as mpl
import matplotlib.pyplot as plt

# Generate a phase .gif
from IPython.display import Image
from scipy.optimize import curve_fit

from utils import tool_belt as tb

mpl.rc("image", cmap="Blues")

from slmsuite import example_library
from slmsuite.hardware.cameras.thorlabs import ThorCam
from slmsuite.hardware.cameraslms import FourierSLM
from slmsuite.hardware.slms.thorlabs import ThorSLM
from slmsuite.holography import analysis, toolbox
from slmsuite.holography.algorithms import FeedbackHologram, SpotHologram
from slmsuite.misc import fitfunctions


from majorroutines.widefield import image_sample

def cam_plot():
    cam.set_exposure(0.0001)
    img = cam.get_image()
    # Plot the result
    plt.figure(figsize=(12, 9))
    plt.imshow(img)
    plt.show()


def load_fourier_calibration():
    calibration_file_path = r"C:\Users\matth\GitHub\dioptric\slmsuite\fourier_calibration\26438-SLM-fourier-calibration_00003.h5"
    fs.load_fourier_calibration(calibration_file_path)
    print("Fourier calibration loaded from:", calibration_file_path)


def load_wavefront_calibration():
    calibration_file_path = r"C:\Users\matth\GitHub\dioptric\slmsuite\wavefront_calibration\26438-SLM-wavefront-calibration_00004.h5"
    fs.load_wavefront_calibration(calibration_file_path)
    print("Wavefront calibration loaded from:", calibration_file_path)


# Example usage with a list of Nuvu pixel coordinates
nuvu_pixel_coords = [
    [110.186, 129.281],
    [128.233, 88.007],
    [86.294, 103.0],
]
# thesre are coord at which
# 87.614, 104.484
# 129.637, 89.599
# 111.651, 130.532
nuvu_pixel_coords_array = np.array(nuvu_pixel_coords)
# Calibrate the coordinates from Nuvu to Thorlabs system
thorcam_coords = example_library.nuvu2thorcam_calibration(nuvu_pixel_coords_array)
# thorcam_coords = thorcam_coords.T  #Transpose


def scanning_range(x1, x2, num_points=10):
    shifts = np.linspace(x1, x2, num=num_points)
    x_path, y_path = []
    for shift in shifts:
        x_path.append((shift, -x2))  # Right
    for shift in shifts:
        y_path.append((x2, shift))  # Up
    return path

def calculate_initial_phase():
    # Ensure thorcam_coords has shape (2, n)
    thorcam_coords = example_library.nuvu2thorcam_calibration(nuvu_pixel_coords_array).T

    if thorcam_coords.shape[0] != 2:
        raise ValueError(
            f"Expected thorcam_coords to have shape (2, n), but got {thorcam_coords.shape}"
        )

    hologram = SpotHologram(
        shape=(2048, 2048), spot_vectors=thorcam_coords, basis="ij", cameraslm=fs
    )

    # Precondition computationally
    hologram.optimize(
        "WGS-Kim",
        maxiter=20,
        feedback="computational_spot",
        stat_groups=["computational_spot"],
    )

    initial_phase = hologram.extract_phase()
    # path = r"C:\Users\Saroj Chand\Documents\slm_phase"
    # filename = "initial_phase.npy"
    # save(initial_phase, path, filename)

    def opimize_slm_calibartion():
        number_attempt = 5
        try:
            while num_attemp <number_attempt:
                for angle in range(0, 360, 30):
                    # Update phase with live rotation
                    delta_phase = 
                    phase = initial_phase + delta_phase
                    # Display the phase pattern on the SLM
                    slm.write(phase, settle=True)
                    image_array = image_sample.widefield_image(nv_sig, num_reps)
                    # Capture image from the camera
                    cam.set_exposure(0.001)
                    im = cam.get_image()
        finally:
            print("Real-time dynamical tweezers operation stopped.")


# Define the save function
def save(data, path, filename):
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(os.path.join(path, filename), data)


def main():
    try:
        slm = ThorSLM(serialNumber="00429430")
        cam = ThorCam(serial="26438", verbose=True)
        fs = FourierSLM(cam, slm)
        load_fourier_calibration()
        # load_wavefront_calibration()
        # camp2phase_calibration()
        Optimize_phase_calibarion()
    finally:
        print("Closing")
        slm.close_window()
        slm.close_device()
        cam.close()
 
 if __name__ = "__main__"