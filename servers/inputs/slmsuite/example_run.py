import os, sys, time
# Add slmsuite to the python path (for the case where it isn't installed via pip).
sys.path.append(os.path.join(os.getcwd(), 'c:/Users/Saroj Chand/Documents/dioptric/servers/inputs'))

import numpy as np
import cv2
import scipy.ndimage as ndimage

import warnings
warnings.filterwarnings("ignore")

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc('image', cmap='Blues')

from slmsuite.holography import analysis, toolbox
from slmsuite.hardware.slms.thorlabs import ThorSLM
from slmsuite.hardware.cameras.thorlabs import ThorCam
from slmsuite.hardware.cameraslms import FourierSLM
from slmsuite.holography.algorithms import FeedbackHologram, SpotHologram

# funtions
def plot_phase(phase, title="", zoom=True):
    # One plot if no camera; two otherwise.
    _, axs = plt.subplots(1, 2 - (cam is None), figsize=(24,6))

    if cam is None:
        axs = [axs]

    # Plot the phase.
    axs[0].set_title("SLM Phase")
    im = axs[0].imshow(
        np.mod(phase, 2*np.pi),
        vmin=0,
        vmax=2*np.pi,
        interpolation="none",
        cmap="twilight"
    )
    plt.colorbar(im, ax=axs[0])

    # Grab an image of the resulting pattern and plot.
    slm.write(phase, settle=True)
    img = cam.get_image()

    axs[1].set_title("Camera Result")
    axs[1].imshow(img)
    if zoom:
        xlim = axs[1].get_xlim()
        ylim = axs[1].get_ylim()
        axs[1].set_xlim([xlim[0] * .7 + xlim[1] * .3, xlim[0] * .3 + xlim[1] * .7])
        axs[1].set_ylim([ylim[0] * .7 + ylim[1] * .3, ylim[0] * .3 + ylim[1] * .7])

    # Make a title, if given.
    plt.suptitle(title)
    plt.show()

def write_to_slm():
    slm_size = (1080, 1920)
    hologram = SpotHologram.make_rectangular_array(
    slm_size,
    array_shape=(15,10),
    array_pitch=(60,40),
    basis='knm'
    )
    hologram.optimize('WGS-Kim', feedback='computational_spot', stat_groups=['computational_spot'], maxiter=50)
    phase = hologram.extract_phase()
    slm.write(phase, settle=True)

def fourier_calibration():
    cam.set_exposure(.0001)               # Increase exposure because power will be split many ways
    fs.fourier_calibrate(
        array_shape=[30, 20],           # Size of the calibration grid (Nx, Ny) [knm]
        array_pitch=[30, 40],           # Pitch of the calibration grid (x, y) [knm]
        plot=True
    )
    cam.set_exposure(.0002)
    #save calibation
    calibration_file = fs.save_fourier_calibration()
    print("Fouri er calibration saved to:", calibration_file)

def load_fourier_calibration():
    calibration_file_path = r"C:\Users\Saroj Chand\Documents\dioptric\26438-SLM-fourier-calibration_00000.h5"
    fs.load_fourier_calibration(calibration_file_path)
    print("Fourier calibration loaded from:", calibration_file_path)

def cam_plot():
    cam.set_exposure(.001)
    img = cam.get_image()

    # Plot the result
    plt.figure(figsize=(18, 9))
    plt.imshow(img)
    plt.show()

def computaional_feedback():
    xlist = np.arange(900, 1300, 100)
    ylist = np.arange(300, 700, 100)                       # Get the coordinates for one edge
    xgrid, ygrid = np.meshgrid(xlist, ylist)
    square = np.vstack((xgrid.ravel(), ygrid.ravel()))      # Make an array of points in a grid
    hologram = SpotHologram(shape=(2048, 2048), spot_vectors=square, basis='ij', cameraslm=fs)

    # Precondition computationally.
    hologram.optimize(
        'WGS-Kim',
        maxiter=20,
        feedback='computational_spot',
        stat_groups=['computational_spot']
    )
    phase = hologram.extract_phase()
    slm.write(phase, settle=True)
    cam_plot()

def experiment_feedback():
    xlist = np.arange(100, 1000, 100)                      # Get the coordinates for one edge
    xgrid, ygrid = np.meshgrid(xlist, xlist)
    square = np.vstack((xgrid.ravel(), ygrid.ravel()))      # Make an array of points in a grid
    hologram = SpotHologram(shape=(2048, 2048), spot_vectors=square, basis='ij', cameraslm=fs)

    # Precondition computationally.
    hologram.optimize(
        'WGS-Kim',
        maxiter=20,
        feedback='computational_spot',
        stat_groups=['computational_spot']
    )
    # Hone the result with experimental feedback.
    hologram.optimize(
        'WGS-Kim',
        maxiter=20,
        feedback='experimental_spot',
        stat_groups=['computational_spot', 'experimental_spot'],
        fixed_phase=False
    )

# run commands
slm = ThorSLM(serialNumber='00429430')
# write_to_slm()
try:
    cam = ThorCam(serial="26438", verbose=True)
    fs = FourierSLM(cam, slm)
    # fourier_calibration()
    load_fourier_calibration()
    computaional_feedback()
    # experiment_feedback()
    # cam_plot()

finally:
    # After you're done using the camera
    cam.close()  # Add this line

    # Then close the SDK
    ThorCam.close_sdk()
# cam = None

#Connect to SLM 
# slm = ThorSLM(serialNumber='00429430') 
# slm_size = (1080, 1920)
# slm.fourier_calibrate(
#     array_shape=[30, 20],           # Size of the calibration grid (Nx, Ny) [knm]
#     array_pitch=[30, 40],           # Pitch of the calibration grid (x, y) [knm]
#     plot=True
# )
# make a rectangular grid in the knm basis
# hologram = SpotHologram.make_rectangular_array(
#     slm_size,
#     array_shape=(20,30),
#     array_pitch=(40,20),
#     basis='knm'
# )
# hologram.plot_farfield(title="Before Optimization")
# hologram.optimize('WGS-Kim', feedback='computational_spot', stat_groups=['computational_spot'], maxiter=50)
# # hologram.plot_farfield(title="After Optimization")

# # hologram.plot_nearfield(title="Padded", padded=True)
# # hologram.plot_nearfield(title="Unpadded")

# phase = hologram.extract_phase()
# plot_phase(phase, title="optical Tweezers")

# Get .2 degrees in normalized units.
# vector_deg = (.2, .2)
# vector = toolbox.convert_blaze_vector(vector_deg, from_units="deg", to_units="norm")

# # Get the phase for the new vector
# blaze_phase = toolbox.phase.blaze(grid=slm, vector=vector)

# plot_phase(blaze_phase, title="Blaze at {} deg".format(vector_deg))