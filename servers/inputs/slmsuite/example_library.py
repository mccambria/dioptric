import os, sys, time
# Add slmsuite to the python path (for the case where it isn't installed via pip).
sys.path.append(os.path.join(os.getcwd(), 'c:/Users/Saroj Chand/Documents/dioptric/servers/inputs'))

import numpy as np
import pandas as pd
import cv2
import scipy.ndimage as ndimage
import warnings
warnings.filterwarnings("ignore")
import matplotlib as mpl
import matplotlib.pyplot as plt
# Generate a phase .gif
from IPython.display import Image
import imageio
import io 


mpl.rc('image', cmap='Blues')

from slmsuite.holography import analysis, toolbox
from slmsuite.hardware.slms.thorlabs import ThorSLM
from slmsuite.hardware.cameras.thorlabs import ThorCam
from slmsuite.hardware.cameraslms import FourierSLM
from slmsuite.holography.algorithms import FeedbackHologram, SpotHologram

# funtions
# region "plot_phase" function
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

# integrate Intensity
def integrate_intensity():
    xlist = np.arange(650, 750, 25)                      # Get the coordinates for one edge
    ylist = np.arange(340, 440, 25) 
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
    hologram.plot_nearfield(title="Padded", padded=True)
    hologram.plot_nearfield(title="Unpadded")
    phase = hologram.extract_phase()
    # for ind in range(phase.shape[0]):
    #     for jnd in range(phase.shape[1]):
    #         phase[ind, jnd] *= np.dot((ind, jnd), (0.5, 0))
    # print(phase)
    # phase *= np.exp(1j * np.dot(np.meshgrid(phase.shape), (10,15))), 855,502.5
    slm.write(phase, settle=True)
    intensity_map = cam_plot()
    cam.set_exposure(.0001)
    img = cam.get_image()
    # Define the region of interest (ROI) around the center spot and compute intensity
    x, y = 700, 390  # Center spot coordinates, adjust as necessary
    roi_size = 110  # Size of the region around the center spot to analyze
    center_intensity = img[y-roi_size:y+roi_size, x-roi_size:x+roi_size].sum()
    roi_size = 50
    x, y = 830, 540
    # Compute intensities for each spot in the grid
    total_intensity = img[y-roi_size:y+roi_size, x-roi_size:x+roi_size].sum()
    # Print or plot the intensities for analysis
    print("Center spot intensity:", center_intensity)
    print(f"all intensity: {total_intensity}")

def plot_intensity():
    # Data from the image
    data = {
        "Power": [1, 1.2, 1.4, 1.6, 2, 2.4, 3, 4, 6, 8],
        "all_array_spot Intensity": [73240, 74239, 75155, 76969, 80057, 84386, 89551, 103747, 153496, 296225],
        "oth order": [13010, 13154, 13213, 13333, 13638, 14087, 14599, 15705, 20399, 33304]
    }

    df = pd.DataFrame(data)
    # Compute the relative intensity
    df['Relative Intensity'] = df['all_array_spot Intensity'] / df['oth order']
    # Plot the data
    plt.figure(figsize=(10, 6))
    # Plot relative intensity
    plt.plot(df['Power'], df['Relative Intensity'], label='Relative Intensity (All Array / 0th Order)', marker='s', color='blue')
    # plt.plot(df['Power'], df['all_array_spot Intensity'], label='Array Spots Integrated Intensity', marker='o', color='orange')
    # plt.plot(df['Power'], df['oth order'], label='0th Order Intensity', marker='x', color='red')

    plt.xlabel('Power')
    plt.ylabel('Intensity')
    plt.title('Intensity vs Power')
    plt.legend()
    plt.grid(True)
    plt.show()
# plot_intensity()


def shift_phase(phase, shift_x, shift_y):
    for ind in range(phase.shape[0]):
        for jnd in range(phase.shape[1]):
            phase[ind, jnd] += np.dot((ind, jnd), (shift_x, shift_y))
    return phase

def save_initial_phase():
    xlist = np.arange(550, 1150, 25)  # Get the coordinates for one edge
    ylist = np.arange(240, 840, 25)
    xgrid, ygrid = np.meshgrid(xlist, ylist)
    square = np.vstack((xgrid.ravel(), ygrid.ravel()))  # Make an array of points in a grid
    hologram = SpotHologram(shape=(2048, 2048), spot_vectors=square, basis='ij', cameraslm=fs)

    # Precondition computationally
    hologram.optimize(
        'WGS-Kim',
        maxiter=20,
        feedback='computational_spot',
        stat_groups=['computational_spot']
    )

    initial_phase = hologram.extract_phase()
    np.save('initial_phase.npy', initial_phase)  # Save the phase to a file
    print("Initial phase saved.")

def animate_wavefront_shifts():
    initial_phase = np.load('initial_phase.npy')  # Load the saved phase

    frames = []
    shifts = np.linspace(0, 0.3, num=10)  # Define the range and number of steps for the shifts

    # Define the square path
    path = []
    for shift in shifts:
        path.append((shift, 0))  # Right
    for shift in shifts:
        path.append((shifts[-1], shift))  # Up
    for shift in shifts[::-1]:
        path.append((shift, shifts[-1]))  # Left
    for shift in shifts[::-1]:
        path.append((0, shift))  # Down

    for shift_x, shift_y in path:
        shifted_phase = shift_phase(np.copy(initial_phase), shift_x=shift_x, shift_y=shift_y)
        slm.write(shifted_phase, settle=True)
        cam.set_exposure(0.0001)
        im = cam.get_image()
        if im is not None:
            plt.imshow(im)  # Use custom colormap
            plt.draw()
            # Save the current figure to a buffer
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image = imageio.imread(buffer)
            frames.append(image)
            buffer.close()
            plt.pause(0.01)  # Small pause to update the plot

    # Save frames as a GIF
    imageio.mimsave('wavefront_shift_animation.gif', frames, fps=5)
    print("Animation saved as wavefront_shift_animation.gif")