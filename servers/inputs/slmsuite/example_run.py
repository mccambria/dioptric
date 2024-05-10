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
# Generate a phase .gif
from IPython.display import Image
import imageio

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

# region "blaze" function
def blaze():
    # Get .2 degrees in normalized units.
    vector_deg = (.2, .2)
    vector = toolbox.convert_blaze_vector(vector_deg, from_units="deg", to_units="norm")

    # Get the phase for the new vector
    blaze_phase = toolbox.phase.blaze(grid=slm, vector=vector)

    plot_phase(blaze_phase, title="Blaze at {} deg".format(vector_deg))

# region "calibration"
def fourier_calibration():
    cam.set_exposure(.003)               # Increase exposure because power will be split many ways
    fs.fourier_calibrate(
        array_shape=[25, 16],           # Size of the calibration grid (Nx, Ny) [knm]
        array_pitch=[30, 40],           # Pitch of the calibration grid (x, y) [knm]
        plot=True
    )
    cam.set_exposure(.0001)
    #save calibation
    calibration_file = fs.save_fourier_calibration(path=r"C:\Users\Saroj Chand\Documents\fourier_calibration")
    print("Fouri er calibration saved to:", calibration_file)

def wavefront_calibration():
    cam.set_exposure(.001)
    fs.wavefront_calibrate(
        interference_point=(1100, 300),
        field_point=(.25, 0),
        field_point_units="freq",
        superpixel_size=120,
        autoexposure=False
    )
    #save calibation
    calibration_file = fs.save_wavefront_calibration(path=r"C:\Users\Saroj Chand\Documents\wavefront_calibration")
    print("Fourier calibration saved to:", calibration_file)
    
# region "load calibration" 
def load_fourier_calibration():
    calibration_file_path = r"C:\Users\Saroj Chand\Documents\fourier_calibration\26438-SLM-fourier-calibration_00000.h5"
    fs.load_fourier_calibration(calibration_file_path)
    print("Fourier calibration loaded from:", calibration_file_path)

def load_wavefront_calibration():
    calibration_file_path = r""
    fs.load_wavefront_calibration(calibration_file_path)
    print("Fourier calibration loaded from:", calibration_file_path)

def test_wavefront_calibration():
    cam.set_exposure(.0001)
    movie = fs.wavefront_calibrate(
        interference_point=(1100, 300),
        field_point=(.25, 0),
        field_point_units="freq",
        superpixel_size=50,
        test_superpixel=(16, 16),           # Testing mode
        autoexposure=False,
        plot=3                              # Special mode to generate a phase .gif
    )
    imageio.mimsave('wavefront.gif', movie)
    Image(filename="wavefront.gif")
    
# region "cam_plot" function 
def cam_plot():
    cam.set_exposure(.0001)
    img = cam.get_image()

    # Plot the result
    plt.figure(figsize=(18, 9))
    plt.imshow(img)
    plt.show()

# region "evaluate" function 
def evaluate_uniformity(vectors=None, size=25):
    # Set exposure and capture image
    cam.set_exposure(0.00001)
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

    # return subimages, powers

# region "square" function 
def square_array():
    xlist = np.arange(350, 1150, 100)                      # Get the coordinates for one edge
    ylist = np.arange(240, 1040, 100) 
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
    slm.write(phase, settle=True)
    cam_plot()
    evaluate_uniformity(vectors=square)
    # Hone the result with experimental feedback.
    hologram.optimize(
        'WGS-Kim',
        maxiter=20,
        feedback='experimental_spot',
        stat_groups=['computational_spot', 'experimental_spot'],
        fixed_phase=False
    )
    phase = hologram.extract_phase()
    slm.write(phase, settle=True)
    cam_plot()
    evaluate_uniformity(vectors=square)

# region "circle" function 
def circle_pattern():
    cam.set_exposure(0.001)
   # Define parameters for the circle
    center = (720, 540)  # Center of the circle
    radius = 300  # Radius of the circle

    # Generate points within the circle using polar coordinates
    num_points = 20  # Number of points to generate
    theta = np.linspace(0, 2*np.pi, num_points)  # Angle values
    x_circle = center[0] + radius * np.cos(theta)  # X coordinates
    y_circle = center[1] + radius * np.sin(theta)  # Y coordinates

    # Convert to grid format if needed
    circle = np.vstack((x_circle, y_circle))

    hologram = SpotHologram(shape=(2048, 2048), spot_vectors=circle, basis='ij', cameraslm=fs)

    # # Precondition computationally.
    hologram.optimize(
        'WGS-Kim',
        maxiter=20,
        feedback='computational_spot',
        stat_groups=['computational_spot']
    )
    phase = hologram.extract_phase()
    slm.write(phase, settle=True)
    cam_plot()
    evaluate_uniformity(vectors=circle)


    # Hone the result with experimental feedback.
    hologram.optimize(
        'WGS-Kim',
        maxiter=20,
        feedback='experimental_spot',
        stat_groups=['computational_spot', 'experimental_spot'],
        fixed_phase=False
    )
    phase = hologram.extract_phase()
    slm.write(phase, settle=True)
    cam_plot()
    evaluate_uniformity(vectors=circle)

# region "scatter" function 
def scatter_pattern():
    lloyds_points = toolbox.lloyds_points(
        grid=tuple(int(s/5) for s in fs.cam.shape), 
        n_points=100, 
        iterations=40
        ) * 5
    
    hologram = SpotHologram((2048, 2048), lloyds_points, basis='ij', cameraslm=fs)

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
    # # Hone the result with experimental feedback.
    hologram.optimize(
        'WGS-Kim',
        maxiter=20,
        feedback='experimental_spot',
        stat_groups=['computational_spot', 'experimental_spot'],
        fixed_phase=False
    )
    phase = hologram.extract_phase()
    slm.write(phase, settle=True)
    cam_plot()

# region "smiley" function 

def smiley_pattern():
  # Define points for the smiley face
    x_eyes = [1100, 1200]  # X coordinates for the eyes
    y_eyes = [400, 400]     # Y coordinates for the eyes

    # Define points for the mouth (a semi-circle)
    theta = np.linspace(0, np.pi, 15)  # Angle values for the semi-circle
    mouth_radius = 150  # Radius of the mouth
    x_mouth = 1150 + mouth_radius * np.cos(theta)  # X coordinates for the mouth
    y_mouth = 500 + mouth_radius * np.sin(theta)   # Y coordinates for the mouth

    # Combine all points into a single array
    smiley = np.vstack((np.concatenate((x_eyes, x_mouth)), np.concatenate((y_eyes, y_mouth))))
    hologram = SpotHologram(shape=(2048, 2048), spot_vectors=smiley, basis='ij', cameraslm=fs)

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

# region "UCB"  function 
def UCB_pattern():
    # Define coordinates for each letter in "UCB"
    letters = {
        'U': [(700, 400), (650, 500), (650, 600), (700, 700), (750, 600), (750, 500)],
        'C': [(800, 500), (800, 600), (850, 600), (900, 550), (850, 500), (900, 500)],
        'B': [(950, 500), (950, 600), (1000, 600), (1050, 550), (1000, 500), (1050, 500)],
    }

    # Combine coordinates for "UCB"
    ucb = np.vstack([letters[letter] for letter in "UCB"]).T

    hologram = SpotHologram(shape=(2048, 2048), spot_vectors=ucb, basis='ij', cameraslm=fs)

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


# region run funtions
slm = ThorSLM(serialNumber='00429430')
try:
    cam = ThorCam(serial="26438", verbose=True)
    fs = FourierSLM(cam, slm)
    # fs.load_wavefront_calibration(plot=True)
    # blaze()
    # fourier_calibration()
    load_fourier_calibration()
    wavefront_calibration()
    # test_wavefront_calibration()
    # square_array()
    # circle_pattern()
    # smiley()
    # scatter_pattern()
    # cam_plot()

finally:
    # After you're done using the camera
    cam.close()  # Add this line
    # Then close the SDK
    ThorCam.close_sdk()
# endregions