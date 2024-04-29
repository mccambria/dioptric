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

# from slmsuite.holography.algorithms import SpotHologram

cam = None
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
    # slm.write(phase, settle=True)
    # # img = cam.get_image()

    # axs[1].set_title("Camera Result")
    # axs[1].imshow(img)
    # if zoom:
    #     xlim = axs[1].get_xlim()
    #     ylim = axs[1].get_ylim()
    #     axs[1].set_xlim([xlim[0] * .7 + xlim[1] * .3, xlim[0] * .3 + xlim[1] * .7])
    #     axs[1].set_ylim([ylim[0] * .7 + ylim[1] * .3, ylim[0] * .3 + ylim[1] * .7])

    # Make a title, if given.
    plt.suptitle(title)
    plt.show()


# Move to a larger grid size
slm_size = (1080, 1920)
# phase = np.zeros(slm_size)

# xlist = np.arange(100, 1000, 100)  # Adjusted range
# ylist = np.arange(100, 1000, 100)  # Adjusted range   
# xgrid, ygrid = np.meshgrid(xlist, ylist)
# square = np.vstack((xgrid.ravel(), ygrid.ravel())) 
# hologram = SpotHologram(shape=(1080, 1920), spot_vectors=square, basis='knm')

# slm = ThorSLM(serialNumber='00429430')
# ThorCam.info(verbose=True)
# cam = ThorCam(serial="02C5V", verbose=True)
# fs = FourierSLM(slm)
# Move to a larger grid size
# Instead of picking a few points, make a rectangular grid in the knm basis
hologram = SpotHologram.make_rectangular_array(
    slm_size,
    array_shape=(10,10),
    array_pitch=(100,100),
    basis='knm'
)

hologram.optimize('WGS-Kim', feedback='computational_spot', stat_groups=['computational_spot'], maxiter=30)
hologram.plot_farfield(title="After Optimization")
slm = ThorSLM(serialNumber='00429430') 
# slm.write(hologram.extract_phase())     
phase = hologram.extract_phase()        # Write hologram.

# print(xlist)

# xgrid, ygrid = np.meshgrid(xlist, xlist)
# square = np.vstack((xgrid.ravel(), ygrid.ravel()))                  # Make an array of points in a grid

# plt.scatter(square[0,:], square[1,:])                               # Plot the points
# plt.xlim([0, fs.cam.shape[1]]); plt.ylim([fs.cam.shape[0], 0])
# plt.show(block=False)

# hologram = SpotHologram(shape=(1920, 1080), spot_vectors=square, basis='ij', cameraslm=fs)
# hologram.optimize('WGS-Kim', feedback='computational_spot', stat_groups=['computational_spot'], maxiter=50)
# hologram.plot_farfield(limits=[[800, 1200], [800, 1200]], title="After Optimization")

# fs.slm.write(hologram.extract_phase(), settle=True)             # Write hologram.
# fs.cam.set_exposure(.001)
# fs.cam.flush()
# img = fs.cam.get_image()                                        # Grab image.

# plt.figure(figsize=(24,24));    plt.imshow(img, vmax=50)        # This is saturated for visibility.
# plt.xlabel("Image $x$ [pix]");  plt.ylabel("Image $y$ [pix]")
# plt.show()


from slmsuite.holography import toolbox

vector = (.002, .002)                                       # Radians (== normalized units kx/k).
blaze_phase = toolbox.phase.blaze(grid=slm, vector=vector)  # Phase in units of 2pi.
plot_phase(phase, title="blaze at {}".format(vector))


# array_phase = hologram.extract_farfield()
# plot_phase(array_phase, title="array Phase")
