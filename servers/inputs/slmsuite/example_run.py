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

# Move to a larger grid size
slm_size = (1920, 1080)

# Instead of picking a few points, make a rectangular grid in the knm basis
array_holo = SpotHologram.make_rectangular_array(
    slm_size,
    array_shape=(10,20),
    array_pitch=(10,10),
    basis='knm'
)
zoom = array_holo.plot_farfield(source=array_holo.target, title='Initialized Nearfield')