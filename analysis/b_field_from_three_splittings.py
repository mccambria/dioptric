# -*- coding: utf-8 -*-
"""
Extract the magnitude and orientation of a b field from three 
NV splittings with different orientations

Created on November 29th, 2023

@author: mccambria
"""


import os
import sys
import time
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from majorroutines.widefield import optimize
from utils import common
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils import widefield as widefield
from utils.constants import NVSpinState, VirtualLaser
from utils.positioning import get_scan_1d as calculate_freqs

degrees_1095 = 109.5 * (2 * np.pi / 360)
degrees_120 = 120 * (2 * np.pi / 360)


### Splittings here

splittings = [59, 0, 63]  # MHz

### Calculate

nv_orientations = [
    (
        0,
        0,
        1,
    ),
    (
        np.sin(degrees_1095),
        0,
        np.cos(degrees_1095),
    ),
    (
        np.cos(degrees_120) * np.sin(degrees_1095),
        np.sin(degrees_120) * np.sin(degrees_1095),
        np.cos(degrees_1095),
    ),
]
nv_field_projections = [el / 5.6 for el in splittings]
b_z = nv_field_projections[0]
orient = nv_orientations[1]
b_x = (nv_field_projections[1] - (orient[2] * b_z)) / orient[0]
orient = nv_orientations[2]
b_y = (nv_field_projections[2] - orient[0] * b_x - orient[2] * b_z) / orient[1]
print(np.sqrt(b_x**2 + b_y**2 + b_z**2))
