# -*- coding: utf-8 -*-
"""
Extract the magnitude and orientation of a b field from three
NV splittings with different orientations

Created on November 29th, 2023

@author: mccambria
"""

import numpy as np

degrees_1095 = 109.5 * (2 * np.pi / 360)
degrees_120 = 120 * (2 * np.pi / 360)


### Splittings here

splittings = [214, 162, 85, 50]  # new spliting
splittings = [splittings[jnd] for jnd in range(4) if jnd != 3]
# splittings = [59, 0, 63]  # MHz

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
