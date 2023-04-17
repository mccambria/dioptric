# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 18:05:32 2023

@author: kolkowitz
"""


import utils.tool_belt as tool_belt
import utils.positioning as positioning
import utils.kplotlib as kpl
from utils.kplotlib import KplColors
from utils.kplotlib import Size
from numpy import pi
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json


file = "2022_09_08-13_02_34-rubin-nv1_2022_08_10"

file_name = file + '.txt'
with open(file_name) as f:
    data = json.load(f)
        


img_array = np.array(data["img_array"])
readout = data["readout"]
img_array_kcps = (img_array / 1000) / (readout * 1e-9)
# scan_type = data['scan_type']
x_center = 0
y_center = 0

x_range = data["x_range"]*35
y_range = data["y_range"]*35
num_steps = data["num_steps"]
ret_vals = positioning.get_scan_grid_2d(
    x_center, y_center, x_range, y_range, num_steps, num_steps
)
extent = ret_vals[-1]

kpl.init_kplotlib()
fig, ax = plt.subplots()
im = kpl.imshow(
    ax,
    img_array_kcps,
    # title=title,
    x_label="um",
    y_label="um",
    cbar_label="kcps",
    # vmax = 60,
    extent=extent,
    aspect="auto",
)
