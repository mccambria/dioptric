# -*- coding: utf-8 -*-
"""
Extract hysteresis offsets from file

2021/08/03 mccambria
"""


from os import listdir
from os.path import isfile, join
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy
from scipy.optimize import curve_fit


def linear_line(x, m, b):
    return m * x + b


# Get the files from August 6th
path = Path(
    "/home/mccambria/E/nvdata/pc_rabi/branch_piezo-hysteresis/determine_positioning_hysteresis/2021_08/"
)
file_paths = [f for f in listdir(path) if isfile(join(path, f))]
file_paths = [f for f in file_paths if f.startswith("2021_08_06")]
file_paths = [f for f in file_paths if f.endswith("txt")]
displacement_list = []
opti_delta_list = []
for file_path in file_paths:
    with open(path / file_path) as f:
        data = json.load(f)
        try:
            displacement_list.append(data["movement_displ"])
            opti_delta_list.append(data["optimum_delta"])
        except Exception as e:
            print("skipping {}".format(file_path))
            continue

plt.ion()

opti_params, cov_arr = curve_fit(linear_line, displacement_list, opti_delta_list)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.plot(displacement_list, opti_delta_list, "bo")

lin_x_vals = numpy.linspace(min(displacement_list), max(displacement_list), 100)
ax.plot(lin_x_vals, linear_line(lin_x_vals, *opti_params), "r-")
text = "y = {:.4f} x + {:.5f}".format(opti_params[0], opti_params[1])
props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
ax.text(
    0.5,
    0.1,
    text,
    transform=ax.transAxes,
    fontsize=12,
    verticalalignment="top",
    bbox=props,
)

ax.set_xlabel("Displacement in axis {} (V)".format(2))
ax.set_ylabel("Added adjustment to return to original position (V)")
ax.set_title("Movement in axis {}".format(2))
plt.show(block=True)
