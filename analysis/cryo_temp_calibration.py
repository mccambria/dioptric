import errno
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import csv
import matplotlib.patches as patches
import matplotlib.lines as mlines
from scipy.optimize import curve_fit
import pandas as pd
import utils.tool_belt as tool_belt
import utils.common as common
from scipy.odr import ODR, Model, RealData
import sys
from pathlib import Path
import math
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.gridspec import GridSpec
import csv

# fmt: off
sample_readings = [5.15, 6.00, 7.00, 8.00, 9.00, 10.00, 11.00, 12.00, 13.00, 14.00, 15.00, 16.00, 17.00, 18.00, 19.00, 20.00, 21.50, 23.00, 24.50, 26.00, 27.50, 29.00, 31.00, 33.00, 35.00, 37.00, 39.00, 42.00, 45.00, 48.00, 52.00, 56.00, 60.00, 65.00, 70.00, 80.00, 90.00, 100.00, 110.00, 120.00, 130.00, 140.00, 150.00, 160.00, 170.00, 180.00, 190.00, 200.00, 210.00, 220.00, 230.00, 240.00, 250.00, 260.00, 270.00, 280.00, 290.00, 300.00]
calibr_readings = [8.75, 9.11, 9.67, 10.29, 10.99, 11.73, 12.50, 13.43, 14.30, 15.17, 16.04, 16.97, 17.88, 18.78, 19.70, 20.67, 22.10, 23.53, 24.97, 26.32, 27.84, 29.30, 31.19, 33.17, 35.16, 37.09, 39.06, 42.10, 45.01, 47.97, 52.00, 55.90, 59.89, 64.77, 69.69, 79.70, 89.44, 99.35, 109.14, 119.13, 129.00, 138.78, 148.40, 158.55, 168.12, 177.78, 187.30, 196.90, 205.93, 216.35, 226.44, 235.91, 244.72, 253.92, 263.92, 273.62, 284.00, 293.51 ]
# fmt: on
num_readings = len(sample_readings)


def calibrate(val):
    """Calibrate an arbitrary value by linear interpolation"""
    if val >= 295:
        return None
    for ind in range(num_readings - 1):
        cur_sample_reading = sample_readings[ind]
        next_sample_reading = sample_readings[ind + 1]

        if (val >= cur_sample_reading) and (val < next_sample_reading):

            diff = val - cur_sample_reading
            extent = next_sample_reading - cur_sample_reading
            frac_diff = diff / extent

            cur_calibr_reading = calibr_readings[ind]
            next_calibr_reading = calibr_readings[ind + 1]
            calibr_extent = next_calibr_reading - cur_calibr_reading
            ret_val = cur_calibr_reading + (frac_diff * calibr_extent)
            return ret_val


def main():
    # fmt: off
    nominal_temps = [295, 275, 200, 262.5, 225, 212.5, 237.5, 287.5, 300, 250, 150, 85, 175, 125, 295, 350, 400, 350, 487.5, 425, 412.5, 450, 437.5, 325, 312.5, 475, 462.5, 400, 387.5, 375, 362.5, 337.5, 327.5, 160, 187.5, 162.5, 50, 5.5, 5.5, 295, 250, 200, 100, 150, 5.5, 50, 350, 450, 400, 485, 485, 485, 450, 350, 400, 295, 295]
    # fmt: on
    for val in nominal_temps:
        print(calibrate(val))


if __name__ == "__main__":

    print(calibrate(237.5))
    print(calibrate(187.5))

    # main()
