# -*- coding: utf-8 -*-
"""
This file contains functions to control the CFM. Just change the function call
in the main section at the bottom of this file and run the file.

Created on Sun Nov 25 14:00:28 2018

@author: mccambria
"""


# %% Imports


import majorroutines.scan_sample as scan_sample


# %% Major Routines

def do_scan_sample(name, x_center, y_center, z_center,
                   x_range, y_range, num_steps):

    readout = 10 * 10**6
    apd_index = 0

    scan_sample.main(name, x_center, y_center, z_center,
                     x_range, y_range, num_steps, readout, apd_index)


# %% Script Code


# Functions only run when called. Since this part of the script is not in a
# function, it will run when the script is run.
# __name__ will only be __main__ if we're running the file as a program.
# The below pattern enables us to import this file as a module without
# running it as a program.
if __name__ == '__main__':

    # %% Shared parameters
    # The file has minimal documentation.
    # For more, view the function definitions in their respective file.

    name = 'Ayrton9'

    x_center = 0.0
    y_center = 0.0
    z_center = 50.0

    # 1 V => ~100 um
    # With gold nanoparticles 0.4 is good for broad field
    # 0.04 is good for a single particle

    scan_range = 1.5
    x_scan_range = scan_range
    y_scan_range = scan_range
    num_scan_steps = 100

    # %% Functions to run

    do_scan_sample(name, x_center, y_center, z_center,
                   x_scan_range, y_scan_range, num_scan_steps)
