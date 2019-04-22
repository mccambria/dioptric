# -*- coding: utf-8 -*-
"""
This file contains functions to control the CFM. Just change the function call
in the main section at the bottom of this file and run the file.

Created on Sun Nov 25 14:00:28 2018

@author: mccambria
"""


# %% Imports

import labrad
import majorroutines.image_sample as image_sample
import majorroutines.optimize as optimize
import utils.tool_belt as tool_belt


# %% Major Routines

def do_image_sample(name, coords, x_range, y_range, num_steps):

    x_center, y_center, z_center = coords
    readout = 10 * 10**6
    apd_index = 0

    with labrad.connect() as cxn:
        image_sample.main(cxn, name, x_center, y_center, z_center,
                           x_range, y_range, num_steps, readout, apd_index)

def do_optimize(name, coords):

    x_center, y_center, z_center = coords
    apd_index = 0

    with labrad.connect() as cxn:
        optimize.main(cxn, name, x_center, y_center, z_center, apd_index)


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

    name = 'Hopper'
    
    coords = [-0.046, -0.005, 51.4]

    # 1 V => ~100 um
    # With gold nanoparticles 0.4 is good for broad field
    # 0.04 is good for a single particle

#    scan_range = 0.4
    scan_range = 0.2
    x_scan_range = scan_range
    y_scan_range = scan_range
    num_scan_steps = 60

    # %% Functions to run
    
    try:
        do_image_sample(name, coords, x_scan_range, y_scan_range, num_scan_steps)
#        do_optimize(name, coords)
    finally:
        pass
        # Kill safe stop
#        if tool_belt.check_safe_stop_alive():
#            tool_belt.poll_safe_stop()
