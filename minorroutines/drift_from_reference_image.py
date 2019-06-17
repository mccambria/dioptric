# -*- coding: utf-8 -*-
"""Determine the drift by taking an image using the same parameters as a
reference image and then comparing the reference image to the new image.

Created on Sun Jun 16 11:38:17 2019

@author: mccambria
"""


# %% Imports


import majorroutines.image_sample as image_sample
import utils.tool_belt as tool_belt


# %% Main


def main(cxn, ref_file_name, nd_filter, apd_indices):
    """When you run the file, we'll call into main, which should contain the
    body of the routine.
    """

    # %% Get the reference image
    ref_data = tool_belt.get_raw_data('image_sample', ref_file_name)

    # %% Get a new image

    # Take a new image using the same parameters
    new_data = image_sample.main(cxn, ref_data['coords'], nd_filter,
                          ref_data['x_range'], ref_data['y_range'],
                          ref_data['num_steps'], apd_indices,
                          save_data=False, plot_data=False)

    # %% Calculate the shift




# %% Run the file


# The __name__ variable will only be '__main__' if you run this file directly.
# This allows a file's functions, classes, etc to be imported without running
# the script that you set up here.
if __name__ == '__main__':

    pass
