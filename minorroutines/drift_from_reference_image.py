# -*- coding: utf-8 -*-
"""Determine the drift by taking an image using the same parameters as a
reference image and then comparing the reference image to the new image.

Created on Sun Jun 16 11:38:17 2019

@author: mccambria
"""


# %% Imports


import majorroutines.image_sample as image_sample
import utils.tool_belt_test as tool_belt
import cv2
import utils.image_processing as image_processing
from matplotlib import pyplot as plt
import numpy


# %% Main


def main(cxn, ref_file_name, nd_filter, apd_indices):
    """When you run the file, we'll call into main, which should contain the
    body of the routine.
    """

    # %% Get the reference image

    ref_data = tool_belt.get_raw_data('image_sample', ref_file_name)

    # %% Get a new image

    # Take a new image using the same parameters
    # new_data = image_sample.main(cxn, ref_data['coords'], nd_filter,
    #                       ref_data['x_range'], ref_data['y_range'],
    #                       ref_data['num_steps'], apd_indices,
    #                       save_data=False, plot_data=False)

    # Test
    new_data = tool_belt.get_raw_data('image_sample',
                                      '2019-06-14_16-36-48_ayrton12')

    # %% Calculate the shift

    # Get the image arrays
    ref_img_array = numpy.array(ref_data['img_array'])
    new_img_array = numpy.array(new_data['img_array'])

    # convert to 8 bit
    ref_img_array = image_processing.convert_to_8bit(ref_img_array)
    new_img_array = image_processing.convert_to_8bit(new_img_array)

    # Add a border to the reference image so that we can use template matching
    img_shape = ref_img_array.shape
    ver_border_size = img_shape[0] // 2
    hor_border_size = img_shape[1] // 2
    ref_img_array = cv2.copyMakeBorder(ref_img_array,
                       top=ver_border_size, bottom=ver_border_size,
                       left=hor_border_size, right=hor_border_size,
                       borderType=cv2.BORDER_CONSTANT, value=0)

    res = cv2.matchTemplate(ref_img_array, new_img_array,
                            cv2.TM_CCOEFF_NORMED)

    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(res)


# %% Run the file


# The __name__ variable will only be '__main__' if you run this file directly.
# This allows a file's functions, classes, etc to be imported without running
# the script that you set up here.
if __name__ == '__main__':

    main(None, '2019-06-14_16-28-13_ayrton12', None, None)
