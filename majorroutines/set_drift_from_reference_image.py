# -*- coding: utf-8 -*-
"""Determine the drift by taking an image using the same parameters as a
reference image and then comparing the reference image to the new image.
To calculate the drift we use opencv's template matching. Template matching
assumes your template is smaller than the image you're searching through so we
pad the reference image to double its size in each dimension. I think this
could bias the values to small shifts, but I'm not sure this actually happens
with the matching method we use (cv2.TM_CCOEFF_NORMED). I need to think more
about the formula for that method. In my tests at least, this works
consistently.

This needs some work to support non-square images yet.

Created on Sun Jun 16 11:38:17 2019

@author: mccambria
"""

import cv2
import labrad
import numpy
from matplotlib import pyplot as plt

import majorroutines.image_sample as image_sample
import majorroutines.targeting as targeting
import utils.image_processing as image_processing
import utils.tool_belt as tool_belt

# region Functions


def get_drift_from_file_images(ref_image, new_image):
    ref_data = tool_belt.get_raw_data(ref_image)
    new_data = tool_belt.get_raw_data(new_image)

    ref_img_array = numpy.array(ref_data["img_array"])
    new_img_array = numpy.array(new_data["img_array"])

    return calc_offset(ref_img_array, new_img_array)


def calc_offset(ref_img_array, new_img_array):
    # Convert to 8 bit
    ref_img_array = image_processing.convert_to_8bit(ref_img_array)
    new_img_array = image_processing.convert_to_8bit(new_img_array)

    # Add a border to the reference image so that we can use template matching
    # img_shape = ref_img_array.shape
    # ver_border = img_shape[0] // 2
    # hor_border = img_shape[1] // 2
    # ref_img_array = cv2.copyMakeBorder(ref_img_array,
    #                    top=ver_border, bottom=ver_border,
    #                    left=hor_border, right=hor_border,
    #                    borderType=cv2.BORDER_CONSTANT, value=128)

    res = cv2.matchTemplate(ref_img_array, new_img_array, cv2.TM_CCOEFF_NORMED)

    # Find where we best matched to determine the shift
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    # borders = (ver_border, hor_border)
    # return max_val, max_loc, borders
    return max_val, max_loc


# endregion


# region Main


def main(ref_file_name, apd_indices):
    with labrad.connect() as cxn:
        main_with_cxn(cxn, ref_file_name, apd_indices)


def main_with_cxn(cxn, ref_file_name, apd_indices):
    ### Get the reference and new images

    ref_data = tool_belt.get_raw_data(ref_file_name)

    # Take a new image at half the range and the same pixel size
    nv_sig = ref_data["nv_sig"]
    img_range = ref_data["x_range"]  # Assume square image and pixels
    num_steps = ref_data["num_steps"]
    pixel_size = img_range / num_steps
    num_steps = num_steps // 2
    x_range = pixel_size * num_steps
    y_range = x_range
    new_data = image_sample.main_with_cxn(
        cxn, nv_sig, x_range, y_range, num_steps, apd_indices
    )

    ### Calculate the correlations

    ref_img_array = numpy.array(ref_data["img_array"])
    new_img_array = numpy.array(new_data["img_array"])
    max_val, max_loc, borders = calc_offset(ref_img_array, new_img_array)

    ### If a correlation was good enough, set the shift accordingly

    success = max_val > 0.8
    if success:
        # Adjust for the borders
        # ver_border, hor_border = borders
        # x_shift_pixels = max_loc[0] - hor_border
        # y_shift_pixels = max_loc[1] - ver_border
        x_shift_pixels = max_loc[0]
        y_shift_pixels = max_loc[1]

        # Determine the pixel size in volts / pixel
        ref_x_voltages = ref_data["x_voltages"]
        x_pixel_size = (ref_x_voltages[-1] - ref_x_voltages[0]) / num_steps
        ref_y_voltages = ref_data["y_voltages"]
        y_pixel_size = (ref_y_voltages[-1] - ref_y_voltages[0]) / num_steps

        # Convert to volts
        x_shift_volts = x_shift_pixels * x_pixel_size
        y_shift_volts = y_shift_pixels * y_pixel_size

        # Make sure we have floats and set the drift
        current_drift = tool_belt.get_drift(cxn)
        tool_belt.set_drift(
            [float(x_shift_volts), float(y_shift_volts), float(current_drift[2])]
        )
    return success

    ### Optimize to set drift more precisely

    # optimize.main(cxn, nv_sig, nd_filter, apd_indices)


# endregion

if __name__ == "__main__":
    # New image should be smaller than and have same pixel size as ref image
    ref_image = "2022_01_01-09_57_45-wu-nv6_2021_12_25"
    # new_image = "2022_01_01-09_57_45-wu-nv6_2021_12_25"
    new_image = "2022_01_12-16_59_20-wu-nv6_2021_12_25"

    max_val, max_loc, _ = get_drift_from_file_images(ref_image, new_image)

    print(max_val)
    print(max_loc)
