# -*- coding: utf-8 -*-
"""
Optimize on an NV by taking an image and comparing it to a reference image

Created on April 30th, 2022

@author: mccambria
"""


# %% Imports


import utils.tool_belt as tool_belt
import labrad
import majorroutines.image_sample as image_sample
import numpy as np
import cv2
import utils.image_processing as image_processing


# %% Functions


def find_offset(new_img_data, ref_image_data):

    ref_x_voltages = ref_image_data["x_voltages"]
    ref_y_voltages = ref_image_data["y_voltages"]
    ref_num_steps = ref_image_data["num_steps"]
    ref_img_array = np.array(ref_image_data["img_array"])

    # Convert to 8 bit
    ref_img_array = image_processing.convert_to_8bit(ref_img_array)
    new_img_array = image_processing.convert_to_8bit(new_img_array)

    # Add a border to the reference image so that we can use template matching
    img_shape = ref_img_array.shape
    ver_border_size = img_shape[0] // 2
    hor_border_size = img_shape[1] // 2
    ref_img_array = cv2.copyMakeBorder(
        ref_img_array,
        top=ver_border_size,
        bottom=ver_border_size,
        left=hor_border_size,
        right=hor_border_size,
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )

    res = cv2.matchTemplate(ref_img_array, new_img_array, cv2.TM_CCOEFF_NORMED)

    # Find where we best matched to determine the shift
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    print(max_val)

    # Adjust for the borders
    x_shift_pixels = max_loc[0] - hor_border_size
    y_shift_pixels = max_loc[1] - ver_border_size

    # Determine the pixel size in volts / pixel
    x_pixel_size = (ref_x_voltages[-1] - ref_x_voltages[0]) / ref_num_steps
    y_pixel_size = (ref_y_voltages[-1] - ref_y_voltages[0]) / ref_num_steps

    # Convert to volts
    x_shift_volts = x_shift_pixels * x_pixel_size
    y_shift_volts = y_shift_pixels * y_pixel_size

    return x_shift_volts, y_shift_volts
    # return x_shift_pixels, y_shift_pixels


# %% Main


def main(
    nv_sig,
    apd_indices,
    set_to_opti_coords=True,
    save_data=False,
    plot_data=False,
):

    with labrad.connect() as cxn:
        main_with_cxn(
            cxn, nv_sig, apd_indices, set_to_opti_coords, save_data, plot_data
        )


def main_with_cxn(
    cxn,
    nv_sig,
    apd_indices,
    set_to_opti_coords=True,
    save_data=False,
    plot_data=False,
    set_drift=True,
):

    # Make sure there's a reference image
    if "reference_image" in nv_sig:
        ref_image_file = nv_sig["reference_image"]
    else:
        return

    tool_belt.reset_cfm(cxn)

    # Get what we need from the reference image file
    ref_image_data = tool_belt.get_raw_data(ref_image_file)
    x_range = ref_image_data["x_range"]
    y_range = ref_image_data["y_range"]
    ref_num_steps = ref_image_data["num_steps"]

    new_img_array, _, _ = image_sample.main(
        nv_sig, x_range, y_range, ref_num_steps, apd_indices
    )

    x_shift, y_shift = find_offset(new_img_array, ref_image_data)

    # Make sure we have floats and set the drift
    drift = tool_belt.get_drift()
    tool_belt.set_drift([float(x_shift), float(y_shift), float(drift[2])])


if __name__ == "__main__":
    
    new_image_file = "2022_04_30-08_09_42-wu-nv6_2022_04_14"
    ref_image_file = "2022_04_29-22_20_49-wu-nv6_2022_04_14"
    # new_image_file = ref_image_file

    new_image_data = tool_belt.get_raw_data(new_image_file)
    new_img_array = new_image_data["img_array"]
    new_img_array = np.array(new_img_array)
    ref_image_data = tool_belt.get_raw_data(ref_image_file)
    x_shift_volts, y_shift_volts = find_offset(new_img_array, ref_image_data)

    print(x_shift_volts, y_shift_volts)
