# -*- coding: utf-8 -*-
"""
Fit circles to superresolution rings in images demonstrating resolved
images of two NVs separated by less than the diffraction limit.

Created on February 25, 2022

@author: mccambria
"""

# %% Imports


import utils.tool_belt as tool_belt
import utils.common as common
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, brute
from numpy import pi
from matplotlib.patches import Circle
import cv2 as cv


# %% Constants

num_circle_samples = 100
phi_linspace = np.linspace(0, 2 * pi, num_circle_samples)
cos_phi_linspace = np.cos(phi_linspace)
sin_phi_linspace = np.sin(phi_linspace)


# %% Functions


def cost0(params, image, debug):
    """
    Faux-integrate the pixel values around the circle. Then muliply by -1 so that
    lower values are better and we can use scipy.optimize.minimize.
    By faux-integrate I mean average the values under a 1000 point, linearly spaced
    sampling of the circle.
    """

    circle_center_x, circle_center_y, circle_radius = params

    integrand = 0
    num_valid_samples = 0
    image_domain = image.shape

    circle_samples_x = circle_center_x + circle_radius * np.cos(phi_linspace)
    circle_samples_y = circle_center_y + circle_radius * np.sin(phi_linspace)
    circle_samples_x_round = [round(el) for el in circle_samples_x]
    circle_samples_y_round = [round(el) for el in circle_samples_y]
    # plt.plot(phi_linspace, circle_samples_x_round)
    # plt.plot(phi_linspace, circle_samples_y_round)
    circle_samples = zip(circle_samples_x_round, circle_samples_y_round)

    # if debug:
    #     circle_samples_list = [el for el in circle_samples]
    #     print(circle_samples_list)
    #     circle_samples = zip(circle_samples_x_round, circle_samples_y_round)

    # We'll use this to keep track of how much of the circle is actually
    # in the domain of the image
    for sample in circle_samples:
        sample_x = sample[0]
        sample_y = sample[1]
        if (sample_x < 0) or (sample_x >= image_domain[0]):
            continue
        if (sample_y < 0) or (sample_y >= image_domain[1]):
            continue
        num_valid_samples += 1
        integrand += image[sample_x, sample_y]

    # Ignore points that don't have enough valid samples
    # if num_valid_samples < num_circle_samples // 2:
    #     cost = 0
    # else:
    #     cost = integrand / num_valid_samples
    cost = integrand / num_valid_samples
    return -1 * cost


def cost(params, image, debug):
    """
    Faux-integrate the pixel values around the circle. Then muliply by -1 so that
    lower values are better and we can use scipy.optimize.minimize.
    By faux-integrate I mean average the values under a 1000 point, linearly spaced
    sampling of the circle.
    """

    circle_center_x, circle_center_y, circle_radius = params

    # Use a double circle to account for the ring width
    half_width = 3
    radii = [circle_radius - half_width, circle_radius + half_width]
    integrand = 0
    num_valid_samples = 0
    image_domain = image.shape

    for radius in radii:

        circle_samples_x = circle_center_x + radius * np.cos(phi_linspace)
        circle_samples_y = circle_center_y + radius * np.sin(phi_linspace)
        circle_samples_x_round = [round(el) for el in circle_samples_x]
        circle_samples_y_round = [round(el) for el in circle_samples_y]
        # plt.plot(phi_linspace, circle_samples_x_round)
        # plt.plot(phi_linspace, circle_samples_y_round)
        circle_samples = zip(circle_samples_x_round, circle_samples_y_round)

        # if debug:
        #     circle_samples_list = [el for el in circle_samples]
        #     print(circle_samples_list)
        #     circle_samples = zip(circle_samples_x_round, circle_samples_y_round)

        # We'll use this to keep track of how much of the circle is actually
        # in the domain of the image
        for sample in circle_samples:
            sample_x = sample[0]
            sample_y = sample[1]
            if (sample_x < 0) or (sample_x >= image_domain[0]):
                continue
            if (sample_y < 0) or (sample_y >= image_domain[1]):
                continue
            num_valid_samples += 1
            integrand += image[sample_x, sample_y]

    # # Ignore points that don't have enough valid samples
    # if num_valid_samples < len(radii) * num_circle_samples // 2:
    #     cost = 0
    # else:
    #     cost = integrand / num_valid_samples
    cost = integrand / num_valid_samples
    return -1 * cost


def cost2(params, image, debug):

    circle_center_x, circle_center_y, circle_radius = params

    half_width = 2
    # half_width = 3
    num_diff_points = (2 * half_width) + 1
    diff_points_list = np.linspace(-half_width, half_width, num_diff_points)
    diff_points_list = [int(val) for val in diff_points_list]
    diffs = []
    for diff_x in diff_points_list:
        for diff_y in diff_points_list:
            if np.sqrt(diff_x ** 2 + diff_y ** 2) <= half_width:
                diffs.append((diff_x, diff_y))
    len_diffs = len(diffs)
    integrand = 0
    num_valid_samples = 0
    image_domain = image.shape

    circle_samples_x = circle_center_x + circle_radius * np.cos(phi_linspace)
    circle_samples_y = circle_center_y + circle_radius * np.sin(phi_linspace)
    circle_samples_x_round = [round(el) for el in circle_samples_x]
    circle_samples_y_round = [round(el) for el in circle_samples_y]
    # plt.plot(phi_linspace, circle_samples_x_round)
    # plt.plot(phi_linspace, circle_samples_y_round)
    circle_samples = zip(circle_samples_x_round, circle_samples_y_round)

    # if debug:
    #     circle_samples_list = [el for el in circle_samples]
    #     print(circle_samples_list)
    #     circle_samples = zip(circle_samples_x_round, circle_samples_y_round)

    # We'll use this to keep track of how much of the circle is actually
    # in the domain of the image
    for sample in circle_samples:

        sample_x = sample[0]
        sample_y = sample[1]

        # if (sample_x < half_width) or (
        #     sample_x >= image_domain[0] - half_width
        # ):
        #     continue
        # if (sample_y < half_width) or (
        #     sample_y >= image_domain[1] - half_width
        # ):
        #     continue
        num_valid_samples += 1

        sample_integrand = 0
        for el in diffs:
            diff_x, diff_y = el
            sample_integrand += image[sample_x + diff_x, sample_y + diff_y]
        # if sample_integrand / len_diffs > 1:
        # if sample_integrand / len_diffs > 0.7:
        #     integrand += 1
        integrand += sample_integrand

    # Ignore points that don't have enough valid samples
    # if num_valid_samples < num_circle_samples // 2:
    #     cost = 0
    # else:
    #     cost = integrand / num_valid_samples
    cost = integrand / num_valid_samples
    return -1 * cost


def cost3(params, image, debug):

    circle_center_x, circle_center_y, circle_radius = params

    # Ignore circles outside a reasonable range
    if image[round(circle_center_x), round(circle_center_y)] > 0.4:
        return 0

    # half_width = 2
    half_width = 4
    num_diff_points = (2 * half_width) + 1
    diff_points_list = np.linspace(-half_width, half_width, num_diff_points)
    diff_points_list = [int(val) for val in diff_points_list]
    diffs = []
    for diff_x in diff_points_list:
        for diff_y in diff_points_list:
            if np.sqrt(diff_x ** 2 + diff_y ** 2) <= half_width:
                diffs.append((diff_x, diff_y))
    len_diffs = len(diffs)
    integrand = 0
    num_valid_samples = 0
    image_domain = image.shape

    circle_samples_x = circle_center_x + circle_radius * np.cos(phi_linspace)
    circle_samples_y = circle_center_y + circle_radius * np.sin(phi_linspace)
    circle_samples_x_round = [round(el) for el in circle_samples_x]
    circle_samples_y_round = [round(el) for el in circle_samples_y]
    # plt.plot(phi_linspace, circle_samples_x_round)
    # plt.plot(phi_linspace, circle_samples_y_round)
    circle_samples = zip(circle_samples_x_round, circle_samples_y_round)

    # if debug:
    #     circle_samples_list = [el for el in circle_samples]
    #     print(circle_samples_list)
    #     circle_samples = zip(circle_samples_x_round, circle_samples_y_round)

    # We'll use this to keep track of how much of the circle is actually
    # in the domain of the image
    for sample in circle_samples:

        sample_x = sample[0]
        sample_y = sample[1]

        # if (sample_x < half_width) or (
        #     sample_x >= image_domain[0] - half_width
        # ):
        #     continue
        # if (sample_y < half_width) or (
        #     sample_y >= image_domain[1] - half_width
        # ):
        #     continue
        num_valid_samples += 1

        center_val = image[sample_x, sample_y]
        # diff_vals = []
        # for el in diffs:
        #     diff_x, diff_y = el
        #     im_val = image[sample_x + diff_x, sample_y + diff_y]
        #     diff_vals.append(im_val)
        # integrand += center_val / min(diff_vals)
        min_nonzero = None
        for el in diffs:
            diff_x, diff_y = el
            im_val = image[sample_x + diff_x, sample_y + diff_y]
            if im_val == 0:
                continue
            if (min_nonzero is None) or (im_val < min_nonzero):
                min_nonzero = im_val
        integrand += center_val / min_nonzero

    # Ignore points that don't have enough valid samples
    # if num_valid_samples < num_circle_samples // 2:
    #     cost = 0
    # else:
    #     cost = integrand / num_valid_samples
    cost = integrand / num_valid_samples
    return -1 * cost


def cost4(params, image, debug):
    """
    Faux-integrate the pixel values around the circle. Then muliply by -1 so that
    lower values are better and we can use scipy.optimize.minimize.
    By faux-integrate I mean average the values under a 1000 point, linearly spaced
    sampling of the circle.
    """

    circle_center_x, circle_center_y, circle_radius = params

    # Ignore circles outside a reasonable range
    # min_center_val = 0.4
    # min_center_val = 15
    # if image[round(circle_center_x), round(circle_center_y)] > min_center_val:
    #     return 0

    image_domain = image.shape
    image_domain_x = image_domain[0]
    image_domain_y = image_domain[1]

    mid_point_x = image_domain_x // 2
    mid_point_y = image_domain_y // 2
    sample_range = 5
    noise_floor = np.average(
        image[
            mid_point_x - sample_range : mid_point_x + sample_range,
            mid_point_y - sample_range : mid_point_y + sample_range,
        ]
    )

    half_width = 3
    inner_radius = circle_radius - half_width
    outer_radius = circle_radius + half_width
    integrand = 0
    num_valid_samples = 0

    phi_samples = zip(cos_phi_linspace, sin_phi_linspace)

    for phi_sample in phi_samples:

        cos_phi = phi_sample[0]
        sin_phi = phi_sample[1]

        mid_sample_x = round(circle_center_x + circle_radius * cos_phi)
        mid_sample_y = round(circle_center_y + circle_radius * sin_phi)
        inner_sample_x = round(circle_center_x + inner_radius * cos_phi)
        inner_sample_y = round(circle_center_y + inner_radius * sin_phi)
        outer_sample_x = round(circle_center_x + outer_radius * cos_phi)
        outer_sample_y = round(circle_center_y + outer_radius * sin_phi)

        if (outer_sample_x < 0) or (outer_sample_x >= image_domain_x):
            continue
        if (outer_sample_y < 0) or (outer_sample_y >= image_domain_y):
            continue

        mid_val = image[mid_sample_x, mid_sample_y]
        if mid_val == 0:
            continue
        inner_val = image[inner_sample_x, inner_sample_y]
        outer_val = image[outer_sample_x, outer_sample_y]
        # inner_outer_avg = (inner_val + outer_val) / 2
        # if inner_outer_avg < 0.1:
        #     inner_outer_avg = 0.1
        # ratio = mid_val / inner_outer_avg
        # integrand += ratio
        # integrand += (mid_val - inner_outer_avg) / (inner_outer_avg + 0.1)
        # inner_diff = (mid_val - inner_val) / (inner_val + 0.1)
        # outer_diff = (mid_val - outer_val) / (outer_val + 0.1)
        # integrand += np.sqrt(inner_diff ** 2 + outer_diff ** 2)
        # inner_err = (inner_val + 0.1) / (mid_val - inner_val)
        # outer_err = (outer_val + 0.1) / (mid_val - outer_val)
        inner_err = np.exp(-(mid_val - inner_val) / (inner_val + noise_floor))
        outer_err = np.exp(-(mid_val - outer_val) / (outer_val + noise_floor))
        integrand += np.sqrt(inner_err ** 2 + outer_err ** 2)
        num_valid_samples += 1

    # if debug:
    #     print(integrand)
    cost = integrand / num_valid_samples
    # return -1 * cost
    return cost


# x, y >= 0, finite

# df/dx > 0
# df/dy < 0
# over domain, f is finite
# f should capture value of x relative to y rather than either value absolutely

# d g / df < 0
# g >= 0
# for finite f, g(f) is finite

# %% Main


def main(image_file_name, circle_a, circle_b, brute_range):

    # %% Setup

    cost_func = cost0
    optimize = True
    # optimize = False

    # Get the image as a 2D ndarray
    image_file_dict = tool_belt.get_raw_data(image_file_name)
    image = np.array(image_file_dict["readout_image_array"])

    # %% Processing

    # Convert to 8 bit
    eight_bit_image = np.copy(image)
    eight_bit_image -= eight_bit_image.min()
    eight_bit_image *= 256 / eight_bit_image.max()
    eight_bit_image = eight_bit_image.astype(np.uint8)

    # Blur
    gaussian_size = 7
    blur_image = cv.GaussianBlur(
        eight_bit_image, (gaussian_size, gaussian_size), 0
    )
    blur_image = cv.GaussianBlur(blur_image, (gaussian_size, gaussian_size), 0)
    blur_image = cv.GaussianBlur(blur_image, (gaussian_size, gaussian_size), 0)

    gradient_image = cv.Laplacian(blur_image, cv.CV_64F, ksize=gaussian_size)
    # gradient_image = cv.Laplacian(gradient_image, cv.CV_64F, ksize=5)

    # Threshold
    # thresh_image = np.array(image > 0.7, dtype=int)
    # thresh_image = cv.adaptiveThreshold(
    #     blur_image,
    #     255,
    #     cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    #     cv.THRESH_BINARY,
    #     55,
    #     -2,
    # )

    # Set which image to optimize on
    # opti_image = image
    # opti_image = thresh_image
    # opti_image = blur_image
    opti_image = gradient_image

    # Plot the image
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    img = ax.imshow(opti_image, cmap="inferno")
    _ = plt.colorbar(img)
    return

    # %% Circle finding + plotting

    # circles = cv.HoughCircles(
    #     blur_image,
    #     cv.HOUGH_GRADIENT,
    #     1,
    #     3,
    #     param1=50,
    #     param2=30,
    #     minRadius=20,
    #     maxRadius=35,
    # )
    # circles = np.uint16(np.around(circles))
    # for circle in circles[0, :]:
    #     # Plot the circle
    #     circle_patch = Circle(
    #         (circle[1], circle[0]),
    #         circle[2],
    #         fill=False,
    #         color="w",
    #     )
    #     ax.add_patch(circle_patch)

    # return

    if optimize:

        bounds_a = [(el - brute_range, el + brute_range) for el in circle_a]
        bounds_b = [(el - brute_range, el + brute_range) for el in circle_b]
        mid_x = (bounds_a[1][1] + bounds_b[1][0]) / 2
        bounds_a[1] = (bounds_a[1][0], mid_x)
        bounds_b[1] = (mid_x, bounds_b[1][1])
        bounds_a = tuple(bounds_a)
        bounds_b = tuple(bounds_b)

        for bounds in [bounds_a, bounds_b]:

            # Start with a brute force optimization, then we'll do some fine tuning
            opti_circle = brute(
                cost_func,
                bounds,
                Ns=20,
                args=(opti_image, False),  # finish=None
            )
            # res = minimize(
            #     cost_func, x0=opti_circle, args=(opti_image, False), bounds=bounds
            # )
            # opti_circle = res.x

            print(opti_circle)
            print(cost_func(opti_circle, opti_image, True))

            # Plot the circle
            circle_patch = Circle(
                (opti_circle[1], opti_circle[0]),
                opti_circle[2],
                fill=False,
                color="w",
            )
            ax.add_patch(circle_patch)

    else:

        # Fig. 3
        # circle_a = [41.5, 36.5, 27.75]
        # circle_b = [40, 44, 27.75]
        # Fig. 4
        # circle_a = [51, 46.5, 26]
        # circle_b = [51.5, 56, 27.5]
        for opti_circle in [circle_a, circle_b]:
            # opti_circle[2] -= 3
            print(opti_circle)
            print(cost_func(opti_circle, opti_image, True))

            # Plot the circle
            circle_patch = Circle(
                (opti_circle[1], opti_circle[0]),
                opti_circle[2],
                fill=False,
                color="w",
            )
            ax.add_patch(circle_patch)


# %% Run the file


if __name__ == "__main__":

    tool_belt.init_matplotlib()

    for circle in [3, 4]:

        # Fig. 3
        if circle == 3:
            image_file_name = "2021_09_30-13_18_47-johnson-dnv7_2021_09_23"
            # Best circles by hand
            circle_a = [41.5, 37, 27.5]
            circle_b = [40, 44, 27.75]
            brute_range = 3

        # Fig. 4
        elif circle == 4:
            image_file_name = "2021_10_17-19_02_22-johnson-dnv5_2021_09_23"
            # Best circles by hand
            circle_a = [50, 46, 26]
            circle_b = [51.7, 56.5, 27.3]
            brute_range = 3

        main(image_file_name, circle_a, circle_b, brute_range)

    plt.show(block=True)
