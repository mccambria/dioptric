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


def cost4(
    params,
    image,
    two_point_cost,
    noise_floor,
    image_domain_x,
    image_domain_y,
    debug,
):
    """
    Faux-integrate the pixel values around the circle. Then muliply by -1 so that
    lower values are better and we can use scipy.optimize.minimize.
    By faux-integrate I mean average the values under a 1000 point, linearly spaced
    sampling of the circle.
    """

    circle_center_x, circle_center_y, circle_radius = params

    # Ignore circles outside a reasonable range
    min_center_val = 4 * noise_floor
    if image[round(circle_center_x), round(circle_center_y)] > min_center_val:
        return 10

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
        inner_val = image[inner_sample_x, inner_sample_y]
        outer_val = image[outer_sample_x, outer_sample_y]
        inner_err = two_point_cost(mid_val, inner_val)
        outer_err = two_point_cost(mid_val, outer_val)
        integrand += inner_err ** 2 + outer_err ** 2
        num_valid_samples += 1

    # if debug:
    #     print(integrand)
    cost = np.sqrt(integrand / num_valid_samples)
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

    cost_func = cost4

    # If do_minimize is True we'll run the minimization routine and
    # plot the circles returned by that. Otherwise we'll just plot
    # the passed circles.{exp(-(x-0.0)/(0.0+0.1))} for 0<x<1
    do_minimize = True
    # do_minimize = Fals{exp(-(x-0.0)/(0.0+0.1))} for 0<x<1e

    # Get the image as a 2D ndarray
    image_file_dict = tool_belt.get_raw_data(image_file_name)
    image = np.array(image_file_dict["readout_image_array"])

    # Plot the image
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    img = ax.imshow(image, cmap="inferno")
    _ = plt.colorbar(img)
    # return

    image_domain = image.shape
    image_domain_x = image_domain[0]
    image_domain_y = image_domain[1]

    # Get a noise floor, defined as the average of the 100 pixels
    # in a 10 x 10 square at the center of the image
    mid_point_x = image_domain_x // 2
    mid_point_y = image_domain_y // 2
    half_range = 5
    noise_floor = np.average(
        image[
            mid_point_x - half_range : mid_point_x + half_range,
            mid_point_y - half_range : mid_point_y + half_range,
        ]
    )

    # %% Circle finding

    # Define the two point cost function (see note) ahead of time so that
    # it doesn't have to be generated each time we calculate the full cost
    two_point_cost = lambda a, b: np.exp(-(a - b) / (b + noise_floor))
    # two_point_cost = lambda a, b: np.exp(-(a - b) / (b + noise_floor))
    args = (
        image,
        two_point_cost,
        noise_floor,
        image_domain_x,
        image_domain_y,
        False,
    )

    if do_minimize:

        # Define the bounds of the optimization
        bounds_a = [(el - brute_range, el + brute_range) for el in circle_a]
        bounds_b = [(el - brute_range, el + brute_range) for el in circle_b]
        # Assume one circle is in the left half of the image and the other
        # is in the right half
        mid_x = (bounds_a[1][1] + bounds_b[1][0]) / 2
        bounds_a[1] = (bounds_a[1][0], mid_x)
        bounds_b[1] = (mid_x, bounds_b[1][1])
        bounds_a = tuple(bounds_a)
        bounds_b = tuple(bounds_b)

        plot_circles = []

        for bounds in [bounds_a, bounds_b]:

            # Run a brute force optimization to avoid local minima. This function
            # automatically includes a fine tuning minimization at the end
            opti_circle = brute(
                cost_func,
                bounds,
                Ns=20,
                args=args,
                # finish=None
            )

            plot_circles.append(opti_circle)

    # Just use the passed circles
    else:
        plot_circles = [circle_a, circle_b]

    # region Circle plotting

    for circle in plot_circles:

        # Debug tweak
        # circle[2] -= 3

        # Report what we found
        rounded_circle = [round(el, 2) for el in circle]
        rounded_cost = round(cost_func(circle, *args), 2)
        print("{} & {} & {} & {}".format(*rounded_circle, rounded_cost))
        # print("{}, {}, {}, {}".format(*rounded_circle, rounded_cost)

        # Plot the circle
        circle_patch = Circle(
            (circle[1], circle[0]),
            circle[2],
            fill=False,
            color="w",
        )
        ax.add_patch(circle_patch)

    # endregion


# region Run the file

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

# endregion

# %%
