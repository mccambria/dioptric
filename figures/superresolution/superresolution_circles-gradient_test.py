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
        # integrand += image[sample_x, sample_y] ** 2
        integrand += image[sample_x, sample_y]

    # Ignore points that don't have enough valid samples
    # if num_valid_samples < num_circle_samples // 2:
    #     cost = 0
    # else:
    #     cost = integrand / num_valid_samples
    # cost = np.sqrt(integrand / num_valid_samples)
    cost = integrand / num_valid_samples
    return cost


def sigmoid_quotient(laplacian, gradient):

    # Get the zeros from the gradient so we can avoid divide by zeros.
    # At the end we'll just set the sigmoid to the sign of the Laplacian
    # for these values.
    gradient_zeros = gradient < 1e-10
    gradient_not_zeros = np.logical_not(gradient_zeros)
    masked_gradient = (gradient * gradient_not_zeros) + gradient_zeros
    quotient = laplacian / masked_gradient
    sigmoid = 1 / (1 + np.exp(-1 * quotient))
    laplacian_positive = np.sign(laplacian) == 1
    sigmoid = (sigmoid * gradient_not_zeros) + (
        laplacian_positive * gradient_zeros
    )
    return sigmoid


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
    gaussian_size = 5
    blur_image = cv.GaussianBlur(image, (gaussian_size, gaussian_size), 0)
    # blur_image = cv.GaussianBlur(blur_image, (gaussian_size, gaussian_size), 0)
    # blur_image = cv.GaussianBlur(blur_image, (gaussian_size, gaussian_size), 0)

    gradient_root = blur_image
    # gradient_root = image

    laplacian_image = cv.Laplacian(
        gradient_root, cv.CV_64F, ksize=gaussian_size
    )

    sobel_x = cv.Sobel(gradient_root, cv.CV_64F, 1, 0, ksize=gaussian_size)
    sobel_y = cv.Sobel(gradient_root, cv.CV_64F, 0, 1, ksize=gaussian_size)
    gradient_image = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    sigmoid_image = sigmoid_quotient(laplacian_image, gradient_image)
    # sigmoid_image = cv.GaussianBlur(
    #     sigmoid_image, (gaussian_size, gaussian_size), 0
    # )

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
    # opti_image = gradient_image
    opti_image = sigmoid_image

    # Plot the image
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    img = ax.imshow(image, cmap="inferno")
    # img = ax.imshow(thresh_image, cmap="inferno")
    # img = ax.imshow(blur_image, cmap="inferno")
    # img = ax.imshow(laplacian_image, cmap="inferno")
    # img = ax.imshow(gradient_image, cmap="inferno")
    # img = ax.imshow(sigmoid_image, cmap="inferno")
    # img = ax.imshow(opti_image, cmap="inferno")
    _ = plt.colorbar(img)
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
            # circle_b = [40, 44, 27.75]
            circle_b = [40, 44, 25]
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
