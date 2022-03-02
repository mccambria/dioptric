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
phi_linspace = np.linspace(0, 2 * pi, num_circle_samples, endpoint=False)
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

    image_domain = image.shape
    x_lim = image_domain[1]
    y_lim = image_domain[0]

    circle_samples_x = circle_center_x + circle_radius * cos_phi_linspace
    circle_samples_y = circle_center_y + circle_radius * sin_phi_linspace
    circle_samples_x_round = [round(el) for el in circle_samples_x]
    circle_samples_y_round = [round(el) for el in circle_samples_y]
    circle_samples = zip(circle_samples_x_round, circle_samples_y_round)

    # circle_samples = np.column_stack(
    #     (circle_samples_x_round, circle_samples_y_round)
    # )

    check_valid = (
        lambda el: (el[0] >= 0)
        and (el[0] < x_lim)
        and (el[1] >= 0)
        and (el[1] < y_lim)
    )
    # valid_samples = check_valid(circle_samples)
    # valid_samples = np.where()
    # integrand_vals = np.take(image, circle_samples)
    # integrand_vals *= valid_samples
    # valid_samples = [el for el in circle_samples if check_valid(el)]
    # num_valid_samples = len(valid_samples)

    integrand_vals = [image[el] for el in circle_samples if check_valid(el)]

    cost = np.sum(integrand_vals) / len(integrand_vals)
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
    # minimize_type = "manual"
    # minimize_type = "auto"
    minimize_type = "recursive"
    # minimize_type = "none"

    # Get the image as a 2D ndarray
    image_file_dict = tool_belt.get_raw_data(image_file_name)
    image = np.array(image_file_dict["readout_image_array"])

    image_domain = image.shape
    image_len_x = image_domain[1]
    image_len_y = image_domain[0]

    # %% Processing

    # Blur
    gaussian_size = 7
    blur_image = cv.GaussianBlur(image, (gaussian_size, gaussian_size), 0)

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

    # Set which image to optimize on
    # opti_image = image
    # opti_image = blur_image
    # opti_image = gradient_image
    opti_image = sigmoid_image

    # Plot the image
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    img = ax.imshow(image, cmap="inferno")
    # img = ax.imshow(blur_image, cmap="inferno")
    # img = ax.imshow(gradient_image, cmap="inferno")
    # img = ax.imshow(laplacian_image, cmap="inferno")
    # img = ax.imshow(sigmoid_image, cmap="inferno")
    # img = ax.imshow(opti_image, cmap="inferno")
    _ = plt.colorbar(img)
    # return

    # %% Circle finding

    args = (opti_image, False)
    plot_circles = []

    if minimize_type == "manual":

        x_linspace = np.linspace(0, image_len_x, image_len_x, endpoint=False)
        y_linspace = np.linspace(0, image_len_x, image_len_y, endpoint=False)
        rad_linspace = np.linspace(20, 35, 16)

        left_best_circle = None
        left_best_cost = 1
        right_best_circle = None
        right_best_cost = 1
        half_x = image_len_x / 2

        # Manual brute force optimization for left/right halves
        for x in x_linspace:
            for y in y_linspace:
                for r in rad_linspace:
                    circle = [y, x, r]
                    cost = cost_func(circle, *args)
                    if x < half_x:
                        if cost < left_best_cost:
                            left_best_circle = circle
                            left_best_cost = cost
                    else:
                        if cost < right_best_cost:
                            right_best_circle = circle
                            right_best_cost = cost

        brute_circles = [left_best_circle, right_best_circle]
        for circle in brute_circles:
            print(circle)
            bounds = [(val - 1, val + 1) for val in circle]
            res = minimize(
                cost_func, circle, bounds=bounds, args=args, method="L-BFGS-B"
            )
            opti_circle = res.x
            plot_circles.append(opti_circle)

    elif minimize_type == "auto":

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

    elif minimize_type == "recursive":

        # Define the bounds of the optimization
        bounds_a = [
            ((1 / 4) * image_len_y, (3 / 4) * image_len_y),
            (0, image_len_x / 2),
            (20, 35),
        ]
        bounds_b = [
            ((1 / 4) * image_len_y, (3 / 4) * image_len_y),
            (image_len_x / 2, image_len_x),
            (20, 35),
        ]

        for bounds in [bounds_a, bounds_b]:

            best_cost = 1
            while True:

                opti_circle = brute(
                    cost_func, bounds, Ns=20, args=args, finish=None
                )
                new_best_cost = cost_func(opti_circle, *args)

                threshold = 0.0001 * best_cost
                if best_cost - new_best_cost < threshold:
                    break
                best_cost = new_best_cost

                bounds_span = [el[1] - el[0] for el in bounds]
                half_new_span = [0.1 * el for el in bounds_span]
                bounds = [
                    (
                        opti_circle[ind] - half_new_span[ind],
                        opti_circle[ind] + half_new_span[ind],
                    )
                    for ind in range(3)
                ]
                # print(new_best_cost)
                # print(bounds)

            plot_circles.append(opti_circle)

    # Just use the passed circles
    else:
        plot_circles = [circle_a, circle_b]

    # region Circle plotting

    for circle in plot_circles:

        # Debug tweak
        circle[0] -= 0.5

        # Report what we found
        rounded_circle = [round(el, 2) for el in circle]
        rounded_cost = round(cost_func(circle, *args), 5)
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

    # circles = [3]
    circles = [3, 4]
    for circle in circles:

        # Fig. 3
        if circle == 3:
            image_file_name = "2021_09_30-13_18_47-johnson-dnv7_2021_09_23"
            # Best circles by hand
            # circle_a = [41.5, 37, 27.5]
            # circle_b = [40, 44, 27.75]
            # Brute guesses
            circle_a = [42.0, 37.0, 28.0]
            circle_b = [39.0, 43.0, 28.0]
            brute_range = 3

        # Fig. 4
        elif circle == 4:
            image_file_name = "2021_10_17-19_02_22-johnson-dnv5_2021_09_23"
            # Best circles by hand
            # circle_a = [50, 46, 26]
            # circle_b = [51.7, 56.5, 27.3]
            # Brute guesses
            circle_a = [51.0, 46.0, 26.0]
            circle_b = [51.0, 57.0, 27.0]
            brute_range = 3

        main(image_file_name, circle_a, circle_b, brute_range)

    plt.show(block=True)

# endregion
