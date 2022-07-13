# -*- coding: utf-8 -*-
"""
Fit circles to superresolution rings in images demonstrating resolved
images of two NVs separated by less than the diffraction limit.
By full auto I mean you don't have to tell the program how many circles
there are - it'll figure it out for you by finding the best circle,
finding the next best circle, and so on until there are no more good
circles left.

Created on February 25, 2022

@author: mccambria
"""

# region Imports

import utils.tool_belt as tool_belt
import utils.common as common
import copy
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, brute
from numpy import pi
from matplotlib.patches import Circle
import cv2 as cv
import sys
import multiprocessing
from functools import partial
import time
import superresolution_circles_fake_data as fake_data

# endregion

# region Constants

num_circle_samples = 1000

phi_linspace = np.linspace(0, 2 * pi, num_circle_samples, endpoint=False)
cos_phi_linspace = np.cos(phi_linspace)
sin_phi_linspace = np.sin(phi_linspace)

# endregion

# region Functions


def cost0(params, image, x_lim, y_lim, debug, excluded_centers):
    """
    Faux-integrate the pixel values around the circle. By faux-integrate I mean
    average the values under a 1000 point, linearly spaced sampling of the circle.

    excluded_centers: list of tuples (x,y) describing potential centers that we don't
    want to consider. Useful for ignoring circles we've already found when optimizing.
    """

    circle_center_x, circle_center_y, circle_radius = params

    # Check if the circle is in the exclusion range
    exclusion_range = 4  # Pixels
    for excluded_x, excluded_y in excluded_centers:
        dist = np.sqrt(
            (excluded_x - circle_center_x) ** 2
            + (excluded_y - circle_center_y) ** 2
        )
        if dist < exclusion_range:
            return 1

    circle_samples_x = circle_center_x + circle_radius * cos_phi_linspace
    circle_samples_y = circle_center_y + circle_radius * sin_phi_linspace
    circle_samples_x_round = [round(el) for el in circle_samples_x]
    circle_samples_y_round = [round(el) for el in circle_samples_y]
    circle_samples = zip(circle_samples_x_round, circle_samples_y_round)

    check_valid = lambda el: (0 <= el[1] < x_lim) and (0 <= el[0] < y_lim)
    integrand = [image[el] for el in circle_samples if check_valid(el)]

    cost = np.sum(integrand) / len(integrand)

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
    # sigmoid = 1 / (1 + np.exp(-5 * quotient - 0.0))
    # sigmoid = quotient
    laplacian_positive = np.sign(laplacian) == 1
    sigmoid = (sigmoid * gradient_not_zeros) + (
        laplacian_positive * gradient_zeros
    )
    return sigmoid


def calc_errors(image_file_name, circle_a, circle_b):

    cost_func = cost0

    # Get the image as a 2D ndarray
    image_file_dict = tool_belt.get_raw_data(image_file_name)
    image = np.array(image_file_dict["readout_image_array"])
    image_domain = image.shape
    image_len_x = image_domain[1]
    image_len_y = image_domain[0]

    ret_vals = process_image(image)
    sigmoid_image = ret_vals[-1]

    fig, axes_pack = plt.subplots(1, 3)
    fig.set_tight_layout(True)

    num_points = 1000
    sweep_half_range = 10
    # for circle in [circle_a]:
    for circle in [circle_a, circle_b]:

        print(circle)
        args = [sigmoid_image, image_len_x, image_len_y, False]
        opti_cost = cost_func(circle, *args)

        for param_ind in range(3):

            ax = axes_pack[param_ind]
            sweep_center = circle[param_ind]
            sweep_vals = np.linspace(
                sweep_center - sweep_half_range,
                sweep_center + sweep_half_range,
                num_points,
            )

            cost_vals = []
            for sweep_ind in range(num_points):
                test_circle = list(circle)
                test_circle[param_ind] = sweep_vals[sweep_ind]
                cost_vals.append(0.5 - cost_func(test_circle, *args))

            ax.plot(sweep_vals, cost_vals)

            left_width = None
            right_width = None
            if param_ind == 2:
                target_max_ratio = 0.5
            else:
                target_max_ratio = 1 - (1 / pi)
            half_max = target_max_ratio * (0.5 - opti_cost)
            half_ind = num_points // 2
            for delta in range(half_ind):
                test_ind = half_ind - delta
                if (cost_vals[test_ind] < half_max) and (left_width is None):
                    left_width = sweep_vals[test_ind]
                test_ind = half_ind + delta
                if (cost_vals[test_ind] < half_max) and (right_width is None):
                    right_width = sweep_vals[test_ind]
                if (left_width is not None) and (right_width is not None):
                    break
            if right_width is not None and left_width is not None:
                half_width_at_half_max = (right_width - left_width) / 2
                print(half_width_at_half_max)

    print()


def process_image(image):

    # Blur
    gaussian_size = 7
    blur_image = cv.GaussianBlur(image, (gaussian_size, gaussian_size), 0)

    gradient_root = blur_image

    laplacian_image = cv.Laplacian(
        gradient_root, cv.CV_64F, ksize=gaussian_size
    )
    # offset = np.average(np.abs(laplacian_image))
    # print(offset)
    # offset = np.sqrt(np.average(laplacian_image ** 2))
    # # # print(offset)
    # laplacian_image += offset
    # laplacian_image -= np.min(laplacian_image)

    sobel_x = cv.Sobel(gradient_root, cv.CV_64F, 1, 0, ksize=gaussian_size)
    sobel_y = cv.Sobel(gradient_root, cv.CV_64F, 0, 1, ksize=gaussian_size)
    gradient_image = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    # gradient_image += 100
    # gradient_image = gradient_image**2

    sigmoid_image = sigmoid_quotient(laplacian_image, gradient_image)
    # sigmoid_image = cv.GaussianBlur(
    #     sigmoid_image, (gaussian_size, gaussian_size), 0
    # )

    return blur_image, laplacian_image, gradient_image, sigmoid_image


def calc_distance(fig, x0, x1, y0, y1, sx0, sx1, sy0, sy1):

    dx = x1 - x0
    dy = y1 - y0
    distance = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
    err = np.sqrt(
        ((dx / distance) ** 2) * (sx0 ** 2 + sx1 ** 2)
        + ((dy / distance) ** 2) * (sy0 ** 2 + sy1 ** 2)
    )

    print(distance)
    print(err)

    # 0.0004375 V, for Fig 4 each pixel is 0.0005 V. And the conversion is 34.8 um/V
    if fig == 3:
        conversion = 15.225  # nm / pixel
    elif fig == 4:
        conversion = 17.4  # nm / pixel
    distance_nm = conversion * distance
    err_nm = conversion * err
    print(distance_nm)
    print(err_nm)

    print()


# endregion


def main(
    image_file_name,
    circle_a=None,
    circle_b=None,
    fast_recursive=False,
    brute_range=None,
):

    # region Setup

    cost_func = cost0
    # minimize_type = "manual"
    # minimize_type =   "publication"
    # minimize_type = "recursive"
    minimize_type = "full_auto"

    if image_file_name == "fake":
        image = fake_data.main()
    else:
        # Get the image as a 2D ndarray
        image_file_dict = tool_belt.get_raw_data(image_file_name)
        image = np.array(image_file_dict["readout_image_array"])

    image_domain = image.shape
    image_len_x = image_domain[1]
    image_len_y = image_domain[0]

    ret_vals = process_image(image)
    blur_image, laplacian_image, gradient_image, sigmoid_image = ret_vals

    opti_image = sigmoid_image
    plot_image = sigmoid_image

    # Plot the image
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    img = ax.imshow(plot_image, cmap="inferno")
    _ = plt.colorbar(img)
    # return

    # endregion

    # region Circle finding

    excluded_centers = []
    args = [opti_image, image_len_x, image_len_y, False, excluded_centers]
    # Partial function with everything but the circle parameters filled in
    cost_func_partial = partial(
        cost_func, image=args[0], x_lim=args[1], y_lim=args[2], debug=args[2]
    )
    plot_circles = []

    start = time.time()

    if minimize_type == "manual":

        x_linspace = np.linspace(0, image_len_x, image_len_x, endpoint=False)
        y_linspace = np.linspace(0, image_len_x, image_len_y, endpoint=False)
        rad_linspace = np.linspace(25, 30, 10)

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
            # print(circle)
            bounds = [(val - 1, val + 1) for val in circle]
            res = minimize(
                cost_func, circle, bounds=bounds, args=args, method="L-BFGS-B"
            )
            opti_circle = res.x
            plot_circles.append(opti_circle)

    elif minimize_type == "publication":
        half_range = 18  # for circle 3
        # half_range = 20 #for circle 4
        num_points = 100
        half_len_x = image_len_x // 2
        half_len_y = image_len_x // 2
        x_linspace = np.linspace(
            half_len_x - half_range, half_len_x + half_range, num_points
        )
        y_linspace = np.linspace(
            half_len_y - half_range, half_len_y + half_range, num_points
        )
        reconstruction = []
        print(image_len_x)
        print(x_linspace)

        # return

        # x_linspace = np.linspace(0, image_len_x, image_len_x, endpoint=False)
        # y_linspace = np.linspace(0, image_len_y, image_len_y, endpoint=False)
        rad_linspace = np.linspace(26, 28, 21)
        # r = np.average([circle_a[2], circle_b[2]])

        # image_copy = copy.deepcopy(image)
        # image_copy[:] = np.nan
        # image_copy_log = np.copy(image_copy)

        half_x = image_len_x / 2

        # Manual brute force optimization for left/right halves
        for y in y_linspace:
            reconstruction.append([])
            for x in x_linspace:

                # set the r value
                # if x < half_x:
                #     r = circle_a[2]
                # else:
                #     r = circle_b[2]

                # Cost at a fixed r
                # circle = [y, x, r]
                # cost_value = 0.5 - cost_func(circle, *args)

                # just recording min cost value over rad_linspace
                # test_circles = [[y, x, r] for r in rad_linspace]
                # with multiprocessing.Pool() as pool:
                #     cost_vals = pool.map(cost_func_partial, test_circles)
                # Slow singlethreaded version
                cost_func_lambda = lambda r: cost_func([y, x, r], *args)
                cost_vals = [cost_func_lambda(r) for r in rad_linspace]
                # cost_value = 0.5 - min(cost_vals)
                cost_value = min(cost_vals)

                # print(cost_value)
                # image_copy[int(y)][int(x)] = cost_value
                # image_copy_log[int(y)][int(x)] = np.log(cost_value)
                reconstruction[-1].append(cost_value)

        fig2, ax = plt.subplots()
        fig2.set_tight_layout(True)
        # img = ax.imshow(image_copy, cmap="YlGnBu_r")
        extent = [
            min(x_linspace),
            max(x_linspace),
            max(y_linspace),
            min(y_linspace),
        ]
        img = ax.imshow(reconstruction, cmap="inferno_r", extent=extent)
        _ = plt.colorbar(img)

        # figlog, ax = plt.subplots()
        # figlog.set_tight_layout(True)
        # img = ax.imshow(image_copy_log, cmap="inferno")
        # _ = plt.colorbar(img)

        plot_circles = [circle_a, circle_b]

    elif minimize_type == "recursive":

        # Define the bounds of the optimization
        if fast_recursive:
            bounds_a = [(el - 1, el + 1) for el in circle_a]
            bounds_b = [(el - 1, el + 1) for el in circle_b]
        else:
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
                    cost_func,
                    bounds,
                    Ns=20,
                    args=args,
                    finish=None,
                    workers=-1,  # Multiprocessing: -1 means use as many cores as available
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

    elif minimize_type == "full_auto":

        while True:

            # Define the bounds of the optimization
            bounds = [
                (image_len_y / 4, 3 * image_len_y / 4),
                (image_len_x / 4, 3 * image_len_x / 4),
                # (20, 35),
                (40, 60),
            ]

            # Anything worse than minimum_cost and we stop searching for circles
            minimum_cost = 0.5

            # Initialize best_cost
            best_cost = 1

            while True:

                # Update args with the last found opti_circle
                args[-1] = excluded_centers

                opti_circle = brute(
                    cost_func,
                    bounds,
                    Ns=20,
                    args=args,
                    finish=None,
                    workers=-1,  # Multiprocessing: -1 means use as many cores as available
                )
                args[-1] = []
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

            # Quit if our best is worse than the threshold we set, but make sure
            # we have at least one circle to plot
            if (len(plot_circles) > 0) and (new_best_cost > minimum_cost):
                break
            excluded_centers.append(opti_circle[0:2])
            plot_circles.append(opti_circle)
            if len(plot_circles) > 5:
                break

    end = time.time()
    print(f"Elapsed time: {end-start}")

    # endregion

    # region Circle plotting

    for circle in plot_circles:

        # Debug tweak
        # circle[0] -= 0.5

        # Report what we found
        rounded_circle = [round(el, 2) for el in circle]
        rounded_cost = round(cost_func(circle, *args), 5)
        # print("{} & {} & {} & {}".format(*rounded_circle, rounded_cost))
        print("{}, {}, {}, {}".format(*rounded_circle, rounded_cost))

        # Plot the circle
        circle_patch = Circle(
            (circle[1], circle[0]),
            circle[2],
            fill=False,
            color="w",
        )
        ax.add_patch(circle_patch)

        # Plot the center
        circle_patch = Circle(
            (circle[1], circle[0]),
            0.5,
            fill="w",
            color="w",
        )
        ax.add_patch(circle_patch)

    # endregion
    return


# region Run the file

if __name__ == "__main__":

    # # Fig 3
    # calc_distance(3, 36.85, 43.87, 41.74, 39.05, 1.4, 0.9, 1.5, 1.3)
    # # Fig 4
    # calc_distance(4, 45.79, 56.32, 50.98, 51.2, 1.0, 1.1, 1.4, 1.4)

    # sys.exit()

    # tool_belt.init_matplotlib()

    # # circles = [3]
    # circles = [4]
    # # circles = [3, 4]
    # for circle in circles:

    #     # Fig. 3
    #     if circle == 3:
    #         image_file_name = "2021_09_30-13_18_47-johnson-dnv7_2021_09_23"
    #         # Best circles by hand
    #         # circle_a = [41.5, 37, 27.5]
    #         # circle_b = [40, 44, 27.75]
    #         # Recursive brute results, 1000 point circle
    #         circle_a = [41.74, 36.85, 27.73]  # 0.31941
    #         # errs_a = [1.4, 1.4, 1.2]
    #         circle_b = [39.05, 43.87, 27.59]  # 0.36108
    #         # errs_b = [1.5, 0.9, 1.0]

    #     # Fig. 4
    #     elif circle == 4:
    #         image_file_name = "2021_10_17-19_02_22-johnson-dnv5_2021_09_23"
    #         # Best circles by hand
    #         # circle_a = [50, 46, 26]
    #         # circle_b = [51.7, 56.5, 27.3]
    #         # Recursive brute results, 1000 point circle
    #         circle_a = [50.98, 45.79, 26.14]  # 0.3176
    #         # circle_a = [50.98 - 0, 45.79 + 0, 26.14 - 0]  # 0.3176
    #         # errs_a = [2.1, 1.1, 1.3]
    #         circle_b = [51.2, 56.32, 27.29]  # 0.35952
    #         # errs_b = [1.8, 1.1, 1.2]

    #     main(image_file_name, circle_a, circle_b, fast_recursive=True)
    #     # calc_errors(image_file_name, circle_a, circle_b)

    main("fake")

    plt.show(block=True)

# endregion


#  0.0004375 V, for Fig 4 each pixel is 0.0005 V. And the conversion is 34.8 um/V

#  Fig 3: 15.225 nm / pixel
#  Fig 4: 17.4 nm / pixel
