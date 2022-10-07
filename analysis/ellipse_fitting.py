# -*- coding: utf-8 -*-
"""
Trial ellipse fitting

Created October 3rd, 2022

@author: mccambria
"""

### Imports
import numpy as np
from numpy.core.shape_base import block
from scipy.optimize import root_scalar, minimize_scalar, minimize
import utils.tool_belt as tool_belt
import time
import matplotlib.pyplot as plt
from utils import common
from utils import kplotlib as kpl
from scipy.optimize import curve_fit
import csv

cent = 0.5
amp = 0.65 / 2

# region Functions


def ellipse_point(theta, phi):
    return (cent + amp * np.cos(theta + phi), cent + amp * np.cos(theta - phi))


def ellipse_cost(phi, points, debug=False):

    if debug:
        test = 1

    ellipse_lambda = lambda theta: ellipse_point(theta, phi)

    # The cost is the root mean square distance between the
    # point and the ellipse
    # Finding the closest distance between an arbitary point and an ellipse,
    # of course, turns out to be a hard problem, so let's just run another
    # minimization for it
    cost = 0
    for point in points:
        theta_cost = lambda theta: sum(
            [(el[0] - el[1]) ** 2 for el in zip(ellipse_lambda(theta), point)]
        )
        # Guess theta by assuming theta and the ellipse amplitude are free and
        # solving for the position of the point on the ellipse
        # x = cent + amp * np.cos(theta + phi)
        # y = cent + amp * np.cos(theta - phi)
        # x-y = amp * (np.cos(theta + phi) - np.cos(theta - phi))
        #     = -2 amp sin(theta) sin(phi)
        # amp = (x-y) / (-2 sin(theta) sin(phi))
        # x+y = 2cent + 2 amp cos(theta) cos(phi)
        #     = 2cent - (x-y) cot(theta) cot(phi)
        x, y = point
        guess_arg = ((x - y)) / (np.tan(phi) * (2 * cent - (x + y)))
        # guess_arg = abs(guess_arg)
        guess_theta_p = np.arctan(guess_arg)
        guess_theta_m = np.pi - guess_theta_p
        # guess_theta = np.arctan(point[1] / point[0])
        # bounds = (guess_theta - np.pi / 4, guess_theta + np.pi / 4)
        # res = minimize_scalar(theta_cost, bounds=bounds)
        # res = minimize_scalar(theta_cost)
        res_p = minimize(theta_cost, guess_theta_p)
        point_cost_p = res_p.fun
        res_m = minimize(theta_cost, guess_theta_m)
        point_cost_m = res_m.fun
        if point_cost_p < point_cost_m:
            opti_theta = res_p.x
            point_cost = point_cost_p
        else:
            opti_theta = res_m.x
            point_cost = point_cost_m
        if debug and point_cost > 0.005:
            test = 1
        cost += point_cost

    num_points = len(points)
    cost = np.sqrt(cost / num_points)

    return cost


def gen_ellipses():

    num_ellipses = 20
    theta_linspace = np.linspace(0, 2 * np.pi, 20)
    phi_linspace = np.linspace(0, np.pi, num_ellipses)
    phis = phi_linspace.tolist()

    ellipses = []
    for phi in phi_linspace:
        ellipse = [[phi]]
        ellipse_lambda = lambda theta: ellipse_point(theta, phi)
        points = [ellipse_lambda(val) for val in theta_linspace]
        ellipse.extend(points)
        ellipses.append(ellipse)

    return ellipses


def populate_imported_phis(path, files):

    phis_by_file = []

    for el in files:
        sub_phis = []
        with open(path / el, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                float_row = [float(el) for el in row]
                sub_phis.append(float_row)
        phis_by_file.append(sub_phis)

    # Re-sort by ellipse
    num_files = len(files)
    num_ellipses = len(phis[0])
    phis = []
    for ellipse_ind in range(num_ellipses):
        sub_phis = []
        for sub_ind in range(num_files):
            sub_phis.append(phis_by_file[sub_ind][ellipse_ind])
        phis.append(sub_phis)

    return phis


def import_ellipses(path):

    x_vals = []
    y_vals = []
    phis = []
    file_x = "testingX.csv"
    file_y = "testingY.csv"
    phi_files = ["testingPhi_d.csv", "phi_LS.csv", "phi_NN.csv"]

    trans_x = zip(*csv.reader(open(path / file_x, newline="")))
    for row in trans_x:
        float_row = [float(el) for el in row]
        x_vals.append(float_row)

    trans_y = zip(*csv.reader(open(path / file_y, newline="")))
    for row in trans_y:
        float_row = [float(el) for el in row]
        y_vals.append(float_row)

    phis = populate_imported_phis(path, phi_files)

    ellipses = []
    for ellipse_x, ellipse_y, ellipse_phi in zip(x_vals, y_vals, phis):
        points = list(zip(ellipse_x, ellipse_y))
        ellipse = [ellipse_phi]
        ellipse.extend(points)
        ellipses.append(ellipse)

    return ellipses


# endregion

# region Main


def main(path):

    # ellipses = import_ellipses(path)
    ellipses = gen_ellipses()
    theta_linspace = np.linspace(0, 2 * np.pi, 100)

    opti_phis = []
    ellipse_inds = range(len(ellipses))

    phi_errors = []

    if True:
        phi_errors_sub = []
        ellipse = ellipses[1]
        # for ellipse in ellipses[0]:
        ellipse_phis = ellipse[0]
        true_phi = ellipse_phis.pop(0)
        algo_phis = ellipse_phis
        points = ellipse[1:]

        res = minimize_scalar(ellipse_cost, args=(points,), bounds=(0, np.pi))
        opti_phi = res.x
        algo_phis.insert(0, opti_phi)

        # Plot the data points
        for point in points:
            plt.scatter(*point, color=kpl.KplColors.BLUE.value)

        # Plot the fit
        ellipse_lambda = lambda theta: ellipse_point(theta, opti_phi)
        x_vals, y_vals = zip(ellipse_lambda(theta_linspace))
        x_vals = x_vals[0]
        y_vals = y_vals[0]
        plt.plot(x_vals, y_vals)

        # Get the costs
        test_phis = [true_phi].extend(algo_phis)
        for phi in test_phis:
            cost = ellipse_cost(phi, points, True)
            print(f"{phi}: {cost}")

        # Get the phi errors
        for phi in algo_phis:
            phi_errors_sub.append(phi - true_phi)
        phi_errors.append(phi_errors_sub)

    print(phi_errors)
    phi_errors = np.array(phi_errors)
    mean_phi_errors = np.mean(phi_errors, axis=0)
    print(mean_phi_errors)


# endregion

if __name__ == "__main__":

    kpl.init_kplotlib()

    home = common.get_nvdata_dir()
    path = home / "ellipse_data"

    main(path)

    plt.show(block=True)
