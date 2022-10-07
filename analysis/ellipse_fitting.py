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
from utils.kplotlib import KplColors
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

    # The cost is the square of the closest distance between the
    # point and the ellipse
    # This, of course, turns out to be a hard problem, so let's just
    # run another minimization for it
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
        base_guess = np.arctan(abs(guess_arg)) % np.pi
        # Check the guess and its compliments
        guesses = [
            base_guess,
            np.pi - base_guess,
            base_guess + np.pi,
            2 * np.pi - base_guess,
        ]

        best_cost = None
        for guess in guesses:
            res = minimize(theta_cost, guess)
            opti_theta = res.x
            opti_cost = res.fun
            if (best_cost is None) or (opti_cost < best_cost):
                best_cost = opti_cost

        # Error checking
        if debug and best_cost > 0.005:
            test = 1
        cost += best_cost

    return cost


def gen_ellipses():

    num_ellipses = 20
    theta_linspace = np.linspace(0, 2 * np.pi, 20)
    phi_linspace = np.linspace(0, np.pi / 2, num_ellipses)
    phis = phi_linspace.tolist()

    ellipses = []
    for phi in phi_linspace:
        ellipse = [phi]
        ellipse_lambda = lambda theta: ellipse_point(theta, phi)
        points = [ellipse_lambda(val) for val in theta_linspace]
        ellipse.extend(points)
        ellipses.append(ellipse)

    return ellipses


def import_ellipses(path):

    x_vals = []
    y_vals = []
    phis = []
    file_x = "testingX.csv"
    file_y = "testingY.csv"
    file_phi = "testingPhi_d.csv"

    trans_x = zip(*csv.reader(open(path / file_x, newline="")))
    for row in trans_x:
        float_row = [float(el) for el in row]
        x_vals.append(float_row)

    trans_y = zip(*csv.reader(open(path / file_y, newline="")))
    for row in trans_y:
        float_row = [float(el) for el in row]
        y_vals.append(float_row)

    with open(path / file_phi, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            float_row = [float(el) for el in row]
            phis.append(float_row)

    ellipses = []
    for ellipse_x, ellipse_y, ellipse_phi in zip(x_vals, y_vals, phis):
        points = list(zip(ellipse_x, ellipse_y))
        ellipse = ellipse_phi
        ellipse.extend(points)
        ellipses.append(ellipse)

    return ellipses


# endregion

# region Main


def main(path):

    ellipses = import_ellipses(path)
    # ellipses = gen_ellipses()
    theta_linspace = np.linspace(0, 2 * np.pi, 100)

    # if True:
    #     ellipse = ellipses[15]
    for ellipse in ellipses[::10]:
        
        fig, ax = plt.subplots()
        
        true_phi = ellipse[0]
        points = ellipse[1:]

        res = minimize(ellipse_cost, np.pi / 4, args=(points,))
        opti_phi = res.x[0]

        for point in points:
            kpl.plot_data(ax, *point, color=KplColors.BLUE.value)
        ellipse_lambda = lambda theta: ellipse_point(theta, opti_phi)
        # ellipse_points = ellipse_lambda(theta_linspace)
        # for el in ellipse_points:

        x_vals, y_vals = zip(ellipse_lambda(theta_linspace))
        x_vals = x_vals[0]
        y_vals = y_vals[0]
        kpl.plot_line(ax, x_vals, y_vals)
        
        kpl.tight_layout(fig)

        print(true_phi)
        print(opti_phi)
        print(ellipse_cost(true_phi, points, True))
        print(ellipse_cost(opti_phi, points))
        print()


# endregion

if __name__ == "__main__":

    kpl.init_kplotlib()

    home = common.get_nvdata_dir()
    path = home / "ellipse_data"

    main(path)

    plt.show(block=True)
