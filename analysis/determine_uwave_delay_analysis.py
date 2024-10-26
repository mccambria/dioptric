import os
import time
from random import shuffle

import labrad
import matplotlib.pyplot as plt
import numpy
from scipy.optimize import curve_fit, fsolve

import majorroutines.targeting as targeting
import utils.kplotlib as kpl
import utils.positioning as positioning
import utils.tool_belt as tool_belt
from utils.kplotlib import KplColors
from utils.tool_belt import NormStyle


def solve_linear(m, b, y, z):
    x = z[0]

    F = numpy.empty((1))

    F[0] = m * x + b - y
    return F


kpl.init_kplotlib()


def find_uwave_delay(file):
    """
    A file that fits a downward linear line to the first 1/3 of the data and
    finds the intercept with the horizontal linear line of the last 1/3 of the data
    
    This works best for data that is roughly centered on the value of the delay:
    | \
    |  \
    |   \___
    |
    ----------
    """
    data = tool_belt.get_raw_data(file)

    delay_range = data["delay_range"]
    num_steps = data["num_steps"]
    norm_avg_sig = data["norm_avg_sig"]
    state = data["state"]

    taus = numpy.linspace(delay_range[0], delay_range[1], num_steps)
    third_num_steps = int(num_steps / 3)

    decline_points = norm_avg_sig[0:third_num_steps]
    line_points = norm_avg_sig[int(2 * third_num_steps) : -1]

    decline_taus = taus[0:third_num_steps]

    line_avg = numpy.average(line_points)

    fit_func = tool_belt.linear
    init_params = [-1, 0]
    popt, pcov = curve_fit(fit_func, decline_taus, decline_points, p0=init_params)

    fig, ax = plt.subplots()
    ax.set_xlabel("Delay (ns)")
    ax.set_ylabel("Signal")
    ax.set_title("MW delay for {}".format(state))
    # ax.set_title('AWG trigger delay')
    kpl.plot_line(ax, taus, norm_avg_sig)
    smooth_taus = numpy.linspace(taus[0], taus[-1], num=1000)
    kpl.plot_line(
        ax,
        smooth_taus,
        fit_func(smooth_taus, *popt),
        color=KplColors.RED,
    )
    ax.axhline(line_avg)

    # # find intersection of linear line and offset (half of full range)
    solve_linear_func = lambda z: solve_linear(popt[0], popt[1], line_avg, z)
    zGuess = numpy.array(60)
    solve = fsolve(solve_linear_func, zGuess)
    x_intercept = solve[0]
    print(x_intercept)

    base_text = "MW delay {:.1f} ns"
    # base_text = "AWG delay {:.1f} ns"
    size = kpl.Size.SMALL
    text = base_text.format(x_intercept)
    kpl.anchored_text(ax, text, kpl.Loc.UPPER_RIGHT, size=size)


# %%%
file = "2023_03_21-09_58_57-siena-nv0_2023_03_20"
find_uwave_delay(file)
