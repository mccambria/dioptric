# -*- coding: utf-8 -*-
"""
Created on August 9th, 2021

Service for controlling the temperature with a multimeter and power supply

@author: mccambria
"""


import utils.tool_belt as tool_belt
import time
import numpy


def meas_temp():
    pass


def set_power(power):
    pass


def calc_error(run_params, actual):
    target, _, error_history, time_history = run_params
    error = target - actual
    error_history.append(error)
    now = time.time()
    time_history.append(now)
    # Keep the last 50 or 1 minute of values, whichever is greater
    if (now - time_history[0] > 60) or (len(time_history) >= 50):
        error_history = error_history[1:]
        time_history = time_history[1:]


def pid(run_params, actual):
    """Returns the power to set given the current actual temperature 
    measurement and the target temperature

    Parameters
    ----------
    actual : float
        Current actual temperature (K)
    target : float
        Current target temperature (K)
        
    Returns
    ----------
    float
        The power to set the power supply to
    """

    calc_error(run_params, actual)

    _, pid_coeffs, error_history, time_history = run_params

    p_coeff, i_coeff, d_coeff = pid_coeffs

    # Proportional component
    p_comp = p_coeff * error_history[-1]

    # Integral component
    # Riemann sum over the full 1 second memory
    time_diffs = []
    for ind in range(len(time_history)) - 1:
        time_diffs.append(time_history[ind + 1] - time_history[ind])
    time_diffs.append(time.time() - time_history[-1])
    integral = numpy.dot(error_history, time_diffs)
    i_comp = i_coeff * integral

    # Derivative component
    # Find the index of the first error recorded more than 0.1 s back
    current_time = time_history[-1]
    ind = -2
    while True:
        diff = time_history[ind] - current_time
        if diff > 0.1:
            break
        else:
            ind -= 1
    error_diff = error_history[-1] - error_history[ind]
    time_diff = time_history[-1] - time_history[ind]
    derivative = error_diff / time_diff
    d_comp = d_coeff * derivative

    return p_comp + i_comp + d_comp


def main_loop(run_params):

    actual = meas_temp()
    power = pid(run_params, actual)
    set_power(power)


if __name__ == "__main__":

    target = 300.0
    pid_coeffs = []
    error_history = []
    time_history = []
    run_params = [target, pid_coeffs, error_history, time_history]

    # Get a few errors to boot strap the PID loop
    start_time = time.time()
    while time.time() - start_time < 1.0:
        actual = meas_temp()
        calc_error(run_params, actual)

    # Start 'Press enter to stop...'
    tool_belt.init_safe_stop()

    # Break out of the while if the user says stop
    while not tool_belt.safe_stop():
        # Just run as fast as we can
        main_loop(run_params)
