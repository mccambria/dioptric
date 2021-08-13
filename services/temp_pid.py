# -*- coding: utf-8 -*-
"""
Created on August 9th, 2021

Service for controlling the temperature with a multimeter and power supply

@author: mccambria
"""


import utils.tool_belt as tool_belt
import time
import numpy
import labrad


def set_power(cxn, power):
    # Thorlabs HT24S
    if power > 24:
        power = 24
    # P = V2 / R
    # V = sqrt(P R)
    resistance = 23.5  
    voltage = numpy.sqrt(power * resistance)
    # print(voltage)
    cxn.power_supply_mp710087.set_voltage(voltage)


def calc_error(run_params, actual):
    target, _, error_history, time_history = run_params
    error = target - actual
    error_history.append(error)
    now = time.time()
    time_history.append(now)
    # Keep the last 50 or 1 minute of values, whichever is greater
    if (now - time_history[0] > 60) or (len(time_history) >= 50):
        error_history.pop(0)
        time_history.pop(0)


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
    for ind in range(len(time_history) - 1):
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
        if (diff > 0.1) or (ind == -len(time_history)):
            break
        else:
            ind -= 1
    error_diff = error_history[-1] - error_history[ind]
    time_diff = time_history[-1] - time_history[ind]
    derivative = error_diff / time_diff
    d_comp = d_coeff * derivative

    return p_comp + i_comp + d_comp


def main_with_cxn(cxn, target, pid_coeffs):
    
    error_history = []
    time_history = []
    run_params = [target, pid_coeffs, error_history, time_history]

    # Get a few errors to boot strap the PID loop
    start_time = time.time()
    while time.time() - start_time < 1.0:
        actual = cxn.multimeter_mp730028.measure()
        calc_error(run_params, actual)
    power = pid(run_params, actual)

    # Start 'Press enter to stop...'
    tool_belt.init_safe_stop()

    # Break out of the while if the user says stop
    while not tool_belt.safe_stop():
        # Just run as fast as we can
        actual = cxn.multimeter_mp730028.measure()
        power = pid(run_params, actual)
        # print(power)
        set_power(cxn, power)


if __name__ == "__main__":

    target = 300.0
    pid_coeffs = [0.001, 0, 0]
    
    with labrad.connect() as cxn:
        # Set up the multimeter for temperature measurement
        cxn.multimeter_mp730028.config_temp_measurement("PT100", "K")
        cxn.power_supply_mp710087.set_current_limit(1)
        cxn.power_supply_mp710087.set_voltage_limit(24)
        cxn.power_supply_mp710087.set_current(0)
        cxn.power_supply_mp710087.set_voltage(0)
        cxn.power_supply_mp710087.output_on()
        
        # temp = cxn.multimeter_mp730028.measure()
        # print(temp)
        
        main_with_cxn(cxn, target, pid_coeffs)
