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


def calc_error(state, target, actual):
    last_meas_time, last_error, integral, derivative = state
    cur_meas_time = time.time()
    state[0] = cur_meas_time
    cur_error = target - actual
    state[1] = cur_error
    state[2] = integral + [(cur_meas_time-last_meas_time) * last_error]
    cur_derivative = (cur_error - last_error) / (cur_meas_time - last_meas_time)
    # If we just use the current derivative, then we'll likely be overly
    # sensitive to measurement noise. By mixing the current measured
    # derivative in with the current derivative value in state, we're 
    # performing a rolling average over all past derivative measurements
    # with geometric weighting such that past values, which we don't care
    # about, are exponentially suppressed. 
    state[3] = 0.5 * (derivative + cur_derivative)


def pid(state, pid_coeffs, actual):
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

    _, last_error, integral, derivative = state
    p_coeff, i_coeff, d_coeff = pid_coeffs

    # Proportional component
    p_comp = p_coeff * last_error

    # Integral component
    i_comp = i_coeff * integral

    # Derivative component
    d_comp = d_coeff * derivative

    return p_comp + i_comp + d_comp


def main_with_cxn(cxn, target, pid_coeffs):

    # Initialize the state
    # Last meas time, last error, integral, derivative
    state = [time.time(), 
             cxn.multimeter_mp730028.measure(), 
             0.0,
             0.0]

    # Start 'Press enter to stop...'
    tool_belt.init_safe_stop()

    # Break out of the while if the user says stop
    while not tool_belt.safe_stop():
        actual = cxn.multimeter_mp730028.measure()
        calc_error(state, target, actual)
        power = pid(state)
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

        main_with_cxn(cxn, target, pid_coeffs)
