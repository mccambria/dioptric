# -*- coding: utf-8 -*-
"""
Created on August 9th, 2021

PID temperature control service with a multimeter and power supply

@author: mccambria
"""


import matplotlib.pyplot as plt
import utils.tool_belt as tool_belt
import time
import numpy
import labrad
import socket
import os


def set_power(cxn, power):
    # Thorlabs HT24S
    if power > 24:
        power = 24
    if power <= 0.01:
        power = 0.01  # Can't actually set 0 exactly, but this is close enough
    # P = V2 / R
    # V = sqrt(P R)
    resistance = 23.5
    voltage = numpy.sqrt(power * resistance)
    # print(voltage)
    cxn.power_supply_mp710087.set_voltage(voltage)


def calc_error(state, target, actual):

    # Last meas time
    last_meas_time, last_error, integral, derivative = state
    cur_meas_time = time.time()
    state[0] = cur_meas_time

    # Last error
    cur_error = target - actual
    state[1] = cur_error

    # Integral
    new_integral_term = (cur_meas_time - last_meas_time) * last_error
    state[2] = integral + new_integral_term

    # Derivative
    # Ignore noise (0.05 K)
    cur_diff = cur_error - last_error
    if abs(cur_diff) > 0.05:
        cur_derivative = cur_diff / (cur_meas_time - last_meas_time)
        state[3] = cur_derivative
    else:
        state[3] = 0.0


def pid(state, pid_coeffs):

    _, last_error, integral, derivative = state
    p_coeff, i_coeff, d_coeff = pid_coeffs

    p_comp = p_coeff * last_error
    i_comp = i_coeff * integral
    d_comp = d_coeff * derivative

    return p_comp + i_comp + d_comp


def main_with_cxn(cxn, do_plot, target, pid_coeffs):

    # We'll log all the plotted data to this file. (If plotting is turned off,
    # we'll log the data that we would've plotted.) The format is:
    # <rounded time.time()>, <temp in K> \n
    pc_name = socket.gethostname()
    file_name = os.path.basename(__file__)
    file_name_no_ext = os.path.splitext(file_name)[0]
    logging_file = (
        "E:/Shared drives/Kolkowitz Lab"
        " Group/nvdata/pc_{}/service_logging/{}.log"
    ).format(pc_name, file_name_no_ext)

    # Initialize the state
    # Last meas time, last error, integral, derivative
    now = time.time()
    actual = cxn.multimeter_mp730028.measure()
    state = [time.time(), actual, 0.0, 0.0]

    cycle_dur = 0.1
    start_time = round(now)
    prev_time = now

    if do_plot:
        plt.ion()
        fig, ax = plt.subplots()
        plot_times = [now]
        plot_temps = [actual]
        plot_period = 10  # Plot every plot_period seconds
        ax.plot(plot_times, plot_temps)
        max_plot_vals = 600
        plot_x_extent = int(1.1 * max_plot_vals * plot_period)
        ax.set_xlim(0, plot_x_extent)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Temp (K)")

    # Break out of the while if the user says stop
    tool_belt.init_safe_stop()
    while not tool_belt.safe_stop():

        # Timing work
        now = time.time()
        time_diff = now - prev_time
        prev_time = now
        if time_diff < cycle_dur:
            # print(time_diff)
            time.sleep(cycle_dur - time_diff)

        actual = cxn.multimeter_mp730028.measure()

        # Plotting and logging
        if now - last_plot_time > plot_period:
            if do_plot:
                elapsed_time = round(now) - start_time
                plot_times.append(elapsed_time)
                plot_temps.append(actual)
                ax.plot(plot_times, plot_temps)
                # Relim as necessary
                if len(plot_times) > max_plot_vals:
                    plot_times.pop(0)
                    plot_temps.pop(0)
                    min_plot_time = min(plot_times)
                    ax.set_xlim(min_plot_time, min_plot_time + plot_x_extent)
                # Allow the plot to update
                plt.pause(0.01)
            last_plot_time = now
            with open(logging_file, "a") as f:
                f.write("{}, {} \n".format(now, actual))

        # Update state and set the power accordingly
        # print(actual)
        calc_error(state, target, actual)
        power = pid(state, pid_coeffs)
        # print(power)
        set_power(cxn, power)


if __name__ == "__main__":

    do_plot = True
    target = 310.0
    pid_coeffs = [0.5, 0.01, 0]

    with labrad.connect() as cxn:
        # Set up the multimeter for temperature measurement
        cxn.multimeter_mp730028.config_temp_measurement("PT100", "K")
        cxn.power_supply_mp710087.set_current_limit(1.0)
        cxn.power_supply_mp710087.set_voltage_limit(24.0)
        cxn.power_supply_mp710087.set_current(0.0)
        cxn.power_supply_mp710087.set_voltage(0.01)
        cxn.power_supply_mp710087.output_on()

        # temp = cxn.multimeter_mp730028.measure()
        # print(temp)

        main_with_cxn(cxn, do_plot, target, pid_coeffs)

        # input("Press enter to stop...")
        cxn.power_supply_mp710087.output_off()
