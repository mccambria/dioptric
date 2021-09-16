# -*- coding: utf-8 -*-
"""
Created on August 9th, 2021

PID temperature control service using a multimeter and power supply

@author: mccambria
"""


import matplotlib.pyplot as plt
import utils.tool_belt as tool_belt
import time
import numpy
import labrad
import socket
import os

power_limit_factor = 0.96
integral_max = power_limit_factor * 24 / 0.01


def set_power(cxn, power, resistance):
    global power_limit_factor
    # Thorlabs HT24S: limit is 24 W, but let's not quite max it out
    power_limit = power_limit_factor * 24
    if power > power_limit:
        power = power_limit
    if power <= 0.01:
        power = 0.01  # Can't actually set 0 exactly, but this is close enough
    # P = V2 / R
    # V = sqrt(P R)
    voltage = numpy.sqrt(power * resistance)
    # return
    cxn.power_supply_mp710087.set_voltage(voltage)


def calc_error(pid_state, target, actual):

    new_pid_state = []

    # Last meas time
    last_meas_time, last_error, integral, derivative = pid_state
    cur_meas_time = time.time()
    new_pid_state.append(cur_meas_time)

    # Last error
    cur_error = target - actual
    new_pid_state.append(cur_error)

    # Integral
    new_integral_term = (cur_meas_time - last_meas_time) * last_error
    new_integral = integral + new_integral_term
    global integral_max
    integral_min = 0.0
    if new_integral > integral_max:
        new_integral = integral_max
    elif new_integral < integral_min:
        new_integral = integral_min
    new_pid_state.append(new_integral)

    # Derivative
    # Ignore noise (0.05 K)
    cur_diff = cur_error - last_error
    if abs(cur_diff) > 0.05:
        cur_derivative = cur_diff / (cur_meas_time - last_meas_time)
        new_derivative = cur_derivative
    else:
        new_derivative = 0.0
    new_pid_state.append(new_derivative)

    return new_pid_state


def pid(pid_state, pid_coeffs):

    _, last_error, integral, derivative = pid_state
    p_coeff, i_coeff, d_coeff = pid_coeffs

    p_comp = p_coeff * last_error
    i_comp = i_coeff * integral
    d_comp = d_coeff * derivative

    return p_comp + i_comp + d_comp


def update_resistance(cxn):
    """The resistance may change as a function of temperature.
    The real limit we care about is the power, so let's feed the current
    measured resistance forward when we go to set the voltage (we can't
    set power directly). This measurement is slow so only do it occasionally
    """

    # Thorlabs HT24S: 23.5 ohms
    nominal_resistance = 23.5
    resistance = cxn.power_supply_mp710087.meas_resistance()
    # If the measured resistance is more than 2.5 X or less than 1/2.5 X
    # the nominal value, assume something funny is happening (maybe the
    # output is not on yet) and just use the nominal resistance
    too_high = resistance > 2.5 * nominal_resistance
    too_low = resistance < nominal_resistance / 2.5
    if too_high or too_low:
        resistance = nominal_resistance
    # print(resistance)
    return resistance


def main_with_cxn(cxn, do_plot, target, pid_coeffs, integral_bootstrap=0.0):

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
    resistance = update_resistance(cxn)
    first_set = True
    # Update the resistance every resistance_period seconds
    resistance_period = 10
    last_resistance_time = now
    # Check the third-party temperature monitor to make sure nothing is melting
    safety_check_period = 10 
    last_safety_check_time = now + 5  # Offset this from the resistance checks
    last_meas_time = now
    last_error = target - actual
    integral = integral_bootstrap
    derivative = 0.0
    pid_state = [last_meas_time, last_error, integral, derivative]

    cycle_dur = 0.1
    start_time = now
    prev_time = now

    plot_log_period = 2  # Plot and log every plot_period seconds
    last_plot_log_time = now
    if do_plot:
        plt.ion()
        fig, ax = plt.subplots()
        plot_times = [0]
        plot_temps = [actual]
        ax.plot(plot_times, plot_temps)
        history = 600
        max_plot_vals = history / plot_log_period
        plot_x_extent = int(1.1 * max_plot_vals * plot_log_period)
        ax.set_xlim(0, plot_x_extent)
        ax.set_ylim(actual - 2, actual + 2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Temp (K)")
        fig.canvas.draw()
        fig.canvas.flush_events()

    # Break out of the while if the user says stop
    tool_belt.init_safe_stop()
    while not tool_belt.safe_stop():

        # Timing work
        now = time.time()
        time_diff = now - prev_time
        prev_time = now
        if time_diff < cycle_dur:
            time.sleep(cycle_dur - time_diff)

        actual = cxn.multimeter_mp730028.measure()

        # Plotting and logging
        if now - last_plot_log_time > plot_log_period:
            if do_plot:

                elapsed_time = round(now - start_time)
                plot_times.append(elapsed_time)
                plot_temps.append(actual)

                lines = ax.get_lines()
                line = lines[0]
                line.set_xdata(plot_times)
                line.set_ydata(plot_temps)

                # Relim as necessary
                if len(plot_times) > max_plot_vals:
                    plot_times.pop(0)
                    plot_temps.pop(0)
                    min_plot_time = min(plot_times)
                    ax.set_xlim(min_plot_time, min_plot_time + plot_x_extent)
                ax.set_ylim(min(plot_temps) - 2, max(plot_temps) + 2)

                # Redraw the plot with the new data
                fig.canvas.draw()
                fig.canvas.flush_events()

            with open(logging_file, "a+") as f:
                f.write("{}, {} \n".format(round(now), round(actual, 3)))
            last_plot_log_time = now

        # Update state and set the power accordingly
        pid_state = calc_error(pid_state, target, actual)
        power = pid(pid_state, pid_coeffs)
        if now - last_resistance_time > resistance_period:
            resistance = update_resistance(cxn)
            last_resistance_time = now
        if now - last_safety_check_time > safety_check_period:
            safety_check_temp = cxn.temp_controller_tc200.measure()
            with open(logging_file, "a+") as f:
                f.write("safety check: {}, {} \n".format(round(now), round(safety_check_temp, 1)))
            last_safety_check_time = now
            if (safety_check_temp < 285) or (safety_check_temp > 305):
                print("Safety check temperature out of bounds at {}! Exiting.".format(safety_check_temp))
                return
        set_power(cxn, power, resistance)
        # Immediately get a better resistance measurement after the first set
        if first_set:
            resistance = update_resistance(cxn)
            first_set = False


if __name__ == "__main__":

    do_plot = False
    target = 337.5
    pid_coeffs = [0.5, 0.01, 0]
    # Bootstrap the integral term after restarting to mitigate windup,
    # ringing, etc
    # integral_bootstrap = 0.0
    integral_bootstrap = 0.3 * integral_max
    # integral_bootstrap = 0.6 * integral_max
    # integral_bootstrap = integral_max

    with labrad.connect() as cxn:
        # Set up the multimeter for resistance measurement
        cxn.multimeter_mp730028.config_temp_measurement("PT100")
        cxn.temp_controller_tc200.config_measurement("ptc100", "k")
        cxn.power_supply_mp710087.set_current_limit(1.0)
        # Allow the voltage to be somewhat higher than the specs suggest
        # it should max out at to account for temperature dependence of
        # the heating element resistance. 36 is the voltage we'd get
        # driving 90% of 24 W for a resistance 2.5 times the nominal 23.5 ohms
        cxn.power_supply_mp710087.set_voltage_limit(36)
        cxn.power_supply_mp710087.set_current(0.0)
        cxn.power_supply_mp710087.set_voltage(0.01)
        cxn.power_supply_mp710087.output_on()

        # temp = cxn.multimeter_mp730028.measure()
        # print(temp)

        try:
            main_with_cxn(cxn, do_plot, target, pid_coeffs, integral_bootstrap)
        finally:
            cxn.power_supply_mp710087.output_off()
