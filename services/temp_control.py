# -*- coding: utf-8 -*-
"""Temperature control and monitoring service. Control is achieved with a
PID loop 

Created on August 9th, 2021

@author: mccambria
"""

# region Imports and constants

import matplotlib.pyplot as plt
import utils.tool_belt as tool_belt
import utils.kplotlib as kpl
import utils.common as common
import time
import numpy as np
import labrad
import socket
from enum import Enum, auto
import os


class Mode(Enum):
    MONITOR = auto()
    CONTROL = auto()


# endregion
# region PID functions


def pid(pid_state, pid_coeffs, target, actual, integral_max):

    new_pid_state = []

    # Unpack last measurement
    last_meas_time, last_error, integral, derivative = pid_state
    cur_meas_time = time.time()
    new_pid_state.append(cur_meas_time)

    # Proportional
    cur_error = target - actual
    new_pid_state.append(cur_error)

    # Integral
    new_integral_term = (cur_meas_time - last_meas_time) * last_error
    new_integral = integral + new_integral_term
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

    # Put it all together
    p_coeff, i_coeff, d_coeff = pid_coeffs
    p_comp = p_coeff * cur_error
    i_comp = i_coeff * new_integral
    d_comp = d_coeff * new_derivative

    return new_pid_state, p_comp + i_comp + d_comp


# endregion
# region Main


def main_with_cxn(
    cxn,
    mode=None,
    do_plot=True,
    do_email=False,
    email_recipient=None,
    email_stability=0.1,
    target=None,
    pid_coeffs=[0.5, 0.01, 0],
    integral_bootstrap=0.0,
    integral_max=None,
    safety_check=False,
    safety_check_bounds=[285, 305],
):

    # We'll log all the plotted data to this file. (If plotting is turned off,
    # we'll log the data that we would've plotted.) The format is:
    # <rounded time.time()>, <temp in K> \n
    pc_name = socket.gethostname()
    file_name = os.path.basename(__file__)
    file_name_no_ext = os.path.splitext(file_name)[0]
    nvdata_dir = common.get_nvdata_dir()
    logging_file = nvdata_dir / f"pc_{pc_name}/service_logging/{file_name_no_ext}.log"

    temp_monitor = tool_belt.get_server_temp_monitor(cxn)
    power_supply = tool_belt.get_server_power_supply(cxn)
    now = time.time()
    email_sent = False
    cycle_dur = 0.1
    start_time = now
    prev_time = now
    actual = temp_monitor.measure()

    # Initialize the state for control mode
    if mode == Mode.CONTROL:
        # Last meas time, last error, integral, derivative
        # Check the third-party temperature monitor to make sure nothing is melting
        if safety_check:
            safety_check_period = 10
            last_safety_check_time = now
            safety_lower, safety_upper = safety_check_bounds
        last_meas_time = now
        last_error = target - actual
        integral = integral_bootstrap
        derivative = 0.0
        pid_state = [last_meas_time, last_error, integral, derivative]

    # Plotting setup
    plot_log_period = 2  # Plot and log every plot_period seconds
    last_plot_log_time = now
    if do_plot:
        kpl.init_kplotlib()
        fig, ax = plt.subplots()
        plot_times = [0]
        plot_temps = [actual]
        kpl.plot_line(ax, plot_times, plot_temps)
        history = 600
        max_plot_vals = history / plot_log_period
        plot_x_extent = int(1.1 * max_plot_vals * plot_log_period)
        ax.set_xlim(0, plot_x_extent)
        ax.set_ylim(actual - 2, actual + 2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Temp (K)")
        cur_temp_str = f"Current temp: {actual} K"
        cur_temp_text_box = kpl.anchored_text(
            ax, cur_temp_str, kpl.Loc.UPPER_LEFT, size=kpl.Size.SMALL
        )

    # Break out of the while if the user says stop
    tool_belt.init_safe_stop()
    while not tool_belt.safe_stop():

        # Timing work
        now = time.time()
        if now - start_time > 2.25*60*60:
            break
        time_diff = now - prev_time
        prev_time = now
        if time_diff < cycle_dur:
            time.sleep(cycle_dur - time_diff)

        actual = temp_monitor.measure()

        # Plotting and logging
        if now - last_plot_log_time > plot_log_period:
            if do_plot:

                elapsed_time = round(now - start_time)
                plot_times.append(elapsed_time)
                plot_temps.append(actual)

                kpl.plot_line_update(ax, x=plot_times, y=plot_temps)

                # Relim as necessary
                if len(plot_times) > max_plot_vals:
                    plot_times.pop(0)
                    plot_temps.pop(0)
                min_plot_time = min(plot_times)
                ax.set_xlim(min_plot_time, min_plot_time + 1.05*plot_x_extent)
                ax.set_ylim(min(plot_temps) - 2, max(plot_temps) + 2)

                cur_temp_str = f"Current temp: {round(actual, 3)} K"
                cur_temp_text_box.txt.set_text(cur_temp_str)

                kpl.flush_update(ax)

            # Notify the user once the temp is stable (ptp < email_stability over current plot history)
            if do_email:
                temp_check = max(plot_temps) - min(plot_temps) < email_stability
                time_check = len(plot_times) == max_plot_vals
                if temp_check and time_check and not email_sent:
                    msg = "Temp is stable!"
                    tool_belt.send_email(msg, email_to=email_recipient)
                    email_sent = True

            with open(logging_file, "a+") as f:
                f.write(f"{round(now)}, {round(actual, 3)} \n")
            last_plot_log_time = now

        # Update state and set the power accordingly
        if mode == Mode.CONTROL:
            pid_state, power = pid(pid_state, pid_coeffs, target, actual, integral_max)
            if safety_check:
                if now - last_safety_check_time > safety_check_period:
                    safety_check_temp = cxn.temp_controller_tc200.measure()
                    with open(logging_file, "a+") as f:
                        f.write(
                            f"safety check: {round(now)}, {round(safety_check_temp, 1)} \n"
                        )
                    last_safety_check_time = now
                    if not safety_lower < safety_check_temp < safety_upper:
                        print(
                            f"Safety check temperature out of bounds at {safety_check_temp}! Exiting."
                        )
                        return
            power_supply.set_power(power)


# endregion

if __name__ == "__main__":

    do_plot = True
    target = 310
    pid_coeffs = [0.5, 0.01, 0]
    integral_max = 2500
    # Bootstrap the integral term after restarting to mitigate windup,
    # ringing, etc
    # integral_bootstrap = 0.0
    # integral_bootstrap = 0.3 * integral_max
    integral_bootstrap = 0.1 * integral_max
    # integral_bootstrap = integral_max

    with labrad.connect() as cxn:

        # Set up the multimeter for resistance measurement
        temp_monitor = tool_belt.get_server_temp_monitor(cxn)
        power_supply = tool_belt.get_server_power_supply(cxn)
        temp_monitor.config_temp_measurement("PT100")

        # Set up the safety check temp monitor
        # cxn.temp_controller_tc200.config_measurement("ptc100", "k")

        # Set up the power supply
        power_supply.set_power_limit(30)
        power_supply.set_voltage_limit(30)
        # Bootstrap to get a faster initial resistance reading
        power_supply.set_current(0.0)
        power_supply.set_voltage(0.1)
        power_supply.output_on()

        # temp = cxn.multimeter_mp730028.measure()
        # print(temp)

        try:
            main_with_cxn(
                cxn,
                mode=Mode.CONTROL,
                do_plot=do_plot,
                target=target,
                pid_coeffs=pid_coeffs,
                integral_bootstrap=integral_bootstrap,
                integral_max=integral_max,
                safety_check=True,
            )
        finally:
            power_supply.output_off()
