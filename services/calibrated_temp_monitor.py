# -*- coding: utf-8 -*-
"""
Created on August 9th, 2021

Temperature monitoring code 

@author: mccambria
"""


import matplotlib.pyplot as plt
import utils.tool_belt as tool_belt
import time
import labrad
import socket
import os


def main(channel=1, do_plot=True):
    with labrad.connect() as cxn:
        main_with_cxn(cxn, channel, do_plot)


def main_with_cxn(cxn, channel, do_plot):

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
    now = time.time()
    actual = cxn.temp_monitor_lakeshore218.measure(channel)

    cycle_dur = 0.1
    start_time = now
    prev_time = now

    plot_log_period = 2  # Plot and log every plot_period seconds
    last_plot_log_time = now
    if do_plot:
        plt.ion()
        fig, ax = plt.subplots()
        # fig = plt.figure()
        # ax = fig.add_axes([0, 0, 1, 1])
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
        cur_temp_str = "Current temp: {} K".format(actual)
        cur_temp_text_box = fig.text(0.15, 0.90, cur_temp_str)
        fig.tight_layout()
        # fig.canvas.draw()
        # fig.canvas.flush_events()

    # Break out of the while if the user says stop
    tool_belt.init_safe_stop()
    while not tool_belt.safe_stop():

        # Timing work
        now = time.time()
        time_diff = now - prev_time
        prev_time = now
        if time_diff < cycle_dur:
            time.sleep(cycle_dur - time_diff)

        actual = cxn.temp_monitor_lakeshore218.measure(channel)

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

                cur_temp_str = "Current temp: {} K".format(actual)
                cur_temp_text_box.set_text(cur_temp_str)

                # Redraw the plot with the new data
                fig.canvas.draw()
                fig.canvas.flush_events()

                # Notify the user once the temp is stable (ptp < 0.1 over current plot history)
                if max(plot_temps) - min(plot_temps) < 0.1:
                    msg = "Temp is stable!"
                    recipient = "cambria@wisc.edu"
                    tool_belt.send_email(msg, email_to=recipient)

            with open(logging_file, "a+") as f:
                f.write("{}, {} \n".format(round(now), round(actual, 3)))
            last_plot_log_time = now

    sample_name = "wu"
    timestamp = tool_belt.get_time_stamp()
    file_path = tool_belt.get_file_path(__file__, timestamp, sample_name)
    tool_belt.save_figure(fig, file_path)


if __name__ == "__main__":

    channel = 1
    sensor_serial = "X162689"
    do_plot = True

    with labrad.connect() as cxn:

        # temp = cxn.temp_monitor_lakeshore218.measure(channel)
        # print(temp)

        # cxn.temp_monitor_lakeshore218.enter_calibration_curve(channel, sensor_serial)

        main_with_cxn(cxn, channel, do_plot)
