# -*- coding: utf-8 -*-
"""Temperature monitoring code

Created on August 9th, 2021

@author: mccambria
"""


import matplotlib.pyplot as plt
import utils.tool_belt as tool_belt
import utils.kplotlib as kpl
import utils.common as common
import time
import labrad
import socket
import csv
import os
import numpy as np


def main(
    channel=1,
    do_plot=True,
    do_email=False,
    email_recipient=None,
    email_stability=0.1,
):
    """Monitor the temperature on the monitor listed in the registry

    Parameters
    ----------
    channel : int, optional
        Channel of the temp monitor to use, by default 1
    do_plot : bool, optional
        Whether the temp is plotted and logged or just logged, by default True
    do_email : bool, optional
        Whether an email is sent when the temp is stable to within email_stability, by default False
    email_recipient : str, optional
        Address to email, by default None
    email_stability : float, optional
        10-minute peak-to-peak stability threshold for emailing, by default 0.1
    """
    with labrad.connect() as cxn:
        main_with_cxn(
            cxn, channel, do_plot, do_email, email_recipient, email_stability
        )


def main_with_cxn(
    cxn, channel, do_plot, do_email, email_recipient, email_stability
):

    # We'll log all the plotted data to this file. (If plotting is turned off,
    # we'll log the data that we would've plotted.) The format is:
    # <rounded time.time()>, <temp in K> \n
    pc_name = socket.gethostname()
    file_name = os.path.basename(__file__)
    file_name_no_ext = os.path.splitext(file_name)[0]
    nvdata_dir = common.get_nvdata_dir()
    logging_file = (
        nvdata_dir / f"pc_{pc_name}/service_logging/{file_name_no_ext}.log"
    )

    # Initialize the state
    now = time.time()
    temp_monitor = tool_belt.get_server_temp_monitor(cxn)
    actual = temp_monitor.measure(channel)

    cycle_dur = 0.1
    start_time = now
    prev_time = now

    email_sent = False

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
        cur_temp_text_box = kpl.anchored_text(ax, cur_temp_str, 
                                    kpl.Loc.UPPER_LEFT, size=kpl.Size.SMALL)

    # Break out of the while if the user says stop
    tool_belt.init_safe_stop()
    while not tool_belt.safe_stop():

        # Timing work
        now = time.time()
        time_diff = now - prev_time
        prev_time = now
        if time_diff < cycle_dur:
            time.sleep(cycle_dur - time_diff)

        actual = temp_monitor.measure(channel)

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
                ax.set_xlim(min_plot_time, min_plot_time + plot_x_extent)
                ax.set_ylim(min(plot_temps) - 2, max(plot_temps) + 2)

                cur_temp_str = f"Current temp: {actual} K"
                cur_temp_text_box.txt.set_text(cur_temp_str)

                kpl.flush_update(ax)

                # Notify the user once the temp is stable (ptp < email_stability over current plot history)
                temp_check = (
                    max(plot_temps) - min(plot_temps) < email_stability
                )
                time_check = len(plot_times) == max_plot_vals
                if do_email and temp_check and time_check and not email_sent:
                    msg = "Temp is stable!"
                    tool_belt.send_email(msg, email_to=email_recipient)
                    email_sent = True

            with open(logging_file, "a+") as f:
                f.write("{}, {} \n".format(round(now), round(actual, 3)))
            last_plot_log_time = now

    timestamp = tool_belt.get_time_stamp()
    file_path = tool_belt.get_file_path(__file__, timestamp, "none")
    tool_belt.save_figure(fig, file_path)


if __name__ == "__main__":

    channel = 1
    do_plot = True
    do_email = True
    email_recipient = "cambria@wisc.edu"
    email_stability = 0.1

    main(
        channel,
        do_plot,
        do_email=do_email,
        email_recipient=email_recipient,
        email_stability=email_stability,
    )
