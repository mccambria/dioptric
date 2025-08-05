# -*- coding: utf-8 -*-
"""
g(2) routine. For each event on one channel, calculates the deltas relative to
the events on the opposite channel and plots a histogram of the deltas. Here
the events are photon detections from the same source, but split over two
APDs. The splitting is necessary to compensate for the APD dead time, which
is typically significantly longer than the lifetime of the excited state we
are interested in.

Created on Wed Apr 24 17:33:26 2019

@author: mccambria
"""


# %% Imports

import json
import time

import labrad
import matplotlib.pyplot as plt
import numpy

import majorroutines.targeting as targeting
import utils.kplotlib as kpl
import utils.tool_belt as tool_belt
from utils.kplotlib import KplColors

# %% Functions


def illuminate(cxn, laser_power, laser_name):
    pulse_gen = tool_belt.get_server_pulse_gen(cxn)
    wiring = tool_belt.get_pulse_streamer_wiring(cxn)
    if not laser_power:
        pulse_gen.constant([wiring["do_{}_dm".format(laser_name)]], 0.0, 0.0)
    else:
        analog_channel = wiring["ao_{}_am".format(laser_name)]
        if analog_channel == 1:
            pulse_gen.constant([], laser_power, 0.0)
        elif analog_channel == 2:
            pulse_gen.constant([], 0.0, laser_power)


def calculate_relative_g2_zero(hist):
    # We take the values on the wings to be representatives for g2(inf)
    # We take the wings to be the first and last 1/6 of collected data
    num_bins = len(hist)
    wing_length = num_bins // 12
    neg_wing = hist[0:wing_length]
    pos_wing = hist[num_bins - wing_length :]
    inf_delay_differences = numpy.average([neg_wing, pos_wing])

    # Use the parity of num_bins to determine differences at 0 ns
    if num_bins % 2 == 0:
        # As an example, say there are 6 bins. Then we want the differences
        # from bins 2 and 3 (indexing starts from 0).
        midpoint_high = num_bins // 2
        zero_delay_differences = numpy.average(hist[midpoint_high - 1, midpoint_high])
    else:
        # Now say there are 7 bins. We'd like bin 3.
        midpoint = int(numpy.floor(num_bins / 2))
        zero_delay_differences = hist[midpoint]

    return zero_delay_differences / inf_delay_differences


def process_raw_buffer(
    timestamps,
    channels,
    diff_window_ps,
    afterpulse_window,
    differences_append,
    apd_a_chan_name,
    apd_b_chan_name,
):
    indices_to_delete = []
    indices_to_delete_append = indices_to_delete.append

    # Throw out probable afterpulses
    # Something is wrong with this... see 2019-06-03_17-05-01_ayrton12
    if False:
        num_vals = timestamps.size
        click_index = 0
        last_deleted_a_index = -1
        last_deleted_b_index = -1
        for click_index in range(num_vals):
            click_time = timestamps[click_index]
            click_channel = channels[click_index]

            # if click_channel == apd_a_chan_name:
            #     continue

            # Skip over events we've already decided to delete
            if (click_channel == apd_a_chan_name) and (
                click_index <= last_deleted_a_index
            ):
                continue
            elif (click_channel == apd_b_chan_name) and (
                click_index <= last_deleted_b_index
            ):
                continue

            # Calculate relevant differences
            next_index = click_index + 1
            while next_index < num_vals:
                diff = timestamps[next_index] - click_time
                if diff > afterpulse_window:
                    break
                if channels[next_index] == click_channel:
                    if click_channel == apd_a_chan_name:
                        last_deleted_a_index = next_index
                    elif click_channel == apd_a_chan_name:
                        last_deleted_b_index = next_index
                    indices_to_delete_append(next_index)
                next_index += 1

        timestamps = numpy.delete(timestamps, indices_to_delete)
        channels = numpy.delete(channels, indices_to_delete)

    # Calculate differences
    num_vals = timestamps.size
    for click_index in range(num_vals):
        click_time = timestamps[click_index]

        # Determine the channel to take the difference with
        click_channel = channels[click_index]
        if click_channel == apd_b_chan_name:
            diff_channel = apd_a_chan_name
        elif click_channel == apd_a_chan_name:
            diff_channel = apd_b_chan_name

        # if click_channel == apd_a_chan_name:
        #     continue

        # Calculate relevant differences
        next_index = click_index + 1
        while next_index < num_vals:  # Don't go past the buffer end
            # Stop taking differences past the diff window
            diff = timestamps[next_index] - click_time
            if diff > diff_window_ps:
                break
            # Only record the diff between opposite chanels
            if channels[next_index] == diff_channel:
                # Flip the sign for diffs relative to channel b
                if click_channel == apd_b_chan_name:
                    diff = -diff
                differences_append(int(diff))
            next_index += 1

    # MCC
    # Calculate differences
    # num_vals = timestamps.size
    # for click_index in range(num_vals):

    #     click_time = timestamps[click_index]

    #     # Determine the channel to take the difference with
    #     click_channel = channels[click_index]
    #     if click_channel == 1:
    #         diff_channel = apd_a_chan_name
    #     else:
    #         continue

    #     # Calculate relevant differences
    #     next_index = click_index + 1
    #     while next_index < num_vals:  # Don't go past the buffer end
    #         # Stop taking differences past the diff window
    #         diff = timestamps[next_index] - click_time
    #         if diff > diff_window_ps:
    #             break
    #         # Only record the diff between opposite chanels
    #         if channels[next_index] == diff_channel:
    #             # Flip the sign for diffs relative to channel b
    #             if click_channel == apd_b_chan_name:
    #                 diff = -diff
    #             differences_append(int(diff))
    #         next_index += 1


# %% Main


def main(nv_sig, run_time, diff_window):
    with labrad.connect() as cxn:
        main_with_cxn(cxn, nv_sig, run_time, diff_window)


def main_with_cxn(cxn, nv_sig, run_time, diff_window):
    # %% Initial calculations and setup

    tool_belt.reset_cfm(cxn)
    kpl.init_kplotlib()
    tagger_server = tool_belt.get_server_tagger(cxn)

    laser_key = "imaging_laser"
    laser_name = nv_sig[laser_key]

    # 200 ns to account for twilighting and afterpulsing
    afterpulse_window = 200
    #    sleep_time = 2 # (?)
    # afterpulse_window = (diff_window - 200) * 1000

    # apd_indices = [apd_a_index, apd_b_index]

    # Optimize and set filters
    opti_coords = targeting.main_with_cxn(cxn, nv_sig)
    tool_belt.set_filter(cxn, nv_sig, laser_key)
    laser_power = tool_belt.set_laser_power(cxn, nv_sig, laser_key)

    illuminate(cxn, laser_power, laser_name)

    num_tags = 0
    collection_index = 0
    cent_diff_window = diff_window + 0.5

    cent_diff_window_ps = cent_diff_window * 1000
    differences = []  # Create a list to hold the differences
    differences_append = differences.append  # Skip unnecessary lookup
    num_bins = int(2 * cent_diff_window)
    # num_bins = 151
    # num_bins = diff_window + 1
    # num_bins = int(2*mod)-1

    # Get the APD channel names that the tagger will return
    apd_wiring = tool_belt.get_tagger_wiring(cxn)
    apd_a_chan_name = apd_wiring["di_apd_gate"]
    apd_b_chan_name = apd_a_chan_name

    # Expose the stream
    apd_a_di_ind = apd_wiring["di_apd_0"]
    apd_b_di_ind = apd_wiring["di_apd_0"]

    tagger_server.start_tag_stream([apd_a_di_ind, apd_b_di_ind], [], False)

    # %% Collect the data
    start_time = time.time()
    print("Time remaining:")
    tool_belt.init_safe_stop()

    # Python does not have do-while loops so we will use something like
    # a while True
    stop = False
    time_remaining = run_time
    # num_pointsss = 50
    # vals = numpy.linspace(0, num_pointsss, num_pointsss+1)
    # vals *= 8
    total_tags = []
    while not stop:
        # Wait until some data has filled
        #        now = time.time()
        #        calc_time_elapsed = now - start_time
        #        time.sleep(max(sleep_time - calc_time_elapsed, 0))
        # Read the stream and convert from strings to int64s
        #        ret_vals = cxn.apd_tagger.read_tag_stream()
        #        ret_vals = tool_belt.decode_time_tags(ret_vals_string)
        #        buffer_timetags, buffer_channels = ret_vals
        #        buffer_timetags = numpy.array(buffer_timetags, dtype=numpy.int64)

        # Check if we should stop
        new_time_remaining = int((start_time + run_time) - time.time())
        if (time_remaining < 0) or tool_belt.safe_stop():
            stop = True
        # Do not spam the console witth the same number
        elif new_time_remaining < time_remaining:
            print(new_time_remaining)
            time_remaining = new_time_remaining

        time.sleep(1.0)

        # Optimize every 2 minutes
        elapsed_time = run_time - new_time_remaining
        if (
            (elapsed_time % 120 == 0)
            and (elapsed_time > 0)
            and (new_time_remaining > 0)
        ):
            cxn.apd_tagger.stop_tag_stream()
            opti_coords = targeting.main_with_cxn(cxn, nv_sig)
            tool_belt.set_filter(cxn, nv_sig, laser_key)
            laser_power = tool_belt.set_laser_power(cxn, nv_sig, laser_key)
            illuminate(cxn, laser_power, laser_name)
            cxn.apd_tagger.start_tag_stream(clock=False)

        # Read the stream and convert from strings to int64s
        ret_vals = cxn.apd_tagger.read_tag_stream()
        buffer_timetags, buffer_channels = ret_vals
        buffer_timetags = numpy.array(buffer_timetags, dtype=numpy.int64)
        total_tags.extend(buffer_timetags.tolist())
        # print('num tags: {}'.format(len(buffer_channels)))
        #        print(buffer_timetags)
        #        print(buffer_channels)
        # return

        # Process data
        process_raw_buffer(
            buffer_timetags,
            buffer_channels,
            cent_diff_window_ps,
            afterpulse_window,
            differences_append,
            apd_a_chan_name,
            apd_b_chan_name,
        )
        #        print(differences)
        # return

        # Create/update the histogram
        if collection_index == 0:
            fig, ax = plt.subplots()
            hist, bin_edges = numpy.histogram(
                differences, num_bins, (-cent_diff_window_ps, cent_diff_window_ps)
            )
            bin_edges = bin_edges / 1000  # ps to ns
            bin_centers = []
            for ind in range(len(bin_edges) - 1):
                center = (bin_edges[ind] + bin_edges[ind + 1]) // 2
                bin_centers.append(int(center))
            ax.plot(bin_centers, hist)
            # ax.plot(bin_centers, hist, marker='o', linestyle='none', markersize=3)
            xlim = int(1.1 * diff_window)
            ax.set_xlim(-xlim, xlim)

            #####

            # diff_mod = (numpy.array(differences) // 1000) % mod
            # hist, bin_edges = numpy.histogram(diff_mod, num_bins,
            #                           (-mod+1, mod-1))
            # ax.plot(numpy.linspace(-mod,+mod, num_bins), hist)
            # ax.set_xlim(-mod-1, mod+1)

            #####

            ax.set_xlabel("Time (ns)")
            ax.set_ylabel("Differences")
            ax.set_title(r"$g^{(2)}(\tau)$")
            fig.set_tight_layout(True)
            fig.canvas.draw()
            fig.canvas.flush_events()

        elif collection_index > 1:
            # print(buffer_timetags)
            hist, bin_edges = numpy.histogram(
                differences, num_bins, (-cent_diff_window_ps, cent_diff_window_ps)
            )
            # diff_mod = (numpy.array(differences) // 1000) % mod
            # hist, bin_edges = numpy.histogram(diff_mod, num_bins,
            #                           (-mod+1, mod-1))
            tool_belt.update_line_plot_figure(fig, hist)

        collection_index += 1
        num_tags += len(buffer_timetags)

    cxn.apd_tagger.stop_tag_stream()

    # %% Calculate a relative value for g2(0)

    g2_zero = calculate_relative_g2_zero(hist)

    # %% Clean up and save the data

    tool_belt.reset_cfm(cxn)

    timestamp = tool_belt.get_time_stamp()

    raw_data = {
        "timestamp": timestamp,
        "nv_sig": nv_sig,
        "nv_sig-units": tool_belt.get_nv_sig_units(),
        "g2_zero": g2_zero,
        "g2_zero-units": "ratio",
        # 'opti_coords': opti_coords,
        # 'opti_coords-units': 'V',
        "run_time": run_time,
        "run_time-units": "s",
        "diff_window": diff_window,
        "diff_window-units": "ns",
        "num_bins": num_bins,
        "num_tags": num_tags,
        "differences": differences,
        "differences-units": "ps",
    }

    filePath = tool_belt.get_file_path(__file__, timestamp, nv_sig["name"])
    tool_belt.save_figure(fig, filePath)
    tool_belt.save_raw_data(raw_data, filePath)

    print("g2(0) = {}".format(g2_zero))

    return g2_zero


# %% Run the file


def calculate_relative_g2_zero_mod(hist):
    # We take the values on the wings to be representatives for g2(inf)
    # We take the wings to be the first and last 1/6 of collected data
    num_bins = len(hist)
    wing_length = num_bins // 12
    neg_wing = hist[0:wing_length]
    pos_wing = hist[num_bins - wing_length :]
    inf_delay_differences = numpy.average([neg_wing, pos_wing])

    # Use the parity of num_bins to determine differences at 0 ns
    if num_bins % 2 == 0:
        # As an example, say there are 6 bins. Then we want the differences
        # from bins 2 and 3 (indexing starts from 0).
        midpoint_high = num_bins // 2
        zero_delay_differences = numpy.average(hist[midpoint_high - 1, midpoint_high])
    else:
        # Now say there are 7 bins. We'd like bin 3.
        midpoint = int(numpy.floor(num_bins / 2))
        zero_delay_differences = hist[midpoint]

    return zero_delay_differences / inf_delay_differences, inf_delay_differences


if __name__ == "__main__":
    file_name = "2021_12_03-12_10_11-wu-nv3_2021_12_02"
    data = tool_belt.get_raw_data(file_name)

    differences = data["differences"]
    num_bins = data["num_bins"]
    diff_window = data["diff_window"]

    diff_window_ps = diff_window * 1000
    hist, bin_edges = numpy.histogram(
        differences, num_bins, (-diff_window_ps, diff_window_ps)
    )
    bin_edges = bin_edges / 1000  # ps to ns
    bin_center_offset = (bin_edges[1] - bin_edges[0]) / 2
    bin_centers = bin_edges[0:num_bins] + bin_center_offset

    plt.plot(bin_centers, hist)
    g2_zero = calculate_relative_g2_zero(hist)
    print(g2_zero)
