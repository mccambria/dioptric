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


import utils.tool_belt as tool_belt
import majorroutines.optimize as optimize
import numpy
import matplotlib.pyplot as plt
import time


# %% Functions


def calculate_differences(buffer, diff_window, differences_append,
                          tagger_di_apd_a, tagger_di_apd_b):

    # Couple shorthands to speed up the calculation
    buffer_tagTimestamps = buffer.tagTimestamps
    buffer_tagChannels = buffer.tagChannels
    buffer_size = buffer.size

    indices_to_delete = []
    indices_to_delete_append = indices_to_delete.append

    # Throw out probable afterpulses
    for click_index in range(buffer.size):

        click_time = buffer_tagTimestamps[click_index]

        # Determine the afterpulse channel
        click_channel = buffer_tagChannels[click_index]

        # Calculate relevant differences
        next_index = click_index + 1
        while next_index < buffer_size:
            diff = buffer_tagTimestamps[next_index] - click_time
            if diff > 50 * 10**3:
                break
            if buffer_tagChannels[next_index] == click_channel:
                indices_to_delete_append(next_index)
            next_index += 1

    print("num deleted afterpulses: " + str(len(indices_to_delete)))
    buffer_tagTimestamps = numpy.delete(buffer_tagTimestamps, indices_to_delete)
    buffer_tagChannels = numpy.delete(buffer_tagChannels, indices_to_delete)

    num_vals = buffer_tagTimestamps.size

    for click_index in range(num_vals):

        click_time = buffer_tagTimestamps[click_index]

        # Determine the channel to take the difference with
        click_channel = buffer_tagChannels[click_index]
        if click_channel == tagger_di_apd_a:
            diff_channel = tagger_di_apd_b
        else:
            diff_channel = tagger_di_apd_a

        # Calculate relevant differences
        next_index = click_index + 1
        while next_index < num_vals:  # Don't go past the buffer end
            # Stop taking differences past the diff window
            diff = buffer_tagTimestamps[next_index] - click_time
            if diff > diff_window:
                break
            # Only record the diff between opposite chanels
            if buffer_tagChannels[next_index] == diff_channel:
                # Flip the sign for diffs relative to channel 2
                if click_channel == tagger_di_apd_b:
                    diff = -diff
                differences_append(diff)
            next_index += 1


# %% Main


def main(cxn, name, coords, apd_a_index, apd_b_index):

    # %% Initial calculations and setup

    total_size = 0
    collect_time = 0

    run_time = 60 * 7
#    run_time = 30
    sleep_time = 2
    start_time = time.time()

    collection_index = 0

    # Create a list to hold the differences
    differences = []
    # Don't append every loop - just do it once here
    differences_append = differences.append
    diff_window = 100 * 10**3  # 100 ns in ps
    num_bins = int((2 * diff_window) / 1000)  # 1 ns bins in ps

    buffer = None

    tool_belt.init_safe_stop()

    # %% Collect the data

    while time.time() - start_time < run_time:

        if tool_belt.safe_stop():
            break

        start_calc_time = time.time()

        if buffer is not None:

            calculate_differences(buffer, diff_window, differences_append,
                                  tagger_di_apd_a, tagger_di_apd_b)

            total_size += buffer.size
            print("calc time: " + str(time.time() - start_calc_time))

            if collection_index == 1:
                # Plot the differences as a histogram
                fig, ax = plt.subplots()
                hist, bin_edges = numpy.histogram(differences, num_bins)
                bin_edges = bin_edges / 1000  # ps to ns
                bin_center_offset = (bin_edges[1] - bin_edges[0]) / 2
                bin_centers = bin_edges[0: num_bins] + bin_center_offset
                ax.plot(bin_centers, hist)
                ax.set_xlabel('Time (ns)')
                ax.set_ylabel('Differences')
                ax.set_title(r'$g^{(2)}(\tau)$')
                fig.tight_layout()
                # Draw the canvas and flush the events to the backend
                fig.canvas.draw()
                fig.canvas.flush_events()
            elif collection_index > 1:
                hist, bin_edges = numpy.histogram(differences, num_bins)
                tool_belt.update_line_plot_figure(fig, hist)

            collection_index += 1

        now = time.time()
        time_elapsed = now - start_calc_time
        time.sleep(max(sleep_time - time_elapsed, 0))
        stream.getData(buffer_hardware)
        buffer = buffer_hardware
        print("buffer size: " + str(buffer.size))
        if buffer.size != 0:
            print("first tag at: " + str(buffer.tagTimestamps[0]))

    int_differences = list(map(int, differences))

    rawData = {"name": name,
               "xyzCenters": [x_center, y_center, z_center],
               "differences": int_differences,
               "total_size": total_size,
               "collect_time": collect_time}
    print("total collect time: " + str(collect_time))


