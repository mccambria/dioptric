# -*- coding: utf-8 -*-
"""
Created on Thu May 27 22:30:51 2021

@author: matth
"""


from numpy import random
import numpy
import matplotlib.pyplot as plt


def find_afterpulses(timestamps, channels, chan_names, afterpulse_window):
    """Find afterpulses on channels specified in chan_names"""
    
    indices_to_delete = []
    indices_to_delete_append = indices_to_delete.append
    
    num_vals = timestamps.size
    click_index = 0
    last_deleted_indices = [-1] * 4
    
    for click_index in range(num_vals):

        click_time = timestamps[click_index]
        click_channel = channels[click_index]
        
        # Skip over channels that aren't under consideration
        if click_channel not in chan_names:
            continue
        
        # Skip over events we've already decided to delete
        if click_index <= last_deleted_indices[chan_names.index(click_channel)]:
            continue

        # Calculate relevant differences
        next_index = click_index + 1
        while next_index < num_vals:
            diff = timestamps[next_index] - click_time
            if diff > afterpulse_window:
                break
            # if channels[next_index] == click_channel:
            if channels[next_index] in chan_names:
                last_deleted_indices[chan_names.index(click_channel)] = next_index
                indices_to_delete_append(next_index)
            next_index += 1
                
    return indices_to_delete


def process_raw_buffer(timestamps, channels,
                       diff_window, afterpulse_window,
                       differences_append, apd_a_chan_name, apd_b_chan_name,
                       c_chan_name=None, d_chan_name=None):
    
    if c_chan_name is None:
        chan_names = [apd_a_chan_name, apd_b_chan_name]
    else:
        chan_names = [apd_a_chan_name, apd_b_chan_name, c_chan_name, d_chan_name]

    if afterpulse_window > 0:
        indices_to_delete = find_afterpulses(timestamps, channels, chan_names, afterpulse_window)
        timestamps = numpy.delete(timestamps, indices_to_delete)
        channels = numpy.delete(channels, indices_to_delete)

    # Calculate differences
    num_vals = timestamps.size
    for click_index in range(num_vals):

        click_time = timestamps[click_index]

        # Determine the channel to take the difference with
        click_channel = channels[click_index]
        if c_chan_name is None:
            if click_channel == apd_a_chan_name:
                diff_channel = apd_b_chan_name
            elif click_channel == apd_b_chan_name:
                diff_channel = apd_a_chan_name
        else:
            if click_channel == apd_a_chan_name:
                diff_channel = d_chan_name
            elif click_channel == apd_b_chan_name:
                diff_channel = c_chan_name
            elif click_channel == c_chan_name:
                diff_channel = apd_b_chan_name
            elif click_channel == d_chan_name:
                diff_channel = apd_a_chan_name
            
        # if click_channel == apd_a_chan_name:
        #     continue

        # Calculate relevant differences
        next_index = click_index + 1
        while next_index < num_vals:  # Don't go past the buffer end
            # Stop taking differences past the diff window
            diff = timestamps[next_index] - click_time
            # if diff == 0:
            #     print(channels[next_index])
            #     print(diff_channel)
            #     print()
            if diff > diff_window:
                break
            # Only record the diff between opposite chanels
            if (channels[next_index] == diff_channel) and (diff_channel == apd_b_chan_name):
                # Flip the sign for diffs relative to APD 2 
                if click_channel in [apd_b_chan_name, d_chan_name]:
                    diff = -diff
                differences_append(int(diff))
            next_index += 1
            
    
def plot_hist(differences, num_bins, diff_window):
    
    fig, ax = plt.subplots()
    bin_edges = numpy.linspace(-diff_window+0.5, diff_window-0.5, num_bins)
    hist, _ = numpy.histogram(differences, bin_edges)
    bin_center_offset = (bin_edges[1] - bin_edges[0]) / 2
    bin_centers = bin_edges[0: num_bins-1] + bin_center_offset
    # print(bin_centers)
    ax.plot(bin_centers, hist)
    # ax.plot(bin_centers, hist, marker='o', linestyle='none', markersize=3)
    xlim = int(1.1 * diff_window)
    ax.set_xlim(-xlim, xlim)
    
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Differences')
    ax.set_title(r'$g^{(2)}(\tau)$')
    fig.set_tight_layout(True)
    fig.canvas.draw()
    fig.canvas.flush_events()


def sim_background_background(background_count_rate, bin_size,
                              measurement_time, diff_window):
    
    num_counts = background_count_rate * measurement_time
    # num_counts = 100
    timestamps = random.randint(measurement_time*10**9, size=(num_counts), dtype=numpy.int64)
    timestamps = numpy.sort(timestamps)
    channels = random.randint(2, size=(num_counts))
    random.shuffle(channels)

    differences = []  # Create a list to hold the differences
    differences_append = differences.append  # Skip unnecessary lookup
    
    process_raw_buffer(timestamps, channels, diff_window, 0,
                       differences_append, 0, 1)
    # print(differences)
    # unique = numpy.unique(timestamps)
    # print(num_counts)
    # print(len(unique))
    
    # unique, counts = numpy.unique(differences, return_counts=True)
    # hist = dict(zip(unique, counts))
    # print(hist)

    num_bins = int(2 * diff_window) # 1 ns bins
    
    plot_hist(differences, num_bins, diff_window)


def sim_background_nv(background_count_rate, nv_count_rate, bin_size,
                      measurement_time, diff_window):
    
    num_background_counts = background_count_rate * measurement_time
    num_nv_counts = nv_count_rate * measurement_time
    num_counts = num_background_counts + num_nv_counts
    # num_counts = 100
    timestamps = random.randint(measurement_time*10**9, size=(num_counts), dtype=numpy.int64)
    timestamps = numpy.sort(timestamps)
    channels = [0] * (num_background_counts // 2)
    channels.extend([1] * (num_background_counts // 2))
    channels.extend([2] * (num_nv_counts // 2))
    channels.extend([3] * (num_nv_counts // 2))
    channels = numpy.array(channels)
    random.shuffle(channels)
    
    # Account for NV light antibunching by removing counts on channels 2 or 3 
    # that occur within 100 ns of each other. 
    indices_to_delete = find_afterpulses(timestamps, channels, [2,3], 0)
    timestamps = numpy.delete(timestamps, indices_to_delete)
    channels = numpy.delete(channels, indices_to_delete)

    differences = []  # Create a list to hold the differences
    differences_append = differences.append  # Skip unnecessary lookup
    
    process_raw_buffer(timestamps, channels, diff_window, 0,
                       differences_append, 0, 1, 2, 3)
    # print(differences)
    # unique = numpy.unique(timestamps)
    # print(num_counts)
    # print(len(unique))
    
    # unique, counts = numpy.unique(differences, return_counts=True)
    # hist = dict(zip(unique, counts))
    # print(hist)

    num_bins = int(2 * diff_window) # 1 ns bins
    
    plot_hist(differences, num_bins, diff_window)
    


if __name__ == '__main__':
    
    # Second quantities
    background_count_rate = 12000
    nv_count_rate = 50000 - background_count_rate
    # background_count_rate = 500
    # nv_count_rate = 20000
    measurement_time = 60*10
    
    # ns quantities
    bin_size = 1  # The code assumes 1 ns bins right now
    diff_window = 150
    
    # sim_background_background(background_count_rate, bin_size,
    #                           measurement_time, diff_window)
    # expected = background_count_rate**2 * measurement_time * bin_size * 1e-9 / 4
    # print('expected: {}'.format(expected))
    
    sim_background_nv(background_count_rate, nv_count_rate, 
                      bin_size, measurement_time, diff_window)
    expected = background_count_rate* nv_count_rate * measurement_time * bin_size * 1e-9 / 2
    print('expected: {}'.format(expected))
    