# -*- coding: utf-8 -*-
"""
g2 measurement using 2 channels on a Time Tagger

Created on Thu Mar 28 13:48:11 2019

@author: mccambria
"""

# %% Imports


# User modules
import Outputs.xyz as xyz
import MajorRoutines.find_nv_center as find_nv_center
import Utils.tool_belt as tool_belt

# Library modules
import TimeTagger
import numpy
import matplotlib.pyplot as plt


# %% Main


def main(time_tagger_serial, pulser_ip, daq_name,
         daq_ao_galvo_x, daq_ao_galvo_y, piezo_serial,
         tagger_di_apd_0, tagger_di_apd_1,
         name, x_center, y_center, z_center,
         bin_width, num_bins):
    """Entry point for the routine

    Params:
    """

    xyz.write_daq(daq_name, daq_ao_galvo_x, daq_ao_galvo_y, piezo_serial,
                  x_center, y_center, z_center)

    tagger = TimeTagger.createTimeTagger()

    collect_raw = False
    plot_normalized = True

    if collect_raw:
        pass
    else:
        # Total measurement time = binwidth * n_bins
        g_two = TimeTagger.Correlation(tagger, channel_1=0, channel_2=1,
                                       binwidth=bin_width, n_bins=num_bins)

        fig, ax = plt.subplots()
        x_vals = g_two.getIndex() * (bin_width / 10**3)  # In ns
        raw_correlation = g_two.getData()
        normalized_correlation = g_two.getDataNormalized()
        if plot_normalized:
            y_vals = normalized_correlation
        else:
            y_vals = raw_correlation
        ax.plot(x_vals, y_vals)
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Normalized counts')
        ax.set_title('$g^{(2)}(\tau)')
        fig.tight_layout()
        # Draw the canvas and flush the events to the backend
        fig.canvas.draw()
        fig.canvas.flush_events()

    # %% Save the data

    timeStamp = tool_belt.get_time_stamp()

    rawData = {"timeStamp": timeStamp,
               "name": name,
               "xyzCenters": [x_center, y_center, z_center],
               "bin_width": bin_width,
               "num_bins": num_bins,
               "raw_correlation": raw_correlation.astype(int).tolist(),
               "normalized_correlation": normalized_correlation.tolist()}

    filePath = tool_belt.get_file_path("correlation", timeStamp, name)
    tool_belt.save_figure(fig, filePath)
    tool_belt.save_raw_data(rawData, filePath)
