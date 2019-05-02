# -*- coding: utf-8 -*-
"""
Test script for setting microwave power/frequency

Created on Wed May  1 13:52:35 2019

@author: mccambria
"""

import utils.tool_belt as tool_belt
import labrad
import numpy
import matplotlib.pyplot as plt
import time

def scan_freq():

    metadata = ''
    power = 5.0
    freq_center = 2.87
    freq_range = 0.3
    num_steps = 31

    # Calculate the frequencies we need to set
    half_freq_range = freq_range / 2
    freq_low = freq_center - half_freq_range
    freq_high = freq_center + half_freq_range
    set_freqs = numpy.linspace(freq_low, freq_high, num_steps)

    # Initialize arrays
    meas_freqs = numpy.empty(num_steps)
    meas_freqs[:] = numpy.nan
    meas_powers = numpy.copy(meas_freqs)

    with labrad.connect() as cxn:

        for ind in range(len(set_freqs)):

            freq = set_freqs[ind]
            cxn.microwave_signal_generator.set_freq(freq)

            if ind == 0:
                cxn.microwave_signal_generator.set_amp(power)
                cxn.microwave_signal_generator.uwave_on()
                cxn.pulse_streamer.constant(2)

            time.sleep(0.1)
            ret_vals = cxn.spectrum_analyzer.measure_peak()
            meas_freqs[ind], meas_powers[ind] = ret_vals

        cxn.pulse_streamer.constant(0)
        cxn.microwave_signal_generator.uwave_off()

    fig, ax = plt.subplots()
    ax.plot(set_freqs, meas_powers)
    ax.set_title('Power Versus Frequency')
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('Power (dBm)')
    fig.tight_layout()

    timestamp = tool_belt.get_time_stamp()
    raw_data = {'timestamp': timestamp,
               'metadata': metadata,
               'set_freqs': set_freqs.tolist(),
               'meas_freqs': set_freqs.tolist(),
               'meas_powers': set_freqs.tolist(),}

    file_path = tool_belt.get_file_path(__file__, timestamp)
    tool_belt.save_figure(fig, file_path)
    tool_belt.save_raw_data(raw_data, file_path)


def scan_power():
    with labrad.connect() as cxn:
        cxn.microwave_signal_generator.set_freq(2.87)
        cxn.microwave_signal_generator.set_amp(11.0)
        cxn.microwave_signal_generator.uwave_on()
        cxn.pulse_streamer.constant(2)

#        freqs = numpy.linspace(2.72, 3.02, 31)
#
#        for freq in freqs:
#            cxn.microwave_signal_generator.set_freq(freq)
#            print(freq)
#            if input('Press enter to continue...') == 'stop':
#                break

        while True:
            freq = input('Enter a frequency or nothing to stop: ')

            if freq != '':
                cxn.microwave_signal_generator.set_freq(freq)
            else:
                break

        cxn.pulse_streamer.constant(0)
        cxn.microwave_signal_generator.uwave_off()

def plot_data():
    # input power vs output power
    x_vals = numpy.linspace(-10.0, 11.0, 22)
    y_vals = [-0.2, 0.8, 1.8, 2.8, 3.7, 4.6, 5.4, 6.1, 6.8, 7.4,
              8.0, 8.3, 8.7, 9.0, 9.3, 9.5, 9.7, 9.8, 9.9, 10.0, 10.0, 10.1]
    plt.plot(x_vals, y_vals)


if __name__ == '__main__':
    # check_power()
#    check_freq()
    plot_data()