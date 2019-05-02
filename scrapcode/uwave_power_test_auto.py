# -*- coding: utf-8 -*-
"""
Test script for measuring microwave power at various settings

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
    set_power = 5.0
    freq_center = 2.87
    freq_range = 0.3
    resolution = 0.005

    # Calculate the frequencies we need to set
    half_freq_range = freq_range / 2
    freq_low = freq_center - half_freq_range
    freq_high = freq_center + half_freq_range
    set_freqs = numpy.arange(freq_low, freq_high, resolution)
    num_steps = len(set_freqs)

    # Initialize arrays
    meas_freqs = numpy.empty(num_steps)
    meas_freqs[:] = numpy.nan
    meas_powers = numpy.copy(meas_freqs)

    with labrad.connect() as cxn:

        for ind in range(num_steps):

            freq = set_freqs[ind]
            cxn.microwave_signal_generator.set_freq(freq)

            if ind == 0:
                cxn.microwave_signal_generator.set_amp(set_power)
                cxn.microwave_signal_generator.uwave_on()
                cxn.pulse_streamer.constant(2)

            time.sleep(0.1)
            ret_vals = cxn.spectrum_analyzer.measure_peak()
            meas_freqs[ind], meas_powers[ind] = ret_vals

        cxn.pulse_streamer.constant(0)
        cxn.microwave_signal_generator.uwave_off()

    fig, ax = plt.subplots()
    ax.plot(set_freqs, meas_powers)
    ax.set_title('Measured Power Versus Frequency')
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('Measured Power (dBm)')
    fig.tight_layout()

    timestamp = tool_belt.get_time_stamp()
    raw_data = {'timestamp': timestamp,
                'metadata': metadata,
                'set_power': set_power,
                'set_freqs': set_freqs.tolist(),
                'set_freqs-units': 'GHz',
                'meas_freqs': meas_freqs.tolist(),
                'meas_freqs-units': 'GHz',
                'meas_powers': meas_powers.tolist(),
                'meas_powers-units': 'dBm'}

    file_path = tool_belt.get_file_path(__file__, timestamp)
    tool_belt.save_figure(fig, file_path)
    tool_belt.save_raw_data(raw_data, file_path)


def scan_power():

    metadata = ''
    set_freq = 5.0
    power_low = -11.0
    power_high = 11.0
    resolution = 0.5

    # Calculate the powers we need to set
    set_powers = numpy.arange(power_low, power_high, resolution)
    num_steps = len(set_powers)

    # Initialize arrays
    meas_freqs = numpy.empty(num_steps)
    meas_freqs[:] = numpy.nan
    meas_powers = numpy.copy(meas_freqs)

    with labrad.connect() as cxn:

        for ind in range(num_steps):

            power = set_powers[ind]
            cxn.microwave_signal_generator.set_amp(power)

            if ind == 0:
                cxn.microwave_signal_generator.set_freq(set_freq)
                cxn.microwave_signal_generator.uwave_on()
                cxn.pulse_streamer.constant(2)

            time.sleep(0.1)
            ret_vals = cxn.spectrum_analyzer.measure_peak()
            meas_freqs[ind], meas_powers[ind] = ret_vals

        cxn.pulse_streamer.constant(0)
        cxn.microwave_signal_generator.uwave_off()

    fig, ax = plt.subplots()
    ax.plot(set_powers, meas_powers)
    ax.set_title('Measured Power Versus Set Power')
    ax.set_xlabel('Set Power (dBm)')
    ax.set_ylabel('Measured Power (dBm)')
    fig.tight_layout()

    timestamp = tool_belt.get_time_stamp()
    raw_data = {'timestamp': timestamp,
                'metadata': metadata,
                'set_powers': set_powers.tolist(),
                'set_powers-units': 'dBm',
                'meas_freqs': meas_freqs.tolist(),
                'meas_freqs-units': 'GHz',
                'meas_powers': meas_powers.tolist(),
                'meas_powers-units': 'dBm'}

    file_path = tool_belt.get_file_path(__file__, timestamp)
    tool_belt.save_figure(fig, file_path)
    tool_belt.save_raw_data(raw_data, file_path)


if __name__ == '__main__':
    scan_freq()
    # scan_power()