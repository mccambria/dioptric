# -*- coding: utf-8 -*-
"""
Scrap to test galvo and pulse streamer interactions

Created on Tue Dec 22 11:11:50 2020

@author: mccambria
"""

import numpy
import utils.tool_belt as tool_belt
import time
import sys
import json
import matplotlib.pyplot as plt
import labrad

nv_sig = { 'coords':[0.0, 0.0, 5],
            'name': 'test',
            'expected_count_rate': None, 'nd_filter': 'nd_0',
            'pulsed_readout_dur': 350, 'magnet_angle': 0.0,
            'resonance_LOW': None, 'rabi_LOW': None, 'uwave_power_LOW': 9.0,
            'resonance_HIGH': None, 'rabi_HIGH': None, 'uwave_power_HIGH': 10.0}

with labrad.connect() as cxn:
    
    # cxn.pulse_streamer.constant([])
    # sys.exit()
    
    x_range = 3.0
    y_range = x_range
    num_steps = 10
    
    readout = 0.5*10**9

    x_center, y_center, z_center = nv_sig['coords']

    # cxn.galvo.write(x_center, y_center)
    # sys.exit()

    # The galvo's small angle step response is 400 us
    # Let's give ourselves a buffer of 500 us (500000 ns)
    delay = 500000

    total_num_samples = num_steps**2
    # total_num_samples = 5*10**7

    # %% Load the PulseStreamer

    seq_args = [delay, readout, 0]
    # seq_args = [250, 500, 0]
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    ret_vals = cxn.pulse_streamer.stream_load('simple_readout.py',
                                              seq_args_string)
    period = ret_vals[0]

    # %% Initialize at the passed coordinates

    cxn.galvo.write(x_center, y_center)
    # sys.exit()

    # %% Set up the galvo

    x_voltages, y_voltages = cxn.galvo.load_sweep_scan(x_center, y_center,
                                                       x_range, y_range,
                                                       num_steps, period)
    
    cxn.pulse_streamer.stream_start(total_num_samples)
