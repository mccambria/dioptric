# -*- coding: utf-8 -*-
"""
Variation of simple readout but with a constant high line connected to the camera trigger

Created on July 29th, 2023

@author: mccambria
"""

from pulsestreamer import Sequence
from pulsestreamer import OutputState
from utils import tool_belt as tb
from utils import common
import numpy as np

LOW = 0
HIGH = 1


def get_seq(pulse_streamer, config, args):
    # Unpack the args
    delay, readout_time, laser_name, laser_power = args

    # Get what we need out of the wiring dictionary
    pulse_gen_wiring = config["Wiring"]["PulseGen"]
    do_daq_clock = pulse_gen_wiring["do_sample_clock"]
    do_daq_gate = pulse_gen_wiring["do_apd_gate"]
    do_camera_trigger = pulse_gen_wiring["do_camera_trigger"]

    # Convert the 32 bit ints into 64 bit ints
    delay = np.int64(delay)
    readout_time = np.int64(readout_time)

    period = np.int64(delay + readout_time + 300)

    # tb.check_laser_power(laser_name, laser_power)

    # Define the sequence
    seq = Sequence()

    # The clock signal will be high for 100 ns with buffers of 100 ns on
    # either side. During the buffers, everything should be low. The buffers
    # account for any timing jitters/delays and ensure that everything we
    # expect to be on one side of the clock signal is indeed on that side.
    train = [(period - 200, LOW), (100, HIGH), (100, LOW)]
    seq.setDigital(do_daq_clock, train)

    train = [(delay, LOW), (readout_time, HIGH), (300, LOW)]
    tb.process_laser_seq(seq, laser_name, laser_power, train)

    train = [(period, HIGH)]
    seq.setDigital(do_camera_trigger, train)

    final_digital = []
    final = OutputState(final_digital, 0.0, 0.0)

    return seq, final, [period]


if __name__ == "__main__":
    config = common.get_config_dict()
    args = [1000, 3000.0, "laser_INTE_520", 0.0]
    # args = [5000, 10000.0, 1, 'integrated_520',None]
    #    seq_args_string = tool_belt.encode_seq_args(args)
    seq, ret_vals, period = get_seq(None, config, args)
    seq.plot()
