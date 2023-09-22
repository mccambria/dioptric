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
    # Get what we need out of the wiring dictionary
    pulse_gen_wiring = config["Wiring"]["PulseGen"]
    do_camera_trigger = pulse_gen_wiring["do_camera_trigger"]

    # Convert the 32 bit ints into 64 bit ints
    readout_time = np.int64(0.1e9)

    period = np.int64(readout_time)

    # tb.check_laser_power(laser_name, laser_power)

    # Define the sequence
    seq = Sequence()

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
