# -*- coding: utf-8 -*-
"""
Variation of simple readout but with a constant high line connected to the camera trigger

Created on July 29th, 2023

@author: mccambria
"""

import numpy as np
from pulsestreamer import OutputState, Sequence

from utils import common
from utils import tool_belt as tb

LOW = 0
HIGH = 1


# def get_seq(pulse_streamer, config, args):
#     # Unpack the args
#     delay, readout_time, laser_name, laser_power = args

#     # Get what we need out of the wiring dictionary
#     pulse_gen_wiring = config["Wiring"]["PulseGen"]
#     do_daq_clock = pulse_gen_wiring["do_sample_clock"]
#     do_daq_gate = pulse_gen_wiring["do_apd_gate"]
#     do_camera_trigger = pulse_gen_wiring["do_camera_trigger"]

#     # Convert the 32 bit ints into 64 bit ints
#     delay = np.int64(delay)
#     readout_time = np.int64(readout_time)

#     period = np.int64(delay + readout_time + 300)

#     # tb.check_laser_power(laser_name, laser_power)

#     # Define the sequence
#     seq = Sequence()

#     # The clock signal will be high for 100 ns with buffers of 100 ns on
#     # either side. During the buffers, everything should be low. The buffers
#     # account for any timing jitters/delays and ensure that everything we
#     # expect to be on one side of the clock signal is indeed on that side.
#     train = [(period - 200, LOW), (100, HIGH), (100, LOW)]
#     seq.setDigital(do_daq_clock, train)

#     train = [(delay, LOW), (readout_time, HIGH), (300, LOW)]
#     tb.process_laser_seq(seq, laser_name, laser_power, train)

#     train = [(period, HIGH)]
#     seq.setDigital(do_camera_trigger, train)

#     final_digital = []
#     final = OutputState(final_digital, 0.0, 0.0)

#     return seq, final, [period]


# if __name__ == "__main__":
#     config = common.get_config_dict()
#     args = [1000, 3000.0, "laser_INTE_520", 0.0]
#     # args = [5000, 10000.0, 1, 'integrated_520',None]
#     #    seq_args_string = tool_belt.encode_seq_args(args)
#     seq, ret_vals, period = get_seq(None, config, args)
#     seq.plot()

# -*- coding: utf-8 -*-
"""
Simple readout with NO laser control.
- DAQ sample clock: 100 ns HIGH, padded by 100 ns LOW on each side
- APD gate: HIGH only during readout (optional; comment out if you don't want it)
- Camera trigger: held HIGH for the entire period
"""

import numpy as np
from pulsestreamer import OutputState, Sequence

from utils import common

LOW = 0
HIGH = 1


def get_seq(pulse_streamer, config, args):
    # Unpack the args: delay (ns) before readout, readout_time (ns)
    delay, readout_time = args

    # Wiring
    pulse_gen_wiring = config["Wiring"]["PulseGen"]
    do_daq_clock = pulse_gen_wiring["do_sample_clock"]
    do_daq_gate = pulse_gen_wiring["do_apd_gate"]
    do_camera_trigger = pulse_gen_wiring["do_camera_trigger"]

    # Convert to 64-bit ints for Pulsestreamer
    delay = np.int64(delay)
    readout_time = np.int64(readout_time)

    # Padding after readout (ns)
    tail_pad = np.int64(300)
    period = np.int64(delay + readout_time + tail_pad)

    # Define the sequence
    seq = Sequence()

    # DAQ sample clock: 100 ns pulse with 100 ns LOW buffers
    clock_train = [(period - 200, LOW), (100, HIGH), (100, LOW)]
    seq.setDigital(do_daq_clock, clock_train)

    # APD gate (optional): HIGH only during readout window
    gate_train = [(delay, LOW), (readout_time, HIGH), (tail_pad, LOW)]
    seq.setDigital(do_daq_gate, gate_train)

    # Camera trigger: hold HIGH for full period
    cam_train = [(period, HIGH)]
    seq.setDigital(do_camera_trigger, cam_train)

    # Final digital state
    final = OutputState([], 0.0, 0.0)

    return seq, final, [period]


if __name__ == "__main__":
    config = common.get_config_dict()
    # e.g., 1 us delay, 3 us readout (all in ns)
    args = [1000, 3000]
    seq, ret_vals, period = get_seq(None, config, args)
    seq.plot()
