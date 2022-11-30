# -*- coding: utf-8 -*-
"""
Created on Wed Jun 5 14:49:23 2019

@author: gardill
"""

from pulsestreamer import Sequence
from pulsestreamer import OutputState
import utils.tool_belt as tool_belt
from utils.tool_belt import Digital
import numpy


def get_seq(pulse_streamer, config, args):

    # %% Parse wiring and args

    # The first 9 args are ns durations and we need them as int64s
    durations = []
    for ind in range(2):
        durations.append(numpy.int64(args[ind]))

    # Unpack the durations
    uwave_on_time, uwave_off_time = durations
    period = uwave_on_time + uwave_on_time
    pulser_wiring = config["Wiring"]["PulseStreamer"]
    sig_gen_name = args[2]
    sig_gen_gate_chan_name = "do_{}_gate".format(sig_gen_name)
    pulser_do_sig_gen_gate = pulser_wiring[sig_gen_gate_chan_name]

    seq = Sequence()

    train = [(uwave_off_time, Digital.LOW), (uwave_on_time, Digital.HIGH)]
    seq.setDigital(pulser_do_sig_gen_gate, train)

    final_digital = [0]
    final = OutputState(final_digital, 0.0, 0.0)

    return seq, final, [period]

    # %% Define the sequence


if __name__ == "__main__":
    config = tool_belt.get_config_dict()
    args = [100, 100, 1]
    seq = get_seq(None, config, args)[0]
    seq.plot()
