# -*- coding: utf-8 -*-
"""
Created on Wed Jun 5 14:49:23 2019

@author: gardill
"""

from pulsestreamer import Sequence
from pulsestreamer import OutputState
import utils.tool_belt as tool_belt
from utils.tool_belt import States
import numpy

def get_seq(pulse_streamer, config, args):

    # %% Parse wiring and args

    # The first 9 args are ns durations and we need them as int64s
    durations = []
    for ind in range(3):
        durations.append(numpy.int64(args[ind]))
    
    # Unpack the durations
    uwave_on_time, uwave_off_time, iq_mod_delay = durations
    
    
    period = uwave_on_time + uwave_on_time
    
    buffer = max(100, iq_mod_delay)
    # Digital
    LOW = 0
    HIGH = 1

    state = args[3]
    state = States(state)
    pulser_wiring = config['Wiring']['PulseStreamer']
    sig_gen_name = config['Microwaves']['sig_gen_{}'.format(state.name)]
    sig_gen_gate_chan_name = 'do_{}_gate'.format(sig_gen_name)
    pulser_do_sig_gen_gate = pulser_wiring[sig_gen_gate_chan_name]
    pulser_do_arb_wave_trigger = pulser_wiring['do_arb_wave_trigger']
    
    
    
    seq = Sequence()
    
    
    train = [(buffer - iq_mod_delay + uwave_off_time, LOW), (uwave_on_time, HIGH),
             (uwave_off_time, LOW), (uwave_on_time, HIGH), (iq_mod_delay, LOW)]
    seq.setDigital(pulser_do_sig_gen_gate, train)

    train = [(buffer + uwave_off_time, LOW), 
             (10, HIGH), 
             (uwave_on_time - 10 + uwave_off_time, LOW),
             (10, HIGH), (uwave_on_time - 10 , LOW)]
    seq.setDigital(pulser_do_arb_wave_trigger, train)
    
    final_digital = [0]
    final = OutputState(final_digital, 0.0, 0.0)
    
    return seq, final, [period]

    # %% Define the sequence


if __name__ == '__main__':
    config = tool_belt.get_config_dict()
    args = [200, 200, 0, 3]
    seq = get_seq(None, config, args)[0]
    seq.plot()