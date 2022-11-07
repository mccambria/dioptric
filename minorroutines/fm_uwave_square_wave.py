# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 12:47:05 2022

@author: agardill
"""


# %% Imports


from pulsestreamer import PulseStreamer as Pulser
from pulsestreamer import TriggerStart
from pulsestreamer import OutputState
import numpy
from pulsestreamer import Sequence
import labrad
import utils.tool_belt as tool_belt
from utils.tool_belt import States
import time


# %% Constants


LOW = 0
HIGH = 1


# %% Functions



# %% Main


def main(cxn, state, deviation = 4, uwave_freq = 2.87, uwave_power = 0):
    """Run a laser on on a square wave."""

    seq_file = 'fm_square_wave.py'
    period = int(2e6)
    config = tool_belt.get_config_dict(cxn)
    sig_gen_name = config['Microwaves']['sig_gen_{}'.format(state.name)]
    seq_args = [period, sig_gen_name]


    seq_args_string = tool_belt.encode_seq_args(seq_args)
    
    sig_gen_cxn = tool_belt.get_signal_generator_cxn(cxn, state)
    sig_gen_cxn.set_freq(uwave_freq)
    sig_gen_cxn.set_amp(uwave_power)
    sig_gen_cxn.load_fm(deviation)
    sig_gen_cxn.uwave_on()
        
    cxn.pulse_streamer.stream_immediate(seq_file, -1, seq_args_string)

    input('Press enter to stop...')
    
    sig_gen_cxn.uwave_off()
    sig_gen_cxn.mod_off()
    sig_gen_cxn.mod_off()




# %% Run the file


# The __name__ variable will only be '__main__' if you run this file directly.
# This allows a file's functions, classes, etc to be imported without running
# the script that you set up here.
if __name__ == '__main__':

    # Set up your parameters to be passed to main here

    state = States.HIGH
    
    with labrad.connect() as cxn:
        main(cxn, state,)
        
        sig_gen_cxn = tool_belt.get_signal_generator_cxn(cxn, state)
        sig_gen_cxn.uwave_off()
        sig_gen_cxn.mod_off()
        cxn.pulse_streamer.constant()
    

        input('Press enter to stop...')

