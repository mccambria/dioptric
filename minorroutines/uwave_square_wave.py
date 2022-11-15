# -*- coding: utf-8 -*-
"""Template for scripts that should be run directly from the files themselves
(as opposed to from the control panel, for example).

Created on Sun Jun 16 11:22:40 2019

@author: mccambria
"""


# %% Imports


from pulsestreamer import PulseStreamer as Pulser
from pulsestreamer import TriggerStart
from pulsestreamer import OutputState
import numpy as np
from pulsestreamer import Sequence
import labrad
import utils.tool_belt as tool_belt
from utils.tool_belt import States
import time


# %% Constants


LOW = 0
HIGH = 1


def sweep(cxn, uwave_freqs, uwave_power):
    
    tool_belt.init_safe_stop()
    
    sig_gen_cxn = cxn.signal_generator_sg394
    sig_gen_cxn.set_freq(uwave_freqs[0])
    sig_gen_cxn.set_amp(uwave_power)
    sig_gen_cxn.uwave_on()
    cxn.pulse_streamer.constant([7])
    
    time.sleep(2)
    
    for freq in uwave_freqs:
        if tool_belt.safe_stop():
            break
        sig_gen_cxn.set_freq(freq)
        time.sleep(0.03)
        
    cxn.pulse_streamer.constant()
    sig_gen_cxn.uwave_off()
    tool_belt.reset_cfm(cxn)


def constant(cxn, uwave_freq, uwave_power):

    sig_gen_cxn = cxn.signal_generator_sg394
    sig_gen_cxn.set_amp(uwave_power)
    sig_gen_cxn.set_freq(uwave_freq)
    sig_gen_cxn.uwave_on()

    cxn.pulse_streamer.constant([7])
    input('Press enter to stop...')
    cxn.pulse_streamer.constant()

    sig_gen_cxn.uwave_off()
    tool_belt.reset_cfm(cxn)

# %% Main


def main(cxn, uwave_freq, uwave_power, state = States.LOW):
    """When you run the file, we'll call into main, which should contain the
    body of the script.
    """


    sig_gen_cxn = tool_belt.get_signal_generator_cxn(cxn, state)
    sig_gen_cxn = cxn.signal_generator_sg394
    sig_gen_cxn.set_amp(uwave_power)
    sig_gen_cxn.set_freq(uwave_freq)
    # sig_gen_cxn.load_iq()
    # sig_gen_cxn.uwave_on()


    seq_file = 'uwave_square_wave.py'
    uwave_on = int(500)
    uwave_off = int(500)
    seq_args = [uwave_on,  uwave_off, state.value]
    print(seq_args)

    # seq_args_string = tool_belt.encode_seq_args(seq_args)

    # cxn.pulse_streamer.stream_immediate(seq_file, -1, seq_args_string)

    # input('Press enter to stop...')

    sig_gen_cxn.uwave_off()
    tool_belt.reset_cfm(cxn)



# %% Run the file


# The __name__ variable will only be '__main__' if you run this file directly.
# This allows a file's functions, classes, etc to be imported without running
# the script that you set up here.
if __name__ == '__main__':

    # Set up your parameters to be passed to main here

    with labrad.connect() as cxn:
        uwave_freq = 2.87 #GHz
        uwave_power = 16.5 #dBm
        # # main(cxn,uwave_freq, uwave_power, States.HIGH)
        constant(cxn,uwave_freq, uwave_power)
        half_range = 0.5
        uwave_freqs = np.linspace(uwave_freq - half_range, uwave_freq + half_range, 1000)
        # sweep(cxn, uwave_freqs, uwave_power)
