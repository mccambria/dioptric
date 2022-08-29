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
import numpy
from pulsestreamer import Sequence
import labrad
import utils.tool_belt as tool_belt
from utils.tool_belt import States
import time


# %% Constants


LOW = 0
HIGH = 1



# %% Main


def main(cxn, uwave_freq, uwave_power, state = States.LOW):
    """When you run the file, we'll call into main, which should contain the
    body of the script.
    
    unfortunately the SRS sig gen sg394 only handles modulation with frequencies
    above 400 MHz, and our oscilloscope has a bandwidth of 200 MHz...
    """
    
    iq_delay = 100
    
    tool_belt.reset_cfm(cxn)
    
    sig_gen_cxn = tool_belt.get_signal_generator_cxn(cxn, state)
    

    sig_gen_cxn.set_freq(uwave_freq)
    sig_gen_cxn.set_amp(uwave_power)
    sig_gen_cxn.load_iq()
    sig_gen_cxn.uwave_on()
    
    phases = [numpy.pi, numpy.pi ]
    cxn.arbitrary_waveform_generator.load_arb_phases(phases)
        
    seq_file = 'uwave_square_wave_iq_mod.py'
    uwave_on = int(200)
    uwave_off = int(200)
    seq_args = [uwave_on,  uwave_off,iq_delay, state.value]
    print(seq_args)
    
    seq_args_string = tool_belt.encode_seq_args(seq_args)

    cxn.pulse_streamer.stream_immediate(seq_file, -1, seq_args_string)

    input('Press enter to stop...')

    cxn.arbitrary_waveform_generator.wave_off()
    sig_gen_cxn.uwave_off()

    tool_belt.reset_cfm(cxn)
    

# %% Run the file


# The __name__ variable will only be '__main__' if you run this file directly.
# This allows a file's functions, classes, etc to be imported without running
# the script that you set up here.
if __name__ == '__main__':

    # Set up your parameters to be passed to main here

    with labrad.connect() as cxn:
        uwave_freq = 0.5 #GHz
        uwave_power = -2 #dBm
        main(cxn,uwave_freq, uwave_power,  States.HIGH)
