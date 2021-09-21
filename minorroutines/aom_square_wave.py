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
import time


# %% Constants


LOW = 0
HIGH = 1


# %% Functions


def constant(cxn, laser_name, laser_power=None):

    tool_belt.laser_on(cxn, laser_name, laser_power)

    input('Press enter to stop...')

    tool_belt.laser_off(cxn, laser_name)


# %% Main


def main(cxn, laser_name, laser_power=None):
    """When you run the file, we'll call into main, which should contain the
    body of the script.
    """

    seq_file = 'square_wave.py'
    period = int(1e6)
    # period = int(0.25e6)
    # period = int(1000)
    seq_args = [period, laser_name, laser_power]
    seq_args_string = tool_belt.encode_seq_args(seq_args)

    cxn.pulse_streamer.stream_immediate(seq_file, -1, seq_args_string)

    input('Press enter to stop...')

    tool_belt.laser_off(cxn, laser_name)


# %% Run the file


# The __name__ variable will only be '__main__' if you run this file directly.
# This allows a file's functions, classes, etc to be imported without running
# the script that you set up here.
if __name__ == '__main__':

    # Set up your parameters to be passed to main here

    # Rabi
    # laser_name = 'cobolt_515'
    # laser_name = 'cobolt_638'
    # filter_name = 'nd_0.5'
    # pos = [-0.025, -0.009, 4.89]

    # Hahn
    laser_name = 'integrated_520'
    # filter_name = 'nd_1.0'
    pos = [0.0, 0.0, 5.0]

    with labrad.connect() as cxn:
        # tool_belt.set_xyz(cxn, pos)
#        for el in laser_names:
        # tool_belt.set_filter(cxn, optics_name=laser_name, filter_name=filter_name)
        # tool_belt.set_filter(cxn, optics_name='collection', filter_name='630_lp')
        # constant(cxn, laser_name)
        main(cxn, laser_name)
