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


# %% Constants


LOW = 0
HIGH = 1


# %% Functions


def constant(digital_channels, analog_0_voltage, analog_1_voltage):
    
    with labrad.connect() as cxn:
        pulser = cxn.pulse_streamer
        pulser.constant(digital_channels,
                                    analog_0_voltage, analog_1_voltage)
#        val = 0.0
#        cxn.pulse_streamer.constant([], 0.0, 0.0)
#        cxn.pulse_streamer.constant([], val, 0.0)
#        cxn.pulse_streamer.constant([7], 0.0, val)
#        cxn.pulse_streamer.constant([], val, val)

        input('Press enter to stop...')
        
        pulser.constant([], 0.0, 0.0)


# %% Main


def main():
    """When you run the file, we'll call into main, which should contain the
    body of the script.
    """
    
#    period = numpy.int64(100)
#    period = numpy.int64(300)
    period = numpy.int64(10**3)
#    period = numpy.int64(10**5)
#    period = numpy.int64(10**9)
    half_period = period // 2
    
    seq = Sequence()

#    train = [(half_period, HIGH), (half_period, HIGH)]
    train = [(half_period, HIGH), (half_period, LOW)]
    seq.setDigital(3, train)
    
    pulser = Pulser('128.104.160.113')
    pulser.constant(OutputState([]))
    pulser.setTrigger(start=TriggerStart.SOFTWARE)
    pulser.stream(seq, Pulser.REPEAT_INFINITELY)
    pulser.startNow()
    
    input('Press enter to stop...')
    
    pulser.constant(OutputState([]))


# %% Run the file


# The __name__ variable will only be '__main__' if you run this file directly.
# This allows a file's functions, classes, etc to be imported without running
# the script that you set up here.
if __name__ == '__main__':

    # Set up your parameters to be passed to main here

    # Run the script
    # main()
    
    with labrad.connect() as cxn:
        tool_belt.set_xyz(cxn, [0.0, 0.0, 0])
        cxn.filter_slider_ell9k.set_filter('nd_0')
    constant([3], 0.0, 0.0)
