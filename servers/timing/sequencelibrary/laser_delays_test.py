# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 11:33:58 2020

File to stagger the three lasers to test if the delay times are accurate.

Green shines for 1 us, foloowed by 100 ns break, then yellow for 1 us, followed
100 ns break, then red for 1 us.

@author: agardil
"""


# %% Imports


from pulsestreamer import Sequence
from pulsestreamer import OutputState
import numpy


# %% Constants


LOW = 0


# %% Functions


# %% Sequence definition


def get_seq(pulser_wiring, args):
    """This is called by the pulse_streamer server to get the sequence object
    based on the wiring (from the registry) and the args passed by the client.
    """
    
    durations = args[0:3]
    durations = [numpy.int64(el) for el in durations]
    laser_532_delay, aom_589_delay, laser_638_delay = durations[0:3]
  
    apd_index = args[3]
    do_apd_gate = pulser_wiring['do_apd_{}_gate'.format(apd_index)]
    do_532 = pulser_wiring['do_532_aom']
    ao_589 = pulser_wiring['ao_589_aom']
    ao_638 = pulser_wiring['ao_638_laser']
#        HIGH = 0.7
    
    illumination = 10**3
    readout = 3 * illumination + 200
    period = readout + 100

    seq = Sequence()
    
    total_laser_delay = laser_532_delay + aom_589_delay + laser_638_delay
    # Readout over 3 x the illumination times
    train = [(total_laser_delay, LOW), (readout, 1), (100, LOW)]
    seq.setDigital(do_apd_gate, train)
    
#   # Green laser turns on first
#    train = [(aom_589_delay + laser_638_delay, LOW), (illumination, 1.0), (laser_532_delay, LOW)]
#    seq.setDigital(do_532, train)
#    # Yellow laser turns on next, 100 ns after green
#    train = [(laser_532_delay + laser_638_delay + illumination + 100, LOW), (illumination, 1.0), (aom_589_delay, LOW)]
#    seq.setAnalog(ao_589, train)
#    # Red laser turns on next, 100 ns after Yellow
#    train = [(laser_532_delay + aom_589_delay + 2*illumination + 200, LOW), (illumination, 0.76), (laser_638_delay, LOW)]
#    seq.setAnalog(ao_638, train)
    
    # Red laser
    train = [(aom_589_delay + laser_532_delay, LOW), (illumination, 0.7), (laser_638_delay, LOW)]
    seq.setAnalog(ao_638, train)
    # Yellow laser
    train = [(laser_532_delay + laser_638_delay + illumination + 100, LOW), (illumination, 1.0), (aom_589_delay, LOW)]
    seq.setAnalog(ao_589, train)
    # Green laser
    train = [(laser_638_delay + aom_589_delay + 2*illumination + 200, LOW), (illumination, 1), (laser_532_delay, LOW)]
    seq.setDigital(do_532, train)


    final_digital = [pulser_wiring['do_sample_clock']]
    final = OutputState(final_digital, 0.0, 0.0)
    return seq, final, [period]


# %% Run the file


# The __name__ variable will only be '__main__' if you run this file directly.
# This allows a file's functions, classes, etc to be imported without running
# the script that you set up here.
if __name__ == '__main__':

    # The whole point of defining sequences in their own files is so that we
    # can run the file to plot the sequence for debugging and analysis. We'll
    # go through that here.

    # Set up a dummy pulser wiring dictionary
    pulser_wiring = {'do_apd_0_gate': 0, 'do_532_aom': 1, 'do_sample_clock': 2,'ao_589_aom': 1, 'ao_638_laser': 0}

    # Set up a dummy args list
    args = [0,0,0, 0]

    # get_seq returns the sequence and an arbitrary list to pass back to the
    # client. We just want the sequence.
    seq = get_seq(pulser_wiring, args)[0]

    # Plot the sequence
    seq.plot()
