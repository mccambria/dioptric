# -*- coding: utf-8 -*-
"""

Sequence for determining the delay beween the rf and the AOM

Created on Sun Aug 6 11:22:40 2019

@author: agardill
"""


# %% Imports


from pulsestreamer import Sequence
from pulsestreamer import OutputState
import numpy


# %% Constants


LOW = 0
HIGH = 1


# %% Functions


# %% Sequence definition


def get_seq(pulser_wiring, args):
    """This is called by the pulse_streamer server to get the sequence object
    based on the wiring (from the registry) and the args passed by the client.
    """
    
    # Extract the delay (that the rf is applied from the start of the seq),
    # the readout time, and the pi pulse duration
    durations = args[0:6]
    durations = [numpy.int64(el) for el in durations]
    delay, readout, pi_pulse, aom_delay_time, \
                                polarization, wait_time = durations[0:2]
    
    sig_gen = args[6]
    apd_index = args[7]
    
    pulser_do_apd_gate = pulser_wiring['do_apd_{}_gate'.format(apd_index)]
    pulser_do_aom = pulser_wiring['do_532_aom']
    pulser_do_sig_gen_gate = pulser_wiring['do_signal_generator_{}_gate'.format(sig_gen)]
    
    
    sequence = polarization + wait_time + polarization + wait_time + \
                        polarization
    period =  sequence + aom_delay_time

    seq = Sequence()
    
    # The readout windows are what everything else is relative to so keep
    # these fixed
    prep_time = aom_delay_time + polarization + wait_time
    train = [(prep_time, LOW), (readout, HIGH), \
             (polarization + wait_time - readout, LOW), (readout, HIGH), \
             (polarization - readout)]
    seq.setDigital(pulser_do_apd_gate, train)

    # Pulse sequence for the AOM
    train = [(polarization, HIGH), (wait_time, LOW), (polarization, HIGH), (wait_time, LOW), (polarization - aom_delay_time, HIGH)]
    seq.setDigital(pulser_do_aom, train)
    
    # Vary the position of the rf pi pulse
    train = [(aom_delay_time + delay, LOW)]
    train.extend([(pi_pulse, HIGH)])
    train.extend([(sequence - aom_delay_time - delay - pi_pulse, LOW)])
    seq.setDigital(pulser_do_sig_gen_gate, train)

    final_digital = [pulser_wiring['do_532_aom'],
                     pulser_wiring['do_sample_clock']]
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
    pulser_wiring = {'do_apd_0_gate': 0, 'do_532_aom': 1, 'do_sample_clock': 2,
                     'do_signal_generator_tsg4104a_gate': 3, 
                     'do_signal_generator_bnc835_gate': 4}

    # Set up a dummy args list
    args = [0, 2000, 110, 0, 1000, 1000, 'tsg4104a', 0]

    # get_seq returns the sequence and an arbitrary list to pass back to the
    # client. We just want the sequence.
    seq = get_seq(pulser_wiring, args)[0]

    # Plot the sequence
    seq.plot()
