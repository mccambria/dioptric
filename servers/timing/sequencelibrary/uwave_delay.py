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
import utils.tool_belt as tool_belt
from utils.tool_belt import States


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
    durations = args[0:7]
    durations = [numpy.int64(el) for el in durations]
    tau, max_tau, readout, pi_pulse, aom_delay_time, polarization, wait_time = durations
    
    sig_gen = args[7]
    apd_index = args[8]
    laser_name = args[9]
    laser_power = args[10]
    
    pulser_do_apd_gate = pulser_wiring['do_apd_{}_gate'.format(apd_index)]
    pulser_do_sig_gen_gate = pulser_wiring['do_{}_gate'.format(sig_gen)]
    
    # Include a buffer on the front end so that we can delay channels that
    # should start off the sequence HIGH
    front_buffer = max(max_tau, aom_delay_time)
    
    period = front_buffer + polarization + wait_time + polarization

    seq = Sequence()
    
    # The readout windows are what everything else is relative to so keep
    # those fixed
    train = [(front_buffer, LOW),
             (polarization + wait_time - readout, LOW), 
             (readout, HIGH),
             (polarization + wait_time - readout, LOW),
             (readout, HIGH),
             ]
    seq.setDigital(pulser_do_apd_gate, train)

    if laser_power == -1:
        laser_high = HIGH
        laser_low = LOW
    else:
        laser_high = laser_power
        laser_low = 0.0
    train = [(front_buffer - aom_delay_time, laser_low),
             (polarization, laser_high), 
             (wait_time, laser_low), 
             (polarization, laser_high),
             (aom_delay_time, laser_low), 
             ]
    train.extend([(wait_time, laser_low), (polarization - aom_delay_time, laser_high)])
    if laser_power == -1:
        pulser_laser_mod = pulser_wiring['do_{}_dm'.format(laser_name)]
        seq.setDigital(pulser_laser_mod, train)
    else:
        pulser_laser_mod = pulser_wiring['ao_{}_am'.format(laser_name)]
        seq.setAnalog(pulser_laser_mod, train)
    
    # Vary the position of the rf pi pulse
    train = [(front_buffer - tau, LOW),
             (pi_pulse, HIGH),
             (period - (front_buffer - tau) - pi_pulse, LOW),
             ]
    seq.setDigital(pulser_do_sig_gen_gate, train)

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
    pulser_wiring = {'do_apd_0_gate': 0, 'do_laser_515_dm': 1, 'do_sample_clock': 2,
                     'do_signal_generator_sg394_gate': 3, 
                     'do_signal_generator_tsg4104a_gate': 4}

    # Set up a dummy args list
    args = [0, 1500, 350, 81, 90, 1100, 1000, 'signal_generator_sg394', 0, 'laser_515', -1]

    # get_seq returns the sequence and an arbitrary list to pass back to the
    # client. We just want the sequence.
    seq = get_seq(pulser_wiring, args)[0]

    # Plot the sequence
    seq.plot()
