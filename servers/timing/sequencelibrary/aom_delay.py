# -*- coding: utf-8 -*-
"""
Template for Pulse Streamer sequences. If you don't follow this template,
the pulse_streamer server won't be able to read your sequence.

Determine the delay from lasers by illuminating an NV and sweeping the 
readout pulse over the end of the laser pulse. 

Created on Sun Jun 16 11:22:40 2019

@author: mccambria
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
    
    durations = args[0:3]
    durations = [numpy.int64(el) for el in durations]
    tau, max_tau, readout = durations

    apd_index = args[2]
    laser_name = args[3]
    laser_power = args[4]
    
    do_apd_gate = pulser_wiring['do_apd_{}_gate'.format(apd_index)]
    pulser_do_daq_clock = pulser_wiring['do_sample_clock']
        
    illumination = 10**4
    half_illumination = illumination // 2
    readout = illumination // 10
    inter_time = 10**3
    period = max_tau + (2 * illumination) + inter_time
    
    seq = Sequence()

    # Keep the signal readout fixed at the back end of the illumination
    # to account for transients and because the laser should lag the readout.
    # Sweep the illumination delay. Place the second
    # readout (reference) squarely in the middle of the illumination so its 
    # independent of the actual delay.
    train = [(max_tau, LOW),
             (illumination-readout, LOW), (readout, HIGH), 
             (inter_time, LOW),
             (half_illumination, LOW),
             (readout, HIGH),
             (half_illumination-readout, LOW),
             ]
    seq.setDigital(do_apd_gate, train)

    if laser_power == -1:
        laser_high = HIGH
        laser_low = LOW
    else:
        laser_high = laser_power
        laser_low = 0.0
    train = [(max_tau-tau, laser_low), 
             (illumination, laser_high), 
             (inter_time+tau, laser_low), 
             (illumination, laser_high),
             ]
    if laser_power == -1:
        pulser_laser_mod = pulser_wiring['do_{}_dm'.format(laser_name)]
        seq.setDigital(pulser_laser_mod, train)
    else:
        pulser_laser_mod = pulser_wiring['ao_{}_am'.format(laser_name)]
        seq.setAnalog(pulser_laser_mod, train)
    
    final_digital = [pulser_do_daq_clock]
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
    pulser_wiring = {'do_apd_0_gate': 0, 
                     'do_532_aom': 1, 
                     'do_sample_clock': 2,
                     'ao_515_laser': 0,
                     'ao_589_aom': 1, 
                     'do_638_laser': 4}

    # Set up a dummy args list
    args = [ 500, 1500, 200, 0, 0, 1.0, 589]
    args = [1573.3333333333335, 2000, 200, 0, 0, 0.65, '515a']

    # get_seq returns the sequence and an arbitrary list to pass back to the
    # client. We just want the sequence.
    seq = get_seq(pulser_wiring, args)[0]

    # Plot the sequence
    seq.plot()
