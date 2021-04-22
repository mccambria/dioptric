# -*- coding: utf-8 -*-
"""Template for Pulse Streamer sequences. If you don't follow this template,
the pulse_streamer server won't be able to read your sequence."

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


# %% Functions


# %% Sequence definition


def get_seq(pulser_wiring, args):
    """This is called by the pulse_streamer server to get the sequence object
    based on the wiring (from the registry) and the args passed by the client.
    """
    
    durations = args[0:4]
    durations = [numpy.int64(el) for el in durations]
    tau, max_tau, readout, laser_delay = durations[0:4]
    
    apd_index = args[4]
    am_power = args[5]
    color_ind = args[6]
    
    do_apd_gate = pulser_wiring['do_apd_{}_gate'.format(apd_index)]
    if color_ind == 532:
        do_aom = pulser_wiring['do_532_aom']
        HIGH = 1
    elif color_ind == '515a':
        do_aom = pulser_wiring['ao_515_laser']
        HIGH = am_power
    elif color_ind == 589:
        do_aom = pulser_wiring['ao_589_aom']
        HIGH = am_power
    elif color_ind == 638:
#        do_aom = pulser_wiring['ao_638_laser']
#        HIGH = 0.8
        do_aom = pulser_wiring['do_638_laser']
        HIGH = 1
        
    illumination = 10**4
    inter_time = 10**3
    period = 2 * illumination

    seq = Sequence()

    # The readout window is what moved during the delay. scan end of ilumination
    train = [(laser_delay + illumination - 500 + tau, LOW), (readout, 1), (max_tau - tau + inter_time + int(0.5*illumination) , LOW)]
    train.extend([(readout, 1), (100,LOW)])
    seq.setDigital(do_apd_gate, train)

#    # Keep the illumination fixed, starting at 500 ns.
#    train = [(500, LOW), (illumination, HIGH), (inter_time, LOW), (illumination, HIGH), (100, LOW)]
    
    # Keep the illumination fixed, allow the readout to scan over the end of the illumination
    train = [(illumination, HIGH), (inter_time, LOW), (illumination, HIGH), (100, LOW)]
    
    
    if color_ind == 532 or color_ind == 638:
        seq.setDigital(do_aom, train)
    else:
        seq.setAnalog(do_aom, train)

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
