#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 20:40:44 2020
4/8/20 includes initial red ionization pulse

@author: agardill
"""

from pulsestreamer import Sequence
from pulsestreamer import OutputState
import utils.tool_belt as tool_belt
import numpy

LOW = 0
HIGH = 1

def get_seq(pulse_streamer, config, args):

    # Unpack the args
    readout_time, init_ion_pulse_time, reion_time, ion_time, \
            wait_time, laser_515_delay, aom_589_delay, laser_638_delay, \
            apd_indices, aom_ao_589_pwr, \
            reion_laser, readout_laser, ion_laser = args  # green, yellow, red

    # We're in analog yellow mode if the wiring says yellow is wired to an
    # analog output
    pulser_wiring = config['Wiring']['PulseStreamer']
    analog_yellow_key = 'ao_{}_am'.format(readout_laser)
    analog_yellow = (analog_yellow_key in pulser_wiring)
    
    readout_time = numpy.int64(readout_time)
    init_ion_pulse_time = numpy.int64(init_ion_pulse_time)
    ion_time = numpy.int64(ion_time)
    reion_time = numpy.int64(reion_time)
    
    total_laser_delay = laser_515_delay + aom_589_delay + laser_638_delay
    # Test period
    period =  total_laser_delay + (init_ion_pulse_time + reion_time + ion_time + \
                           readout_time + 3 * wait_time)*2
    
    # Get what we need out of the wiring dictionary
    pulser_do_apd_gate = pulser_wiring['do_apd_{}_gate'.format(apd_indices)]
    pulser_do_clock = pulser_wiring['do_sample_clock']
    pulser_do_532_aom = pulser_wiring['do_{}_dm'.format(reion_laser)]
    if analog_yellow:
        pulser_589_chan = pulser_wiring[analog_yellow_key]
    else:
        digital_yellow_key = 'do_{}_dm'.format(readout_laser)
        pulser_589_chan = pulser_wiring[digital_yellow_key]
    pulser_do_638_aom = pulser_wiring['do_{}_dm'.format(ion_laser)]


    # Make sure the ao_aom voltage to the 589 aom is within 0 and 1 V
    if analog_yellow:
        tool_belt.aom_ao_589_pwr_err(aom_ao_589_pwr)
    
    seq = Sequence()


    #collect photons for certain timewindow tR in APD
    train = [(total_laser_delay + init_ion_pulse_time + reion_time + ion_time +  3*wait_time, LOW), 
             (readout_time, HIGH),
             (init_ion_pulse_time + reion_time + ion_time + 4*wait_time, LOW), 
             (readout_time, HIGH), (wait_time, LOW)]
    seq.setDigital(pulser_do_apd_gate, train)

    # green pulse 
    train = [(laser_638_delay + aom_589_delay + init_ion_pulse_time + wait_time, LOW),
             (reion_time, HIGH), 
             (4*wait_time + ion_time + readout_time + init_ion_pulse_time, LOW), 
             (reion_time, HIGH), 
             (3*wait_time + ion_time + readout_time + laser_515_delay, LOW)]
    seq.setDigital(pulser_do_532_aom, train)

    # red pulse 
    train = [(aom_589_delay + laser_515_delay, LOW), (init_ion_pulse_time, HIGH),
             (reion_time + 2*wait_time, LOW), 
             (ion_time, HIGH), 
             (2*wait_time + readout_time, LOW), (init_ion_pulse_time, HIGH),
             (4*wait_time + reion_time + readout_time + ion_time + laser_638_delay, LOW)]       
    seq.setDigital(pulser_do_638_aom, train)
    
    # readout with 589
    yellow_high = aom_ao_589_pwr if analog_yellow else HIGH
    train = [(laser_515_delay + laser_638_delay + init_ion_pulse_time + reion_time + ion_time +  3*wait_time, LOW), 
             (readout_time, yellow_high),
             (init_ion_pulse_time + reion_time + ion_time + 4*wait_time, LOW), 
             (readout_time, yellow_high), (wait_time + aom_589_delay, LOW)]
    if analog_yellow:
        seq.setAnalog(pulser_589_chan, train) 
    else:
        seq.setDigital(pulser_589_chan, train) 
    
    final_digital = [pulser_do_clock]
    final = OutputState(final_digital, 0.0, 0.0)

    return seq, final, [period]


if __name__ == '__main__':
    # wiring = {'do_apd_0_gate': 1,
    #           'do_532_aom': 2,
    #           'sig_gen_gate_chan_name': 3,
    #            'do_sample_clock':4,
    #            'ao_589_aom': 0,
    #            'ao_638_laser': 1,
    #            'do_638_laser': 7             }
    config = tool_belt.get_config_dict()

    args = [600000, 25000, 10000, 1000, 1000, 0, 0, 0, 0, 1, 'integrated_520', 'laserglow_589', 'cobolt_638']
    seq, final, _ = get_seq(None, config, args)
    seq.plot()
    
    # readout_time, init_ion_pulse_time, reion_time, ion_time, \
    #         wait_time, laser_515_delay, aom_589_delay, laser_638_delay, \
    #         apd_indices, aom_ao_589_pwr, \
    #         reion_laser, readout_laser, ion_laser = args  # green, yellow, red