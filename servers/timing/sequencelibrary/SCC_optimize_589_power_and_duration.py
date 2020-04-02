#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 1 20:40:44 2020

@author: agardill
"""

from pulsestreamer import Sequence
from pulsestreamer import OutputState
import utils.tool_belt as tool_belt
import numpy

LOW = 0
HIGH = 1

def get_seq(pulser_wiring, args):

    # Unpack the args
    readout_time, reionization_time, ionization_time,\
            wait_time, laser_515_delay, aom_589_delay, laser_638_delay, \
            apd_indices, aom_ao_589_pwr = args

    readout_time = numpy.int64(readout_time)
    reionization_time = numpy.int64(reionization_time)
    ionization_time = numpy.int64(ionization_time)
    
    total_laser_delay = laser_515_delay + aom_589_delay + laser_638_delay
    # Test period
    period =  total_laser_delay + (reionization_time + ionization_time + \
                           readout_time + 3 * wait_time)*2
    
    # Get what we need out of the wiring dictionary
    pulser_do_apd_gate = pulser_wiring['do_apd_{}_gate'.format(apd_indices)]
    pulser_do_clock = pulser_wiring['do_sample_clock']
    pulser_do_532_aom = pulser_wiring['do_532_aom']
    pulser_ao_589_aom = pulser_wiring['ao_589_aom']
    pulser_do_638_aom = pulser_wiring['do_638_laser']

    # Make sure the ao_aom voltage to the 589 aom is within 0 and 1 V
    tool_belt.aom_ao_589_pwr_err(aom_ao_589_pwr)
    
    seq = Sequence()


    #collect photons for certain timewindow tR in APD
    train = [(total_laser_delay + reionization_time + ionization_time +  \
              2*wait_time, LOW), (readout_time, HIGH),
             (reionization_time + ionization_time + 3*wait_time, LOW), 
             (readout_time, HIGH), (wait_time, LOW)]
    seq.setDigital(pulser_do_apd_gate, train)
    
#    #clock pulse
#    train = [(total_laser_delay + reionization_time + ionization_time + \
#              readout_time + 3*wait_time - clock_time, LOW), (clock_time, HIGH), 
#             (reionization_time + ionization_time + readout_time + \
#              3*wait_time - clock_time, LOW), (clock_time, HIGH), (100, LOW)]
#    seq.setDigital(pulser_do_clock, train)

    # reionization (green) pulse
    train = [(aom_589_delay + laser_638_delay, LOW), (reionization_time, HIGH), 
             (3*wait_time + ionization_time + readout_time, LOW), 
              (reionization_time, HIGH), (3*wait_time + ionization_time + \
              readout_time + laser_515_delay, LOW)]
    seq.setDigital(pulser_do_532_aom, train)
    
    # ionization (red) pulse
    train = [ (aom_589_delay + laser_515_delay + 2*reionization_time + \
               ionization_time + readout_time + 4*wait_time, LOW), 
             (ionization_time, HIGH), 
             (2*wait_time + readout_time + laser_638_delay, LOW)]
    seq.setDigital(pulser_do_638_aom, train)
    
    # readout with 589
    train = [(laser_515_delay + laser_638_delay + reionization_time + \
              ionization_time +  2*wait_time, LOW), 
              (readout_time, aom_ao_589_pwr),
              (reionization_time + ionization_time + 3*wait_time, LOW), 
              (readout_time, aom_ao_589_pwr), (wait_time + aom_589_delay, LOW)]
    seq.setAnalog(pulser_ao_589_aom, train) 
    

    
    final_digital = [pulser_do_clock]
    final = OutputState(final_digital, 0.0, 0.0)

    return seq, final, [period]


if __name__ == '__main__':
    wiring = {'do_apd_0_gate': 1,
              'do_532_aom': 2,
              'sig_gen_gate_chan_name': 3,
               'do_sample_clock':4,
               'ao_589_aom': 0,
               'ao_638_laser': 1,
               'do_638_laser': 7             }
#    args = [10000000, 1000000, 500, 1000, 0, 0, 0, 0, 0.7]
    args = [1000, 100, 10, 1000, 0, 0, 0, 0, 0.7]
    seq, final, _ = get_seq(wiring, args)
    seq.plot()