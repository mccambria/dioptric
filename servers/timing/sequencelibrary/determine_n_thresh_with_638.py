#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 20:40:44 2019

@author: yanfeili
"""
>>>Currently broken AG 4/2/2020<<<<

from pulsestreamer import Sequence
from pulsestreamer import OutputState
import utils.tool_belt as tool_belt
import numpy

LOW = 0
HIGH = 1

def get_seq(pulser_wiring, args):
    
    # Unpack the args
    readout_time, reionization_time, ionization_time, \
            wait_time, laser_515_delay, aom_589_delay, \
                laser_638_delay, apd_indices, aom_ao_589_pwr = args

    readout_time = numpy.int64(readout_time)
    ionization_time = numpy.int64(ionization_time)
    reionization_time = numpy.int64(reionization_time)

    # SCC photon collection test period
    total_laser_delay = laser_515_delay + aom_589_delay + laser_638_delay
    period =  total_laser_delay + (reionization_time + ionization_time + \
                           illumination_time + 4 * wait_time)*2
    
    # Get what we need out of the wiring dictionary
    pulser_do_apd_gate = pulser_wiring['do_apd_{}_gate'.format(apd_indices)]
    pulser_do_clock = pulser_wiring['do_sample_clock']
    pulser_do_532_aom = pulser_wiring['do_532_aom']
    pulser_ao_589_aom = pulser_wiring['ao_589_aom']
    pulser_ao_638_aom = pulser_wiring['ao_638_laser']

    # Make sure the ao_aom voltage to the 589 aom is within 0 and 1 V
    tool_belt.aom_ao_589_pwr_err(aom_ao_589_pwr)
    
    seq = Sequence()

    #collect photons for certain timewindow tR in APD
    train = [(total_laser_delay + reionization_time + ionization_time  + 3*wait_time, LOW), \
             (readout_time, HIGH), (readout_time + 100 + 2*wait_time, LOW),
             (reionization_time + ionization_time  + 2*wait_time, LOW), \
             (readout_time, HIGH), (readout_time + 100 + 2*wait_time, LOW)]
    seq.setDigital(pulser_do_apd_gate, train)
    
    #clock pulse
#    train = [(laser_delay + reionization_time + ionization_time + illumination_time + 3*wait_time, LOW), (100, HIGH), 
#             ( reionization_time + ionization_time + illumination_time + 4*wait_time, LOW), (100, HIGH), (wait_time, LOW)]
#    seq.setDigital(pulser_do_clock, train)

    #reionize with 532
    train = [ (reionization_time, HIGH), (int(period/2) -reionization_time, LOW), 
              (reionization_time, HIGH), (int(period/2) -reionization_time  + laser_delay , LOW)]
    seq.setDigital(pulser_do_532_aom, train)
    
    #readout with 589
    train = [(reionization_time + ionization_time +  2*wait_time, LOW), (illumination_time, aom_ao_589_pwr), (100 + 2*wait_time, LOW),
             (reionization_time + ionization_time +  2*wait_time, LOW), (illumination_time, aom_ao_589_pwr), (100 + 2*wait_time + laser_delay, LOW),]
    seq.setAnalog(pulser_ao_589_aom, train) 
    
    #ionize with 638 
    train = [(reionization_time+ wait_time, LOW), (ionization_time, ao_638_pwr), \
             (3*wait_time + illumination_time + laser_delay + 100 + int(period/2), LOW)]
    seq.setAnalog(pulser_ao_638_aom, train)
    
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
               'do_638_laser': 6              }

    args = [1000000, 1000000, 1001000, 1000000, 1000, 170, 0, 0.4, 0.75]
    seq, final, _ = get_seq(wiring, args)
    seq.plot()