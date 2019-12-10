#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 20:40:44 2019

@author: yanfeili
"""

from pulsestreamer import Sequence
import ultils.tool_belt as tool_belt
import numpy

LOW = 0
HIGH = 1

def get_seq(pulser_wiring, args):

    # Unpack the args
    readout_time, reionization_time, illumination_time, ionization_time, \
            aom_delay, apd_indices, aom_ao_589_pwr, ao_638_pwr = args

    readout_time = numpy.int64(readout_time)
    aom_delay = numpy.int64(aom_delay)
    ionization_time = numpy.int64(ionization_time)
    reionization_time = numpy.int64(reionization_time)
    illumination_time = numpy.int64(illumination_time)

    # SCC photon collection test period
    period =  (reionization_time + ionization_time + aom_delay+ illumination_time + 400)*2
    
    # Get what we need out of the wiring dictionary
    pulser_do_apd_gate = pulser_wiring['do_apd_{}_gate'.format(apd_indices)]
    pulser_do_clock = pulser_wiring['do_sample_clock']
    pulser_do_532_aom = pulser_wiring['do_532_aom']
    pulser_ao_589_aom = pulser_wiring['ao_589_aom']
    pulser_do_638_aom = pulser_wiring['do_638_aom']

    # Make sure the ao_aom voltage to the 589 aom is within 0 and 1 V
    tool_belt.aom_ao_589_pwr_err(aom_ao_589_pwr)
    
    # Make sure the ao voltage to the 638 laser is within 0 and 1 V
    tool_belt.ao_638_pwr_err(ao_638_pwr)
    
    seq = Sequence()


    #collect photons for certain timewindow tR in APD
    train = [(aom_delay + reionization_time + ionization_time + aom_delay + 100, LOW), (readout_time, HIGH), (illumination_time - readout_time + 300, LOW),
             (aom_delay + reionization_time + ionization_time + aom_delay + 100, LOW), (readout_time, HIGH), (illumination_time - readout_time + 300, LOW)]
    seq.setDigital(pulser_do_apd_gate, train)
    
    #clock pulse
    train = [(aom_delay + reionization_time + ionization_time + aom_delay + 100 + illumination_time + 100, LOW), (100, HIGH), (100, LOW),
             (aom_delay + reionization_time + ionization_time + aom_delay + 100 + illumination_time + 100, LOW), (100, HIGH), (100, LOW)]
    seq.setDigital(pulser_do_clock, train)

    #reionize with 532
    train = [ (reionization_time, HIGH), (period/2 -reionization_time , LOW), 
              (reionization_time, HIGH), (period/2 -reionization_time , LOW)]
    seq.setDigital(pulser_do_532_aom, train)
    
    #readout with 589
    train = [(reionization_time + ionization_time + aom_delay + 100, LOW), (illumination_time, aom_ao_589_pwr), (aom_delay + 300, LOW),
             (reionization_time + ionization_time + aom_delay + 100, LOW), (illumination_time, aom_ao_589_pwr), (aom_delay + 300, LOW)]
    seq.setAnalog(pulser_ao_589_aom, train) 
    
    #ionize with 638 
    train = [(reionization_time+ 100, LOW), (ionization_time, ao_638_pwr), (aom_delay + illumination_time + aom_delay + 300, LOW), (period/2, LOW)]
    seq.setDigital(pulser_do_638_aom, train)
    
    final_digital = []
    final = OutputState(final_digital, 0.0, 0.0)

    return seq, final, [period]


if __name__ == '__main__':
    wiring = {'do_apd_0_gate': 1,
              'do_532_aom': 2,
              'sig_gen_gate_chan_name': 3,
               'do_sample_clock':4,
               'ao_589_aom': 0,
               'do_638_aom': 6              }
    args = [1000, 500, 1200, 1000, 100, 0, 1.0, 300]
    seq, final, _ = get_seq(wiring, args)
    seq.plot()