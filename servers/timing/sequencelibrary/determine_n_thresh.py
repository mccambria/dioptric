#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 16:45:20 2019

Collect the photon counts under yellow illumination, after reionizing NV into 
NV- with green light.

@author: yanfeili
"""

from pulsestreamer import Sequence
from pulsestreamer import OutputState
import utils.tool_belt as tool_belt
import numpy

LOW = 0
HIGH = 1

def get_seq(pulser_wiring, args):

    # Unpack the args
    readout_time, reionization_time, illumination_time, aom_delay, apd_indices, \
                                                    aom_ao_589_pwr = args

    readout_time = numpy.int64(readout_time)
    illumination_time = numpy.int64(illumination_time)
    reionization_time = numpy.int64(reionization_time)
    aom_delay = numpy.int64(aom_delay)

    # SCC photon collection test period
    period =  reionization_time + illumination_time + aom_delay + 400
    
    # Get what we need out of the wiring dictionary
    pulser_do_apd_gate = pulser_wiring['do_apd_{}_gate'.format(apd_indices)]
    pulser_do_clock = pulser_wiring['do_sample_clock']
    pulser_do_532_aom = pulser_wiring['do_532_aom']
    pulser_ao_589_aom = pulser_wiring['ao_589_aom']
    
    # Make sure the ao_aom voltage to the 589 aom is within 0 and 1 V
    tool_belt.aom_ao_589_pwr_err(aom_ao_589_pwr)
    
    seq = Sequence()


    #collect photons for certain timewindow tR in APD
    train = [(aom_delay + reionization_time + 100, LOW), (readout_time, HIGH), (100, LOW)]
    seq.setDigital(pulser_do_apd_gate, train)
    
    #clock pulse
    train = [(aom_delay + reionization_time + 100 + readout_time + 100, LOW), (100, HIGH), (100, LOW)]
    seq.setDigital(pulser_do_clock, train)

    # illuminate with 532 
    train = [(reionization_time, HIGH), (aom_delay, LOW)]
    seq.setDigital(pulser_do_532_aom, train)
#    train = [(reionization_time, HIGH), (100, LOW), (readout_time, HIGH), (aom_delay, LOW)]
#    seq.setDigital(pulser_do_532_aom, train)
    
    # readout with 589
    train = [(reionization_time + 100, LOW), (illumination_time, aom_ao_589_pwr), (aom_delay, LOW)]
    seq.setAnalog(pulser_ao_589_aom, train)
    
    final_digital = []
    final = OutputState(final_digital, 0.0, 0.0)

    return seq, final, [period]


if __name__ == '__main__':
    wiring = {'do_apd_0_gate': 1,
              'do_532_aom': 2,
              'sig_gen_gate_chan_name': 3,
               'do_sample_clock':4,
               'ao_589_aom': 0,
               'do_638_aom': 6}
    args = [1000, 500 ,500,0, 0,1.0]
    seq, final, _ = get_seq(wiring, args)
    seq.plot()
