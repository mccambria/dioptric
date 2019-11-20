#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 16:45:20 2019

@author: yanfeili
"""

from pulsestreamer import Sequence
from pulsestreamer import OutputState
import numpy

LOW = 0
HIGH = 1

def get_seq(pulser_wiring, args):

    # Unpack the args
    gate_time, aom_delay589, apd_indices, aom_power = args



    readout_time = numpy.int64(gate_time)
    aom_delay589 = numpy.int64(aom_delay589)

   #SCC photon collection test period
    period =  readout_time + aom_delay589
    # Get what we need out of the wiring dictionary
    pulser_do_apd_gate = pulser_wiring['do_apd_{}_gate'.format(apd_indices)]
    pulser_ao_aom589 = pulser_wiring['ao_589_aom']


    seq = Sequence()


    #collect photons for certain timewindow tR in APD
    train = [(readout_time, HIGH), (aom_delay589, LOW)]
    seq.setDigital(pulser_do_apd_gate, train)


    #readout with 589
    train = [(readout_time, aom_power),(aom_delay589, LOW)]
    seq.setAnalog(pulser_ao_aom589, train)
    
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
    args = [8*10**6, 1*10**6,0,1.0]
    seq, final, _ = get_seq(wiring, args)
    seq.plot()
