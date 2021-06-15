# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 21:24:36 2019

Similar to the simple readout sequence, except green light irs used to reionize
NVs into NV- before yellow light is used to readout photons.

@author: agardill
"""

from pulsestreamer import Sequence
from pulsestreamer import OutputState
import numpy

LOW = 0
HIGH = 1


def get_seq(pulser_wiring, args):

    # Unpack the args
    delay, readout_time, illumin, reioniz, aom_ao_589_pwr, apd_index = args

    # Get what we need out of the wiring dictionary
    pulser_do_daq_clock = pulser_wiring['do_sample_clock']
    pulser_do_daq_gate = pulser_wiring['do_apd_{}_gate'.format(apd_index)]
    pulser_do_532_aom = pulser_wiring['do_532_aom']
    pulser_ao_589_aom = pulser_wiring['ao_589_aom']

    # Convert the 32 bit ints into 64 bit ints
    delay = numpy.int64(delay)
    readout_time = numpy.int64(readout_time)
    illumin = numpy.int64(illumin)
    reioniz = numpy.int64(reioniz)
    period = numpy.int64(delay + illumin + reioniz + readout_time + 1300)
    
    # Make sure the aom_ao_589_pwer is within range of +1 and 0
    tool_belt.aom_ao_589_pwr_err(aom_ao_589_pwr)
    
    seq = Sequence()
    
    # The clock signal will be high for 100 ns with buffers of 100 ns on
    # either side. During the buffers, everything should be low. The buffers
    # account for any timing jitters/delays and ensure that everything we
    # expect to be on one side of the clock signal is indeed on that side.
    train = [(delay + reioniz + 1000 + readout_time + 100, LOW), (100, HIGH), (100, LOW)]
    seq.setDigital(pulser_do_daq_clock, train)

    train = [(delay + reioniz + 100, LOW), (readout_time, HIGH), (300, LOW)]
    seq.setDigital(pulser_do_daq_gate, train)
    
    # pulse the green laser to reionize to NV-
    train = [(reioniz, HIGH), (1000 + illumin, LOW)]
    seq.setDigital(pulser_do_532_aom, train)
    
    # pulse the yellow laser to readout the counts
    train = [(reioniz + 1000, 0.0), (illumin, aom_ao_589_pwr), (100, 0.0)]
    seq.setAnalog(pulser_ao_589_aom, train)  

    
    final_digital = [3]
    final = OutputState(final_digital, 0.0, 0.0)


    return seq, final, [period]


if __name__ == '__main__':
    wiring = {'do_sample_clock': 0,
              'do_apd_0_gate': 1,
              'do_532_aom': 2,
              'ao_589_aom': 0}
    args = [0, 500, 500, 250, 1.0, 0]
    seq, ret_vals, _ = get_seq(wiring, args)
    seq.plot()
