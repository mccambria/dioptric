# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 21:24:36 2019

@author: mccambria
"""

from pulsestreamer import Sequence
from pulsestreamer import OutputState
import utils.tool_belt as tool_belt
import numpy

LOW = 0
HIGH = 1


def get_seq(pulser_wiring, args):

    # Unpack the args
    delay, readout_time, aom_ao_589_pwr, apd_index, color_ind = args

    # Get what we need out of the wiring dictionary
    pulser_do_daq_clock = pulser_wiring['do_sample_clock']
    pulser_do_daq_gate = pulser_wiring['do_apd_{}_gate'.format(apd_index)]
    pulser_do_532_aom = pulser_wiring['do_532_aom']
    pulser_ao_589_aom = pulser_wiring['ao_589_aom']
    pulser_do_638_aom = pulser_wiring['do_638_laser']

    # Convert the 32 bit ints into 64 bit ints
    delay = numpy.int64(delay)
    readout_time = numpy.int64(readout_time)
    
    period = numpy.int64(delay + readout_time + 100)
    
    # Make sure the aom_ao_589_pwer is within range of +1 and 0
    tool_belt.aom_ao_589_pwr_err(aom_ao_589_pwr)
        
    # make sure only the color passed is either 532 or 589
#    tool_belt.color_ind_err(color_ind)
        
    # Define the sequence
    seq = Sequence()

#    # The clock signal will be high for 100 ns with buffers of 100 ns on
#    # either side. During the buffers, everything should be low. The buffers
#    # account for any timing jitters/delays and ensure that everything we
#    # expect to be on one side of the clock signal is indeed on that side.
#    train = [(delay + readout_time + 100, LOW), (100, HIGH), (100, LOW)]
#    seq.setDigital(pulser_do_daq_clock, train)

    train = [(delay, LOW), (readout_time, HIGH), (100, LOW)]
    seq.setDigital(pulser_do_daq_gate, train)
    
    if color_ind == 589:
        
        train = [(period, aom_ao_589_pwr)]
        seq.setAnalog(pulser_ao_589_aom, train)
            
    elif color_ind == 532:
               
        train = [(period, HIGH)]
        seq.setDigital(pulser_do_532_aom, train)

    elif color_ind == 638:
        
        train = [(period, HIGH)]
        seq.setDigital(pulser_do_638_aom, train)
        
    final_digital = [pulser_do_daq_clock]
    final = OutputState(final_digital, 0.0, 0.0)

    return seq, final, [period]


if __name__ == '__main__':
    wiring = {'do_sample_clock': 0,
              'do_apd_0_gate': 1,
              'do_638_laser': 3,
              'do_532_aom': 2,
              'ao_589_aom': 1}
    args = [500000, 10000000, 0.3, 0, 532]
#    seq_args_string = tool_belt.encode_seq_args(args)
    seq, ret_vals, period = get_seq(wiring, args)
    seq.plot()
