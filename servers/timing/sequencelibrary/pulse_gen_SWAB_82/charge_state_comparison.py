# -*- coding: utf-8 -*-
"""
Created on Wed Aug  18 21:24:36 2021

This routine collects readout under yellow twice, once after a shrot red pusle 
and once after a green pulse. 

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
    nvm_prep_laser_dur, nv0_prep_laser_dur, charge_readout_dur, \
    nvm_prep_laser_key, nv0_prep_laser_key, charge_readout_laser_key,\
    nvm_prep_laser_power, nv0_prep_laser_power, charge_read_laser_power, \
    initial_delay, apd_index  = args

    # Get what we need out of the wiring dictionary
    pulser_wiring = config['Wiring']['PulseStreamer']
    pulser_do_daq_clock = pulser_wiring['do_sample_clock']
    pulser_do_daq_gate = pulser_wiring['do_apd_{}_gate'.format(apd_index)]
    
    nvm_prep_aom_delay_time = config['Optics'][nvm_prep_laser_key]['delay']
    nv0_prep_aom_delay_time = config['Optics'][nv0_prep_laser_key]['delay']
    charge_readout_aom_delay_time = config['Optics'][charge_readout_laser_key]['delay']
    
    # Convert the 32 bit ints into 64 bit ints
    nvm_prep_laser_dur = numpy.int64(nvm_prep_laser_dur)
    nv0_prep_laser_dur = numpy.int64(nv0_prep_laser_dur)
    charge_readout_dur = numpy.int64(charge_readout_dur)
    
    intra_pulse_delay = config['CommonDurations']['cw_meas_buffer']
    
    total_delay = nvm_prep_aom_delay_time + nv0_prep_aom_delay_time + \
                                            charge_readout_aom_delay_time
    
    period = initial_delay + total_delay + nvm_prep_laser_dur + nv0_prep_laser_dur +\
             charge_readout_dur*2 + intra_pulse_delay*3 + 300
        
    #%% Define the sequence
    seq = Sequence()

    # Clock
    train = [(period - 200, LOW), (100, HIGH), (100, LOW)]
    seq.setDigital(pulser_do_daq_clock, train)

    # APD gate
    train = [(initial_delay + total_delay + nvm_prep_laser_dur + intra_pulse_delay, LOW), 
             (charge_readout_dur, HIGH), (intra_pulse_delay + nv0_prep_laser_dur + intra_pulse_delay, LOW),
             (charge_readout_dur, HIGH), (300, LOW)]
    seq.setDigital(pulser_do_daq_gate, train)
    
    # nvm train
    train_nvm = [(initial_delay + total_delay - nvm_prep_aom_delay_time, LOW), 
                        (nvm_prep_laser_dur, HIGH), 
                        (nv0_prep_laser_dur + charge_readout_dur*2 + intra_pulse_delay*3 + 300, LOW )]
    tool_belt.process_laser_seq(pulse_streamer, seq, config,
                            nvm_prep_laser_key, nvm_prep_laser_power, train_nvm)
    
    # nv0
    train_nv0 = [(initial_delay + total_delay - nv0_prep_aom_delay_time + \
                  nvm_prep_laser_dur + intra_pulse_delay*2 + charge_readout_dur, LOW), 
                        (nv0_prep_laser_dur, HIGH), 
                        (charge_readout_dur + intra_pulse_delay + 300, LOW )]
    tool_belt.process_laser_seq(pulse_streamer, seq, config,
                            nv0_prep_laser_key, nv0_prep_laser_power, train_nv0)
    
    
    train_read = [(initial_delay + total_delay - charge_readout_aom_delay_time + \
                   nvm_prep_laser_dur + intra_pulse_delay, LOW), 
                  (charge_readout_dur, HIGH), (intra_pulse_delay*2 + nv0_prep_laser_dur ,LOW ),
                  (charge_readout_dur, HIGH), (300 ,LOW )]
    tool_belt.process_laser_seq(pulse_streamer, seq, config,
                            charge_readout_laser_key, charge_read_laser_power, train_read)
        
    final_digital = []
    final = OutputState(final_digital, 0.0, 0.0)

    return seq, final, [period]


if __name__ == '__main__':
    config = tool_belt.get_config_dict()
    
    # args = [10000, 10000, 50000, 'laserglow_532', 'cobolt_638', 'laserglow_589', None, None, 0.8, 0, 0]
    args= [1000.0, 1000.0, 15000000.0, "laserglow_532", "cobolt_638", "laserglow_589", None, None, 0.2, 500000, 0]
    seq = get_seq(None, config, args)[0]
    seq.plot()
