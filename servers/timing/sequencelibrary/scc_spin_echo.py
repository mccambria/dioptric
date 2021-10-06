# -*- coding: utf-8 -*-
"""
Created on Sat May  4 08:34:08 2019

@author: Aedan
"""

from pulsestreamer import Sequence
from pulsestreamer import OutputState
import numpy
import utils.tool_belt as tool_belt

LOW = 0
HIGH = 1


def get_seq(pulse_streamer, config, args):

    # %% Parse wiring and args

    # Unpack the args
    readout_time, reion_time, ion_time, shelf_time,\
            tau_shrt, tau_long, pi_pulse, pi_on_2_pulse, \
            green_laser_name, yellow_laser_name, red_laser_name, sig_gen_name, \
            apd_indices, readout_power, shelf_power = args
            
    # Convert all times to int64
    readout_time = numpy.int64(readout_time)
    reion_time = numpy.int64(reion_time)
    ion_time = numpy.int64(ion_time)
    shelf_time = numpy.int64(shelf_time)
    pi_pulse = numpy.int64(pi_pulse)
    pi_on_2_pulse = numpy.int64(pi_on_2_pulse)
    tau_shrt = numpy.int64(tau_shrt)
    tau_long = numpy.int64(tau_long)
            
        
    wait_time = config['CommonDurations']['uwave_buffer']
    
    # delays
    green_delay_time = config['Optics'][green_laser_name]['delay']
    yellow_delay_time = config['Optics'][yellow_laser_name]['delay']
    red_delay_time = config['Optics'][red_laser_name]['delay']
    rf_delay_time = config['Microwaves'][sig_gen_name]['delay']
    
    # TESTING
    # wait_time =100
    # galvo_move_time = 500
    # galvo_move_time = numpy.int64(galvo_move_time)
    # green_delay_time = 0
    # yellow_delay_time = 0
    # red_delay_time =0
    # rf_delay_time = 0
    
    total_delay = green_delay_time + yellow_delay_time + red_delay_time + rf_delay_time
    
    # Get what we need out of the wiring dictionary
    pulser_wiring = config['Wiring']['PulseStreamer']
    pulser_do_apd_gate = pulser_wiring['do_apd_{}_gate'.format(apd_indices)]
    pulser_ao_589_aom = pulser_wiring['ao_{}_am'.format(yellow_laser_name)]
    sig_gen_gate_chan_name = 'do_{}_gate'.format(sig_gen_name)
    pulser_do_sig_gen_gate = pulser_wiring[sig_gen_gate_chan_name]
    

    # %% Write the microwave sequence to be used.

    # In t1, the sequence is just a pi pulse, wait for a relaxation time, then
    # then a second pi pulse

    # I define both the time of this experiment, which is useful for the AOM
    # and gate sequences to dictate the time for them to be LOW
    # And I define the actual uwave experiement to be plugged into the rf
    # sequence. I hope that this formatting works.

    # With future protocols--ramsey, spin echo, etc--it will be easy to use
    # this format of sequence building and just change this section of the file

    uwave_experiment_shrt = pi_on_2_pulse + tau_shrt + pi_pulse + \
                            tau_shrt + pi_on_2_pulse
    uwave_experiment_long = pi_on_2_pulse + tau_long + pi_pulse + \
                            tau_long + pi_on_2_pulse

    # %% Couple calculated values

    readout_scheme = shelf_time + ion_time + wait_time + readout_time
    
    reference_experiment_shrt = reion_time + wait_time + \
        uwave_experiment_shrt + wait_time
        
    reference_experiment_long = reion_time + wait_time + \
        uwave_experiment_long + wait_time

    # up_to_long_gates = prep_time + readout_scheme + sig_to_ref_wait_time_shrt + \
    #     reference_time + pre_uwave_exp_wait_time + \
    #     uwave_experiment_long + post_uwave_exp_wait_time
        

    # %% Calclate total period. This is fixed for each tau index

    # The period is independent of the particular tau, but it must be long
    # enough to accomodate the longest tau
    period = total_delay + ( reference_experiment_shrt + 2* wait_time + readout_scheme)*2 +\
        ( reference_experiment_long + 2* wait_time + readout_scheme)*2

    # %% Define the sequence

    seq = Sequence()

    # APD gating
    train = [(total_delay + reference_experiment_shrt + readout_scheme - readout_time, LOW),
             (readout_time, HIGH),
             (wait_time + reference_experiment_shrt + readout_scheme - readout_time, LOW),
             (readout_time, HIGH),
             (wait_time + reference_experiment_long + readout_scheme - readout_time, LOW),
             (readout_time, HIGH),
             (wait_time + reference_experiment_long + readout_scheme - readout_time, LOW),
             (readout_time, HIGH), (wait_time, LOW)]
    seq.setDigital(pulser_do_apd_gate, train)

    # Green laser
    delay = total_delay - green_delay_time
    train = [(delay + reion_time, HIGH),
             (wait_time + uwave_experiment_shrt + wait_time + readout_scheme + wait_time, LOW),
             (reion_time, HIGH),
             (wait_time + uwave_experiment_shrt + wait_time + readout_scheme + wait_time, LOW),
             (reion_time, HIGH),
             (wait_time + uwave_experiment_long + wait_time + readout_scheme + wait_time, LOW),
             (reion_time, HIGH),
             (wait_time + uwave_experiment_long + wait_time + readout_scheme + wait_time, LOW),
             (green_delay_time, LOW)]
    tool_belt.process_laser_seq(pulse_streamer, seq, config,
                                green_laser_name, None, train)
    
    # Red laser
    delay = total_delay - red_delay_time
    train = [(delay + reference_experiment_shrt + shelf_time, LOW),
             (ion_time, HIGH),
             (wait_time + readout_time + wait_time, LOW),
             (reference_experiment_shrt +  shelf_time, LOW),
             (ion_time, HIGH),
             (wait_time + readout_time + wait_time, LOW),
             (reference_experiment_long +  shelf_time, LOW),
             (ion_time, HIGH),
             (wait_time + readout_time + wait_time, LOW),
             (reference_experiment_long + shelf_time, LOW),
             (ion_time, HIGH),
             (wait_time + readout_time + wait_time, LOW),
             (red_delay_time, LOW)]
    tool_belt.process_laser_seq(pulse_streamer, seq, config,
                                red_laser_name, None, train)
    
    # Yellow laser
    delay = total_delay - yellow_delay_time
    train = [(delay + reference_experiment_shrt, LOW),
             (shelf_time + ion_time, shelf_power),
             (wait_time, LOW),
             (readout_time, readout_power),
             (wait_time, LOW),
             (reference_experiment_shrt, LOW),
             (shelf_time + ion_time, shelf_power),
             (wait_time, LOW),
             (readout_time, readout_power),
             (wait_time, LOW),
             (reference_experiment_long, LOW),
             (shelf_time + ion_time, shelf_power),
             (wait_time, LOW),
             (readout_time, readout_power),
             (wait_time, LOW),
             (reference_experiment_long, LOW),
             (shelf_time + ion_time, shelf_power),
             (wait_time, LOW),
             (readout_time, readout_power),
             (wait_time, LOW),
             (yellow_delay_time, LOW)]
    seq.setAnalog(pulser_ao_589_aom, train) 
    
    

    # Pulse the microwave for tau
    delay = total_delay - rf_delay_time
    train = [(delay + reion_time + wait_time, LOW),
             (pi_on_2_pulse, HIGH), (tau_shrt, LOW),
             (pi_pulse, HIGH),
             (tau_shrt, LOW), (pi_on_2_pulse, HIGH),
             (wait_time + readout_scheme + wait_time, LOW),
             (reference_experiment_shrt + readout_scheme + wait_time, LOW),
             
             (reion_time + wait_time, LOW),
             (pi_on_2_pulse, HIGH), (tau_long, LOW),
             (pi_pulse, HIGH),
             (tau_long, LOW), (pi_on_2_pulse, HIGH),
             (wait_time + readout_scheme + wait_time, LOW),
             (reference_experiment_long + readout_scheme + wait_time, LOW),
             (rf_delay_time, LOW)
             ]
    seq.setDigital(pulser_do_sig_gen_gate, train)

    final_digital = [pulser_wiring['do_sample_clock']]
    final = OutputState(final_digital, 0.0, 0.0)
    return seq, final, [period]

if __name__ == '__main__':
    config = tool_belt.get_config_dict()
            
    # seq_args = [1000, 500, 500, 100, 0, 1000, 100, 50, 
    #             'cobolt_515', 'laserglow_589','cobolt_638', 
    #             'signal_generator_bnc835', 0, 0.5, 1.0]
    seq_args = [500000.0, 100000.0, 300, 0, 20000, 40000, 72, 36,
                'cobolt_515', 'laserglow_589', 'cobolt_638', 
                'signal_generator_bnc835', 0, 0.3, 0.4]
    seq, final, ret_vals = get_seq(None, config, seq_args)
    seq.plot()
