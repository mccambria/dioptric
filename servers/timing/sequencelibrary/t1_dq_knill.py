# -*- coding: utf-8 -*-
"""
Created on Sat May  4 08:34:08 2019

@author: Aedan
"""

from pulsestreamer import Sequence
from pulsestreamer import OutputState
import numpy
import utils.tool_belt as tool_belt
from utils.tool_belt import States

LOW = 0
HIGH = 1

def get_seq(pulser_wiring, args):

    # %% Parse wiring and args

    # The first 11 args are ns durations and we need them as int64s
    durations = []
    for ind in range(15):
        durations.append(numpy.int64(args[ind]))

    # Unpack the durations
    tau_shrt, polarization_time, signal_time, reference_time,  \
            sig_to_ref_wait_time, pre_uwave_exp_wait_time,  \
            post_uwave_exp_wait_time, aom_delay_time, \
            sig_gen_tsg4104a_delay, sig_gen_sg394_delay, iq_delay_time, \
            gate_time, pi_pulse_low, pi_pulse_high, tau_long = durations

    # Get the APD indices
    apd_index = args[15]

    # Specify the initial and readout states
    init_state_value = args[16]
    read_state_value = args[17]

    pulser_do_apd_gate = pulser_wiring['do_apd_{}_gate'.format(apd_index)]

    low_sig_gen_name = tool_belt.get_signal_generator_name(States.LOW)
    low_sig_gen_gate_chan_name = 'do_{}_gate'.format(low_sig_gen_name)
    pulser_do_sig_gen_low_gate = pulser_wiring[low_sig_gen_gate_chan_name]
    if low_sig_gen_name == 'signal_generator_tsg4104a':
        rf_low_delay = sig_gen_tsg4104a_delay
    elif low_sig_gen_name == 'signal_generator_sg394':
        rf_low_delay = sig_gen_sg394_delay
    
    high_sig_gen_name = tool_belt.get_signal_generator_name(States.HIGH)
    high_sig_gen_gate_chan_name = 'do_{}_gate'.format(high_sig_gen_name)
    pulser_do_sig_gen_high_gate = pulser_wiring[high_sig_gen_gate_chan_name]
    if high_sig_gen_name == 'signal_generator_tsg4104a':
        rf_high_delay = sig_gen_tsg4104a_delay
    elif high_sig_gen_name == 'signal_generator_sg394':
        rf_high_delay = sig_gen_sg394_delay
    
    pulser_do_arb_wave_trigger = pulser_wiring['do_arb_wave_trigger']

    # %% Some further setup

    # Default the pulses to 0
    init_pi_low = 0
    init_pi_high = 0
    read_pi_low = 0
    read_pi_high = 0

    # Set pulse durations for Knill composite pulses
    if init_state_value == States.LOW.value:
        init_pi_low = 5*pi_pulse_low
    elif init_state_value == States.HIGH.value:
        init_pi_high = 5*pi_pulse_high
    if read_state_value == States.LOW.value:
        read_pi_low = 5*pi_pulse_low
    elif read_state_value == States.HIGH.value:
        read_pi_high = 5*pi_pulse_high
        
    pulser_do_aom = pulser_wiring['do_532_aom']
    
    base_uwave_experiment_dur = init_pi_high + init_pi_low + \
                    read_pi_high + read_pi_low
    uwave_experiment_shrt = base_uwave_experiment_dur + tau_shrt
    uwave_experiment_long = base_uwave_experiment_dur + tau_long

    prep_time = aom_delay_time + \
        polarization_time + pre_uwave_exp_wait_time + \
        uwave_experiment_shrt + post_uwave_exp_wait_time

    up_to_long_gates = prep_time + signal_time + sig_to_ref_wait_time + \
        reference_time + pre_uwave_exp_wait_time + \
        uwave_experiment_long + post_uwave_exp_wait_time

    # The period is independent of the particular tau, but it must be long
    # enough to accomodate the longest tau
    period = aom_delay_time + polarization_time + \
        pre_uwave_exp_wait_time + uwave_experiment_shrt + post_uwave_exp_wait_time + \
        signal_time + sig_to_ref_wait_time + reference_time + pre_uwave_exp_wait_time + \
        uwave_experiment_long + post_uwave_exp_wait_time + \
        signal_time + sig_to_ref_wait_time + gate_time

    seq = Sequence()
    
    # %% APD

    pre_duration = prep_time
    short_sig_to_short_ref = signal_time + sig_to_ref_wait_time - gate_time
    short_ref_to_long_sig = up_to_long_gates - (prep_time + signal_time + sig_to_ref_wait_time + gate_time)
    long_sig_to_long_ref = signal_time + sig_to_ref_wait_time - gate_time
    train = [(pre_duration, LOW),
             (gate_time, HIGH),
             (short_sig_to_short_ref, LOW),
             (gate_time, HIGH),
             (short_ref_to_long_sig, LOW),
             (gate_time, HIGH),
             (long_sig_to_long_ref, LOW),
             (gate_time, HIGH)]
    seq.setDigital(pulser_do_apd_gate, train)
    
    # %% Green laser

    train = [(polarization_time, HIGH),
             (pre_uwave_exp_wait_time + uwave_experiment_shrt + post_uwave_exp_wait_time, LOW),
             (signal_time, HIGH),
             (sig_to_ref_wait_time, LOW),
             (reference_time, HIGH),
             (pre_uwave_exp_wait_time + uwave_experiment_long + post_uwave_exp_wait_time, LOW),
             (signal_time, HIGH),
             (sig_to_ref_wait_time, LOW),
             (gate_time + aom_delay_time, HIGH)]
    seq.setDigital(pulser_do_aom, train)

    # %% Microwaves

    pre_duration = aom_delay_time + polarization_time + pre_uwave_exp_wait_time
    mid_duration = post_uwave_exp_wait_time + signal_time + sig_to_ref_wait_time + \
        reference_time + pre_uwave_exp_wait_time
    post_duration = post_uwave_exp_wait_time + signal_time + \
        sig_to_ref_wait_time + gate_time

    train = [(pre_duration - rf_high_delay, LOW)]
    train.extend([(init_pi_high, LOW), (init_pi_low + tau_shrt, LOW), (read_pi_high, LOW)])
    train.extend([(read_pi_low + mid_duration, LOW)])
    train.extend([(init_pi_high, LOW), (init_pi_low + tau_long, LOW), (read_pi_high, LOW)])
    train.extend([(read_pi_low + post_duration + rf_high_delay, LOW)])
    seq.setDigital(pulser_do_sig_gen_high_gate, train)

    train = [(pre_duration - rf_low_delay, LOW)]
    train.extend([(init_pi_low, LOW), (init_pi_high + tau_shrt, LOW), (read_pi_low, LOW)])
    train.extend([(read_pi_high + mid_duration, LOW)])
    train.extend([(init_pi_low, LOW), (init_pi_high + tau_long, LOW), (read_pi_low, LOW)])
    train.extend([(read_pi_high + post_duration + rf_low_delay, LOW)])
    seq.setDigital(pulser_do_sig_gen_low_gate, train)
    
    # %% IQ modulation
    
    composite_low_seq = [(10, HIGH), (pi_pulse_low-10, LOW)] * 5
    composite_high_seq = [(10, HIGH), (pi_pulse_high-10, LOW)] * 5

    train = [(pre_duration - iq_delay_time, LOW)]
    
    if init_state_value == States.LOW.value:
        train.extend(composite_low_seq)
    elif init_state_value == States.HIGH.value:
        train.extend(composite_high_seq)
        
    train.extend([(tau_shrt, LOW)])
    
    if read_state_value == States.LOW.value:
        train.extend(composite_low_seq)
    elif read_state_value == States.HIGH.value:
        train.extend(composite_high_seq)
        
    train.extend([(mid_duration, LOW)])
    
    if init_state_value == States.LOW.value:
        train.extend(composite_low_seq)
    elif init_state_value == States.HIGH.value:
        train.extend(composite_high_seq)
        
    train.extend([(tau_long, LOW)])
    
    if read_state_value == States.LOW.value:
        train.extend(composite_low_seq)
    elif read_state_value == States.HIGH.value:
        train.extend(composite_high_seq)
        
    train.extend([(post_duration + iq_delay_time, LOW)])
    
    seq.setDigital(pulser_do_arb_wave_trigger, train)
    
    # %% Return the sequence

    final_digital = [pulser_wiring['do_532_aom'],
                     pulser_wiring['do_sample_clock']]
    final = OutputState(final_digital, 0.0, 0.0)
    return seq, final, [period]


if __name__ == '__main__':
    wiring = {'do_sample_clock': 0,
              'do_apd_0_gate': 4,
              'do_532_aom': 1,
              'do_signal_generator_sg394_gate': 2,
              'do_signal_generator_tsg4104a_gate': 3,
              'do_arb_wave_trigger': 5,}
    
    # seq_args = [6428, 3000, 3000, 3000, 2000, 1000, 1000, 0, 0, 510, 51, 80, 3571, 0, 3, 3]
    # seq_args = [0, 3000, 3000, 3000, 2000, 1000, 1000, 0, 0, 510, 51, 80, 5000, 0, 3, 3]
    # seq_args = [3000, 1000, 1000, 1000, 2000, 1000, 1000, 1080, 1005, 995, 560, 350, 121, 73, 0, 0, 3, 3]
    seq_args = [3000, 1000, 1000, 1000, 2000, 1000, 1000, 0, 0, 0, 0, 350, 121, 7300, 0, 0, 3, 3]

    seq, final, ret_vals = get_seq(wiring, seq_args)
    seq.plot()
