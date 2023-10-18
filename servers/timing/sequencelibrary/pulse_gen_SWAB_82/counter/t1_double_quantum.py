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

def get_seq(pulse_streamer, config, args):

    # %% Parse wiring and args

    # The first 11 args are ns durations and we need them as int64s
    durations = []
    for ind in range(6):
        durations.append(numpy.int64(args[ind]))

    # Unpack the durations
    tau_shrt, polarization_time,   \
            gate_time, pi_pulse_low, pi_pulse_high, tau_long = durations

    
    # Specify the initial and readout states
    init_state_value = args[6]
    read_state_value = args[7]
    
    laser_name = args[8]
    laser_power = args[9]
    
    
    # time of illumination during which signal readout occurs
    signal_time = polarization_time
    # time of illumination during which reference readout occurs
    reference_time = polarization_time
    pre_uwave_exp_wait_time = config['CommonDurations']['uwave_buffer']
    post_uwave_exp_wait_time = config['CommonDurations']['uwave_buffer']
    # time between signal and reference without illumination
    sig_to_ref_wait_time = pre_uwave_exp_wait_time + post_uwave_exp_wait_time
    
    aom_delay_time = config['Optics'][laser_name]['delay']
    low_sig_gen_name = config['Servers']['sig_gen_LOW']
    high_sig_gen_name = config['Servers']['sig_gen_HIGH']
    
    rf_low_delay = config['Microwaves'][low_sig_gen_name]['delay']
    rf_high_delay = config['Microwaves'][high_sig_gen_name]['delay']
    
    
    pulser_wiring = config['Wiring']['PulseGen']
    pulser_do_apd_gate = pulser_wiring['do_apd_gate']
    low_sig_gen_gate_chan_name = 'do_{}_gate'.format(low_sig_gen_name)
    pulser_do_sig_gen_low_gate = pulser_wiring[low_sig_gen_gate_chan_name]
    high_sig_gen_gate_chan_name = 'do_{}_gate'.format(high_sig_gen_name)
    pulser_do_sig_gen_high_gate = pulser_wiring[high_sig_gen_gate_chan_name]

    # Default the pulses to 0
    init_pi_low = 0
    init_pi_high = 0
    read_pi_low = 0
    read_pi_high = 0

    if init_state_value == States.LOW.value:
        init_pi_low = pi_pulse_low
    elif init_state_value == States.HIGH.value:
        init_pi_high = pi_pulse_high

    if read_state_value == States.LOW.value:
        read_pi_low = pi_pulse_low
    elif read_state_value == States.HIGH.value:
        read_pi_high = pi_pulse_high

    # pulser_do_aom = pulser_wiring['do_532_aom']

    # %% Write the microwave sequence to be used.

    # In t1, the sequence is just a pi pulse, wait for a relaxation time, then
    # then a second pi pulse

    # I define both the time of this experiment, which is useful for the AOM
    # and gate sequences to dictate the time for them to be LOW
    # And I define the actual uwave experiement to be plugged into the rf
    # sequence. I hope that this formatting works.

    # With future protocols--ramsey, spin echo, etc--it will be easy to use
    # this format of sequence building and just change this secion of the file

    base_uwave_experiment_dur = init_pi_high + init_pi_low + \
                    read_pi_high + read_pi_low
    uwave_experiment_shrt = base_uwave_experiment_dur + tau_shrt
    uwave_experiment_long = base_uwave_experiment_dur + tau_long

#    uwave_experiment_seq_shrt = [(pi_pulse_init, HIGH), (tau_shrt, LOW),
#                                     (pi_pulse_read, HIGH)]


#    uwave_experiment_seq_long = [(pi_pulse_init, HIGH), (tau_long, LOW),
#                                     (pi_pulse_read, HIGH)]

    # %% Couple calculated values
                                                #??? Right now just using rf_low_delay for rf delay
    prep_time = aom_delay_time + rf_low_delay + \
        polarization_time + pre_uwave_exp_wait_time + \
        uwave_experiment_shrt + post_uwave_exp_wait_time

    up_to_long_gates = prep_time + signal_time + sig_to_ref_wait_time + \
        reference_time + pre_uwave_exp_wait_time + \
        uwave_experiment_long + post_uwave_exp_wait_time

    # %% Calclate total period. This is fixed for each tau index

    # The period is independent of the particular tau, but it must be long
    # enough to accomodate the longest tau
    period = aom_delay_time + rf_low_delay + polarization_time + \
        pre_uwave_exp_wait_time + uwave_experiment_shrt + post_uwave_exp_wait_time + \
        signal_time + sig_to_ref_wait_time + reference_time + pre_uwave_exp_wait_time + \
        uwave_experiment_long + post_uwave_exp_wait_time + \
        signal_time + sig_to_ref_wait_time + reference_time

    # %% Define the sequence

    seq = Sequence()

    # APD gating
    pre_duration = prep_time
    short_sig_to_short_ref = signal_time + sig_to_ref_wait_time - gate_time
    short_ref_to_long_sig = up_to_long_gates - (prep_time + signal_time + sig_to_ref_wait_time + gate_time)
    long_sig_to_long_ref = signal_time + sig_to_ref_wait_time - gate_time
    post_duration = reference_time - gate_time
    train = [(pre_duration, LOW),
             (gate_time, HIGH),
             (short_sig_to_short_ref, LOW),
             (gate_time, HIGH),
             (short_ref_to_long_sig, LOW),
             (gate_time, HIGH),
             (long_sig_to_long_ref, LOW),
             (gate_time, HIGH),
             (post_duration, LOW)]
    seq.setDigital(pulser_do_apd_gate, train)

    # Pulse the laser with the AOM for polarization and readout
    train = [(rf_low_delay + polarization_time, HIGH),
             (pre_uwave_exp_wait_time + uwave_experiment_shrt + post_uwave_exp_wait_time, LOW),
             (signal_time, HIGH),
             (sig_to_ref_wait_time, LOW),
             (reference_time, HIGH),
             (pre_uwave_exp_wait_time + uwave_experiment_long + post_uwave_exp_wait_time, LOW),
             (signal_time, HIGH),
             (sig_to_ref_wait_time, LOW),
             (reference_time + aom_delay_time, HIGH)]
    tool_belt.process_laser_seq(pulse_streamer, seq, config, 
                                laser_name, laser_power, train)
    # seq.setDigital(pulser_do_aom, train)

    # Pulse the microwave for tau
    pre_duration = aom_delay_time + polarization_time + pre_uwave_exp_wait_time
    mid_duration = post_uwave_exp_wait_time + signal_time + sig_to_ref_wait_time + \
        reference_time + pre_uwave_exp_wait_time
    post_duration = post_uwave_exp_wait_time + signal_time + \
        sig_to_ref_wait_time + reference_time + rf_low_delay

    train = [(pre_duration, LOW)]
    train.extend([(init_pi_high, HIGH), (tau_shrt + init_pi_low, LOW), (read_pi_high, HIGH)])
    train.extend([(read_pi_low + mid_duration, LOW)])
    train.extend([(init_pi_high, HIGH), (tau_long + init_pi_low, LOW), (read_pi_high, HIGH)])
    train.extend([(read_pi_low + post_duration, LOW)])
    seq.setDigital(pulser_do_sig_gen_high_gate, train)

    train = [(pre_duration, LOW)]
    train.extend([(init_pi_low, HIGH), (tau_shrt + init_pi_high, LOW), (read_pi_low, HIGH)])
    train.extend([(read_pi_high + mid_duration, LOW)])
    train.extend([(init_pi_low, HIGH), (tau_long + init_pi_high, LOW), (read_pi_low, HIGH)])
    train.extend([(read_pi_high + post_duration, LOW)])
    seq.setDigital(pulser_do_sig_gen_low_gate, train)

#    if init_state == 1 and read_state == 1:
#        pulser_do_uwave = pulser_do_uwave_read
#        train = [(pre_duration, LOW)]
#        train.extend([(pi_pulse_init, HIGH), (tau_shrt, LOW), (pi_pulse_read, HIGH)])
#        train.extend([(mid_duration, LOW)])
#        train.extend([(pi_pulse_init, HIGH), (tau_long, LOW), (pi_pulse_read, HIGH)])
#        train.extend([(post_duration, LOW)])
#        seq.setDigital(pulser_do_uwave, train)
#
#    if init_state == -1 and read_state == -1:
#        pulser_do_uwave = pulser_do_uwave_read
#        train = [(pre_duration, LOW)]
#        train.extend([(pi_pulse_init, HIGH), (tau_shrt, LOW), (pi_pulse_read, HIGH)])
#        train.extend([(mid_duration, LOW)])
#        train.extend([(pi_pulse_init, HIGH), (tau_long, LOW), (pi_pulse_read, HIGH)])
#        train.extend([(post_duration, LOW)])
#        seq.setDigital(pulser_do_uwave, train)

    final_digital = [pulser_wiring['do_sample_clock']]
    final = OutputState(final_digital, 0.0, 0.0)
    return seq, final, [period]

if __name__ == '__main__':
    config = tool_belt.get_config_dict()
    tool_belt.set_delays_to_zero(config)
    # tool_belt.set_feedthroughs_to_false(config)
    
    seq_args = [1000, 1000.0, 350, 59, 53, 2000,  1, 2, 'integrated_520', None]
    seq = get_seq(None, config, seq_args)[0]
    seq.plot()
