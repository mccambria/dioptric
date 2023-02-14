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

    # The first few args are ns durations and we need them as int64s
    durations = []
    for ind in range(6):
        durations.append(numpy.int64(args[ind]))

    # Unpack the durations
    tau_shrt, polarization_time, gate_time, pi_pulse_low, pi_pulse_high, tau_long = durations

    # Get the rest of the arguments
    init_state_value, read_state_value, laser_name, laser_power = args[6:10]
            
    # time of illumination during which signal readout occurs
    signal_time = polarization_time
    # time of illumination during which reference readout occurs
    reference_time = polarization_time
    pre_uwave_exp_wait_time = config['CommonDurations']['uwave_buffer']
    post_uwave_exp_wait_time = config['CommonDurations']['uwave_buffer']
    # time between signal and reference without illumination
    sig_to_ref_wait_time = pre_uwave_exp_wait_time + post_uwave_exp_wait_time
            
    aom_delay_time = config['Optics'][laser_name]['delay']
    low_sig_gen_name = config["Servers"][f"sig_gen_{States.LOW.name}"]
    high_sig_gen_name = config["Servers"][f"sig_gen_{States.HIGH.name}"]
    
    rf_low_delay = config['Microwaves'][low_sig_gen_name]['delay']
    rf_high_delay = config['Microwaves'][high_sig_gen_name]['delay']
    iq_delay_time = config['Microwaves']['iq_delay']

    pulser_wiring = config['Wiring']['PulseGen']
    pulser_do_apd_gate = pulser_wiring['do_apd_gate']
    low_sig_gen_gate_chan_name = 'do_{}_gate'.format(low_sig_gen_name)
    pulser_do_sig_gen_low_gate = pulser_wiring[low_sig_gen_gate_chan_name]
    high_sig_gen_gate_chan_name = 'do_{}_gate'.format(high_sig_gen_name)
    pulser_do_sig_gen_high_gate = pulser_wiring[high_sig_gen_gate_chan_name]
    pulser_do_arb_wave_trigger = pulser_wiring['do_arb_wave_trigger']

    # %% Some further setup

    # Default the pulses to 0
    init_pi_low = 0
    init_pi_high = 0
    read_pi_low = 0
    read_pi_high = 0

    # This ensures that all sequences run the same duty cycle. It compensates
    # for finite pulse length.
    pi_pulse_buffer = max(5*pi_pulse_low, 5*pi_pulse_high)
    # pi_pulse_buffer = 0

    # Set pulse durations for Knill composite pulses
    if init_state_value == States.LOW.value:
        init_pi_low = 5*pi_pulse_low
    elif init_state_value == States.HIGH.value:
        init_pi_high = 5*pi_pulse_high
    if read_state_value == States.LOW.value:
        read_pi_low = 5*pi_pulse_low
    elif read_state_value == States.HIGH.value:
        read_pi_high = 5*pi_pulse_high

    base_uwave_experiment_dur = 2*pi_pulse_buffer
    uwave_experiment_shrt = base_uwave_experiment_dur + tau_shrt
    uwave_experiment_long = base_uwave_experiment_dur + tau_long

    prep_time = aom_delay_time + \
        polarization_time + pre_uwave_exp_wait_time + \
        uwave_experiment_shrt + post_uwave_exp_wait_time

    up_to_long_gates = prep_time + signal_time + sig_to_ref_wait_time + \
        reference_time + pre_uwave_exp_wait_time + \
        uwave_experiment_long + post_uwave_exp_wait_time
        
    # Microsecond buffer at the end where everything is off
    end_buffer = 1000

    # The period is independent of the particular tau, but it must be long
    # enough to accomodate the longest tau
    period = aom_delay_time + polarization_time + \
        pre_uwave_exp_wait_time + uwave_experiment_shrt + post_uwave_exp_wait_time + \
        signal_time + sig_to_ref_wait_time + reference_time + pre_uwave_exp_wait_time + \
        uwave_experiment_long + post_uwave_exp_wait_time + \
        signal_time + sig_to_ref_wait_time + gate_time + end_buffer

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
             (gate_time, HIGH),
             (end_buffer, LOW)]
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
             (gate_time + aom_delay_time, HIGH),
             (end_buffer, LOW)]
    tool_belt.process_laser_seq(pulse_streamer, seq, config, 
                                laser_name, laser_power, train)

    # %% Microwaves

    pre_duration = aom_delay_time + polarization_time + pre_uwave_exp_wait_time
    mid_duration = post_uwave_exp_wait_time + signal_time + sig_to_ref_wait_time + \
        reference_time + pre_uwave_exp_wait_time
    post_duration = post_uwave_exp_wait_time + signal_time + \
        sig_to_ref_wait_time + gate_time

    train = [(pre_duration - rf_high_delay, LOW)]
    train.extend([(init_pi_high, HIGH), (pi_pulse_buffer-init_pi_high + tau_shrt, LOW), (read_pi_high, HIGH)])
    train.extend([(pi_pulse_buffer-read_pi_high + mid_duration, LOW)])
    train.extend([(init_pi_high, HIGH), (pi_pulse_buffer-init_pi_high + tau_long, LOW), (read_pi_high, HIGH)])
    train.extend([(pi_pulse_buffer-read_pi_high + post_duration + rf_high_delay, LOW), (end_buffer, LOW)])
    seq.setDigital(pulser_do_sig_gen_high_gate, train)

    train = [(pre_duration - rf_low_delay, LOW)]
    train.extend([(init_pi_low, HIGH), (pi_pulse_buffer-init_pi_low + tau_shrt, LOW), (read_pi_low, HIGH)])
    train.extend([(pi_pulse_buffer-read_pi_low + mid_duration, LOW)])
    train.extend([(init_pi_low, HIGH), (pi_pulse_buffer-init_pi_low + tau_long, LOW), (read_pi_low, HIGH)])
    train.extend([(pi_pulse_buffer-read_pi_low + post_duration + rf_low_delay, LOW), (end_buffer, LOW)])
    seq.setDigital(pulser_do_sig_gen_low_gate, train)

    # %% IQ modulation

    composite_low_seq = [(10, HIGH), (pi_pulse_low-10, LOW)] * 5
    composite_high_seq = [(10, HIGH), (pi_pulse_high-10, LOW)] * 5

    train = [(pre_duration - iq_delay_time, LOW)]

    if init_state_value == States.LOW.value:
        train.extend(composite_low_seq)
        train.extend([(pi_pulse_buffer-init_pi_low, LOW)])
    elif init_state_value == States.HIGH.value:
        train.extend(composite_high_seq)
        train.extend([(pi_pulse_buffer-init_pi_high, LOW)])
    else:
        train.extend([(pi_pulse_buffer, LOW)])

    train.extend([(tau_shrt, LOW)])

    if read_state_value == States.LOW.value:
        train.extend(composite_low_seq)
        train.extend([(pi_pulse_buffer-read_pi_low, LOW)])
    elif read_state_value == States.HIGH.value:
        train.extend(composite_high_seq)
        train.extend([(pi_pulse_buffer-read_pi_high, LOW)])
    else:
        train.extend([(pi_pulse_buffer, LOW)])

    train.extend([(mid_duration, LOW)])

    if init_state_value == States.LOW.value:
        train.extend(composite_low_seq)
        train.extend([(pi_pulse_buffer-init_pi_low, LOW)])
    elif init_state_value == States.HIGH.value:
        train.extend(composite_high_seq)
        train.extend([(pi_pulse_buffer-init_pi_high, LOW)])
    else:
        train.extend([(pi_pulse_buffer, LOW)])

    train.extend([(tau_long, LOW)])

    if read_state_value == States.LOW.value:
        train.extend(composite_low_seq)
        train.extend([(pi_pulse_buffer-read_pi_low, LOW)])
    elif read_state_value == States.HIGH.value:
        train.extend(composite_high_seq)
        train.extend([(pi_pulse_buffer-read_pi_high, LOW)])
    else:
        train.extend([(pi_pulse_buffer, LOW)])

    train.extend([(post_duration + iq_delay_time, LOW), (end_buffer, LOW)])

    seq.setDigital(pulser_do_arb_wave_trigger, train)

    # %% Return the sequence

    final_digital = [pulser_wiring['do_sample_clock']]
    final = OutputState(final_digital, 0.0, 0.0)
    return seq, final, [str(period)]


if __name__ == '__main__':
    
    seq_args = [3000, 1000, 350, 121, 105, 7300, 1, 1, "laserglow_532", None]

    config = tool_belt.get_config_dict()
    tool_belt.set_delays_to_zero(config)
    
    seq, final, ret_vals = get_seq(None, config, seq_args)
    seq.plot()
