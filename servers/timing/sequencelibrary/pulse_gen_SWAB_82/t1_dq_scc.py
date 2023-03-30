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
        val = args[ind]
        val = 0 if val == None else val
        durations.append(numpy.int64(val))

    # Unpack the durations
    (
        tau_shrt,
        _,
        _,
        pi_pulse_low,
        pi_pulse_high,
        tau_long,
    ) = durations

    # Get other arguments
    (
        init_state_value,
        read_state_value,
        _,
        _,
    ) = args[6:10]

    # Get the scc args
    (
        pol_laser_name,
        pol_laser_power,
        polarization_dur,
        ion_laser_name,
        ion_laser_power,
        ionization_dur,
        shelf_laser_name,
        shelf_laser_power, # removed the power from the sequence, the power is always set to LOW
        shelf_dur,
        readout_laser_name,
        readout_laser_power,
        readout_dur,
    ) = args[10:]
    ionization_dur = numpy.int64(ionization_dur)
    shelf_dur = numpy.int64(shelf_dur)
    readout_dur = numpy.int64(readout_dur)

    uwave_buffer = config["CommonDurations"]["uwave_buffer"]
    pre_uwave_exp_wait_time = uwave_buffer
    post_uwave_exp_wait_time = uwave_buffer
    scc_ion_readout_buffer = config["CommonDurations"][
        "scc_ion_readout_buffer"
    ]
    # time between signal and reference without illumination
    sig_to_ref_wait_time = pre_uwave_exp_wait_time + post_uwave_exp_wait_time

    pol_laser_delay = config["Optics"][pol_laser_name]["delay"]
    ion_laser_delay = config["Optics"][ion_laser_name]["delay"]
    shelf_laser_delay = config["Optics"][shelf_laser_name]["delay"]
    readout_laser_delay = config["Optics"][readout_laser_name]["delay"]
    state_low = States.LOW
    state_high = States.HIGH
    low_sig_gen_name = config['Servers']['sig_gen_{}'.format(state_low.name)]
    high_sig_gen_name = config['Servers']['sig_gen_{}'.format(state_high.name)]

    rf_low_delay = config["Microwaves"][low_sig_gen_name]["delay"]
    rf_high_delay = config["Microwaves"][high_sig_gen_name]["delay"]

    common_delay = (
        max(
            pol_laser_delay,
            ion_laser_delay,
            shelf_laser_delay,
            readout_laser_delay,
            rf_low_delay,
            rf_high_delay,
        )
        + 100
    )

    pulser_wiring = config['Wiring']['PulseGen']
    pulser_do_apd_gate = pulser_wiring["do_apd_gate"]
    low_sig_gen_gate_chan_name = "do_{}_gate".format(low_sig_gen_name)
    pulser_do_sig_gen_low_gate = pulser_wiring[low_sig_gen_gate_chan_name]
    high_sig_gen_gate_chan_name = "do_{}_gate".format(high_sig_gen_name)
    pulser_do_sig_gen_high_gate = pulser_wiring[high_sig_gen_gate_chan_name]
    # readout_laser_gate = pulser_wiring["ao_{}_am".format(readout_laser_name)]

    # %% Some further setup

    # Default the pulses to 0
    init_pi_low = 0
    init_pi_high = 0
    read_pi_low = 0
    read_pi_high = 0

    # This ensures that all sequences run the same duty cycle. It compensates
    # for finite pulse length.
    pi_pulse_buffer = max(pi_pulse_low, pi_pulse_high)
    # pi_pulse_buffer = 0

    total_readout_dur = (
        shelf_dur + ionization_dur + scc_ion_readout_buffer + readout_dur
    )

    # Set pi pulse durations
    if init_state_value == States.LOW.value:
        init_pi_low = pi_pulse_low
    elif init_state_value == States.HIGH.value:
        init_pi_high = pi_pulse_high
    if read_state_value == States.LOW.value:
        read_pi_low = pi_pulse_low
    elif read_state_value == States.HIGH.value:
        read_pi_high = pi_pulse_high

    base_uwave_experiment_dur = 2 * pi_pulse_buffer
    uwave_experiment_shrt = base_uwave_experiment_dur + tau_shrt
    uwave_experiment_long = base_uwave_experiment_dur + tau_long

    prep_time = (
        common_delay
        + polarization_dur
        + pre_uwave_exp_wait_time
        + uwave_experiment_shrt
        + post_uwave_exp_wait_time
    )

    up_to_long_gates = (
        prep_time
        + total_readout_dur
        + polarization_dur
        + sig_to_ref_wait_time
        + total_readout_dur
        + polarization_dur
        + pre_uwave_exp_wait_time
        + uwave_experiment_long
        + post_uwave_exp_wait_time
    )

    # Microsecond buffer at the end of each experiment where everything is off
    end_buffer = 1000
    # end_buffer = 0

    # The period is independent of the particular tau, but it must be long
    # enough to accomodate the longest tau
    period = numpy.int64(
        common_delay
        + polarization_dur
        + pre_uwave_exp_wait_time
        + uwave_experiment_shrt
        + post_uwave_exp_wait_time
        + total_readout_dur
        + end_buffer
        + polarization_dur
        + sig_to_ref_wait_time
        + total_readout_dur
        + end_buffer
        + polarization_dur
        + pre_uwave_exp_wait_time
        + uwave_experiment_long
        + post_uwave_exp_wait_time
        + total_readout_dur
        + end_buffer
        + polarization_dur
        + sig_to_ref_wait_time
        + total_readout_dur
        + end_buffer
    )

    seq = Sequence()

    # %% APD

    pre_duration = prep_time
    short_sig_to_short_ref = (
        total_readout_dur
        + polarization_dur
        + sig_to_ref_wait_time
        - total_readout_dur
    )
    short_ref_to_long_sig = up_to_long_gates - (
        prep_time
        + total_readout_dur
        + polarization_dur
        + sig_to_ref_wait_time
        + total_readout_dur
    )
    long_sig_to_long_ref = (
        total_readout_dur
        + polarization_dur
        + sig_to_ref_wait_time
        - total_readout_dur
    )
    train = [
        (pre_duration, LOW),
        (shelf_dur + ionization_dur + scc_ion_readout_buffer, LOW),
        (readout_dur, HIGH),
        (end_buffer, LOW),
        (short_sig_to_short_ref, LOW),
        (shelf_dur + ionization_dur + scc_ion_readout_buffer, LOW),
        (readout_dur, HIGH),
        (end_buffer, LOW),
        (short_ref_to_long_sig, LOW),
        (shelf_dur + ionization_dur + scc_ion_readout_buffer, LOW),
        (readout_dur, HIGH),
        (end_buffer, LOW),
        (long_sig_to_long_ref, LOW),
        (shelf_dur + ionization_dur + scc_ion_readout_buffer, LOW),
        (readout_dur, HIGH),
        (end_buffer, LOW),
    ]
    seq.setDigital(pulser_do_apd_gate, train)
    # durs_only = [el[0] for el in train]
    # total_dur = sum(durs_only)
    # print(total_dur)

    # %% Polarization (green laser)

    train = [
        (common_delay - pol_laser_delay, LOW),
        (polarization_dur, HIGH),
        (
            pre_uwave_exp_wait_time
            + uwave_experiment_shrt
            + post_uwave_exp_wait_time,
            LOW,
        ),
        (total_readout_dur, LOW),
        (end_buffer, LOW),
        (polarization_dur, HIGH),
        (sig_to_ref_wait_time, LOW),
        (total_readout_dur, LOW),
        (end_buffer, LOW),
        (polarization_dur, HIGH),
        (
            pre_uwave_exp_wait_time
            + uwave_experiment_long
            + post_uwave_exp_wait_time,
            LOW,
        ),
        (total_readout_dur, LOW),
        (end_buffer, LOW),
        (polarization_dur, HIGH),
        (sig_to_ref_wait_time, LOW),
        (total_readout_dur + pol_laser_delay, LOW),
        (end_buffer, LOW),
    ]
    tool_belt.process_laser_seq(
        pulse_streamer,
        seq,
        config,
        pol_laser_name,
        pol_laser_power,
        train,
    )
    # durs_only = [el[0] for el in train]
    # total_dur = sum(durs_only)
    # print(total_dur)

    # %% Ionization (red laser)

    train = [
        (pre_duration - ion_laser_delay, LOW),
        (shelf_dur, LOW),
        (ionization_dur, HIGH),
        (scc_ion_readout_buffer + readout_dur, LOW),
        (end_buffer, LOW),
        (short_sig_to_short_ref, LOW),
        (shelf_dur, LOW),
        (ionization_dur, HIGH),
        (scc_ion_readout_buffer + readout_dur, LOW),
        (end_buffer, LOW),
        (short_ref_to_long_sig, LOW),
        (shelf_dur, LOW),
        (ionization_dur, HIGH),
        (scc_ion_readout_buffer + readout_dur, LOW),
        (end_buffer, LOW),
        (long_sig_to_long_ref, LOW),
        (shelf_dur, LOW),
        (ionization_dur, HIGH),
        (scc_ion_readout_buffer + readout_dur, LOW),
        (end_buffer + ion_laser_delay, LOW),
    ]
    tool_belt.process_laser_seq(
        pulse_streamer,
        seq,
        config,
        ion_laser_name,
        ion_laser_power,
        train,
    )
    # durs_only = [el[0] for el in train]
    # total_dur = sum(durs_only)
    # print(total_dur)

    # %% Shelf/readout (yellow laser)

    train = [
        (pre_duration - readout_laser_delay, LOW),
        (shelf_dur, LOW),
        (ionization_dur + scc_ion_readout_buffer, LOW),
        (readout_dur, HIGH),
        (end_buffer, LOW),
        (short_sig_to_short_ref, LOW),
        (shelf_dur, LOW),
        (ionization_dur + scc_ion_readout_buffer, LOW),
        (readout_dur, HIGH),
        (end_buffer, LOW),
        (short_ref_to_long_sig, LOW),
        (shelf_dur, LOW),
        (ionization_dur + scc_ion_readout_buffer, LOW),
        (readout_dur, HIGH),
        (end_buffer, LOW),
        (long_sig_to_long_ref, LOW),
        (shelf_dur, LOW),
        (ionization_dur + scc_ion_readout_buffer, LOW),
        (readout_dur, HIGH),
        (end_buffer + readout_laser_delay, LOW),
    ]
    tool_belt.process_laser_seq(
        pulse_streamer,
        seq,
        config,
        readout_laser_name,
        readout_laser_power,
        train,
    )
    # durs_only = [el[0] for el in train]
    # total_dur = sum(durs_only)
    # print(total_dur)

    # %% Microwaves

    pre_duration = common_delay + polarization_dur + pre_uwave_exp_wait_time
    mid_duration = (
        post_uwave_exp_wait_time
        + polarization_dur
        + total_readout_dur
        + end_buffer
        + sig_to_ref_wait_time
        + polarization_dur
        + total_readout_dur
        + end_buffer
        + pre_uwave_exp_wait_time
    )
    post_duration = (
        post_uwave_exp_wait_time
        + polarization_dur
        + total_readout_dur
        + end_buffer
        + sig_to_ref_wait_time
        + total_readout_dur
    )

    train = [(pre_duration - rf_high_delay, LOW)]
    train.extend(
        [
            (init_pi_high, HIGH),
            (pi_pulse_buffer - init_pi_high + tau_shrt, LOW),
            (read_pi_high, HIGH),
        ]
    )
    train.extend([(pi_pulse_buffer - read_pi_high + mid_duration, LOW)])
    train.extend(
        [
            (init_pi_high, HIGH),
            (pi_pulse_buffer - init_pi_high + tau_long, LOW),
            (read_pi_high, HIGH),
        ]
    )
    train.extend(
        [
            (
                pi_pulse_buffer - read_pi_high + post_duration + rf_high_delay,
                LOW,
            ),
            (end_buffer, LOW),
        ]
    )
    seq.setDigital(pulser_do_sig_gen_high_gate, train)
    # durs_only = [el[0] for el in train]
    # total_dur = sum(durs_only)
    # print(total_dur)

    train = [(pre_duration - rf_low_delay, LOW)]
    train.extend(
        [
            (init_pi_low, HIGH),
            (pi_pulse_buffer - init_pi_low + tau_shrt, LOW),
            (read_pi_low, HIGH),
        ]
    )
    train.extend([(pi_pulse_buffer - read_pi_low + mid_duration, LOW)])
    train.extend(
        [
            (init_pi_low, HIGH),
            (pi_pulse_buffer - init_pi_low + tau_long, LOW),
            (read_pi_low, HIGH),
        ]
    )
    train.extend(
        [
            (
                pi_pulse_buffer - read_pi_low + post_duration + rf_low_delay,
                LOW,
            ),
            (end_buffer, LOW),
        ]
    )
    seq.setDigital(pulser_do_sig_gen_low_gate, train)
    # durs_only = [el[0] for el in train]
    # total_dur = sum(durs_only)
    # print(total_dur)

    # %% Return the sequence

    final_digital = [pulser_wiring["do_sample_clock"]]
    final = OutputState(final_digital, 0.0, 0.0)
    return seq, final, [str(period)]


if __name__ == "__main__":

    config = tool_belt.get_config_dict()
    tool_belt.set_delays_to_zero(config) 
    # tool_belt.set_feedthroughs_to_false(config)
    config["CommonDurations"]["scc_ion_readout_buffer"] = 1000
    
    args = [100, None, None, 50, 50, 1000,  1, 0, None, None, 
            'integrated_520', None, 1000.0, 'cobolt_638', None, 400, 
            'laser_LGLO_589', 1.0, 0, 'laser_LGLO_589', 1.0, 100.0]
    # args = [2000, None, None, 67, 91, 4000, 1, 1, 3, None, None, 
    #         'laserglow_532', None, 1000.0, 'cobolt_638', None, 200, 
    #         'laserglow_589', 1.0, 0, 'laserglow_589', 1.0, 1000.0]

    seq, final, ret_vals = get_seq(None, config, args)
    seq.plot()
