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
    for ind in range(8):
        durations.append(numpy.int64(args[ind]))

    # Unpack the durations
    tau_shrt, polarization_time, gate_time, pi_pulse_low_dur, pi_on_2_pulse_low_dur, \
        pi_pulse_high_dur, pi_on_2_pulse_high_dur, tau_long = durations

    # Get the APD indices
    state_ini, state_opp, laser_name, laser_power, do_ramsey = args[8:13]
    state_ini = States(state_ini)
    state_opp = States(state_opp)
        
    uwave_buffer = config['CommonDurations']['uwave_buffer']
    laser_delay = config['Optics'][laser_name]['delay']
    sig_gen_low_name = config['Servers']['sig_gen_LOW']
    uwave_delay_low = config['Microwaves'][sig_gen_low_name]['delay']
    sig_gen_high_name = config['Servers']['sig_gen_HIGH']
    uwave_delay_high = config['Microwaves'][sig_gen_high_name]['delay']
    short_buffer = 10
    common_delay = max(laser_delay, uwave_delay_low, uwave_delay_high) + short_buffer
    back_buffer = 200

    pulser_wiring = config['Wiring']['PulseGen']
    pulser_do_apd_gate = pulser_wiring['do_apd_gate']
    sig_gen_gate_chan_name_low = 'do_{}_gate'.format(sig_gen_low_name)
    pulser_do_sig_gen_gate_low = pulser_wiring[sig_gen_gate_chan_name_low]
    sig_gen_gate_chan_name_high = 'do_{}_gate'.format(sig_gen_high_name)
    pulser_do_sig_gen_gate_high = pulser_wiring[sig_gen_gate_chan_name_high]


    init_pi_pulse_low = 0
    read_pi_pulse_low = 0
    init_pi_pulse_high = 0
    read_pi_pulse_high = 0
    
    # echo_pi_pulse_low_1 = 0
    # echo_pi_pulse_low_2 = 0
    # echo_pi_pulse_high_1 = 0
    # echo_pi_pulse_high_2 = 0
    
    
    if state_ini.value == States.LOW.value:
        init_pi_pulse_low = pi_pulse_low_dur
    elif state_ini.value == States.HIGH.value:
        init_pi_pulse_high = pi_pulse_high_dur
        
    if state_opp.value == States.LOW.value:
        read_pi_pulse_low = pi_pulse_low_dur
    elif state_opp.value == States.HIGH.value:
        read_pi_pulse_high = pi_pulse_high_dur
        
    
    swap_pi_pulse_low_1 = pi_pulse_low_dur
    swap_pi_pulse_low_2 = 0
    swap_pi_pulse_high_1 = 0
    swap_pi_pulse_high_2 = pi_pulse_high_dur
    
    
        
    uwave_init_pulse = init_pi_pulse_low + init_pi_pulse_high
    uwave_read_pulse = read_pi_pulse_low + read_pi_pulse_high
    
    uwave_swap_pulse_dur = swap_pi_pulse_low_1 + swap_pi_pulse_low_2 + \
                swap_pi_pulse_high_1 + swap_pi_pulse_high_2 + \
                swap_pi_pulse_low_1 + swap_pi_pulse_high_1



    ### Define the sequence

    seq = Sequence()

    # APD gating
    train = [(common_delay, LOW),
             (polarization_time, LOW),
             (uwave_buffer, LOW),
             (uwave_init_pulse, LOW),
             (tau_shrt, LOW),
             (uwave_swap_pulse_dur, LOW),
             (tau_shrt, LOW),
             (uwave_read_pulse, LOW),
             (uwave_buffer, LOW),
             (gate_time, HIGH),
             (polarization_time-gate_time, LOW),
             (uwave_buffer, LOW),
             (uwave_buffer, LOW),
             (gate_time, HIGH),
             (polarization_time-gate_time, LOW),
             (uwave_buffer, LOW),
             (uwave_init_pulse, LOW),
             (tau_long, LOW),
             (uwave_swap_pulse_dur, LOW),
             (tau_long, LOW),
             (uwave_read_pulse, LOW),
             (uwave_buffer, LOW),
             (gate_time, HIGH),
             (polarization_time-gate_time, LOW),
             (uwave_buffer, LOW),
             (uwave_buffer, LOW),
             (gate_time, HIGH),
             (polarization_time-gate_time, LOW),
             (back_buffer, LOW)]
    seq.setDigital(pulser_do_apd_gate, train)
    period = 0
    for el in train:
        period += el[0]
    print(period)

    # Laser
    train = [(common_delay - laser_delay, LOW),
             (polarization_time, HIGH),
             (uwave_buffer, LOW),
             (uwave_init_pulse, LOW),
             (tau_shrt, LOW),
             (uwave_swap_pulse_dur, LOW),
             (tau_shrt, LOW),
             (uwave_read_pulse, LOW),
             (uwave_buffer, LOW),
             (polarization_time, HIGH),
             (uwave_buffer, LOW),
             (uwave_buffer, LOW),
             (polarization_time, HIGH),
             (uwave_buffer, LOW),
             (uwave_init_pulse, LOW),
             (tau_long, LOW),
             (uwave_swap_pulse_dur, LOW),
             (tau_long, LOW),
             (uwave_read_pulse, LOW),
             (uwave_buffer, LOW),
             (polarization_time, HIGH),
             (uwave_buffer, LOW),
             (uwave_buffer, LOW),
             (polarization_time, HIGH),
             (back_buffer + laser_delay, LOW)]
    tool_belt.process_laser_seq(pulse_streamer, seq, config,
                                laser_name, laser_power, train)
    period = 0
    for el in train:
        period += el[0]
    print(period)

    # Microwaves for LOW sig gen
    train = [(common_delay - uwave_delay_low, LOW),
             (polarization_time, LOW),
             
             (uwave_buffer, LOW),
             
             (init_pi_pulse_low, HIGH),
             (init_pi_pulse_high, LOW),
             
             (tau_shrt, LOW),
             
             (swap_pi_pulse_low_1, HIGH),
             (swap_pi_pulse_high_1, LOW),
             (swap_pi_pulse_low_2, HIGH),
             (swap_pi_pulse_high_2, LOW),
             (swap_pi_pulse_low_1, HIGH),
             (swap_pi_pulse_high_1, LOW),
             
             (tau_shrt, LOW),
             
             (read_pi_pulse_low, HIGH),
             (read_pi_pulse_high, LOW),
             
             (uwave_buffer, LOW),
             
             (polarization_time, LOW),
             (uwave_buffer, LOW),
             (uwave_buffer, LOW),
             
             (polarization_time, LOW),
             
             (uwave_buffer, LOW),
             
             (init_pi_pulse_low, HIGH),
             (init_pi_pulse_high, LOW),
             
             (tau_long, LOW),
             
             (swap_pi_pulse_low_1, HIGH),
             (swap_pi_pulse_high_1, LOW),
             (swap_pi_pulse_low_2, HIGH),
             (swap_pi_pulse_high_2, LOW),
             (swap_pi_pulse_low_1, HIGH),
             (swap_pi_pulse_high_1, LOW),
             
             (tau_long, LOW),
             
             (read_pi_pulse_low, HIGH),
             (read_pi_pulse_high, LOW),
             
             (uwave_buffer, LOW),
             
             (polarization_time, LOW),
             (uwave_buffer, LOW),
             (uwave_buffer, LOW),
             (polarization_time, LOW),
             (back_buffer + uwave_delay_low, LOW)]
    seq.setDigital(pulser_do_sig_gen_gate_low, train)
    period = 0
    for el in train:
        period += el[0]
    print(period)

    # Microwaves for HIGH sig gen
    train = [(common_delay - uwave_delay_high, LOW),
             (polarization_time, LOW),
             
             (uwave_buffer, LOW),
             
             (init_pi_pulse_low, LOW),
             (init_pi_pulse_high, HIGH),
             
             (tau_shrt, LOW),
             
             (swap_pi_pulse_low_1, LOW),
             (swap_pi_pulse_high_1, HIGH),
             (swap_pi_pulse_low_2, LOW),
             (swap_pi_pulse_high_2, HIGH),
             (swap_pi_pulse_low_1, LOW),
             (swap_pi_pulse_high_1, HIGH),
             
             (tau_shrt, LOW),
             
             (read_pi_pulse_low, LOW),
             (read_pi_pulse_high, HIGH),
             
             (uwave_buffer, LOW),
             
             (polarization_time, LOW),
             (uwave_buffer, LOW),
             (uwave_buffer, LOW),
             
             (polarization_time, LOW),
             
             (uwave_buffer, LOW),
             
             (init_pi_pulse_low, LOW),
             (init_pi_pulse_high, HIGH),
             
             (tau_long, LOW),
             
             (swap_pi_pulse_low_1, LOW),
             (swap_pi_pulse_high_1, HIGH),
             (swap_pi_pulse_low_2, LOW),
             (swap_pi_pulse_high_2, HIGH),
             (swap_pi_pulse_low_1, LOW),
             (swap_pi_pulse_high_1, HIGH),
             
             (tau_long, LOW),
             
             (read_pi_pulse_low, LOW),
             (read_pi_pulse_high, HIGH),
             
             (uwave_buffer, LOW),
             
             (polarization_time, LOW),
             (uwave_buffer, LOW),
             (uwave_buffer, LOW),
             
             (polarization_time, LOW),
             (back_buffer + uwave_delay_high, LOW)]
    seq.setDigital(pulser_do_sig_gen_gate_high, train)
    period = 0
    for el in train:
        period += el[0]
    print(period)
    
    final_digital = [pulser_wiring['do_sample_clock']]
    final = OutputState(final_digital, 0.0, 0.0)
    return seq, final, [period]

if __name__ == '__main__':
    config = tool_belt.get_config_dict()
    tool_belt.set_delays_to_zero(config)
    
    
    seq_args = [0, 10000.0, 300, 66, 33, 68, 34, 100000, 1, 3, 'integrated_520', None, False]
    seq, final, ret_vals = get_seq(None, config, seq_args)
    seq.plot()
