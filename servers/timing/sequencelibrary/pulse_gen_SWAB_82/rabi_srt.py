# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 14:44:30 2022

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
    uwave_srt_shrt, polarization_time,   \
            gate_time, pi_pulse_low, pi_pulse_high, uwave_srt_long = durations


    dev_high_sign = args[6]
    dev_low_sign = args[7]
    
    
    # Specify the initial and readout states
    init_state_value = args[8]
    read_state_value = args[9]
    
    laser_name = args[10]
    laser_power = args[11]
    
    low_sig_gen_name = config['Servers']['sig_gen_LOW']
    high_sig_gen_name = config['Servers']['sig_gen_HIGH']
    
    uwave_laser_buffer= config['CommonDurations']['uwave_buffer']
    fm_mod_bandwidth_LOW = config['Microwaves'][low_sig_gen_name]['fm_mod_bandwidth'] # in Hz
    fm_mod_bandwidth_HIGH = 100*config['Microwaves'][high_sig_gen_name]['fm_mod_bandwidth'] # in Hz
    min_fm_mod_bandwidth = min(fm_mod_bandwidth_LOW, fm_mod_bandwidth_HIGH)
    fm_bandwidth_buffer =  1/min_fm_mod_bandwidth * 1e9
    uwave_detune_buffer = uwave_laser_buffer
    # time between signal and reference without illumination
    
    aom_delay_time = config['Optics'][laser_name]['delay']
    
    rf_low_delay = config['Microwaves'][low_sig_gen_name]['delay']
    rf_high_delay = config['Microwaves'][high_sig_gen_name]['delay']
    
    
    pulser_wiring = config['Wiring']['PulseGen']
    pulser_do_apd_gate = pulser_wiring['do_apd_gate']
    low_sig_gen_gate_chan_name = 'do_{}_gate'.format(low_sig_gen_name)
    pulser_do_sig_gen_low_gate = pulser_wiring[low_sig_gen_gate_chan_name]
    high_sig_gen_gate_chan_name = 'do_{}_gate'.format(high_sig_gen_name)
    pulser_do_sig_gen_high_gate = pulser_wiring[high_sig_gen_gate_chan_name]
    
    pulser_ao_fm_LOW = pulser_wiring['ao_fm_{}'.format(low_sig_gen_name)]
    pulser_ao_fm_HIGH = pulser_wiring['ao_fm_{}'.format(high_sig_gen_name)]
    
    delay_buffer = max(aom_delay_time,rf_low_delay,rf_high_delay, 100)
    back_buffer = 200

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


    # %% Write the microwave sequence to be used.

    init_pi_dur = init_pi_high + init_pi_low
    read_pi_dur = read_pi_high + read_pi_low
    

    # %% Define the sequence

    seq = Sequence()

    # APD gating
    train = [(delay_buffer, LOW),
            (polarization_time, LOW),
            (uwave_laser_buffer, LOW),
            (init_pi_dur, LOW),
            (uwave_detune_buffer, LOW),
            (fm_bandwidth_buffer, LOW),
            (uwave_srt_shrt, LOW),
            (fm_bandwidth_buffer, LOW),
            (uwave_detune_buffer, LOW),
            (read_pi_dur, LOW),
            (uwave_laser_buffer, LOW),
            
            (gate_time, HIGH),
            (polarization_time - gate_time, LOW),
            (uwave_laser_buffer, LOW),
            (init_pi_dur, LOW),
            (uwave_detune_buffer, LOW),
            (fm_bandwidth_buffer, LOW),
            (fm_bandwidth_buffer, LOW),
            (uwave_detune_buffer, LOW),
            (read_pi_dur, LOW),
            (uwave_laser_buffer, LOW),
            
            (gate_time, HIGH),
            (polarization_time - gate_time, LOW),
            (uwave_laser_buffer, LOW),
            (init_pi_dur, LOW),
            (uwave_detune_buffer, LOW),
            (fm_bandwidth_buffer, LOW),
            (uwave_srt_long, LOW),
            (fm_bandwidth_buffer, LOW),
            (uwave_detune_buffer, LOW),
            (read_pi_dur, LOW),
            (uwave_laser_buffer, LOW),
            
            (gate_time, HIGH),
            (polarization_time - gate_time, LOW),
            (uwave_laser_buffer, LOW),
            (init_pi_dur, LOW),
            (uwave_detune_buffer, LOW),
            (fm_bandwidth_buffer, LOW),
            (fm_bandwidth_buffer, LOW),
            (uwave_detune_buffer, LOW),
            (read_pi_dur, LOW),
            (uwave_laser_buffer, LOW),
            (gate_time, HIGH) ,
            (polarization_time - gate_time, LOW) ,
            (back_buffer, LOW)
            ]
    seq.setDigital(pulser_do_apd_gate, train)
    period = 0
    for el in train:
        period += el[0]
    print(period)

    # Pulse the laser with the AOM for polarization and readout
    train = [(delay_buffer - aom_delay_time, LOW),
            (polarization_time, HIGH),
            (uwave_laser_buffer, LOW),
            (init_pi_dur, LOW),
            (uwave_detune_buffer, LOW),
            (fm_bandwidth_buffer, LOW),
            (uwave_srt_shrt, LOW),
            (fm_bandwidth_buffer, LOW),
            (uwave_detune_buffer, LOW),
            (read_pi_dur, LOW),
            (uwave_laser_buffer, LOW),
            
            (polarization_time, HIGH),
            (uwave_laser_buffer, LOW),
            (init_pi_dur, LOW),
            (uwave_detune_buffer, LOW),
            (fm_bandwidth_buffer, LOW),
            (fm_bandwidth_buffer, LOW),
            (uwave_detune_buffer, LOW),
            (read_pi_dur, LOW),
            (uwave_laser_buffer, LOW),
            
            (polarization_time, HIGH),
            (uwave_laser_buffer, LOW),
            (init_pi_dur, LOW),
            (uwave_detune_buffer, LOW),
            (fm_bandwidth_buffer, LOW),
            (uwave_srt_long, LOW),
            (fm_bandwidth_buffer, LOW),
            (uwave_detune_buffer, LOW),
            (read_pi_dur, LOW),
            (uwave_laser_buffer, LOW),
            
            (polarization_time, HIGH),
            (uwave_laser_buffer, LOW),
            (init_pi_dur, LOW),
            (uwave_detune_buffer, LOW),
            (fm_bandwidth_buffer, LOW),
            (fm_bandwidth_buffer, LOW),
            (uwave_detune_buffer, LOW),
            (read_pi_dur, LOW),
            (uwave_laser_buffer, LOW),
            (polarization_time, HIGH) ,
            (back_buffer + aom_delay_time, LOW)
            ]
    tool_belt.process_laser_seq(pulse_streamer, seq, config, 
                                laser_name, laser_power, train)
    period = 0
    for el in train:
        period += el[0]
    print(period)

    # Pulse the microwave for tau HIGH

    train = [(delay_buffer - rf_high_delay, LOW),
            (polarization_time, LOW),
            (uwave_laser_buffer, LOW),
            (init_pi_high, HIGH),
            (uwave_detune_buffer + init_pi_low, LOW),
            (fm_bandwidth_buffer, LOW),
            (uwave_srt_shrt, HIGH),
            (fm_bandwidth_buffer, LOW),
            (uwave_detune_buffer, LOW),
            (read_pi_high, HIGH),
            (uwave_laser_buffer + read_pi_low, LOW),
            
            (polarization_time, LOW),
            (uwave_laser_buffer, LOW),
            (init_pi_high, LOW), ####
            (uwave_detune_buffer + init_pi_low, LOW),
            (fm_bandwidth_buffer, LOW),
            (fm_bandwidth_buffer, LOW),
            (uwave_detune_buffer, LOW),
            (read_pi_high, LOW), ####
            (uwave_laser_buffer + read_pi_low, LOW),
            
            (polarization_time, LOW),
            (uwave_laser_buffer, LOW),
            (init_pi_high, HIGH),
            (uwave_detune_buffer + init_pi_low, LOW),
            (fm_bandwidth_buffer, LOW),
            (uwave_srt_long, HIGH),
            (fm_bandwidth_buffer, LOW),
            (uwave_detune_buffer, LOW),
            (read_pi_high, HIGH),
            (uwave_laser_buffer + read_pi_low, LOW),
            
            (polarization_time, LOW),
            (uwave_laser_buffer, LOW),
            (init_pi_high, LOW), ####
            (uwave_detune_buffer + init_pi_low, LOW),
            (fm_bandwidth_buffer, LOW),
            (fm_bandwidth_buffer, LOW),
            (uwave_detune_buffer, LOW),
            (read_pi_high, LOW), ####
            (uwave_laser_buffer + read_pi_low, LOW),
            (polarization_time, LOW) ,
            (back_buffer + rf_high_delay, LOW)
            ]
    seq.setDigital(pulser_do_sig_gen_high_gate, train)
    period = 0
    print(train)
    for el in train:
        period += el[0]
    print(period)
    
    # MW gate LOW
    train = [(delay_buffer - rf_low_delay, LOW),
            (polarization_time, LOW),
            (uwave_laser_buffer, LOW),
            (init_pi_low, HIGH),
            (uwave_detune_buffer + init_pi_high, LOW),
            (fm_bandwidth_buffer, LOW),
            (uwave_srt_shrt, HIGH),
            (fm_bandwidth_buffer, LOW),
            (uwave_detune_buffer, LOW),
            (read_pi_low, HIGH),
            (uwave_laser_buffer + read_pi_high, LOW),
            
            (polarization_time, LOW),
            (uwave_laser_buffer, LOW),
            (init_pi_low, LOW),
            (uwave_detune_buffer + init_pi_high, LOW),
            (fm_bandwidth_buffer, LOW),
            (fm_bandwidth_buffer, LOW),
            (uwave_detune_buffer, LOW),
            (read_pi_low, LOW),
            (uwave_laser_buffer + read_pi_high, LOW),
            
            (polarization_time, LOW),
            (uwave_laser_buffer, LOW),
            (init_pi_low, HIGH),
            (uwave_detune_buffer + init_pi_high, LOW),
            (fm_bandwidth_buffer, LOW),
            (uwave_srt_long, HIGH),
            (fm_bandwidth_buffer, LOW),
            (uwave_detune_buffer, LOW),
            (read_pi_low, HIGH),
            (uwave_laser_buffer + read_pi_high, LOW),
            
            (polarization_time, LOW),
            (uwave_laser_buffer, LOW),
            (init_pi_low, LOW),
            (uwave_detune_buffer + init_pi_high, LOW),
            (fm_bandwidth_buffer, LOW),
            (fm_bandwidth_buffer, LOW),
            (uwave_detune_buffer, LOW),
            (read_pi_low, LOW),
            (uwave_laser_buffer + read_pi_high, LOW),
            (polarization_time, LOW) ,
            (back_buffer + rf_low_delay, LOW)
            ]
    seq.setDigital(pulser_do_sig_gen_low_gate, train)
    period = 0
    for el in train:
        period += el[0]
    print(period)
    
    if False:
        HIGH_fm_high = 1.0 * dev_high_sign
        # Turn on the FM for HIGH sig gen
        train = [
            (delay_buffer - rf_high_delay, LOW),
                (polarization_time, LOW),
                (uwave_laser_buffer, LOW),
                (init_pi_high, LOW),
                (uwave_detune_buffer + init_pi_low, LOW), 
                (fm_bandwidth_buffer, HIGH_fm_high),
                (uwave_srt_shrt, HIGH_fm_high),
                (fm_bandwidth_buffer, HIGH_fm_high),
                (uwave_detune_buffer, LOW),
                (read_pi_high, LOW),
                (uwave_laser_buffer + read_pi_low, LOW),
                
                (polarization_time, LOW),
                (uwave_laser_buffer, LOW),
                (init_pi_high, LOW),
                (uwave_detune_buffer + init_pi_low, LOW),
                (fm_bandwidth_buffer, LOW),
                (fm_bandwidth_buffer, LOW),
                (uwave_detune_buffer, LOW),
                (read_pi_high, LOW),
                (uwave_laser_buffer + read_pi_low, LOW),
                
                (polarization_time, LOW),
                (uwave_laser_buffer, LOW),
                (init_pi_high, LOW),
                (uwave_detune_buffer + init_pi_low, LOW),
                (fm_bandwidth_buffer, HIGH_fm_high),
                (uwave_srt_long, HIGH_fm_high),
                (fm_bandwidth_buffer, HIGH_fm_high),
                (uwave_detune_buffer, LOW),
                (read_pi_high, LOW),
                (uwave_laser_buffer + read_pi_low, LOW),
                
                (polarization_time, LOW),
                (uwave_laser_buffer, LOW),
                (init_pi_high, LOW),
                (uwave_detune_buffer + init_pi_low, LOW), 
                (fm_bandwidth_buffer, LOW),
                (fm_bandwidth_buffer, LOW),
                (uwave_detune_buffer, LOW),
                (read_pi_high, LOW),
                (uwave_laser_buffer + read_pi_low, LOW),
                (polarization_time, LOW) ,
                (back_buffer + rf_high_delay, LOW)
                ]
        seq.setAnalog(pulser_ao_fm_HIGH, train)
        period = 0
        for el in train:
            period += el[0]
        print(period)
    
    # Turn on the FM for LOW sig gen
    HIGH_fm_low = 1.0 * dev_low_sign
    train = [
        (delay_buffer - rf_high_delay, LOW),
            (polarization_time, LOW),
            (uwave_laser_buffer, LOW),
            (init_pi_low, LOW),
            (uwave_detune_buffer + init_pi_high, LOW), 
            (fm_bandwidth_buffer, HIGH_fm_low),
            (uwave_srt_shrt, HIGH_fm_low),
            (fm_bandwidth_buffer, HIGH_fm_low),
            (uwave_detune_buffer, LOW),
            (read_pi_low, LOW),
            (uwave_laser_buffer + read_pi_high, LOW),
            
            (polarization_time, LOW),
            (uwave_laser_buffer, LOW),
            (init_pi_low, LOW),
            (uwave_detune_buffer + init_pi_high, LOW),
            (fm_bandwidth_buffer, LOW),
            (fm_bandwidth_buffer, LOW),
            (uwave_detune_buffer, LOW),
            (read_pi_low, LOW),
            (uwave_laser_buffer + read_pi_high, LOW),
            
            (polarization_time, LOW),
            (uwave_laser_buffer, LOW),
            (init_pi_low, LOW),
            (uwave_detune_buffer + init_pi_high, LOW),
            (fm_bandwidth_buffer, HIGH_fm_low),
            (uwave_srt_long, HIGH_fm_low),
            (fm_bandwidth_buffer, HIGH_fm_low),
            (uwave_detune_buffer, LOW),
            (read_pi_low, LOW),
            (uwave_laser_buffer + read_pi_high, LOW),
            
            (polarization_time, LOW),
            (uwave_laser_buffer, LOW),
            (init_pi_low, LOW),
            (uwave_detune_buffer + init_pi_high, LOW), 
            (fm_bandwidth_buffer, LOW),
            (fm_bandwidth_buffer, LOW),
            (uwave_detune_buffer, LOW),
            (read_pi_low, LOW),
            (uwave_laser_buffer + read_pi_high, LOW),
            (polarization_time, LOW) ,
            (back_buffer + rf_high_delay, LOW)
            ]
    seq.setAnalog(pulser_ao_fm_LOW, train)
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
    # tool_belt.set_feedthroughs_to_false(config)
    
    
    seq_args = [0, 10000.0, 300, 68, 68, 300, 1, 1, 1.0, 1, 3, 'integrated_520', None]
    seq = get_seq(None, config, seq_args)[0]
    seq.plot()
