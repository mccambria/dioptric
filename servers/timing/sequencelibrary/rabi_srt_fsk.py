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

    # Get the APD indices
    dev_high_sign = args[6]
    dev_low_sign = args[7]
    
    apd_index = args[8]
    
    # Specify the initial and readout states
    init_state_value = args[9]
    read_state_value = args[10]
    
    laser_name = args[11]
    laser_power = args[12]
    
    low_sig_gen_name = config['Microwaves']['sig_gen_LOW']
    high_sig_gen_name = config['Microwaves']['sig_gen_HIGH']
    
    uwave_laser_buffer= 100e3#config['CommonDurations']['uwave_buffer']
    uwave_detune_buffer = 200e3#uwave_laser_buffer*10
    trigger_dur = 10
    
    aom_delay_time = config['Optics'][laser_name]['delay']
    
    rf_low_delay = config['Microwaves'][low_sig_gen_name]['delay']
    rf_high_delay = config['Microwaves'][high_sig_gen_name]['delay']
    
    
    pulser_wiring = config['Wiring']['PulseStreamer']
    pulser_do_apd_gate = pulser_wiring['do_apd_{}_gate'.format(apd_index)]
    low_sig_gen_gate_chan_name = 'do_{}_gate'.format(low_sig_gen_name)
    pulser_do_sig_gen_low_gate = pulser_wiring[low_sig_gen_gate_chan_name]
    high_sig_gen_gate_chan_name = 'do_{}_gate'.format(high_sig_gen_name)
    pulser_do_sig_gen_high_gate = pulser_wiring[high_sig_gen_gate_chan_name]
    
    pulser_do_fsk_trigger = pulser_wiring['do_fsk_trigger']
    # pulser_ao_fm_HIGH = pulser_wiring['ao_fm_{}'.format(high_sig_gen_name)]
    
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
            (uwave_srt_shrt, LOW),
            (uwave_detune_buffer, LOW),
            (read_pi_dur, LOW),
            (uwave_laser_buffer, LOW),
            
            (gate_time, HIGH),
            (polarization_time - gate_time, LOW),
            (uwave_laser_buffer, LOW),
            (init_pi_dur, LOW),
            (uwave_detune_buffer, LOW),
            (uwave_detune_buffer, LOW),
            (read_pi_dur, LOW),
            (uwave_laser_buffer, LOW),
            
            (gate_time, HIGH),
            (polarization_time - gate_time, LOW),
            (uwave_laser_buffer, LOW),
            (init_pi_dur, LOW),
            (uwave_detune_buffer, LOW),
            (uwave_srt_long, LOW),
            (uwave_detune_buffer, LOW),
            (read_pi_dur, LOW),
            (uwave_laser_buffer, LOW),
            
            (gate_time, HIGH),
            (polarization_time - gate_time, LOW),
            (uwave_laser_buffer, LOW),
            (init_pi_dur, LOW),
            (uwave_detune_buffer, LOW),
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
            (uwave_srt_shrt, LOW),
            (uwave_detune_buffer, LOW),
            (read_pi_dur, LOW),
            (uwave_laser_buffer, LOW),
            
            (polarization_time, HIGH),
            (uwave_laser_buffer, LOW),
            (init_pi_dur, LOW),
            (uwave_detune_buffer, LOW),
            (uwave_detune_buffer, LOW),
            (read_pi_dur, LOW),
            (uwave_laser_buffer, LOW),
            
            (polarization_time, HIGH),
            (uwave_laser_buffer, LOW),
            (init_pi_dur, LOW),
            (uwave_detune_buffer, LOW),
            (uwave_srt_long, LOW),
            (uwave_detune_buffer, LOW),
            (read_pi_dur, LOW),
            (uwave_laser_buffer, LOW),
            
            (polarization_time, HIGH),
            (uwave_laser_buffer, LOW),
            (init_pi_dur, LOW),
            (uwave_detune_buffer, LOW),
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
            (uwave_srt_shrt, HIGH),
            (uwave_detune_buffer, LOW),
            (read_pi_high, HIGH),
            (uwave_laser_buffer + read_pi_low, LOW),
            
            (polarization_time, LOW),
            (uwave_laser_buffer, LOW),
            (init_pi_high, LOW),
            (uwave_detune_buffer + init_pi_low, LOW),
            (uwave_detune_buffer, LOW),
            (read_pi_high, LOW), 
            (uwave_laser_buffer + read_pi_low, LOW),
            
            (polarization_time, LOW),
            (uwave_laser_buffer, LOW),
            (init_pi_high, HIGH),
            (uwave_detune_buffer + init_pi_low, LOW),
            (uwave_srt_long, HIGH),
            (uwave_detune_buffer, LOW),
            (read_pi_high, HIGH),
            (uwave_laser_buffer + read_pi_low, LOW),
            
            (polarization_time, LOW),
            (uwave_laser_buffer, LOW),
            (init_pi_high, LOW), 
            (uwave_detune_buffer + init_pi_low, LOW),
            (uwave_detune_buffer, LOW),
            (read_pi_high, LOW), 
            (uwave_laser_buffer + read_pi_low, LOW),
            (polarization_time, LOW) ,
            (back_buffer + rf_high_delay, LOW)
            ]
    seq.setDigital(pulser_do_sig_gen_high_gate, train)
    period = 0
    for el in train:
        period += el[0]
    print(period)
    
    # MW gate LOW
    train = [(delay_buffer - rf_low_delay, LOW),
            (polarization_time, LOW),
            (uwave_laser_buffer, LOW),
            (init_pi_low, HIGH),
            (uwave_detune_buffer + init_pi_high, LOW),
            (uwave_srt_shrt, HIGH),
            (uwave_detune_buffer, LOW),
            (read_pi_low, HIGH),
            (uwave_laser_buffer + read_pi_high, LOW),
            
            (polarization_time, LOW),
            (uwave_laser_buffer, LOW),
            (init_pi_low, LOW),
            (uwave_detune_buffer + init_pi_high, LOW),
            (uwave_detune_buffer, LOW),
            (read_pi_low, LOW),
            (uwave_laser_buffer + read_pi_high, LOW),
            
            (polarization_time, LOW),
            (uwave_laser_buffer, LOW),
            (init_pi_low, HIGH),
            (uwave_detune_buffer + init_pi_high, LOW),
            (uwave_srt_long, HIGH),
            (uwave_detune_buffer, LOW),
            (read_pi_low, HIGH),
            (uwave_laser_buffer + read_pi_high, LOW),
            
            (polarization_time, LOW),
            (uwave_laser_buffer, LOW),
            (init_pi_low, LOW),
            (uwave_detune_buffer + init_pi_high, LOW),
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
    
    
    # FSK trigger
    uwave_detune_buffer_half = int(uwave_detune_buffer/2)
    train = [
        (delay_buffer - rf_high_delay, LOW),
        (trigger_dur, HIGH),
            (polarization_time- trigger_dur, LOW),
            (uwave_laser_buffer, LOW),
            (init_pi_low + init_pi_high, LOW),
            (uwave_detune_buffer_half, LOW), 
            (trigger_dur, HIGH),
            (uwave_detune_buffer_half - trigger_dur, LOW), 
            (uwave_srt_shrt, LOW),
            (uwave_detune_buffer_half, LOW), 
            (trigger_dur, HIGH),
            (uwave_detune_buffer_half - trigger_dur, LOW), 
            (read_pi_low + read_pi_high, LOW),
            (uwave_laser_buffer, LOW),
            
            (polarization_time, LOW),
            (uwave_laser_buffer, LOW),
            (init_pi_low + init_pi_high, LOW),
            (uwave_detune_buffer, LOW),
            (uwave_detune_buffer, LOW),
            (read_pi_low + read_pi_high, LOW),
            (uwave_laser_buffer, LOW),
            
            (trigger_dur, HIGH),
            (polarization_time- trigger_dur, LOW),
            (uwave_laser_buffer, LOW),
            (init_pi_low + init_pi_high, LOW),
            (uwave_detune_buffer_half, LOW), 
            (trigger_dur, HIGH),
            (uwave_detune_buffer_half - trigger_dur, LOW), 
            (uwave_srt_long, LOW),
            (uwave_detune_buffer_half, LOW), 
            (trigger_dur, HIGH),
            (uwave_detune_buffer_half - trigger_dur, LOW), 
            (read_pi_low + read_pi_high, LOW),
            (uwave_laser_buffer, LOW),
            
            (polarization_time, LOW),
            (uwave_laser_buffer, LOW),
            (init_pi_low + init_pi_high, LOW),
            (uwave_detune_buffer, LOW), 
            (uwave_detune_buffer, LOW),
            (read_pi_low + read_pi_high, LOW),
            (uwave_laser_buffer, LOW),
            (polarization_time, LOW) ,
            (back_buffer + rf_high_delay, LOW)
            ]
    seq.setDigital(pulser_do_fsk_trigger, train)
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
    tool_belt.set_feedthroughs_to_false(config)
    
    
    seq_args = [0, 10000.0, 300, 65, 65, 200, 1, 1, 1, 3, 3, 'integrated_520', None]
    seq = get_seq(None, config, seq_args)[0]
    seq.plot()
