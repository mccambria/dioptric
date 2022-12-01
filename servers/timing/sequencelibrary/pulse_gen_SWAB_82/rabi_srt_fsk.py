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
            gate_time, \
                init_pi_dur, read_pi_dur, uwave_srt_long = durations

    # Get the APD indices    
    apd_index = args[6]
    
    laser_name = args[7]
    laser_power = args[8]
    
    single_sig_gen_name=config['Microwaves']['sig_gen_single']
    omni_sig_gen_name=config['Microwaves']['sig_gen_omni']
    
    # uwave_laser_buffer= config['CommonDurations']['uwave_buffer']
    detune_buffer = 200e3
    trigger_dur = 10
    
    aom_delay_time = config['Optics'][laser_name]['delay']
    
    rf_single_delay = config['Microwaves'][single_sig_gen_name]['delay']
    rf_omni_delay = config['Microwaves'][omni_sig_gen_name]['delay']
    fsk_trig_delay= config['Microwaves']['fsk_trig_delay']
    
    
    pulser_wiring = config['Wiring']['PulseStreamer']
    pulser_do_apd_gate = pulser_wiring['do_apd_{}_gate'.format(apd_index)]
    omni_sig_gen_gate_chan_name = 'do_{}_gate'.format(omni_sig_gen_name)
    pulser_do_sig_gen_omni_gate = pulser_wiring[omni_sig_gen_gate_chan_name]
    single_sig_gen_gate_chan_name = 'do_{}_gate'.format(single_sig_gen_name)
    pulser_do_sig_gen_single_gate = pulser_wiring[single_sig_gen_gate_chan_name]
    
    pulser_do_fsk_trigger = pulser_wiring['do_fsk_trigger']
    # pulser_ao_fm_HIGH = pulser_wiring['ao_fm_{}'.format(high_sig_gen_name)]
    
    delay_buffer = max(aom_delay_time,rf_single_delay,rf_omni_delay, 100)
    back_buffer = 200

    # # Default the pulses to 0
    # init_pi_low = 0
    # init_pi_high = 0
    # read_pi_low = 0
    # read_pi_high = 0

    # if init_state_value == States.LOW.value:
    #     init_pi_low = pi_pulse_low
    # elif init_state_value == States.HIGH.value:
    #     init_pi_high = pi_pulse_high

    # if read_state_value == States.LOW.value:
    #     read_pi_low = pi_pulse_low
    # elif read_state_value == States.HIGH.value:
    #     read_pi_high = pi_pulse_high


    

    # %% Define the sequence

    seq = Sequence()

    # APD gating
    train = [(delay_buffer, LOW),
            (polarization_time, LOW),
            (detune_buffer - polarization_time, LOW),
            (init_pi_dur, LOW),
            (detune_buffer, LOW),
            (uwave_srt_shrt, LOW),
            (detune_buffer, LOW),
            (read_pi_dur, LOW),
            (detune_buffer - polarization_time, LOW),
                
            (gate_time, HIGH),
            (polarization_time - gate_time, LOW),
            (detune_buffer - polarization_time, LOW),
            (init_pi_dur, LOW),
            (detune_buffer, LOW),
            (detune_buffer, LOW),
            (read_pi_dur, LOW),
            (detune_buffer - polarization_time, LOW),
            
            (gate_time, HIGH),
            (polarization_time - gate_time, LOW),
            (detune_buffer - polarization_time, LOW),
            (init_pi_dur, LOW),
            (detune_buffer, LOW),
            (uwave_srt_long, LOW),
            (detune_buffer, LOW),
            (read_pi_dur, LOW),
            (detune_buffer - polarization_time, LOW),
                        
            (gate_time, HIGH),
            (polarization_time - gate_time, LOW),
            (detune_buffer - polarization_time, LOW),
            (init_pi_dur, LOW),
            (detune_buffer, LOW),
            (detune_buffer, LOW),
            (read_pi_dur, LOW),
            (detune_buffer - polarization_time, LOW),
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
            (detune_buffer - polarization_time, LOW),
            (init_pi_dur, LOW),
            (detune_buffer, LOW),
            (uwave_srt_shrt, LOW),
            (detune_buffer, LOW),
            (read_pi_dur, LOW),
            (detune_buffer - polarization_time, LOW),
            
            (polarization_time, HIGH),
            (detune_buffer - polarization_time, LOW),
            (init_pi_dur, LOW),
            (detune_buffer, LOW),
            (detune_buffer, LOW),
            (read_pi_dur, LOW),
            (detune_buffer - polarization_time, LOW),
            
            (polarization_time, HIGH),
            (detune_buffer - polarization_time, LOW),
            (init_pi_dur, LOW),
            (detune_buffer, LOW),
            (uwave_srt_long, LOW),
            (detune_buffer, LOW),
            (read_pi_dur, LOW),
            (detune_buffer - polarization_time, LOW),
            
            (polarization_time, HIGH),
            (detune_buffer - polarization_time, LOW),
            (init_pi_dur, LOW),
            (detune_buffer, LOW),
            (detune_buffer, LOW),
            (read_pi_dur, LOW),
            (detune_buffer - polarization_time, LOW),
            (polarization_time, HIGH) ,
            (back_buffer + aom_delay_time, LOW)
            ]
    tool_belt.process_laser_seq(pulse_streamer, seq, config, 
                                laser_name, laser_power, train)
    period = 0
    for el in train:
        period += el[0]
    print(period)

    # Pulse the microwave for tau omni

    train = [(delay_buffer - rf_omni_delay, LOW),
            (polarization_time, LOW),
            (detune_buffer - polarization_time, LOW),
            (init_pi_dur, HIGH),
            (detune_buffer, LOW),
            (uwave_srt_shrt, HIGH),
            (detune_buffer, LOW),
            (read_pi_dur, HIGH),
            (detune_buffer - polarization_time, LOW),
            
            (polarization_time, LOW),
            (detune_buffer - polarization_time, LOW),
            (init_pi_dur, LOW),
            (detune_buffer, LOW),
            (detune_buffer, LOW),
            (read_pi_dur, LOW),
            (detune_buffer - polarization_time, LOW),
            
            (polarization_time, LOW),
            (detune_buffer - polarization_time, LOW),
            (init_pi_dur, HIGH),
            (detune_buffer, LOW),
            (uwave_srt_long, HIGH),
            (detune_buffer, LOW),
            (read_pi_dur, HIGH),
            (detune_buffer - polarization_time, LOW),
            
            (polarization_time, LOW),
            (detune_buffer - polarization_time, LOW),
            (init_pi_dur, LOW),
            (detune_buffer, LOW),
            (detune_buffer, LOW),
            (read_pi_dur, LOW),
            (detune_buffer - polarization_time, LOW),
            (polarization_time, LOW) ,
            (back_buffer + rf_omni_delay, LOW)
            ]
    seq.setDigital(pulser_do_sig_gen_omni_gate, train)
    period = 0
    for el in train:
        period += el[0]
    print(period)
    
    # MW gate just for the sig gen that puts out a single signal
    train = [(delay_buffer - rf_single_delay, LOW),
            (polarization_time, LOW),
            (detune_buffer - polarization_time, LOW),
            (init_pi_dur, LOW),
            (detune_buffer, LOW),
            (uwave_srt_shrt, HIGH),
            (detune_buffer, LOW),
            (read_pi_dur, LOW),
            (detune_buffer - polarization_time, LOW),
            
            (polarization_time, LOW),
            (detune_buffer - polarization_time, LOW),
            (init_pi_dur, LOW),
            (detune_buffer, LOW),
            (detune_buffer, LOW),
            (read_pi_dur, LOW),
            (detune_buffer - polarization_time, LOW),
            
            (polarization_time, LOW),
            (detune_buffer - polarization_time, LOW),
            (init_pi_dur, LOW),
            (detune_buffer, LOW),
            (uwave_srt_long, HIGH),
            (detune_buffer, LOW),
            (read_pi_dur, LOW),
            (detune_buffer - polarization_time, LOW),
            
            (polarization_time, LOW),
            (detune_buffer - polarization_time, LOW),
            (init_pi_dur, LOW),
            (detune_buffer, LOW),
            (detune_buffer, LOW),
            (read_pi_dur, LOW),
            (detune_buffer - polarization_time, LOW),
            (polarization_time, LOW) ,
            (back_buffer + rf_single_delay, LOW)
            ]
    seq.setDigital(pulser_do_sig_gen_single_gate, train)
    period = 0
    for el in train:
        period += el[0]
    print(period)
    
    
    # FSK list trigger (sig gen omni)
    trigger_buffer = 2e3
    train = [
            (delay_buffer - fsk_trig_delay + trigger_buffer, LOW),
            (trigger_dur, HIGH),
            (polarization_time - trigger_dur, LOW),
            (detune_buffer - polarization_time, LOW),
            (init_pi_dur, LOW),
            (trigger_dur, HIGH),
            (detune_buffer - trigger_dur, LOW),
            (uwave_srt_shrt, LOW),
            (trigger_dur, HIGH),
            (detune_buffer - trigger_dur, LOW),
            (read_pi_dur, LOW),
            (detune_buffer - polarization_time, LOW),
            
            (polarization_time, LOW),
            (detune_buffer - polarization_time, LOW),
            (init_pi_dur, LOW),
            (detune_buffer, LOW),
            (detune_buffer, LOW),
            (read_pi_dur, LOW),
            (detune_buffer - polarization_time, LOW),
            
            (trigger_dur, HIGH),
            (polarization_time - trigger_dur, LOW),
            (detune_buffer - polarization_time, LOW),
            (init_pi_dur, LOW),
            (trigger_dur, HIGH),
            (detune_buffer - trigger_dur, LOW),
            (uwave_srt_long, LOW),
            (trigger_dur, HIGH),
            (detune_buffer - trigger_dur, LOW),
            (read_pi_dur, LOW),
            (detune_buffer - polarization_time, LOW),
            
            (polarization_time, LOW),
            (detune_buffer - polarization_time, LOW),
            (init_pi_dur, LOW),
            (detune_buffer, LOW),
            (detune_buffer, LOW),
            (read_pi_dur, LOW),
            (detune_buffer - polarization_time, LOW),
            (polarization_time, LOW) ,
            (back_buffer + fsk_trig_delay - trigger_buffer, LOW)
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
    
    
    seq_args = [100, 10000.0, 300, 65, 65, 200, 1, 'integrated_520', None, 
                "signal_generator_sg394", "signal_generator_bnc835"]
    seq = get_seq(None, config, seq_args)[0]
    seq.plot()
