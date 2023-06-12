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
    for ind in range(7):
        durations.append(numpy.int64(args[ind]))

    # Unpack the durations
    tau,   readout_time, reion_time, ion_time,  \
            pi_pulse_low, pi_pulse_high, max_tau = durations
 
    
    # Specify the initial and readout states
    activ_state_value = args[7]
    # proxy_state_value = args[9]
    
    green_laser_name = args[8]
    yellow_laser_name = args[9]
    red_laser_name = args[10]
    reion_power = args[11]
    ion_power = args[12]
    readout_power = args[13]
    
    low_sig_gen_name = config['Servers']['sig_gen_LOW']
    high_sig_gen_name = config['Servers']['sig_gen_HIGH']
    
    uwave_buffer= config['CommonDurations']['uwave_buffer']
    scc_ion_readout_buffer = config['CommonDurations']['scc_ion_readout_buffer']
    sig_ref_buffer = uwave_buffer
    coh_buffer = 134
    iq_trigger_time = 100
    
    green_delay_time = config['Optics'][green_laser_name]['delay']
    yellow_delay_time = config['Optics'][yellow_laser_name]['delay']
    red_delay_time = config['Optics'][red_laser_name]['delay']
    rf_delay_low = config['Microwaves'][low_sig_gen_name]['delay']
    rf_delay_high = config['Microwaves'][high_sig_gen_name]['delay']
    iq_delay_time = config['Microwaves']['iq_delay']
    
    
    pulser_wiring = config['Wiring']['PulseGen']
    pulser_do_apd_gate = pulser_wiring['do_apd_gate']
    sig_gen_gate_chan_name_low = 'do_{}_gate'.format(low_sig_gen_name)
    pulser_do_sig_gen_gate_low = pulser_wiring[sig_gen_gate_chan_name_low]
    sig_gen_gate_chan_name_high = 'do_{}_gate'.format(high_sig_gen_name)
    pulser_do_sig_gen_gate_high = pulser_wiring[sig_gen_gate_chan_name_high]
    pulser_do_arb_wave_trigger = pulser_wiring['do_arb_wave_trigger']
    
    
    delay_buffer = max(green_delay_time, yellow_delay_time, red_delay_time,rf_delay_low,rf_delay_high,iq_delay_time, 100)
    back_buffer = 200

    
    init_pi_pulse_low = 0
    init_pi_pulse_high = 0
    read_pi_pulse_low = 0
    read_pi_pulse_high = 0
    
    uwave_pulse_activ_low = 0
    uwave_pulse_activ_high = 0
    
    uwave_pulse_proxy_low = 0
    uwave_pulse_proxy_high = 0
    
    if activ_state_value == States.LOW.value:
        uwave_pulse_activ_low = tau
        uwave_pulse_proxy_high = pi_pulse_high
        init_pi_pulse_high = pi_pulse_high
        read_pi_pulse_low = pi_pulse_low
    elif activ_state_value == States.HIGH.value:
        uwave_pulse_activ_high = tau
        uwave_pulse_proxy_low = pi_pulse_low
        init_pi_pulse_low = pi_pulse_low
        read_pi_pulse_high = pi_pulse_high



    # %% Write the microwave sequence to be used.
    
    uwave_experiment_train_low = [
                                    # (init_pi_pulse_low, HIGH), (init_pi_pulse_high, LOW),
                                    (read_pi_pulse_low, HIGH), (read_pi_pulse_high, LOW),
                                  (coh_buffer, LOW),
                                  (uwave_pulse_proxy_low, HIGH), (uwave_pulse_proxy_high, LOW),
                                    (coh_buffer, LOW),
                                    (uwave_pulse_activ_low, HIGH), (uwave_pulse_activ_high, LOW),
                                    (coh_buffer, LOW),
                                    (uwave_pulse_proxy_low, HIGH), (uwave_pulse_proxy_high, LOW),
                                    (coh_buffer, LOW),
                                    (read_pi_pulse_low, HIGH), (read_pi_pulse_high, LOW)
                                    ]
    uwave_experiment_dur = 0
    for el in uwave_experiment_train_low:
        uwave_experiment_dur += el[0]
        
        
    uwave_experiment_train_high = [
                                    # (init_pi_pulse_low, LOW), (init_pi_pulse_high, HIGH),
                                    (read_pi_pulse_low, LOW), (read_pi_pulse_high, HIGH),
                                    (coh_buffer, LOW),
                                   (uwave_pulse_proxy_low, LOW), (uwave_pulse_proxy_high, HIGH),
                                       (coh_buffer, LOW),
                                       (uwave_pulse_activ_low, LOW), (uwave_pulse_activ_high, HIGH),
                                       (coh_buffer, LOW),
                                       (uwave_pulse_proxy_low, LOW), (uwave_pulse_proxy_high, HIGH),
                                       (coh_buffer, LOW),
                                       (read_pi_pulse_low, LOW), (read_pi_pulse_high, HIGH)
                                       ]
    uwave_experiment_dur = 0
    for el in uwave_experiment_train_high:
        uwave_experiment_dur += el[0]
    

    # %% Define the sequence

    seq = Sequence()

    # APD gating
    train = [(delay_buffer, LOW),
            (reion_time, LOW),
            (uwave_buffer, LOW),
            (max_tau, LOW),
            (uwave_buffer, LOW),
            (ion_time, LOW),
            (scc_ion_readout_buffer, LOW),
            (readout_time, HIGH),
            (sig_ref_buffer, LOW),
            (reion_time, LOW),
            (uwave_buffer, LOW),
            (max_tau-tau, LOW),
            (uwave_experiment_dur, LOW),
            (uwave_buffer, LOW),
            (ion_time, LOW),
            (scc_ion_readout_buffer, LOW),
            (readout_time, HIGH),
            (back_buffer, LOW)
            ]
    seq.setDigital(pulser_do_apd_gate, train)
    period = 0
    for el in train:
        period += el[0]
    print(period)

    # Green laser
    train = [(delay_buffer - green_delay_time, LOW),
            (reion_time, HIGH),
            (uwave_buffer, LOW),
            (max_tau, LOW),
            (uwave_buffer, LOW),
            (ion_time, LOW),
            (scc_ion_readout_buffer, LOW),
            (readout_time, LOW),
            (sig_ref_buffer, LOW),
            (reion_time, HIGH),
            (uwave_buffer, LOW),
            (max_tau-tau, LOW),
            (uwave_experiment_dur, LOW),
            (uwave_buffer, LOW),
            (ion_time, LOW),
            (scc_ion_readout_buffer, LOW),
            (readout_time, LOW),
            (back_buffer + green_delay_time, LOW)
            ]
    power_list = reion_power
    tool_belt.process_laser_seq(
        pulse_streamer, seq, config, green_laser_name,power_list , train
            )
    period = 0
    for el in train:
        period += el[0]
    print(period)

    # Red
    train = [(delay_buffer - red_delay_time, LOW),
            (reion_time, LOW),
            (uwave_buffer, LOW),
            (max_tau, LOW),
            (uwave_buffer, LOW),
            (ion_time, HIGH),
            (scc_ion_readout_buffer, LOW),
            (readout_time, LOW),
            (sig_ref_buffer, LOW),
            (reion_time, LOW),
            (uwave_buffer, LOW),
            (max_tau-tau, LOW),
            (uwave_experiment_dur, LOW),
            (uwave_buffer, LOW),
            (ion_time, HIGH),
            (scc_ion_readout_buffer, LOW),
            (readout_time, LOW),
            (back_buffer + red_delay_time, LOW)
            ]
    power_list = [ion_power, ion_power]
    tool_belt.process_laser_seq(
        pulse_streamer, seq, config, red_laser_name, power_list, train
    )
    period = 0
    for el in train:
        period += el[0]
    print(period)
    
    
    # Yellow
    train = [(delay_buffer - yellow_delay_time, LOW),
            (reion_time, LOW),
            (uwave_buffer, LOW),
            (max_tau, LOW),
            (uwave_buffer, LOW),
            (ion_time, LOW),
            (scc_ion_readout_buffer, LOW),
            (readout_time, HIGH),
            (sig_ref_buffer, LOW),
            (reion_time, LOW),
            (uwave_buffer, LOW),
            (max_tau-tau, LOW),
            (uwave_experiment_dur, LOW),
            (uwave_buffer, LOW),
            (ion_time, LOW),
            (scc_ion_readout_buffer, LOW),
            (readout_time, HIGH),
            (back_buffer + yellow_delay_time, LOW)
            ]
    tool_belt.process_laser_seq(pulse_streamer, seq, config,
                                yellow_laser_name, 
                                [readout_power], train)
    period = 0
    for el in train:
        period += el[0]
    print(period)
    
    # MW gate HIGH
    train = [(delay_buffer - rf_delay_high, LOW),
             (reion_time, LOW),
             (uwave_buffer, LOW),
             (max_tau, LOW),
             (uwave_buffer, LOW),
             (ion_time, LOW),
             (scc_ion_readout_buffer, LOW),
             (readout_time, LOW),
             (sig_ref_buffer, LOW),
             (reion_time, LOW),
            (uwave_buffer, LOW),
            (max_tau-tau, LOW)]
    train.extend(uwave_experiment_train_high)
    train.extend([
             (uwave_buffer, LOW),
             (ion_time, LOW),
             (scc_ion_readout_buffer, LOW),
             (readout_time, LOW),
             (back_buffer + rf_delay_high, LOW)])
    seq.setDigital(pulser_do_sig_gen_gate_high, train)
    period = 0
    # print(train)
    for el in train:
        period += el[0]
    print(period)
    
    # MW gate LOW
    train = [(delay_buffer - rf_delay_low, LOW),
             (reion_time, LOW),
             (uwave_buffer, LOW),
             (max_tau, LOW),
             (uwave_buffer, LOW),
             (ion_time, LOW),
             (scc_ion_readout_buffer, LOW),
             (readout_time, LOW),
             (sig_ref_buffer, LOW),
             (reion_time, LOW),
            (uwave_buffer, LOW),
            (max_tau-tau, LOW)]
    train.extend(uwave_experiment_train_low)
    train.extend([
             (uwave_buffer, LOW),
             (ion_time, LOW),
             (scc_ion_readout_buffer, LOW),
             (readout_time, LOW),
             (back_buffer + rf_delay_low, LOW)])
    seq.setDigital(pulser_do_sig_gen_gate_low, train)
    period = 0
    for el in train:
        period += el[0]
    print(period)
    
    # Set a trigger to the IQ modulation at the beginning to set the phase
    train = [(delay_buffer - iq_delay_time, LOW),
            (iq_trigger_time,HIGH),
            (period - iq_trigger_time -delay_buffer+ + iq_delay_time, LOW)]
    seq.setDigital(pulser_do_arb_wave_trigger, train)
    print(train)
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
    
    
    # # Unpack the durations
    # tau,   readout_time, reion_time, ion_time,  \
    #         pi_pulse_low, pi_pulse_high, max_tau = durations
 
    
    # # Specify the initial and readout states
    # activ_state_value = args[8]
    # # proxy_state_value = args[9]
    
    # green_laser_name = args[9]
    # yellow_laser_name = args[10]
    # red_laser_name = args[11]
    # reion_power = args[12]
    # ion_power = args[13]
    # readout_power = args[14]
    
    
    # 
    # tau,   readout_time, reion_time, ion_time,  \
    #         pi_pulse_low, pi_pulse_high, max_tau = durations
    seq_args =[396, 1000.0, 6000.0, 400, 39, 42, 600, 3, 
               'integrated_520', 'laser_LGLO_589', 'cobolt_638', None, None, None]
    seq = get_seq(None, config, seq_args)[0]
    seq.plot()
