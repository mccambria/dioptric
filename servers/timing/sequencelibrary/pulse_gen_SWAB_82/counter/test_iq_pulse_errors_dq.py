# -*- coding: utf-8 -*-
"""

based off this paper: https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.105.077601


Created on Sun Aug 6 11:22:40 2019

@author: agardill
"""


# %% Imports


from pulsestreamer import Sequence
from pulsestreamer import OutputState
import numpy
import utils.tool_belt as tool_belt
from utils.tool_belt import States


# %% Constants


LOW = 0
HIGH = 1


# %% Functions


# %% Sequence definition


def get_seq(pulse_streamer, config, args):
    """This is called by the pulse_streamer server to get the sequence object
    based on the wiring (from the registry) and the args passed by the client.
    """
    
    # Extract the delay (that the rf is applied from the start of the seq),
    # the readout time, and the pi pulse duration
    durations = args[0:9]
    durations = [numpy.int64(el) for el in durations]
    readout, pi_pulse_low,pi_pulse_high, \
        uwave_pulse_dur_1, uwave_pulse_dur_2, uwave_pulse_dur_3, \
        polarization, inter_pulse_time ,inter_uwave_buffer = durations
    
    num_uwave_pulses, state_activ,state_proxy, laser_name, laser_power = args[9:14]
    state_activ = States(state_activ)
    state_proxy = States(state_proxy)
    
    wait_time = config['CommonDurations']['uwave_buffer']
    aom_delay_time = config['Optics'][laser_name]['delay']
    sig_gen_low_name = config['Servers']['sig_gen_LOW']
    uwave_delay_low = config['Microwaves'][sig_gen_low_name]['delay']
    sig_gen_high_name = config['Servers']['sig_gen_HIGH']
    uwave_delay_high = config['Microwaves'][sig_gen_high_name]['delay']
    # inter_uwave_buffer = 100
    iq_delay_time = config['Microwaves']['iq_delay']
    iq_trigger_time = 100#numpy.int64(min(pi_pulse_low/2, pi_pulse_high/2, 10))
    
    # make sure uwave_sig_wait_time is a factor of two!
    uwave_sig_wait =inter_pulse_time # int(iq_trigger_time*2 //2)
    half_uwave_sig_wait = int(uwave_sig_wait/2)
    half_inter_uwave_buffer = int(inter_uwave_buffer/2)
    
    pulser_wiring = config['Wiring']['PulseGen']
    pulser_do_apd_gate = pulser_wiring['do_apd_gate']
    sig_gen_gate_chan_name_low = 'do_{}_gate'.format(sig_gen_low_name)
    pulser_do_sig_gen_gate_low = pulser_wiring[sig_gen_gate_chan_name_low]
    sig_gen_gate_chan_name_high = 'do_{}_gate'.format(sig_gen_high_name)
    pulser_do_sig_gen_gate_high = pulser_wiring[sig_gen_gate_chan_name_high]
    pulser_do_arb_wave_trigger = pulser_wiring['do_arb_wave_trigger']
    
    
    
    prep_pulse_proxy_low = 0
    prep_pulse_proxy_high = 0
    
    norm_pulse_activ_low = 0
    norm_pulse_activ_high = 0
    
    uwave_pulse_dur_1_low = 0
    uwave_pulse_dur_2_low = 0
    uwave_pulse_dur_3_low = 0
    uwave_pulse_dur_1_high = 0
    uwave_pulse_dur_2_high = 0
    uwave_pulse_dur_3_high = 0
    
    if state_activ.value == States.LOW.value:
        norm_pulse_activ_low = pi_pulse_low
        prep_pulse_proxy_high = pi_pulse_high
        uwave_pulse_dur_1_low = uwave_pulse_dur_1
        uwave_pulse_dur_2_low = uwave_pulse_dur_2
        uwave_pulse_dur_3_low = uwave_pulse_dur_3
        
    elif state_activ.value == States.HIGH.value:
        norm_pulse_activ_high = pi_pulse_high
        prep_pulse_proxy_low = pi_pulse_low
        uwave_pulse_dur_1_high = uwave_pulse_dur_1
        uwave_pulse_dur_2_high = uwave_pulse_dur_2
        uwave_pulse_dur_3_high = uwave_pulse_dur_3
    
    norm_pi_pulse_dur = norm_pulse_activ_low + norm_pulse_activ_high
    prep_pulse_dur = prep_pulse_proxy_low + prep_pulse_proxy_high
    total_uwave_pulse_dur_1 = prep_pulse_dur + \
            inter_uwave_buffer + uwave_pulse_dur_1 + inter_uwave_buffer + \
            prep_pulse_dur
            
    total_uwave_pulse_dur_2 = prep_pulse_dur + \
            inter_uwave_buffer + uwave_pulse_dur_2 + inter_uwave_buffer + \
            prep_pulse_dur
            
    total_uwave_pulse_dur_3 = prep_pulse_dur + \
            inter_uwave_buffer + uwave_pulse_dur_3 + inter_uwave_buffer + \
            prep_pulse_dur
    
    # Include a buffer on the front end so that we can delay channels that
    # should start off the sequence HIGH
    front_buffer = max(aom_delay_time, uwave_delay_low, uwave_delay_high, iq_delay_time, 100)
    
    seq = Sequence() 
    
    if num_uwave_pulses == 1:
        micowave_signal_train_low = [(norm_pulse_activ_low, HIGH), (norm_pulse_activ_high, LOW),
                                     (uwave_sig_wait, LOW), #uwave_sig_wait
                                     (prep_pulse_proxy_low, HIGH), (prep_pulse_proxy_high, LOW),
                                     (inter_uwave_buffer, LOW),
                                     (uwave_pulse_dur_1_low, HIGH), (uwave_pulse_dur_1_high, LOW),
                                     (inter_uwave_buffer, LOW),
                                     (prep_pulse_proxy_low, HIGH), (prep_pulse_proxy_high, LOW),
                                     (uwave_sig_wait, LOW),
                                     (norm_pulse_activ_low, HIGH), (norm_pulse_activ_high, LOW)]
        
        micowave_signal_train_high = [(norm_pulse_activ_low, LOW), (norm_pulse_activ_high, HIGH),
                                     (uwave_sig_wait, LOW),
                                     (prep_pulse_proxy_low, LOW), (prep_pulse_proxy_high, HIGH),
                                     (inter_uwave_buffer, LOW),
                                     (uwave_pulse_dur_1_low, LOW), (uwave_pulse_dur_1_high, HIGH),
                                     (inter_uwave_buffer, LOW),
                                     (prep_pulse_proxy_low, LOW), (prep_pulse_proxy_high, HIGH),
                                     (uwave_sig_wait, LOW),
                                     (norm_pulse_activ_low, LOW), (norm_pulse_activ_high, HIGH)]
        iq_signal_train = [(wait_time ,LOW),
                           (norm_pi_pulse_dur, LOW),
                           (half_uwave_sig_wait, LOW),
                           (iq_trigger_time, HIGH),
                           (half_uwave_sig_wait - iq_trigger_time +
                            total_uwave_pulse_dur_1 + half_uwave_sig_wait, LOW),
                           (iq_trigger_time, HIGH),
                          ( half_uwave_sig_wait + norm_pi_pulse_dur, LOW)]
        
        
    elif num_uwave_pulses == 2:
        micowave_signal_train_low = [(norm_pulse_activ_low, HIGH), (norm_pulse_activ_high, LOW),
                                     (uwave_sig_wait, LOW),
                                     (prep_pulse_proxy_low, HIGH), (prep_pulse_proxy_high, LOW),
                                     (inter_uwave_buffer, LOW),
                                     (uwave_pulse_dur_1_low, HIGH), (uwave_pulse_dur_1_high, LOW),
                                     (inter_uwave_buffer, LOW),
                                     (prep_pulse_proxy_low, HIGH), (prep_pulse_proxy_high, LOW),
                                     (uwave_sig_wait, LOW),
                                     (prep_pulse_proxy_low, HIGH), (prep_pulse_proxy_high, LOW),
                                     (inter_uwave_buffer, LOW),
                                     (uwave_pulse_dur_2_low, HIGH), (uwave_pulse_dur_2_high, LOW),
                                     (inter_uwave_buffer, LOW),
                                     (prep_pulse_proxy_low, HIGH), (prep_pulse_proxy_high, LOW),
                                     (uwave_sig_wait, LOW),
                                     (norm_pulse_activ_low, HIGH), (norm_pulse_activ_high, LOW)]
        
        micowave_signal_train_high = [(norm_pulse_activ_low, LOW), (norm_pulse_activ_high, HIGH),
                                     (uwave_sig_wait, LOW),
                                     (prep_pulse_proxy_low, LOW), (prep_pulse_proxy_high, HIGH),
                                     (inter_uwave_buffer, LOW),
                                     (uwave_pulse_dur_1_low, LOW), (uwave_pulse_dur_1_high, HIGH),
                                     (inter_uwave_buffer, LOW),
                                     (prep_pulse_proxy_low, LOW), (prep_pulse_proxy_high, HIGH),
                                     (uwave_sig_wait, LOW),
                                     (prep_pulse_proxy_low, LOW), (prep_pulse_proxy_high, HIGH),
                                     (inter_uwave_buffer, LOW),
                                     (uwave_pulse_dur_2_low, LOW), (uwave_pulse_dur_2_high, HIGH),
                                     (inter_uwave_buffer, LOW),
                                     (prep_pulse_proxy_low, LOW), (prep_pulse_proxy_high, HIGH),
                                     (uwave_sig_wait, LOW),
                                     (norm_pulse_activ_low, LOW), (norm_pulse_activ_high, HIGH)]
        
        
        iq_signal_train = [(wait_time + norm_pi_pulse_dur + half_uwave_sig_wait, LOW),
                           (iq_trigger_time, HIGH),
                           (half_uwave_sig_wait - iq_trigger_time +
                               total_uwave_pulse_dur_1  + half_uwave_sig_wait, LOW),
                           (iq_trigger_time, HIGH),
                           (half_uwave_sig_wait - iq_trigger_time + total_uwave_pulse_dur_2+
                               half_uwave_sig_wait, LOW),
                           (iq_trigger_time, HIGH),
                           (half_uwave_sig_wait + norm_pi_pulse_dur, LOW)]
                                
    elif num_uwave_pulses == 3:
        
        micowave_signal_train_low = [(norm_pulse_activ_low, HIGH), (norm_pulse_activ_high, LOW),
                                     (uwave_sig_wait, LOW),
                                     (prep_pulse_proxy_low, HIGH), (prep_pulse_proxy_high, LOW),
                                     (inter_uwave_buffer, LOW),
                                     (uwave_pulse_dur_1_low, HIGH), (uwave_pulse_dur_1_high, LOW),
                                     (inter_uwave_buffer, LOW),
                                     (prep_pulse_proxy_low, HIGH), (prep_pulse_proxy_high, LOW),
                                     (uwave_sig_wait, LOW),
                                     (prep_pulse_proxy_low, HIGH), (prep_pulse_proxy_high, LOW),
                                     (inter_uwave_buffer, LOW),
                                     (uwave_pulse_dur_2_low, HIGH), (uwave_pulse_dur_2_high, LOW),
                                     (inter_uwave_buffer, LOW),
                                     (prep_pulse_proxy_low, HIGH), (prep_pulse_proxy_high, LOW),
                                     (uwave_sig_wait, LOW),
                                     (prep_pulse_proxy_low, HIGH), (prep_pulse_proxy_high, LOW),
                                     (inter_uwave_buffer, LOW),
                                     (uwave_pulse_dur_3_low, HIGH), (uwave_pulse_dur_3_high, LOW),
                                     (inter_uwave_buffer, LOW),
                                     (prep_pulse_proxy_low, HIGH), (prep_pulse_proxy_high, LOW),
                                     (uwave_sig_wait, LOW),
                                     (norm_pulse_activ_low, HIGH), (norm_pulse_activ_high, LOW)]
                                         
        micowave_signal_train_high = [(norm_pulse_activ_low, LOW), (norm_pulse_activ_high, HIGH),
                                     (uwave_sig_wait, LOW),
                                     (prep_pulse_proxy_low, LOW), (prep_pulse_proxy_high, HIGH),
                                     (inter_uwave_buffer, LOW),
                                     (uwave_pulse_dur_1_low, LOW), (uwave_pulse_dur_1_high, HIGH),
                                     (inter_uwave_buffer, LOW),
                                     (prep_pulse_proxy_low, LOW), (prep_pulse_proxy_high, HIGH),
                                     (uwave_sig_wait, LOW),
                                     (prep_pulse_proxy_low, LOW), (prep_pulse_proxy_high, HIGH),
                                     (inter_uwave_buffer, LOW),
                                     (uwave_pulse_dur_2_low, LOW), (uwave_pulse_dur_2_high, HIGH),
                                     (inter_uwave_buffer, LOW),
                                     (prep_pulse_proxy_low, LOW), (prep_pulse_proxy_high, HIGH),
                                     (uwave_sig_wait, LOW),
                                     (prep_pulse_proxy_low, LOW), (prep_pulse_proxy_high, HIGH),
                                     (inter_uwave_buffer, LOW),
                                     (uwave_pulse_dur_3_low, LOW), (uwave_pulse_dur_3_high, HIGH),
                                     (inter_uwave_buffer, LOW),
                                     (prep_pulse_proxy_low, LOW), (prep_pulse_proxy_high, HIGH),
                                     (uwave_sig_wait, LOW),
                                     (norm_pulse_activ_low, LOW), (norm_pulse_activ_high, HIGH)]
        
        iq_signal_train = [(wait_time + norm_pi_pulse_dur + half_uwave_sig_wait, LOW),
                           (iq_trigger_time, HIGH),
                           (half_uwave_sig_wait - iq_trigger_time +
                               total_uwave_pulse_dur_1  + half_uwave_sig_wait, LOW),
                           (iq_trigger_time, HIGH),
                           (half_uwave_sig_wait - iq_trigger_time + total_uwave_pulse_dur_2+
                               half_uwave_sig_wait, LOW),
                           (iq_trigger_time, HIGH),
                           (half_uwave_sig_wait - iq_trigger_time + total_uwave_pulse_dur_3+
                               half_uwave_sig_wait, LOW),
                           (iq_trigger_time, HIGH),
                           (half_uwave_sig_wait + norm_pi_pulse_dur, LOW)]    
    micowave_signal_dur = 0
    for el in micowave_signal_train_low:
        micowave_signal_dur += el[0]

    
    
    # The readout windows are what everything else is relative to so keep
    # those fixed
    train = [(front_buffer, LOW),
             (polarization, LOW),
             (wait_time, LOW),
             (wait_time, LOW),
             (readout, HIGH),
             (polarization - readout, LOW),
             (wait_time, LOW),
             (norm_pi_pulse_dur, LOW), 
             (wait_time, LOW),
             (readout, HIGH),
             (polarization - readout, LOW),
             (wait_time, LOW),
             (micowave_signal_dur, LOW), 
             (wait_time, LOW),
             (readout, HIGH),
             (polarization - readout, LOW),
             (100, LOW)
             ]
    seq.setDigital(pulser_do_apd_gate, train)
    period = 0
    for el in train:
        period += el[0]
    print(period)

    # Laser sequence
    train = [(front_buffer-aom_delay_time, LOW),
             (polarization, HIGH),
             (wait_time, LOW),
             (wait_time, LOW),
             (polarization, HIGH),
             (wait_time, LOW),
             (norm_pi_pulse_dur, LOW),
             (wait_time, LOW),
             (polarization, HIGH),
             (wait_time, LOW),
             (micowave_signal_dur, LOW), 
             (wait_time, LOW),
             (polarization, HIGH),
             (100 + aom_delay_time, LOW)
             ]
    tool_belt.process_laser_seq(pulse_streamer, seq, config, 
                                laser_name, laser_power, train)
    
    period = 0
    for el in train:
        period += el[0]
    print(period)
    
    # uwave LOW sequence
    train = [(front_buffer-uwave_delay_low, LOW),
             (polarization, LOW),
             (wait_time, LOW),
             (wait_time, LOW),
             (polarization, LOW),
             (wait_time, LOW),
             (norm_pulse_activ_low, HIGH),
             (norm_pulse_activ_high, LOW),
             (wait_time, LOW),
             (polarization, LOW),
             (wait_time, LOW)
             ]
    train.extend(micowave_signal_train_low)
    train.extend([(wait_time, LOW),
             (polarization, LOW),
             (100 + uwave_delay_low, LOW)
             ])
    seq.setDigital(pulser_do_sig_gen_gate_low, train)
    
    period = 0
    for el in train:
        period += el[0]
    print(period)
        
    
    # uwave HIGH sequence
    train = [(front_buffer-uwave_delay_high, LOW),
             (polarization, LOW),
             (wait_time, LOW),
             (wait_time, LOW),
             (polarization, LOW),
             (wait_time, LOW),
             (norm_pulse_activ_low, LOW),
             (norm_pulse_activ_high, HIGH),
             (wait_time, LOW),
             (polarization, LOW),
             (wait_time, LOW)
             ]
    train.extend(micowave_signal_train_high)
    train.extend([(wait_time, LOW),
             (polarization, LOW),
             (100 + uwave_delay_high, LOW)
             ])
    seq.setDigital(pulser_do_sig_gen_gate_high, train)
    
    period = 0
    for el in train:
        period += el[0]
    print(period)
    
    # IQ trigger
    train = [(front_buffer-iq_delay_time, LOW),
             (iq_trigger_time, HIGH),
             (polarization-iq_trigger_time, LOW),
             (wait_time, LOW),
             (wait_time, LOW),
             (polarization, LOW),
             (iq_trigger_time, HIGH),
             (wait_time- iq_trigger_time, LOW),
             (norm_pi_pulse_dur, LOW), 
             (wait_time, LOW),
             (polarization, LOW)]
    train.extend(iq_signal_train)
    train.extend([
             (wait_time, LOW),
             (polarization - iq_trigger_time, LOW),
             (100 + iq_delay_time, LOW)
             ])
    seq.setDigital(pulser_do_arb_wave_trigger, train)
    
    period = 0
    for el in train:
        period += el[0]
    print(period)

    final_digital = [pulser_wiring['do_sample_clock']]
    final = OutputState(final_digital, 0.0, 0.0)
    return seq, final, [period]


# %% Run the file


# The __name__ variable will only be '__main__' if you run this file directly.
# This allows a file's functions, classes, etc to be imported without running
# the script that you set up here.
if __name__ == '__main__':
    config = tool_belt.get_config_dict()
    tool_belt.set_delays_to_zero(config) 
    
    # readout, pi_pulse_low,pi_pulse_high, \
    #     uwave_pulse_dur_1, uwave_pulse_dur_2, uwave_pulse_dur_3, \
    #     polarization, inter_pulse_time = durations
    # num_uwave_pulses, state_activ,state_proxy, laser_name, laser_power = args[8:13]
    args = [340, 69.71, 56.38, 28.75, 28.75, 0, 1000.0, 0, 114, 2, 3, 1, 'integrated_520', None]
    seq = get_seq(None, config, args)[0]

    # Plot the sequence
    seq.plot()
