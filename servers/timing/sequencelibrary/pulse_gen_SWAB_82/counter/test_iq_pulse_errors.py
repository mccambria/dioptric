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
    durations = args[0:7]
    durations = [numpy.int64(el) for el in durations]
    readout, pi_pulse, uwave_pulse_dur_1, uwave_pulse_dur_2,uwave_pulse_dur_3, \
        polarization, inter_pulse_time = durations
    
    num_uwave_pulses, state, laser_name, laser_power = args[7:11]
    state = States(state)
    sig_gen = config['Servers']['sig_gen_{}'.format(state.name)]
    
    wait_time = config['CommonDurations']['uwave_buffer']
    aom_delay_time = config['Optics'][laser_name]['delay']
    rf_delay_time = config['Microwaves'][sig_gen]['delay']
    iq_delay_time = config['Microwaves']['iq_delay']
    iq_trigger_time = 100# numpy.int64(min(pi_pulse/2, 10))
    
    # make sure uwave_sig_wait_time is a factor of two!
    uwave_sig_wait =inter_pulse_time # int(iq_trigger_time*2 //2)
    half_uwave_sig_wait = int(uwave_sig_wait/2)
    
    pulser_wiring = config['Wiring']['PulseGen']
    pulser_do_apd_gate = pulser_wiring['do_apd_gate']
    pulser_do_sig_gen_gate = pulser_wiring['do_{}_gate'.format(sig_gen)]
    pulser_do_arb_wave_trigger = pulser_wiring['do_arb_wave_trigger']
    
    
    # Include a buffer on the front end so that we can delay channels that
    # should start off the sequence HIGH
    front_buffer = max(aom_delay_time, rf_delay_time, iq_delay_time, 100)
    
    seq = Sequence()
    
    if num_uwave_pulses == 1:
        micowave_signal_train = [(uwave_pulse_dur_1, HIGH)]
        iq_signal_train = [(iq_trigger_time, HIGH),
                           (wait_time + uwave_pulse_dur_1, LOW)]
    elif num_uwave_pulses == 2:
        micowave_signal_train = [(uwave_pulse_dur_1, HIGH), 
                                (uwave_sig_wait, LOW), 
                                (uwave_pulse_dur_2, HIGH)]
        iq_signal_train = [(iq_trigger_time, HIGH),
                                (wait_time - iq_trigger_time + uwave_pulse_dur_1 + half_uwave_sig_wait, LOW),
                                (iq_trigger_time, HIGH),
                                (half_uwave_sig_wait + uwave_pulse_dur_2, LOW),]
    elif num_uwave_pulses == 3:
        micowave_signal_train = [(uwave_pulse_dur_1, HIGH), 
                                 (uwave_sig_wait, LOW), 
                                 (uwave_pulse_dur_2, HIGH),
                                 (uwave_sig_wait, LOW), 
                                 (uwave_pulse_dur_3, HIGH)]
        iq_signal_train = [(iq_trigger_time, HIGH),
                                (wait_time - iq_trigger_time + uwave_pulse_dur_1 + half_uwave_sig_wait, LOW),
                                (iq_trigger_time, HIGH),
                                (half_uwave_sig_wait - iq_trigger_time + uwave_pulse_dur_2 + half_uwave_sig_wait, LOW),
                                (iq_trigger_time, HIGH),
                                (half_uwave_sig_wait + uwave_pulse_dur_3, LOW),]
    
        
    micowave_signal_dur = 0
    for el in micowave_signal_train:
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
             (pi_pulse, LOW), 
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
             (pi_pulse, LOW), 
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
    
    # uwave sequence
    train = [(front_buffer-rf_delay_time, LOW),
             (polarization, LOW),
             (wait_time, LOW),
             (wait_time, LOW),
             (polarization, LOW),
             (wait_time, LOW),
             (pi_pulse, HIGH),
             (wait_time, LOW),
             (polarization, LOW),
             (wait_time, LOW)
             ]
    train.extend(micowave_signal_train)
    train.extend([(wait_time, LOW),
             (polarization, LOW),
             (100 + rf_delay_time, LOW)
             ])
    seq.setDigital(pulser_do_sig_gen_gate, train)
    
    period = 0
    for el in train:
        period += el[0]
    print(period)
        
    # iq pulses
    train = [(front_buffer-iq_delay_time, LOW),
             (iq_trigger_time, HIGH),
             (polarization-iq_trigger_time, LOW),
             (wait_time, LOW),
             (wait_time, LOW),
             (polarization, LOW),
             (iq_trigger_time, HIGH),
             (wait_time- iq_trigger_time, LOW),
             (pi_pulse, LOW), 
             (wait_time, LOW),
             (polarization, LOW)]
    train.extend(iq_signal_train)
    train.extend([
             (wait_time, LOW),
             (polarization - iq_trigger_time, LOW),
             (100 + iq_delay_time, LOW)
             ])
    seq.setDigital(pulser_do_arb_wave_trigger, train)
    
    print(train)
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
    
    # readout, pi_pulse, uwave_pulse_dur_1, uwave_pulse_dur_2,uwave_pulse_dur_3, \
    #     polarization, inter_pulse_time = durations
    # num_uwave_pulses, state, laser_name, laser_power
    
    args = [480, 39.13, 39.13, 22.6, 0, 1000.0, 30, 2, 3, 'integrated_520', None]
    seq = get_seq(None, config, args)[0]

    # Plot the sequence
    seq.plot()
