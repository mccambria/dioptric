# -*- coding: utf-8 -*-
"""

Sequence for determining the delay beween the rf and the AOM

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
    durations = args[0:4]
    durations = [numpy.int64(el) for el in durations]
    readout, pi_pulse, pi_on_2_pulse, polarization = durations
    
    num_pi_pulse, state, apd_index, laser_name, laser_power = args[4:9]
    state = States(state)
    sig_gen = config['Microwaves']['sig_gen_{}'.format(state.name)]
    
    wait_time = config['CommonDurations']['uwave_buffer']
    aom_delay_time = config['Optics'][laser_name]['delay']
    rf_delay_time = config['Microwaves'][sig_gen]['delay']
    iq_delay_time = config['Microwaves']['iq_delay']
    iq_trigger_time = numpy.int64(min(pi_on_2_pulse, 10))
    
    pulser_wiring = config['Wiring']['PulseStreamer']
    pulser_do_apd_gate = pulser_wiring['do_apd_{}_gate'.format(apd_index)]
    pulser_do_sig_gen_gate = pulser_wiring['do_{}_gate'.format(sig_gen)]
    pulser_do_arb_wave_trigger = pulser_wiring['do_arb_wave_trigger']
    
    
    # Include a buffer on the front end so that we can delay channels that
    # should start off the sequence HIGH
    front_buffer = max(aom_delay_time, rf_delay_time, iq_delay_time, 100)
    
    seq = Sequence()
    
    reference_microwave_train = [(pi_pulse, HIGH)]
    if num_pi_pulse == 0: 
        micorwave_duration = pi_pulse
        signal_iq_train = [(iq_trigger_time, HIGH), (pi_pulse - iq_trigger_time, LOW)]
    else:
        micorwave_duration = pi_on_2_pulse + (pi_pulse)*num_pi_pulse + pi_on_2_pulse
        signal_iq_train = [(iq_trigger_time, HIGH), (pi_on_2_pulse- iq_trigger_time, LOW)]
        rep_pi_pulse_train = [(iq_trigger_time, HIGH), (pi_pulse - iq_trigger_time, LOW)]*num_pi_pulse
        signal_iq_train = signal_iq_train + rep_pi_pulse_train + signal_iq_train
        
    signal_microwave_train = [(micorwave_duration , HIGH) ]
    
    
    # The readout windows are what everything else is relative to so keep
    # those fixed
    train = [(front_buffer, LOW),
             (polarization, LOW),
             (wait_time, LOW),
             (pi_pulse, LOW), 
             (wait_time, LOW),
             (readout, HIGH),
             (polarization - readout, LOW),
             (wait_time, LOW),
             (micorwave_duration, LOW), 
             (wait_time, LOW),
             (readout, HIGH),
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
             (pi_pulse, LOW), 
             (wait_time, LOW),
             (polarization, HIGH),
             (wait_time, LOW),
             (micorwave_duration, LOW), 
             (wait_time, LOW),
             (readout, HIGH),
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
             (wait_time, LOW)]
    train.extend(reference_microwave_train)
    train.extend([(wait_time, LOW),
             (polarization, LOW),
             (wait_time, LOW)])
    train.extend(signal_microwave_train)
    train.extend([(wait_time, LOW),
             (readout, LOW),
             (100 + rf_delay_time, LOW)
             ])
    seq.setDigital(pulser_do_sig_gen_gate, train)
    
    period = 0
    for el in train:
        period += el[0]
    print(period)
    
    # Vary the position of the iq pulses
    train = [(front_buffer-iq_delay_time, LOW),
             (polarization, LOW),
             (wait_time, LOW),
             (iq_trigger_time, HIGH),
             (pi_pulse - iq_trigger_time, LOW), 
             (wait_time, LOW),
             (polarization, LOW),
             (wait_time, LOW)]
    train.extend(signal_iq_train)
    train.extend([
             (wait_time, LOW),
             (readout, LOW),
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
    
    # readout, pi_pulse, pi_on_2_pulse, polarization = durations
    # num_pi_pulse, state, apd_index, laser_name, laser_power = args[5:9]
    args = [350, 100, 50, 1000, 0, 3, 1, 'integrated_520', None]
    seq = get_seq(None, config, args)[0]

    # Plot the sequence
    seq.plot()
