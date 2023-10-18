# -*- coding: utf-8 -*-
"""

Sequence for determining the delay beween the fm of the sig gen and the other pulses

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
    durations = args[0:5]
    durations = [numpy.int64(el) for el in durations]
    tau, max_tau, readout, pi_pulse, polarization = durations
    
    state, apd_index, laser_name, laser_power = args[5:9]
    state = States(state)
    sig_gen = config['Servers']['sig_gen_{}'.format(state.name)]
    
    wait_time = config['CommonDurations']['uwave_buffer']
    aom_delay_time = config['Optics'][laser_name]['delay']
    rf_delay_time = config['Microwaves'][sig_gen]['delay']
    fm_mod_bandwidth = 100e3#config['CommonDurations']['fm_mod_bandwidth'] # in Hz
    fm_bandwidth_buffer = 1/fm_mod_bandwidth * 1e9
    
    pulser_wiring = config['Wiring']['PulseStreamer']
    pulser_do_apd_gate = pulser_wiring['do_apd_{}_gate'.format(apd_index)]
    pulser_do_sig_gen_gate = pulser_wiring['do_{}_gate'.format(sig_gen)]
    pulser_ao_fm = pulser_wiring['ao_fm']
    
    
    # Include a buffer on the front end so that we can delay channels that
    # should start off the sequence HIGH
    front_buffer = max(max_tau+pi_pulse, aom_delay_time, rf_delay_time)
    short_buffer = 10
    fm_pol_sub = max(fm_bandwidth_buffer, polarization) + short_buffer
    
    seq = Sequence()
    
    # The readout windows are what everything else is relative to so keep
    # those fixed
    train = [(front_buffer, LOW),
             (polarization, LOW),
             (fm_pol_sub - polarization, LOW),
             (wait_time, LOW),
             (readout, HIGH),
             (polarization - readout, LOW),
             (fm_pol_sub - polarization, LOW),
             (wait_time, LOW),
             (readout, HIGH),
             (polarization - readout, LOW),
             ]
    seq.setDigital(pulser_do_apd_gate, train)
    period = 0
    for el in train:
        period += el[0]
    print(period)

    # Laser sequence
    train = [(front_buffer - aom_delay_time, LOW),
             (polarization, HIGH),
             (fm_pol_sub - polarization, LOW),
             (wait_time, LOW),
             (polarization, HIGH),
             (fm_pol_sub - polarization, LOW),
             (wait_time, LOW),
             (polarization, HIGH),
             (aom_delay_time, LOW), 
             ]
    tool_belt.process_laser_seq(pulse_streamer, seq, config, 
                                laser_name, laser_power, train)
    
    period = 0
    for el in train:
        period += el[0]
    print(period)
    
    # uwave sequence
    train = [(front_buffer - rf_delay_time, LOW),
             (fm_pol_sub - pi_pulse, LOW),
             (pi_pulse, HIGH), 
             (wait_time, LOW),
             (fm_pol_sub, LOW),
             (wait_time, LOW),
             (polarization, LOW),
             (rf_delay_time, LOW), 
             ]
    seq.setDigital(pulser_do_sig_gen_gate, train)
    
    period = 0
    for el in train:
        period += el[0]
    print(period)
    
    # Vary the position of the fm
    train = [(front_buffer - tau, LOW),
             (fm_pol_sub, HIGH),
             (wait_time, LOW),
             (fm_pol_sub, HIGH),
             (wait_time, LOW),
             (polarization + tau, LOW)
             ]
    seq.setAnalog(pulser_ao_fm, train)
    
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
    
    #tau, max_tau, readout, pi_pulse, polarization = durations
    args = [200, 500, 350, 89, 1000.0, 3, 1, 'integrated_520', None]
    seq = get_seq(None, config, args)[0]

    # Plot the sequence
    seq.plot()
