# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 17:39:27 2019

@author: mccambria
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

    # The first 6 args are ns durations and we need them as int64s
    durations = []
    for ind in range(4):
        durations.append(numpy.int64(args[ind]))

    # Unpack the durations
    polarization_dur, exp_dur, readout_dur, pi_pulse = durations

    # Get the APD index
    apd_index = args[4]
    state_value = args[5]  # Spin state tells us which get sig gen to use
    laser_name = args[6]
    
    # Get what we need out of the wiring dictionary
    pulser_wiring = config['Wiring']['PulseStreamer']
    key = 'do_apd_{}_gate'.format(apd_index)
    apd_index = pulser_wiring[key]
    
    sig_gen_name = config["Microwaves"]["sig_gen_{}".format(States(state_value).name)]
    sig_gen_gate_chan_name = 'do_{}_gate'.format(sig_gen_name)
    pulser_do_sig_gen_gate = pulser_wiring[sig_gen_gate_chan_name]
    
    pulser_do_aom = pulser_wiring['do_{}_dm'.format(laser_name)]
    
    # Get delays
    aom_delay = config["Optics"][laser_name]["delay"]
    rf_delay = config["Microwaves"][sig_gen_name]["delay"]

    # %% Couple calculated values

    half_exp_dur = exp_dur // 2
    exp_dur = half_exp_dur * 2  # This will prevent any rounding errors
    
    half_clock_pulse = numpy.int64(50)
    
    if pi_pulse % 2 == 0:
        half_pi_pulse_short = pi_pulse // 2
        half_pi_pulse_long = half_pi_pulse_short
    else:
        half_pi_pulse_short = pi_pulse // 2
        half_pi_pulse_long = half_pi_pulse_short + 1
    
    period = (2 * polarization_dur) + (2 * exp_dur) + half_clock_pulse

    # %% Define the sequence

    seq = Sequence()

    # APD gating
    train = [(polarization_dur, LOW),
             (exp_dur, LOW),
             (readout_dur, HIGH),
             (polarization_dur - readout_dur, LOW),
             (exp_dur, LOW),
             (readout_dur, HIGH),
             (half_clock_pulse, LOW)]
    seq.setDigital(apd_index, train)

    # Pulse the laser with the AOM for polarization and readout
    train = [(polarization_dur - aom_delay, HIGH),
             (exp_dur, LOW),
             (polarization_dur, HIGH),
             (exp_dur, LOW),
             (half_clock_pulse + aom_delay, HIGH)]
    seq.setDigital(pulser_do_aom, train)

    # Pulse the microwave for tau
    train = [(polarization_dur - rf_delay, LOW),
             (half_exp_dur - half_pi_pulse_short, LOW),
             (pi_pulse, HIGH),
             (half_exp_dur - half_pi_pulse_long, LOW),
             (polarization_dur, LOW),
             (exp_dur, LOW),
             (half_clock_pulse + rf_delay, LOW)]
    seq.setDigital(pulser_do_sig_gen_gate, train)

    final_digital = [pulser_do_aom,
                     pulser_wiring['do_sample_clock']]
    final = OutputState(final_digital, 0.0, 0.0)
    return seq, final, [period]


if __name__ == '__main__':
    wiring = {'do_sample_clock': 0,
              'do_apd_0_gate': 1,
              'do_532_aom': 3,
              'do_uwave_gate_0': 4,
              'do_uwave_gate_1': 5}
    
    # polarization_dur, exp_dur, aom_delay, rf_delay, 
    # readout_dur, pi_pulse, apd_index, uwave_gate_index
    args = [3000, 3000, 0, 0, 
            300, 100, 0, 0]
    seq, final, ret_vals = get_seq(wiring, args)
    seq.plot()   
