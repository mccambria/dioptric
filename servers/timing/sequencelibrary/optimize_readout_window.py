# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 17:39:27 2019

@author: mccambria
"""

from pulsestreamer import Sequence
import numpy

LOW = 0
HIGH = 1


def get_seq(pulser_wiring, args):

    # %% Parse wiring and args

    # The first 6 args are ns durations and we need them as int64s
    durations = []
    for ind in range(6):
        durations.append(numpy.int64(args[ind]))

    # Unpack the durations
    polarization_dur, exp_dur, aom_delay, rf_delay, \
    readout_dur, pi_pulse = durations

    # Get the APD index
    apd_index = args[6]
    
    #Signify which signal generator to use
    do_uwave_gate = args[7]

    # Get what we need out of the wiring dictionary
    key = 'do_apd_gate_{}'.format(apd_index)
    apd_index = pulser_wiring[key]
    if do_uwave_gate == 0:
        pulser_do_uwave = pulser_wiring['do_uwave_gate_0']
    if do_uwave_gate == 1:
        pulser_do_uwave = pulser_wiring['do_uwave_gate_1']
    pulser_do_aom = pulser_wiring['do_aom']

    # %% Couple calculated values

    half_exp_dur = exp_dur // 2
    exp_dur = half_exp_dur * 2  # This will prevent any rounding errors
    
    if pi_pulse % 2 == 0:
        half_pi_pulse_short = pi_pulse // 2
        half_pi_pulse_long = half_pi_pulse_short
    else:
        half_pi_pulse_short = pi_pulse // 2
        half_pi_pulse_long = half_pi_pulse_short + 1
    
    period = (3 * polarization_dur) + (2 * exp_dur)

    # %% Define the sequence

    seq = Sequence()

    # APD gating
    train = [(polarization_dur, LOW),
             (exp_dur, LOW),
             (readout_dur, HIGH),
             (polarization_dur - readout_dur, LOW),
             (exp_dur, LOW),
             (readout_dur, HIGH),
             (polarization_dur - readout_dur, LOW)]
    seq.setDigital(apd_index, train)

    # Pulse the laser with the AOM for polarization and readout
    train = [(polarization_dur - aom_delay, HIGH),
             (exp_dur, LOW),
             (polarization_dur, HIGH),
             (exp_dur, LOW),
             (polarization_dur + aom_delay, HIGH)]
    seq.setDigital(pulser_do_aom, train)

    # Pulse the microwave for tau
    train = [(polarization_dur - rf_delay, LOW),
             (half_exp_dur - half_pi_pulse_short, LOW),
             (pi_pulse, HIGH),
             (half_exp_dur - half_pi_pulse_long, LOW),
             (polarization_dur, LOW),
             (exp_dur, LOW),
             (polarization_dur + rf_delay, LOW)]
    seq.setDigital(pulser_do_uwave, train)

    return seq, [period]


if __name__ == '__main__':
    wiring = {'do_daq_clock': 0,
              'do_apd_gate_0': 1,
              'do_apd_gate_1': 2,
              'do_aom': 3,
              'do_uwave_gate_0': 4,
              'do_uwave_gate_1': 5}
    
    # polarization_dur, exp_dur, aom_delay, rf_delay, 
    # readout_dur, pi_pulse, apd_index, uwave_gate_index
    args = [3000, 3000, 500, 200, 
            300, 100, 0, 0]
    seq, ret_vals = get_seq(wiring, args)
    seq.plot()   
