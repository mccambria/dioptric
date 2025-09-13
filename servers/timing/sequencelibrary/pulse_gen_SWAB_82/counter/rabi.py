# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 17:39:27 2019

@author: mccambria
modified by Saroj Chand on August 2, 2025
"""

import numpy as np
from pulsestreamer import OutputState, Sequence

import utils.tool_belt as tb
from utils import common
from utils.constants import VirtualLaserKey
from utils.tool_belt import Digital


def get_seq(pulse_streamer, config, args):
    ### Unpack and get what we need from config

    # Unpack the durations
    tau, max_tau, uwave_ind = args
    # The pulse streamer expects 64 bit ints
    tau = np.int64(tau)
    max_tau = np.int64(max_tau)

    # Signify which signal generator to use
    virtual_sig_gen_dict = tb.get_virtual_sig_gen_dict(uwave_ind)
    sig_gen_name = virtual_sig_gen_dict["physical_name"]

    # Get which laser to use. Same laser will also be used for readout and polarization
    laser_name = tb.get_physical_laser_name(VirtualLaserKey.SPIN_READOUT)
    readout_dur = tb.get_virtual_laser_dict(VirtualLaserKey.SPIN_READOUT)["duration"]
    polarization_dur = tb.get_virtual_laser_dict(VirtualLaserKey.SPIN_POL)["duration"]
    if readout_dur > polarization_dur:
        raise ValueError("Readout duration must be shorter than polarization duration")

    # Get what we need out of the wiring dictionary
    pulser_wiring = config["Wiring"]["PulseGen"]

    pulser_do_apd_gate = pulser_wiring["do_apd_gate"]
    pulser_do_sig_gen_dm = pulser_wiring[f"do_{sig_gen_name}_dm"]

    # Get the other durations we need
    # print(laser_name)
    laser_delay = config["Optics"]["PhysicalLasers"][laser_name]["delay"]
    uwave_delay = config["Microwaves"]["PhysicalSigGens"][sig_gen_name]["delay"]
    common_delay = max(laser_delay, uwave_delay)
    uwave_buffer = config["CommonDurations"]["uwave_buffer"]

    ### Define the sequence

    seq = Sequence()

    # APD gating - first high is for signal, second high is for reference
    train = [
        (common_delay, Digital.LOW),
        (polarization_dur - readout_dur, Digital.LOW),
        (uwave_buffer, Digital.LOW),
        (max_tau, Digital.LOW),
        (uwave_buffer, Digital.LOW),
        (readout_dur, Digital.HIGH),
        (polarization_dur - readout_dur, Digital.LOW),
        (uwave_buffer, Digital.LOW),
        (max_tau, Digital.LOW),
        (uwave_buffer, Digital.LOW),
        (readout_dur, Digital.HIGH),
    ]
    seq.setDigital(pulser_do_apd_gate, train)
    # Track the total duration for one rep
    total_dur = 0
    for el in train:
        total_dur += el[0]
    print(total_dur)

    # Laser for polarization and readout
    train = [
        (common_delay - laser_delay, Digital.HIGH),
        (polarization_dur - readout_dur, Digital.HIGH),
        (uwave_buffer, Digital.LOW),
        (max_tau, Digital.LOW),
        (uwave_buffer, Digital.LOW),
        (polarization_dur, Digital.HIGH),
        (uwave_buffer, Digital.LOW),
        (max_tau, Digital.LOW),
        (uwave_buffer, Digital.LOW),
        (readout_dur, Digital.HIGH),
        (laser_delay, Digital.HIGH),
    ]
    tb.process_laser_seq(seq, VirtualLaserKey.SPIN_READOUT, train)
    total_dur = 0
    for el in train:
        total_dur += el[0]
    print(total_dur)

    # Pulse the microwave for tau
    train = [
        (common_delay - uwave_delay, Digital.LOW),
        (polarization_dur - readout_dur, Digital.LOW),
        (uwave_buffer, Digital.LOW),
        (tau, Digital.HIGH),
        (max_tau - tau, Digital.LOW),
        (uwave_buffer, Digital.LOW),
        (polarization_dur, Digital.LOW),
        (uwave_buffer, Digital.LOW),
        (max_tau, Digital.LOW),
        (uwave_buffer, Digital.LOW),
        (readout_dur, Digital.LOW),
        (uwave_delay, Digital.LOW),
    ]
    seq.setDigital(pulser_do_sig_gen_dm, train)
    total_dur = 0
    for el in train:
        total_dur += el[0]
    print(total_dur)

    final_digital = [pulser_wiring["do_sample_clock"]]
    final = OutputState(final_digital, 0.0, 0.0)
    return seq, final, [total_dur]


if __name__ == "__main__":
    config = common.get_config_dict()
    # tb.set_delays_to_zero(config)
    args = [100, 1000.0, 0]
    seq = get_seq(None, config, args)[0]
    seq.plot()
