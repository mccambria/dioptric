# -*- coding: utf-8 -*-
"""Template for scripts that should be run directly from the files themselves
(as opposed to from the control panel, for example).

Created on Sun Jun 16 11:22:40 2019

@author: mccambria
"""


import numpy as np
import labrad
import utils.tool_belt as tool_belt
import time


def sweep(cxn, sig_gen_name, uwave_freqs, uwave_power):

    sig_gen_cxn = eval(f"cxn.{sig_gen_name}")
    sig_gen_cxn.set_freq(uwave_freqs[0])
    sig_gen_cxn.set_amp(uwave_power)
    sig_gen_cxn.uwave_on()
    pulse_gen = tool_belt.get_server_pulse_gen(cxn)
    pulse_gen.constant([7])

    time.sleep(2)

    for freq in uwave_freqs:
        if tool_belt.safe_stop():
            break
        sig_gen_cxn.set_freq(freq)
        time.sleep(0.03)

    pulse_gen.constant()
    sig_gen_cxn.uwave_off()
    tool_belt.reset_cfm(cxn)
    tool_belt.reset_safe_stop()


def constant(cxn, sig_gen_name, uwave_freq, uwave_power):

    sig_gen_cxn = eval(f"cxn.{sig_gen_name}")
    sig_gen_cxn.set_amp(uwave_power)
    sig_gen_cxn.set_freq(uwave_freq)
    sig_gen_cxn.uwave_on()

    config = tool_belt.get_config_dict()
    pulser_wiring = config["Wiring"]["PulseGen"]
    sig_gen_gate_chan_name = "do_{}_gate".format(sig_gen_name)
    pulser_do_sig_gen_gate = pulser_wiring[sig_gen_gate_chan_name]
    
    pulse_gen = tool_belt.get_server_pulse_gen(cxn)
    pulse_gen.constant([7])
    tool_belt.poll_safe_stop()
    pulse_gen.constant()

    sig_gen_cxn.uwave_off()


def square_wave(cxn, sig_gen_name, uwave_freq, uwave_power):

    uwave_on = int(500)
    uwave_off = int(500)

    sig_gen_cxn = eval(f"cxn.{sig_gen_name}")
    sig_gen_cxn.set_amp(uwave_power)
    sig_gen_cxn.set_freq(uwave_freq)

    seq_file = "uwave_square_wave.py"
    seq_args = [uwave_on, uwave_off, sig_gen_name]
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    pulse_gen = tool_belt.get_server_pulse_gen(cxn)
    pulse_gen.stream_immediate(seq_file, -1, seq_args_string)
    tool_belt.poll_safe_stop()
    sig_gen_cxn.uwave_off()
    tool_belt.reset_cfm(cxn)


if __name__ == "__main__":

    sig_gen_name = "sig_gen_STAN_sg394"

    uwave_freq = 2.87
    half_range = 0.5
    uwave_freqs = np.linspace(
        uwave_freq - half_range, uwave_freq + half_range, 1000
    )

    uwave_power = 0  # dBm

    tool_belt.init_safe_stop()

    with labrad.connect() as cxn:
        # Some parameters you'll need to set in these functions
        # sweep(cxn, sig_gen_name, uwave_freqs, uwave_power)
        constant(cxn, sig_gen_name, uwave_freq, uwave_power)
        # square_wave(cxn, sig_gen_name, uwave_freq, uwave_power)

    tool_belt.reset_cfm()
    tool_belt.reset_safe_stop()
