# -*- coding: utf-8 -*-
"""
Test for composite pulse

Created on Wed Mar 17 14:13:32 2021

@author: mccambria
"""


import utils.tool_belt as tool_belt
import majorroutines.optimize as optimize
from scipy.optimize import minimize_scalar
from numpy import pi
import numpy
import time
import matplotlib.pyplot as plt
from random import shuffle
import labrad
from utils.tool_belt import States
from scipy.optimize import curve_fit
from numpy.linalg import eigvals


def main():
    
    try:
        with labrad.connect() as cxn:
            main_with_cxn(cxn)
    finally:
        tool_belt.reset_cfm()
        if tool_belt.check_safe_stop_alive():
            print('\n\nRoutine complete. Press enter to exit.')
            tool_belt.poll_safe_stop()


def main_with_cxn(cxn):
    
    # sig_gen_cxn = cxn.signal_generator_tsg4104a
    sig_gen_cxn = cxn.signal_generator_sg394
    sig_gen_cxn.set_freq(0.4)
    sig_gen_cxn.set_amp(10)
    sig_gen_cxn.load_iq()
    sig_gen_cxn.uwave_on()
    
    cxn.arbitrary_waveform_generator.iq_switch()
    # uwave_dur, gap, switch_delay, iq_delay, sig_gen
    # seq_args = [3200, 48, 0, 0, 'tsg4104a']
    seq_args = [160, 16, 0, 0, 'sg394']
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    ret_vals = cxn.pulse_streamer.stream_immediate('composite_pulses.py',
                                                    -1, seq_args_string)
    
    # cxn.arbitrary_waveform_generator.test_sin()
    
    input('Press enter to stop...')
    
    

if __name__ == '__main__':

    main()