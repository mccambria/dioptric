#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 11:10:35 2022
file to help find fidelity emperically, outside of having to use determine_charge_readout_params file

@author: carterfox
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
import utils.tool_belt as tool_belt
import scipy.stats as stats
import majorroutines.optimize as optimize
from majorroutines.charge_majorroutines.determine_scc_pulse_params import calc_histogram, calculate_threshold_no_model
import json


def calc_fidelity(nv0_counts, nvm_counts):
    
    readout_dur = 10e9 # just make it super big because we just want to use what was measured
    
    occur_0, x_vals_0, occur_m, x_vals_m = calc_histogram(nv0_counts, nvm_counts, readout_dur, bins=None)
    
    max_x_val = max(list(x_vals_0) + list(x_vals_m)) + 10

    num_reps = len(nv0_counts)
    mean_0 = sum(occur_0 * x_vals_0) / num_reps
    mean_m = sum(occur_m * x_vals_m) / num_reps
    
    threshold, fidelity, fig = calculate_threshold_no_model(
        readout_dur,
        occur_0,
        occur_m,
        mean_0,
        mean_m,
        x_vals_0,
        x_vals_m,
        1,
        None,
    )
    
    return threshold, fidelity


if __name__ == "__main__":
    
    filename = '2022_12_10-08_56_16-johnson-search-ion_pulse_dur'
    
    data = tool_belt.get_raw_data(filename)
    print(data)

    # calc_fidelity(nv0_counts, nvm_counts)


