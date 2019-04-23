# -*- coding: utf-8 -*-
"""
Rabi flopping routine. Sweeps the pulse duration of a fixed uwave frequency.

Created on Tue Apr 23 11:49:23 2019

@author: mccambria
"""


# %% Imports


import Utils.tool_belt as tool_belt
import majorroutines.optimize as optimize
import numpy
import os
import matplotlib.pyplot as plt


# %% Main


def main(cxn, name, coords, apd0_index, apd1_index,
         switch_delay, uwave_freq, uwave_power,
         uwave_time_range, num_steps, num_runs):

    # %% Initial calculations and setup

    # Array of times to sweep through
    taus = numpy.linspace(uwave_time_range[0], uwave_time_range[1],
                          num=num_steps)

    # Define some times

    polarizationTime = numpy.int64(3 * 10**3)
    referenceTime = numpy.int64(1 * 10**3)
    signalWaitTime = numpy.int64(1 * 10**3)
    referenceWaitTime = numpy.int64(2 * 10**3)
    backgroundWaitTime = numpy.int64(1 * 10**3)
    AOMDelay = numpy.int64(AOMDelayTmp)
    gateTime = numpy.int64(300)


    # Total period
    totalTime = AOMDelay + polarizationTime + referenceWaitTime + \
        referenceWaitTime + polarizationTime + referenceWaitTime + \
        referenceTime + rfMaxTime

    # %% Set up the microwaves

    cxn.microwave_signal_generator.set_freq(uwave_freq)
    cxn.microwave_signal_generator.set_amp(uwave_power)
    cxn.microwave_signal_generator.uwave_on()'

    # %% Collect the data

    # Start 'Press enter to stop...'
    tool_belt.init_safe_stop()

    for run_ind in range(num_runs):

        optimize.main(cxn, name, coords, apd_index)
