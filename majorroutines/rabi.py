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


def main(cxn, name, coords, sig_apd_index, ref_apd_index,
         uwave_freq, uwave_power, uwave_time_range,
         num_steps, num_reps, num_runs):

    # %% Initial calculations and setup

    # Define some times (in ns)
    polarization_time = 3 * 10**3
    reference_time = 1 * 10**3
    signal_wait_time = 1 * 10**3
    reference_wait_time = 2 * 10**3
    background_wait_time = 1 * 10**3
    aom_delay_time = 750
    gate_time = 300

    # Array of times to sweep through
    min_uwave_time = uwave_time_range[0]
    max_uwave_time = uwave_time_range[1]
    taus = numpy.linspace(min_uwave_time, max_uwave_time, num=num_steps)

    # Analyze the sequence
    file_name = os.path.basename(__file__)
    file_name_no_ext = os.path.splitext(file_name)[0]
    args = [taus[0], polarization_time, reference_time, signal_wait_time,
            reference_wait_time, background_wait_time,
            aom_delay_time, gate_time, max_uwave_time]
    period = cxn.pulse_streamer.stream_load(file_name, args)

    # The period is independent of the particular tau, but it must be long
    # enough to accomodate the longest tau
    # period = aom_delay_time + polarization_time + reference_wait_time + \
    #     reference_wait_time + polarization_time + reference_wait_time + \
    #     reference_time + max_uwave_time

    # Set up our data structure, an array of NaNs that we'll fill
    # incrementally. NaNs are ignored by matplotlib, which is why they're
    # useful for us here.
    # We define 2D arrays, with the horizontal dimension for the frequency and
    # the veritical dimension for the index of the run.
    sig_counts = numpy.empty([num_runs, num_steps], dtype=numpy.uint32)
    sig_counts[:] = numpy.nan
    ref_counts = numpy.copy(sig_counts)
    # norm_counts = numpy.empty([num_runs, num_steps])


    # %% Load the APD tasks

    num_samples_per_run = 2 * num_steps  # Two samples for each frequency step
    num_samples = num_runs * num_samples_per_run
    cxn.apd_counter.load_stream_reader(sig_apd_index,
                                       period, num_samples)
    cxn.apd_counter.load_stream_reader(ref_apd_index,
                                       period, num_samples)

    # %% Set up the microwaves

    cxn.microwave_signal_generator.set_freq(uwave_freq)
    cxn.microwave_signal_generator.set_amp(uwave_power)
    cxn.microwave_signal_generator.uwave_on()

    # %% Collect the data

    tool_belt.set_xyz(cxn, coords)

    # Start 'Press enter to stop...'
    tool_belt.init_safe_stop()

    for run_ind in range(num_runs):

        # optimize.main(cxn, name, coords, apd_index)

        for tau_ind in range(len(taus)):

            # Load the sequence if it hasn't already been loaded
            if (run_ind != 0) and (tau_ind != 0):
                args = [taus[0], polarization_time, reference_time,
                        signal_wait_time, reference_wait_time,
                        background_wait_time, aom_delay_time,
                        gate_time, max_uwave_time]
                cxn.pulse_streamer.stream_load(file_name, args)

            cxn.pulse_streamer.stream_start(num_reps)

            count = cxn.apd_counter.read_stream(sig_apd_index, 1)
            sig_counts[run_ind, tau_ind] = count

            count = cxn.apd_counter.read_stream(ref_apd_index, 1)
            ref_counts[run_ind, tau_ind] = count

    # %% Average the counts over the iterations

    sig_counts_avg = numpy.average(sig_counts, axis=0)
    ref_counts_avg = numpy.average(ref_counts, axis=0)

    # %% Calculate the Rabi data, signal / reference over different Tau

    norm_counts = (countsSignalAveraged) / (countsReferenceAveraged)

    # %% Fit the data and extract piPulse

    # Estimated fit parameters
    offset = 0.9
    amplitude = 0.01
    frequency = 1/100
    phase = 1.57
    decay = 10**-7

    popt, pcov = curve_fit(tool_belt.sinexp, taus, norm_counts,
                           p0=[offset, amplitude, frequency, phase, decay])

    period = 1 / popt[2]

    # %% Plot the Rabi signal

    fig1, axesPack = plt.subplots(1, 2, figsize=(17, 8.5))

    ax = axesPack[0]
    ax.plot(tauArray, countsSignalAveraged, 'r-')
    ax.plot(tauArray, countsReferenceAveraged, 'g-')
#    ax.plot(tauArray, countsBackground, 'o-')
    ax.set_xlabel('rf time (ns)')
    ax.set_ylabel('Counts')

    ax = axesPack[1]
    ax.plot(tauArray , countsRabi, 'b-')
    ax.set_title('Normalized Signal with varying rf time')
    ax.set_xlabel('rf time (ns)')
    ax.set_ylabel('Normalized signal')

    fig1.canvas.draw()
    fig1.set_tight_layout(True)
    fig1.canvas.flush_events()

    # %% Plot the data itself and the fitted curve

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.plot(tauArray, countsRabi,'bo',label='data')
    ax.plot(tauArray, sinexp(tauArray,*popt),'r-',label='fit')
    ax.set_xlabel('rf Time (ns)')
    ax.set_ylabel('Contrast (arb. units)')
    ax.set_title('Rabi Oscillation of Nitrogen-Vacancy Center electron spin')
    ax.legend()
    text = "\n".join((r'$C + A_0 \mathrm{sin}(\nu * 2 \pi * t + \phi) e^{-d * t}$',
                      r'$\frac{1}{\nu} = $' + "%.1f"%(period) + " ns",
                      r'$A_0 = $' + "%.3f"%(popt[1]),
                      r'$d = $' + "%.3f"%(popt[4]) + ' ' + r'$ ns^{-1}$'))


    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(0.55, 0.25, text, transform=ax.transAxes, fontsize=12,
                            verticalalignment="top", bbox=props)

    fig.canvas.draw()
    fig.set_tight_layout(True)
    fig.canvas.flush_events()

# %% Turn off the RF and save the data

    sigGen.write("ENBR 0")

    timeStamp = tool_belt.get_time_stamp()

    rawData = {"timeStamp": timeStamp,
               "name": name,
               "xyzCenters": [xCenter, yCenter, zCenter],
               "rfFrequency": rfFrequency,
               "rfPower": rfPower,
               "rfMinTime": int(rfMinTime),
               "rfMaxTime": int(rfMaxTime),
               "numTimeSteps": numTimeSteps,
               "numSamples": numSamples,
               "numAveraged": numAverage,
               "rabi": countsRabi.astype(float).tolist(),
               "rawAveragedCounts": [countsSignalAveraged.astype(int).tolist(),
                        countsReferenceAveraged.astype(int).tolist()],
               "rawCounts": [countsSignal.astype(int).tolist(),
                        countsReference.astype(int).tolist()]}


    filePath = tool_belt.get_file_path("rabi", timeStamp, name)
    tool_belt.save_figure(fig1, filePath)
    tool_belt.save_figure(fig, filePath + 'fitting')
    tool_belt.save_raw_data(rawData, filePath)

# %% Return value for pi pulse

    return numpy.int64(period)
