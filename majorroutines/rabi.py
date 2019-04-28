# -*- coding: utf-8 -*-
"""
Rabi flopping routine. Sweeps the pulse duration of a fixed uwave frequency.

Created on Tue Apr 23 11:49:23 2019

@author: mccambria
"""


# %% Imports


import utils.tool_belt as tool_belt
import majorroutines.optimize as optimize
import numpy
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# %% Main


def main(cxn, coords, sig_apd_index, ref_apd_index,
         uwave_freq, uwave_power, uwave_time_range,
         num_steps, num_reps, num_runs, name='untitled'):

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
    # Must be ints since the pulse streamer only works with int64s
    min_uwave_time = uwave_time_range[0]
    max_uwave_time = uwave_time_range[1]
    taus = numpy.linspace(min_uwave_time, max_uwave_time,
                          num=num_steps, dtype=numpy.int32)

    # Analyze the sequence
    file_name = os.path.basename(__file__)
    file_name_no_ext = os.path.splitext(file_name)[0]
    sequence_args = [taus[0], polarization_time, reference_time,
                    signal_wait_time, reference_wait_time,
                    background_wait_time, aom_delay_time,
                    gate_time, max_uwave_time,
                    sig_apd_index, ref_apd_index]
    ret_vals = cxn.pulse_streamer.stream_load(file_name, sequence_args, 1)
    period = ret_vals[0]

    # Set up our data structure, an array of NaNs that we'll fill
    # incrementally. NaNs are ignored by matplotlib, which is why they're
    # useful for us here.
    # We define 2D arrays, with the horizontal dimension for the frequency and
    # the veritical dimension for the index of the run.
    sig_counts = numpy.empty([num_runs, num_steps], dtype=numpy.uint32)
    sig_counts[:] = numpy.nan
    ref_counts = numpy.copy(sig_counts)
    # norm_avg_sig = numpy.empty([num_runs, num_steps])

    # %% Set up the microwaves

    cxn.microwave_signal_generator.set_freq(uwave_freq)
    cxn.microwave_signal_generator.set_amp(uwave_power)
    cxn.microwave_signal_generator.uwave_on()

    # %% Collect the data

#    tool_belt.set_xyz(cxn, coords)

    # Start 'Press enter to stop...'
    tool_belt.init_safe_stop()

    for run_ind in range(num_runs):

        # Break out of the while if the user says stop
        if tool_belt.safe_stop():
            break

        optimize.main(cxn, coords, sig_apd_index)

        # Load the APD tasks
        cxn.apd_counter.load_stream_reader(sig_apd_index, period, num_steps)
        cxn.apd_counter.load_stream_reader(ref_apd_index, period, num_steps)

        for tau_ind in range(len(taus)):

            # Break out of the while if the user says stop
            if tool_belt.safe_stop():
                break

            # Stream the sequence
            args = [taus[tau_ind], polarization_time, reference_time,
                    signal_wait_time, reference_wait_time,
                    background_wait_time, aom_delay_time,
                    gate_time, max_uwave_time,
                    sig_apd_index, ref_apd_index]
            cxn.pulse_streamer.stream_immediate(file_name, num_reps, args, 1)

            count = cxn.apd_counter.read_stream(sig_apd_index, 1)
            sig_counts[run_ind, tau_ind] = count

            count = cxn.apd_counter.read_stream(ref_apd_index, 1)
            ref_counts[run_ind, tau_ind] = count

    # %% Turn off the signal generator

    cxn.microwave_signal_generator.uwave_off()

    # %% Average the counts over the iterations

    sig_counts_avg = numpy.average(sig_counts, axis=0)
    ref_counts_avg = numpy.average(ref_counts, axis=0)

    # %% Calculate the Rabi data, signal / reference over different Tau

    norm_avg_sig = (sig_counts_avg) / (ref_counts_avg)

    # %% Fit the data and extract piPulse

    # Estimated fit parameters
    offset = 0.9
    amplitude = 0.01
    frequency = 1/100
    phase = 1.57
    decay = 10**-7

    init_params = [offset, amplitude, frequency, phase, decay]

    opti_params, cov_arr = curve_fit(tool_belt.sinexp, taus, norm_avg_sig,
                                     p0=init_params)

    period = 1 / opti_params[2]

    # %% Plot the Rabi signal

    fig, axes_pack = plt.subplots(1, 2, figsize=(17, 8.5))

    ax = axes_pack[0]
    ax.plot(taus, sig_counts_avg, 'r-')
    ax.plot(taus, ref_counts_avg, 'g-')
    # ax.plot(tauArray, countsBackground, 'o-')
    ax.set_xlabel('rf time (ns)')
    ax.set_ylabel('Counts')

    ax = axes_pack[1]
    ax.plot(taus , norm_avg_sig, 'b-')
    ax.set_title('Normalized Signal With Varying Microwave Duration')
    ax.set_xlabel('Microwave duration (ns)')
    ax.set_ylabel('Contrast (arb. units)')

    fig.canvas.draw()
    # fig.set_tight_layout(True)
    fig.canvas.flush_events()

    # %% Plot the data itself and the fitted curve

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.plot(taus, norm_avg_sig,'bo',label='data')
    ax.plot(taus, tool_belt.sinexp(taus, *opti_params), 'r-', label='fit')
    ax.set_xlabel('Microwave duration (ns)')
    ax.set_ylabel('Contrast (arb. units)')
    ax.set_title('Rabi Oscillation Of NV Center Electron Spin')
    ax.legend()
    text = '\n'.join((r'$C + A_0 \mathrm{sin}(\nu * 2 \pi * t + \phi) e^{-d * t}$',
                      r'$\frac{1}{\nu} = $' + '%.1f'%(period) + ' ns',
                      r'$A_0 = $' + '%.3f'%(opti_params[1]),
                      r'$d = $' + '%.3f'%(opti_params[4]) + ' ' + r'$ ns^{-1}$'))


    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.55, 0.25, text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    fig.canvas.draw()
    # fig.set_tight_layout(True)
    fig.canvas.flush_events()

    # %% Save the data

    timestamp = tool_belt.get_time_stamp()

    raw_data = {'timestamp': timestamp,
               'name': name,
               'xyz_centers': coords,
               'uwave_freq': uwave_freq,
               'uwave_power': uwave_power,
               'uwave_time_range': uwave_time_range,
               'num_steps': num_steps,
               'num_reps': num_reps,
               'num_runs': num_runs,
               'sig_counts': sig_counts.astype(int).tolist(),
               'ref_counts': ref_counts.astype(int).tolist(),
               'norm_avg_sig': norm_avg_sig.astype(int).tolist()}

    file_path = tool_belt.get_file_path(file_name_no_ext, timestamp, name)
    tool_belt.save_figure(fig, file_path)
    tool_belt.save_figure(fig, file_path + '_fitting')
    tool_belt.save_raw_data(raw_data, file_path)

    # %% Return value for pi pulse

    return numpy.int64(period)
