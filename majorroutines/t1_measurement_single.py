# -*- coding: utf-8 -*-
"""
T1 measurement routine.

We'll start by using just one delay to see if this sequence is working.

Created on Wed Apr 24 15:01:04 2019

@author: agardill
"""

# %% Imports


import utils.tool_belt as tool_belt
import majorroutines.optimize as optimize
import numpy
import os
import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import json
from scipy import asarray as ar,exp

# %% Main

def main(cxn, coords, nd_filter, sig_apd_index, ref_apd_index,
         uwave_freq, uwave_power, uwave_pi_pulse, relaxation_time_range,
         num_steps, num_reps, num_runs, 
         name='untitled', measure_spin_0=True):
    
    # %% Defiene the times to be used in the sequence

    # Define some times (in ns)
    # time to intially polarize the nv
    polarization_time = 3 * 10**3
    # time of illumination during which signal readout occurs
    signal_time = 3 * 10**3
    # time of illumination during which reference readout occurs
    reference_time = 3 * 10**3
    # time between polarization and experiment without illumination
    pre_uwave_exp_wait_time = 1 * 10**3
    # time between the end of the experiment and signal without illumination
    post_uwave_exp_wait_time = 1 * 10**3
    # time between signal and reference without illumination
    sig_to_ref_wait_time = pre_uwave_exp_wait_time + post_uwave_exp_wait_time
    # the amount of time the AOM delays behind the gate and rf
    aom_delay_time = 750
    # the amount of time the rf delays behind the AOM and rf
    rf_delay_time = 40
    # the length of time the gate will be open to count photons
    gate_time = 300  
            
    # %% Conditional rf on or off depending on which type of t1 to meassure
    
    if measure_spin_0 == True:
        uwave_pi_pulse = 0

    # %% Create the array of relaxation times
    
    # Array of times to sweep through
    # Must be ints since the pulse streamer only works with int64s
    
    min_relaxation_time = int( relaxation_time_range[0] )
    max_relaxation_time = int( relaxation_time_range[1] )
    
    taus = numpy.linspace(min_relaxation_time, max_relaxation_time,
                          num=num_steps, dtype=numpy.int32)
        
    # %% Create data structure to save the counts
    
    # We create an array of NaNs that we'll fill
    # incrementally for the signal and reference counts. 
    # NaNs are ignored by matplotlib, which is why they're useful for us here.
    # We define 2D arrays, with the horizontal dimension for the frequency and
    # the veritical dimension for the index of the run.
    
    sig_counts = numpy.empty([num_runs, num_steps], dtype=numpy.uint32)
    sig_counts[:] = numpy.nan
    ref_counts = numpy.copy(sig_counts)
    
    coords_used = numpy.array(coords)
    
    # %% Analyze the sequence
    
    # pulls the file of the sequence from serves/timing/sequencelibrary
    file_name = os.path.basename(__file__)
    sequence_args = [min_relaxation_time, polarization_time, signal_time, reference_time, 
                    sig_to_ref_wait_time, pre_uwave_exp_wait_time, 
                    post_uwave_exp_wait_time, aom_delay_time, rf_delay_time, 
                    gate_time, uwave_pi_pulse, max_relaxation_time,
                    sig_apd_index, ref_apd_index]
    ret_vals = cxn.pulse_streamer.stream_load(file_name, sequence_args, 1)
    seq_time = ret_vals[0]
    print(sequence_args)
    print(seq_time)
    
    # %% Ask user if they wish to run experiment based on run time
    
    seq_time_s = seq_time / (10**9)  # s
    expected_run_time = num_steps * num_reps * num_runs * seq_time_s  # s
    expected_run_time_m = expected_run_time / 60 # s

    
    msg = 'Expected run time: {} minutes. ' \
        'Enter \'y\' to continue: '.format(expected_run_time_m)
    if input(msg) != 'y':
        return
    
    # %% Get the starting time of the function, to be used to calculate run time

    startFunctionTime = time.time()
    
     # %% Set up the microwaves

    cxn.microwave_signal_generator.set_freq(uwave_freq)
    cxn.microwave_signal_generator.set_amp(uwave_power)
    cxn.microwave_signal_generator.uwave_on()
    
    # %% Collect the data
    
    optimize_failed = False
    
    # Start 'Press enter to stop...'
    tool_belt.init_safe_stop()
    
    for run_ind in range(num_runs):

        print(str(run_ind))
        
        # Break out of the while if the user says stop
        if tool_belt.safe_stop():
            break
        
        coords = optimize.main(cxn, coords, nd_filter, sig_apd_index)
        if None in coords:
            optimize_failed = True
        coords_used = numpy.vstack([coords_used, numpy.array(coords)])
        
        # Load the APD tasks
        cxn.apd_counter.load_stream_reader(sig_apd_index, seq_time, num_steps)
        cxn.apd_counter.load_stream_reader(ref_apd_index, seq_time, num_steps)   
                
        for tau_ind in range(len(taus)):
            
            # Break out of the while if the user says stop
            if tool_belt.safe_stop():
                break  
            
            # Stream the sequence
            args = [taus[tau_ind], polarization_time, signal_time, reference_time, 
                    sig_to_ref_wait_time, pre_uwave_exp_wait_time, 
                    post_uwave_exp_wait_time, aom_delay_time, rf_delay_time, 
                    gate_time, uwave_pi_pulse, max_relaxation_time,
                    sig_apd_index, ref_apd_index]
            
            cxn.pulse_streamer.stream_immediate(file_name, num_reps, args, 1)
    
            count = cxn.apd_counter.read_stream(sig_apd_index, 1)
            sig_counts[run_ind, tau_ind] = count
#            print('sig = ' + str(count))
            
            count = cxn.apd_counter.read_stream(ref_apd_index, 1)
            ref_counts[run_ind, tau_ind] = count  
#            print('ref = ' + str(count))

    # %% Turn off the signal generator

    cxn.microwave_signal_generator.uwave_off()
    
    # %% Average the counts over the iterations

    avg_sig_counts = numpy.average(sig_counts, axis=0)
    avg_ref_counts = numpy.average(ref_counts, axis=0)
    
    # %% Calculate the t1 data, signal / reference over different relaxation times

    norm_avg_sig = avg_sig_counts / avg_ref_counts
    
    # %% Plot the t1 signal

    # Different title for the plot based on the measurement
    if measure_spin_0 == True:
        spin = 'ms = 0'
    else:
        spin = 'ms = +/- 1'

    raw_fig, axes_pack = plt.subplots(1, 2, figsize=(17, 8.5))

    ax = axes_pack[0]
    ax.plot(taus / 10**6, avg_sig_counts, 'r-', label = 'signal')
    ax.plot(taus / 10**6, avg_ref_counts, 'g-', label = 'reference')
    ax.set_xlabel('Relaxation time (ms)')
    ax.set_ylabel('Counts')
    ax.legend()

    ax = axes_pack[1]
    ax.plot(taus / 10**6, norm_avg_sig, 'b-')
    ax.set_title('T1 Measurement of ' + spin)
    ax.set_xlabel('Relaxation time (ms)')
    ax.set_ylabel('Contrast (arb. units)')

    raw_fig.canvas.draw()
    # fig.set_tight_layout(True)
    raw_fig.canvas.flush_events()
    
    # %% Save the data

    endFunctionTime = time.time()

    timeElapsed = endFunctionTime - startFunctionTime

    timestamp = tool_belt.get_time_stamp()
    
    raw_data = {'timestamp': timestamp,
            'timeElapsed': timeElapsed,
            'name': name,
            'spin_measured?': spin,
            'coords': coords.tolist(),
            'coords_used': coords_used.tolist(),
            'coords-units': 'V',
            'optimize_failed': optimize_failed,
            'nd_filter': nd_filter,
            'uwave_freq': uwave_freq,
            'uwave_freq-units': 'GHz',
            'uwave_power': uwave_power,
            'uwave_power-units': 'dBm',
            'uwave_pi_pulse': uwave_pi_pulse,
            'uwave_pi_pulse-units': 'ns',
            'relaxation_time_range': relaxation_time_range,
            'relaxation_time_range-units': 'ns',
            'num_steps': num_steps,
            'num_reps': num_reps,
            'num_runs': num_runs,
            'sig_counts': sig_counts.astype(int).tolist(),
            'sig_counts-units': 'counts',
            'ref_counts': ref_counts.astype(int).tolist(),
            'ref_counts-units': 'counts',
            'norm_avg_sig': norm_avg_sig.astype(float).tolist(),
            'norm_avg_sig-units': 'arb'}
    
    file_path = tool_belt.get_file_path(__file__, timestamp, name)
    tool_belt.save_figure(raw_fig, file_path)
    tool_belt.save_raw_data(raw_data, file_path)
    
# %%    
    
def decayExp(t, offset, amplitude, decay):
    return offset + amplitude * exp(-decay * t)    
    
# %% Fitting the data
    
def t1_exponential_decay(open_file_name, save_file_type):
    
    directory = 'E:/Team Drives/Kolkowitz Lab Group/nvdata/t1_measurement/'
   
    # Open the specified file
    with open(directory + open_file_name + '.txt') as json_file:
        
        # Load the data from the file
        data = json.load(json_file)
        countsT1 = data["norm_avg_sig"]
        relaxation_time_range = data["relaxation_time_range"]
        num_steps = data["num_steps"]
        spin = data["spin_measured?"]
        
    min_relaxation_time = relaxation_time_range[0] 
    max_relaxation_time = relaxation_time_range[1]
        
    timeArray = numpy.linspace(min_relaxation_time, max_relaxation_time,
                              num=num_steps, dtype=numpy.int32)
    
    offset = 0.8
    amplitude = 0.1
    decay = 1/10000 # inverse ns

    popt,pcov = curve_fit(decayExp, timeArray, countsT1, 
                              p0=[offset, amplitude, decay])
    
    decay_time = 1 / popt[2]
            
    first = timeArray[0]
    last = timeArray[len(timeArray)-1]
    linspaceTime = numpy.linspace(first, last, num=1000)
    
    
    fig, ax= plt.subplots(1, 1, figsize=(10, 8))
    ax.plot(timeArray / 10**6, countsT1,'bo',label='data')
    ax.plot(linspaceTime / 10**6, decayExp(linspaceTime,*popt),'r-',label='fit')
    ax.set_xlabel('Dark Time (ms)')
    ax.set_ylabel('Contrast (arb. units)')
    ax.set_title('T1 of ' + str(spin))
    ax.legend()
    
    text = "\n".join((r'$C + A_0 e^{-t / d}$',
                      r'$C = $' + '%.1f'%(popt[0]),
                      r'$A_0 = $' + '%.1f'%(popt[1]),
                      r'$d = $' + "%.3f"%(decay_time / 10**6) + " ms"))
    
    
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(0.70, 0.95, text, transform=ax.transAxes, fontsize=12,
                            verticalalignment="top", bbox=props)
    
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    fig.savefig(open_file_name + 'replot.' + save_file_type)

    