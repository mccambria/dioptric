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

# %% Main

def main(cxn, coords, nd_filter, sig_shrt_apd_index, ref_shrt_apd_index,
         sig_long_apd_index, ref_long_apd_index,
         uwave_freq, uwave_power, uwave_pi_pulse, relaxation_time_range,
         num_steps, num_reps, num_runs, 
         name='untitled', measure_spin_0 = True):
    
    # %% Get the starting time of the function, to be used to calculate run time

    startFunctionTime = time.time()

    # %% Initial calculations and setup

    # Define some times (in ns)
    # time to intially polarize the nv
    polarization_time = 3 * 10**3
    # time of illumination during which signal readout occurs
    signal_time = 3 * 10**3
    # time of illumination during which reference readout occurs
    reference_time = 3 * 10**3
    # time between signal and reference without illumination
    sig_to_ref_wait_time = 1 * 10**3
    # time between polarization and pi pulse without illumination
    pol_to_piPulse_wait_time = 500
    # time between the second pi pulse and signal without illumination
    piPulse_to_pol_wait_time = 500
    # the amount of time the AOM delays behind the gate and rf ##do we really know it lags behind the rf???
    aom_delay_time = 750
    # the amount of time the rf delays behind the AOM and rf
    rf_delay_time = 40
    # the length of time the gate will be open to count photons
    gate_time = 300    
    
    # %% Create the array of relaxation times
    
    # Array of times to sweep through
    # Must be ints since the pulse streamer only works with int64s
    
    # We don't want the min time to be 0, but instead some amount of time so 
    # that the two rf signals or polariation pulses don't overlap.
    # Let's make that time 1 us, and we will redefine the min time
    
    # We also want to add this time to the maxTime so that the time divisions
    # work out to what we expect
    
    min_relaxation_time = relaxation_time_range[0] + 1 * 10**3
    max_relaxation_time = relaxation_time_range[1] + 1 * 10**3
    
    taus = numpy.linspace(min_relaxation_time, max_relaxation_time,
                          num=num_steps, dtype=numpy.int32)
 
    # %% Fix the length of the sequence FIXXX
     
    # Our sequence pairs the longest time with the shortest time, and steps 
    # toward the middle. This means we only step through half of the length
    # of the time array. If the number of elements is odd, we need to make the
    # half length and integer to appease python. Adding 1.5 to the divided 
    # number allows the loop to step through the middle value twice
    
#    if len(timeArray) % 2 == 0:
#        lengthTimeArray = len(timeArray)
#    elif len(timeArray) % 2 == 1:
#        lengthTimeArray = len(timeArray) + 1  
    
    # %% Conditional rf on or off depending on which type of t1 to meassure
    
    if measure_spin_0 == True:
        uwave_pi_pulse = 0
        pol_to_piPulse_wait_time = 0
        piPulse_to_pol_wait_time = 0
    
    # %% Analyze the sequence
    
    # pulls the file of the sequence from serves/timing/sequencelibrary
    file_name = os.path.basename(__file__)
    sequence_args = [taus[0], polarization_time, signal_time, reference_time, 
                    sig_to_ref_wait_time, pol_to_piPulse_wait_time, 
                    piPulse_to_pol_wait_time, aom_delay_time, rf_delay_time, 
                    gate_time, uwave_pi_pulse, max_relaxation_time,
                    sig_shrt_apd_index, ref_shrt_apd_index,
                    sig_long_apd_index, ref_long_apd_index]
    ret_vals = cxn.pulse_streamer.stream_load(file_name, sequence_args, 1)
    period = ret_vals[0]
    
    # %% Create data structure to save the counts
    
    # We create an array of NaNs that we'll fill
    # incrementally for the signal and reference counts. 
    # NaNs are ignored by matplotlib, which is why they're useful for us here.
    # We define 2D arrays, with the horizontal dimension for the frequency and
    # the veritical dimension for the index of the run.
    
    sig_counts = numpy.empty([num_runs, num_steps], dtype=numpy.uint32)
    sig_counts[:] = numpy.nan
    ref_counts = numpy.copy(sig_counts)
    
     # %% Set up the microwaves

    cxn.microwave_signal_generator.set_freq(uwave_freq)
    cxn.microwave_signal_generator.set_amp(uwave_power)
    cxn.microwave_signal_generator.uwave_on()
    
    # %% Collect the data
    
    optimize_failed = False
    
    # Start 'Press enter to stop...'
    tool_belt.init_safe_stop()
    
    for run_ind in range(num_runs):

        print(' ' +
              str(run_ind))
        
        # Break out of the while if the user says stop
        if tool_belt.safe_stop():
            break
        
        xyz_centers = optimize.main(cxn, coords, nd_filter, sig_shrt_apd_index)
        if None in xyz_centers:
            optimize_failed = True
            
        # Load the APD tasks
        cxn.apd_counter.load_stream_reader(sig_shrt_apd_index, period, num_steps)
        cxn.apd_counter.load_stream_reader(ref_shrt_apd_index, period, num_steps)
#        cxn.apd_counter.load_stream_reader(sig_long_apd_index, period, num_steps)
#        cxn.apd_counter.load_stream_reader(ref_long_apd_index, period, num_steps)    
                
        for tau_ind in range(len(taus)):
  
            # Break out of the while if the user says stop
            if tool_belt.safe_stop():
                break  
            # Stream the sequence
            args = [taus[tau_ind], polarization_time, signal_time, reference_time, 
                    sig_to_ref_wait_time, pol_to_piPulse_wait_time, 
                    piPulse_to_pol_wait_time, aom_delay_time, rf_delay_time, 
                    gate_time, uwave_pi_pulse, max_relaxation_time,
                    sig_shrt_apd_index, ref_shrt_apd_index,
                    sig_long_apd_index, ref_long_apd_index]
            
            cxn.pulse_streamer.stream_immediate(file_name, num_reps, args, 1)
    
            count = cxn.apd_counter.read_stream(sig_shrt_apd_index, 1)
            sig_counts[run_ind, tau_ind] = count

            count = cxn.apd_counter.read_stream(ref_shrt_apd_index, 1)
            ref_counts[run_ind, tau_ind] = count    
            
    # %% Turn off the signal generator

    cxn.microwave_signal_generator.uwave_off()
    
    # %% Average the counts over the iterations

    avg_sig_counts = numpy.average(sig_counts, axis=0)
    avg_ref_counts = numpy.average(ref_counts, axis=0)
    
    # %% Calculate the t1 data, signal / reference over different relaxation times

    avg_norm_sig = (avg_sig_counts) / (avg_ref_counts)
    
    # %% Plot the t1 signal

    # Different title for the plot based on the measurement
    if measure_spin_0 == True:
        spin = 'ms = 0'
    else:
        spin = 'ms = +/- 1'

    raw_fig, axes_pack = plt.subplots(1, 2, figsize=(17, 8.5))

    ax = axes_pack[0]
    ax.plot(taus / 10**6, avg_sig_counts, 'r-')
    ax.plot(taus / 10**6, avg_ref_counts, 'g-')
    ax.set_xlabel('Relaxation time (ms)')
    ax.set_ylabel('Counts')

    ax = axes_pack[1]
    ax.plot(taus / 10**6, avg_norm_sig, 'b-')
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
            'coords': coords,
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
            'avg_norm_sig': avg_norm_sig.astype(float).tolist(),
            'avg_norm_sig-units': 'arb'}
    
    file_path = tool_belt.get_file_path(__file__, timestamp, name)
    tool_belt.save_figure(raw_fig, file_path)
    tool_belt.save_raw_data(raw_data, file_path)