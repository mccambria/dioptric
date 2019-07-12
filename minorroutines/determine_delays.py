# -*- coding: utf-8 -*-
"""
Plot the counts obtained by moving the AOM on time so that we can
determine the delay of the AOM relative to the APD gating.

Created on Fri Jul 12 13:53:45 2019

@author: mccambria
"""


# %% Imports


import labrad
import utils.tool_belt as tool_belt
import majorroutines.optimize as optimize
from random import shuffle
import numpy
import matplotlib.pyplot as plt


# %% Functions


# %% Main


def aom_delay(cxn, nv_sig, nd_filter, readout, apd_indices,
              delay_range, num_steps, num_reps, 
              name='untitled'):
    
    taus = numpy.linspace(delay_range[0], delay_range[1],
                          num_steps, dtype=numpy.int32)
    tau_ind_list = list(range(num_steps))
    shuffle(tau_ind_list)
    
    sig_counts = numpy.empty(num_steps, dtype=numpy.uint32)
    sig_counts[:] = numpy.nan
    ref_counts = numpy.copy(sig_counts)
    
    optimize.main(cxn, nv_sig, nd_filter, apd_indices)
    
    tool_belt.reset_cfm(cxn)
    cxn.apd_tagger.start_tag_stream(apd_indices)  
    
    tool_belt.init_safe_stop()
    
    for tau_ind in tau_ind_list:
        
        # Break out of the while if the user says stop
        if tool_belt.safe_stop():
            break
        
        tau = taus[tau_ind]
        seq_args = [tau, readout, apd_indices[0]]
        cxn.pulse_streamer.stream_immediate('aom_delay.py', num_reps, seq_args, 1)
        
        new_counts = cxn.apd_tagger.read_counter_separate_gates(1)
        sample_counts = new_counts[0]
        sig_counts[tau_ind] = sum(sample_counts[0::2])
        ref_counts[tau_ind] = sum(sample_counts[1::2])
            
    cxn.apd_tagger.stop_tag_stream()
    tool_belt.reset_cfm(cxn)
    
    # kcps
    sig_count_rates = (sig_counts / (num_reps * 1000)) / (readout / (10**9))
    ref_count_rates = (ref_counts / (num_reps * 1000)) / (readout / (10**9))
    norm_avg_sig = sig_counts / ref_counts

    fig, axes_pack = plt.subplots(1, 2, figsize=(17, 8.5))
    ax = axes_pack[0]
    ax.plot(taus, sig_count_rates, 'r-', label = 'signal')
    ax.plot(taus, ref_count_rates, 'g-', label = 'reference')
    ax.set_title('Counts vs AOM Delay Time')
    ax.set_xlabel('Delay time (ns)')
    ax.set_ylabel('Count rate (kcps)')
    ax = axes_pack[1]
    ax.plot(taus, norm_avg_sig, 'b-')
    ax.set_title('Contrast vs AOM Delay Time')
    ax.set_xlabel('Delay time (ns)')
    ax.set_ylabel('Contrast (arb. units)')
    fig.canvas.draw()
    fig.set_tight_layout(True)
    fig.canvas.flush_events()
    
    timestamp = tool_belt.get_time_stamp()
    raw_data = {'timestamp': timestamp,
            'nv_sig': nv_sig,
            'nv_sig-units': tool_belt.get_nv_sig_units(),
            'nv_sig-format': tool_belt.get_nv_sig_format(),
            'nd_filter': nd_filter,
            'readout': readout,
            'readout-units': 'ns',
            'delay_range': delay_range,
            'delay_range-units': 'ns',
            'num_steps': num_steps,
            'num_reps': num_reps,
            'sig_counts': sig_counts.astype(int).tolist(),
            'sig_counts-units': 'counts',
            'ref_counts': ref_counts.astype(int).tolist(),
            'ref_counts-units': 'counts',
            'norm_avg_sig': norm_avg_sig.astype(float).tolist(),
            'norm_avg_sig-units': 'arb'}
    
    file_path = tool_belt.get_file_path(__file__, timestamp, name)
    tool_belt.save_figure(fig, file_path)
    tool_belt.save_raw_data(raw_data, file_path)

def uwave_delay(cxn, nv_sig, nd_filter, apd_indices):
    
    optimize.main(cxn, nv_sig, nd_filter, apd_indices)


# %% Run the file


# The __name__ variable will only be '__main__' if you run this file directly.
# This allows a file's functions, classes, etc to be imported without running
# the script that you set up here.
if __name__ == '__main__':

    # Set up your parameters to be passed to main here
    name = 'johnson1'
    nv_sig = [-0.169, -0.306, 38.74, 40, 2]
    nd_filter = 0.5
    apd_indices = [0]
    num_reps = 10**5
    readout = 2000

    # aom_delay
    delay_range = [900, 1500]
    num_steps = 51
    with labrad.connect() as cxn:
        aom_delay(cxn, nv_sig, nd_filter, readout, apd_indices,
                  delay_range, num_steps, num_reps, name)
