# -*- coding: utf-8 -*-
"""Template for scripts that should be run directly from the files themselves
(as opposed to from the control panel, for example).

Created on Sun Jun 16 11:22:40 2019

@author: mccambria
"""


# %% Imports


import utils.tool_belt as tool_belt
import numpy
import matplotlib.pyplot as plt


# %% Constants


# %% Functions


# %% Main


def main(data):
    """When you run the file, we'll call into main, which should contain the
    body of the script.
    """

    ref_counts = data['ref_counts']
    num_runs = data['num_runs']
    num_steps = data['num_steps']
    
    x_low = None
    x_high = None
    
    ref_counts = numpy.array(ref_counts)
    std = numpy.std(ref_counts)
    print('standard deviation: {}'.format(std))
    avg = numpy.average(ref_counts)
    print('average: {}'.format(avg))
    print('relative standard deviation: {}'.format(std / avg))
    
    tau_inds = data['tau_index_master_list']
    ref_counts = numpy.array(ref_counts)
    ref_counts_time = numpy.zeros(num_runs * num_steps)
    time_ind = 0
    for run_ind in range(num_runs):
        tau_ind_run = tau_inds[run_ind]
        tau_ind_run = numpy.array(tau_ind_run)
        tau_ind_run = tau_ind_run % num_steps  # Convert negative indices
        tau_ind_run_dedupe = []
        for tau_ind in tau_ind_run:
            if tau_ind not in tau_ind_run_dedupe:
                tau_ind_run_dedupe.append(tau_ind)
        for tau_ind in tau_ind_run_dedupe:
            ref_counts_time[time_ind] = ref_counts[run_ind, tau_ind]
            time_ind += 1
    
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    if (x_low is not None) and (x_high is not None):
        ref_counts_slice = ref_counts_time[x_low: x_high]
    else:
        ref_counts_slice = ref_counts_time

    ax.plot(ref_counts_slice, 'g-', label = 'reference')
    ax.set_xlabel('Time index')
    ax.set_ylabel('Counts')
    ax.legend()
    run_lines = numpy.array(range(num_runs)) * num_steps
    for line in run_lines:
        if (x_low is not None) and (x_high is not None):
            if x_low < line < x_high:
                ax.axvline(line - x_low)
        else:
            ax.axvline(line)
    fig.canvas.draw()
    fig.set_tight_layout(True)
    fig.canvas.flush_events()


# %% Run the file


# The __name__ variable will only be '__main__' if you run this file directly.
# This allows a file's functions, classes, etc to be imported without running
# the script that you set up here.
if __name__ == '__main__':

    # Set up your parameters to be passed to main here
    file_name = '2019-07-30-02_04_25-ayrton12-nv27_2019_07_25' # [0,1]
    data = tool_belt.get_raw_data('t1_double_quantum', file_name)

    # Run the script
    main(data)
