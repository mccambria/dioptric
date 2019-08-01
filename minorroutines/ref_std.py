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


def st_devs(data):
    ref_counts = data['ref_counts']
    num_runs = data['num_runs']
    num_steps = data['num_steps']
    ref_counts_trans = numpy.transpose(ref_counts)
    print('Num cols: {}'.format(len(ref_counts_trans)))
    # for col in ref_counts_trans:
    #     avg_col = numpy.average(col)
    #     print('avg: {}'.format(avg_col))
    #     print('expected st dev: {}'.format(numpy.sqrt(avg_col)))
    #     std_col = numpy.std(col)
    #     print('st dev: {}'.format(std_col))
        # print('unadjusted st dev: {}'.format(std_col))
        # # Adjust for binning
        # print('adjusted st dev: {}'.format(std_col*numpy.sqrt(num_runs)))
    summed = numpy.sum(ref_counts, axis=0)
    print(summed)
    print('expected st dev: {}'.format(numpy.sqrt(numpy.average(summed))))
    print('st dev: {}'.format(numpy.std(summed)*(num_steps/(num_steps-1))))
    # print(numpy.std(data['norm_avg_sig']))


def expected_st_dev_norm(ref_counts, expected_contrast):
    sig_counts = expected_contrast * ref_counts
    rel_std_sig = numpy.sqrt(sig_counts) / sig_counts
    rel_std_ref = numpy.sqrt(ref_counts) / ref_counts
    # Propogate the error
    print(expected_contrast * numpy.sqrt((rel_std_sig**2) + (rel_std_ref**2)))



# %% Main


def main(data):
    """When you run the file, we'll call into main, which should contain the
    body of the script.
    """

    # print(numpy.std(data['norm_avg_sig']))
    # return

    ref_counts = data['ref_counts']
    num_runs = data['num_runs']
    num_steps = data['num_steps']
    
    x_low = None
    x_high = None
    
    ref_counts = numpy.array(ref_counts)
    avg = numpy.average(ref_counts)
    print('average: {}'.format(avg))
    print('expected standard deviation: {}'.format(numpy.sqrt(avg)))
    std = numpy.std(ref_counts)
    print('standard deviation: {}'.format(std))
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
    # file_name = '2019-07-31-15_41_43-ayrton12-nv27_2019_07_25'
    # file_name = '2019-08-01-09_30_37-ayrton12-nv27_2019_07_25'
    file_name = '2019-07-29-23_05_39-ayrton12-nv27_2019_07_25'
    data = tool_belt.get_raw_data('t1_double_quantum', file_name)

    exp_count_rate = 25 # kcps
    readout_window = 450 # ns
    num_reps = 8 * 10 ** 4
    num_runs = 20
    
    counts = (exp_count_rate * 10 ** 3) * (readout_window * 10**-9) * num_reps * num_runs 
    # Run the script
    # main(data)
    # st_devs(data)
    expected_st_dev_norm(counts, 0.95)
