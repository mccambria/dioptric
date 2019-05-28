# -*- coding: utf-8 -*-
"""
Created on Wed May  8 09:25:44 2019

@author: kolkowitz
"""

import json
import numpy
import matplotlib.pyplot as plt
import time

import utils.tool_belt as tool_belt

def replot_t1(open_file_name, save_file_type):
    
    directory = 'E:/Team Drives/Kolkowitz Lab Group/nvdata/t1_measurement/'
   
    # Open the specified file
    with open(directory + open_file_name + '.txt') as json_file:
        data = json.load(json_file)
        sig_counts = numpy.array(data["sig_counts"])
        ref_counts = numpy.array(data["ref_counts"])
        relaxation_time_range = data["relaxation_time_range"]
        num_steps = data["num_steps"]
        spin = data["spin_measured?"]
        name = data["name"]
        num_runs = data["num_runs"]

    
    list_to_delete = (1,2,3,4,5,6,7,8,9)
    
    sig_counts_to_keep = numpy.delete(sig_counts, list_to_delete, axis = 0) 
    ref_counts_to_keep = numpy.delete(ref_counts, list_to_delete, axis = 0)
     
    
    avg_sig_counts_to_keep = numpy.average(sig_counts_to_keep, axis=0)
    avg_ref_counts_to_keep = numpy.average(ref_counts_to_keep, axis=0)
    
    norm_avg_sig_to_keep = avg_sig_counts_to_keep / avg_ref_counts_to_keep

#    print(norm_avg_sig_to_keep)
    
    min_relaxation_time = relaxation_time_range[0] 
    max_relaxation_time = relaxation_time_range[1]
        
    taus = numpy.linspace(min_relaxation_time, max_relaxation_time,
                              num=num_steps, dtype=numpy.int32)
        
    
    fig, axes_pack = plt.subplots(1, 2, figsize=(17, 8.5))
    
    ax = axes_pack[0]
    ax.plot(taus / 10**6, avg_sig_counts_to_keep, 'r-', label = 'signal')
    ax.plot(taus / 10**6, avg_ref_counts_to_keep, 'g-', label = 'reference')
    ax.set_xlabel('Relaxation time (ms)')
    ax.set_ylabel('Counts')
    ax.legend()
    
    ax = axes_pack[1]
    ax.plot(taus / 10**6, norm_avg_sig_to_keep, 'b-')
    ax.set_title('T1 Measurement of ' + spin)
    ax.set_xlabel('Relaxation time (ms)')
    ax.set_ylabel('Contrast (arb. units)')

    fig.canvas.draw()
    # fig.set_tight_layout(True)
    fig.canvas.flush_events()
    
    timestamp = tool_belt.get_time_stamp()
    
    raw_data = {'timestamp': timestamp,
            'original_file': '2019-05-08_07-11-25_ayrton12',
            'spin_measured?': spin,
            'relaxation_time_array': taus.tolist(),
            'relaxation_time_array_units': 'ns',
            'num_runs': num_runs,
            'deleted_runs': list_to_delete,
            'avg_sig_counts_to_keep': avg_sig_counts_to_keep.astype(int).tolist(),
            'sig_counts-units': 'counts',
            'avg_ref_counts_to_keep': avg_ref_counts_to_keep.astype(int).tolist(),
            'ref_counts-units': 'counts',
            'norm_avg_sig_to_keep': norm_avg_sig_to_keep.astype(float).tolist(),
            'norm_avg_sig-units': 'arb'}
    
#    file_path = tool_belt.get_file_path('t1_measurement_single', timestamp, name + 'replot')
#    tool_belt.save_figure(fig, file_path)
#    tool_belt.save_raw_data(raw_data, file_path)
        
        
        
        
if __name__ == '__main__':
    
    replot_t1('2019-05-09_15-04-22_ayrton12', 'svg')