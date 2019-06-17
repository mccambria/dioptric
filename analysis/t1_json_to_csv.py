# -*- coding: utf-8 -*-
"""
Specify folders and run to produce csv files containing the taus and
norm_avg_sigs within each json-formatted text file in the folder.

The csv files are named as:
<init state>_to_<read state>_<max relaxation time in us>.csv

Created on Mon May 27 11:26:49 2019

@author: mccambria
"""

import os
import json
import numpy
import csv

def convert(folder_name):
    
    folder_items = os.listdir(folder_name)

    for json_file_name in folder_items:
    
        # Only process txt files, which we assume to be json files
        if not json_file_name.endswith('.txt'):
            continue
    
        with open('{}/{}'.format(folder_name, json_file_name)) as json_file:
    
            try:
                data = json.load(json_file)
                relaxation_time_range = data['relaxation_time_range']
                norm_avg_sig = data['norm_avg_sig']
                num_steps = data['num_steps']
                init_state = data['init_state']
                read_state = data['read_state']
            except Exception:
                # Skip txt files that are evidently not data files
                print('skipped {}'.format(json_file_name))
                continue
    
        # Calculate the taus
        taus = numpy.linspace(relaxation_time_range[0], relaxation_time_range[1],
                              num=num_steps, dtype=numpy.int32)
    
        # Populate the data to save
        csv_data = []
        for tau_ind in range(len(taus)):
            row = []
            row.append(taus[tau_ind])
            row.append(norm_avg_sig[tau_ind])
            csv_data.append(row)
    
        max_relaxation_us = relaxation_time_range[1] // 1000
    
        csv_file_name = '{}_to_{}_{}'.format(init_state, read_state,
                         max_relaxation_us)
    
        with open('{}/{}.csv'.format(folder_name, csv_file_name),
                  'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',',
                                    quoting=csv.QUOTE_NONE)
            csv_writer.writerows(csv_data)
            
if __name__ == '__main__':
    
    top_folder_name = 'E:/Shared drives/Kolkowitz Lab Group/nvdata/' \
        't1_double_quantum/{}'
        
#    sub_folder_names = ['nv0_2019_06_06 _48MHz',
#                        'nv1_2019_05_10_20MHz',
#                        'nv1_2019_05_10_28MHz',
#                        'nv1_2019_05_10_30MHz',
#                        'nv1_2019_05_10_32MHz',
#                        'nv1_2019_05_10_52MHz',
#                        'nv1_2019_05_10_98MHz',
#                        'nv1_2019_05_10_116MHz',
#                        'nv2_2019_04_30_29MHz',
#                        'nv2_2019_04_30_45MHz',
#                        'nv2_2019_04_30_56MHz',
#                        'nv2_2019_04_30_57MHz',
#                        'nv2_2019_04_30_70MHz',
#                        'nv2_2019_04_30_85MHz',
#                        'nv2_2019_04_30_101MHz',
#                        'nv4_2019_06_06_28MHz']
#    
#    for el in sub_folder_names:
#        
#        convert(top_folder_name.format(el))
        
    convert(top_folder_name.format('nv13_2019_06_10_72MHz'))
