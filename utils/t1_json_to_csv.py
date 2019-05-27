# -*- coding: utf-8 -*-
"""
Point this at a folder and it'll produce csv files containing the taus and
norm_avg_sigs within each json file in the folder. 

Created on Mon May 27 11:26:49 2019

@author: mccambria
"""

import os
import json
import numpy
import csv

folder_name = 'E:/Team Drives/Kolkowitz Lab Group/nvdata/t1_double_quantum/' \
    '2019-04-30-NV2_70MHzSplitting_important_data'
    
folder_items = os.listdir(folder_name)

for json_file_name in folder_items:
    
    # Only process txt files, which we assume to be json files
    if not json_file_name.endswith('.txt'):
        continue
    
    with open('{}/{}'.format(folder_name, json_file_name)) as json_file:
        data = json.load(json_file)
        
        try:
            relaxation_time_range = data['relaxation_time_range']
            norm_avg_sig = data['norm_avg_sig']
            num_steps = data['num_steps']
            init_state = data['init_state']
            read_state = data['read_state']
        except Exception:
            # Skip txt files that are evidently not data files
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
    
    csv_file_name = str(init_state) + '_to_' + str(read_state)
    
    with open('{}/{}.csv'.format(folder_name, csv_file_name),
              'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',',
                                quoting=csv.QUOTE_NONE)
        csv_writer.writerows(csv_data)
    