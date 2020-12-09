# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 13:55:53 2020

@author: Aedan
"""

import os
import json
import numpy
import csv
import utils.tool_belt as tool_belt

def convert(folder_name):
    
    folder_items = os.listdir(folder_name)

    for json_file_name in folder_items:
    
        # Only process txt files, which we assume to be json files
        if not json_file_name.endswith('.txt'):
            continue
        
        with open('{}/{}'.format(folder_name, json_file_name)) as json_file:
            # print(json_file_name)
            try:
                data = json.load(json_file)
                img_array = numpy.array(data['img_array'])
                x_voltages = data['x_voltages']
                y_voltages = data['y_voltages']
                x_range = data['x_range']
                y_range=data['y_range']
                num_steps = data['num_steps']
                timestamp = data['timestamp']
                
            except Exception:
                # Skip txt files that are evidently not data files
                print('skipped {} when looking for img file'.format(json_file_name))
                continue

        
    # Populate the data to save
    csv_data = []
    
    for ind in range(num_steps+1):
        row = []
        if ind==0:
            row.append(num_steps)
            row.append(x_range)
        else:
            for i in range(num_steps):
                row.append(img_array[ind-1][i])
        csv_data.append(row)
    csv_file_name = '{}-image'.format(timestamp)

    with open('{}/{}.csv'.format(folder_name, csv_file_name),
              'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',',
                                quoting=csv.QUOTE_NONE)
        csv_writer.writerows(csv_data)

        
                        
if __name__ == '__main__':
    
    top_folder_name = 'E:/Shared drives/Kolkowitz Lab Group/nvdata/image_sample/branch_Spin_to_charge/2020_12'
        
    sub_folder_names = ['to convert'
                        ]
#    
    for el in sub_folder_names:
        
        convert(top_folder_name.format(el))
        
