# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 14:01:31 2019

@author: Aedan
"""

import os
import utils.tool_belt as tool_belt



path = 'E:/Shared drives/Kolkowitz Lab Group/nvdata/t1_double_quantum'

data_folder = 't1_double_quantum'

rate_list = []
ste_list = []
                
for i in range(3,29):
    folder = 'nv2_2019_04_30_29MHz_{}'.format(i)
    
    folder_dir = 'E:/Shared drives/Kolkowitz Lab Group/nvdata/t1_double_quantum/{}'.format(folder)
    
    file_list = tool_belt.get_file_list(data_folder, '.txt', folder)
    for file in file_list:
        if 'splitting_rate_analysis.txt' in file:
            try:
                data = tool_belt.get_raw_data(data_folder, file[:-4], folder)
                rate = data['gamma']
                ste = data['gamma_ste']
                
                rate_list.append(rate)
                ste_list.append(ste)
            except Exception:
                continue
            
print(rate_list)
print(ste_list)
