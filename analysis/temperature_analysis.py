# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 09:33:29 2020

@author: matth
"""


import csv
import os
import json
import numpy

path = 'E:/Shared drives/Kolkowitz Lab Group/nvdata/papers/bulk_dq_relaxation/temperature/'


def get_voltages(folder_name):
    
    sig_voltages = []
    ref_voltages = []
    # r=root, d=directories, f = file_names
    for r, d, f in os.walk(path + folder_name):
        for file_name in f:
            if not file_name.endswith('.txt'):
                continue
            with open(os.path.join(r, file_name)) as file:
                data = json.load(file)
            temps_list = numpy.array(data['temps_list'])
            # print(data['temps_list'])
            # print(file_name)
            # return
            if len(temps_list) == 0:
                print(file_name)
            sig_voltages.extend(temps_list[:,0])
            ref_voltages.extend(temps_list[:,1])
        
    for voltages in [sig_voltages, ref_voltages]:
        for val in voltages:
            if val > 1.0:
                print(val)
        print()
        
    
def get_temperatures(file_name, lookup_name):
    
    header = True
    resistances = []
    with open(os.path.join(path, file_name), newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            # Create columns from the header (first row)
            if header:
                header = False
                continue
            resistances.append(float(row[2]))
    
    lookup_temps = []
    lookup_resistances = []
    with open(os.path.join(path, lookup_name), newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            # print(row)
            lookup_temps.append(float(row[0]))
            lookup_resistances.append(float(row[1]))
            
    temps = []
    for res in resistances:
        for ind in range(len(lookup_resistances)):
            lookup_res = lookup_resistances[ind]
            if res >= lookup_res:
                # Linearly interpolate
                low_lookup_res = lookup_res
                high_lookup_res = lookup_resistances[ind-1]
                low_lookup_temp = lookup_temps[ind]
                high_lookup_temp = lookup_temps[ind-1]
                norm_diff = (res-low_lookup_res) / (high_lookup_res-low_lookup_res)
                temp_diff = norm_diff * (high_lookup_temp-low_lookup_temp)
                temp = low_lookup_temp + temp_diff
                temps.append(temp)
                break
                
    for temp in temps:
        print(temp)
    
    
        
    
if __name__ == '__main__':
    
    # get_voltages('t1_double_quantum_overnight')
    get_voltages('t1_double_quantum_week')
    # get_temperatures('temperature.csv', 'thermistor_lookup.csv')
    