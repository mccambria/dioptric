# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 23:22:03 2022

@author: agard
"""
import numpy
import csv
import utils.tool_belt as tool_belt
threshold = False

folder = 'pc_rabi/branch_master/SPaCE/2021_09'
# file = '2021_09_30-13_18_47-johnson-dnv7_2021_09_23'
# file = '2021_09_09-23_04_14-johnson-dnv0_2021_09_09'
# file = '2021_09_13-08_23_22-johnson-dnv0_2021_09_09'
# file = '2021_09_13-14_03_04-johnson-dnv0_2021_09_09'

# folder = 'pc_rabi/branch_CFMIII/SPaCE_digital/2021_12'
# file = '2021_12_27-11_24_53-johnson-nv0_2021_12_22'


# folder = 'pc_rabi/branch_master/SPaCE/2021_10'
# file = '2021_10_17-19_02_22-johnson-dnv5_2021_09_23'


folder = 'pc_rabi/branch_master/SPaCE/2021_09'
file = '2021_09_06-01_46_43-johnson-nv1_2021_09_03'
threshold = 4

data = tool_belt.get_raw_data(file, folder)
timestamp = data['timestamp']
nv_sig = data['nv_sig']
try:
    num_steps_b = data['num_steps']
except Exception:
    num_steps_b = data['num_steps_b']

readout_counts_avg = numpy.array(data['readout_counts_avg'])

# print(len(readout_counts_avg))

raw_counts = numpy.array(data['readout_counts_array'])
if threshold: 
    for r in range(len(raw_counts)):
        row = raw_counts[r]
        for c in range(len(row)):
            current_val = raw_counts[r][c]
            if current_val < threshold:
                set_val = 0
            elif current_val >= threshold:
                set_val = 1
            raw_counts[r][c] = set_val
    readout_counts_avg = numpy.average(raw_counts, axis = 1)

split_counts = numpy.split(readout_counts_avg, num_steps_b)
readout_image_array = numpy.vstack(split_counts)
r = 0

for i in reversed(range(len(readout_image_array))):
    if r % 2 == 0:
        readout_image_array[i] = list(reversed(readout_image_array[i]))
    r += 1

readout_image_array = numpy.flipud(readout_image_array)


adds = 4
newrow = [0]*num_steps_b
for i in range(adds):
    readout_image_array = numpy.delete(readout_image_array, 0, 0)
    readout_image_array = numpy.vstack([readout_image_array, newrow])

    
for r in range(len(readout_image_array)):
    row = readout_image_array[r].tolist()
    # print(row)
    del row[-1]
    # print(row)
    row = [0] + row 
    # print(row)
    readout_image_array[r] = row
        
# print(len(readout_counts_avg))
csv_data = []
for ind in range(len(readout_image_array)):
    row = readout_image_array[ind]
    # row.append(readout_counts_avg[ind])
    csv_data.append(row)

csv_file_name = file
file_path = "E:\\Shared drives\\Kolkowitz Lab Group\\nvdata\\" + folder

with open('{}/{}.csv'.format(file_path, csv_file_name),
          'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',',
                            quoting=csv.QUOTE_NONE)
    csv_writer.writerows(csv_data)            
