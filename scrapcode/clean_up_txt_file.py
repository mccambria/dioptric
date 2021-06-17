# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 10:02:05 2021

@author: Matt
"""

import numpy
import csv

file = 'C:\\Users\Matt\\Desktop\\oscillations_raw_tags.txt'
timestamps = []

with open(file) as f:
    for i, l in enumerate(f):
        if i == 0:
            first_time = l.lstrip()
            if ',' in first_time:
                first_time = first_time[:first_time.index(',')]
            first_time = numpy.int64(first_time)
        time = l.lstrip()
        if ',' in time:
            time = time[:time.index(',')]
        time = numpy.int64(time)
        time = time-first_time
        timestamps.append(time)
    
with open('C:\\Users\Matt\\Desktop\\oscillations_raw_tags_trim2.txt', 'w', newline='') as f:
     wr = csv.writer(f, quoting=csv.QUOTE_NONE)
     wr.writerow(timestamps)