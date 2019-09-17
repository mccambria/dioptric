# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 18:46:26 2019

Figure for the supplemental mateirals of magnetically forbidden rates paper
showing all 9 possible measurements

@author: Aedan
"""
import json
import numpy
import matplotlib.pyplot as plt

folder = 'E:/Shared drives/Kolkowitz Lab Group/nvdata/t1_double_quantum/nv1_2019_05_10_28MHz_3'

file_list = [
        '2019-08-23-17_36_12-ayrton12-nv1_2019_05_10.txt', #1,-1 short
        '2019-08-23-23_20_27-ayrton12-nv1_2019_05_10.txt', # 1,-1 long
        '2019-08-24-01_07_22-ayrton12-nv1_2019_05_10.txt', # 1,1
        '2019-08-24-06_51_38-ayrton12-nv1_2019_05_10.txt', # 1,1
        '2019-08-24-08_38_20-ayrton12-nv1_2019_05_10.txt', # 1,0
        '2019-08-24-14_22_27-ayrton12-nv1_2019_05_10.txt', # 1,0
        '2019-08-24-16_58_56-ayrton12-nv1_2019_05_10.txt', #-1,-1
        '2019-08-24-22_43_02-ayrton12-nv1_2019_05_10.txt', #-1,-1
        '2019-08-25-00_29_24-ayrton12-nv1_2019_05_10.txt', # -1,1
        '2019-08-25-06_13_26-ayrton12-nv1_2019_05_10.txt', #-1,1
        '2019-08-25-07_59_35-ayrton12-nv1_2019_05_10.txt', #-1,0
        '2019-08-25-13_43_30-ayrton12-nv1_2019_05_10.txt', #-1,0
        '2019-08-25-16_42_47-ayrton12-nv1_2019_05_10.txt', #0,-1
        '2019-08-25-22_26_53-ayrton12-nv1_2019_05_10.txt', #0,-1
        '2019-08-26-00_13_23-ayrton12-nv1_2019_05_10.txt',#0,1
        '2019-08-26-05_57_27-ayrton12-nv1_2019_05_10.txt', #0,1
        '2019-08-26-07_43_57-ayrton12-nv1_2019_05_10.txt', #0,0
        '2019-08-26-13_27_51-ayrton12-nv1_2019_05_10.txt' # 0,0
        ]


with open('{}/{}'.format(folder, file_list[1])) as file:
    data = json.load(file)
    
    relaxation_time_range = data['relaxation_time_range']
    num_steps = data['num_steps']
    
    taus = numpy.linspace(relaxation_time_range[0],
                          relaxation_time_range[1], num_steps) / 10**6
    norm_avg_sig = ['norm_avg_sig']
    
fig , axes = plt.subplots(1, 1, figsize=(8, 8))
ax.plot(taus,norm_avg_sig)
    
    
    
    