# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 16:04:37 2021

@author: samli
"""


import json 
import scipy.stats
import scipy.special
import math  
import numpy as np
import matplotlib.pyplot as plt


filename1 = "Data_4_21_100ms.json"
filename2 = "Data_4_21_150ms.json"
filename3 = "Data_4_21_200ms.json"
filename4 = "Data_4_21_250ms.json"
filename5 = "Data_4_21_300ms.json"

#load the data in json format
filename_list = [filename1,filename2,filename3,filename4,filename5]
nv0_array = []
nvm_array = []
for j in range(len(filename_list)): 
    filename = filename_list[j]
    with open(filename) as f:
        data = json.load(f)

    nv0_list = []    
    for i in data['nv0_list']: 
        nv0_list.append(i)
    nv0_array.append(nv0_list)

    nvm_list = []    
    for i in data['nvm_list']: 
        nvm_list.append(i)
    nvm_array.append(nvm_list)
    f.close()


Data_4_21_nv0_array = np.array(nv0_array)
Data_4_21_nvm_array = np.array(nvm_array)


