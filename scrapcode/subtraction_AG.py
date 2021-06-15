# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 09:39:20 2019

subtract data

@author: kolkowitz
"""
import utils.tool_belt as tool_belt
import numpy
import json

folder_name = 'E:/Shared drives/Kolkowitz Lab Group/nvdata/determine_n_thresh/branch_Spin_to_charge/2019_11'
file_on = '2019_11_21-11_07_58-goeppert_mayer_SCC-lifetime.txt'
file_off = '2019_11_21-11_20_42-goeppert_mayer_SCC-lifetime.txt'

with open('{}/{}'.format(folder_name, file_on)) as file:
    data = json.load(file)
    differences = data['differences']
    num_bins = data['num_bins']
