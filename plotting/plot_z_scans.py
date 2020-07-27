# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 14:53:54 2020

@author: kolkowitz
"""

import utils.tool_belt as tool_belt
import matplotlib.pyplot as plt

path = 'optimize/branch_Spin_to_charge/2020_07'
A_file = '2020_07_13-15_55_02-hopper-ensemble'
B_file = '2020_07_14-14_47_17-hopper-ensemble'
C_file = '2020_07_14-15_55_30-hopper-ensemble'

data = tool_belt.get_raw_data(path, A_file)
z_voltages_A = data['z_voltages']
z_counts_A = data['z_counts']

data = tool_belt.get_raw_data(path, B_file)
z_voltages_B = data['z_voltages']
z_counts_B = data['z_counts']

data = tool_belt.get_raw_data(path, C_file)
z_voltages_C = data['z_voltages']
z_counts_C = data['z_counts']

fig, ax = plt.subplots(1,1, figsize = (10, 8))
ax.plot(z_voltages_A, z_counts_A, label = 'area A 7/12')
ax.plot(z_voltages_B, z_counts_B, label = 'area B 7/14')
ax.plot(z_voltages_C, z_counts_C, label = 'area C 7/14')
ax.legend()