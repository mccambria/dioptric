# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 16:50:29 2022

@author: carter fox

this is to analyze ramsey data with one run at one free precession time with many reps, where we will bin the data temporally, 
looking for nuclear spin flips

"""

import matplotlib.pylab as plt
import numpy as np
import utils.tool_belt as tool_belt


folder = "pc_carr/branch_opx-setup/ramsey/2022_11"
file = '2022_11_11-11_59_10-johnson-search'

# detuning = 0
data = tool_belt.get_raw_data(file, folder)

sig_counts = data['sig_counts']

period = 0 #get this from simuling the program

plt.plot(sig_counts)
