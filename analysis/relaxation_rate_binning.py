# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 09:52:43 2019

This analysis script will take a set of T1 experiments and fit the fucntions
defined in the Myer's paper ((0,0) - (0,1) and (1,1) - (1,-1)) to extract a 
value for Omega nad Gamma. 

Additionally, this fucntion allows the user to pass in a variable to define the
number of bins to seperate the data into. We can split the data up into bins
based on the num_runs. This allows us to see the data does not significantly 
changes over the course of the experiment.

@author: Aedan
"""

# %% Imports

import os
import numpy
import json
from scipy import asarray as ar, exp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import utils.tool_belt as tool_belt

# %% Constants

# Define the directory to get the folders from
directory = 'G:/Shared drives/Kolkowitz Lab Group/nvdata/t1_double_quantum/' 

# %% Functions

# The functions we will fit the data to

def zero_relaxation_eq(t, omega, amp, offset):
    return offset + amp * exp(-3 * omega * t)

    
def plus_relaxation_eq(t, gamma, omega, amp, offset):
    return offset + amp * exp(-(omega + gamma * 2) * t)

# Function to get the file list
    
def get_file_list(folder_name):
    
    # Create a list of all the files in the folder for one experiment
    file_list = []
    for file in os.listdir('{}/{}'.format(directory, folder_name)):
        if file.endswith(".txt") and not file.endswith("bins.txt") \
                                    and not file.endswith("analysis.txt"):
            file_list.append(file)
      
    return file_list

