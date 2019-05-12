# -*- coding: utf-8 -*-
"""
Created on Mon May  6 13:09:45 2019

Script to run various analysis techniques on the data.

@author: Aedan
"""

# %% Imports


# User modules
#import Utils.tool_belt as tool_belt
import majorroutines.image_sample as image_sample
import majorroutines.t1_measurement as t1_measurement


# %% Analysis Routines

def create_scan_image_in_position_space():
    # This fucntion  takes the data from an image sample and plots it with 
    # position instead of voltage
    
    # Function specific parameters
    colorMap = 'inferno'
    save_file_type = 'png'
    
    # Run the function
    image_sample.reformat_plot(colorMap, save_file_type)
    
def fit_exponential_decay(open_file_name):
    # This function fits an exponential decay to a t1 measurement
    
    save_file_type = 'png'
    
    t1_measurement.t1_exponential_decay(open_file_name, save_file_type)

# %% Script Code


# Functions only run when called. Since this part of the script is not in a
# function, it will run when the script is run.
# __name__ will only be __main__ if we're running the file as a program.
# The below pattern enables us to import this file as a module without
# running it as a program.
if __name__ == "__main__":


    # %% Functions to run
    
#    recreate_image_sample()
    fit_exponential_decay('2019-05-11_19-53-32_ayrton12')
    
    
    
    