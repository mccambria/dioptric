# -*- coding: utf-8 -*-
"""
Created on Mon May  6 13:09:45 2019

Script to run various analysis techniques on the data.

@author: Aedan
"""

# %% Imports


# User modules
#import Utils.tool_belt as tool_belt
import image_sample as image_sample
import t1_measurement as t1_measurement


# %% Analysis Routines

def recreate_image_sample():
    
    # Function specific parameters
    colorMap = 'inferno'
    saveAs = 'png'
    
    # Run the function
    image_sample.recreate_scan_image(colorMap, saveAs)
    
def fit_exponential_decay(fileName):
    
    saveAs = 'png'
    
    t1_measurement.t1_exponential_decay(fileName, saveAs)

# %% Script Code


# Functions only run when called. Since this part of the script is not in a
# function, it will run when the script is run.
# __name__ will only be __main__ if we're running the file as a program.
# The below pattern enables us to import this file as a module without
# running it as a program.
if __name__ == "__main__":


    # %% Functions to run
    
#    recreate_image_sample()
    fit_exponential_decay('2019-05-06_13-50-00_ayrton12')
    
    
    
    