# -*- coding: utf-8 -*-
"""
Created on Mon May  6 13:09:45 2019

Script to run various analysis techniques on the data.

@author: Aedan
"""

# %% Imports


# User modules
#import utils.tool_belt as tool_belt
#import majorroutines.image_sample as image_sample
import analysis.relaxation_rate_plot_data as relaxation_rate_plot_data
import analysis.relaxation_rate_stdev as relaxation_rate_stdev


# %% Analysis Routines

def create_scan_image_in_position_space():
    # This fucntion  takes the data from an image sample and plots it with 
    # position instead of voltage
    
    # Function specific parameters
    colorMap = 'inferno'
    save_file_type = 'png'
    
    # Run the function
    image_sample.reformat_plot(colorMap, save_file_type)
    
    
def t1_analysis(folder_name):
    
    omega = 0.34
    omega_unc = 0.07
    
    relaxation_rate_plot_data.main(folder_name,  omega, omega_unc, 
                                   True, offset = True)
    
    relaxation_rate_stdev.main(folder_name, omega, omega_unc, offset = True)
    

# %% Script Code


# Functions only run when called. Since this part of the script is not in a
# function, it will run when the script is run.
# __name__ will only be __main__ if we're running the file as a program.
# The below pattern enables us to import this file as a module without
# running it as a program.
if __name__ == "__main__":


    # %% Functions to run
    
    #t1 folder
    folder_name = 'nv2_2019_04_30_29MHz_16'
    
    
#    create_scan_image_in_position_space()
#    fit_exponential_decay_from_basic_t1('2019-05-11_19-53-32_ayrton12')
    t1_analysis(folder_name)
    
    
    