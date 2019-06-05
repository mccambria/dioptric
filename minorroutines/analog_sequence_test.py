# -*- coding: utf-8 -*-
"""
This is a test to create a pulse sequence using analog outputs of the pulse 
streamer

Created on Wed Jun 5 14:49:23 2019

@author: gardill
"""


# %% Imports

import os
import labrad

# %% Main


def main(cxn, aom_on_time, aom_off_time, low_voltage, high_voltage):

    # %% Initial set up
    
    # Define some times (in ns)
    aom_on_time = int(aom_on_time)
    aom_off_time = int(aom_off_time)        
    
    # %% Run the sequence
    
    file_name = os.path.basename(__file__)    
    args = [aom_on_time, aom_off_time, low_voltage, high_voltage]
    
    cxn.pulse_streamer.stream_immediate(file_name, args, 0)
    
    # %%
    
if __name__ == '__main__':
    
    with labrad.connect() as cxn:
        main(1000, 1000, 0, 1)
    
            
