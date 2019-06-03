# -*- coding: utf-8 -*-
"""
Will allow tests of rise/fall time of AOM. The rise/fall time can be seen
on an oscilloscope connected to a photodiode.

We first want to test the rise/fall time of the AOM driver. test_aom determines
if the AOM driver will be used to d othe switching or a switch

Created on Mon Jun 3 11:49:23 2019

@author: gardill
"""


# %% Imports

import os
import labrad

# %% Main


def main(cxn, aom_on_time, aom_off_time, test_aom_driver = True):

    # %% Initial set up
    
    # Define some times (in ns)
    aom_on_time = int(aom_on_time)
    aom_off_time = int(aom_off_time)        
    
    # %% Run the sequence
    
    file_name = os.path.basename(__file__)    
    args = [aom_on_time, aom_off_time, test_aom_driver]
    
    cxn.pulse_streamer.stream_immediate(file_name, args, 0)
    
    # %%
    
if __name__ == '__main__':
    
    with labrad.connect() as cxn:
        main(1000,1000, test_aom_driver = True)
    
            
