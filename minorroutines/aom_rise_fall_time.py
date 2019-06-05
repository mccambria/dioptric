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
import utils.tool_belt as tool_belt

# %% Main


def main(cxn, aom_on_time, aom_off_time, test_aom_driver = True):

    # %% Initial set up
    
    # Define some times (in ns)
    aom_on_time = int(aom_on_time)
    aom_off_time = int(aom_off_time)        
    
    # %% Run the sequence
    
    file_name = os.path.basename(__file__)    
    args = [aom_on_time, aom_off_time, test_aom_driver]
    
    cxn.pulse_streamer.stream_immediate(file_name, 1 * 10**8, args, 1)
#    cxn.pulse_streamer.stream_immediate(file_name, 3, args, 1)
    
    # %%
    
if __name__ == '__main__':
    try:
        
        with labrad.connect() as cxn:
            main(cxn, 100, 100, False)
        
    finally:
        # Kill safe stop
        if tool_belt.check_safe_stop_alive():
            print("\n\nRoutine complete. Press enter to exit.")
            tool_belt.poll_safe_stop()
            
