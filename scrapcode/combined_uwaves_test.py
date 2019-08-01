# -*- coding: utf-8 -*-
"""Template for scripts that should be run directly from the files themselves
(as opposed to from the control panel, for example).

Created on Sun Jun 16 11:22:40 2019

@author: mccambria
"""


# %% Imports


import labrad
import utils.tool_belt as tool_belt


# %% Constants


# %% Functions


# %% Main


def main():
    """When you run the file, we'll call into main, which should contain the
    body of the script.
    """

    with labrad.connect() as cxn:
        main_with_cxn(cxn)
        
def main_with_cxn(cxn):
    
    freq_center = 2.8202
    dev = 0.010
    uwave_power = -5.0
    
    tool_belt.reset_cfm(cxn)
    
    sig_gen_cxn = cxn.signal_generator_tsg4104a
    sig_gen_cxn.set_freq(freq_center - dev)
    sig_gen_cxn.set_amp(uwave_power)
    sig_gen_cxn.uwave_on()
    
#    sig_gen_cxn = cxn.signal_generator_bnc835
#    sig_gen_cxn.set_freq(freq_center + dev)
#    sig_gen_cxn.set_amp(uwave_power)
#    sig_gen_cxn.uwave_on()
    
    cxn.pulse_streamer.constant([4])
    
    input('Press enter to stop...')
    
    cxn.pulse_streamer.constant()
    tool_belt.reset_cfm(cxn)
    

# %% Run the file


# The __name__ variable will only be '__main__' if you run this file directly.
# This allows a file's functions, classes, etc to be imported without running
# the script that you set up here.
if __name__ == '__main__':

    # Set up your parameters to be passed to main here

    # Run the script
    main()
