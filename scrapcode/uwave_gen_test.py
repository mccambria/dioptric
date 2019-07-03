# -*- coding: utf-8 -*-
"""Template for scripts that should be run directly from the files themselves
(as opposed to from the control panel, for example).

Created on Sun Jun 16 11:22:40 2019

@author: mccambria
"""


# %% Imports


import visa
import nidaqmx
import labrad


# %% Constants


# %% Functions


# %% Main


def main(cxn):
    """When you run the file, we'll call into main, which should contain the
    body of the script.
    """
    
    cxn.signal_generator_bnc835.set_freq(2.87)
    cxn.signal_generator_bnc835.set_amp(1.0)
    cxn.signal_generator_bnc835.uwave_on()
    
    # %% Sig gen tests
    
#    resource_manager = visa.ResourceManager()
#    sig_gen_address = 'TCPIP::10.128.226.175::inst0::INSTR'
#    sig_gen = resource_manager.open_resource(sig_gen_address)
#    #    sig_gen.read_termination = '\n'
#    #    sig_gen.write_termination = '\n'
#    
#    print(sig_gen.query('FREQ?'))
#    print(sig_gen.query('*IDN?'))
#    #    print(sig_gen.write('FREQ?'))
#    #    print(sig_gen.read_raw())  # Use this to determine the termination
#    print(sig_gen.query('AMPR?'))
#    print(sig_gen.query('FDEV?'))
#    print(sig_gen.query('MODL?'))
    
    # %% DAQ tests
    
#    with nidaqmx.Task() as task:
#        chan_name = 'dev1/_ao3_vs_aognd'
#        task.ai_channels.add_ai_voltage_chan(chan_name,
#                                             min_val=-10.0, max_val=10.0)
#        print(task.read())


# %% Run the file


# The __name__ variable will only be '__main__' if you run this file directly.
# This allows a file's functions, classes, etc to be imported without running
# the script that you set up here.
if __name__ == '__main__':

    # Set up your parameters to be passed to main here

    # Run the script
    with labrad.connect() as cxn:
        try:
            main(cxn)
        finally:
            cxn.signal_generator_bnc835.uwave_off()
