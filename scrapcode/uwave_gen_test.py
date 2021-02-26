# -*- coding: utf-8 -*-
"""Template for scripts that should be run directly from the files themselves
(as opposed to from the control panel, for example).

Created on Sun Jun 16 11:22:40 2019

@author: mccambria
"""


# %% Imports


import pyvisa as visa
import nidaqmx
import labrad
import time
import numpy
from pulsestreamer import PulseStreamer as Pulser
from pulsestreamer import TriggerStart
from pulsestreamer import OutputState


# %% Constants


# %% Functions


# %% Main


def main(cxn=None):
    """When you run the file, we'll call into main, which should contain the
    body of the script.
    """
    
    instr = cxn.signal_generator_sg394
    # instr = cxn.signal_generator_bnc835
    instr.set_amp(12.0)
    instr.set_freq(2.87)
    instr.uwave_on()
    pulser = cxn.pulse_streamer
    pulser.constant([7])
    pulser.constant([4])
    input('Press enter to stop...')
    pulser.constant([])
    
#    cxn.signal_generator_bnc835.reset()
#            
#    num_steps = 11
#    cxn.signal_generator_bnc835.set_amp(5.0)
##    cxn.signal_generator_bnc835.set_freq(2.85)
#    cxn.signal_generator_bnc835.load_freq_list(numpy.linspace(2.82, 2.92, num_steps))
##    cxn.signal_generator_bnc835.load_freq_list([2.87])
##    cxn.signal_generator_bnc835.load_freq_sweep(2.82, 2.92, num_steps)
#    cxn.signal_generator_bnc835.uwave_on()
#    
#    pulser = Pulser('128.104.160.111')
#    
#    while True:
#        if input('Enter "q" to stop or nothing to continue: ')=='q':
#            break
#        else:
##            pass
#            # Send out a pulse to channel 7 for a tenth of a second
#            pulser.constant(OutputState([7]))
#            time.sleep(0.1)
#            pulser.constant(OutputState([]))
#            # Measure the peak within 200 MHz about 2.87 GHz
#    #        print(cxn.spectrum_analyzer.measure_peak(2.87, 0.2))
    
    # %% Sig gen tests
    
    # resource_manager = visa.ResourceManager()
    # address = 'TCPIP::128.104.160.115::5025::SOCKET'
    # sig_gen = resource_manager.open_resource(address)
    # print(sig_gen)
    # Set the VISA read and write termination. This is specific to the
    # instrument - you can find it in the instrument's programming manual
#    sig_gen.read_termination = '\r\n'
#    sig_gen.write_termination = '\r\n'
    
#    print(sig_gen.query('MODL?'))
#    print(sig_gen.query('FREQ?'))
#    print(sig_gen.query('AMPR?'))
#    print(sig_gen.query('FDEV?'))
#    print(sig_gen.query('ENBR?'))
    
    # print(sig_gen.query('OUTP?'))
    # print(sig_gen.query('FREQ?'))
    # print(sig_gen.query('FREQ:MODE?'))
    # print(sig_gen.query('POW?'))
    # print(sig_gen.query('POW:MODE?'))
    # print(sig_gen.query('AM:STAT?'))
    # print(sig_gen.query('FM:STAT?'))
    
    # sig_gen.close()
    
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
    
    # main()
    
    # Run the script
    with labrad.connect() as cxn:
        try:
            main(cxn)
        finally:
            cxn.signal_generator_sg394.reset()
