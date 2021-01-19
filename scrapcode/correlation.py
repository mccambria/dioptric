# -*- coding: utf-8 -*-
"""
Simple working example code using Swabian's Time Tagger to collect and print
out count rates on two channels and their coincidence rate

Created on Mon Jan 18 16:38:35 2021

@author: mccambria
"""


# %% Imports


import TimeTagger
import time


# %% Setup


tagger_serial = '1948000SIP'  # Serial number of your Time Tagger
run_time = 5  # Run time in seconds (this will not be a very precise time)
coincidence_window = 150  # Coincidence window in ns
# Time Tagger hardware channels that your single photon detectors are wired to
hardware_channels = [2, 5]  


# %% Stuff specific to Kolkowitz lab


import labrad

# Turn on the laser
with labrad.connect() as cxn:
    cxn.pulse_streamer.constant([3])
    
    
# %% Main body


# Connect to the Time Tagger and reset it
tagger = TimeTagger.createTimeTagger(tagger_serial)
try:
    tagger.reset()
    
    # Set up a virtual coincidence channel between the hardward channels
    coincidence_window_ps = coincidence_window * 1000
    coincidence_channel = TimeTagger.Coincidence(tagger, hardware_channels, 
                                                 coincidence_window_ps)
    
    # Start a countrate measurement using the two hardware channels and the
    # virtual coincidence channel
    channels = [*hardware_channels, coincidence_channel.getChannel()]
    # channels = hardware_channels
    measurement = TimeTagger.Countrate(tagger, channels)
    
    # When you set up a measurement, it will not start recording data
    # immediately. It takes some time for the tagger to configure the fpga,
    # etc. The sync call waits until this process is complete. 
    tagger.sync()
    
    # Run the measurement for the duration of run_time
    start_time = time.time()
    time_remaining = run_time
    while time_remaining > 0:
        time_remaining = (start_time + run_time) - time.time()
        
    # Record the data and stop the measurement
    count_rates = measurement.getData()
    measurement.stop()
    
    # Print the results
    int_rates = [int(rate) for rate in count_rates]
    print('Hardware channel A: {} counts per second'.format(int_rates[0]))
    print('Hardware channel B: {} counts per second'.format(int_rates[1]))
    print('Coincidences: {:d} coincidences per second'.format(int_rates[2]))

finally:  # Do this even if we crash
    # Release the connection to the Time Tagger
    TimeTagger.freeTimeTagger(tagger)


# %% Stuff specific to Kolkowitz lab


# Turn off the laser
with labrad.connect() as cxn:
    cxn.pulse_streamer.constant([])
