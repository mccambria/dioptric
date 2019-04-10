# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 17:41:00 2019

@author: mccambria
"""


import Utils.tool_belt as tool_belt
import numpy
import sys


# %% Parameters


freqRange = (2.4, 3.4)  # GHz
stepSize = 0.01
power = -30.0  # dbm


# %% Initial calculations and setup

# Calculate how many samples we'll take
freqRangeDiff = freqRange[1] - freqRange[0]
totalNumSamples = int(numpy.floor(freqRangeDiff / stepSize) + 1)

# Calculate the frequencies we need to set
freqSteps = numpy.arange(totalNumSamples)
freqs = (stepSize * freqSteps) + freqRange[0]


# %% Get the signal generator


sigGen = tool_belt.get_VISA_instr("ASRL1::INSTR")

# Exit program if no signal generator found 
if sigGen is None:
    sys.exit()


# %% Take a sample and increment the frequency


for ind in range(totalNumSamples):

    sigGen.write("FREQ " + str(freqs[ind]) + " GHz")

    # If this is the first sample then we have to enable the signal
    if ind == 0:
        sigGen.write("AMPL " + str(power))
        sigGen.write("ENBR 1")
    
    val = input("Press enter to advance or enter 'q' to quit...")
    
    if val == "q":
        break

sigGen.write("ENBR 0")
print("Done!")