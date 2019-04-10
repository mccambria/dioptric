# -*- coding: utf-8 -*-
"""
Messin with line plots

Created on Sun Feb 10 16:45:26 2019

@author: mccambria
"""


# %% Imports


import numpy
import Utils.tool_belt as tool_belt
import time


# %% Input


vals = [10, 4.5, 7.3, 5.5, 9.4, 9.5, 9.5, 7.3, 6.0, 5.9]
vals = numpy.array(vals)

freqRange = (2.80, 2.98)
resolution = 0.02


# %% Plotting


# Calculate how many samples we'll take
freqRangeDiff = freqRange[1] - freqRange[0]
totalNumSamples = int(numpy.floor(freqRangeDiff / resolution) + 1)

allSamplesDiff = numpy.empty(totalNumSamples)
allSamplesDiff[:] = numpy.nan

# Calculate the frequencies we need to set
freqSteps = numpy.arange(totalNumSamples)
freqs = (resolution * freqSteps) + freqRange[0]

fig = tool_belt.create_line_plot_figure(allSamplesDiff, freqs)

for ind in range(totalNumSamples):

    time.sleep(1)

    allSamplesDiff[ind] = vals[ind]

    tool_belt.update_line_plot_figure(fig, allSamplesDiff)
