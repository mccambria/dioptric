# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 22:33:25 2018

@author: Matt
"""


# %% Imports


# User modules
import Utils.tool_belt as tool_belt

# Library modules
import numpy
import time

# %% Data

rawVals = [[1000, 2000, 8000],
           [5000, 2000],
           [1000, 1000, 3000, 10000, 5000],
           [5000, 4000]]


# %% Figure and image setup

counts = numpy.empty(12)
counts.fill(numpy.nan)
print(counts)

fig = tool_belt.create_line_plot_figure(counts)

# %% Update data

writePos = 0

for valsToAdd in rawVals:
    
    newWritePos = writePos + len(valsToAdd)

    counts[writePos: newWritePos] = valsToAdd
    
    writePos = newWritePos

    tool_belt.update_line_plot_figure(fig, counts)

    time.sleep(1)
