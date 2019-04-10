# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 22:33:25 2018

@author: Matt
"""


# %% Imports


# User modules
import Utils.tool_belt as tool_belt
import Utils.sweep_utils as sweep_utils

# Library modules
import numpy

# %% Data

xDim = 16
yDim = 12

resolution = .1

offset = [0.0,  0.0]

rawVals = [[1, 2, 8], [5, 2], [1, 1, 3, 10, 5], [5, 4],
           [1, 2, 8], [5, 2], [1, 1, 3, 10, 5], [5, 4],
           [1, 2, 8], [5, 2], [1, 1, 3, 10, 5], [5, 4],
           [1, 2, 8], [5, 2], [1, 1, 3, 10, 5], [5, 4],
           [1, 2, 8], [5, 2], [1, 1, 3, 10, 5], [5, 4],
           [1, 2, 8], [5, 2], [1, 1, 3, 10, 5], [5, 4],
           [1, 2, 8], [5, 2], [1, 1, 3, 10, 5], [5, 4],
           [1, 2, 8], [5, 2], [1, 1, 3, 10, 5], [5, 4],
           [1, 2, 8], [5, 2], [1, 1, 3, 10, 5], [5, 4],
           [1, 2, 8], [5, 2], [1, 1, 3, 10, 5], [5, 4],
           [1, 2, 8], [5, 2], [1, 1, 3, 10, 5], [5, 4],
           [1, 2, 8], [5, 2], [1, 1, 3, 10, 5], [5, 4],
           [1, 2, 8], [5, 2], [1, 1, 3, 10, 5], [5, 4],
           [1, 2, 8], [5, 2], [1, 1, 3, 10, 5], [5, 4],
           [1, 2, 8], [5, 2], [1, 1, 3, 10, 5], [5, 4],
           [1, 2, 8], [5, 2], [1, 1, 3, 10, 5], [5, 4]]

# %% Figure and image setup

imgArray = numpy.empty((yDim, xDim))
imgArray[:] = numpy.nan

minX = offset[0]
maxX = minX + ((xDim - 1) * resolution)
minY = offset[1]
maxY = minY + ((yDim - 1) * resolution)

# For the image extent, we need to bump out the min/max x/y by half the
# resolution in each direction so that the center of each pixel properly
# lies at its x/y voltages.
halfRes = resolution / 2
imageExtent = [maxX + halfRes, minX - halfRes,
               minY - halfRes, maxY + halfRes]

fig = tool_belt.create_image_figure(imgArray, imageExtent)

# %% Update data

writePos = []

for valsToAdd in rawVals:

    valsToAdd = numpy.array(valsToAdd)
    sweep_utils.populate_img_array(valsToAdd, imgArray, writePos)

    tool_belt.update_image_figure(fig, imgArray)
