# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 22:33:25 2018

@author: Matt
"""

import matplotlib.pyplot as plt
import numpy
import matplotlib.animation as animation
import time

# %% Data

xDim = 4
yDim = 3

resolution = .1

offset = [0.0,  0.0]

vals = numpy.array([1, 2, 10, 5,
                    0, 1, 1, 9,
                    10, 10, 5, 4])

# %% Figure and image setup

imgVals = numpy.zeros((yDim, xDim))

left = offset[0]
right = left + (xDim * resolution)
top = offset[1]
bottom = top + (yDim * resolution)

imageExtent = [left, right, bottom, top]
centImageExtent = [x - (resolution / 2) for x in imageExtent]

fig = plt.figure()
img = plt.imshow(imgVals, cmap="gray",
                 extent=tuple(centImageExtent), animated=True)

# %% Update image function


def updateFig(*args):
    global imgVals
    img.set_array(imgVals)
    return img,


# %% Show and set up animation

ani = animation.FuncAnimation(fig, updateFig, interval=100, blit=True)
plt.show()

# %% Update data

for index in range(yDim):
    start = index * xDim
    if index % 2 == 0:
        extension = vals[start: start+xDim]
        imgVals[index, 0: xDim] = extension
    else:
        extension = vals[start: start+xDim]
        extension = extension[::-1]  # Reverse the list
        imgVals[index, 0: xDim] = extension
    print(imgVals)
    time.sleep(1)
