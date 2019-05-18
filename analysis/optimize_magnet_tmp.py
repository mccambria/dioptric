# -*- coding: utf-8 -*-
"""
Created on Fri May 17 15:01:49 2019

@author: kolkowitz
"""
import numpy
import matplotlib.pyplot as plt

minus_one = numpy.array([2.907, 2.919, 2.924, 2.922, 2.913])
plus_one = numpy.array([2.826, 2.815, 2.807, 2.810, 2.817])

splitting = minus_one - plus_one

angles = [20, 40, 60, 80, 100]

fig, ax = plt.subplots(1, 1, figsize=(12, 8.5))

ax.plot(angles, splitting)
ax.set_xlabel('Angle (deg)')
ax.set_ylabel('Splitting (GHz)')

fig.canvas.draw()
fig.canvas.flush_events()

