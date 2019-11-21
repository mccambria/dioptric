# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 15:24:08 2019

@author: kolkowitz
"""

import numpy
import matplotlib.pyplot as plt

data = [0,0,0,1,1,1,2,2,2,2,2,3,2,3,3,3,4,3,5,4,5,5,6,6,6,7,4,5,6,7,9,9]
data = [el+0.5 for el in data]

hist, bin_edges = numpy.histogram(data, 10, (0, 10))
bin_center_offset = (bin_edges[1] - bin_edges[0]) / 2
bin_centers = bin_edges[0: len(hist)] + bin_center_offset
plt.plot(bin_centers, hist)
