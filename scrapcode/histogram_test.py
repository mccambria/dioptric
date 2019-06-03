# -*- coding: utf-8 -*-
"""
histogram test

Created on Mon Jun  3 16:00:39 2019

@author: mccambria
"""

from matplotlib import pyplot as plt 
import numpy

fig, ax = plt.subplots()
differences = numpy.array([0, 22,87,5,43,56,73,55,54,11,20,51,5,79,31,27, 100]) 
num_bins = 5

hist, bin_edges = numpy.histogram(differences, num_bins)
print(bin_edges)
bin_center_offset = (bin_edges[1] - bin_edges[0]) / 2
bin_centers = bin_edges[0: num_bins] + bin_center_offset
ax.plot(bin_centers, hist)
#ax.plot(bin_edges[0: len(bin_edges)-1], hist)
