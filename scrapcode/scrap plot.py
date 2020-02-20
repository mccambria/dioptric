# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 15:44:49 2020

@author: kolkowitz
"""

import numpy
import utils.tool_belt as tool_belt
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

data_path = 'E:/Shared drives/Kolkowitz Lab Group/nvdata'

folder = 'lifetime_v2/2020_02'

file_1 = '2020_02_19-14_36_09-graphene_Y2O3'
data = tool_belt.get_raw_data(folder, file_1)

counts_1 = numpy.array(data["binned_samples"])
bin_centers_1 = numpy.array(data["bin_centers"])/10**3
background_1 = numpy.average(counts_1[75:150])

file_2 = '2020_02_19-19_34_23-graphene_Y2O3'
data = tool_belt.get_raw_data(folder, file_2)

counts_2 = numpy.array(data["binned_samples"])
bin_centers_2 = numpy.array(data["bin_centers"])/10**3
background_2 = numpy.average(counts_2[75:150])

fig, ax = plt.subplots(1,1,figsize=(10, 8))

ax.plot(bin_centers_1, counts_1-background_1, label = '0V_1')
ax.plot(bin_centers_2, counts_2-background_2, label = '0V_2')
ax.legend()