# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 13:16:41 2019

basic saturation plot

@author: gardill
"""

import matplotlib.pyplot as plt

polarization_time = [1,3,5,10,15,20,25,30,35,40, 45, 50]
counts = [200,540,730,950,1120,1200,1250,1300,1320,1340, 1360, 1360]

plt.plot(polarization_time, counts)
plt.xlabel('Illumination time (us)')
plt.ylabel('Counts')