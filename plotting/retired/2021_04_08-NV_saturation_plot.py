# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 14:29:15 2021

@author: kolkowitz
"""
import matplotlib.pyplot as plt 
import numpy

cts = numpy.array([0, 9, 23, 31, 37, 41, 46, 53, 47, 54, 15, 0.5, 52, 59, 60, 80, 100])
bkgd = numpy.array([0.1, 0.1, 0.5, 2.5, 4, 6, 8, 10, 11, 12, 0.2, 0.2, 14, 16, 20, 37, 55])
pwr = [0.0074, 0.011, 0.09, 0.43, 0.71, 0.99, 1.2, 1.5, 1.75, 2.0, 0.017, 0.089, 2.25, 2.5, 3.25, 6.84, 11.7]

#fig, ax = plt.subplots()
#ax.plot(pwr, cts, 'bo', label = 'Total counts')
#ax.plot(pwr, bkgd, 'ko', label = 'Background counts')
#ax.plot(pwr, cts - bkgd, 'ro', label = 'Adusted NV counts')
#ax.set_xlabel('Power (mW)')
#ax.set_ylabel('kcps')
#ax.legend()

cts_c = numpy.array([0, 0, 0.4, 2.5, 6, 6, 10, 10, 16.6, 15.2, 0.2, 0.1, 19, 22, 32, 95, 220])
cts_1 = numpy.array([0.3, 2, 4.8, 5.9, 12, 11, 16, 17, 21.1, 23, 3.5, 0.5, 29, 36, 45.7, 100, 250])
fig, ax = plt.subplots()
ax.plot(pwr, cts_c, 'bo', label = 'Center counts')
ax.plot(pwr, cts_1, 'ko', label = 'First count')
ax.set_xlabel('Power (mW)')
ax.set_ylabel('kcps')
ax.legend()