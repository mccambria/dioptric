# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 13:01:00 2019

Finding the most probable value for omega based on it's std dev

https://math.stackexchange.com/questions/2101139/weighted-average-from-errors

@author: Aedan
"""
import numpy

# The data
nv2_omega_avg = [0.37, 0.52, 0.33, 0.41, 0.33, 0.32, 0.27, 0.34, 0.25, 0.35, 0.30, 0.34]
nv2_omega_error = [0.06, 0.11, 0.07, 0.06, 0.06, 0.04, 0.04, 0.02, 0.03, 0.02, 0.03, 0.07]

num = []
den = []

for i in range(len(nv2_omega_avg)):
    D = nv2_omega_avg[i]
    sigma = nv2_omega_error[i]
    
    num.append(D/sigma**2)
    den.append(1/sigma**2)
    
unc = numpy.array(nv2_omega_error)
uncertainty = 1/3 * numpy.sqrt( numpy.sum(unc**2) )

Y = sum(num)/sum(den) 

avg = numpy.average(nv2_omega_avg)


print(Y)
print(avg)
print(uncertainty)