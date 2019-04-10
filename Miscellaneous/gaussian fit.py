# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 19:11:29 2019

@author: Aedan
"""

import numpy

import pylab as plb
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp

def gaussian(x, *params):
	"""
	Calculates the value of a gaussian for the given input and parameters

	Params:
		x: float
			Input value
		params: tuple
			The parameters that define the Gaussian
			1: coefficient that defines the peak height
			2: mean, defines the center of the Gaussian
			3: standard deviation, defines the width of the Gaussian
			4: constant y value to account for background
	"""

	coeff, mean, stdev, offset = params
	var = stdev**2  # variance
	centDist = x-mean  # distance from the center
	return offset + coeff**2 *numpy.exp(-(centDist**2)/(2*var))

xData = [43, 35, 43, 44, 41, 34, 52, 48, 40, 48, 40, 37, 49, 33, 37, 35, 34,
53, 69, 73, 106, 156, 252, 487, 547, 662, 654, 650, 590, 528, 363,
284, 177, 123, 87, 44, 43, 48, 47, 51, 53, 43, 57, 82, 58, 48, 53,
44, 49, 50, 46, 43, 38, 35, 30, 45, 51, 42, 46, 42, 45]

#print(xData)

xLow = 0.029

resolution = 3.27 * 10**(-4)


i = 0

steps = [xLow]

for i in range(1, 61):
    xNextStep = steps[i-1] + resolution
    steps.append(xNextStep)
    
    
#print(steps)    

#optimizationFailed = False
#
#try:
#    optiParams, varianceArr = curve_fit(gaussian, steps,
#                                                    xData, p0=(25., -0.039, 0.001, 40.))
#    
#    print(optiParams)
#    
#except Exception:
#            optimizationFailed = True
#            
#    fig, ax = plt.subplots()
#    ax.plot(steps, xData)
#    ax.set_title('X Plot')
#    
#    first = steps[0]
#    last = steps[len(steps)-1]
#    linspaceVoltages = numpy.linspace(first, last, num=1000)
#    gaussianFit = gaussian(linspaceVoltages, *optiParams)
#    ax.plot(linspaceVoltages, gaussianFit)


#print(data)
#
x = ar(steps)
y = ar(xData)

def gaus(x,const,a,x0,sigma):
    return const + a*exp(-(x-x0)**2/(2*sigma**2))

popt,pcov = curve_fit(gaus,x,y,p0=[70., 500. ,-0.032, 0.005])

plt.plot(x,gaus(x,*popt))
plt.show()






