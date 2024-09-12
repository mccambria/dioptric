#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 14:51:18 2024

@author: sean
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy

x = []
mag = np.array([])
error = np.array([])
for ii in data:
    x.append(ii[0])
    xfield = np.array(ii[1]['X magnetic field'])
    yfield = np.array(ii[1]['Y magnetic field'])
    zfield = np.array(ii[1]['Z magnetic field'])
    fields = np.sqrt(np.power(xfield,2)+np.power(yfield,2)+np.power(zfield,2))
    
    mag = np.append(mag,np.mean(fields))
    error = np.append(error,np.std(fields)/np.sqrt(len(fields)))

def f(x,a,b):
    return a*x+b
popt, pcov = scipy.optimize.curve_fit(f,x,mag*10,sigma=error*10)
perr = np.sqrt(np.diag(pcov))

print(popt,perr)

xx = np.arange(0,1.5,0.01)

plt.errorbar(x,mag*10,yerr=error*10,fmt='b.',ecolor="r")
plt.plot(xx,popt[0]*xx+popt[1])
plt.title("linear behavior of big coil")
plt.xlabel("current A")
plt.ylabel("field strength G")

