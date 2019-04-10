# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 10:42:07 2019

This script plots ESR data to a Gaussian. It only works for a single peak currently.

Just put in the file name you want to plot.

@author: gardill
"""
import Utils.tool_belt as tool_belt

import numpy
import json
from scipy import asarray as ar,exp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

fileName = '2019-03-21_13-23-46_Ayrton9'
fileType = ".png"

with open(fileName + '.txt') as json_file:
    data = json.load(json_file)
    
    freqCenter = data["freqCenter"]
    freqRange = data["freqRange"]
    freqResolution = data["freqResolution"]
    
    allCounts = data["averageCounts"]

countsSubAverage = numpy.asarray(allCounts[2])

freqCenterPlusMinus = freqRange / 2
freqStepSize = freqRange / freqResolution
freqMin = freqCenter - freqCenterPlusMinus

freq = numpy.empty(freqResolution)
freq[0] = freqMin


for i in range(freqResolution - 1):
    freq[i + 1] = freq[i] + freqStepSize
             
mean = 2.875    
sigma = numpy.sqrt(sum(countsSubAverage * (freq - mean)**2) / sum(countsSubAverage))               
#sigma = sum(countsSubAverage*(freq-mean)**2)/freqResolution

def gaus(x,offset,a,x0,sigma):
    return offset - a*exp(-(x-x0)**2/(2*sigma**2))

popt,pcov = curve_fit(gaus, freq, countsSubAverage ,p0=[1,0.1,mean,sigma])

#print(popt)

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.plot(freq, countsSubAverage,'b',label='data')
ax.plot(freq,gaus(freq,*popt),'r-',label='fit')
ax.set_xlabel('Frequency (GHz)')
ax.set_ylabel('Contrast (arb. units)')
ax.legend()
text = "\n".join(("Fluorescent Contrast=" + "%.3f"%(popt[1]),
                  "Center Frequency=" + "%.3f"%(popt[2]) + " GHz",
                  "Std Deviation=" + "%.3f"%(popt[3]) + " GHz"))
props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

ax.text(0.05, 0.15, text, transform=ax.transAxes, fontsize=12,
                        verticalalignment="top", bbox=props)

fig.canvas.draw()
fig.set_tight_layout(True)
fig.canvas.flush_events()

# Save the file in the same file directory
fig.savefig(fileName + 'replot.' + fileType)
