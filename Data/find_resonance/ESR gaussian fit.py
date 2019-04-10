# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 10:42:07 2019

This script plots ESR data to a Gaussian fit. It guesses the frequencies of the
resonances, and uses those to curve fit to either one or two gaussians 
(depending on how many resonances it finds)

The protocol to find the resonances is simple and could use improvement.

@author: gardill
"""
# %% Imports

import numpy
import json
from scipy import asarray as ar,exp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# %% Call in file and pass in data

# Call the file and define the file format for saving in the end
fileName = '2019-03-27_17-00-29_Ayrton9'
fileType = ".png"

# Open the file with JSON
with open(fileName + '.txt') as json_file:
    data = json.load(json_file)
    
    # Get information about the frequency
    freqCenter = data["freqCenter"]
    freqRange = data["freqRange"]
    freqResolution = data["freqResolution"]
    
    # Get the counts from the ESR 
    allCounts = data["averageCounts"]

# Pick off just the normalized counts, which is the third array in the data
countsSubAverage = numpy.asarray(allCounts[2])
    
#countsNormAverage = numpy.asarray(allCounts[0])
#countsESRAverage = numpy.asarray(allCounts[1])
#countsSubAverage = countsESRAverage / countsNormAverage

# Calculate some of the necessary value about the frequency scanned 
freqCenterPlusMinus = freqRange / 2
freqStepSize = freqRange / freqResolution
freqMin = freqCenter - freqCenterPlusMinus

# Create an array for the frequency values scanned over. Fill in the first value
# with the minimum freq scanned value
freq = numpy.empty(freqResolution)
freq[0] = freqMin

# populate the frequencies scanned by adding the previous frequency value with
# the scan step size
for i in range(freqResolution - 1):
    freq[i + 1] = freq[i] + freqStepSize
   
# Guess the st dev for the fitting           
sigma = 0.01

# Make emty lists to put the script's guesses for the resonances
minCountsGuess = []
minFreqGuess = []

# We want to step through each value (starting at i+3 and ending at 3 from the 
# end) and see if the three points before and after are larger than it. This 
# should be a decent guess as to a minimum value. There is room for improvement
# Add any guesses for a minimum to a list
#for i in range(freqResolution):
#    if i > 2 and i < freqResolution - 3:
#        if countsSubAverage[i] < 0.95:
#            if countsSubAverage[i-3] > countsSubAverage[i] and countsSubAverage[i-2] > countsSubAverage[i] and countsSubAverage[i-1] > countsSubAverage[i] \
#                and countsSubAverage[i+3] > countsSubAverage[i] and countsSubAverage[i+2] > countsSubAverage[i] and countsSubAverage[i+1] > countsSubAverage[i]:
#                
#                minCountsGuess.append(countsSubAverage[i])
#                minFreqGuess.append(freq[i])

# Now try to fit the data. If the numbr of minimums is not 1 or two, then give
# an error message
#if len(minFreqGuess) > 2 or len(minFreqGuess) == 0:
#    print('Error: more than 2 minimums found')
#
## If there is one minimum found, then fit a single gaussian to it
#elif len(minFreqGuess) == 1:
#    
#    # Define the gaussina function to fit the data to
#    def gaus(x,offset,a,x0,sigma):
#        return offset - a*exp(-(x-x0)**2/(2*sigma**2))
#
#    # Curve fit the data, using the gaussian function previously defined, with x 
#    # values of freq, y values with the counts, and initial guesses on params    
#    popt,pcov = curve_fit(gaus, freq, countsSubAverage, 
#                          p0=[1, 1 - minCountsGuess[0], minFreqGuess[0], sigma])
#
#    # Plot the data itself and the fitted curve
#    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
#    ax.plot(freq, countsSubAverage,'b',label='data')
#    ax.plot(freq,gaus(freq,*popt),'r-',label='fit')
#    ax.set_xlabel('Frequency (GHz)')
#    ax.set_ylabel('Contrast (arb. units)')
##    ax.set_title('ESR (75\N{DEGREE SIGN})')
#    ax.legend()
#    text = "\n".join(("Fluorescent Contrast=" + "%.3f"%(popt[1]),
#                      "Center Frequency=" + "%.3f"%(popt[2]) + " GHz",
#                      "Std Deviation=" + "%.3f"%(popt[3]) + " GHz"))
#    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
#    
#    ax.text(0.05, 0.15, text, transform=ax.transAxes, fontsize=12,
#                            verticalalignment="top", bbox=props)

# If there are two minimums guessed, fit to two gaussians
#elif len(minFreqGuess) == 2:
    
center1 = 2.82
center2 = 2.91

# Define the gaussina function to fit the data to
def gaus(x,offset,a1,center1,sigma1, a2, center2, sigma2):
    return offset - a1*exp(-(x-center1)**2/(2*sigma1**2)) - a2*exp(-(x-center2)**2/(2*sigma2**2))

# Curve fit the data, using the gaussian function previously defined, with x 
# values of freq, y values with the counts, and initial guesses on params
popt,pcov = curve_fit(gaus, freq, countsSubAverage, 
                      p0=[1, 0.1, center1, sigma,
                          0.1, center2, sigma])

    
# Plot the data itself and the fitted curve
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.plot(freq, countsSubAverage,'b',label='data')
ax.plot(freq,gaus(freq,*popt),'r-',label='fit')
ax.set_xlabel('Frequency (GHz)')
ax.set_ylabel('Contrast (arb. units)')
#    ax.set_title('ESR (75\N{DEGREE SIGN})')
ax.legend()
text1 = "\n".join(("Fluorescent Contrast=" + "%.3f"%(popt[1]),
                  "Center Frequency=" + "%.3f"%(popt[2]) + " GHz",
                  "Std Deviation=" + "%.3f"%(popt[3]) + " GHz"))
text2 = "\n".join(("Fluorescent Contrast=" + "%.3f"%(popt[4]),
                  "Center Frequency=" + "%.3f"%(popt[5]) + " GHz",
                  "Std Deviation=" + "%.3f"%(popt[6]) + " GHz"))
props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

ax.text(0.05, 0.15, text1, transform=ax.transAxes, fontsize=12,
                        verticalalignment="top", bbox=props)
ax.text(0.55, 0.15, text2, transform=ax.transAxes, fontsize=12,
                        verticalalignment="top", bbox=props)
            
print(minCountsGuess)
print(minFreqGuess)

fig.canvas.draw()
fig.set_tight_layout(True)
fig.canvas.flush_events()

# Save the file in the same file directory
fig.savefig(fileName + 'replot' + fileType)
