# -*- coding: utf-8 -*-
"""
Test for Gaussian fit

Created on Mon Feb 25 20:22:38 2019

@author: mccambria
"""

# User modules
import Utils.tool_belt as tool_belt\

# Library modules
import numpy
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Pack up
centersPack = [None]*3
rangesPack = [None]*3
voltagesPack = [None]*3
countsPack = [None]*3
titlesPack = ("X Axis", "Y Axis", "Z Axis")

for ind in range(3):

	centersPack[ind] = 0.0
	rangesPack[ind] = 1.0

	# Almost Gaussian random data as a histogram
	data = numpy.random.normal(size=1000)
	hist, bin_edges = numpy.histogram(data)
	countsPack[ind] = hist
	voltagesPack[ind] = (bin_edges[:-1] + bin_edges[1:])/2

# Create 3 plots in the figure, one for each axis
fig, axesPack = plt.subplots(1, 3)

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

# Loop over each dimension
for ind in range(3):

	# Unpack for the dimension
	center = centersPack[ind]
	voltageRange = rangesPack[ind]
	voltages = voltagesPack[ind]
	counts = countsPack[ind]
	ax = axesPack[ind]
	title = titlesPack[ind]

	# Guess initial Gaussian fit parameters: coeff, mean, stdev, constY
	initParams = (1000., center, voltageRange/3, 1000.)

	# Least squares
	optiParams, varianceArr = curve_fit(tool_belt.gaussian, voltages,
									    counts, p0=initParams)

	# Plot the fit
	first = voltages[0]
	last = voltages[len(voltages)-1]
	linspaceVoltages = numpy.linspace(first, last, num=1000)
	gaussianFit = tool_belt.gaussian(linspaceVoltages, *optiParams)
	ax.plot(voltages, counts)
	ax.plot(linspaceVoltages, gaussianFit)

	# Add info to the axes
	ax.set_title(title)
	text = "\n".join(("a=" + "%.3f"%(optiParams[0]),
				"$\mu$=" + "%.3f"%(optiParams[1]),
				"$\sigma$=" + "%.3f"%(optiParams[2]),
				"offset=" + "%.3f"%(optiParams[3])))

	props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
	ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=14,
	        verticalalignment="top", bbox=props)
