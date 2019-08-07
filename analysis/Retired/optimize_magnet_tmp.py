# -*- coding: utf-8 -*-
"""
Created on Fri May 17 15:01:49 2019

@author: kolkowitz
"""
import numpy
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

minus_one = numpy.array([2.8813, 2.8795, 2.8773, 2.8760, 2.8759, 2.8769, 2.8779,  2.8773, 2.8760, 2.8757, 2.8767])
plus_one = numpy.array([2.8499, 2.8518, 2.8547, 2.8551, 2.8555, 2.8542, 2.8525, 2.8534, 2.8550, 2.8552, 2.8540])

splitting = minus_one - plus_one

angles = [-100, -60, -30, -10, 10, 40, 70, 100, 130, 160, 190]

def AbsCos(angle, offset, amp, phase):
    return offset + abs(amp * numpy.cos(angle * numpy.pi / 180 + phase * numpy.pi / 180))

offset = 1
amp = 0.02
phase = 10

#popt,pcov = curve_fit(AbsCos, angles, splitting, 
#                  p0=[offset, amp, phase])

fig, ax = plt.subplots(1, 1, figsize=(12, 8.5))

ax.plot(angles, splitting,'b', label='data')


#ax.plot(angles, AbsCos(angles,*popt),'r-', label='fit')
ax.set_xlabel('Angle (deg)')
ax.set_ylabel('Splitting (GHz)')
#text1 = "\n".join(("offset=" + "%.3f"%(popt[0]),
#                  "amp=" + "%.4f"%(popt[1]),
#                  "phase=" + "%.4f"%(popt[2])))
#props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
#
#ax.text(0.05, 0.15, text1, transform=ax.transAxes, fontsize=12,
#                        verticalalignment="top", bbox=props)

fig.canvas.draw()
fig.canvas.flush_events()

