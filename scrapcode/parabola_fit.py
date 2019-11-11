# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 15:25:31 2019
scrap program just to fit a parabola and find a maximum position

@author: gardill
"""
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy


def parabola(x, offset, amplitude, opti_param):
    return offset + amplitude * (x - opti_param)**2 

def fit_parabola(x, y, init_guess_list):
    
    popt,pcov = curve_fit(parabola, x, y,
                          p0=init_guess_list) 
    
    return popt

 # %%

voltages = [5.25, 5.2,5.175, 5.16,5.15,5.125,5.1,5.05,5.0]
counts = [938, 1032, 1064, 1042, 1076, 1054, 1036, 876, 808]
init_guess_list = [1000, 400, 5.15]

popt = fit_parabola(voltages, counts, init_guess_list)

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.plot(voltages, counts, 'ro', label = 'data')
ax.set_xlabel('Piezo voltage z (V)')
ax.set_ylabel('Counts') 
linspace_time = numpy.linspace(voltages[0], voltages[-1], num=1000)
ax.plot(linspace_time, parabola(linspace_time,*popt), 'b-', label = 'fit')

text = ('Optimal piezo voltage = {:.3f} V'.format(popt[2]))

props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
ax.text(0.60, 0.05, text, transform=ax.transAxes, fontsize=12,
                        verticalalignment="top", bbox=props)
ax.legend() 
fig.canvas.draw()
fig.canvas.flush_events()