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

voltages = [ 0.5, 0.25, 0.1, 0.08, 0.06, 0.05]
counts = [ 600, 110, 6.8, 3, 1.1, 0.65]
init_guess_list = [0, 1000, 0]

popt = fit_parabola(voltages, counts, init_guess_list)

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.plot(voltages, counts, 'ro', label = 'data')
ax.set_xlabel('AOM AO power setting')
ax.set_ylabel('Power (uW)') 
linspace_time = numpy.linspace(voltages[-1], voltages[0], num=1000)
ax.plot(linspace_time, parabola(linspace_time,*popt), 'b-', label = 'fit')

text = '\n'.join((r'$C + A_0 (x - x_off)^2$',
                      r'$C = $' + '%.3f'%(popt[0]),
                      r'$A_0 = $' + '%.3f'%(popt[1]),
                      r'$x_{off} = $' + '%.1f'%(popt[2])))

props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
ax.text(0.60, 0.15, text, transform=ax.transAxes, fontsize=12,
                        verticalalignment="top", bbox=props)
ax.legend() 
fig.canvas.draw()
fig.canvas.flush_events()