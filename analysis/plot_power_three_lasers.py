# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 11:38:28 2020

Plot power of three lasers

@author: agardill
"""

import matplotlib.pyplot as plt
import numpy
from scipy.optimize import curve_fit
# %%

def lin(x, m ,b):
    return m*x + b

# %%
    
green_cobalt = [0,1,2,3,4,5,10,20]
green_power_meter = [ 0.0068, 0.0107 , 0.040  ,0.41, 0.69, 0.95, 2.24, 4.7]
green_pd = [-6.6, -2.2, 0, 31, 52, 71, 161, 351]

red_AI = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
red_power_meter = [0, 0, 0, 0, 0, 0, 0, 0.000082,0.023, 0.184, 0.458, 1.02, 7.74, 9.6, 13.1, 15.1,
                   14.1, 7.6]
red_pd = [-2.2,-2.2,-2.2,-2.2,-2.2,-2.2,-2.2,-2.1,1.65, 26.1, 68, 155, 1320, 1670,
          2290, 2640, 2400, 1300] 

red_cobalt = [0, 1, 2,3, 4, 5, 10, 20, 40, 60, 80, 100, 120, 140, 160, 180]
red_cb_power_meter = [0.0088, 0.0114 , 0.015, 0.021, 0.029, 0.036, 0.050, 0.085, 0.52, 0.99, 7.22, 9.66, 12.1, 13.9, 15.4, 13.2]
red_cb_pd = [0, 0.5, 1, 1.9, 3.11, 4.27, 6.26, 11.4, 81, 153, 1230, 1700, 2150, 2460, 2700, 2270]


yellow_AI = [0, 0.01, 0.05, 0.1, 0.12, 0.15, 0.17, 0.2, 0.25, 0.3, 0.35, 0.4, 
             0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
yellow_power_meter = [0, 0, 0.00053, 0.0055 , 0.010, 0.022, 0.033, 0.058, 0.115,
                      0.195, 0.283, 0.380, 0.466, 0.539, 0.6, 0.655, 0.68, 
                      0.71, 0.73, 0.74, 0.75, 0.75, 0.75, 0.75]
yellow_pd = [-3.1, -3.1, -3.1, -2.5, -2.5, -0.4, 0.4, 4.1, 11.9, 22.3, 33.5, 
             45.5, 56.3, 67, 75.5, 80.1, 84.7, 86, 90.2, 91.2, 91, 91, 91, 91]
# %%
init_params = [0, 1/200]

popt_g, arr = curve_fit(lin, green_pd, green_power_meter,
                               p0=init_params)
g_linspace = numpy.linspace(0, green_pd[-1], 1000)

popt_r, arr = curve_fit(lin, red_pd + red_cb_pd, red_power_meter + red_cb_power_meter,
                               p0=init_params)
r_linspace = numpy.linspace(0, 2700, 1000)


popt_y, arr = curve_fit(lin, yellow_pd, yellow_power_meter,
                               p0=init_params)
y_linspace = numpy.linspace(0, yellow_pd[-1], 1000)
#%%
#fig_optical_power, axes_pack = plt.subplots(2, 2, figsize=(11, 11))
#ax = axes_pack[0,0]
#ax.plot(green_cobalt, green_power_meter, 'gs')
#ax.set_title('Green Cobalt (using digital mod. an cobalt application)')
#ax.set_xlabel('Power setting on cobalt App (mW)')
#ax.set_ylabel('Measured optical power (mW)')
#
#
#ax = axes_pack[0,1]
#ax.plot(yellow_AI, yellow_power_meter, 'yo')
#ax.set_title('Yellow LaserGlow (using analog mod. on AOM)')
#ax.set_xlabel('Analog votlage (V)')
#ax.set_ylabel('Measured optical power (mW)')
#
#ax = axes_pack[1,0]
#ax.plot(red_cobalt, red_cb_power_meter, 'rs')
#ax.set_title('Red Cobalt (using digital mod. an cobalt application)')
#ax.set_xlabel('Power setting on cobalt App (mW)')
#ax.set_ylabel('Measured optical power (mW)')
#
#ax = axes_pack[1,1]
#ax.plot(red_AI, red_power_meter, 'ro')
#ax.set_title('Red Cobalt (using analog mod.)')
#ax.set_xlabel('Analog votlage (V)')
#ax.set_ylabel('Measured optical power (mW)')

fig_photodiode, ax = plt.subplots(1,1 , figsize=(11, 11))
#power_meter = green_power_meter + red_power_meter + red_cb_power_meter + yellow_power_meter
#pd = green_pd +red_pd + red_cb_pd +yellow_pd
ax.plot(green_pd, green_power_meter, 'gs', label = 'Green DM')
ax.plot(g_linspace, lin(g_linspace, *popt_g), 'g-', label = 'Green fit')
ax.plot(red_pd, red_power_meter, 'ro', label = 'Red AM')
ax.plot(r_linspace, lin(r_linspace, *popt_r), 'r-', label = 'Re fit')
ax.plot(red_cb_pd, red_cb_power_meter, 'rs', label = 'Red DM')
ax.plot(yellow_pd, yellow_power_meter, 'yo', label = 'Yellow AM')
ax.plot(y_linspace, lin(y_linspace, *popt_y), 'y-', label = 'Yellow fit')
ax.set_xlabel('Photodiode Voltage (mV)')
ax.set_ylabel('Measured optical Power (mW)')
ax.legend()

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
eq_popt = r'y = mx + b'
g_popt = '\n'.join((r'Green',
                      r'm = ' + '%.3f'%(popt_g[0]) + 'mW/mV',
                      r'b  = ' + '%.3f'%(popt_g[1]) + 'mW'))
r_popt = '\n'.join((r'Red',
                      r'm = ' + '%.4f'%(popt_r[0]) + 'mW/mV',
                      r'b  = ' + '%.3f'%(popt_r[1]) + 'mW'))
y_popt = '\n'.join((r'Yellow',
                      r'm = ' + '%.4f'%(popt_y[0]) + 'mW/mV',
                      r'b  = ' + '%.3f'%(popt_y[1]) + 'mW'))


ax.text(0.75, 0.5, eq_popt, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)
ax.text(0.75, 0.45, g_popt, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)
ax.text(0.75, 0.25, r_popt, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)
ax.text(0.75, 0.35, y_popt, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)