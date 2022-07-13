# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 09:19:51 2022

@author: agard
"""

import utils.tool_belt as tool_belt
import numpy
import matplotlib.pyplot as plt
import airy_disk_simulation 
from scipy.optimize import curve_fit
from scipy.special import j1
from scipy.special import jv
import copy

# %%
NA = 1.3
wavelength = 638
fwhm =1.825 # 2* (ln(2))^1/4
scale = 0.99e3


# %%  Data from fitting
y2 = numpy.array([0.10725226150792853, 0.06190012343462058, 0.22799568710400495, 
               0.27458371231064505, 0.3914980260623759, 0.41733672866339744, 
               0.5271959432866865, 0.5145959956581315, 0.5775909397584066, 
               0.5921214987564455, 0.5875214037295355, 0.5903589782122276, 
               0.5499300166778723]) #heights list, in norm. NV population
y2_err = numpy.array([8.34948890712762e-05, 0.00010793648980823788, 
                   0.00013014837012369804, 0.00013073034217673103,
                   9.806252983981712e-05, 7.480836398390166e-05, 
                   8.181131441195924e-05, 5.642847964186914e-05,
                   5.724635703157688e-05, 8.968477049723815e-05, 
                   8.700462838501345e-05, 4.813162265693258e-05, 
                   8.425765285908609e-05])

t = numpy.array([10.0, 11.0, 7.5, 5.0, 2.5, 1.0, 0.75, 0.5,
              0.25, 0.1, 0.075, 0.05, 0.01]) # in ms

lin_x_vals = numpy.linspace((0.01), (11), 100)
# %%

def exp_decay(t, A, alpha, e):
    return A * numpy.exp(-t* (numpy.log(2)/alpha) * e**2)


fit_func = exp_decay
e =  0.0008723132950598539
A = 0.581186150703034
alpha = 0.00000309380937


params = [A, alpha, e]

fig, ax = plt.subplots()
ax.plot(lin_x_vals, fit_func(lin_x_vals, *params), 
            color = 'blue',  linestyle = 'dashed' , linewidth = 1)


ax.errorbar(t, numpy.array(y2),  
            yerr = y2_err,
                fmt='o', color = 'black', 
                linewidth = 1, markersize = 5, mfc='#d6d6d6')

ax.set_xlabel(r'Depletion pulse duration, $\tau$ (ms)')
ax.set_ylabel('Normalized NV pop. height')
# ax.set_xscale('log')
ax.set_yscale('log')
# ax.set_ylim([13.8,208.5])