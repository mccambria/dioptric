# -*- coding: utf-8 -*-
"""
Created on Jan 3 2023

@author: agardill
"""


import utils.tool_belt as tool_belt
import utils.kplotlib as kpl
from utils.kplotlib import KplColors
from numpy import pi
import numpy
import time
import matplotlib.pyplot as plt
from random import shuffle
import labrad
from utils.tool_belt import States
from scipy.optimize import curve_fit
from numpy.linalg import eigvals
import majorroutines.optimize as optimize
from utils.tool_belt import NormStyle

def cosine_fit(x, offset, amp, freq, phase):
    return offset + amp * numpy.cos(x* freq + phase)


def plot(x_data, y_data, title):
    kpl.init_kplotlib()
    
    x_smooth = numpy.linspace(x_data[0], x_data[-1], 1000)
    
    fit_func = lambda x, offset, amp, phase: cosine_fit(x, offset, amp, 1, phase)
    init_params = [ 0.5, 30,  1]
    popt, pcov = curve_fit(
        fit_func,
          x_data,
        y_data,
        # sigma=t2_sq_unc,
        # absolute_sigma=True,
        p0=init_params,
    )
    print(popt)
        

    # Plot setup
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('Phase (rad)')
    ax.set_ylabel("IF voltage (mV)")
    ax.set_title(title)

    # Plotting
    kpl.plot_points(ax,  x_data, y_data, label = 'data', color=KplColors.BLACK)
    
    kpl.plot_line(ax, x_smooth, fit_func(x_smooth,*popt ), label = 'fit', color=KplColors.RED)
    
    ax.legend()
            
    

phases = [0
,0.174532925
,0.34906585
,0.523598776
,0.698131701
,0.872664626
,1.047197551
,1.221730476
,1.396263402
,1.570796327
,1.745329252
,1.919862177
,2.094395102
,2.268928028
,2.443460953
,2.617993878
,2.792526803
,2.967059728
,3.141592654
,3.316125579
,3.490658504
,3.665191429
,3.839724354
,4.01425728
,4.188790205
,4.36332313
,4.537856055
,4.71238898
,4.886921906
,5.061454831
,5.235987756
,5.410520681
,5.585053606
,5.759586532
,5.934119457
,6.108652382
,6.283185307
]
voltages = [135
,120
,90
,63
,35
,5
,-12
,-40
,-63
,-83
,-103
,-115
,-125
,-133
,-136
,-132
,-121
,-110
,-90
,-68
,-45
,-20
,7
,38
,63
,90
,111
,132
,150
,163
,174
,182
,185
,181
,171
,157
,135

]
plot(phases, voltages, '',)


