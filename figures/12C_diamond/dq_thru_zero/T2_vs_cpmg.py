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


def T2_scale_alt(N,n, T2_0): #https://www.nature.com/articles/s41467-019-11776-8
    return (N)**(n)*T2_0

def plot(N, t2_sq, t2_sq_unc, t2_dq, t2_dq_unc, title, do_fit = True):
    kpl.init_kplotlib()
    
    N_lin = numpy.linspace(N[0], N[-1], 100)
    
    if do_fit:
        fit_func = T2_scale_alt
        init_params = [ 0.05, 1.5]
        # popt_sq, pcov_sq = curve_fit(
        #     fit_func,
        #     N,
        #     t2_sq,
        #     sigma=t2_sq_unc,
        #     absolute_sigma=True,
        #     p0=init_params,
        # )
        
        popt_dq, pcov_dq = curve_fit(
            fit_func,
            N,
            t2_dq,
            sigma=t2_dq_unc,
            absolute_sigma=True,
            p0=init_params,
        )
        
        # print('T2_0 = {} +/- {} us'.format(popt[0], numpy.sqrt(pcov[0][0])))
        # print(popt)

    # Plot setup
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('CPMG-N')
    ax.set_ylabel("T2 time (us)")
    ax.set_title(title)

    # Plotting
    # kpl.plot_points(ax, N, t2_sq, yerr=t2_sq_unc, label = 'SQ', color=KplColors.RED)
    kpl.plot_points(ax, N, t2_dq, yerr=t2_dq_unc, label = 'DQ', color=KplColors.GREEN)
    
    if do_fit:
        # kpl.plot_line(ax, N_lin, fit_func(N_lin,*popt_sq ), label = 'SQ fit', color=KplColors.RED)
        kpl.plot_line(ax, N_lin, fit_func(N_lin,*popt_dq ), label = 'DQ fit', color=KplColors.GREEN)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.legend()
            
    

# NV4

N = [1, 2, 4, 8 ,16, 32, 64, 128, 256, 512]

t2_sq = []
t2_sq_unc = []

t2_dq = [0.88, 1.50, 1.74, 1.54, 2.53,2.69, 1.24, 1.79, 1.94, 3.37]
t2_dq_unc = [0.10, 0.16, 0.18, 0.22, 0.3, 0.31, 0.15, 0.3, 0.3, 0.47]

title = 'NV4-2023_01_16'
plot(N, t2_sq, t2_sq_unc, t2_dq, t2_dq_unc, title, do_fit = True)


