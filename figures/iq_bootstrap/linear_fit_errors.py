
import utils.tool_belt as tool_belt
import utils.kplotlib as kpl
from utils.kplotlib import KplColors
import utils.positioning as positioning
import numpy
import os
import time
import matplotlib.pyplot as plt
from random import shuffle
from scipy.optimize import curve_fit
import labrad
import majorroutines.optimize as optimize
from utils.tool_belt import NormStyle

from scipy.optimize import fsolve

def solve_linear(m, b, y, z):
   x = z[0]

   F = numpy.empty((1))
   
   F[0] = m*x + b - y
   return F
kpl.init_kplotlib()


def linear_fit_errors(t, vals, errs, title):
    

    ### fit linear line to initial slope
    mid_value = vals[int(len(vals)/2)]
    fit_func = tool_belt.linear
    init_params = [-1, 0]
    
    popt, pcov = curve_fit(fit_func, t, vals,
                        p0=init_params,
                        sigma=errs,
                        absolute_sigma=True)
    
    print(popt)
    
    fig, ax = plt.subplots()
    ax.set_xlabel('Pulse duration (ns)')
    ax.set_ylabel("Error")
    ax.set_title(title)
    
    kpl.plot_points(ax, t, vals, yerr=errs)
                
    smooth_t = numpy.linspace(t[0], t[-1], num=1000)
    kpl.plot_line(
        ax,
        smooth_t,
        fit_func(smooth_t, *popt),
        color=KplColors.RED,
    )
    
    # # find intersection of linear line and offset (half of full range)        
    solve_linear_func = lambda z: solve_linear(popt[0], popt[1], 0, z)
    zGuess = numpy.array(mid_value)
    solve= fsolve(solve_linear_func,zGuess)
    x_intercept =  solve[0]
    print(x_intercept)
    
   
    
t = [91, 101, 111, 121, 131, 141, 151]

# phi_p = [0.2457,0.1089,-0.0586,-0.0668,-0.1942]
# phi_p_unc = [0.0645,0.062,0.0589,0.059,0.0571]
# title = "Error on pi/2 x (-2 Phi')"
# linear_fit_errors(t, phi_p, phi_p_unc, title)

pulse_error_array = numpy.array([-0.4446802061230676, -0.3147691978151431, -0.13482034941215293, 
                                 -0.15161908604862506, 0.12593086584416802, 0.2986200948684772, 0.27383452330953495])
pulse_error_ste_array = numpy.array([0.051662016249883896, 0.05615136893298137,
                                     0.05951554729106733, 0.05481553400025979, 0.06391072024317843, 0.0638959442662516, 0.062438830582250165])

chi_p = pulse_error_array/2 - 0.0#3
chi_p_unc = numpy.sqrt(pulse_error_ste_array**2 + 0.03**2)

title = "Error on pi y (Chi')"
linear_fit_errors(t, chi_p, chi_p_unc, title)
print(chi_p)
print(chi_p_unc)
