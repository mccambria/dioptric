
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
    
   


    
t = [59-10, 59-5, 59, 59+5, 59+10]

phi_p = [0.12426383981154315, 0.0835897435897438, 0.1919999999999999, 0.15593561368209258, 0.12241887905604742]
phi_p_unc = [0.10151806948931795, 0.08837157863240727, 0.08021342629332615, 0.08240040808863484, 0.08315089188815634]
chi_p = [0.1597222222222215, 0.3275862068965518, 0.2629310344827586, 0.39648241206030166, 0.18432203389830515]
chi_p_unc = [0.0926832226312471, 0.09074993935776376, 0.11479614168671791, 0.07881849323635998, 0.08741664746710423]
title = "Error on pi/2 x (Phi')"
linear_fit_errors(t, phi_p, phi_p_unc, title)
title = "Error on pi/2 y (Chi')"
linear_fit_errors(t, chi_p, chi_p_unc, title)


t = [59-10, 59-5, 59, 59+5, 59+10]
# phi_array = numpy.array([-0.4446802061230676, -0.3147691978151431, -0.13482034941215293, 
#                                  -0.15161908604862506, 0.12593086584416802, 0.2986200948684772, 0.27383452330953495])
# phi_unc_array = numpy.array([0.051662016249883896, 0.05615136893298137,
#                                      0.05951554729106733, 0.05481553400025979, 0.06391072024317843, 0.0638959442662516, 0.062438830582250165])

# chi_array = numpy.array([-0.4446802061230676, -0.3147691978151431, -0.13482034941215293, 
#                                  -0.15161908604862506, 0.12593086584416802, 0.2986200948684772, 0.27383452330953495])
# chi_unc_array = numpy.array([0.051662016249883896, 0.05615136893298137,
#                                      0.05951554729106733, 0.05481553400025979, 0.06391072024317843, 0.0638959442662516, 0.062438830582250165])

# phi = phi_array/2 - (0.1)
# phi_unc = numpy.sqrt(phi_unc_array**2 + (0.1)**2)

# chi = chi_array/2 - (0.1)
# chi_unc = numpy.sqrt(chi_unc_array**2 + (0.1)**2)

# title = "Error on pi y (Chi')"
# linear_fit_errors(t, chi, chi_unc, title)
# print(chi)
# print(chi_unc)
