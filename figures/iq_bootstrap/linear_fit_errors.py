
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
    
    text = "Optimum pulse dur {:.1f} ns".format(x_intercept)
    size = kpl.Size.SMALL
    kpl.anchored_text(ax, text, kpl.Loc.LOWER_LEFT, size=size)
    
   


    
t = [25, 35, 45, 55, 65, 75, 85]
#x
phi_p = [ 0.29817906,  0.14291188,  0.20430906, -0.00972763, -0.10694051,
       -0.36014493, -0.70697674]
phi_p_unc =[ 0.11237101,  0.104745  ,  0.10559904,  0.09732338,  0.08637986,
        0.08234573, -0.0869573 ]
#y
chi_p=[-0.01276276, -0.1684492 , -0.24384949, -0.48762376, -0.56785714,
       -0.59947299, -0.83002129]
chi_p_unc=[ 0.09521654,  0.09032296,  0.08367208,  0.09250574, -0.07837704,
       -0.07292712, -0.08274836]

title = "Error on pi/2 x (Phi')"
linear_fit_errors(t, phi_p, phi_p_unc, title)
title = "Error on pi/2 y (Chi')"
linear_fit_errors(t, chi_p, chi_p_unc, title)


t = [100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170]
phi_array = numpy.array(
[0.639779005524862, 1.0013413816230732, 0.9748953974895396, 0.7576772752652148, 0.3450351053159486, 0.4871220604703237, 0.6033452807646349, 0.3799890650628752, 0.3183716075156584, 0.2590233545647551, 0.03106267029972698, 0.1595935445307819, -0.18507890961262474, -0.35813436979455926, -0.4664310954063613])
phi_unc_array = numpy.array(
[0.1584239238570766, 0.20667107231383558, 0.18705768464797826, 0.16564944938046194, 0.1372831218630389, 0.15718800697517368, 0.1701257578109078, 0.1467125446497577, 0.13979015462377933, 0.1410305773450991, 0.13595573733773125, 0.1563326486618463, 0.11483256024645887, 0.12602174537055272, 0.13436943081546282])

chi_array = numpy.array([0.11767810026385184, 0.15321637426900536, 0.4041013268998792, 0.08492022645393749, 0.23294593228903415, 0.07573964497041441, -0.19020907700153034, -0.1604938271604932, -0.42015321154979435, -0.2773917691074298, -0.3466275659824044, -0.5115789473684202, -0.6657093624353823, -0.5429403202328962, -0.7708880714661049])
chi_unc_array = numpy.array([0.1348080459863919, 0.15040609449778544, 0.16501004751082826, 0.12909564468464466, 0.1322888926783776, 0.14933222199592028, 0.11997495581137042, 0.13488801878294412, 0.13373894228046573, 0.12559360357962923, 0.13396366144179966, 0.12101697155157963, 0.12748899187504323, 0.11001821042085075, 0.11296880241634169])


phi = phi_array/2 - (0.005)
phi_unc = numpy.sqrt(phi_unc_array**2 + (0.03)**2)

chi = chi_array/2 - (0.1)
chi_unc = numpy.sqrt(chi_unc_array**2 + (0.03)**2)

title = "Error on pi_x (Phi)"
# linear_fit_errors(t, phi, phi_unc, title)
#print(phi)
#print(phi_unc)
title = "Error on pi y (Chi)"
# linear_fit_errors(t, chi, chi_unc, title)
#print(chi)
#print(chi_unc)
