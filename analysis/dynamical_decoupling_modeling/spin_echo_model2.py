# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 12:42:32 2022

@author: kolkowitz
"""

import numpy
import utils.tool_belt as tool_belt
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import quad
from numpy import pi

def X_SE(t, fL ):
    sigma= 0.1*fL
    lambd = 0.5
    a0 = 1
    a1 = -2/2
    a2 = (1/2)/2
    a_list = [a1, a2]
    a_sum = a0
    
    for i in range(len(a_list)):
        n = i+1
        a_sum += a_list[i]*numpy.exp(n**2 * t**2 * sigma**2 / 8) * numpy.cos(n*t*fL*2*pi)

    return lambd**2 * a_sum

def F_SE(f, t):
    return 8*numpy.sin(f*2*pi*t/2)**4

def F_FID(f, t):
    return 2*numpy.sin(f*2*pi*t/2)**2

def SS(f ,t, fL):
    sigma = 1 * f #??? fL
    lamd = 0.25
    # fL = 0.10
    return numpy.sqrt(2*pi)/sigma * lamd**2 * (f*2*pi)**2 * numpy.exp(-(f*2*pi - fL*2*pi )**2 / (2*sigma**2))

def integrand(f, t, fL):
    
    return F_SE(f, t) * SS(f ,t, fL) / (2 * pi * (f*2*pi)**2)

# def S_bath_SE(t, fL):
#     I = quad(integrand, -numpy.inf, numpy.inf, args=(t, fL))
    
#     return numpy.exp(-I[0])
def S_bath_SE(t, fL):
    # I = quad(integrand, -numpy.inf, numpy.inf, args=(t, fL))
    
    return numpy.exp(-X_SE(t, fL ))


fig, ax = plt.subplots()
f_lin = numpy.linspace(0, 2,101)
fit_func = S_bath_SE
init_params=[1e5]


t0 = 0
tf = 40 # us
t_lin = numpy.linspace(t0, tf, 101)
f_log = numpy.logspace(1, 7, 1001)
d_ts = numpy.linspace(t0, tf, 101)
fL =0.1 #MHz
for index in range(t_lin.size):
        d_ts[index] = S_bath_SE(t_lin[index], fL)
# print(d_ts)
ax.plot(
        t_lin,
        S_bath_SE(t_lin, fL),
        "-",
        color="black",
        # label="Spin bath",
    ) 
# ax.set_xscale("log")
ax.set_xlabel(r"t (us)")
ax.set_ylabel(r"Signal")
# ax.set_title('Spin Echo')
# ax.legend()

# ax.plot(
#         f_log,
#         SS(f_log, 1e-6, 1e5),
#         # F_FID(f_log, 5e-6)/f_log**2,
#         # F_FID(t_lin, 5e-6)/t_lin**2,
#         "-",
#         color="blue",
#         label="Integrand",
#     ) 

# ax.plot(
#         f_log,
#         F_FID(f_log, 5e-6)/f_log**2,
#         # F_FID(t_lin, 5e-6)/t_lin**2,
#         "-",
#         color="blue",
#         label="Ramsey",
#     ) 
# ax.plot(
#         f_log,
#         F_SE(f_log, 5e-6)/f_log**2,
#         # F_FID(t_lin, 5e-6)/t_lin**2,
#         "-",
#         color="red",
#         label="Spin Echo",
#         ) 
# ax.set_xscale("log")
# ax.set_xlabel(r"Frequency, $\omega/2\pi$ (Hz)")
# ax.set_ylabel(r"F($\omega, \tau = 5$ us)/$\omega^2$")
# ax.set_title('Spin Echo')
# ax.legend()
