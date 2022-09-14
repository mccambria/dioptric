# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 13:19:47 2022

@author: kolkowitz
"""

import numpy
import utils.tool_belt as tool_belt
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numpy import pi

uN = 0.762e-3 #MHz/G
uN_13C = 0.7*uN

Bz= 94.6 #G
# Ax = 0.100 #MHz
# Ay = 0 #MHz
# Az = -0.075 #MHz

B_vec = numpy.array([0,0,Bz])


def S_1(t, a, b, f0, T):
    term_ss = numpy.sin(f0*t*2*pi/2)**2 
    return (a - b* term_ss)*numpy.exp((-t/T)**3)

def S_2(t, a, b, f0, f1, T):
    term_ss = numpy.sin(f0*t*2*pi/2)**2 * numpy.sin(f1*t*2*pi/2)**2
    return (a - b* term_ss)*numpy.exp((-t/T)**3)

def X_SE(t, fL, lambd, sigma_per ):
    a_list = [6, -8, 2]
    
    sum_expr = a_list[0]
    # lambd = 0.25
    # sigma = 0.1 * fL #/ (2*pi)
    for i in range(len(a_list)-1):
        n=i+1
        # print(n)
        sum_expr += a_list[n]*numpy.exp(-n**2 * (t)**2 * sigma_per**2 / 2) * numpy.cos(n*t*(fL*2*pi))

    # X =
    return 4* lambd**2 * sum_expr

def S_bath_SE(t, fL, lambd, sigma_per, T2):
    # I = quad(integrand, -numpy.inf, numpy.inf, args=(t, fL))
    # t = t/2
    return numpy.exp(-X_SE(t, fL, lambd, sigma_per )) * numpy.exp(-(2*t/T2)**3)

    
def pop_S_1(t, a, b, f0, T):
    return (S_1(t, a, b, f0,  T) + 1)/2

def pop_S_2(t, a, b, f0, f1, T):
    return (S_2(t, a, b, f0, f1, T) + 1)/2
    
file = "2022_08_22-17_48_47-rubin-nv1"
folder = 'pc_rabi/branch_master/spin_echo/2022_08'

data = tool_belt.get_raw_data(file, folder)
sig_counts = numpy.array(data['sig_counts'])
ref_counts = numpy.array(data['ref_counts'])
precession_time_range = data['precession_time_range']
num_steps = data['num_steps']
min_precession_time = int(precession_time_range[0])
max_precession_time = int(precession_time_range[1])

taus = numpy.linspace(
    min_precession_time,
    max_precession_time,
    num=num_steps,
)
plot_taus = taus/ 1000

avg_sig_counts = numpy.average(sig_counts, axis=0)
avg_ref_counts = numpy.average(ref_counts, axis=0)

max_ref =  numpy.average(avg_ref_counts)
min_ref = avg_sig_counts[0]
contrast = 0.167*2#max_ref - min_ref

norm_avg_sig = avg_sig_counts / numpy.average(avg_ref_counts)
norm_avg_sig = (norm_avg_sig - (1-contrast))/(contrast)

# norm_avg_sig = (max_ref- avg_sig_counts) / (1 - contrast) 
    
# fit_func = lambda t, b, f0, T: pop_S_1(t, 1, b, f0, T)
# init_params = [1,  0.1, 200]
fit_func = lambda t, fL, lambd, sigma_per, T2: (S_bath_SE(t,fL, lambd, sigma_per, T2) + 1)/2
init_params = [0.1, 0.19, 0.02, 100]

popt, pcov = curve_fit(
    fit_func,
    plot_taus,
    norm_avg_sig,
    # sigma=norm_avg_sig_ste,
    # absolute_sigma=True,
    p0=init_params,
    # bounds=(min_bounds, max_bounds),
)
print(popt)

fig, ax = plt.subplots()
taus_lin = numpy.linspace(plot_taus[0], plot_taus[-1],600)
# taus_lin = numpy.linspace(0, 4,600)
ax.plot(
        taus_lin,
        fit_func(taus_lin, *popt),
        "-",
        color="red",
        label="model",
    ) 

ax.plot(
        plot_taus,
        norm_avg_sig,
        "o",
        color="blue",
        label="data",
    )    
ax.set_xlabel("Inter pulse wait time, tau (us)")
ax.set_ylabel("Normalized signal Counts")
ax.set_title('Spin echo')


